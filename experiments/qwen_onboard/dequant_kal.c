/* dequant_kal.c — Qwen Path A K-aligned INT4 -> INT8 dequant, scalar + RVV.
 *
 * Layout (weights_kal_b2 / layerN_kal_b2.bin, per K-block of 32, KG=32):
 *   nib16[16][N] : 16 row-pair planes, each N bytes contiguous.  Plane j byte n
 *                  holds column n: low nibble = K-row 2j, high nibble = K-row 2j+1,
 *                  raw int4 -8..7.  (Design B2 re-layout: old nib[N][16] transposed
 *                  to nib16[16][N]; bit-exactness preserved by construction.)
 * Output:
 *   w[32][ncols] K-major  (w[k*ncols + n])  -> feeds TIU right operand [K=32, ncols].
 *
 * RVV strategy: for each j (row-pair), CONTIGUOUS load (vle8) of the column window
 * [col_off, col_off+ncols) from plane j, sign-extend both nibbles, and store into
 * rows 2*j / 2*j+1.  vs. the old nib[N][16] strided layout (vlse8 stride-16 gather)
 * this touches ~16x fewer cache lines per load and drops the gather instruction.
 *
 * Sign extension MUST stay vsra(vsll(v,4),4)/vsra(v,4): raw int4 8..15 means -8..-1;
 * a plain v&0xF mask would produce 0..15 and break bit-exactness.
 *
 * Build: riscv64 cross with -march=rv64imafdcv0p7xthead (RVV 0.7.1).
 */
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <riscv_vector.h>

/* ---- scalar reference (nib16 layout; matches dequant semantics) ---- */
void dequant_kal_scalar(const uint8_t *nib, int N, int col_off, int ncols, int8_t *w) {
    for (int n = 0; n < ncols; n++) {
        for (int j = 0; j < 16; j++) {
            uint8_t b = nib[(size_t)j * N + col_off + n];
            int lo = b & 0xF, hi = b >> 4;
            lo = lo > 7 ? lo - 16 : lo; hi = hi > 7 ? hi - 16 : hi;
            w[(size_t)(2 * j) * ncols + n] = (int8_t)lo;
            w[(size_t)(2 * j + 1) * ncols + n] = (int8_t)hi;
        }
    }
}

/* ---- RVV vectorized (C906 VLEN=128bit; m8 -> 128 int8/vector) ----
 * nib = nib16 base of one K-block (32 rows): plane j at nib + j*N.  Column window
 * [col_off, col_off+ncols).  Output w[32][ncols] K-major.  Continuous vle8 loads
 * (no stride-16 gather; whole 128B load touches ~2 cache lines instead of ~32). */
void dequant_kal_rvv(const uint8_t *nib, int N, int col_off, int ncols, int8_t *w) {
    const size_t VLM = 128;                 /* VLEN/SEW * LMUL8 = 128/8*8 */
    for (int j = 0; j < 16; j++) {
        const uint8_t *src = nib + (size_t)j * N + col_off;
        int8_t *wlo = w + (size_t)(2 * j) * ncols;
        int8_t *whi = w + (size_t)(2 * j + 1) * ncols;
        for (size_t off = 0; off < (size_t)ncols; off += VLM) {
            size_t vl = (size_t)ncols - off;
            if (vl > VLM) vl = VLM;
            vint8m8_t v = vle8_v_i8m8(src + off, vl);      /* continuous load */
            vint8m8_t lo = vsra_vx_i8m8(vsll_vx_i8m8(v, 4, vl), 4, vl); /* low  nibble */
            vint8m8_t hi = vsra_vx_i8m8(v, 4, vl);                      /* high nibble */
            vse8_v_i8m8(wlo + off, lo, vl);
            vse8_v_i8m8(whi + off, hi, vl);
        }
    }
}
