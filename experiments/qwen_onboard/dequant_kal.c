/* dequant_kal.c — Qwen Path A K-aligned INT4 -> INT8 dequant, scalar + RVV.
 *
 * Layout (weights_kal / layerN_kal.bin, per K-block of 32, KG=32):
 *   nib[N][16]  : for each output column n, 16 bytes.  Byte j holds K-row
 *                 k=2*j (low nibble) and k=2*j+1 (high nibble), raw int4 -8..7.
 * Output:
 *   w[32][N] K-major  (w[k*N + n])  -> feeds TIU right operand [K=32, N].
 *
 * RVV strategy: for each j (row-pair), gather byte j across columns with a
 * strided load (stride 16), sign-extend both nibbles, and do CONTIGUOUS
 * stores into rows 2*j / 2*j+1.  Avoids the scalar path's stride-N scatter.
 *
 * Build: riscv64 cross with -march=rv64imafdcv0p7xthead (RVV 0.7.1).
 */
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <riscv_vector.h>

/* ---- scalar reference (matches qwen_m1_chunk.c dequant_block) ---- */
void dequant_kal_scalar(const uint8_t *nib, int N, int8_t *w) {
    for (int n = 0; n < N; n++) {
        const uint8_t *b = nib + (size_t)n * 16;
        for (int j = 0; j < 16; j++) {
            int lo = b[j] & 0xF, hi = b[j] >> 4;
            lo = lo > 7 ? lo - 16 : lo; hi = hi > 7 ? hi - 16 : hi;
            w[(size_t)(2 * j) * N + n] = (int8_t)lo;
            w[(size_t)(2 * j + 1) * N + n] = (int8_t)hi;
        }
    }
}

/* ---- RVV vectorized (C906 VLEN=128bit; m8 -> 128 int8/vector) ----
 * Column-blocked: for large N (down_proj N=4864) a full-N stride-16 gather
 * spans 77KB and thrashes L2.  Processing 256-column blocks keeps the gather
 * working set at 4KB resident in L1/L2. */
void dequant_kal_rvv(const uint8_t *nib, int N, int8_t *w) {
    const size_t VLM = 128;                 /* VLEN/SEW * LMUL8 = 128/8*8 */
    const size_t CB = 256;                  /* column block */
    for (size_t nb = 0; nb < (size_t)N; nb += CB) {
        size_t ncols = (size_t)N - nb; if (ncols > CB) ncols = CB;
        const uint8_t *nibb = nib + nb * 16;
        int8_t *wb = w + nb;
        for (int j = 0; j < 16; j++) {
            const int8_t *src = (const int8_t *)nibb + j;   /* byte j of col nb */
            for (size_t off = 0; off < ncols; off += VLM) {
                size_t vl = ncols - off;
                if (vl > VLM) vl = VLM;
                vint8m8_t v = vlse8_v_i8m8(src + off * 16, 16, vl);   /* gather byte j */
                vint8m8_t lo = vsra_vx_i8m8(vsll_vx_i8m8(v, 4, vl), 4, vl); /* low  nibble */
                vint8m8_t hi = vsra_vx_i8m8(v, 4, vl);                     /* high nibble */
                vse8_v_i8m8(wb + (size_t)(2 * j) * N + off, lo, vl);
                vse8_v_i8m8(wb + (size_t)(2 * j + 1) * N + off, hi, vl);
            }
        }
    }
}
