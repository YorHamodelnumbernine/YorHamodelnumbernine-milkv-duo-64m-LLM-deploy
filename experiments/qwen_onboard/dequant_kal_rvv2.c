/* dequant_kal_rvv2.c — P2 candidate kernels for Qwen Path A INT4->INT8 dequant.
 *
 * Goal: beat the merged dequant_kal_rvv (378-384 MB/s) by 5-10% on C906.
 * Interface (FIXED, per design A §5): void dequant_kal_rvv(const uint8_t*, int, int8_t*)
 * Layout: nib[N][16] -> w[32][N] K-major.  Build -march=rv64imafdcv0p7xthead (RVV 0.7.1).
 *
 * This file is EXPERIMENTAL. It does not touch the merged dequant_kal.c.
 * Variants are suffixed; the benchmark harness compares each vs scalar.
 */
#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>

/* ==== baseline: verbatim copy of merged dequant_kal_rvv (reference for A/B) ==== */
void dequant_kal_rvv_baseline(const uint8_t *nib, int N, int8_t *w) {
    const size_t VLM = 128;
    const size_t CB = 256;
    for (size_t nb = 0; nb < (size_t)N; nb += CB) {
        size_t ncols = (size_t)N - nb; if (ncols > CB) ncols = CB;
        const uint8_t *nibb = nib + nb * 16;
        int8_t *wb = w + nb;
        for (int j = 0; j < 16; j++) {
            const int8_t *src = (const int8_t *)nibb + j;
            for (size_t off = 0; off < ncols; off += VLM) {
                size_t vl = ncols - off;
                if (vl > VLM) vl = VLM;
                vint8m8_t v = vlse8_v_i8m8(src + off * 16, 16, vl);
                vint8m8_t lo = vsra_vx_i8m8(vsll_vx_i8m8(v, 4, vl), 4, vl);
                vint8m8_t hi = vsra_vx_i8m8(v, 4, vl);
                vse8_v_i8m8(wb + (size_t)(2 * j) * N + off, lo, vl);
                vse8_v_i8m8(wb + (size_t)(2 * j + 1) * N + off, hi, vl);
            }
        }
    }
}

/* ==== variant swap: off-outer / j-inner for L1 locality + no per-iter tail branch.
 * For a fixed 128-col sub-block (2048B contiguous input), all 16 j's are gathered
 * while it is hot in L1/L2; the original swept the whole 256-col block per j. */
void dequant_kal_rvv_swap(const uint8_t *nib, int N, int8_t *w) {
    const size_t VLM = 128;
    const size_t CB = 256;
    for (size_t nb = 0; nb < (size_t)N; nb += CB) {
        size_t ncols = (size_t)N - nb; if (ncols > CB) ncols = CB;
        const uint8_t *nibb = nib + nb * 16;
        int8_t *wb = w + nb;
        /* N is a multiple of VLM for all real shapes (128/896/768/4864), so the
         * last sub-block is always full; keep the tail guard for generality. */
        for (size_t off = 0; off < ncols; off += VLM) {
            size_t vl = ncols - off; if (vl > VLM) vl = VLM;
            const int8_t *sb = (const int8_t *)(nibb + off * 16);
            int8_t *wbk = wb + off;
            for (int j = 0; j < 16; j++) {
                vint8m8_t v = vlse8_v_i8m8(sb + j, 16, vl);
                vint8m8_t lo = vsra_vx_i8m8(vsll_vx_i8m8(v, 4, vl), 4, vl);
                vint8m8_t hi = vsra_vx_i8m8(v, 4, vl);
                vse8_v_i8m8(wbk + (size_t)(2 * j) * N, lo, vl);
                vse8_v_i8m8(wbk + (size_t)(2 * j + 1) * N, hi, vl);
            }
        }
    }
}

/* ==== variant cb128: same loop order, smaller column block (2048B working set). */
void dequant_kal_rvv_cb128(const uint8_t *nib, int N, int8_t *w) {
    const size_t VLM = 128;
    const size_t CB = 128;
    for (size_t nb = 0; nb < (size_t)N; nb += CB) {
        size_t ncols = (size_t)N - nb; if (ncols > CB) ncols = CB;
        const uint8_t *nibb = nib + nb * 16;
        int8_t *wb = w + nb;
        for (size_t off = 0; off < ncols; off += VLM) {
            size_t vl = ncols - off; if (vl > VLM) vl = VLM;
            const int8_t *sb = (const int8_t *)(nibb + off * 16);
            int8_t *wbk = wb + off;
            for (int j = 0; j < 16; j++) {
                vint8m8_t v = vlse8_v_i8m8(sb + j, 16, vl);
                vint8m8_t lo = vsra_vx_i8m8(vsll_vx_i8m8(v, 4, vl), 4, vl);
                vint8m8_t hi = vsra_vx_i8m8(v, 4, vl);
                vse8_v_i8m8(wbk + (size_t)(2 * j) * N, lo, vl);
                vse8_v_i8m8(wbk + (size_t)(2 * j + 1) * N, hi, vl);
            }
        }
    }
}

/* ==== variant unroll2: off-outer / j-inner, j unrolled by 2 (fewer loop deps). */
void dequant_kal_rvv_unroll2(const uint8_t *nib, int N, int8_t *w) {
    const size_t VLM = 128;
    const size_t CB = 256;
    for (size_t nb = 0; nb < (size_t)N; nb += CB) {
        size_t ncols = (size_t)N - nb; if (ncols > CB) ncols = CB;
        const uint8_t *nibb = nib + nb * 16;
        int8_t *wb = w + nb;
        for (size_t off = 0; off < ncols; off += VLM) {
            size_t vl = ncols - off; if (vl > VLM) vl = VLM;
            const int8_t *sb = (const int8_t *)(nibb + off * 16);
            int8_t *wbk = wb + off;
            size_t j = 0;
            for (; j + 1 < 16; j += 2) {
                vint8m8_t v0 = vlse8_v_i8m8(sb + j, 16, vl);
                vint8m8_t v1 = vlse8_v_i8m8(sb + j + 1, 16, vl);
                vint8m8_t l0 = vsra_vx_i8m8(vsll_vx_i8m8(v0, 4, vl), 4, vl);
                vint8m8_t h0 = vsra_vx_i8m8(v0, 4, vl);
                vint8m8_t l1 = vsra_vx_i8m8(vsll_vx_i8m8(v1, 4, vl), 4, vl);
                vint8m8_t h1 = vsra_vx_i8m8(v1, 4, vl);
                vse8_v_i8m8(wbk + (size_t)(2 * j) * N, l0, vl);
                vse8_v_i8m8(wbk + (size_t)(2 * j + 1) * N, h0, vl);
                vse8_v_i8m8(wbk + (size_t)(2 * j + 2) * N, l1, vl);
                vse8_v_i8m8(wbk + (size_t)(2 * j + 3) * N, h1, vl);
            }
            for (; j < 16; j++) {
                vint8m8_t v = vlse8_v_i8m8(sb + j, 16, vl);
                vint8m8_t lo = vsra_vx_i8m8(vsll_vx_i8m8(v, 4, vl), 4, vl);
                vint8m8_t hi = vsra_vx_i8m8(v, 4, vl);
                vse8_v_i8m8(wbk + (size_t)(2 * j) * N, lo, vl);
                vse8_v_i8m8(wbk + (size_t)(2 * j + 1) * N, hi, vl);
            }
        }
    }
}

/* ==== variant unroll4: off-outer / j-inner, j unrolled by 4. */
void dequant_kal_rvv_unroll4(const uint8_t *nib, int N, int8_t *w) {
    const size_t VLM = 128;
    const size_t CB = 256;
    for (size_t nb = 0; nb < (size_t)N; nb += CB) {
        size_t ncols = (size_t)N - nb; if (ncols > CB) ncols = CB;
        const uint8_t *nibb = nib + nb * 16;
        int8_t *wb = w + nb;
        for (size_t off = 0; off < ncols; off += VLM) {
            size_t vl = ncols - off; if (vl > VLM) vl = VLM;
            const int8_t *sb = (const int8_t *)(nibb + off * 16);
            int8_t *wbk = wb + off;
            size_t j = 0;
            for (; j + 3 < 16; j += 4) {
                vint8m8_t v0 = vlse8_v_i8m8(sb + j, 16, vl);
                vint8m8_t v1 = vlse8_v_i8m8(sb + j + 1, 16, vl);
                vint8m8_t v2 = vlse8_v_i8m8(sb + j + 2, 16, vl);
                vint8m8_t v3 = vlse8_v_i8m8(sb + j + 3, 16, vl);
                vint8m8_t l0 = vsra_vx_i8m8(vsll_vx_i8m8(v0, 4, vl), 4, vl);
                vint8m8_t h0 = vsra_vx_i8m8(v0, 4, vl);
                vint8m8_t l1 = vsra_vx_i8m8(vsll_vx_i8m8(v1, 4, vl), 4, vl);
                vint8m8_t h1 = vsra_vx_i8m8(v1, 4, vl);
                vint8m8_t l2 = vsra_vx_i8m8(vsll_vx_i8m8(v2, 4, vl), 4, vl);
                vint8m8_t h2 = vsra_vx_i8m8(v2, 4, vl);
                vint8m8_t l3 = vsra_vx_i8m8(vsll_vx_i8m8(v3, 4, vl), 4, vl);
                vint8m8_t h3 = vsra_vx_i8m8(v3, 4, vl);
                vse8_v_i8m8(wbk + (size_t)(2 * j) * N, l0, vl);
                vse8_v_i8m8(wbk + (size_t)(2 * j + 1) * N, h0, vl);
                vse8_v_i8m8(wbk + (size_t)(2 * j + 2) * N, l1, vl);
                vse8_v_i8m8(wbk + (size_t)(2 * j + 3) * N, h1, vl);
                vse8_v_i8m8(wbk + (size_t)(2 * j + 4) * N, l2, vl);
                vse8_v_i8m8(wbk + (size_t)(2 * j + 5) * N, h2, vl);
                vse8_v_i8m8(wbk + (size_t)(2 * j + 6) * N, l3, vl);
                vse8_v_i8m8(wbk + (size_t)(2 * j + 7) * N, h3, vl);
            }
            for (; j < 16; j++) {
                vint8m8_t v = vlse8_v_i8m8(sb + j, 16, vl);
                vint8m8_t lo = vsra_vx_i8m8(vsll_vx_i8m8(v, 4, vl), 4, vl);
                vint8m8_t hi = vsra_vx_i8m8(v, 4, vl);
                vse8_v_i8m8(wbk + (size_t)(2 * j) * N, lo, vl);
                vse8_v_i8m8(wbk + (size_t)(2 * j + 1) * N, hi, vl);
            }
        }
    }
}
