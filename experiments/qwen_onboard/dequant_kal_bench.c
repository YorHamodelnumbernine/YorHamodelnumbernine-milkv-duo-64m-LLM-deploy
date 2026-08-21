/* dequant_kal_bench.c — verify + benchmark Qwen Path A K-aligned dequant (Design B2).
 *
 * 1. Correctness: B2 RVV vs scalar on nib16[16][N] layout, random data,
 *    N=896/4864, full-block AND (col_off,ncols) window shapes, element-exact.
 * 2. Throughput A/B: B2 new layout (vle8 continuous) vs old nib[N][16] layout
 *    (vlse8 stride-16 gather), processing one full Qwen layer nibble set.
 *
 * Build: riscv64 cross; run: duo_run.py dequant_kal_bench
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include "dequant_kal.c"

#define LAYER_NIB 7453184   /* 24 layers -> 178,876,416 B nibble/token */
#define AB_BYTES  (4 * 1024 * 1024)  /* A/B per-buffer budget (fits 14MB RAM) */

static inline double now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

/* ---- old-layout (nib[N][16], stride-16 gather) RVV kernel, verbatim pre-B2
 * copy — A/B throughput reference. ---- */
static void dequant_kal_rvv_old(const uint8_t *nib, int N, int8_t *w) {
    const size_t VLM = 128;
    const size_t CB = 256;
    for (size_t nb = 0; nb < (size_t)N; nb += CB) {
        size_t ncols = (size_t)N - nb; if (ncols > CB) ncols = CB;
        const uint8_t *nibb = nib + nb * 16;
        int8_t *wb = w + nb;
        for (int j = 0; j < 16; j++) {
            const int8_t *src = (const int8_t *)nibb + j;
            for (size_t off = 0; off < ncols; off += VLM) {
                size_t vl = ncols - off; if (vl > VLM) vl = VLM;
                vint8m8_t v = vlse8_v_i8m8(src + off * 16, 16, vl);
                vint8m8_t lo = vsra_vx_i8m8(vsll_vx_i8m8(v, 4, vl), 4, vl);
                vint8m8_t hi = vsra_vx_i8m8(v, 4, vl);
                vse8_v_i8m8(wb + (size_t)(2 * j) * N + off, lo, vl);
                vse8_v_i8m8(wb + (size_t)(2 * j + 1) * N + off, hi, vl);
            }
        }
    }
}

/* Verify a single (col_off, ncols) window: RVV vs scalar on nib16 layout.
 * Returns number of mismatched int8 outputs. */
static int verify_window(const uint8_t *nib, int N, int col_off, int ncols) {
    int8_t *w_s = malloc((size_t)32 * ncols);
    int8_t *w_r = malloc((size_t)32 * ncols);
    dequant_kal_scalar(nib, N, col_off, ncols, w_s);
    dequant_kal_rvv(nib, N, col_off, ncols, w_r);
    int bad = 0;
    for (int k = 0; k < 32; k++)
        for (int n = 0; n < ncols; n++)
            if (w_s[(size_t)k * ncols + n] != w_r[(size_t)k * ncols + n]) {
                if (bad < 6)
                    printf("  MISMATCH N=%d col_off=%d n=%d k=%d s=%d r=%d\n",
                           N, col_off, col_off + n, k,
                           w_s[(size_t)k * ncols + n], w_r[(size_t)k * ncols + n]);
                bad++;
            }
    free(w_s); free(w_r);
    return bad;
}

static int verify(int N, int nblk) {
    size_t niblen = (size_t)16 * N * nblk;      /* nib16[blk][16][N] */
    uint8_t *nib = malloc(niblen);
    int8_t *w_s = malloc((size_t)32 * N * nblk);
    int8_t *w_r = malloc((size_t)32 * N * nblk);
    for (size_t i = 0; i < niblen; i++) nib[i] = (uint8_t)rand();

    /* full-block: all 7 mat shapes covered by N=896/4864 */
    for (int b = 0; b < nblk; b++) {
        dequant_kal_scalar(nib + (size_t)b * 16 * N, N, 0, N, w_s + (size_t)b * 32 * N);
        dequant_kal_rvv   (nib + (size_t)b * 16 * N, N, 0, N, w_r + (size_t)b * 32 * N);
    }
    size_t nvals = (size_t)32 * N * nblk;
    size_t bad = 0;
    for (size_t i = 0; i < nvals; i++) if (w_s[i] != w_r[i]) bad++;

    /* representative tile windows (per-tile + merged tile-major shapes) */
    static const int wins[][2] = {
        {0, 128}, {0, 256}, {0, 608}, {128, 768}, {448, 896}, {0, 896}, {0, 4864},
        {608, 608}, {2432, 2432}, {896, 3968}, {2432, 1216},
    };
    int nwin = (int)(sizeof(wins) / sizeof(wins[0]));
    int win_bad = 0;
    for (int wi = 0; wi < nwin; wi++) {
        int co = wins[wi][0], nc = wins[wi][1];
        if (co + nc > N) continue;
        win_bad += verify_window(nib, N, co, nc);
    }
    if (bad) {
        printf("  [N=%d] full-block RVV vs scalar: %zu mismatches\n", N, bad);
    }
    if (win_bad) {
        printf("  [N=%d] window RVV vs scalar: %d mismatches\n", N, win_bad);
    }
    free(nib); free(w_s); free(w_r);
    return (int)(bad + win_bad);
}

typedef void (*bench_fn)(const uint8_t *, int, int8_t *);
static void dequant_kal_rvv_full(const uint8_t *nib, int N, int8_t *w) {
    dequant_kal_rvv(nib, N, 0, N, w);   /* B2 kernel, full-block window */
}
static void bench_one(const char *tag, bench_fn fn,
                      const uint8_t *nib, int8_t *w, int N, int nblk) {
    int reps = 3;
    double t0 = now();
    for (int r = 0; r < reps; r++)
        for (int b = 0; b < nblk; b++)
            fn(nib + (size_t)b * N * 16, N, w);   /* reuse w block buffer */
    double dt = (now() - t0) / reps;
    double MBs = (double)(size_t)N * 16 * nblk / dt / 1e6;
    double per_token = dt * (double)(LAYER_NIB * 24) / ((size_t)N * 16 * nblk);
    printf("%-30s: %7.1f MB/s nibble  (%.2f ms/layer -> %.2f s/token)\n",
           tag, MBs, dt * 1e3, per_token);
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);   /* unbuffered: see progress on OOM/signal */
    printf("===== dequant_kal bench: Qwen Path A INT4->INT8 (Design B2 nib16) =====\n");
    int bad896 = verify(896, 8);
    int bad4864 = verify(4864, 4);
    printf("correctness RVV vs scalar: N=896 bad=%d  N=4864 bad=%d  %s\n",
           bad896, bad4864, (bad896 + bad4864) == 0 ? "ALL EXACT" : "FAIL");

    /* ---- A/B throughput: old nib[N][16] vs new nib16[16][N] ----
     * Use a bounded buffer (AB_BYTES) and loop over it; per-token estimate
     * scales by LAYER_NIB*24 (full 24-layer set). */
    int blk896 = AB_BYTES / (896 * 16);
    int blk4864 = AB_BYTES / (4864 * 16);
    uint8_t *nib16 = malloc(AB_BYTES);       /* new layout: nib16[KG][16][N] */
    uint8_t *nib_old = malloc(AB_BYTES);     /* old layout: nib[KG][N][16]   */
    int8_t *w = malloc((size_t)32 * 4864);
    if (!nib16 || !nib_old || !w) {
        fprintf(stderr, "A/B malloc FAILED (nib16=%p nib_old=%p w=%p)\n",
                (void *)nib16, (void *)nib_old, (void *)w);
        return 2;
    }
    for (size_t i = 0; i < AB_BYTES; i++) nib16[i] = (uint8_t)rand();
    /* inverse transpose: old[g*N*16 + n*16 + j] = new[g*16*N + j*N + n] */
    for (int g = 0; g < blk4864; g++) {
        int N = 4864;
        for (int j = 0; j < 16; j++)
            for (int n = 0; n < N; n++)
                nib_old[g * N * 16 + n * 16 + j] = nib16[g * 16 * N + j * N + n];
    }

    printf("-- N=896 (q/k/v/o/up/gate), %d blocks/pass --\n", blk896);
    bench_one("rvv_old (gather, nib[N][16])", dequant_kal_rvv_old, nib_old, w, 896, blk896);
    bench_one("rvv_b2  (vle8, nib16[16][N])", dequant_kal_rvv_full, nib16, w, 896, blk896);
    printf("-- N=4864 (down_proj), %d blocks/pass --\n", blk4864);
    bench_one("rvv_old (gather, nib[N][16])", dequant_kal_rvv_old, nib_old, w, 4864, blk4864);
    bench_one("rvv_b2  (vle8, nib16[16][N])", dequant_kal_rvv_full, nib16, w, 4864, blk4864);

    printf("===== done =====\n");
    free(nib16); free(nib_old); free(w);
    return 0;
}
