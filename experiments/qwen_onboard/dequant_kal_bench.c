/* dequant_kal_bench.c — verify + benchmark Qwen Path A K-aligned dequant.
 *
 * 1. Correctness: RVV vs scalar, random data, N=896/4864, element-exact.
 * 2. Throughput: process one full Qwen layer (7,453,184 nibble bytes),
 *    reuse a single output block buffer (avoid 30MB alloc -> swap thrash).
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

static inline double now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static int verify(int N, int nblk) {
    size_t niblen = (size_t)N * 16 * nblk;
    uint8_t *nib = malloc(niblen);
    int8_t *w_s = malloc((size_t)32 * N * nblk);
    int8_t *w_r = malloc((size_t)32 * N * nblk);
    for (size_t i = 0; i < niblen; i++) nib[i] = (uint8_t)rand();
    for (int b = 0; b < nblk; b++) {
        dequant_kal_scalar(nib + (size_t)b * N * 16, N, w_s + (size_t)b * 32 * N);
        dequant_kal_rvv(nib + (size_t)b * N * 16, N, w_r + (size_t)b * 32 * N);
    }
    size_t nvals = (size_t)32 * N * nblk;
    size_t bad = 0, dumped = 0;
    for (size_t i = 0; i < nvals && dumped < 6; i++) {
        if (w_s[i] != w_r[i]) {
            size_t blk = i / ((size_t)32 * N), rem = i % ((size_t)32 * N);
            size_t k = rem / N, n = rem % N, j = k / 2, nibv = (k & 1) ? (nib[blk*N*16 + n*16 + j] >> 4) : (nib[blk*N*16 + n*16 + j] & 0xF);
            printf("  MISMATCH i=%zu blk=%zu k=%zu n=%zu j=%zu nib=0x%02x s=%d r=%d\n",
                   i, blk, k, n, j, nib[blk*N*16 + n*16 + j], w_s[i], w_r[i]);
            dumped++;
        }
    }
    for (size_t i = 0; i < nvals; i++) if (w_s[i] != w_r[i]) bad++;
    free(nib); free(w_s); free(w_r);
    return (int)bad;
}

static void bench_one(const char *tag, void (*fn)(const uint8_t *, int, int8_t *),
                      uint8_t *nib, int8_t *w, int N, int nblk) {
    int reps = 3;
    double t0 = now();
    for (int r = 0; r < reps; r++)
        for (int b = 0; b < nblk; b++)
            fn(nib + (size_t)b * N * 16, N, w);   /* reuse w block buffer */
    double dt = (now() - t0) / reps;
    double MBs = (double)(size_t)N * 16 * nblk / dt / 1e6;
    double per_token = dt * (double)(LAYER_NIB * 24) / ((size_t)N * 16 * nblk);
    printf("%-24s: %7.1f MB/s nibble  (%.2f ms/layer -> %.2f s/token)\n",
           tag, MBs, dt * 1e3, per_token);
}

int main(void) {
    printf("===== dequant_kal bench: Qwen Path A K-aligned INT4->INT8 =====\n");
    int bad896 = verify(896, 8);
    int bad4864 = verify(4864, 4);
    printf("correctness RVV vs scalar: N=896 bad=%d  N=4864 bad=%d  %s\n",
           bad896, bad4864, (bad896 + bad4864) == 0 ? "ALL EXACT" : "FAIL");

    int blk896 = LAYER_NIB / (896 * 16);
    int blk4864 = LAYER_NIB / (4864 * 16);
    uint8_t *nib = malloc(LAYER_NIB);
    int8_t *w = malloc((size_t)32 * 4864);
    for (size_t i = 0; i < LAYER_NIB; i++) nib[i] = (uint8_t)rand();

    printf("-- N=896 (q/k/v/o/up/gate), %d blocks --\n", blk896);
    bench_one("scalar", dequant_kal_scalar, nib, w, 896, blk896);
    bench_one("rvv",    dequant_kal_rvv,    nib, w, 896, blk896);
    printf("-- N=4864 (down_proj), %d blocks --\n", blk4864);
    bench_one("scalar", dequant_kal_scalar, nib, w, 4864, blk4864);
    bench_one("rvv",    dequant_kal_rvv,    nib, w, 4864, blk4864);

    printf("===== done =====\n");
    free(nib); free(w);
    return 0;
}
