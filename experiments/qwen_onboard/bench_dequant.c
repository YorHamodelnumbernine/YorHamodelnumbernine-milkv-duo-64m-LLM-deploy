/* bench_dequant.c — micro-benchmark dequant_kal_rvv against real layer0 up nib.
 * Reads /data/qwen/layer0_kal.bin up nib (same offset as engine), dequants
 * repeatedly, reports MB/s read + total time.  Lets us iterate dequant variants
 * without the full engine/ION/TIU.
 *
 * Build (host): riscv64 cross, link nothing.
 *   riscv64-unknown-linux-musl-gcc -mcpu=c906fdv -march=rv64imafdcv0p7xthead \
 *     -mcmodel=medany -mabi=lp64d -O3 -std=gnu11 -fsigned-char \
 *     -o bench_dequant bench_dequant.c -lm -s
 * Run (board): ./bench_dequant [N] [iters]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include "dequant_kal.c"

static inline double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * ts.tv_nsec;
}

int main(int argc, char **argv) {
    int N = argc > 1 ? atoi(argv[1]) : 4864;
    int ITERS = argc > 2 ? atoi(argv[2]) : 200;
    const int D = 896;
    const int G = 32;
    const int DKV = 128;
    const int F = 4864;

    FILE *f = fopen("/data/qwen/layer0_kal.bin", "rb");
    if (!f) { fprintf(stderr, "no layer0_kal.bin\n"); return 2; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *lb = malloc(sz); if (fread(lb, 1, sz, f) != (size_t)sz) return 2; fclose(f);
    size_t off = (size_t)D * 4;
    off += (size_t)(D / G) * D * (16 + 2);
    off += (size_t)(D / G) * DKV * (16 + 2);
    off += (size_t)(D / G) * DKV * (16 + 2);
    off += (size_t)(D / G) * D * (16 + 2);
    const uint8_t *nib = lb + off;   /* up nib, [28][4864][16] */
    const uint8_t *nib0 = nib;       /* K-block 0 */

    int8_t *w = malloc((size_t)32 * N);
    if (!w) { perror("malloc w"); return 2; }

    /* warm + correctness vs scalar */
    dequant_kal_rvv(nib0, N, w);
    int8_t *wsc = malloc((size_t)32 * N);
    dequant_kal_scalar(nib0, N, wsc);
    int bad = 0, first = -1;
    for (int i = 0; i < 32 * N; i++)
        if (w[i] != wsc[i]) { if (bad < 5) printf("  MIS i=%d got=%d exp=%d\n", i, w[i], wsc[i]); if (first < 0) first = i; bad++; }
    printf("correctness vs scalar: %s (%d bad, first=%d)\n", bad ? "BROKEN" : "OK", bad, first);

    double t0 = now_s();
    for (int it = 0; it < ITERS; it++) dequant_kal_rvv(nib0, N, w);
    double dt = now_s() - t0;
    size_t read_bytes = (size_t)N * 16 * ITERS;
    size_t write_bytes = (size_t)32 * N * ITERS;
    printf("N=%d iters=%d  dequant total=%.3fs  per-call=%.3fms\n", N, ITERS, dt, 1000.0 * dt / ITERS);
    printf("  read  %.2f MB -> %.1f MB/s\n", read_bytes / 1e6, read_bytes / 1e6 / dt);
    printf("  write %.2f MB -> %.1f MB/s\n", write_bytes / 1e6, write_bytes / 1e6 / dt);
    printf("  r+w   %.2f MB -> %.1f MB/s\n", (read_bytes + write_bytes) / 1e6, (read_bytes + write_bytes) / 1e6 / dt);
    return 0;
}
