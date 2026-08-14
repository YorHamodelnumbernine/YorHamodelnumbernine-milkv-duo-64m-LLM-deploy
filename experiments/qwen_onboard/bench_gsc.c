/* bench_gsc.c — measure gsc sequential-read bandwidth vs nib dequant bandwidth,
 * to confirm whether readahead helps gsc when read as one contiguous region.
 * Reads layer0_kal.bin up_nib / up_gsc regions repeatedly (each iter re-reads
 * cold from page cache perspective -> use madvise(MADV_DONTNEED) to drop between iters).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <sys/mman.h>
#include "dequant_kal.c"

static inline double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * ts.tv_nsec;
}
static float fp16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16, exp = (h >> 10) & 0x1f, man = h & 0x3ff, f;
    if (exp == 0) { if (man == 0) f = sign; else { exp = 127 - 15 + 1; while (!(man & 0x400)) { man <<= 1; exp--; } man &= 0x3ff; f = sign | (exp << 23) | (man << 13); } }
    else if (exp == 31) f = sign | 0x7f800000u | (man << 13);
    else f = sign | ((exp + (127 - 15)) << 23) | (man << 13);
    float out; memcpy(&out, &f, 4); return out;
}

int main(void) {
#define D 896
#define G 32
#define DKV 128
#define F 4864
#define KGB (D / G)   /* K-blocks in K dimension for up/gate (896/32 = 28) */
#define GF (KGB * F)  /* up gsc entries = 28 * 4864 = 136192 (272KB fp16) */
    FILE *f = fopen("/data/qwen/layer0_kal.bin", "rb");
    if (!f) { fprintf(stderr, "no layer0_kal.bin\n"); return 2; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *lb = malloc(sz); if (fread(lb, 1, sz, f) != (size_t)sz) return 2; fclose(f);

    size_t off = (size_t)D * 4;
    size_t nib_off = off + (size_t)(D / G) * D * (16 + 2) + (size_t)(D / G) * DKV * (16 + 2) + (size_t)(D / G) * DKV * (16 + 2) + (size_t)(D / G) * D * (16 + 2);
    const uint8_t *up_nib = lb + nib_off;                    /* [KGB][F][16] = 2.18MB */
    const uint16_t *up_gsc = (const uint16_t *)(lb + nib_off + (size_t)KGB * F * 16); /* [KGB*F] = 272KB */

    const int ITER = 200;
    static float gbuf[GF];
    static uint16_t gbuf16[GF];
    static int8_t wbuf[32 * F];

    /* nib dequant bandwidth (hot) */
    double t0 = now_s();
    for (int it = 0; it < ITER; it++) dequant_kal_rvv(up_nib, F, wbuf);
    double dt_nib_hot = now_s() - t0;

    /* gsc sequential read + convert (hot) */
    t0 = now_s();
    for (int it = 0; it < ITER; it++)
        for (int i = 0; i < GF; i++) gbuf[i] = fp16_to_f32(up_gsc[i]);
    double dt_gsc_hot = now_s() - t0;

    /* gsc raw memcpy (hot) */
    t0 = now_s();
    for (int it = 0; it < ITER; it++) memcpy(gbuf16, up_gsc, sizeof(uint16_t) * GF);
    double dt_cpy_hot = now_s() - t0;

    printf("HOT (all resident):\n");
    printf("  nib dequant  %6.2f MB -> %7.1f MB/s  (%.3f ms/call)\n", (double)GF * 16 / 1e6 * ITER, (double)GF * 16 / 1e6 * ITER / dt_nib_hot, 1000.0 * dt_nib_hot / ITER);
    printf("  gsc conv     %6.2f MB -> %7.1f MB/s  (%.3f ms/call)\n", (double)GF * 2 / 1e6 * ITER, (double)GF * 2 / 1e6 * ITER / dt_gsc_hot, 1000.0 * dt_gsc_hot / ITER);
    printf("  gsc memcpy   %6.2f MB -> %7.1f MB/s  (%.3f ms/call)\n", (double)GF * 2 / 1e6 * ITER, (double)GF * 2 / 1e6 * ITER / dt_cpy_hot, 1000.0 * dt_cpy_hot / ITER);

    /* cold-ish: drop pages each iter via madvise(DONTNEED) — re-reads from page cache/SD */
    long nib_len = (long)GF * 16, gsc_len = (long)GF * 2;
    madvise(up_nib, nib_len, MADV_DONTNEED);
    madvise(up_gsc, gsc_len, MADV_DONTNEED);
    t0 = now_s();
    for (int it = 0; it < 40; it++) {
        dequant_kal_rvv(up_nib, F, wbuf);
        madvise(up_nib, nib_len, MADV_DONTNEED);
    }
    double dt_nib_cold = now_s() - t0;
    t0 = now_s();
    for (int it = 0; it < 40; it++) {
        for (int i = 0; i < GF; i++) gbuf[i] = fp16_to_f32(up_gsc[i]);
        madvise(up_gsc, gsc_len, MADV_DONTNEED);
    }
    double dt_gsc_cold = now_s() - t0;
    printf("COLD (madvise DONTNEED each iter, page-cache re-read):\n");
    printf("  nib dequant  %6.2f MB -> %7.1f MB/s\n", (double)nib_len / 1e6 * 40, (double)nib_len / 1e6 * 40 / dt_nib_cold);
    printf("  gsc conv     %6.2f MB -> %7.1f MB/s\n", (double)gsc_len / 1e6 * 40, (double)gsc_len / 1e6 * 40 / dt_gsc_cold);
    return 0;
}
