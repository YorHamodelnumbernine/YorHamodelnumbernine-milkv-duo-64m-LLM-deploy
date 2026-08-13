/* int4_bench.c — on-device INT4 unpack micro-benchmark (Design A).
 *
 * Two numbers:
 *   (1) read+unpack — realistic pf_worker path: SD read layerN_i4.bin +
 *                     scalar fixed-point unpack -> INT8 (3,543,552 B/layer),
 *                     total over 30 layers.
 *   (2) unpack-only — pure CPU scalar unpack throughput measured on a cached
 *                     matrix block (no SD), scaled to 30 layers.
 *
 * If (2) total for 30 layers > 200 ms, RVV vectorized unpack is warranted
 * (per CEO acceptance plan).
 *
 * Build (cross): riscv64-unknown-linux-musl-gcc -O3 -mcpu=c906fdv \
 *       -march=rv64imafdcv0p7xthead -mcmodel=medany -mabi=lp64d \
 *       -o int4_bench int4_bench.c int4_common.c -lm
 * Run (device): ./int4_bench <weight_dir> [n_repeat]
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include "int4_common.h"

#define D    576
#define DKVD 192
#define FFN  1536
#define G    64
#define N_MAT 7
#define LAYER_SZ (D*4 + D*D + D*DKVD + D*DKVD + D*D + D*4 + D*FFN + D*FFN + FFN*D)  /* 3,543,552 */

static const int mat_rows[] = { D, D, D, D, D, D, FFN };
static const int mat_cols[] = { D, DKVD, DKVD, D, FFN, FFN, D };

static inline double now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static int read_full(int fd, void *buf, size_t n) {
    size_t got = 0; uint8_t *p = (uint8_t *)buf;
    while (got < n) {
        int r = read(fd, p + got, n - got);
        if (r <= 0) return -1;
        got += r;
    }
    return 0;
}

#if defined(__riscv_vector)
#include <riscv_vector.h>
/* RVV unpack into dst; scratch layout identical to scalar path. */
static int layer_unpack_rvv(const char *dir, int layer, uint8_t *dst, uint8_t *scratch) {
    char path[256];
    snprintf(path, sizeof(path), "%s/layer%d_i4.bin", dir, layer);
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    if (read_full(fd, dst, D*4) != 0) { close(fd); return -1; }
    uint8_t *w = dst + D*4;
    for (int m = 0; m < 4; m++) {
        int n = mat_rows[m] * mat_cols[m];
        int nib = n/2, sc = (n/G)*2;
        if (read_full(fd, scratch, nib) || read_full(fd, scratch+nib, sc)) { close(fd); return -1; }
        int16_t *scf = (int16_t*)(scratch + nib);
        for (int g = 0; g < n/G; g++) {
            uint16_t h; memcpy(&h, scratch + nib + (size_t)g*2, 2);
            scf[g] = (int16_t)lrintf(fp16_to_float(h) * 256.0f);
        }
        int4_unpack_fixed_rvv(scratch, scf, n, G, w);
        w += n;
    }
    if (read_full(fd, w, D*4) != 0) { close(fd); return -1; }
    w += D*4;
    for (int m = 4; m < N_MAT; m++) {
        int n = mat_rows[m] * mat_cols[m];
        int nib = n/2, sc = (n/G)*2;
        if (read_full(fd, scratch, nib) || read_full(fd, scratch+nib, sc)) { close(fd); return -1; }
        int16_t *scf = (int16_t*)(scratch + nib);
        for (int g = 0; g < n/G; g++) {
            uint16_t h; memcpy(&h, scratch + nib + (size_t)g*2, 2);
            scf[g] = (int16_t)lrintf(fp16_to_float(h) * 256.0f);
        }
        int4_unpack_fixed_rvv(scratch, scf, n, G, w);
        w += n;
    }
    close(fd);
    return 0;
}
#endif /* __riscv_vector */

/* Unpack one layerN_i4.bin into dst (INT8, LAYER_SZ bytes). scratch >= 512KB. */
static int layer_unpack(const char *dir, int layer, uint8_t *dst, uint8_t *scratch) {
    char path[256];
    snprintf(path, sizeof(path), "%s/layer%d_i4.bin", dir, layer);
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;

    if (read_full(fd, dst, D*4) != 0) { close(fd); return -1; }   /* rms_attn */
    uint8_t *w = dst + D*4;
    for (int m = 0; m < 4; m++) {                                  /* Wq Wk Wv Wo */
        int n = mat_rows[m] * mat_cols[m];
        int nib = n/2, sc = (n/G)*2;
        if (read_full(fd, scratch, nib) || read_full(fd, scratch+nib, sc)) { close(fd); return -1; }
        int16_t *scf = (int16_t*)(scratch + nib);
        for (int g = 0; g < n/G; g++) {
            uint16_t h; memcpy(&h, scratch + nib + (size_t)g*2, 2);
            scf[g] = (int16_t)lrintf(fp16_to_float(h) * 256.0f);
        }
        int4_unpack_fixed(scratch, scf, n, G, w);
        w += n;
    }
    if (read_full(fd, w, D*4) != 0) { close(fd); return -1; }     /* rms_ffn */
    w += D*4;
    for (int m = 4; m < N_MAT; m++) {                              /* up gate down */
        int n = mat_rows[m] * mat_cols[m];
        int nib = n/2, sc = (n/G)*2;
        if (read_full(fd, scratch, nib) || read_full(fd, scratch+nib, sc)) { close(fd); return -1; }
        int16_t *scf = (int16_t*)(scratch + nib);
        for (int g = 0; g < n/G; g++) {
            uint16_t h; memcpy(&h, scratch + nib + (size_t)g*2, 2);
            scf[g] = (int16_t)lrintf(fp16_to_float(h) * 256.0f);
        }
        int4_unpack_fixed(scratch, scf, n, G, w);
        w += n;
    }
    close(fd);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <weight_dir> [n_repeat]\n", argv[0]); return 1; }
    const char *dir = argv[1];
    int nrep = (argc > 2) ? atoi(argv[2]) : 1;
    const int L = 30;
    const int LAYER_ELEMS = LAYER_SZ - 2*D*4;   /* int8 matrix bytes per layer */

    uint8_t *dst = (uint8_t*)malloc(LAYER_SZ);
    uint8_t *scr = (uint8_t*)malloc(512*1024);
    if (!dst || !scr) { fprintf(stderr, "OOM\n"); return 1; }

    /* (1) read+unpack over 30 layers */
    double t_tot = 0;
    int errs = 0;
    for (int rep = 0; rep < nrep; rep++) {
        for (int l = 0; l < L; l++) {
            double t0 = now();
            if (layer_unpack(dir, l, dst, scr) != 0) { errs++; continue; }
            t_tot += now() - t0;
        }
    }
    if (errs) fprintf(stderr, "WARN: %d layer load errors\n", errs);
    printf("read+unpack 30 layers: %.1f ms total (%.2f ms/layer)  [SD read + scalar CPU]\n",
           t_tot/(nrep?nrep:1)*1e3, t_tot/(nrep?nrep:1)*1e3/L);

    /* (2) pure scalar unpack throughput on a cached D*F matrix (CPU only) */
    {
        int n = D * FFN;
        int nib = n/2, sc = (n/G)*2;
        int8_t *src = (int8_t*)malloc(n);
        for (int i = 0; i < n; i++) src[i] = (int8_t)(((i * 7) & 0xFF) - 128);
        int nsc = int4_pack(src, n, G, 1, scr, scr + nib);
        int16_t *scf = (int16_t*)(scr + nib);
        for (int g = 0; g < nsc; g++) {
            uint16_t h; memcpy(&h, scr + nib + (size_t)g*2, 2);
            scf[g] = (int16_t)lrintf(fp16_to_float(h) * 256.0f);
        }
        int8_t *out = (int8_t*)malloc(n);
        double reps = 300.0;
        double t0 = now();
        for (int r = 0; r < (int)reps; r++) int4_unpack_fixed(scr, scf, n, G, out);
        double dt = (now() - t0) / reps;             /* s per FFN matrix unpack */
        double per_elem = dt / n;                    /* s per element */
        double tot30 = per_elem * LAYER_ELEMS * L;   /* s for 30 layers */
        double MBps = n / dt / 1e6;
        printf("unpack_fixed scalar: %.0f MB/s int8-out  (%.2f ms/FFN-matrix, %.1f ms est for 30 layers)\n",
               MBps, dt*1e3, tot30*1e3);
        free(src); free(out);
    }

#if defined(__riscv_vector)
    /* (3) RVV correctness vs scalar on layer0, then RVV timing. */
    {
        uint8_t *ref = (uint8_t*)malloc(LAYER_SZ);
        uint8_t *out = (uint8_t*)malloc(LAYER_SZ);
        if (ref && out) {
            if (layer_unpack(dir, 0, ref, scr) != 0) { fprintf(stderr, "ref layer0 fail\n"); }
            else if (layer_unpack_rvv(dir, 0, out, scr) != 0) { fprintf(stderr, "rvv layer0 fail\n"); }
            else {
                int nerr = 0, first = -1;
                for (int i = 0; i < LAYER_SZ; i++) if (ref[i] != out[i]) { nerr++; if (first < 0) first = i; }
                printf("RVV vs scalar layer0: %s (%d mismatches", nerr ? "MISMATCH" : "MATCH", nerr);
                if (first >= 0) printf(", first@%d ref=%d rvv=%d", first, ref[first], out[first]);
                printf(")\n");
            }
            double t_rvv = 0;
            for (int l = 0; l < L; l++) {
                double t0 = now();
                if (layer_unpack_rvv(dir, l, out, scr) != 0) { fprintf(stderr, "rvv layer %d fail\n", l); break; }
                t_rvv += now() - t0;
            }
            printf("read+unpack(RVV) 30 layers: %.1f ms total (%.2f ms/layer)\n",
                   t_rvv*1e3, t_rvv*1e3/L);

            /* pure RVV unpack throughput on cached FFN matrix */
            int n = D * FFN;
            int nib = n/2, sc = (n/G)*2;
            int8_t *src = (int8_t*)malloc(n);
            for (int i = 0; i < n; i++) src[i] = (int8_t)(((i * 7) & 0xFF) - 128);
            int nsc = int4_pack(src, n, G, 1, scr, scr + nib);
            int16_t *scf = (int16_t*)(scr + nib);
            for (int g = 0; g < nsc; g++) {
                uint16_t h; memcpy(&h, scr + nib + (size_t)g*2, 2);
                scf[g] = (int16_t)lrintf(fp16_to_float(h) * 256.0f);
            }
            double reps = 3000.0;
            double t0 = now();
            for (int r = 0; r < (int)reps; r++) int4_unpack_fixed_rvv(scr, scf, n, G, out);
            double dt = (now() - t0) / reps;
            double per_elem = dt / n;
            double tot30 = per_elem * LAYER_ELEMS * L;
            printf("unpack_fixed_rvv: %.0f MB/s int8-out  (%.3f ms/FFN-matrix, %.1f ms est for 30 layers)\n",
                   n / dt / 1e6, dt*1e3, tot30*1e3);
            free(src);
        }
        free(ref); free(out);
    }
#endif /* __riscv_vector */

    free(dst); free(scr);
    return 0;
}
