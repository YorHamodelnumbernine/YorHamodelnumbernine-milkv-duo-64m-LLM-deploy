/* int4_proto.c — host-side INT4 weight pack/unpack prototype (no Duo dependency).
 *
 * Purpose: validate the INT4 + per-group-scale format from DESIGN_INT4_WEIGHTS.md,
 * measure unpack throughput (x86 proxy), and provide the two pure functions that
 * will later be ported into smollm2_pool_demo.c's pf_worker / initial-load path.
 *
 * Build (host):  gcc -O3 -o int4_proto int4_proto.c int4_common.c -lm
 * Run:           ./int4_proto [n_elements] [group_size] [scale_type(0=fp32,1=fp16)] [dist(0=uniform,1=realistic±32)]
 *
 * Formats:
 *   nibble byte:  low nibble = element 2i, high nibble = element 2i+1  (row-major)
 *   scales:       one per G elements, fp32 (4B) or fp16 (2B)
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "int4_common.h"   /* shared pack/unpack — same code as convert_i4.c */

static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec + ts.tv_nsec/1e9; }

int main(int argc, char **argv) {
    int n = (argc > 1) ? atoi(argv[1]) : (576*1536);   /* default: one FFN matrix */
    int G = (argc > 2) ? atoi(argv[2]) : 64;
    int st = (argc > 3) ? atoi(argv[3]) : 1;            /* scale type: 1=fp16 */
    int dist = (argc > 4) ? atoi(argv[4]) : 0;          /* 0=uniform full-range, 1=realistic(±32) */
    if (n % G) { fprintf(stderr, "n must be multiple of G\n"); return 1; }

    /* deterministic pseudo-random int8 weights
     * dist=0: uniform -128..127 (worst case: tests format robustness)
     * dist=1: bounded -32..31 (typical magnitude of per-channel-scaled int8) */
    int8_t *src = (int8_t*)malloc(n);
    uint32_t seed = 12345;
    for (int i = 0; i < n; i++) {
        seed = seed*1103515245 + 12345;
        uint32_t r = (seed>>16) & 0xff;
        src[i] = dist ? (int8_t)((int)(r & 0x3f) - 32) : (int8_t)r;
    }

    size_t nib_sz = n/2, sc_sz = (size_t)(n/G)*(st?2:4);
    uint8_t *nib = (uint8_t*)malloc(nib_sz);
    uint8_t *sc  = (uint8_t*)malloc(sc_sz);
    int8_t  *out = (int8_t*)malloc(n);

    int nsc = int4_pack(src, n, G, st, nib, sc);
    printf("packed: n=%d G=%d scale=%s nibbles=%zu scales=%zu(%d) total=%zu bytes\n",
           n, G, st?"fp16":"fp32", nib_sz, sc_sz, nsc, nib_sz + sc_sz);

    /* correctness: max & RMS error of dequantized int8 */
    double t0 = now();
    int4_unpack(nib, sc, n, G, st, out);
    double t1 = now();
    long long maxe = 0; double ss = 0;
    for (int i = 0; i < n; i++) { long long e = (long long)src[i] - out[i]; if (e<0)e=-e; if(e>maxe)maxe=e; ss+=(double)e*e; }
    printf("unpack(flt)  : %.1f us  max_err=%lld  rms_err=%.3f\n",
           (t1-t0)*1e6, maxe, sqrt(ss/n));

    /* fixed-point path */
    int16_t *scf = (int16_t*)malloc((size_t)(n/G)*2);
    for (int g = 0; g < nsc; g++) {
        float s; if (st==0) memcpy(&s, sc+(size_t)g*4,4); else { uint16_t h; memcpy(&h, sc+(size_t)g*2,2); s = fp16_to_float(h); }
        scf[g] = (int16_t)lrintf(s * 256.0f);
    }
    t0 = now();
    int4_unpack_fixed(nib, scf, n, G, out);
    t1 = now();
    maxe = 0; ss = 0;
    for (int i = 0; i < n; i++) { long long e = (long long)src[i] - out[i]; if (e<0)e=-e; if(e>maxe)maxe=e; ss+=(double)e*e; }
    printf("unpack(fix)  : %.1f us  max_err=%lld  rms_err=%.3f\n",
           (t1-t0)*1e6, maxe, sqrt(ss/n));

    /* throughput (bytes of int8 output per second, x86 proxy) */
    double reps = 5;
    t0 = now();
    for (int r = 0; r < (int)reps; r++) int4_unpack_fixed(nib, scf, n, G, out);
    t1 = now();
    double MBps = reps * n / 1e6 / (t1 - t0);
    printf("unpack(fix) throughput: %.0f MB/s (int8 out), %.1f us per %d-elem block\n",
           MBps, (t1-t0)*1e6/reps, n);

    free(src); free(nib); free(sc); free(out); free(scf);
    return 0;
}
