/* int4_proto.c — host-side INT4 weight pack/unpack prototype (no Duo dependency).
 *
 * Purpose: validate the INT4 + per-group-scale format from DESIGN_INT4_WEIGHTS.md,
 * measure unpack throughput (x86 proxy), and provide the two pure functions that
 * will later be ported into smollm2_pool_demo.c's pf_worker / initial-load path.
 *
 * Build (host):  gcc -O3 -o int4_proto int4_proto.c -lm
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

/* ---------- tiny fp16 helpers (host) ---------- */
static uint16_t float_to_fp16(float x) {
    uint32_t u; memcpy(&u, &x, 4);
    uint32_t sign = (u >> 16) & 0x8000;
    int32_t  exp  = (u >> 23) & 0xff;
    uint32_t mant = u & 0x7fffff;
    if (exp == 0xff) return (uint16_t)(sign | 0x7c00 | (mant ? 0x200 : 0)); /* inf/nan */
    int32_t e = exp - 127 + 15;
    if (e >= 0x1f) return (uint16_t)(sign | 0x7c00);       /* overflow -> inf */
    if (e <= 0) {
        /* subnormal */
        if (e < -10) return (uint16_t)sign;
        mant |= 0x800000;
        int shift = 14 - e;
        uint32_t half = mant >> shift;
        if (half & 0x200) half += 0x400;                    /* round */
        return (uint16_t)(sign | (half >> 3));
    }
    uint32_t h = sign | ((uint32_t)e << 10) | (mant >> 13);
    if (mant & 0x1000) h += 1;                              /* round */
    return (uint16_t)h;
}
static float fp16_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    uint32_t u;
    if (exp == 0) {
        if (mant == 0) u = sign;
        else { /* subnormal */
            exp = 1;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3ff;
            u = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 0x1f) {
        u = sign | 0x7f800000 | (mant << 13);
    } else {
        u = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f; memcpy(&f, &u, 4);
    return f;
}

/* ---------- pack: int8 row-major -> nibbles + per-group scale ---------- */
/* out_nib:   n/2 bytes
 * out_scale: (n/G) scales (fp32 or fp16)
 * scale_type: 0=fp32, 1=fp16
 * returns number of scales written */
static int int4_pack(const int8_t *src, int n, int G, int scale_type,
                     uint8_t *out_nib, uint8_t *out_scale) {
    int nsc = n / G;
    for (int g = 0; g < nsc; g++) {
        const int8_t *blk = src + (size_t)g * G;
        int maxv = 0;
        for (int i = 0; i < G; i++) { int a = blk[i]; if (a < 0) a = -a; if (a > maxv) maxv = a; }
        float s = (float)maxv / 7.0f;
        if (s < 1e-9f) s = 1.0f;
        /* write scale */
        if (scale_type == 0) memcpy(out_scale + (size_t)g * 4, &s, 4);
        else { uint16_t h = float_to_fp16(s); memcpy(out_scale + (size_t)g * 2, &h, 2); }
        /* write nibbles: q = clamp(round(v/s), -8, 7) */
        uint8_t *nb = out_nib + (size_t)g * G / 2;
        for (int i = 0; i < G; i += 2) {
            int q0 = (int)lrintf((float)blk[i] / s);     if (q0 < -8) q0 = -8; if (q0 > 7) q0 = 7;
            int q1 = (int)lrintf((float)blk[i+1] / s);   if (q1 < -8) q1 = -8; if (q1 > 7) q1 = 7;
            nb[i/2] = (uint8_t)((q0 & 0xF) | ((q1 & 0xF) << 4));
        }
    }
    return nsc;
}

/* ---------- unpack: nibbles + scale -> int8 (dequantized) ---------- */
static void int4_unpack(const uint8_t *src_nib, const uint8_t *src_scale,
                        int n, int G, int scale_type, int8_t *dst) {
    int nsc = n / G;
    for (int g = 0; g < nsc; g++) {
        float s;
        if (scale_type == 0) memcpy(&s, src_scale + (size_t)g * 4, 4);
        else { uint16_t h; memcpy(&h, src_scale + (size_t)g * 2, 2); s = fp16_to_float(h); }
        const uint8_t *nb = src_nib + (size_t)g * G / 2;
        int8_t *d = dst + (size_t)g * G;
        for (int i = 0; i < G; i += 2) {
            uint8_t b = nb[i/2];
            int q0 = (int)(b & 0xF); if (q0 >= 8) q0 -= 16;   /* 8..15 -> -8..-1 */
            int q1 = (int)(b >> 4);  if (q1 >= 8) q1 -= 16;
            int v0 = (int)lrintf((float)q0 * s); if (v0 < -128) v0 = -128; if (v0 > 127) v0 = 127;
            int v1 = (int)lrintf((float)q1 * s); if (v1 < -128) v1 = -128; if (v1 > 127) v1 = 127;
            d[i]   = (int8_t)v0;
            d[i+1] = (int8_t)v1;
        }
    }
}

/* fixed-point unpack (faster on device; s_fixed = round(s * 2^Q), Q=8) */
static void int4_unpack_fixed(const uint8_t *src_nib, const int16_t *src_scale,
                              int n, int G, int8_t *dst) {
    int nsc = n / G;
    for (int g = 0; g < nsc; g++) {
        int16_t sf = src_scale[g];
        const uint8_t *nb = src_nib + (size_t)g * G / 2;
        int8_t *d = dst + (size_t)g * G;
        for (int i = 0; i < G; i += 2) {
            uint8_t b = nb[i/2];
            int q0 = (int)(b & 0xF); if (q0 >= 8) q0 -= 16;
            int q1 = (int)(b >> 4);  if (q1 >= 8) q1 -= 16;
            int v0 = (q0 * sf + 128) >> 8; if (v0 < -128) v0 = -128; if (v0 > 127) v0 = 127;
            int v1 = (q1 * sf + 128) >> 8; if (v1 < -128) v1 = -128; if (v1 > 127) v1 = 127;
            d[i] = (int8_t)v0; d[i+1] = (int8_t)v1;
        }
    }
}

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
