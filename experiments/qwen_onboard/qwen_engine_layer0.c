/* qwen_engine_layer0.c — M2 engine: REAL layer-0 forward on-board probe (TIU).
 *
 * Runs the full layer-0 forward for P0 prompt-1 first token (105538) on the
 * CV1800B TIU with the Path A' engine microkernel:
 *
 *   rms_attn -> per_row_quant -> [q,k,v two-pass TIU] -> bias -> rope(pos0)
 *   -> GQA attention (host, seq=1) -> per_row_quant -> wo TIU -> residual
 *   -> rms_ffn -> per_row_quant -> [up,gate TIU] -> silu -> mid
 *   -> down (K-chunk 1024, per-chunk per-row quant, TIU) -> residual
 *
 * Cross-checks three checkpoints against the trusted host reference
 * (gen_layer0_ref.py -> qwen_engine_layer0_ref.h):
 *   ref_attn / ref_after_wo / ref_after_ffn.
 *
 * Two-pass matmul: pass1 rsafe across ALL N-tiles of a K-block -> block_max ->
 * r_opt -> pass2 rshift=r_opt.  up/gate N=4864 use 6 N-tiles (5x896+384) with
 * block-shared r_opt so the result stays bit-exact vs the reference.  P1/P2
 * bit-exactness and r_opt are verified against a host int32 reference inside
 * eng_matmul (same proof as qwen_engine_tiu.c).
 *
 * Build (riscv64 cross):
 *   riscv64-unknown-linux-musl-gcc -mcpu=c906fdv -march=rv64imafdcv0p7xthead \
 *     -mcmodel=medany -mabi=lp64d -O3 -std=gnu11 -fsigned-char \
 *     -o qwen_engine_layer0 qwen_engine_layer0.c -lm -s
 * Run: python3 ~/Documents/MilkV_duo_project/duo_run.py qwen_engine_layer0
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include "cviruntime_context.h"
#include "bmkernel/bm1822/bmkernel_1822.h"
#include "dequant_kal.c"              /* scalar + RVV K-aligned dequant */
#include "qwen_engine_tiu_data.h"     /* tq_act[896] rms_norm cross-check */
#include "qwen_engine_layer0_ref.h"   /* ref_attn/ref_after_wo/ref_after_ffn */

#define NEURON_SZ 262144
#define ACTQ_OFF  0
#define DQ_OFF    4096
#define P1_OFF    65536
#define P2_OFF    131072

#define D 896
#define H 14
#define KVH 2
#define HD 64
#define F 4864
#define DKV 128
#define G 32
#define GROUPS 7
#define NT_MAX 6
#define ROPE_THETA 1000000.0
#define EPS 1e-6f

#define LAYER_PATH "/data/qwen/layer0_kal.bin"
#define BIAS_PATH  "/data/qwen/layer0_bias.f32"
#define EMBED_PATH "/data/qwen/embed_i8.bin"
#define ESC_PATH   "/data/qwen/embed_scales.f32"

#define TOKEN 105538

static inline double now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

/* ---------------- host semantic helpers (qwen_kal_ref exact) ---------------- */
static inline int8_t int8_round_div(int32_t acc, int rshift) {
    int32_t half = 1 << (rshift - 1);
    int32_t r = (acc + half) >> rshift;
    if (r > 127) r = 127;
    if (r < -128) r = -128;
    return (int8_t)r;
}
static int matmul_rshift_w(int K, int wmax) {
    int r = 0; long long md = (long long)K * 127 * wmax;
    while ((md >> r) > 127) r++;
    return r;
}
static float fp16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16, exp = (h >> 10) & 0x1f, man = h & 0x3ff, f;
    if (exp == 0) { if (man == 0) f = sign; else { exp = 127 - 15 + 1; while (!(man & 0x400)) { man <<= 1; exp--; } man &= 0x3ff; f = sign | (exp << 23) | (man << 13); } }
    else if (exp == 31) f = sign | 0x7f800000u | (man << 13);
    else f = sign | ((exp + (127 - 15)) << 23) | (man << 13);
    float out; memcpy(&out, &f, 4); return out;
}
static inline int32_t round_bankers(float v) {
    float f = floorf(v); float d = v - f;
    if (d > 0.5f) return (int32_t)(f + 1.0f);
    if (d < 0.5f) return (int32_t)f;
    return ((int32_t)f % 2 == 0) ? (int32_t)f : (int32_t)(f + 1.0f);
}
/* sc[m] = max|x|/127 (floor 1e-12); q = clamp(round_bankers(x/sc), -128,127) */
static void per_row_quant(const float *x, int M, int K, int8_t *q, float *sc) {
    for (int m = 0; m < M; m++) {
        const float *xr = x + (size_t)m * K; float mx = 0;
        for (int k = 0; k < K; k++) { float a = fabsf(xr[k]); if (a > mx) mx = a; }
        float s = mx / 127.0f; if (s < 1e-12f) s = 1e-12f; sc[m] = s;
        int8_t *qr = q + (size_t)m * K;
        for (int k = 0; k < K; k++) {
            int32_t ri = round_bankers(xr[k] / s); if (ri > 127) ri = 127; if (ri < -128) ri = -128;
            qr[k] = (int8_t)ri;
        }
    }
}
static void rms_norm(const float *x, const float *g, int n, float *out) {
    double ss = 0; for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    float inv = (float)(1.0 / sqrt(ss / n + EPS));
    for (int i = 0; i < n; i++) out[i] = x[i] * inv * g[i];
}
static inline float silu(float x) { return x / (1.0f + expf(-x)); }

/* ---------------- prebuilt cmdbuf pool (single LoadDmabuf each) ---------------- */
typedef struct {
    int nshape;                        /* right-operand width this pool is for */
    CVI_RT_MEM ld[16][2], src[16][2];  /* [rshift][dest 0=P1 1=P2] */
    CVI_RT_MEM ld_g2l, src_g2l;
} Pool;

static bmk1822_matrix_lmem_shape_t SL = {.n = 1, .c = 1, .w = 32, .col = 32};

static void pool_build(CVI_RT_HANDLE rt, uint64_t pa, int nshape, Pool *p) {
    p->nshape = nshape;
    bmk1822_matrix_lmem_shape_t SR = {.n = 32, .c = 1, .w = (uint32_t)nshape, .col = (uint32_t)nshape};
    bmk1822_matrix_lmem_shape_t SO = {.n = 1, .c = 1, .w = (uint32_t)nshape, .col = (uint32_t)nshape};
    /* g2l: left [1,32] from ACTQ_OFF, right [32,nshape] from DQ_OFF (dense) */
    {
        uint8_t cmdbuf[65536] __attribute__((aligned(16)));
        bmk_info_t info = { .chip_version = 1822, .cmdbuf_size = sizeof(cmdbuf), .cmdbuf = cmdbuf };
        bmk1822_context_t *bmk = bmk1822_register(&info);
        bmk1822_matrix_lmem_t *ml_l = bmk1822_lmem_alloc_matrix(bmk, SL, FMT_I8, 1);
        bmk1822_matrix_lmem_t *ml_r = bmk1822_lmem_alloc_matrix(bmk, SR, FMT_I8, 1);
        bmk1822_matrix_tgmem_t mg_l = {0, ACTQ_OFF, FMT_I8, {1, 32}, {32}};
        bmk1822_matrix_tgmem_t mg_r = {0, DQ_OFF, FMT_I8, {32, (uint32_t)nshape}, {(uint32_t)nshape}};
        bmk1822_tdma_g2l_matrix_copy(bmk, &(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_l, ml_l});
        bmk1822_tdma_g2l_matrix_copy(bmk, &(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_r, ml_r});
        uint32_t cmd_sz; uint8_t *cmd = bmk1822_acquire_cmdbuf(bmk, &cmd_sz);
        uint32_t psize, pmu; bmk1822_dmabuf_size(cmd, cmd_sz, &psize, &pmu);
        p->src_g2l = CVI_RT_MemAlloc(rt, psize);
        uint8_t *db = CVI_RT_MemGetVAddr(p->src_g2l);
        bmk1822_dmabuf_convert(cmd, cmd_sz, db);
        bmk1822_arraybase_set(db, pa, 0, 0, 0);
        CVI_RT_MemFlush(rt, p->src_g2l);
        CVI_RT_LoadDmabuf(rt, p->src_g2l, psize, pa, 0, false, &p->ld_g2l);
        bmk1822_cleanup(bmk);
    }
    for (int r = 0; r < 16; r++) {
        for (int dest = 0; dest < 2; dest++) {
            uint32_t ooff = dest ? P2_OFF : P1_OFF;
            uint8_t cmdbuf[65536] __attribute__((aligned(16)));
            bmk_info_t info = { .chip_version = 1822, .cmdbuf_size = sizeof(cmdbuf), .cmdbuf = cmdbuf };
            bmk1822_context_t *bmk = bmk1822_register(&info);
            bmk1822_matrix_lmem_t *ml_l = bmk1822_lmem_alloc_matrix(bmk, SL, FMT_I8, 1);
            bmk1822_matrix_lmem_t *ml_r = bmk1822_lmem_alloc_matrix(bmk, SR, FMT_I8, 1);
            bmk1822_matrix_lmem_t *ml_o = bmk1822_lmem_alloc_matrix(bmk, SO, FMT_I8, 1);
            bmk1822_matrix_tgmem_t mg_o = {0, ooff, FMT_I8, {1, (uint32_t)nshape}, {(uint32_t)nshape}};
            bmk1822_tiu_matrix_multiplication_param_t mm = {
                .res = ml_o, .left = ml_l, .right = ml_r, .bias = NULL,
                .lshift_bits = 0, .rshift_bits = (uint8_t)r, .res_is_int8 = 1, .relu_enable = 0,
                .add_result = 0, .ps32_mode = 0, .layer_id = 1 };
            bmk1822_tiu_matrix_multiplication(bmk, &mm);
            bmk1822_tdma_l2g_matrix_copy(bmk, &(bmk1822_tdma_l2tg_matrix_copy_param_t){ml_o, &mg_o});
            uint32_t cmd_sz; uint8_t *cmd = bmk1822_acquire_cmdbuf(bmk, &cmd_sz);
            uint32_t psize, pmu; bmk1822_dmabuf_size(cmd, cmd_sz, &psize, &pmu);
            p->src[r][dest] = CVI_RT_MemAlloc(rt, psize);
            uint8_t *db = CVI_RT_MemGetVAddr(p->src[r][dest]);
            bmk1822_dmabuf_convert(cmd, cmd_sz, db);
            bmk1822_arraybase_set(db, pa, 0, 0, 0);
            CVI_RT_MemFlush(rt, p->src[r][dest]);
            CVI_RT_LoadDmabuf(rt, p->src[r][dest], psize, pa, 0, false, &p->ld[r][dest]);
            bmk1822_cleanup(bmk);
        }
    }
}

static Pool *pick_pool(Pool *p128, Pool *p384, Pool *p896, int tn) {
    if (tn <= 128) return p128;
    if (tn <= 384) return p384;
    return p896;
}

/* ---------------- engine two-pass matmul (M=1) ----------------
 * x_i8[K] quantized activation row; nib = K-aligned INT4 [KG][N][16];
 * gscf fp16 [KG*N]; K divisible by G.  out[N] must be pre-zeroed and receives
 * out[n] += (p2<<r_opt)*gscale[n], then all *sc_row at the end.
 * up/gate N=4864 -> 6 N-tiles (5x896+384); r_opt is block-shared (max over
 * ALL tiles of the K-block), matching the reference chunk_matmul_twopass.
 */
static void eng_matmul(CVI_RT_HANDLE rt, CVI_RT_MEM mem, uint64_t pa, uint8_t *va,
                       const char *name, const int8_t *x_i8, const uint8_t *nib, const uint16_t *gscf,
                       int K, int N, float sc_row, float *out,
                       Pool *p128, Pool *p384, Pool *p896,
                       int *bad1, int *bad2, int *rbad) {
    int KG = K / G;
    int wmax = 0;
    for (size_t i = 0; i < (size_t)KG * N * 16; i++) {
        uint8_t b = nib[i];
        int lo = b & 0xF, hi = b >> 4; if (lo > 7) lo -= 16; if (hi > 7) hi -= 16;
        int a = lo < 0 ? -lo : lo, bb = hi < 0 ? -hi : hi;
        if (a > wmax) wmax = a; if (bb > wmax) wmax = bb;
    }
    int rsafe = matmul_rshift_w(G, wmax) - 3; if (rsafe < 4) rsafe = 4;

    int ntiles = (N + 895) / 896;
    int toff[NT_MAX], tn[NT_MAX];
    for (int t = 0; t < ntiles; t++) { toff[t] = t * 896; tn[t] = (t == ntiles - 1) ? N - toff[t] : 896; }

    int32_t acc[896];   /* host int32 reference for one tile */

    for (int g = 0; g < KG; g++) {
        const int8_t *xr = x_i8 + (size_t)g * 32;
        int block_max = 0, gold_max = 0;

        /* ---- pass1 phase: all tiles, rshift=rsafe ---- */
        for (int t = 0; t < ntiles; t++) {
            const uint8_t *nibt = nib + (size_t)g * N * 16 + (size_t)toff[t] * 16;
            dequant_kal_rvv(nibt, tn[t], (int8_t *)(va + DQ_OFF));
            memcpy(va + ACTQ_OFF, xr, 32);
            CVI_RT_MemFlush(rt, mem);
            const int8_t *w = (const int8_t *)(va + DQ_OFF);
            for (int n = 0; n < tn[t]; n++) {
                int32_t s = 0;
                for (int k = 0; k < 32; k++) s += (int32_t)xr[k] * (int32_t)w[(size_t)k * tn[t] + n];
                acc[n] = s;
            }
            Pool *pl = pick_pool(p128, p384, p896, tn[t]);
            CVI_RT_RunCmdbufEx(rt, pl->ld_g2l, &(CVI_RT_ARRAYBASE){.gaddr_base0 = pa});
            CVI_RT_RunCmdbufEx(rt, pl->ld[rsafe][0], &(CVI_RT_ARRAYBASE){.gaddr_base0 = pa});
            CVI_RT_MemInvld(rt, mem);
            int8_t *p1 = (int8_t *)(va + P1_OFF);
            for (int n = 0; n < tn[t]; n++) {
                if (p1[n] != int8_round_div(acc[n], rsafe)) (*bad1)++;
                int av = p1[n]; if (av < 0) av = -av; if (av > block_max) block_max = av;
                int gv = int8_round_div(acc[n], rsafe); if (gv < 0) gv = -gv; if (gv > gold_max) gold_max = gv;
            }
        }
        long long est = (long long)block_max << rsafe;
        int r_opt = 0; while (est > (127LL << r_opt)) r_opt++;
        long long gest = (long long)gold_max << rsafe;
        int r_ref = 0; while (gest > (127LL << r_ref)) r_ref++;
        if (r_opt != r_ref) (*rbad)++;

        /* ---- pass2 phase: all tiles, rshift=r_opt ---- */
        for (int t = 0; t < ntiles; t++) {
            const uint8_t *nibt = nib + (size_t)g * N * 16 + (size_t)toff[t] * 16;
            dequant_kal_rvv(nibt, tn[t], (int8_t *)(va + DQ_OFF));
            memcpy(va + ACTQ_OFF, xr, 32);
            CVI_RT_MemFlush(rt, mem);
            const int8_t *w = (const int8_t *)(va + DQ_OFF);
            for (int n = 0; n < tn[t]; n++) {
                int32_t s = 0;
                for (int k = 0; k < 32; k++) s += (int32_t)xr[k] * (int32_t)w[(size_t)k * tn[t] + n];
                acc[n] = s;
            }
            Pool *pl = pick_pool(p128, p384, p896, tn[t]);
            CVI_RT_RunCmdbufEx(rt, pl->ld_g2l, &(CVI_RT_ARRAYBASE){.gaddr_base0 = pa});
            CVI_RT_RunCmdbufEx(rt, pl->ld[r_opt][1], &(CVI_RT_ARRAYBASE){.gaddr_base0 = pa});
            CVI_RT_MemInvld(rt, mem);
            int8_t *p2 = (int8_t *)(va + P2_OFF);
            float f2 = (float)(1 << r_opt);
            for (int n = 0; n < tn[t]; n++) {
                if (p2[n] != int8_round_div(acc[n], r_opt)) (*bad2)++;
                size_t gidx = (size_t)g * N + toff[t] + n;
                if (gidx >= (size_t)KG * N) {
                    fprintf(stderr, "  [%s] OOB gsc g=%d t=%d n=%d gidx=%zu KG*N=%zu\n",
                            name, g, t, n, gidx, (size_t)KG * N);
                    exit(3);
                }
                float gsc = fp16_to_f32(gscf[gidx]);
                out[(size_t)toff[t] + n] += (float)p2[n] * f2 * gsc;
            }
        }
    }
    for (int n = 0; n < N; n++) out[n] *= sc_row;
}

/* ---------------- comparison helper ---------------- */
static void cmp(const char *tag, const float *got, const float *ref, int n, double thr) {
    double mrel = 0, mabs = 0;
    for (int j = 0; j < n; j++) {
        double rel = fabs((double)ref[j]) > 1e-30 ? fabs((double)got[j] - ref[j]) / fabs((double)ref[j])
                                                  : fabs((double)got[j] - ref[j]);
        if (rel > mrel) mrel = rel;
        double d = fabs((double)got[j] - ref[j]); if (d > mabs) mabs = d;
    }
    printf("%-22s maxrel=%.3e maxabs=%.3e  %s\n", tag, mrel, mabs, (mrel <= thr) ? "OK" : "MISMATCH");
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);
    printf("===== M2 engine layer0 forward (TIU, token %d) =====\n", TOKEN);

    /* ---- layer0 weights ---- */
    FILE *f = fopen(LAYER_PATH, "rb");
    if (!f) { fprintf(stderr, "cannot open %s (deploy weights first)\n", LAYER_PATH); return 2; }
    fseek(f, 0, SEEK_END); long fsz = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *layer = malloc((size_t)fsz);
    if (fread(layer, 1, (size_t)fsz, f) != (size_t)fsz) { fprintf(stderr, "short layer read\n"); return 2; }
    fclose(f);
    printf("layer0_kal.bin: %ld bytes\n", fsz);

    size_t off = 0;
    const float  *rms_attn  = (const float  *)(layer + off); off += D * 4;
    const uint8_t *Wq_nib   = layer + off; off += (size_t)(D / G) * D * 16;
    const uint16_t *Wq_gsc  = (const uint16_t *)(layer + off); off += (size_t)(D / G) * D * 2;
    const uint8_t *Wk_nib   = layer + off; off += (size_t)(D / G) * DKV * 16;
    const uint16_t *Wk_gsc  = (const uint16_t *)(layer + off); off += (size_t)(D / G) * DKV * 2;
    const uint8_t *Wv_nib   = layer + off; off += (size_t)(D / G) * DKV * 16;
    const uint16_t *Wv_gsc  = (const uint16_t *)(layer + off); off += (size_t)(D / G) * DKV * 2;
    const uint8_t *Wo_nib   = layer + off; off += (size_t)(D / G) * D * 16;
    const uint16_t *Wo_gsc  = (const uint16_t *)(layer + off); off += (size_t)(D / G) * D * 2;
    const uint8_t *up_nib   = layer + off; off += (size_t)(D / G) * F * 16;
    const uint16_t *up_gsc  = (const uint16_t *)(layer + off); off += (size_t)(D / G) * F * 2;
    const uint8_t *gate_nib = layer + off; off += (size_t)(D / G) * F * 16;
    const uint16_t *gate_gsc = (const uint16_t *)(layer + off); off += (size_t)(D / G) * F * 2;
    const uint8_t *down_nib = layer + off; off += (size_t)(F / G) * D * 16;
    const uint16_t *down_gsc = (const uint16_t *)(layer + off); off += (size_t)(F / G) * D * 2;
    const float  *rms_ffn   = (const float  *)(layer + off); off += D * 4;
    if (off != (size_t)fsz) { fprintf(stderr, "layout mismatch %zu vs %ld\n", off, fsz); return 2; }
    printf("layer layout ok (%zu bytes)\n", off);

    /* ---- bias ---- */
    float bias[D + DKV + DKV];
    {
        FILE *bf = fopen(BIAS_PATH, "rb");
        if (!bf) { fprintf(stderr, "cannot open %s\n", BIAS_PATH); return 2; }
        if (fread(bias, 4, D + DKV + DKV, bf) != D + DKV + DKV) { fprintf(stderr, "short bias read\n"); return 2; }
        fclose(bf);
    }
    const float *bq = bias, *bk = bias + D, *bv = bias + D + DKV;

    /* ---- embed row + scale (streamed, no full-embed load) ---- */
    int8_t er[D]; float es;
    {
        FILE *ef = fopen(EMBED_PATH, "rb");
        if (!ef) { fprintf(stderr, "cannot open %s\n", EMBED_PATH); return 2; }
        if (fseek(ef, (long)TOKEN * D, SEEK_SET)) { fprintf(stderr, "seek embed\n"); return 2; }
        if (fread(er, 1, D, ef) != D) { fprintf(stderr, "short embed read\n"); return 2; }
        fclose(ef);
    }
    {
        FILE *ef = fopen(ESC_PATH, "rb");
        if (!ef) { fprintf(stderr, "cannot open %s\n", ESC_PATH); return 2; }
        if (fseek(ef, (long)TOKEN * 4, SEEK_SET)) { fprintf(stderr, "seek esc\n"); return 2; }
        if (fread(&es, 4, 1, ef) != 1) { fprintf(stderr, "short esc read\n"); return 2; }
        fclose(ef);
    }
    float x[D];
    for (int j = 0; j < D; j++) x[j] = (float)er[j] * es;
    printf("embed[%d]: x[0..3]=%.6f %.6f %.6f %.6f\n", TOKEN, x[0], x[1], x[2], x[3]);

    /* ---- activation: rms_attn -> quant ---- */
    static float h[D]; static int8_t xi[D]; static float scr[1];
    rms_norm(x, rms_attn, D, h);
    {
        double mrel = 0;
        for (int j = 0; j < D; j++) {
            double rel = fabs(tq_act[j]) > 1e-30 ? fabs((double)h[j] - tq_act[j]) / fabs(tq_act[j]) : fabs((double)h[j] - tq_act[j]);
            if (rel > mrel) mrel = rel;
        }
        printf("rms_attn h vs tq_act (gen_engine_tiu_data): maxrel=%.3e\n", mrel);
    }
    per_row_quant(h, 1, D, xi, scr);
    printf("sc_row=%.6f\n", scr[0]);

    /* ---- ION + prebuilt pools ---- */
    CVI_RT_HANDLE rt; CVI_RT_Init(&rt);
    CVI_RT_MEM mem = CVI_RT_MemAlloc(rt, NEURON_SZ);
    uint64_t pa = CVI_RT_MemGetPAddr(mem);
    uint8_t *va = CVI_RT_MemGetVAddr(mem);
    CVI_RT_SetBaseReg(rt, 0, pa);
    memset(va, 0, NEURON_SZ);

    Pool p128, p384, p896; memset(&p128, 0, sizeof p128); memset(&p384, 0, sizeof p384); memset(&p896, 0, sizeof p896);
    double t0 = now();
    pool_build(rt, pa, 128, &p128);
    pool_build(rt, pa, 384, &p384);
    pool_build(rt, pa, 896, &p896);
    printf("pools built (128/384/896) in %.3fs\n", now() - t0);

    int bad1 = 0, bad2 = 0, rbad = 0;
    static float qb[D], kb[DKV], vb[DKV], attn[D], wob[D], x_after_wo[D];
    static float h2[D], upb[F], gateb[F], mid[F], oout[D], sub[D];
    static int8_t ai[D], x2i[D], mch_i8[1024];
    static float sca[1], sc2[1], mch_sc[1];

    /* ---- QKV ---- */
    t0 = now();
    memset(qb, 0, sizeof qb); eng_matmul(rt, mem, pa, va, "q", xi, Wq_nib, Wq_gsc, D, D,   scr[0], qb, &p128, &p384, &p896, &bad1, &bad2, &rbad);
    memset(kb, 0, sizeof kb); eng_matmul(rt, mem, pa, va, "k", xi, Wk_nib, Wk_gsc, D, DKV, scr[0], kb, &p128, &p384, &p896, &bad1, &bad2, &rbad);
    memset(vb, 0, sizeof vb); eng_matmul(rt, mem, pa, va, "v", xi, Wv_nib, Wv_gsc, D, DKV, scr[0], vb, &p128, &p384, &p896, &bad1, &bad2, &rbad);
    printf("QKV TIU in %.3fs (q/k/v) bad1=%d bad2=%d rbad=%d\n", now() - t0, bad1, bad2, rbad);
    for (int j = 0; j < D; j++) qb[j] += bq[j];
    for (int j = 0; j < DKV; j++) { kb[j] += bk[j]; vb[j] += bv[j]; }
    printf("  q[0..3]=%.6f %.6f %.6f %.6f\n", qb[0], qb[1], qb[2], qb[3]);
    printf("  k[0..3]=%.6f %.6f %.6f %.6f\n", kb[0], kb[1], kb[2], kb[3]);
    printf("  v[0..3]=%.6f %.6f %.6f %.6f\n", vb[0], vb[1], vb[2], vb[3]);

    /* ---- rope pos 0 (identity: cos=1 sin=0) + GQA attention (seq=1) ---- */
    for (int hh = 0; hh < H; hh++) {
        int kvh = hh / GROUPS;
        memcpy(attn + (size_t)hh * HD, vb + (size_t)kvh * HD, HD * sizeof(float));
    }
    cmp("attn vs ref_attn", attn, ref_attn, D, 1e-4);

    /* ---- wo ---- */
    per_row_quant(attn, 1, D, ai, sca);
    t0 = now();
    memset(wob, 0, sizeof wob); eng_matmul(rt, mem, pa, va, "wo", ai, Wo_nib, Wo_gsc, D, D, sca[0], wob, &p128, &p384, &p896, &bad1, &bad2, &rbad);
    printf("wo TIU in %.3fs  bad1=%d bad2=%d rbad=%d\n", now() - t0, bad1, bad2, rbad);
    for (int j = 0; j < D; j++) x_after_wo[j] = x[j] + wob[j];
    cmp("after_wo vs ref", x_after_wo, ref_after_wo, D, 1e-4);

    /* ---- ffn: up / gate ---- */
    rms_norm(x_after_wo, rms_ffn, D, h2);
    per_row_quant(h2, 1, D, x2i, sc2);
    t0 = now();
    memset(upb, 0, sizeof upb); eng_matmul(rt, mem, pa, va, "up", x2i, up_nib, up_gsc, D, F, sc2[0], upb, &p128, &p384, &p896, &bad1, &bad2, &rbad);
    memset(gateb, 0, sizeof gateb); eng_matmul(rt, mem, pa, va, "gate", x2i, gate_nib, gate_gsc, D, F, sc2[0], gateb, &p128, &p384, &p896, &bad1, &bad2, &rbad);
    printf("up/gate TIU in %.3fs bad1=%d bad2=%d rbad=%d\n", now() - t0, bad1, bad2, rbad);
    for (int j = 0; j < F; j++) mid[j] = upb[j] * silu(gateb[j]);

    /* ---- ffn: down (K-chunk 1024, per-chunk per-row quant) ---- */
    t0 = now();
    memset(oout, 0, sizeof oout);
    for (int kc = 0; kc < F; kc += 1024) {
        int kcn = (F - kc < 1024) ? F - kc : 1024;
        float mch[1024];
        memcpy(mch, mid + kc, (size_t)kcn * sizeof(float));
        per_row_quant(mch, 1, kcn, mch_i8, mch_sc);
        const uint8_t  *dnib = down_nib + (size_t)(kc / G) * D * 16;
        const uint16_t *dgsc = down_gsc + (size_t)(kc / G) * D;   /* halfword units */
        memset(sub, 0, sizeof sub);
        eng_matmul(rt, mem, pa, va, "down", mch_i8, dnib, dgsc, kcn, D, mch_sc[0], sub, &p128, &p384, &p896, &bad1, &bad2, &rbad);
        for (int j = 0; j < D; j++) oout[j] += sub[j];
        printf("  down chunk kc=%d kcn=%d done\n", kc, kcn);
    }
    printf("down TIU in %.3fs bad1=%d bad2=%d rbad=%d\n", now() - t0, bad1, bad2, rbad);
    float x_after_ffn[D];
    for (int j = 0; j < D; j++) x_after_ffn[j] = x_after_wo[j] + oout[j];
    cmp("after_ffn vs ref", x_after_ffn, ref_after_ffn, D, 1e-4);

    printf("==== P1/P2 bit-exact: bad1=%d bad2=%d (total blocks-checked)  r_opt mismatches=%d ====\n",
           bad1, bad2, rbad);
    printf("==== layer0 forward %s ====\n",
           (bad1 + bad2 + rbad == 0) ? "ALL BIT-EXACT" : "HAS MISMATCHES");

    free(layer);
    for (int r = 0; r < 16; r++) { for (int d = 0; d < 2; d++) { if (p128.src[r][d]) CVI_RT_MemFree(rt, p128.src[r][d]); if (p384.src[r][d]) CVI_RT_MemFree(rt, p384.src[r][d]); if (p896.src[r][d]) CVI_RT_MemFree(rt, p896.src[r][d]); } }
    if (p128.src_g2l) CVI_RT_MemFree(rt, p128.src_g2l);
    if (p384.src_g2l) CVI_RT_MemFree(rt, p384.src_g2l);
    if (p896.src_g2l) CVI_RT_MemFree(rt, p896.src_g2l);
    CVI_RT_MemFree(rt, mem); CVI_RT_DeInit(rt);
    return (bad1 + bad2 + rbad) ? 1 : 0;
}
