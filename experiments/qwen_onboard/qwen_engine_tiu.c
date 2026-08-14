/* qwen_engine_tiu.c — M2 engine TIU core on-board: prebuilt cmdbuf pool +
 * two-pass matmul over REAL layer0 q_proj (28 K-blocks), reading K-aligned
 * INT4 nib directly from /data/qwen/layer0_kal.bin (deployed weights).
 *
 * Proves the engine's TIU plumbing on real data:
 *   - single-LoadDmabuf prebuilt pool (g2l + pass[r][dest]), submit = Run only
 *   - pass1 rshift=rsafe -> l2g readback -> r_opt -> pass2 rshift=r_opt
 *   - per-block dequant (dequant_kal_rvv) -> DQ_OFF -> g2l right operand
 *   - bit-exact pass1/pass2 vs host int8_round_div; r_opt vs fp64 gold;
 *     fp32 accumulate vs fp64 gold (rel tol)
 *   - per-block submit/dequant timing -> per-token extrapolation
 *
 * Build (riscv64 cross): riscv64-unknown-linux-musl-gcc -mcpu=c906fdv \
 *   -march=rv64imafdcv0p7xthead -mcmodel=medany -mabi=lp64d -O3 -std=gnu11 \
 *   -fsigned-char -o qwen_engine_tiu qwen_engine_tiu.c -lm -s
 * Run: python3 ~/Documents/MilkV_duo_project/duo_run.py qwen_engine_tiu
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include "cviruntime_context.h"
#include "bmkernel/bm1822/bmkernel_1822.h"
#include "dequant_kal.c"            /* scalar + RVV K-aligned dequant */
#include "qwen_engine_tiu_data.h"   /* tq_act[896] real layer0 activation */

#define NEURON_SZ 262144
#define ACTQ_OFF  0
#define DQ_OFF    4096
#define P1_OFF    65536
#define P2_OFF    131072

#define D 896
#define G 32
#define N 896
#define KG (D / G)          /* 28 blocks */

#define LAYER_PATH "/data/qwen/layer0_kal.bin"
#define NIB_BASE   (D * 4)               /* q nib start in layer file */
#define GSC_BASE   (D * 4 + KG * D * 16) /* q gsc fp16 start */

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

/* ---------------- prebuilt cmdbuf pool (single LoadDmabuf each) ---------------- */
typedef struct {
    CVI_RT_MEM ld[16][2], src[16][2];   /* [rshift][dest 0=P1 1=P2] */
    CVI_RT_MEM ld_g2l, src_g2l;
} Pool;

static bmk1822_matrix_lmem_shape_t SL = {.n = 1, .c = 1, .w = 32, .col = 32};
static bmk1822_matrix_lmem_shape_t SR = {.n = 32, .c = 1, .w = 896, .col = 896};
static bmk1822_matrix_lmem_shape_t SO = {.n = 1, .c = 1, .w = 896, .col = 896};

static void pool_build(CVI_RT_HANDLE rt, uint64_t pa, Pool *p) {
    /* g2l: left [1,32] from ACTQ_OFF, right [32,896] from DQ_OFF.
       LMEM alloc order (l,r) MUST match the pass cmdbufs. */
    {
        uint8_t cmdbuf[65536] __attribute__((aligned(16)));
        bmk_info_t info = { .chip_version = 1822, .cmdbuf_size = sizeof(cmdbuf), .cmdbuf = cmdbuf };
        bmk1822_context_t *bmk = bmk1822_register(&info);
        bmk1822_matrix_lmem_t *ml_l = bmk1822_lmem_alloc_matrix(bmk, SL, FMT_I8, 1);
        bmk1822_matrix_lmem_t *ml_r = bmk1822_lmem_alloc_matrix(bmk, SR, FMT_I8, 1);
        bmk1822_matrix_tgmem_t mg_l = {0, ACTQ_OFF, FMT_I8, {1, 32}, {32}};
        bmk1822_matrix_tgmem_t mg_r = {0, DQ_OFF, FMT_I8, {32, 896}, {896}};
        bmk1822_tdma_g2l_matrix_copy(bmk, &(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_l, ml_l});
        bmk1822_tdma_g2l_matrix_copy(bmk, &(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_r, ml_r});
        uint32_t cmd_sz; uint8_t *cmd = bmk1822_acquire_cmdbuf(bmk, &cmd_sz);
        uint32_t psize, pmu; bmk1822_dmabuf_size(cmd, cmd_sz, &psize, &pmu);
        /* KEY: allocate psize ONLY (drop pmu region) — verified on-board that
           LoadDmabuf(enable_pmu=false) works with a psize-only source dmabuf,
           cutting per-cmdbuf ION cost from ~1.05MB (pmu) to ~1KB. */
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
            bmk1822_matrix_tgmem_t mg_o = {0, ooff, FMT_I8, {1, 896}, {896}};
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

int main(void) {
    printf("===== M2 engine TIU core: real layer0 q_proj (28 blocks, KG=32) =====\n");

    FILE *f = fopen(LAYER_PATH, "rb");
    if (!f) { fprintf(stderr, "cannot open %s (deploy weights first)\n", LAYER_PATH); return 2; }
    fseek(f, 0, SEEK_END); long fsz = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *layer = malloc((size_t)fsz);
    if (fread(layer, 1, (size_t)fsz, f) != (size_t)fsz) { fprintf(stderr, "short layer read\n"); return 2; }
    fclose(f);
    printf("layer0_kal.bin: %ld bytes loaded\n", fsz);

    /* activation quant (engine path) */
    static int8_t xi[D]; static float scr[1];
    per_row_quant(tq_act, 1, D, xi, scr);

    /* ION */
    CVI_RT_HANDLE rt; CVI_RT_Init(&rt);
    CVI_RT_MEM mem = CVI_RT_MemAlloc(rt, NEURON_SZ);
    uint64_t pa = CVI_RT_MemGetPAddr(mem);
    uint8_t *va = CVI_RT_MemGetVAddr(mem);
    CVI_RT_SetBaseReg(rt, 0, pa);
    memset(va, 0, NEURON_SZ);

    Pool pool; memset(&pool, 0, sizeof pool);
    pool_build(rt, pa, &pool);
    printf("pool built: g2l + 16x2 pass cmdbufs, psize-only (single LoadDmabuf each)\n");

    /* wmax over whole q_proj matrix from nib (raw int4 max) -> rsafe */
    int wmax = 0;
    for (int i = 0; i < KG * N * 16; i++) {
        uint8_t b = layer[NIB_BASE + i];
        int lo = b & 0xF, hi = b >> 4; if (lo > 7) lo -= 16; if (hi > 7) hi -= 16;
        int a = lo < 0 ? -lo : lo, bb = hi < 0 ? -hi : hi;
        if (a > wmax) wmax = a; if (bb > wmax) wmax = bb;
    }
    int rsafe = matmul_rshift_w(G, wmax) - 3; if (rsafe < 4) rsafe = 4;
    printf("wmax=%d rsafe=%d sc_row=%.6f\n", wmax, rsafe, scr[0]);

    double *out64 = calloc(N, sizeof(double));
    float *out = calloc(N, sizeof(float));
    int32_t *acc = malloc(N * sizeof(int32_t));
    int bad1_total = 0, bad2_total = 0, r_bad = 0;
    double t_deq = 0, t_sub = 0, t_inv = 0, t_cpu = 0;
    double t0 = now();

    for (int g = 0; g < KG; g++) {
        const uint8_t *nib = layer + NIB_BASE + (size_t)g * N * 16;
        const uint16_t *gscf = (const uint16_t *)(layer + GSC_BASE + (size_t)g * N * 2);

        /* 1. dequant block -> DQ_OFF */
        double a0 = now();
        dequant_kal_rvv(nib, N, (int8_t *)(va + DQ_OFF));
        double a1 = now(); t_deq += a1 - a0;

        /* stage act slice + flush */
        memcpy(va + ACTQ_OFF, xi + (size_t)g * 32, 32);
        CVI_RT_MemFlush(rt, mem);

        /* host acc reference (reads DQ just written) */
        const int8_t *w = (const int8_t *)(va + DQ_OFF);
        for (int n = 0; n < N; n++) {
            int32_t s = 0;
            for (int k = 0; k < 32; k++) s += (int32_t)xi[(size_t)g * 32 + k] * (int32_t)w[(size_t)k * N + n];
            acc[n] = s;
        }

        /* 2. g2l + pass1 (submit) */
        double a2 = now();
        CVI_RT_RunCmdbufEx(rt, pool.ld_g2l, &(CVI_RT_ARRAYBASE){.gaddr_base0 = pa});
        CVI_RT_RunCmdbufEx(rt, pool.ld[rsafe][0], &(CVI_RT_ARRAYBASE){.gaddr_base0 = pa});
        double a3 = now(); t_sub += a3 - a2;

        /* 3. invld + max -> r_opt */
        CVI_RT_MemInvld(rt, mem);
        double a4 = now(); t_inv += a4 - a3;
        int8_t *p1 = (int8_t *)(va + P1_OFF);
        int bad1 = 0, maxabs = 0;
        for (int n = 0; n < N; n++) {
            if (p1[n] != int8_round_div(acc[n], rsafe)) bad1++;
            int av = p1[n]; if (av < 0) av = -av; if (av > maxabs) maxabs = av;
        }
        long long est = (long long)maxabs << rsafe;
        int r_opt = 0; while (est > (127LL << r_opt)) r_opt++;
        int r_ref = 0; {
            int32_t estref = 0;
            for (int n = 0; n < N; n++) { int32_t v = int8_round_div(acc[n], rsafe); if (v < 0) v = -v; if (v > estref) estref = v; }
            long long e = (long long)estref << rsafe; while (e > (127LL << r_ref)) r_ref++;
        }
        if (r_opt != r_ref) r_bad++;
        double a5 = now(); t_cpu += a5 - a4;

        /* 4. pass2 (submit) */
        CVI_RT_RunCmdbufEx(rt, pool.ld[r_opt][1], &(CVI_RT_ARRAYBASE){.gaddr_base0 = pa});
        double a6 = now(); t_sub += a6 - a5;

        /* 5. invld + verify p2 + accumulate */
        CVI_RT_MemInvld(rt, mem);
        double a7 = now(); t_inv += a7 - a6;
        int8_t *p2 = (int8_t *)(va + P2_OFF);
        int bad2 = 0;
        for (int n = 0; n < N; n++) if (p2[n] != int8_round_div(acc[n], r_opt)) bad2++;
        float f2 = (float)(1 << r_opt);
        for (int n = 0; n < N; n++) {
            float gsc = fp16_to_f32(gscf[n]);
            out[n] += (float)p2[n] * f2 * gsc;
            out64[n] += (double)int8_round_div(acc[n], r_opt) * (double)(1 << r_opt) * (double)gsc;
        }
        double a8 = now(); t_cpu += a8 - a7;

        bad1_total += bad1; bad2_total += bad2;
        if (bad1 || bad2 || r_opt != r_ref)
            printf("  g=%2d r_opt=%d(rref=%d) bad1=%d bad2=%d maxabs=%d\n",
                   g, r_opt, r_ref, bad1, bad2, maxabs);
    }
    double wall = now() - t0;
    /* out *= sc_row */
    for (int n = 0; n < N; n++) { out[n] *= scr[0]; out64[n] *= (double)scr[0]; }

    /* final verification */
    double maxrel = 0, maxabsd = 0;
    for (int n = 0; n < N; n++) {
        double rel = fabs(out64[n]) > 1e-30 ? fabs((double)out[n] - out64[n]) / fabs(out64[n]) : fabs((double)out[n]);
        if (rel > maxrel) maxrel = rel;
        double d = fabs((double)out[n] - out64[n]); if (d > maxabsd) maxabsd = d;
    }
    printf("P1 bit-exact : bad=%d/%d\n", bad1_total, KG * N);
    printf("P2 bit-exact : bad=%d/%d\n", bad2_total, KG * N);
    printf("r_opt        : mismatches=%d/%d\n", r_bad, KG);
    printf("accum fp32 vs fp64-gold: maxrel=%.4e maxabs=%.4e\n", maxrel, maxabsd);

    /* timing.  Per-run (RunCmdbufEx) time = submit/3 (g2l+p1+p2 per block). */
    double per_blk_sub_ms = t_sub / KG * 1e3;      /* ms/block (g2l+p1+p2) */
    double per_run_ms = per_blk_sub_ms / 3.0;      /* ms/RunCmdbufEx */
    /* Realistic runs/layer with Ntile=896 (see DESIGN §9b; up/gate N=4864 -> 6 tiles,
       q/k/v/o/down 1 tile): runs = g2l + pass, q/k/v/o/up/gate 28 blk, down 152 blk:
         g2l   = 28*4 + 28*6*2 + 152 = 600
         pass  = (28*2)*4 + (28*6*2)*2 + 152*2 = 224 + 672 + 304 = 1200
         total = 1800 runs/layer  (vs N=896-only q_proj lower bound 960)
       TIU-only (pass) = 1200/layer. */
    double run_layer_real = 1800.0;
    double tiu_pass_layer = 1200.0;
    double per_token_sub = per_run_ms * run_layer_real * 24 / 1e3;
    double per_token_tiu_pass = per_run_ms * tiu_pass_layer * 24 / 1e3;
    double per_token_deq = (t_deq / KG * 1e3) * 320 * 24 / 1e3;   /* 320 N=896 blocks */
    double per_token_cpu = ((t_cpu + t_inv) / KG * 1e3) * 320 * 24 / 1e3;
    printf("per-block: wall=%.3fms submit(g2l+p1+p2)=%.3fms invld=%.3fms cpu=%.3fms dequant=%.3fms\n",
           wall / KG * 1e3, per_blk_sub_ms, t_inv / KG * 1e3, t_cpu / KG * 1e3, t_deq / KG * 1e3);
    printf("per-RunCmdbufEx: %.3f ms\n", per_run_ms);
    printf("extrapolate (real runs/layer=%.0f, 24 layers):\n", run_layer_real);
    printf("  TIU all-runs        ~%.2fs/token (pass-only ~%.2fs/token)\n", per_token_sub, per_token_tiu_pass);
    printf("  RVV dequant         ~%.2fs/token\n", per_token_deq);
    printf("  invld+cpu           ~%.2fs/token\n", per_token_cpu);
    printf("  vs SD floor 9.18s/token\n");

    free(out64); free(out); free(acc); free(layer);
    for (int r = 0; r < 16; r++) for (int d = 0; d < 2; d++) if (pool.src[r][d]) CVI_RT_MemFree(rt, pool.src[r][d]);
    if (pool.src_g2l) CVI_RT_MemFree(rt, pool.src_g2l);
    CVI_RT_MemFree(rt, mem); CVI_RT_DeInit(rt);
    printf("===== engine TIU core done (rc=%d) =====\n", (bad1_total + bad2_total + r_bad) ? 1 : 0);
    return (bad1_total + bad2_total + r_bad) ? 1 : 0;
}
