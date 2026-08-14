/* qwen_engine_24l.c — M2 milestone: 24-layer full-model prefill + LM head (TIU Path A').
 *
 * On-board Qwen2.5-0.5B forward for the 3-prompt regression.  Each prompt is run
 * as an M=seq BATCH (seq = 3/5/7) through all 24 layers, matching the host C
 * reference qwen_kal_ref.c numerics exactly:
 *
 *   per token: embed[t]*esc[t]  -> x[seq,D]
 *   layer l (reads /data/qwen/layerN_kal.bin, 8.39MB):
 *     rms_attn -> per_row_quant -> [q,k,v TIU two-pass] + bias -> rope(pos)
 *     -> GQA attention over the batch (causal) -> per_row_quant -> wo TIU
 *     -> residual -> rms_ffn -> per_row_quant -> [up,gate TIU] -> SiLU -> mid
 *     -> down (K-chunk 1024, per-chunk per-row quant, TIU) -> residual
 *   final rms_norm(last row) -> LM head: stream embed_i8.bin (136MB) in chunks,
 *     double-precision dot with embed_scales -> top-5 / gap.
 *
 * Matmul microkernel = the layer0 engine generalized to M=seq:
 *   - two-pass per K-block: pass1 rsafe (all N-tiles) -> block_max over ALL M
 *     rows and ALL tiles -> r_opt -> pass2 rshift=r_opt (block-shared).
 *   - host int32 acc checked vs TIU P1/P2 (bad1/bad2) and r_opt (rbad).
 *   - K-block contributions accumulated in double, out = (float)accd * sc_row[m]
 *     (exactly qwen_kal_ref.c chunk_matmul_twopass semantics).
 *   - N-tile width pool = {128,256,384,512} per M (exact tile match so the g2l
 *     right operand never reads past the dequantized tile).
 *
 * Build (riscv64 cross):
 *   riscv64-unknown-linux-musl-gcc -mcpu=c906fdv -march=rv64imafdcv0p7xthead \
 *     -mcmodel=medany -mabi=lp64d -O3 -std=gnu11 -fsigned-char \
 *     -I <tpu>/include -o qwen_engine_24l qwen_engine_24l.c -lm -s \
 *     -L <tpu>/lib -lcviruntime
 * Run: python3 ~/Documents/MilkV_duo_project/duo_run.py qwen_engine_24l --timeout 600
 * Deps: /data/qwen/layer0..23_kal.bin + layer0..23_bias.f32 + embed_i8.bin +
 *       embed_scales.f32 + final_rms.f32 (all deployed).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>
#include "cviruntime_context.h"
#include "bmkernel/bm1822/bmkernel_1822.h"
#include "dequant_kal.c"              /* scalar + RVV K-aligned dequant */

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
#define L 24
#define V 151936
#define MS 7            /* max prompt seq in the 3-prompt regression */
#define MAX_SEQ 64
#define TILEW 512       /* max N-tile width (M>=2 pools: 128/256/384/512) */
#define MAXT 10         /* ceil(F/TILEW) = 10 */
#define ROPE_THETA 1000000.0
#define EPS 1e-6f

#define WDIR "/data/qwen"

#define EMBED_PATH WDIR "/embed_i8.bin"
#define ESC_PATH   WDIR "/embed_scales.f32"
#define FRMS_PATH  WDIR "/final_rms.f32"

/* Host int32 reference cross-check of TIU P1/P2 (bad1/bad2/rbad).  Proven
 * bit-exact for the full run; can be disabled for the latency-only path
 * (compile with -DVERIFY=0). */
#ifndef VERIFY
#define VERIFY 1
#endif

static inline double now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

/* ---------------- host semantic helpers (qwen_kal_ref exact) ---------------- */
/* TIU-style round: sat8((acc + 2^(r-1)) >> r).  Handles rshift=0 (scale=1). */
static inline int8_t int8_round_div(int32_t acc, int rshift) {
    int32_t scale = 1 << rshift;
    int32_t half = scale >> 1;
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
/* numpy np.round parity: round-half-to-even (banker's) */
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
static void rms_norm(const float *x, int seq, const float *g, int n, float *out) {
    for (int m = 0; m < seq; m++) {
        const float *xr = x + (size_t)m * n; float *or_ = out + (size_t)m * n;
        double ss = 0; for (int i = 0; i < n; i++) ss += (double)xr[i] * xr[i];
        float inv = (float)(1.0 / sqrt(ss / n + EPS));
        for (int i = 0; i < n; i++) or_[i] = xr[i] * inv * g[i];
    }
}
static inline float silu(float x) { return x / (1.0f + expf(-x)); }
static void rope_inplace(float *q, int pos, const float *cos, const float *sin) {
    int half = HD / 2;
    for (int i = 0; i < half; i++) {
        float x0 = q[i], x1 = q[half + i];
        float c = cos[(size_t)pos * half + i], s = sin[(size_t)pos * half + i];
        q[i] = x0 * c - x1 * s;
        q[half + i] = x0 * s + x1 * c;
    }
}

/* ---------------- prebuilt cmdbuf pool (single LoadDmabuf each) ----------------
 * Each cmdbuf is COMBINED: g2l(left)+g2l(right)+TIU matmul+l2g(out) in one
 * submission, so one RunCmdbufEx per (K-block, tile, pass) does the whole job
 * (halves the submit overhead vs a separate g2l cmdbuf).
 */
typedef struct {
    int nshape;
    CVI_RT_MEM ld[16][2], src[16][2];  /* [rshift][dest 0=P1 1=P2] */
} Pool;

static void pool_build(CVI_RT_HANDLE rt, uint64_t pa, int M, int nshape, Pool *p) {
    p->nshape = nshape;
    bmk1822_matrix_lmem_shape_t SL = {.n = (uint32_t)M, .c = 1, .w = 32, .col = 32};
    bmk1822_matrix_lmem_shape_t SR = {.n = 32, .c = 1, .w = (uint32_t)nshape, .col = (uint32_t)nshape};
    bmk1822_matrix_lmem_shape_t SO = {.n = (uint32_t)M, .c = 1, .w = (uint32_t)nshape, .col = (uint32_t)nshape};
    for (int r = 0; r < 16; r++) {
        for (int dest = 0; dest < 2; dest++) {
            uint32_t ooff = dest ? P2_OFF : P1_OFF;
            uint8_t cmdbuf[65536] __attribute__((aligned(16)));
            bmk_info_t info = { .chip_version = 1822, .cmdbuf_size = sizeof(cmdbuf), .cmdbuf = cmdbuf };
            bmk1822_context_t *bmk = bmk1822_register(&info);
            bmk1822_matrix_lmem_t *ml_l = bmk1822_lmem_alloc_matrix(bmk, SL, FMT_I8, 1);
            bmk1822_matrix_lmem_t *ml_r = bmk1822_lmem_alloc_matrix(bmk, SR, FMT_I8, 1);
            bmk1822_matrix_lmem_t *ml_o = bmk1822_lmem_alloc_matrix(bmk, SO, FMT_I8, 1);
            bmk1822_matrix_tgmem_t mg_l = {0, ACTQ_OFF, FMT_I8, {M, 32}, {32}};
            bmk1822_matrix_tgmem_t mg_r = {0, DQ_OFF, FMT_I8, {32, nshape}, {nshape}};
            bmk1822_matrix_tgmem_t mg_o = {0, ooff, FMT_I8, {M, nshape}, {nshape}};
            bmk1822_tdma_g2l_matrix_copy(bmk, &(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_l, ml_l});
            bmk1822_tdma_g2l_matrix_copy(bmk, &(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_r, ml_r});
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

/* Per-M pool set.  Only the widths actually used per M are built:
 *   M=3 : 128, 384, 896   (q/wo/down N=896 -> 1 tile; up/gate 5x896+384)
 *   M=5/7: 128, 256, 768  (N=896 -> 768+128; up/gate 6x768+256)
 */
typedef struct { Pool p128, p256, p384, p768, p896; } PoolSet;

static int max_tile_for_m(int M) { return (M <= 3) ? 896 : 768; }

static Pool *pick_pool(PoolSet *ps, int tn) {
    switch (tn) {
        case 128: return &ps->p128;
        case 256: return &ps->p256;
        case 384: return &ps->p384;
        case 768: return &ps->p768;
        case 896: return &ps->p896;
        default:  fprintf(stderr, "no pool for tn=%d\n", tn); exit(4);
    }
}

/* ---------------- engine two-pass matmul (M=seq) ----------------
 * x_i8[M,K]; nib = K-aligned INT4 [KG][N][16]; gscf fp16 [KG*N]; K divisible by G.
 * r_opt is block-shared: max |pass1| over ALL M rows and ALL N-tiles of a K-block.
 * K-block contributions accumulate in double accd, then out = (float)accd*sc_row[m].
 */
static double accd[MS * F];
static int32_t hacc[MS * 896];
static long g_runs_pass1 = 0, g_runs_pass2 = 0;

static void eng_matmul(CVI_RT_HANDLE rt, CVI_RT_MEM mem, uint64_t pa, uint8_t *va,
                       const char *name, const int8_t *x_i8, int M,
                       const uint8_t *nib, const uint16_t *gscf,
                       int K, int N, const float *sc_row, float *out,
                       PoolSet *ps, int *bad1, int *bad2, int *rbad) {
    int KG = K / G;
    int wmax = 0;
    for (size_t i = 0; i < (size_t)KG * N * 16; i++) {
        uint8_t b = nib[i];
        int lo = b & 0xF, hi = b >> 4; if (lo > 7) lo -= 16; if (hi > 7) hi -= 16;
        int a = lo < 0 ? -lo : lo, bb = hi < 0 ? -hi : hi;
        if (a > wmax) wmax = a; if (bb > wmax) wmax = bb;
    }
    int rsafe = matmul_rshift_w(G, wmax) - 3; if (rsafe < 4) rsafe = 4;

    int tilew = max_tile_for_m(M);
    int ntiles = (N + tilew - 1) / tilew;
    int toff[MAXT], tn[MAXT];
    for (int t = 0; t < ntiles; t++) { toff[t] = t * tilew; tn[t] = (t == ntiles - 1) ? N - toff[t] : tilew; }

    memset(accd, 0, sizeof(double) * (size_t)M * N);

    for (int g = 0; g < KG; g++) {
        int block_max = 0;
#if VERIFY
        int gold_max = 0;
#endif

        /* ---- pass1 phase: all tiles, rshift=rsafe ---- */
        for (int t = 0; t < ntiles; t++) {
            const uint8_t *nibt = nib + (size_t)g * N * 16 + (size_t)toff[t] * 16;
            dequant_kal_rvv(nibt, tn[t], (int8_t *)(va + DQ_OFF));
            for (int m = 0; m < M; m++)
                memcpy(va + ACTQ_OFF + (size_t)m * 32, x_i8 + (size_t)m * K + (size_t)g * 32, 32);
#if VERIFY
            const int8_t *w = (const int8_t *)(va + DQ_OFF);
            for (int m = 0; m < M; m++) {
                const int8_t *xr = x_i8 + (size_t)m * K + (size_t)g * 32;
                int32_t *ar = hacc + (size_t)m * tn[t];
                for (int n = 0; n < tn[t]; n++) {
                    int32_t s = 0;
                    for (int k = 0; k < 32; k++) s += (int32_t)xr[k] * (int32_t)w[(size_t)k * tn[t] + n];
                    ar[n] = s;
                }
            }
#endif
            CVI_RT_MemFlushEx(rt, mem, DQ_OFF + (size_t)32 * tn[t]);
            Pool *pl = pick_pool(ps, tn[t]);
            CVI_RT_RunCmdbufEx(rt, pl->ld[rsafe][0], &(CVI_RT_ARRAYBASE){.gaddr_base0 = pa});
            g_runs_pass1++;
            CVI_RT_MemInvldEx(rt, mem, P2_OFF + (size_t)M * tn[t]);
            int8_t *p1 = (int8_t *)(va + P1_OFF);
            for (int m = 0; m < M; m++) {
                for (int n = 0; n < tn[t]; n++) {
                    int av = p1[(size_t)m * tn[t] + n]; if (av < 0) av = -av; if (av > block_max) block_max = av;
#if VERIFY
                    int gv = int8_round_div(hacc[(size_t)m * tn[t] + n], rsafe);
                    if (p1[(size_t)m * tn[t] + n] != gv) (*bad1)++;
                    if (gv < 0) gv = -gv; if (gv > gold_max) gold_max = gv;
#endif
                }
            }
        }
        long long est = (long long)block_max << rsafe;
        int r_opt = 0; while (est > (127LL << r_opt)) r_opt++;
#if VERIFY
        long long gest = (long long)gold_max << rsafe;
        int r_ref = 0; while (gest > (127LL << r_ref)) r_ref++;
        if (r_opt != r_ref) (*rbad)++;
#endif

        /* ---- pass2 phase: all tiles, rshift=r_opt ---- */
        for (int t = 0; t < ntiles; t++) {
            const uint8_t *nibt = nib + (size_t)g * N * 16 + (size_t)toff[t] * 16;
            dequant_kal_rvv(nibt, tn[t], (int8_t *)(va + DQ_OFF));
            for (int m = 0; m < M; m++)
                memcpy(va + ACTQ_OFF + (size_t)m * 32, x_i8 + (size_t)m * K + (size_t)g * 32, 32);
#if VERIFY
            const int8_t *w = (const int8_t *)(va + DQ_OFF);
            for (int m = 0; m < M; m++) {
                const int8_t *xr = x_i8 + (size_t)m * K + (size_t)g * 32;
                int32_t *ar = hacc + (size_t)m * tn[t];
                for (int n = 0; n < tn[t]; n++) {
                    int32_t s = 0;
                    for (int k = 0; k < 32; k++) s += (int32_t)xr[k] * (int32_t)w[(size_t)k * tn[t] + n];
                    ar[n] = s;
                }
            }
#endif
            CVI_RT_MemFlushEx(rt, mem, DQ_OFF + (size_t)32 * tn[t]);
            Pool *pl = pick_pool(ps, tn[t]);
            CVI_RT_RunCmdbufEx(rt, pl->ld[r_opt][1], &(CVI_RT_ARRAYBASE){.gaddr_base0 = pa});
            g_runs_pass2++;
            CVI_RT_MemInvldEx(rt, mem, P2_OFF + (size_t)M * tn[t]);
            int8_t *p2 = (int8_t *)(va + P2_OFF);
            double f2 = (double)(1 << r_opt);
            for (int m = 0; m < M; m++) {
                for (int n = 0; n < tn[t]; n++) {
#if VERIFY
                    if (p2[(size_t)m * tn[t] + n] != int8_round_div(hacc[(size_t)m * tn[t] + n], r_opt)) (*bad2)++;
#endif
                    size_t gidx = (size_t)g * N + toff[t] + n;
                    if (gidx >= (size_t)KG * N) {
                        fprintf(stderr, "  [%s] OOB gsc g=%d t=%d n=%d gidx=%zu KG*N=%zu\n",
                                name, g, t, n, gidx, (size_t)KG * N);
                        exit(3);
                    }
                    float gsc = fp16_to_f32(gscf[gidx]);
                    accd[(size_t)m * N + toff[t] + n] +=
                        (double)p2[(size_t)m * tn[t] + n] * f2 * (double)gsc;
                }
            }
        }
    }
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++)
            out[(size_t)m * N + n] = (float)accd[(size_t)m * N + n] * sc_row[m];
}

/* ---------------- layer weight parsing ---------------- */
typedef struct {
    const float *rms_attn, *rms_ffn;
    const uint8_t *Wq_nib, *Wk_nib, *Wv_nib, *Wo_nib, *up_nib, *gate_nib, *down_nib;
    const uint16_t *Wq_gsc, *Wk_gsc, *Wv_gsc, *Wo_gsc, *up_gsc, *gate_gsc, *down_gsc;
} LayerRef;

static void parse_layer(const uint8_t *layer, size_t lsz, LayerRef *lr) {
    size_t off = 0;
    lr->rms_attn = (const float *)(layer + off); off += D * 4;
    lr->Wq_nib = layer + off; off += (size_t)(D / G) * D * 16;
    lr->Wq_gsc = (const uint16_t *)(layer + off); off += (size_t)(D / G) * D * 2;
    lr->Wk_nib = layer + off; off += (size_t)(D / G) * DKV * 16;
    lr->Wk_gsc = (const uint16_t *)(layer + off); off += (size_t)(D / G) * DKV * 2;
    lr->Wv_nib = layer + off; off += (size_t)(D / G) * DKV * 16;
    lr->Wv_gsc = (const uint16_t *)(layer + off); off += (size_t)(D / G) * DKV * 2;
    lr->Wo_nib = layer + off; off += (size_t)(D / G) * D * 16;
    lr->Wo_gsc = (const uint16_t *)(layer + off); off += (size_t)(D / G) * D * 2;
    lr->up_nib = layer + off; off += (size_t)(D / G) * F * 16;
    lr->up_gsc = (const uint16_t *)(layer + off); off += (size_t)(D / G) * F * 2;
    lr->gate_nib = layer + off; off += (size_t)(D / G) * F * 16;
    lr->gate_gsc = (const uint16_t *)(layer + off); off += (size_t)(D / G) * F * 2;
    lr->down_nib = layer + off; off += (size_t)(F / G) * D * 16;
    lr->down_gsc = (const uint16_t *)(layer + off); off += (size_t)(F / G) * D * 2;
    lr->rms_ffn = (const float *)(layer + off); off += D * 4;
    if (off != lsz) { fprintf(stderr, "layer layout mismatch %zu vs %zu\n", off, lsz); exit(2); }
}

/* ---------------- GQA attention (M=seq, causal) ---------------- */
static void attention(int seq, const float *qbuf, const float *kbuf, const float *vbuf, float *attn) {
    for (int hh = 0; hh < H; hh++) {
        int kvh = hh / GROUPS;
        for (int m = 0; m < seq; m++) {
            const float *qm = qbuf + (size_t)m * D + (size_t)hh * HD;
            const float *km = kbuf + (size_t)kvh * HD;
            const float *vm = vbuf + (size_t)kvh * HD;
            float lgm[MAX_SEQ], mx = -1e30f;
            for (int s = 0; s < seq; s++) {
                float v = (s > m) ? -1e30f : 0.0f;
                if (s <= m) {
                    const float *ks = km + (size_t)s * DKV;
                    v = 0; for (int j = 0; j < HD; j++) v += qm[j] * ks[j];
                    v *= 1.0f / sqrtf((float)HD);
                }
                lgm[s] = v; if (v > mx) mx = v;
            }
            float sum = 0;
            for (int s = 0; s < seq; s++) { lgm[s] = expf(lgm[s] - mx); sum += lgm[s]; }
            float *attrow = attn + (size_t)m * D + (size_t)hh * HD;
            for (int j = 0; j < HD; j++) {
                float acc = 0;
                for (int s = 0; s < seq; s++) acc += lgm[s] / sum * vm[(size_t)s * DKV + j];
                attrow[j] = acc;
            }
        }
    }
}

/* ---------------- buffers ---------------- */
static float x[MS * D], h[MS * D], qbuf[MS * D], kbuf[MS * DKV], vbuf[MS * DKV], attn[MS * D];
static int8_t xi[MS * D], ai[MS * D];
static float scr[MS], sca[MS], sc2[MS];
static float upb[MS * F], gateb[MS * F], mid[MS * F];
static float oout[MS * D], sub[MS * D];
static float mch[MS * 1024]; static int8_t mch_i8[MS * 1024]; static float mch_sc[MS];
static float cosb[(MAX_SEQ + 8) * (HD / 2)], sinb[(MAX_SEQ + 8) * (HD / 2)];

/* ---------------- one prompt forward ---------------- */
static void run_prompt(CVI_RT_HANDLE rt, CVI_RT_MEM mem, uint64_t pa, uint8_t *va,
                       const int *toks, int seq, int pid,
                       const float *esc, const float *frms,
                       const float (*bias_all)[D + DKV + DKV],
                       PoolSet *ps, uint8_t *layer, size_t lsz,
                       int *bad1, int *bad2, int *rbad,
                       double *t_layers, double *t_head, int *next_token) {
    /* ---- x = embed[t]*esc[t] (streamed single rows) ---- */
    {
        FILE *ef = fopen(EMBED_PATH, "rb");
        if (!ef) { fprintf(stderr, "cannot open %s\n", EMBED_PATH); exit(2); }
        for (int i = 0; i < seq; i++) {
            int t = toks[i];
            int8_t er[D];
            if (fseek(ef, (long)t * D, SEEK_SET)) { fprintf(stderr, "seek embed\n"); exit(2); }
            if (fread(er, 1, D, ef) != D) { fprintf(stderr, "short embed read\n"); exit(2); }
            float es = esc[t];
            for (int k = 0; k < D; k++) x[(size_t)i * D + k] = (float)er[k] * es;
        }
        fclose(ef);
    }

    double t0 = now();
    for (int l = 0; l < L; l++) {
        char path[160];
        snprintf(path, sizeof path, "%s/layer%d_kal.bin", WDIR, l);
        FILE *f = fopen(path, "rb");
        if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
        if (fread(layer, 1, lsz, f) != lsz) { fprintf(stderr, "short layer read %s\n", path); exit(2); }
        fclose(f);

        LayerRef lr; parse_layer(layer, lsz, &lr);
        const float *bias = bias_all[l];
        const float *bq = bias, *bk = bias + D, *bv = bias + D + DKV;

        /* ---- QKV ---- */
        rms_norm(x, seq, lr.rms_attn, D, h);
        per_row_quant(h, seq, D, xi, scr);
        eng_matmul(rt, mem, pa, va, "q", xi, seq, lr.Wq_nib, lr.Wq_gsc, D, D, scr, qbuf, ps, bad1, bad2, rbad);
        eng_matmul(rt, mem, pa, va, "k", xi, seq, lr.Wk_nib, lr.Wk_gsc, D, DKV, scr, kbuf, ps, bad1, bad2, rbad);
        eng_matmul(rt, mem, pa, va, "v", xi, seq, lr.Wv_nib, lr.Wv_gsc, D, DKV, scr, vbuf, ps, bad1, bad2, rbad);
        for (int m = 0; m < seq; m++) {
            for (int j = 0; j < D; j++) qbuf[(size_t)m * D + j] += bq[j];
            for (int j = 0; j < DKV; j++) { kbuf[(size_t)m * DKV + j] += bk[j]; vbuf[(size_t)m * DKV + j] += bv[j]; }
        }
        for (int m = 0; m < seq; m++) {
            for (int hh = 0; hh < H; hh++) rope_inplace(qbuf + (size_t)m * D + (size_t)hh * HD, m, cosb, sinb);
            for (int hh = 0; hh < KVH; hh++) rope_inplace(kbuf + (size_t)m * DKV + (size_t)hh * HD, m, cosb, sinb);
        }
        attention(seq, qbuf, kbuf, vbuf, attn);

        /* ---- wo ---- */
        per_row_quant(attn, seq, D, ai, sca);
        eng_matmul(rt, mem, pa, va, "wo", ai, seq, lr.Wo_nib, lr.Wo_gsc, D, D, sca, oout, ps, bad1, bad2, rbad);
        for (int m = 0; m < seq; m++) for (int j = 0; j < D; j++) x[(size_t)m * D + j] += oout[(size_t)m * D + j];

        /* ---- ffn: up / gate ---- */
        rms_norm(x, seq, lr.rms_ffn, D, h);
        per_row_quant(h, seq, D, xi, scr);
        eng_matmul(rt, mem, pa, va, "up", xi, seq, lr.up_nib, lr.up_gsc, D, F, scr, upb, ps, bad1, bad2, rbad);
        eng_matmul(rt, mem, pa, va, "gate", xi, seq, lr.gate_nib, lr.gate_gsc, D, F, scr, gateb, ps, bad1, bad2, rbad);
        for (int m = 0; m < seq; m++) for (int j = 0; j < F; j++)
            mid[(size_t)m * F + j] = upb[(size_t)m * F + j] * silu(gateb[(size_t)m * F + j]);

        /* ---- ffn: down (K-chunk 1024, per-chunk per-row quant) ---- */
        memset(oout, 0, (size_t)seq * D * sizeof(float));
        for (int kc = 0; kc < F; kc += 1024) {
            int kcn = (F - kc < 1024) ? F - kc : 1024;
            for (int m = 0; m < seq; m++) memcpy(mch + (size_t)m * kcn, mid + (size_t)m * F + kc, (size_t)kcn * sizeof(float));
            per_row_quant(mch, seq, kcn, mch_i8, mch_sc);
            const uint8_t  *dnib = lr.down_nib + (size_t)(kc / G) * D * 16;
            const uint16_t *dgsc = lr.down_gsc + (size_t)(kc / G) * D;   /* halfword units */
            eng_matmul(rt, mem, pa, va, "down", mch_i8, seq, dnib, dgsc, kcn, D, mch_sc, sub, ps, bad1, bad2, rbad);
            for (int m = 0; m < seq; m++) for (int j = 0; j < D; j++) oout[(size_t)m * D + j] += sub[(size_t)m * D + j];
        }
        for (int m = 0; m < seq; m++) for (int j = 0; j < D; j++) x[(size_t)m * D + j] += oout[(size_t)m * D + j];
    }
    *t_layers = now() - t0;

    /* ---- LM head: final rms -> stream embed, top-5 ---- */
    double t1 = now();
    rms_norm(x + (size_t)(seq - 1) * D, 1, frms, D, h);
    int top[5]; double tv[5];
    for (int i = 0; i < 5; i++) { top[i] = -1; tv[i] = -1e300; }
    {
        int CHUNK_ROWS = 8192;                 /* 8192*896 = 7.34MB, fits layer buf */
        FILE *ef = fopen(EMBED_PATH, "rb");
        if (!ef) { fprintf(stderr, "cannot open %s\n", EMBED_PATH); exit(2); }
        int8_t *ebuf = layer;                  /* reuse layer buffer */
        for (long t0 = 0; t0 < V; t0 += CHUNK_ROWS) {
            int nrows = (V - t0 < CHUNK_ROWS) ? V - t0 : CHUNK_ROWS;
            if (fread(ebuf, 1, (size_t)nrows * D, ef) != (size_t)nrows * D) { fprintf(stderr, "short embed stream\n"); exit(2); }
            for (int r = 0; r < nrows; r++) {
                int t = (int)(t0 + r);
                const int8_t *er = ebuf + (size_t)r * D;
                double s = 0;
                for (int j = 0; j < D; j++) s += (double)h[j] * (double)er[j] * (double)esc[t];
                for (int i = 0; i < 5; i++) if (s > tv[i]) {
                    for (int jj = 4; jj > i; jj--) { top[jj] = top[jj - 1]; tv[jj] = tv[jj - 1]; }
                    top[i] = t; tv[i] = s; break;
                }
            }
        }
        fclose(ef);
    }
    *t_head = now() - t1;
    *next_token = top[0];

    printf("PROMPT %d (seq=%d, toks=[%d %d %d %d %d %d %d])\n",
           pid + 1, seq, toks[0], toks[1], toks[2], toks[3], toks[4], toks[5], toks[6]);
    printf("NEXT_TOKEN: %d\n", top[0]);
    printf("TOP5: ");
    for (int i = 0; i < 5; i++) printf("%d ", top[i]);
    printf("\n");
    printf("GAP: %.4f\n", (double)(tv[0] - tv[1]));
    for (int i = 0; i < 5; i++) printf("TOPVAL[%d]=%.4f\n", top[i], tv[i]);
    printf("t_layers=%.2fs t_lmhead=%.2fs t_total=%.2fs per_token=%.2fs\n",
           *t_layers, *t_head, *t_layers + *t_head, (*t_layers + *t_head) / seq);
}

/* ---------------- prompts (P0) ---------------- */
static const int P1_TOKS[MS] = {105538, 59975, 100132, 0, 0, 0, 0};
static const int P2_TOKS[MS] = {785, 6722, 315, 9625, 374, 0, 0};
static const int P3_TOKS[MS] = {100644, 104307, 101243, 3837, 97639, 85336, 102077};
static const int PSEQ[3] = {3, 5, 7};
static const int EXPECTED_NEXT[3] = {2130, 12095, 99366};

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);
    printf("===== M2 24-layer prefill + LM head (TIU Path A', 3-prompt regression) =====\n");

    /* ---- small weights: embed_scales, final_rms, all biases ---- */
    float *esc = malloc(V * sizeof(float));
    {
        FILE *f = fopen(ESC_PATH, "rb"); if (!f) { fprintf(stderr, "cannot open %s\n", ESC_PATH); return 2; }
        if (fread(esc, 4, V, f) != V) { fprintf(stderr, "short esc\n"); return 2; }
        fclose(f);
    }
    static float frms[D];
    {
        FILE *f = fopen(FRMS_PATH, "rb"); if (!f) { fprintf(stderr, "cannot open %s\n", FRMS_PATH); return 2; }
        if (fread(frms, 4, D, f) != D) { fprintf(stderr, "short frms\n"); return 2; }
        fclose(f);
    }
    static float bias_all[L][D + DKV + DKV];
    for (int l = 0; l < L; l++) {
        char path[160]; snprintf(path, sizeof path, "%s/layer%d_bias.f32", WDIR, l);
        FILE *f = fopen(path, "rb"); if (!f) { fprintf(stderr, "cannot open %s\n", path); return 2; }
        if (fread(bias_all[l], 4, D + DKV + DKV, f) != D + DKV + DKV) { fprintf(stderr, "short bias %s\n", path); return 2; }
        fclose(f);
    }
    printf("loaded esc/frms/biases (%zu B)\n", V * 4 + D * 4 + (size_t)L * (D + DKV + DKV) * 4);

    /* ---- rope tables ---- */
    for (int pos = 0; pos < MAX_SEQ + 8; pos++)
        for (int j = 0; j < HD / 2; j++) {
            float freq = powf(ROPE_THETA, -(float)j / (HD / 2));
            cosb[(size_t)pos * (HD / 2) + j] = cosf(pos * freq);
            sinb[(size_t)pos * (HD / 2) + j] = sinf(pos * freq);
        }

    /* ---- ION + prebuilt pools (per M, exact tile widths) ---- */
    CVI_RT_HANDLE rt; CVI_RT_Init(&rt);
    CVI_RT_MEM mem = CVI_RT_MemAlloc(rt, NEURON_SZ);
    uint64_t pa = CVI_RT_MemGetPAddr(mem);
    uint8_t *va = CVI_RT_MemGetVAddr(mem);
    CVI_RT_SetBaseReg(rt, 0, pa);
    memset(va, 0, NEURON_SZ);

    PoolSet ps3, ps5, ps7; memset(&ps3, 0, sizeof ps3); memset(&ps5, 0, sizeof ps5); memset(&ps7, 0, sizeof ps7);
    double t0 = now();
    pool_build(rt, pa, 3, 128, &ps3.p128);
    pool_build(rt, pa, 3, 384, &ps3.p384);
    pool_build(rt, pa, 3, 896, &ps3.p896);
    pool_build(rt, pa, 5, 128, &ps5.p128);
    pool_build(rt, pa, 5, 256, &ps5.p256);
    pool_build(rt, pa, 5, 768, &ps5.p768);
    pool_build(rt, pa, 7, 128, &ps7.p128);
    pool_build(rt, pa, 7, 256, &ps7.p256);
    pool_build(rt, pa, 7, 768, &ps7.p768);
    printf("pools built (M=3:{128,384,896} M=5/7:{128,256,768}) in %.3fs\n", now() - t0);

    /* ---- layer buffer (size from layer0 file) ---- */
    struct stat st;
    if (stat(WDIR "/layer0_kal.bin", &st)) { fprintf(stderr, "stat layer0\n"); return 2; }
    size_t lsz = (size_t)st.st_size;
    uint8_t *layer = malloc(lsz);
    printf("layer file size = %zu B\n", lsz);

    int bad1 = 0, bad2 = 0, rbad = 0;
    const int *PT[3] = {P1_TOKS, P2_TOKS, P3_TOKS};
    int ok_all = 1;
    double tot_all = now();
    for (int p = 0; p < 3; p++) {
        double tl, th; int nxt;
        PoolSet *ps = (PSEQ[p] <= 3) ? &ps3 : (PSEQ[p] <= 5) ? &ps5 : &ps7;
        run_prompt(rt, mem, pa, va, PT[p], PSEQ[p], p, esc, frms, bias_all,
                   ps, layer, lsz, &bad1, &bad2, &rbad, &tl, &th, &nxt);
        int ok = (nxt == EXPECTED_NEXT[p]);
        ok_all &= ok;
        printf("  expected_next=%d  %s\n", EXPECTED_NEXT[p], ok ? "OK" : "MISMATCH");
    }
    printf("total wall = %.2fs (all 3 prompts)\n", now() - tot_all);
    printf("==== P1/P2 bit-exact: bad1=%d bad2=%d  r_opt mismatches=%d ====\n", bad1, bad2, rbad);
    printf("==== TIU runs: pass1=%ld pass2=%ld total=%ld ====\n", g_runs_pass1, g_runs_pass2, g_runs_pass1 + g_runs_pass2);
    printf("==== 24L regression: expected_next 3/3 %s ====\n", ok_all ? "OK" : "FAIL");
    printf("==== 24L regression: TIU internal %s ====\n", (bad1 + bad2 + rbad == 0) ? "BIT-EXACT" : "HAS MISMATCHES");

    free(layer); free(esc);
    CVI_RT_MemFree(rt, mem); CVI_RT_DeInit(rt);
    return 0;
}
