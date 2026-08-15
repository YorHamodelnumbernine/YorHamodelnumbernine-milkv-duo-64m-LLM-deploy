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
 *   final rms_norm(last row) -> LM head (two-stage MDP): Stage1 h·centroid (CPU)
 *     -> top-Kc clusters; Stage2 read cluster-major embed spans -> exact logits.
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
 *       embed_scales.f32 + final_rms.f32 + embed_i8_cl.bin + embed_scales_cl.f32 +
 *       row_to_tok_cl.bin + centroid_f16.bin + clust_idx.bin (all deployed).
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/resource.h>
#include <signal.h>
#include <setjmp.h>
#include "cviruntime_context.h"
#include "bmkernel/bm1822/bmkernel_1822.h"
#include "dequant_kal.c"              /* scalar + RVV K-aligned dequant */

#define NEURON_SZ 262144
#define ACTQ_OFF  0
#define DQ_OFF    4096
#define P1_OFF    163840   /* Phase 7d: 后移给 DQ 腾出 [32,F]=155648B 全块区 (merged up/gate) */
#define P2_OFF    172032

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
#define KV_CAP 48       /* prompt + decode 追加 token 数; 2*DKV*L*4B=24KB/token, 20->48 仅 +0.67MB DDR anon (非 ION) */
                        /* 依据 PLAN_B_NIB_MEMBUDGET_V2: KV cache 是 malloc(DDR 匿名) 不碰 ION carveout;
                           DDR 运行期余量 ~5-6MiB 可容 +0.67MiB. MAX_SEQ=64 >= KV_CAP=48 (rope/softmax 上限). */
#define TILEW 512       /* max N-tile width (M>=2 pools: 128/256/384/512) */
#define MAXT 10         /* ceil(F/TILEW) = 10 */
#define MTILEW 608      /* Phase 7d: merged up/gate N-tile width (F=4864=8*608) */
#define MNT (F / MTILEW)  /* merged tile count = 8 (<= SDK 16-op cmdbuf cap) */

/* ---- Phase 7e: ION gsc cache — per-layer gsc region byte offsets/sizes in the
 * layer file (mirrors parse_layer offset sequence exactly).  All gsc (7 mats)
 * total 931840 B/layer; 24 layers = 22364160 B = 21.3 MiB, CPU-cached in ION
 * (~2200MB/s) => eliminates the per-step cold-mmap gsc reads (~8MB/s on SD). */
#define GSC_WQ_OFF ((size_t)(D * 4 + (D / G) * D * 16))
#define GSC_WQ_SZ  ((size_t)((D / G) * D * 2))
#define GSC_WK_OFF (GSC_WQ_OFF + GSC_WQ_SZ + (size_t)((D / G) * DKV * 16))
#define GSC_WK_SZ  ((size_t)((D / G) * DKV * 2))
#define GSC_WV_OFF (GSC_WK_OFF + GSC_WK_SZ + (size_t)((D / G) * DKV * 16))
#define GSC_WV_SZ  ((size_t)((D / G) * DKV * 2))
#define GSC_WO_OFF (GSC_WV_OFF + GSC_WV_SZ + (size_t)((D / G) * D * 16))
#define GSC_WO_SZ  ((size_t)((D / G) * D * 2))
#define GSC_UP_OFF (GSC_WO_OFF + GSC_WO_SZ + (size_t)((D / G) * F * 16))
#define GSC_UP_SZ  ((size_t)((D / G) * F * 2))
#define GSC_GA_OFF (GSC_UP_OFF + GSC_UP_SZ + (size_t)((D / G) * F * 16))
#define GSC_GA_SZ  ((size_t)((D / G) * F * 2))
#define GSC_DN_OFF (GSC_GA_OFF + GSC_GA_SZ + (size_t)((F / G) * D * 16))
#define GSC_DN_SZ  ((size_t)((F / G) * D * 2))
#define GSC_LAYER_BYTES (GSC_WQ_SZ + GSC_WK_SZ + GSC_WV_SZ + GSC_WO_SZ + GSC_UP_SZ + GSC_GA_SZ + GSC_DN_SZ)
#define GSC_TOTAL_BYTES (GSC_LAYER_BYTES * L)
/* carveout 预检 margin: pools 实测 ~1.7MB + merged ~0.26MB (各含安全余量).
 * 用于 gsc/pools 分配前判断 "整轮 ION 足迹" 是否放得下, 避免 gsc 先占满导致
 * 后续 pools 分配在 SDK 重试路径 SIGABRT. */
#define ION_POOLS_EST (2u << 20)
#define ION_MP_EST    (1u << 20)
#define ROPE_THETA 1000000.0
#define EPS 1e-6f

#define WDIR "/data/qwen"

#define EMBED_PATH WDIR "/embed_i8.bin"
#define ESC_PATH   WDIR "/embed_scales.f32"
#define FRMS_PATH  WDIR "/final_rms.f32"

/* ---- Phase 6 two-stage LM head: cluster-major files (offline-built) ---- */
#define EMBED_CL_PATH WDIR "/embed_i8_cl.bin"
#define ESC_CL_PATH   WDIR "/embed_scales_cl.f32"
#define TOK_CL_PATH   WDIR "/row_to_tok_cl.bin"
#define CENT_PATH     WDIR "/centroid_f16.bin"
#define CLIDX_PATH    WDIR "/clust_idx.bin"
#define LMHEAD_C      1024     /* MDP cluster count */
#define LMHEAD_KC     128      /* top-Kc clusters per token (runtime knob) */

/* Host int32 reference cross-check of TIU P1/P2 (bad1/bad2/rbad).  Proven
 * bit-exact for the full run; runtime-toggleable: VERIFY=0 disables the host
 * reference for the latency-only path (bit-exactness established separately). */
static int g_verify = 1;
static int g_profile = 0;
static int g_use_merged = 1;   /* Phase 7d: MERGE=0 关闭 merged up/gate (A/B 对照) */

/* ---- PROFILE instrumentation (eng_matmul 分段计时; PROFILE=1 末尾打印) ----
 * 分解 decode compute: TIU matmul 实际提交(runcmd, 含 cmdbuf g2l+TIU+l2g) /
 * CPU dequant-RVV(dequant) / g2l-DMA+缓存维护(flush+invld) /
 * host int32 参考(verify) / 其余(other = accd 累加 + block_max 收集 +
 * 页表-缺页等待 + 循环开销). 矩阵名级聚合 (q/k/v/wo/up/gate/down). */
static double g_t_dequant = 0, g_t_copyact = 0, g_t_flush = 0, g_t_runcmd = 0;
static double g_t_invld = 0, g_t_verify = 0, g_t_matmul = 0, g_t_other = 0;
static double g_t_bm = 0, g_t_acc = 0, g_t_fin = 0;   /* Phase 7e: "other" 子项拆分 */
static long g_n_runcmd = 0;
static const char *g_mm_names[7] = {"q", "k", "v", "wo", "up", "gate", "down"};
static double g_t_mm[7]; static long g_n_mm[7];

/* ---- Phase 7c: 离线 rsafe 预标定表 (查表替代 wmax 预扫) ----
 * 转换器对每 (layer, matrix[, down-K-chunk]) 静态算好 rsafe 写入 rsafe.bin:
 * 每层 11 字节 = q,k,v,wo,up,gate 各 1 (idx 0..5) + down 5 个 K-chunk (idx 6..10).
 * 实测对称 INT4 (SYM_QMAX=7, G=32) 下所有 24*11 项 wmax 均 =7 -> rsafe 均 =5
 * (恒定). RSH=1 时 VERIFY=0 直接查表跳过扫描 (省 wmax 预扫全层读); VERIFY=1
 * 仍做一次扫描与表比对 (g_rsh_bad), 证明与运行时逐 tile 位精确一致. */
#define RSH_DCHUNKS 5
static uint8_t g_rsh[L][6 + RSH_DCHUNKS];
static int g_rsh_loaded = 0;   /* rsafe.bin 载入成功 */
static int g_rsh_skip = 0;     /* RSH=1: 查表跳过 wmax 预扫 */
static long g_rsh_bad = 0;     /* 表 vs 运行时扫描 不一致计数 */
static int g_cur_layer = 0;    /* 当前层 (layer loop 设置) */
static int g_cur_dchunk = 0;   /* down K-chunk 索引 (非 down = 0) */
static long g_prev_majflt = 0; /* decode 步内 major fault 计数 (诊断) */
static int rsh_lookup(int l, int mi, int dc) {
    if (l < 0 || l >= L || mi < 0 || mi >= 7) return 5;
    if (mi < 6) return g_rsh[l][mi];
    int idx = 6 + dc;
    if (idx < 6 || idx >= 6 + RSH_DCHUNKS) return 5;
    return g_rsh[l][idx];
}

static inline double now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

/* =====================================================================
 * ION OOM 防御 (libcviruntime CVI_RT_MemAlloc 失败时不返回 NULL: 反汇编
 * mem_alloc_raw 证实其重试 3 次 "reopen ion dev" 后 __assert_fail -> SIGABRT,
 * 直接杀进程, 调用方无法走清理/回退路径).
 *  1) rt_alloc_safe(): sigsetjmp/siglongjmp 捕获 SIGABRT -> 返回 NULL,
 *     调用方走既有 NULL 检查 (回退 GSC_ION=0 冷 mmap / 清理退出), 不 SIGABRT.
 *     assert 前 SDK 已 close 旧 ion fd 且已解锁互斥 (反汇编证实) -> 捕获后
 *     继续用 SDK 安全; 已分配的其他 buffer (neuron 等) 是独立 dmabuf fd, 不受影响.
 *  2) watchdog: 主循环 run_prompt/run_decode_step 每层 wd_kick() 打心跳;
 *     超 WD_TIMEOUT_SEC 无心跳即判定挂死, 强制 _exit(1) 让内核关 fd 释放 ION
 *     (实测 SIGABRT 本身不泄漏, 24.3MB 泄漏源 = 残留存活/孤儿进程).
 * ===================================================================== */
static sigjmp_buf g_ion_abort_jmp;
static volatile sig_atomic_t g_ion_alloc_armed = 0;

static void ion_abort_sighandler(int sig) {
    if (g_ion_alloc_armed) siglongjmp(g_ion_abort_jmp, 1);
    /* 非分配期 SIGABRT = 真实 bug: 恢复默认动作重新 raise, 保持标准 RC=134/core 语义. */
    signal(sig, SIG_DFL);
    raise(sig);
}

static CVI_RT_MEM rt_alloc_safe(CVI_RT_HANDLE rt, size_t sz, const char *what) {
    g_ion_alloc_armed = 1;
    CVI_RT_MEM m = NULL;
    if (sigsetjmp(g_ion_abort_jmp, 1) == 0) {
        m = CVI_RT_MemAlloc(rt, sz);
    } else {
        fprintf(stderr, "[ION-OOM] SIGABRT 在 %s 分配期间被捕获 (sz=%zu B); "
                        "不退出, 走回退/清理路径. 若为残留占用, 请 run_clean.sh --clean 后重试.\n",
                what, sz);
        m = NULL;
    }
    g_ion_alloc_armed = 0;
    return m;
}

/* ---- watchdog (防孤儿进程 ION 泄漏) ---- */
#define WD_TIMEOUT_NS (90LL * 1000000000LL)
static volatile long long g_wd_heartbeat_ns = 0;
static volatile sig_atomic_t g_wd_enabled = 0;
static volatile sig_atomic_t g_wd_die = 0;
static pthread_t g_wd_thread;

static long long wd_now_ns(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}
static void wd_kick(void) {
    if (g_wd_enabled) __atomic_store_n(&g_wd_heartbeat_ns, wd_now_ns(), __ATOMIC_RELAXED);
}
static void *wd_thread_main(void *arg) {
    (void)arg;
    while (!g_wd_die) {
        usleep(500000);   /* 0.5s 检查粒度 */
        long long hb = __atomic_load_n(&g_wd_heartbeat_ns, __ATOMIC_RELAXED);
        if (g_wd_enabled && (wd_now_ns() - hb) > WD_TIMEOUT_NS) {
            fprintf(stderr, "[WATCHDOG] %.0fs 无心跳 (>%.0fs), 判定挂死; 强制 _exit(1) "
                            "让内核释放 ION (防孤儿进程 24MB 泄漏).\n",
                    (double)(wd_now_ns() - hb) / 1e9, WD_TIMEOUT_NS / 1e9);
            _exit(1);
        }
    }
    return NULL;
}
static void wd_start(void) {
    __atomic_store_n(&g_wd_heartbeat_ns, wd_now_ns(), __ATOMIC_RELAXED);
    g_wd_enabled = 1;
    if (pthread_create(&g_wd_thread, NULL, wd_thread_main, NULL) != 0) {
        fprintf(stderr, "[WATCHDOG] 线程创建失败, 继续无看门狗运行\n");
        g_wd_enabled = 0;
    } else {
        pthread_detach(g_wd_thread);
    }
}

/* 安装 SIGABRT 捕获 (rt_alloc_safe 使用); 必须在任何 CVI_RT_MemAlloc 之前调用. */
static void ion_abort_install(void) {
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = ion_abort_sighandler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGABRT, &sa, NULL);
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

static void pool_free(CVI_RT_HANDLE rt, Pool *p);   /* 前置声明 (定义在 pool_build 之后) */

/* 返回 0 OK / -1 失败 (ION 分配失败: 已打印错误, 部分已分配已释放, 不泄漏). */
static int pool_build(CVI_RT_HANDLE rt, uint64_t pa, int M, int nshape, Pool *p) {
    p->nshape = nshape;
    memset(p->ld, 0, sizeof p->ld);
    memset(p->src, 0, sizeof p->src);
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
            p->src[r][dest] = rt_alloc_safe(rt, psize, "pool");
            if (!p->src[r][dest]) {
                fprintf(stderr, "pool ION alloc %u B FAILED (M=%d n=%d r=%d d=%d); carveout 余量不足 — "
                                "run 'run_clean.sh --clean' 清理遗留后重试\n",
                        psize, M, nshape, r, dest);
                pool_free(rt, p);
                return -1;
            }
            uint8_t *db = CVI_RT_MemGetVAddr(p->src[r][dest]);
            bmk1822_dmabuf_convert(cmd, cmd_sz, db);
            bmk1822_arraybase_set(db, pa, 0, 0, 0);
            CVI_RT_MemFlush(rt, p->src[r][dest]);
            CVI_RT_LoadDmabuf(rt, p->src[r][dest], psize, pa, 0, false, &p->ld[r][dest]);
            bmk1822_cleanup(bmk);
        }
    }
    return 0;
}

/* ---- Phase 7d: merged N-tile cmdbuf (up/gate, M=1) ----
 * ONE cmdbuf = g2l(L) + nt x [g2l(R_t from DQ_OFF+t*32*MTILEW) + TIU(r)
 *               + l2g(O_t to ooff+t*MTILEW)] with R/O lmem REUSED across
 * tiles (explicit WAR deps).  Replaces 6 separate per-tile submissions per
 * (K-block, pass).  nt=8 <= SDK 16-bd-op cmdbuf cap (probe-validated).
 * DQ must hold the full [32,F] K-block (P1/P2 shifted to make room). */
typedef struct {
    int nt, tilew;
    CVI_RT_MEM ld[16][2], src[16][2];   /* [rshift][dest 0=P1 1=P2] */
} MergedPool;

/* 返回 0 OK / -1 失败 (ION 分配失败: 已打印错误, 部分已分配已释放, 不泄漏). */
static int mpool_build(CVI_RT_HANDLE rt, uint64_t pa, int M, int tilew, int nt, MergedPool *mp) {
    mp->nt = nt; mp->tilew = tilew;
    memset(mp->ld, 0, sizeof mp->ld);
    memset(mp->src, 0, sizeof mp->src);
    for (int r = 0; r < 16; r++) {
        for (int dest = 0; dest < 2; dest++) {
            uint32_t ooff = dest ? P2_OFF : P1_OFF;
            uint8_t cmdbuf[131072] __attribute__((aligned(16)));
            bmk_info_t info = { .chip_version = 1822, .cmdbuf_size = sizeof(cmdbuf), .cmdbuf = cmdbuf };
            bmk1822_context_t *bmk = bmk1822_register(&info);
            bmk1822_matrix_lmem_shape_t SL = {.n = (uint32_t)M, .c = 1, .w = 32, .col = 32};
            bmk1822_matrix_lmem_shape_t SR = {.n = 32, .c = 1, .w = (uint32_t)tilew, .col = (uint32_t)tilew};
            bmk1822_matrix_lmem_shape_t SO = {.n = (uint32_t)M, .c = 1, .w = (uint32_t)tilew, .col = (uint32_t)tilew};
            bmk1822_matrix_lmem_t *ml_l = bmk1822_lmem_alloc_matrix(bmk, SL, FMT_I8, 1);
            bmk1822_matrix_lmem_t *ml_r = bmk1822_lmem_alloc_matrix(bmk, SR, FMT_I8, 1);
            bmk1822_matrix_lmem_t *ml_o = bmk1822_lmem_alloc_matrix(bmk, SO, FMT_I8, 1);
            bmk1822_matrix_tgmem_t mg_l = {0, ACTQ_OFF, FMT_I8, {M, 32}, {32}};
            bmk1822_op_t *op_g2lL = bmk1822_tdma_g2l_matrix_copy(
                bmk, &(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_l, ml_l});
            bmk1822_op_t *prev_mm = NULL, *prev_l2g = NULL;
            for (int t = 0; t < nt; t++) {
                uint32_t dqoff = DQ_OFF + (size_t)t * 32 * tilew;
                uint32_t ooff_t = ooff + (size_t)t * tilew;
                bmk1822_matrix_tgmem_t mg_r = {0, dqoff, FMT_I8, {32, tilew}, {tilew}};
                bmk1822_matrix_tgmem_t mg_o = {0, ooff_t, FMT_I8, {M, tilew}, {tilew}};
                bmk1822_op_t *op_g2lR = bmk1822_tdma_g2l_matrix_copy(
                    bmk, &(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_r, ml_r});
                bmk1822_tiu_matrix_multiplication_param_t mm = {
                    .res = ml_o, .left = ml_l, .right = ml_r, .bias = NULL,
                    .lshift_bits = 0, .rshift_bits = (uint8_t)r, .res_is_int8 = 1,
                    .relu_enable = 0, .add_result = 0, .ps32_mode = 0, .layer_id = 1 };
                bmk1822_op_t *op_mm = bmk1822_tiu_matrix_multiplication(bmk, &mm);
                bmk1822_op_t *op_l2g = bmk1822_tdma_l2g_matrix_copy(
                    bmk, &(bmk1822_tdma_l2tg_matrix_copy_param_t){ml_o, &mg_o});
                bmk1822_add_dependency(bmk, op_g2lR, op_mm);      /* RAW R */
                bmk1822_add_dependency(bmk, op_mm, op_l2g);       /* RAW O */
                bmk1822_add_dependency(bmk, op_g2lL, op_mm);      /* RAW L */
                if (prev_mm)  bmk1822_add_dependency(bmk, prev_mm, op_g2lR);   /* WAR R */
                if (prev_l2g) bmk1822_add_dependency(bmk, prev_l2g, op_mm);    /* WAR O */
                prev_mm = op_mm; prev_l2g = op_l2g;
            }
            uint32_t cmd_sz; uint8_t *cmd = bmk1822_acquire_cmdbuf(bmk, &cmd_sz);
            uint32_t psize, pmu; bmk1822_dmabuf_size(cmd, cmd_sz, &psize, &pmu);
            mp->src[r][dest] = rt_alloc_safe(rt, psize, "merged-pool");
            if (!mp->src[r][dest]) {
                fprintf(stderr, "merged pool ION alloc %u B FAILED (r=%d d=%d); carveout 余量不足 — "
                                "run 'run_clean.sh --clean' 清理遗留后重试, 或 MERGE=0 关闭 merged pool\n",
                        psize, r, dest);
                for (int rr = 0; rr < 16; rr++)
                    for (int dd = 0; dd < 2; dd++)
                        if (mp->src[rr][dd]) { CVI_RT_MemFree(rt, mp->src[rr][dd]); mp->src[rr][dd] = NULL; }
                return -1;
            }
            uint8_t *db = CVI_RT_MemGetVAddr(mp->src[r][dest]);
            bmk1822_dmabuf_convert(cmd, cmd_sz, db);
            bmk1822_arraybase_set(db, pa, 0, 0, 0);
            CVI_RT_MemFlush(rt, mp->src[r][dest]);
            CVI_RT_LoadDmabuf(rt, mp->src[r][dest], psize, pa, 0, false, &mp->ld[r][dest]);
            bmk1822_cleanup(bmk);
        }
    }
    return 0;
}

/* Per-M pool set.  Only the widths actually used per M are built:
 *   M=3 : 128, 384, 896   (q/wo/down N=896 -> 1 tile; up/gate 5x896+384)
 *   M=5/7: 128, 256, 768  (N=896 -> 768+128; up/gate 6x768+256)
 */
typedef struct { Pool p128, p256, p384, p768, p896; } PoolSet;

/* 释放 pool cmdbuf ION (B-2: decode 前回收 M=5/7 prefill pools ~0.8MB).
 * ld 句柄无独立 Unload API, 释放 src 即释放底层 ION 分配. */
static void pool_free(CVI_RT_HANDLE rt, Pool *p) {
    for (int r = 0; r < 16; r++)
        for (int d = 0; d < 2; d++)
            if (p->src[r][d]) { CVI_RT_MemFree(rt, p->src[r][d]); p->src[r][d] = NULL; }
}
static void pool_set_free(CVI_RT_HANDLE rt, PoolSet *ps) {
    pool_free(rt, &ps->p128); pool_free(rt, &ps->p256);
    pool_free(rt, &ps->p384); pool_free(rt, &ps->p768); pool_free(rt, &ps->p896);
}

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
/* Phase 7e: ION gsc cache — all 24 layers' gsc pread into one 21.3MiB ION buffer
 * at startup (CPU-cached ~2200MB/s), then per-step LayerRef gsc pointers point
 * into it instead of the cold mmap layer pages (~8MB/s on SD).  NULL = disabled
 * (fall back to direct cold-mmap gsc reads, bit-exact either way). */
static CVI_RT_MEM g_gsc_ion_mem = NULL;
static uint16_t *g_gsc_ion = NULL;
/* B-2 gsc 拆分缓存 (CEO 授权形态): 22 层 ION + 2 层 DDR (l >= GSC_ION_LAYERS).
 * DDR 层 = mmap 层文件 (page cache, 可回收 — 避免匿名 malloc 换页顶爆 28MB RAM);
 * 2 层 DDR 在 init 预读入 page cache (CEO: "init 预读入 DDR").
 * ION 预算 (carveout 28,102,656 B): 22*931840 + 2*2179072 + 262144 + pools
 *   prefill 态 1,703,936 = 26,824,704 ✓ (margin 1.28MiB); decode 释放 ps5/7 后
 *   pools ~865,076 -> 25,985,844 (margin ~2.0MiB). DDR gsc = 2 层 (1.86MB page cache).
 * SD 槽: 2 个大槽 (2.18MB) 双缓冲, per-matrix 交替 (槽 = midx&1), 全 ION. */
#define GSC_ION_LAYERS 22
#define GSC_DDR_LAYERS (L - GSC_ION_LAYERS)
#define GSC_ION_CACHE_BYTES (GSC_LAYER_BYTES * GSC_ION_LAYERS)
static uint8_t *g_gsc_ddr_map[GSC_DDR_LAYERS] = {0}; /* 每 DDR 层 layer 文件 mmap 基址 */
static int       g_gsc_ion_n = 0;    /* ION 中 gsc 层数 (GSC_ION=1: L=24; B-2: 20) */
static float    *g_rms_all = NULL;   /* rms 全层缓存 (DDR, L*2*D floats = 172032 B) */
static long g_runs_pass1 = 0, g_runs_pass2 = 0;

static void eng_matmul(CVI_RT_HANDLE rt, CVI_RT_MEM mem, uint64_t pa, uint8_t *va,
                       const char *name, const int8_t *x_i8, int M,
                       const uint8_t *nib, const uint16_t *gscf,
                       int K, int N, const float *sc_row, float *out,
                       PoolSet *ps, int *bad1, int *bad2, int *rbad) {
    double ta, t0 = now();
    int KG = K / G;
    int mi = 0; for (int i = 0; i < 7; i++) if (name && !strcmp(name, g_mm_names[i])) { mi = i; break; }

    /* ---- rsafe: 查表 (Phase 7c, RSH=1) 或运行时 wmax 预扫 ----
     * 表路径跳过 wmax 预扫 (省 201MB/token 全层预读); VERIFY=1 保留扫描做位精确比对. */
    int rsafe;
    if (g_rsh_skip) {
        rsafe = rsh_lookup(g_cur_layer, mi, g_cur_dchunk);
        if (g_verify) {
            int wmax = 0;
            ta = now();
            for (size_t i = 0; i < (size_t)KG * N * 16; i++) {
                uint8_t b = nib[i];
                int lo = b & 0xF, hi = b >> 4; if (lo > 7) lo -= 16; if (hi > 7) hi -= 16;
                int a = lo < 0 ? -lo : lo, bb = hi < 0 ? -hi : hi;
                if (a > wmax) wmax = a; if (bb > wmax) wmax = bb;
            }
            g_t_other += now() - ta;
            int sr = matmul_rshift_w(G, wmax) - 3; if (sr < 4) sr = 4;
            if (sr != rsafe) g_rsh_bad++;
        }
    } else {
        int wmax = 0;
        ta = now();
        for (size_t i = 0; i < (size_t)KG * N * 16; i++) {
            uint8_t b = nib[i];
            int lo = b & 0xF, hi = b >> 4; if (lo > 7) lo -= 16; if (hi > 7) hi -= 16;
            int a = lo < 0 ? -lo : lo, bb = hi < 0 ? -hi : hi;
            if (a > wmax) wmax = a; if (bb > wmax) wmax = bb;
        }
        g_t_other += now() - ta;
        rsafe = matmul_rshift_w(G, wmax) - 3; if (rsafe < 4) rsafe = 4;
    }

    int tilew = max_tile_for_m(M);
    int ntiles = (N + tilew - 1) / tilew;
    int toff[MAXT], tn[MAXT];
    for (int t = 0; t < ntiles; t++) { toff[t] = t * tilew; tn[t] = (t == ntiles - 1) ? N - toff[t] : tilew; }

    memset(accd, 0, sizeof(double) * (size_t)M * N);

    for (int g = 0; g < KG; g++) {
        int block_max = 0;
        int gold_max = 0;

        /* ---- pass1 phase: all tiles, rshift=rsafe ---- */
        for (int t = 0; t < ntiles; t++) {
            const uint8_t *nibt = nib + (size_t)g * N * 16 + (size_t)toff[t] * 16;
            ta = now(); dequant_kal_rvv(nibt, tn[t], (int8_t *)(va + DQ_OFF)); g_t_dequant += now() - ta;
            ta = now();
            for (int m = 0; m < M; m++)
                memcpy(va + ACTQ_OFF + (size_t)m * 32, x_i8 + (size_t)m * K + (size_t)g * 32, 32);
            g_t_copyact += now() - ta;
            if (g_verify) {
                ta = now();
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
                g_t_verify += now() - ta;
            }
            ta = now(); CVI_RT_MemFlushEx(rt, mem, DQ_OFF + (size_t)32 * tn[t]); g_t_flush += now() - ta;
            Pool *pl = pick_pool(ps, tn[t]);
            ta = now(); CVI_RT_RunCmdbufEx(rt, pl->ld[rsafe][0], &(CVI_RT_ARRAYBASE){.gaddr_base0 = pa});
            g_t_runcmd += now() - ta; g_n_runcmd++;
            g_runs_pass1++;
            ta = now(); CVI_RT_MemInvldEx(rt, mem, P2_OFF + (size_t)M * tn[t]); g_t_invld += now() - ta;
            int8_t *p1 = (int8_t *)(va + P1_OFF);
            ta = now();
            for (int m = 0; m < M; m++) {
                for (int n = 0; n < tn[t]; n++) {
                    int av = p1[(size_t)m * tn[t] + n]; if (av < 0) av = -av; if (av > block_max) block_max = av;
                    if (g_verify) {
                        int gv = int8_round_div(hacc[(size_t)m * tn[t] + n], rsafe);
                        if (p1[(size_t)m * tn[t] + n] != gv) (*bad1)++;
                        if (gv < 0) gv = -gv; if (gv > gold_max) gold_max = gv;
                    }
                }
            }
            g_t_bm += now() - ta;
        }
        long long est = (long long)block_max << rsafe;
        int r_opt = 0; while (est > (127LL << r_opt)) r_opt++;
        if (g_verify) {
            long long gest = (long long)gold_max << rsafe;
            int r_ref = 0; while (gest > (127LL << r_ref)) r_ref++;
            if (r_opt != r_ref) (*rbad)++;
        }

        /* ---- pass2 phase: all tiles, rshift=r_opt ---- */
        for (int t = 0; t < ntiles; t++) {
            const uint8_t *nibt = nib + (size_t)g * N * 16 + (size_t)toff[t] * 16;
            ta = now(); dequant_kal_rvv(nibt, tn[t], (int8_t *)(va + DQ_OFF)); g_t_dequant += now() - ta;
            ta = now();
            for (int m = 0; m < M; m++)
                memcpy(va + ACTQ_OFF + (size_t)m * 32, x_i8 + (size_t)m * K + (size_t)g * 32, 32);
            g_t_copyact += now() - ta;
            if (g_verify) {
                ta = now();
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
                g_t_verify += now() - ta;
            }
            ta = now(); CVI_RT_MemFlushEx(rt, mem, DQ_OFF + (size_t)32 * tn[t]); g_t_flush += now() - ta;
            Pool *pl = pick_pool(ps, tn[t]);
            ta = now(); CVI_RT_RunCmdbufEx(rt, pl->ld[r_opt][1], &(CVI_RT_ARRAYBASE){.gaddr_base0 = pa});
            g_t_runcmd += now() - ta; g_n_runcmd++;
            g_runs_pass2++;
            ta = now(); CVI_RT_MemInvldEx(rt, mem, P2_OFF + (size_t)M * tn[t]); g_t_invld += now() - ta;
            int8_t *p2 = (int8_t *)(va + P2_OFF);
            double f2 = (double)(1 << r_opt);
            ta = now();
            for (int m = 0; m < M; m++) {
                for (int n = 0; n < tn[t]; n++) {
                    if (g_verify) {
                        if (p2[(size_t)m * tn[t] + n] != int8_round_div(hacc[(size_t)m * tn[t] + n], r_opt)) (*bad2)++;
                    }
                    float gsc = fp16_to_f32(gscf[(size_t)g * N + toff[t] + n]);
                    accd[(size_t)m * N + toff[t] + n] +=
                        (double)p2[(size_t)m * tn[t] + n] * f2 * (double)gsc;
                }
            }
            g_t_acc += now() - ta;
        }
    }
    ta = now();
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++)
            out[(size_t)m * N + n] = (float)accd[(size_t)m * N + n] * sc_row[m];
    g_t_fin += now() - ta;
    ta = now() - t0;
    g_t_matmul += ta; g_t_mm[mi] += ta; g_n_mm[mi]++;
}

/* ---- Phase 7d: merged N-tile matmul (M=1 decode, up/gate).
 * Per K-block: dequant FULL [32,N] once -> 1 merged pass1 submission ->
 * block_max -> r_opt -> 1 merged pass2 submission (DQ still device-valid).
 * Replaces ntiles per-tile submissions per pass (6 -> 1).  Same semantics /
 * bit-exactness as eng_matmul (host int32 hacc reference, block-shared r_opt). */
static void eng_matmul_merged(CVI_RT_HANDLE rt, CVI_RT_MEM mem, uint64_t pa, uint8_t *va,
                              const char *name, const int8_t *x_i8, int M,
                              const uint8_t *nib, const uint16_t *gscf,
                              int K, int N, const float *sc_row, float *out,
                              MergedPool *mp, int *bad1, int *bad2, int *rbad) {
    double ta, t0 = now();
    int KG = K / G;
    int mi = (name && name[0] == 'u') ? 4 : 5;   /* up=4, gate=5 */

    /* ---- rsafe: 查表 (Phase 7c) 或运行时 wmax 预扫 (同 eng_matmul) ---- */
    int rsafe;
    if (g_rsh_skip) {
        rsafe = rsh_lookup(g_cur_layer, mi, 0);
        if (g_verify) {
            int wmax = 0;
            ta = now();
            for (size_t i = 0; i < (size_t)KG * N * 16; i++) {
                uint8_t b = nib[i];
                int lo = b & 0xF, hi = b >> 4; if (lo > 7) lo -= 16; if (hi > 7) hi -= 16;
                int a = lo < 0 ? -lo : lo, bb = hi < 0 ? -hi : hi;
                if (a > wmax) wmax = a; if (bb > wmax) wmax = bb;
            }
            g_t_other += now() - ta;
            int sr = matmul_rshift_w(G, wmax) - 3; if (sr < 4) sr = 4;
            if (sr != rsafe) g_rsh_bad++;
        }
    } else {
        int wmax = 0;
        ta = now();
        for (size_t i = 0; i < (size_t)KG * N * 16; i++) {
            uint8_t b = nib[i];
            int lo = b & 0xF, hi = b >> 4; if (lo > 7) lo -= 16; if (hi > 7) hi -= 16;
            int a = lo < 0 ? -lo : lo, bb = hi < 0 ? -hi : hi;
            if (a > wmax) wmax = a; if (bb > wmax) wmax = bb;
        }
        g_t_other += now() - ta;
        rsafe = matmul_rshift_w(G, wmax) - 3; if (rsafe < 4) rsafe = 4;
    }

    memset(accd, 0, sizeof(double) * (size_t)M * N);

    for (int g = 0; g < KG; g++) {
        const uint8_t *nibg = nib + (size_t)g * N * 16;
        int block_max = 0, gold_max = 0;

        /* ---- dequant full [32,N] into TILE-MAJOR layout (tile t at
         * DQ_OFF + t*32*tilew, each tile [32,tilew] K-major) — the merged
         * cmdbuf's g2lR reads exactly this layout.  (Full-block row-major
         * dequant would NOT match the per-tile g2lR stride.) ---- */
        ta = now();
        for (int t = 0; t < mp->nt; t++)
            dequant_kal_rvv(nibg + (size_t)t * mp->tilew * 16, mp->tilew,
                            (int8_t *)(va + DQ_OFF) + (size_t)t * 32 * mp->tilew);
        g_t_dequant += now() - ta;
        ta = now(); memcpy(va + ACTQ_OFF, x_i8 + (size_t)g * 32, 32); g_t_copyact += now() - ta;
        if (g_verify) {
            ta = now();
            const int8_t *w = (const int8_t *)(va + DQ_OFF);
            const int8_t *xr = x_i8 + (size_t)g * 32;
            int32_t *ar = hacc;
            for (int t = 0; t < mp->nt; t++)
                for (int n = 0; n < mp->tilew; n++) {
                    int32_t s = 0;
                    for (int k = 0; k < 32; k++) s += (int32_t)xr[k] * (int32_t)w[(size_t)t * 32 * mp->tilew + (size_t)k * mp->tilew + n];
                    ar[(size_t)t * mp->tilew + n] = s;
                }
            g_t_verify += now() - ta;
        }
        ta = now(); CVI_RT_MemFlushEx(rt, mem, DQ_OFF + (size_t)32 * N); g_t_flush += now() - ta;

        /* ---- pass1: one merged submission (all N tiles) ---- */
        ta = now(); CVI_RT_RunCmdbufEx(rt, mp->ld[rsafe][0], &(CVI_RT_ARRAYBASE){.gaddr_base0 = pa});
        g_t_runcmd += now() - ta; g_n_runcmd++; g_runs_pass1++;
        ta = now(); CVI_RT_MemInvldEx(rt, mem, P1_OFF + (size_t)N); g_t_invld += now() - ta;
        int8_t *p1 = (int8_t *)(va + P1_OFF);
        ta = now();
        for (int n = 0; n < N; n++) {
            int av = p1[n]; if (av < 0) av = -av; if (av > block_max) block_max = av;
            if (g_verify) {
                int gv = int8_round_div(hacc[n], rsafe);
                if (p1[n] != gv) (*bad1)++;
                if (gv < 0) gv = -gv; if (gv > gold_max) gold_max = gv;
            }
        }
        g_t_bm += now() - ta;
        long long est = (long long)block_max << rsafe;
        int r_opt = 0; while (est > (127LL << r_opt)) r_opt++;
        if (g_verify) {
            long long gest = (long long)gold_max << rsafe;
            int r_ref = 0; while (gest > (127LL << r_ref)) r_ref++;
            if (r_opt != r_ref) (*rbad)++;
        }

        /* ---- pass2: one merged submission (DQ still valid on device) ---- */
        ta = now(); CVI_RT_RunCmdbufEx(rt, mp->ld[r_opt][1], &(CVI_RT_ARRAYBASE){.gaddr_base0 = pa});
        g_t_runcmd += now() - ta; g_n_runcmd++; g_runs_pass2++;
        ta = now(); CVI_RT_MemInvldEx(rt, mem, P2_OFF + (size_t)N); g_t_invld += now() - ta;
        int8_t *p2 = (int8_t *)(va + P2_OFF);
        double f2 = (double)(1 << r_opt);
        ta = now();
        for (int n = 0; n < N; n++) {
            if (g_verify) {
                if (p2[n] != int8_round_div(hacc[n], r_opt)) (*bad2)++;
            }
            float gsc = fp16_to_f32(gscf[(size_t)g * N + n]);
            accd[n] += (double)p2[n] * f2 * (double)gsc;
        }
        g_t_acc += now() - ta;
    }
    ta = now();
    for (int n = 0; n < N; n++) out[n] = (float)accd[n] * sc_row[0];
    g_t_fin += now() - ta;
    ta = now() - t0;
    g_t_matmul += ta; g_t_mm[mi] += ta; g_n_mm[mi]++;
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

/* ---- Phase 7e: gsc cache — startup load + per-layer pointer override ----
 * gsc regions per layer are small (total 931840 B) and re-read every decode
 * step from cold mmap pages (~8MB/s on SD).  We pread layers' gsc ONCE at startup
 * into an ION buffer (CPU-cached ~2200MB/s) and point the LayerRef gsc fields into
 * it, eliminating the per-step SD gsc reads entirely.
 *   GSC_ION=1 (mmap 基线): 24 层全 ION (GSC_TOTAL_BYTES).
 *   B-2 (ion_db): GSC_ION_LAYERS 层 ION + GSC_DDR_LAYERS 层 DDR mmap (page cache),
 *   配合 nib-only SD_BUF.
 * Bit-exact: same fp16 bytes, same conversion in accum.  ION-only (no anonymous
 * buffers); single LoadDmabuf rule unaffected (data-only buffer, never submitted). */
static int gsc_layer_load(int fd, uint8_t *dst) {
    /* 7 gsc regions in parse_layer order: Wq,Wk,Wv,Wo,up,gate,down */
    if (pread(fd, dst, GSC_WQ_SZ, GSC_WQ_OFF) != (ssize_t)GSC_WQ_SZ) return -1;
    if (pread(fd, dst + GSC_WQ_SZ, GSC_WK_SZ, GSC_WK_OFF) != (ssize_t)GSC_WK_SZ) return -1;
    if (pread(fd, dst + GSC_WQ_SZ + GSC_WK_SZ, GSC_WV_SZ, GSC_WV_OFF) != (ssize_t)GSC_WV_SZ) return -1;
    if (pread(fd, dst + GSC_WQ_SZ + GSC_WK_SZ + GSC_WV_SZ, GSC_WO_SZ, GSC_WO_OFF) != (ssize_t)GSC_WO_SZ) return -1;
    if (pread(fd, dst + GSC_WQ_SZ + GSC_WK_SZ + GSC_WV_SZ + GSC_WO_SZ, GSC_UP_SZ, GSC_UP_OFF) != (ssize_t)GSC_UP_SZ) return -1;
    if (pread(fd, dst + GSC_WQ_SZ + GSC_WK_SZ + GSC_WV_SZ + GSC_WO_SZ + GSC_UP_SZ, GSC_GA_SZ, GSC_GA_OFF) != (ssize_t)GSC_GA_SZ) return -1;
    if (pread(fd, dst + GSC_WQ_SZ + GSC_WK_SZ + GSC_WV_SZ + GSC_WO_SZ + GSC_UP_SZ + GSC_GA_SZ, GSC_DN_SZ, GSC_DN_OFF) != (ssize_t)GSC_DN_SZ) return -1;
    return 0;
}

/* ---- ION carveout 预检 (GSC_ION=1 OOM/泄漏 稳健性) ----
 * 直读 ION debugfs summary: "[0] carveout heap size:<sz> bytes, used:<u> bytes".
 * 返回当前空闲字节数; 文件不可用返回 -1 (调用方跳过预检, 退化为 MemAlloc NULL 检查).
 * 目的: 在进入 libcviruntime mem_alloc_raw 的 "reopen ion dev 重试->assert" 路径
 * 之前拦截空间不足, 避免 SIGABRT + 退出不释放把已有 ION 全量泄漏 (实测 24.3MB). */
static long long ion_carveout_free(void) {
    FILE *f = fopen("/sys/kernel/debug/ion/cvi_carveout_heap_dump/summary", "r");
    if (!f) return -1;
    long long size = 0, used = -1;
    char line[256];
    while (fgets(line, sizeof line, f)) {
        if (sscanf(line, "[%*d] carveout heap size:%lld bytes, used:%lld bytes", &size, &used) == 2) break;
    }
    fclose(f);
    if (size <= 0 || used < 0 || used > size) return -1;
    return size - used;
}

/* 预检诊断: 打印当前 carveout used/free/peak 供 OOM 排障. */
static void ion_carveout_report(const char *tag) {
    FILE *f = fopen("/sys/kernel/debug/ion/cvi_carveout_heap_dump/summary", "r");
    if (!f) { printf("  [%s] ion summary unavailable\n", tag); return; }
    char line[256];
    while (fgets(line, sizeof line, f)) { line[strcspn(line, "\n")] = 0; printf("  [%s] %s\n", tag, line); }
    fclose(f);
}

/* 24 层全 ION (GSC_ION=1 路径, 非 ion_db).
 * 分级降级 (CEO 建议: 空间不够时 gsc 缓存降为部分层):
 *   Tier 1: 24 层全 ION (free >= 24层 + pools/mp 足迹)
 *   Tier 2: 22 层 ION + 2 层 DDR mmap (free >= 22层 + pools/mp 足迹)  — 复用 B-2 混合缓存
 *   Tier 3: 全冷 mmap (GSC_ION=0, 稳定基线)
 * 任何一级失败都走 rt_alloc_safe/预检, 绝不进入 SDK 重试->SIGABRT 路径. */
static int gsc_ion_load(CVI_RT_HANDLE rt) {
    long long free_ion = ion_carveout_free();
    long long pools_need = (long long)ION_POOLS_EST
                         + (g_use_merged ? (long long)ION_MP_EST : 0);
    long long need24 = (long long)GSC_TOTAL_BYTES + pools_need;
    long long need22 = (long long)GSC_ION_CACHE_BYTES + pools_need;
    int ion_layers = L;
    size_t ion_bytes = GSC_TOTAL_BYTES;

    if (free_ion >= 0 && free_ion < need24) {
        if (free_ion >= need22) {
            fprintf(stderr, "GSC_ION pre-check: carveout free=%lld B < 24层需 %lld B; "
                            "降级 %d 层 ION + %d 层 DDR mmap (Tier 2, 留足 pools/mp 余量)\n",
                    free_ion, need24, GSC_ION_LAYERS, GSC_DDR_LAYERS);
            ion_layers = GSC_ION_LAYERS;
            ion_bytes = GSC_ION_CACHE_BYTES;
        } else {
            fprintf(stderr, "GSC_ION pre-check FAIL: carveout free=%lld B < 22层需 %lld B; "
                            "fallback to cold mmap gsc reads (GSC_ION=0, Tier 3).\n"
                            "  If a prior failed run leaked ION, run 'run_clean.sh --clean' first.\n",
                    free_ion, need22);
            g_gsc_ion = NULL;
            return -1;
        }
    }

    g_gsc_ion_mem = rt_alloc_safe(rt, ion_bytes, "gsc");
    if (!g_gsc_ion_mem) {
        fprintf(stderr, "GSC ION alloc %zu B FAILED -> fallback to cold mmap gsc reads\n", ion_bytes);
        g_gsc_ion = NULL;
        return -1;
    }
    uint8_t *base = CVI_RT_MemGetVAddr(g_gsc_ion_mem);
    uint8_t *dst = base;
    for (int l = 0; l < ion_layers; l++, dst += GSC_LAYER_BYTES) {
        char path[160]; snprintf(path, sizeof path, "%s/layer%d_kal.bin", WDIR, l);
        int fd = open(path, O_RDONLY);
        if (fd < 0) { fprintf(stderr, "gsc open %s\n", path); CVI_RT_MemFree(rt, g_gsc_ion_mem); g_gsc_ion_mem = NULL; return -1; }
        if (gsc_layer_load(fd, dst)) goto err;
        close(fd);
        continue;
    err:
        fprintf(stderr, "gsc pread %s short\n", path); close(fd);
        CVI_RT_MemFree(rt, g_gsc_ion_mem); g_gsc_ion_mem = NULL; return -1;
    }
    /* CPU-only reads: ION is CPU-cached, no flush/invld needed. */
    g_gsc_ion = (uint16_t *)base;
    g_gsc_ion_n = ion_layers;

    /* Tier 2: 剩余层 DDR mmap + 预读入 page cache (与 B-2 同机制, decode 期 0 页错误). */
    size_t ddlen[GSC_DDR_LAYERS] = {0};   /* 每个 DDR 层 mmap 长度 (失败回收用; 均等于层文件大小) */
    for (int i = 0; i < GSC_DDR_LAYERS; i++) {
        int l = ion_layers + i;
        char path[160]; snprintf(path, sizeof path, "%s/layer%d_kal.bin", WDIR, l);
        int fd = open(path, O_RDONLY);
        if (fd < 0) { fprintf(stderr, "gsc DDR open %s failed\n", path); goto ddr_err; }
        struct stat st2;
        if (fstat(fd, &st2) != 0 || st2.st_size <= 0) { fprintf(stderr, "gsc DDR fstat %s failed\n", path); close(fd); goto ddr_err; }
        void *mp = mmap(NULL, (size_t)st2.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (mp == MAP_FAILED) { fprintf(stderr, "gsc DDR mmap %s failed\n", path); close(fd); goto ddr_err; }
        g_gsc_ddr_map[i] = (uint8_t *)mp;
        ddlen[i] = (size_t)st2.st_size;
        uint8_t *gb = (uint8_t *)mp + GSC_WQ_OFF;
        size_t glen = GSC_DN_OFF + GSC_DN_SZ - GSC_WQ_OFF;
        madvise(gb, glen, MADV_WILLNEED);
        for (size_t off = 0; off < glen; off += 4096) (void)gb[off];
        (void)gb[glen - 1];
        close(fd);
        continue;
    ddr_err:
        /* DDR 层 mmap 失败: 回收已映射的 DDR 层 + 已分配 ION, 整体回退 Tier 3 冷 mmap. */
        for (int j = 0; j < i; j++) { if (g_gsc_ddr_map[j]) { munmap(g_gsc_ddr_map[j], ddlen[j]); g_gsc_ddr_map[j] = NULL; } }
        CVI_RT_MemFree(rt, g_gsc_ion_mem); g_gsc_ion_mem = NULL;
        g_gsc_ion = NULL; g_gsc_ion_n = 0;
        fprintf(stderr, "GSC_ION DDR layer fallback FAILED -> cold mmap gsc reads (Tier 3)\n");
        return -1;
    }

    printf("GSC_ION: cached %zu B/layer x%d = %.2f MiB in ION",
           (size_t)GSC_LAYER_BYTES, ion_layers, (double)ion_bytes / 1048576.0);
    if (ion_layers < L) printf(" + %d DDR layers (mmap)", GSC_DDR_LAYERS);
    printf("\n");
    return 0;
}

/* B-2 gsc cache: GSC_ION_LAYERS 层 ION + GSC_DDR_LAYERS 层 DDR (l>=GSC_ION_LAYERS).
 * DDR 层 = mmap 层文件 (MAP_PRIVATE page cache, 可回收 — 匿名 malloc 换页顶爆 28MB
 * RAM 已实测 crash); 省出的 ~1.86MB ION 用于第 3 个 SD 大槽. prefill/回归已触碰全部
 * gsc → decode 期间全 page-cache hit. */
static int gsc_cache_load_b2(CVI_RT_HANDLE rt, size_t lsz) {
    /* 分配前预检: 空间不足时干净报错返回, 不触发 SDK assert/泄漏. */
    long long free_ion = ion_carveout_free();
    if (free_ion >= 0 && free_ion < (long long)GSC_ION_CACHE_BYTES) {
        fprintf(stderr, "B-2 gsc pre-check FAIL: carveout free=%lld B < needed=%zu B\n",
                free_ion, (size_t)GSC_ION_CACHE_BYTES);
        return -1;
    }
    g_gsc_ion_mem = rt_alloc_safe(rt, GSC_ION_CACHE_BYTES, "b2-gsc");
    if (!g_gsc_ion_mem) {
        fprintf(stderr, "B-2 gsc ION alloc %zu B FAILED (carveout 余量不足?)\n", (size_t)GSC_ION_CACHE_BYTES);
        return -1;
    }
    uint8_t *base = CVI_RT_MemGetVAddr(g_gsc_ion_mem);
    for (int l = 0; l < GSC_ION_LAYERS; l++) {
        char path[160]; snprintf(path, sizeof path, "%s/layer%d_kal.bin", WDIR, l);
        int fd = open(path, O_RDONLY);
        if (fd < 0 || gsc_layer_load(fd, base + (size_t)l * GSC_LAYER_BYTES)) {
            fprintf(stderr, "B-2 gsc pread %s failed\n", path);
            if (fd >= 0) close(fd);
            CVI_RT_MemFree(rt, g_gsc_ion_mem); g_gsc_ion_mem = NULL; return -1;
        }
        close(fd);
    }
    g_gsc_ion = (uint16_t *)base;
    g_gsc_ion_n = GSC_ION_LAYERS;
    for (int i = 0; i < GSC_DDR_LAYERS; i++) {
        int l = GSC_ION_LAYERS + i;
        char path[160]; snprintf(path, sizeof path, "%s/layer%d_kal.bin", WDIR, l);
        int fd = open(path, O_RDONLY);
        if (fd < 0) { fprintf(stderr, "B-2 gsc DDR open %s failed\n", path); return -1; }
        void *mp = mmap(NULL, lsz, PROT_READ, MAP_PRIVATE, fd, 0);
        if (mp == MAP_FAILED) {
            fprintf(stderr, "B-2 gsc DDR mmap %s failed\n", path); close(fd); return -1;
        }
        g_gsc_ddr_map[i] = (uint8_t *)mp;
        /* CEO: "init 预读入 DDR" — init 阶段把 DDR 层 gsc 区段预读进 page cache,
         * decode 期 0 页错误 (2 层 = 1.86MB). madvise+顺序 touch 填充 cache. */
        uint8_t *gb = (uint8_t *)mp + GSC_WQ_OFF;
        size_t glen = GSC_DN_OFF + GSC_DN_SZ - GSC_WQ_OFF;
        madvise(gb, glen, MADV_WILLNEED);
        for (size_t off = 0; off < glen; off += 4096) (void)gb[off];
        (void)gb[glen - 1];
        close(fd);
    }
    printf("B-2 gsc cache: %d layers ION (%.2f MiB) + %d layers DDR mmap, gsc init-preheat (%.2f MiB)\n",
           GSC_ION_LAYERS, (double)GSC_ION_CACHE_BYTES / 1048576.0,
           GSC_DDR_LAYERS, (double)(GSC_DDR_LAYERS * GSC_LAYER_BYTES) / 1048576.0);
    return 0;
}

/* B-2 rms 全层缓存 (DDR, L*2*D floats). layer 文件首/尾各 3584B (rms_attn/rms_ffn). */
static int rms_cache_load(size_t lsz) {
    g_rms_all = malloc((size_t)L * 2 * D * sizeof(float));
    if (!g_rms_all) { fprintf(stderr, "B-2 rms malloc FAILED\n"); return -1; }
    for (int l = 0; l < L; l++) {
        char path[160]; snprintf(path, sizeof path, "%s/layer%d_kal.bin", WDIR, l);
        int fd = open(path, O_RDONLY);
        if (fd < 0) { fprintf(stderr, "rms open %s\n", path); return -1; }
        if (pread(fd, g_rms_all + (size_t)l * 2 * D, D * 4, 0) != (ssize_t)(D * 4)) { close(fd); return -1; }
        if (pread(fd, g_rms_all + (size_t)l * 2 * D + D, D * 4, (off_t)(lsz - D * 4)) != (ssize_t)(D * 4)) { close(fd); return -1; }
        close(fd);
    }
    printf("B-2 rms cache: %d layers x %zu B in DDR\n", L, (size_t)2 * D * sizeof(float));
    return 0;
}

static void gsc_ion_apply(LayerRef *lr, int l) {
    uint8_t *base;
    if (l < g_gsc_ion_n) {
        if (!g_gsc_ion) return;
        base = (uint8_t *)g_gsc_ion + (size_t)l * GSC_LAYER_BYTES;
        lr->Wq_gsc   = (const uint16_t *)(base + 0);
        lr->Wk_gsc   = (const uint16_t *)(base + GSC_WQ_SZ);
        lr->Wv_gsc   = (const uint16_t *)(base + GSC_WQ_SZ + GSC_WK_SZ);
        lr->Wo_gsc   = (const uint16_t *)(base + GSC_WQ_SZ + GSC_WK_SZ + GSC_WV_SZ);
        lr->up_gsc   = (const uint16_t *)(base + GSC_WQ_SZ + GSC_WK_SZ + GSC_WV_SZ + GSC_WO_SZ);
        lr->gate_gsc = (const uint16_t *)(base + GSC_WQ_SZ + GSC_WK_SZ + GSC_WV_SZ + GSC_WO_SZ + GSC_UP_SZ);
        lr->down_gsc = (const uint16_t *)(base + GSC_WQ_SZ + GSC_WK_SZ + GSC_WV_SZ + GSC_WO_SZ + GSC_UP_SZ + GSC_GA_SZ);
    } else {
        int i = l - g_gsc_ion_n;
        if (i < 0 || i >= GSC_DDR_LAYERS || !g_gsc_ddr_map[i]) return;
        base = g_gsc_ddr_map[i];   /* 层文件 mmap 基址; gsc 区段按文件偏移散布 */
        lr->Wq_gsc   = (const uint16_t *)(base + GSC_WQ_OFF);
        lr->Wk_gsc   = (const uint16_t *)(base + GSC_WK_OFF);
        lr->Wv_gsc   = (const uint16_t *)(base + GSC_WV_OFF);
        lr->Wo_gsc   = (const uint16_t *)(base + GSC_WO_OFF);
        lr->up_gsc   = (const uint16_t *)(base + GSC_UP_OFF);
        lr->gate_gsc = (const uint16_t *)(base + GSC_GA_OFF);
        lr->down_gsc = (const uint16_t *)(base + GSC_DN_OFF);
    }
}

/* ---------------- Phase 7/7b: 层权重读路径 (bandwidth attribution & fix) ----------------
 * decode 瓶颈: 层权重 201MB/token 逐层重读, 裸 mmap demand-paging ~8MB/s
 * (每 4KB 页 fault 串行 + TIU 计算间隙 SD 空闲, readahead 跟不上).
 * SD 顺序读天花板 ~21.5MB/s (Phase 6 实测). 读模式 (LW_READ env):
 *   mmap    : 裸 mmap (Phase 6 基线, swap-safe 但 ~8MB/s)
 *   mmap_ra : mmap + readahead(fd,0,lsz) + madvise(SEQUENTIAL) 整层预取到 page
 *             cache (swap-safe, 无匿名 buffer; Phase 7 已证反而变慢: 本层同步
 *             readahead 与计算串行且争抢 SD)
 *   pread   : 顺序 pread 整层入复用 buffer (proven ~21MB/s; +8.4MB 匿名,
 *             有 swap 抖动风险, 仅作对照)
 *   mmap_db : [Phase 7b] 跨层双缓冲: 算本层 l 时异步 readahead 预取层 l+1 到
 *             page cache (file-backed, clean 可回收, swap-safe). 层 l+1 计算时
 *             全 cache hit -> 读 ~9.7s 藏在计算 ~15.6s 后, t_layers -> max(读,算).
 * 数据字节与 mmap 完全一致 → parse_layer/数值/TIU 路径不变, bit-exact 保持.
 */
#define LW_MMAP    0
#define LW_MMAP_RA 1
#define LW_PREAD   2
#define LW_MMAP_DB 3
#define LW_MMAP_TH 4
static int g_lw_mode = LW_MMAP;
static uint8_t *g_layer_buf = NULL;   /* LW_PREAD 复用 buffer (main 分配) */
static int g_ra_err = 0;              /* mmap_ra / mmap_db 下 readahead 失败计数 (诊断) */
static long g_ra_res = 0, g_ra_tot = 0; /* mmap_ra mincore 驻留快照累计 (诊断) */
static unsigned char g_minc[2050];    /* 层 max 页数 = ceil(8393728/4096) = 2050 */

/* ---- [Phase 7b] LW_MMAP_TH: 双核预取线程 (cross-layer double-buffer on core 2).
 * readahead() 在本系统是同步阻塞的 (mmap_db 主线程发 readahead 每层 ~0.4s,
 * t_layers 反而 +9.6s/token). 用 pthread 把 readahead 挪到第二个核: 主线程算层 l
 * 时, 预取线程同步 readahead 层 l+1 到 page cache (file-backed, clean 可回收,
 * swap-safe). 主线程只发信号不等待 -> 读 ~9.7s 藏在算 ~9.5s 后. */
static pthread_t g_pf_thread;
static pthread_mutex_t g_pf_mtx = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_pf_cv = PTHREAD_COND_INITIALIZER;
static int g_pf_target = -1;        /* -1 = idle; >=0 = 待预取层 */
static int g_pf_shutdown = 0;
static size_t g_pf_lsz = 0;

static void *prefetch_thread_main(void *arg) {
    (void)arg;
    for (;;) {
        int l;
        pthread_mutex_lock(&g_pf_mtx);
        while (!g_pf_shutdown && g_pf_target < 0) pthread_cond_wait(&g_pf_cv, &g_pf_mtx);
        if (g_pf_shutdown) { pthread_mutex_unlock(&g_pf_mtx); return NULL; }
        l = g_pf_target; g_pf_target = -1;
        pthread_mutex_unlock(&g_pf_mtx);
        char path[160];
        snprintf(path, sizeof path, "%s/layer%d_kal.bin", WDIR, l);
        int pfd = open(path, O_RDONLY);
        if (pfd >= 0) {
            long off = 0;
            while (off < (long)g_pf_lsz) {
                size_t n = g_pf_lsz - (size_t)off < 524288 ? g_pf_lsz - (size_t)off : 524288;
                if (readahead(pfd, off, n) != 0) g_ra_err++;
                off += n;
            }
            close(pfd);
        }
    }
}
static void prefetch_issue(int l) {  /* 非阻塞: 只设目标并唤醒预取线程 */
    if (g_pf_target >= 0) return;     /* 上一目标未消费完则跳过 (graceful) */
    pthread_mutex_lock(&g_pf_mtx);
    g_pf_target = l;
    pthread_cond_signal(&g_pf_cv);
    pthread_mutex_unlock(&g_pf_mtx);
}

/* ---- A' (LW_ION_DB): B-2 per-matrix ION 双缓冲 SD 预读 (CEO 授权形态) ----
 * 每层 7 矩阵 (q,k,v,wo,up,gate,down) 的 nib 逐矩阵预读: 背景线程
 * O_DIRECT pread(2MiB 对齐 bounce) -> CPU memcpy -> ION SD_BUF_A/B (per-matrix,
 * 槽 = midx&1, midx = l*7+mi, 槽大小 = max nib 2,179,072). 主线程算矩阵 midx 时
 * 预取 midx+1, SD 读与 TIU 重叠. nib-only: gsc/rms 走缓存 (gsc 22 层 ION + 2 层
 * DDR mmap; rms 全层 DDR), 每 token SD 流量 = 179.17MB -> 地板 8.33s.
 * bit-exact 由构造保持: SD_BUF 字节 = 原层文件 nib 区段字节 (同源), 数值/TIU/LM
 * head 路径全部不动. 仅 CPU 消费 SD_BUF (dequant/accum), ION CPU-cacheable,
 * 无设备访问, 无需 flush/invld. */
#define SD_NSLOT 2          /* 槽数: 2 大槽(2.18MB), 全 ION (per-matrix 双缓冲, 槽=midx&1) */
#define SD_QCAP 16          /* 预取请求 FIFO 深度 (重排用, 给 SD 线程更多匹配选择) */
#define SD_LOOKAHEAD 10     /* 消费完后向前预取的矩阵数 (≥ 一层矩阵数+3, 重排后 SD 线程不空转) */
#define SD_BUF_SZ 2179072    /* 大矩阵 nib (up/gate/down = 2,179,072) 槽大小; 小矩阵弹性占用 */
#define SD_BOUNCE_SZ 2097152 /* 2MiB 对齐 bounce (O_DIRECT 512 对齐; 2MiB 减 pread 次数, ~+1MB anon) */
#define LW_ION_DB 5
#define NMAT 7               /* 每层矩阵数 (q,k,v,wo,up,gate,down) */
/* B-2: 各矩阵 nib 在层文件中的 span (parse_layer 顺序; 偏移/尺寸 512 对齐已验证) */
static const size_t g_mat_off[NMAT] = { 3584, 455168, 519680, 584192, 1035776, 3487232, 5938688 };
static const size_t g_mat_len[NMAT] = { 401408, 57344, 57344, 401408, 2179072, 2179072, 2179072 };
static CVI_RT_MEM g_sd_ion[SD_NSLOT] = {NULL, NULL};   /* 2 槽全 ION (per-matrix 双缓冲) */
static uint8_t   *g_sd_va[SD_NSLOT] = {NULL, NULL};
/* slot 状态: g_sd_midx[s]=槽内矩阵 midx(l*7+mi); -1=空; -2=读入中. g_sd_ready[s]=1 数据就位 */
static volatile int g_sd_midx[SD_NSLOT] = {-1, -1};
static volatile int g_sd_ready[SD_NSLOT] = {0, 0};
static int g_req_q[SD_QCAP];                    /* 预取请求 FIFO (严格升序) */
static int g_req_head = 0, g_req_tail = 0, g_req_n = 0;
static int g_pushed_max = -1;                   /* 已入流水线的最高 midx (单调) */
static int g_consumed_max = -1;                 /* 已消费的最高 midx (单调) */
static int g_nib_fd = -1, g_nib_fd_l = -1;      /* nib 读 fd 缓存 (按层复用, 168->24 opens/step) */
static int g_ion_shutdown = 0;
static pthread_mutex_t g_ion_mtx = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  g_ion_req_cv  = PTHREAD_COND_INITIALIZER; /* SD 线程: 有请求/空槽 */
static pthread_cond_t  g_ion_done_cv = PTHREAD_COND_INITIALIZER; /* 主线程: 矩阵就绪 */
static uint8_t *g_bounce = NULL;
static size_t g_bounce_sz = 0, lsz_global = 0;
static int g_pf_n = 0, g_pf_err = 0;                          /* A' 指标 */
static volatile int g_pf_fail[L * NMAT];                     /* per-matrix 预读失败标记 (rem 索引, 跨 pass 复用;
                                                                main 线程等待矩阵时若置位则转同步补载, 避免死锁) */
static double g_t_sd_read = 0, g_t_memcpy = 0, g_t_sd_wait = 0; /* A' 指标 (SD 读+拷贝 / memcpy / 主线程等待) */
static double g_t_sd_idle = 0;         /* A' 指标: SD 线程阻塞等待时间 (诊断用) */
static double g_t_sd_idle_empty = 0, g_t_sd_idle_noslot = 0; /* 空队列 / 无匹配槽 分项 */
static int g_reorder_n = 0;            /* req_peek_fit 实际发生重排 (idx>0) 的次数 */
static int g_pass_base = 0;              /* 全局单调 pass 基址: 每 pass += L*NMAT (168), 消除跨 pass 流水线重置 */
static volatile int g_busy = 0;          /* 预读线程在途读计数 (ion_drain 用) */

/* ---- B-2 per-matrix prefetch machinery (v2: 2 槽双缓冲 + 请求队列, 多步前瞻) ----
 * 流水线: 主线程消费矩阵 midx, 槽数 SD_NSLOT=2 (全 ION 大槽), 每消费一个矩阵向前预取
 * SD_LOOKAHEAD=10 个 (FIFO 队列), SD 线程串行 O_DIRECT 读 -> bounce -> memcpy 入槽.
 * SD-bound 稳态下 SD 线程永不空转, t_layers -> SD_total (地板 ~8.33s, nib-only).
 * 双缓冲: 任一空闲槽承接在途读, 主线程消费另一槽; 2 槽全 2.18MB, 矩阵弹性放置. */

/* ---- 请求队列 / 槽查找 (均须持锁) ---- */
static void req_push(int midx) {
    if (g_req_n >= SD_QCAP) return;      /* 不会发生: topup 受 g_req_n 限制 */
    g_req_q[g_req_tail] = midx;
    g_req_tail = (g_req_tail + 1) % SD_QCAP;
    g_req_n++;
}
/* 取出队列中第 idx 个请求 (idx=0 即头部). 移动后续元素保持循环数组紧凑. */
static int req_remove_at(int idx) {
    int m = g_req_q[(g_req_head + idx) % SD_QCAP];
    if (idx == 0) {
        g_req_head = (g_req_head + 1) % SD_QCAP;
    } else {
        for (int i = idx; i < g_req_n - 1; i++)
            g_req_q[(g_req_head + i) % SD_QCAP] = g_req_q[(g_req_head + i + 1) % SD_QCAP];
        g_req_tail = (g_req_tail - 1 + SD_QCAP) % SD_QCAP;
    }
    g_req_n--;
    return m;
}
/* 返回能容纳 midx 矩阵的空槽: 2 槽均为 2.18MB 大槽, 任何矩阵可进任意槽 (双缓冲). */
static int slot_find_free(int midx) {
    (void)midx;
    for (int s = 0; s < SD_NSLOT; s++) if (g_sd_midx[s] == -1) return s;
    return -1;
}
static int slot_find_midx(int midx) {
    for (int s = 0; s < SD_NSLOT; s++) if (g_sd_midx[s] == midx) return s;
    return -1;
}
/* 重排预取: 返回队列中第一个能匹配空闲槽的请求下标 (大矩阵只进大槽 0/1, 小矩阵
 * 优先小槽 2), 并回填 midx/slot. 避免 head 是大矩阵被大槽占满阻塞时, 后续小矩阵
 * 空转等槽 — 把 SD 线程的空闲窗口填上可并行的读. head 优先 (i=0 先查), 不延迟
 * 主线程最需要的矩阵. 无匹配返回 -1. */
static int req_peek_fit(int *midx_out, int *slot_out) {
    for (int i = 0; i < g_req_n; i++) {
        int m = g_req_q[(g_req_head + i) % SD_QCAP];
        int s = slot_find_free(m);
        if (s >= 0) { *midx_out = m; *slot_out = s; return i; }
    }
    return -1;
}
/* 填充流水线: 消费完 consumed_max 后, 把后面至多 SD_LOOKAHEAD 个未入队矩阵推入队列. */
static void ion_topup(void) {
    int last = g_pass_base + L * NMAT;   /* 本 pass 上界 (不跨界预取下一 pass) */
    int target = g_consumed_max + SD_LOOKAHEAD;
    while (g_pushed_max + 1 <= target && g_pushed_max + 1 < last && g_req_n < SD_QCAP)
        req_push(++g_pushed_max);
    pthread_cond_signal(&g_ion_req_cv);
}

/* 低层同步补载: 直接 pread+memcpy 装入指定槽 (主线程上下文, 无锁区). */
static void ion_mat_sync(int l, int mi, int slot) {
    char path[160];
    snprintf(path, sizeof path, "%s/layer%d_kal.bin", WDIR, l);
    int fd = open(path, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "[ion_db] open %s\n", path); exit(2); }
    size_t len = g_mat_len[mi], off = g_mat_off[mi], dn = 0;
    uint8_t *dst = g_sd_va[slot];
    while (dn < len) {
        size_t n = (len - dn < g_bounce_sz) ? (len - dn) : g_bounce_sz;
        if (pread(fd, g_bounce, n, off + dn) != (ssize_t)n) { fprintf(stderr, "[ion_db] short pread %s\n", path); exit(2); }
        memcpy(dst + dn, g_bounce, n);
        dn += n;
    }
    close(fd);
}

/* 预读线程: 从请求队列取 midx, O_DIRECT pread -> bounce -> memcpy 入空闲槽.
 * O_DIRECT 读不污染 page cache (给 DDR gsc/rms 缓存让内存); fs 不支持则回退普通读. */
static void *ion_prefetch_thread(void *arg) {
    (void)arg;
    for (;;) {
        int midx, slot;
        pthread_mutex_lock(&g_ion_mtx);
        for (;;) {
            if (g_ion_shutdown) {
                if (g_nib_fd >= 0) { close(g_nib_fd); g_nib_fd = -1; g_nib_fd_l = -1; }
                pthread_mutex_unlock(&g_ion_mtx); return NULL;
            }
            if (g_req_n > 0) {
                int s, idx = req_peek_fit(&midx, &s);
                if (idx >= 0) { slot = s; if (idx > 0) g_reorder_n++; req_remove_at(idx); break; }
            }
            double tw = now();
            int was_empty = (g_req_n == 0);
            pthread_cond_wait(&g_ion_req_cv, &g_ion_mtx);
            double w = now() - tw;
            g_t_sd_idle += w;
            if (was_empty) g_t_sd_idle_empty += w; else g_t_sd_idle_noslot += w;
        }
        g_sd_midx[slot] = -2;            /* 占用 (读入中) */
        g_busy++;                        /* 在途读 +1 (ion_drain 需要) */
        pthread_mutex_unlock(&g_ion_mtx);

        int rem = midx % (L * NMAT);     /* 全局单调 midx -> 层内索引 */
        int l = rem / NMAT, mi = rem % NMAT;
        double t0 = now();
        /* fd 缓存: 矩阵按层顺序读取, 同层复用 fd (168 opens/step -> 24) */
        if (g_nib_fd < 0 || g_nib_fd_l != l) {
            if (g_nib_fd >= 0) close(g_nib_fd);
            char path[160];
            snprintf(path, sizeof path, "%s/layer%d_kal.bin", WDIR, l);
            g_nib_fd = open(path, O_RDONLY | O_DIRECT);
            if (g_nib_fd < 0) g_nib_fd = open(path, O_RDONLY);   /* O_DIRECT 不可用回退 */
            g_nib_fd_l = l;
            /* DDR gsc 层 (2 层): init 已预读入 page cache (gsc_cache_load_b2); 此分支
             * 为兜底 (page cache 被回收时再次 touch, 与 nib 读重叠, 避免 main 线程
             * accum 的 gsc 页错误串行). 每层 ~0.93MB, 2 层 +0.09s/step SD 上限. */
            if (l >= GSC_ION_LAYERS) {
                int di = l - GSC_ION_LAYERS;
                if (di < GSC_DDR_LAYERS && g_gsc_ddr_map[di]) {
                    uint8_t *mp = g_gsc_ddr_map[di];
                    size_t g0 = GSC_WQ_OFF, g1 = GSC_DN_OFF + GSC_DN_SZ;
                    if (g1 > g0) {
                        madvise(mp + g0, g1 - g0, MADV_WILLNEED);
                        for (size_t off = g0; off < g1; off += 4096) (void)mp[off];
                        (void)mp[g1 - 1];
                    }
                }
            }
        }
        if (g_nib_fd < 0) {
            pthread_mutex_lock(&g_ion_mtx);
            g_pf_err++; g_sd_midx[slot] = -1; g_busy--;
            pthread_cond_signal(&g_ion_req_cv);
            pthread_mutex_unlock(&g_ion_mtx);
            continue;
        }
        int fd = g_nib_fd;
        size_t len = g_mat_len[mi], off = g_mat_off[mi], done = 0;
        uint8_t *dst = g_sd_va[slot];
        int perr = 0;
        while (done < len) {
            size_t n = (len - done < g_bounce_sz) ? (len - done) : g_bounce_sz;
            if (pread(fd, g_bounce, n, off + done) != (ssize_t)n) { perr = 1; break; }
            double tm = now();
            memcpy(dst + done, g_bounce, n);
            g_t_memcpy += now() - tm;
            done += n;
        }
        pthread_mutex_lock(&g_ion_mtx);
        if (perr) { g_pf_err++; g_pf_fail[rem] = 1; g_sd_midx[slot] = -1; }  /* 记录失败: main 线程不再永久等待 */
        else      { g_sd_midx[slot] = midx; g_sd_ready[slot] = 1; g_pf_n++; g_pf_fail[rem] = 0; }
        g_busy--;                      /* 在途读 -1 */
        g_t_sd_read += now() - t0;     /* 读+拷贝总耗时 (线程侧统计) */
        pthread_cond_broadcast(&g_ion_done_cv);      /* 主线程可能在等 midx / drain */
        pthread_cond_signal(&g_ion_req_cv);          /* 队列可能还有请求/新空槽 */
        pthread_mutex_unlock(&g_ion_mtx);
    }
}

/* LM head 前排空预读流水线: 等队列空 + 无在途读 (避免 SD 并发读顶爆内存,
 * 导致 embed_cl pread 失败). ion_topup 上界 = g_pass_base+L*NMAT, 层循环结束
 * 后队列自然为空; 本函数作为保险 + 保证下一 pass 冷同步前 SD 线程完全空闲. */
static void ion_drain(void) {
    pthread_mutex_lock(&g_ion_mtx);
    while (!g_ion_shutdown && (g_req_n > 0 || g_busy > 0))
        pthread_cond_wait(&g_ion_done_cv, &g_ion_mtx);
    pthread_mutex_unlock(&g_ion_mtx);
}

/* 计算矩阵 (l, mi) 前调用: 等槽就绪 + 填充流水线. 返回 nib 基址 (指向槽).
 * midx = g_pass_base + l*7+mi. 全局单调, 跨 pass 无缝流水线; 每 pass 起点
 * q(0) 若超出 g_pushed_max 则冷同步 (~18ms), 其余矩阵由预读线程提前入槽. */
static uint8_t *layer_io_mat(int l, int mi) {
    int midx = g_pass_base + l * NMAT + mi;   /* 全局单调, 跨 pass 无缝流水线 */
    pthread_mutex_lock(&g_ion_mtx);
    /* 上一矩阵 midx-1 已消费: 释放其槽供预读线程使用. 全局单调, 无需 pass 起点
     * 重置; midx>0 即释放 (pass 边界 q(0) 时 midx-1 = 上 pass 末 down, 同样要释放). */
    if (midx > 0) {
        int s = slot_find_midx(midx - 1);
        if (s >= 0) { g_sd_midx[s] = -1; g_sd_ready[s] = 0; }
    }
    g_consumed_max = midx - 1;
    int slot;
    if (midx > g_pushed_max) {
        /* 未入流水线: 冷同步 (pass 起点 q(0) / 预取失败兜底) */
        while (!g_ion_shutdown && slot_find_free(midx) < 0)
            pthread_cond_wait(&g_ion_req_cv, &g_ion_mtx);
        slot = slot_find_free(midx);
        if (slot < 0) { pthread_mutex_unlock(&g_ion_mtx); exit(2); }
        g_sd_midx[slot] = -2;
        pthread_mutex_unlock(&g_ion_mtx);
        ion_mat_sync(l, mi, slot);
        pthread_mutex_lock(&g_ion_mtx);
        g_sd_midx[slot] = midx; g_sd_ready[slot] = 1;
        g_pushed_max = midx;
        ion_topup();                    /* 入队 midx+1..midx+LOOKAHEAD */
        pthread_mutex_unlock(&g_ion_mtx);
        return g_sd_va[slot];
    }
    /* 已入流水线: 填充 + 等待就绪. 若预读线程对该矩阵 pread 失败 (g_pf_fail[rem]),
     * 不再永久等 g_ion_done_cv (失败矩阵不会重入队 -> 死锁); 主线程转同步补载
     * (ion_mat_sync), 失败则内部 exit(2) 干净退出 (Phase 8 死锁卫生修复). */
    ion_topup();
    double tw = now();
    int rem = midx % (L * NMAT), failed = 0;
    for (;;) {
        slot = slot_find_midx(midx);
        if (slot >= 0 && g_sd_ready[slot]) break;
        if (g_pf_fail[rem]) { failed = 1; break; }
        pthread_cond_wait(&g_ion_done_cv, &g_ion_mtx);
    }
    if (failed) {
        g_pf_fail[rem] = 0;                              /* 清标记, 只补载一次 */
        while (!g_ion_shutdown && slot_find_free(midx) < 0)
            pthread_cond_wait(&g_ion_req_cv, &g_ion_mtx);
        slot = slot_find_free(midx);
        if (slot < 0) { pthread_mutex_unlock(&g_ion_mtx); exit(2); }
        g_sd_midx[slot] = -2;
        pthread_mutex_unlock(&g_ion_mtx);
        ion_mat_sync(l, mi, slot);                       /* pread 失败内部 exit(2), 不返回 */
        pthread_mutex_lock(&g_ion_mtx);
        g_sd_midx[slot] = midx; g_sd_ready[slot] = 1;
        g_t_sd_wait += now() - tw;
        pthread_mutex_unlock(&g_ion_mtx);
        return g_sd_va[slot];
    }
    g_t_sd_wait += now() - tw;
    pthread_mutex_unlock(&g_ion_mtx);
    return g_sd_va[slot];
}

/* B-2: ion_db 模式下构造 LayerRef — rms/gsc 走缓存, nib 指针由 layer_io_mat 逐矩阵
 * 填入 (先置 NULL 占位; mat_nib 非 ion_db 模式返回 fallback 原指针). */
static void layer_io_lr(LayerRef *lr, int l) {
    lr->rms_attn = g_rms_all + (size_t)l * 2 * D;
    lr->rms_ffn  = g_rms_all + (size_t)l * 2 * D + D;
    lr->Wq_nib = lr->Wk_nib = lr->Wv_nib = lr->Wo_nib = NULL;
    lr->up_nib = lr->gate_nib = lr->down_nib = NULL;
    lr->Wq_gsc = lr->Wk_gsc = lr->Wv_gsc = lr->Wo_gsc = NULL;
    lr->up_gsc = lr->gate_gsc = lr->down_gsc = NULL;
    gsc_ion_apply(lr, l);   /* B-2 gsc cache 已加载 (init 校验), 指针必有值 */
}

/* 取矩阵 mi 的 nib 基址: ion_db 模式走 per-matrix 流水, 其他模式返回 fallback. */
static const uint8_t *mat_nib(LayerRef *lr, int l, int mi, const uint8_t *fallback) {
    (void)lr;
    if (g_lw_mode == LW_ION_DB) return layer_io_mat(l, mi);
    return fallback;
}

typedef struct {
    int mode;
    uint8_t *src;     /* parse_layer 的连续源 (mmap 区域或 pread buffer) */
    size_t lsz;
} LayerIO;

static int layer_io_begin(int l, size_t lsz, LayerIO *io) {
    char path[160];
    snprintf(path, sizeof path, "%s/layer%d_kal.bin", WDIR, l);
    int lfd = open(path, O_RDONLY);
    if (lfd < 0) { fprintf(stderr, "cannot open %s\n", path); return -1; }
    io->mode = g_lw_mode; io->lsz = lsz;
    if (g_lw_mode == LW_ION_DB) {
        /* B-2 per-matrix: 本函数仅占位 (src=NULL), 逐矩阵 nib 由 layer_io_mat 装载,
         * rms/gsc 走缓存 (layer_io_lr). 层文件校验 (存在性) 仍做一次 open. */
        close(lfd);
        io->src = NULL;
        return 0;
    }
    if (g_lw_mode == LW_PREAD) {
        /* 顺序 pread 整层 (offset 升序, 1MB 大块) — 命中 SD 顺序读上限 */
        long remain = (long)lsz, off = 0;
        while (remain > 0) {
            long n = remain < 1048576 ? remain : 1048576;
            if (pread(lfd, g_layer_buf + off, n, off) != n) { close(lfd); return -1; }
            off += n; remain -= n;
        }
        close(lfd);
        io->src = g_layer_buf;
    } else {
        uint8_t *lm = mmap(NULL, lsz, PROT_READ, MAP_PRIVATE, lfd, 0);
        if (lm == MAP_FAILED) { fprintf(stderr, "mmap %s\n", path); close(lfd); return -1; }
        if (g_lw_mode == LW_MMAP_RA) {
            /* 全文件异步预取到 page cache + 顺序 hint.
             * readahead() 为 VFS 通用 syscall (内部自动分片), 把整层顺序大块 I/O
             * 一次性提交到 block layer, SD 队列持续有请求 → 连续 ~21.5MB/s;
             * 消费端 page fault 全变 cache hit. 512KB 用户态分片为无害冗余.
             * fd 关闭不影响已提交的 readahead. */
            long off = 0;
            while (off < (long)lsz) {
                size_t n = lsz - (size_t)off < 524288 ? lsz - (size_t)off : 524288;
                if (readahead(lfd, off, n) != 0) g_ra_err++;
                off += n;
            }
            madvise(lm, lsz, MADV_SEQUENTIAL);
            /* mincore 驻留快照: readahead 提交后、compute 触页前, 已驻留页占比.
             * 辅助区分"readahead 生效"vs"部分预取/被回收" (g_ra_err==0 不够). */
            size_t npg = (lsz + 4095) / 4096;
            if (npg <= sizeof g_minc && mincore(lm, lsz, g_minc) == 0) {
                for (size_t i = 0; i < npg; i++) if (g_minc[i] & 1) g_ra_res++;
                g_ra_tot += npg;
            }
        } else if (g_lw_mode == LW_MMAP_DB) {
            /* [Phase 7b] 跨层双缓冲 (主线程 readahead): 本层 l 刚 mmap(未触页),
             * 立即提交层 l+1 整层 readahead 到 page cache. 实测: readahead() 在本
             * 系统是同步阻塞的, 主线程被卡 ~0.4s/层 -> t_layers +9.6s/token (负结果).
             * 保留此分支供 A/B 对照; 推荐用 LW_MMAP_TH (双核预取线程). */
            if (l + 1 < L) {
                char p2[160];
                snprintf(p2, sizeof p2, "%s/layer%d_kal.bin", WDIR, l + 1);
                int pfd = open(p2, O_RDONLY);
                if (pfd >= 0) {
                    long off = 0;
                    while (off < (long)lsz) {
                        size_t n = lsz - (size_t)off < 524288 ? lsz - (size_t)off : 524288;
                        if (readahead(pfd, off, n) != 0) g_ra_err++;
                        off += n;
                    }
                    close(pfd);
                }
            }
        } else if (g_lw_mode == LW_MMAP_TH) {
            /* [Phase 7b] 双核预取: 主线程只发非阻塞信号, 预取线程 (core 2) 同步
             * readahead 层 l+1 -> 主线程 compute(l) 不被阻塞, SD 读与 TIU 算重叠. */
            if (l + 1 < L) prefetch_issue(l + 1);
        }
        close(lfd);
        io->src = lm;
    }
    return 0;
}

static void layer_io_end(LayerIO *io) {
    if (io->mode != LW_PREAD && io->mode != LW_ION_DB && io->src) munmap(io->src, io->lsz);
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

/* decode 用: 单个 query (位置 pos, 即 len-1) 对 KV cache 0..len-1 做 attention.
 * k/v 为扁平 [KV_CAP][DKV] cache; 输出 attn[1][D] (只写位置 0). */
static void attention_cached(int pos, const float *qbuf, const float *k, const float *v,
                             float *attn, int len) {
    for (int hh = 0; hh < H; hh++) {
        int kvh = hh / GROUPS;
        const float *qm = qbuf + (size_t)hh * HD;
        const float *kbase = k + (size_t)kvh * HD;
        const float *vbase = v + (size_t)kvh * HD;
        float lgm[KV_CAP], mx = -1e30f;
        for (int s = 0; s < len; s++) {
            const float *ks = kbase + (size_t)s * DKV;
            float vv = 0; for (int j = 0; j < HD; j++) vv += qm[j] * ks[j];
            vv *= 1.0f / sqrtf((float)HD);
            lgm[s] = vv; if (vv > mx) mx = vv;
        }
        float sum = 0;
        for (int s = 0; s < len; s++) { lgm[s] = expf(lgm[s] - mx); sum += lgm[s]; }
        float *attrow = attn + (size_t)hh * HD;
        for (int j = 0; j < HD; j++) {
            float acc = 0;
            for (int s = 0; s < len; s++) acc += lgm[s] / sum * vbase[(size_t)s * DKV + j];
            attrow[j] = acc;
        }
    }
}

/* CHAT 泛化 prefill 用: 每个 chunk token 对「已缓存 KV(0..gpos-1) + 本 chunk kbuf/vbuf」做因果 attention.
 * gpos = 本 chunk 首 token 的全局位置 (进入本 chunk 前的 g_kv_len). 输出 attn[cs][D].
 * gpos==0 时退化为满序列因果 (与现有 attention 逐位一致, 保持回归 bit-exact). */
static void attention_ext(int cs, int gpos,
                          const float *qbuf, const float *kbuf, const float *vbuf,
                          const float *kcache, const float *vcache, float *attn) {
    for (int hh = 0; hh < H; hh++) {
        int kvh = hh / GROUPS;
        for (int m = 0; m < cs; m++) {
            const float *qm = qbuf + (size_t)m * D + (size_t)hh * HD;
            int len = gpos + m + 1;   /* 该 token 的完整上下文长度 (0..gpos+m) */
            float lgm[MAX_SEQ], mx = -1e30f;
            for (int s = 0; s < len; s++) {
                const float *ks = (s < gpos) ? kcache + (size_t)kvh * HD + (size_t)s * DKV
                                             : kbuf   + (size_t)kvh * HD + (size_t)(s - gpos) * DKV;
                float v = 0; for (int j = 0; j < HD; j++) v += qm[j] * ks[j];
                v *= 1.0f / sqrtf((float)HD);
                lgm[s] = v; if (v > mx) mx = v;
            }
            float sum = 0;
            for (int s = 0; s < len; s++) { lgm[s] = expf(lgm[s] - mx); sum += lgm[s]; }
            float *attrow = attn + (size_t)m * D + (size_t)hh * HD;
            for (int j = 0; j < HD; j++) {
                float acc = 0;
                for (int s = 0; s < len; s++) {
                    const float *vs = (s < gpos) ? vcache + (size_t)kvh * HD + (size_t)s * DKV
                                                 : vbuf   + (size_t)kvh * HD + (size_t)(s - gpos) * DKV;
                    acc += lgm[s] / sum * vs[j];
                }
                attrow[j] = acc;
            }
        }
    }
}

/* ---------------- buffers ---------------- */
static float x[MS * D], h[MS * D], qbuf[MS * D], kbuf[MS * DKV], vbuf[MS * DKV], attn[MS * D];

/* ---- KV cache (decode 实测): [L][KV_CAP][DKV], malloc'd 在 main ----
 * prefill 阶段 run_prompt 逐层捕获 kbuf/vbuf 到 cache; decode 阶段逐 token 追加. */
static float *g_kvk;      /* [L*KV_CAP*DKV] */
static float *g_kvv;      /* [L*KV_CAP*DKV] */
static int g_kv_len = 0;  /* 已缓存位置数 */
static int8_t xi[MS * D], ai[MS * D];
static float scr[MS], sca[MS], sc2[MS];
static float upb[MS * F], gateb[MS * F], mid[MS * F];
static float oout[MS * D], sub[MS * D];
static float mch[MS * 1024]; static int8_t mch_i8[MS * 1024]; static float mch_sc[MS];
static float cosb[(MAX_SEQ + 8) * (HD / 2)], sinb[(MAX_SEQ + 8) * (HD / 2)];

/* ---------------- Phase 6 two-stage LM head (MDP, CPU Stage1) ----------------
 * Offline-built cluster-major files (see lmhead_cluster_build.py):
 *   centroid_f16.bin [C,D] fp16 unit centroids;  clust_idx.bin [C,2] int32 spans;
 *   embed_i8_cl.bin [V,D] int8 cluster-major rows; embed_scales_cl.f32 [V] fp32;
 *   row_to_tok_cl.bin [V] int32 reordered row -> original token id.
 * Stage1 (per token): score[c]=h·centroid[c] (fp16 upcast, double acc) -> top-Kc.
 * Stage2 (per token): read top-Kc spans, exact logits h·row·esc (double) ->
 *   running top-5 (candidate rows map 1:1 to tokens, so no dense array needed).
 */
static int g_embcl_fd = -1;    /* embed_i8_cl.bin fd (pread spans, 非 mmap) */
static float *g_esccl;     /* [V] fp32 cluster-major scales */
static int32_t *g_tokcl;   /* [V] int32 reordered row -> token */
static float *g_cent;      /* [C*D] fp32 centroids (load 时由 fp16 预转) */
static int32_t *g_clidx;   /* [C*2] int32 (offset, count) */

static inline float f16_to_f32(uint16_t h) {
    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
    uint32_t bits;
    if (e == 0) {
        if (m == 0) bits = s << 31;
        else { int k = 0; uint32_t mm = m; while (!(mm & 1024)) { mm <<= 1; k++; }
               bits = (s << 31) | ((127 - 15 - k) << 23) | ((mm & 1023) << 13); }
    } else if (e == 31) bits = (s << 31) | 0x7f800000u | (m << 13);
    else bits = (s << 31) | ((e + 112) << 23) | (m << 13);
    float out; memcpy(&out, &bits, 4); return out;
}

static int lmhead2_load(void) {
    g_cent = malloc((size_t)LMHEAD_C * D * sizeof(float));
    g_clidx = malloc((size_t)LMHEAD_C * 2 * sizeof(int32_t));
    g_esccl = malloc((size_t)V * sizeof(float));
    g_tokcl = malloc((size_t)V * sizeof(int32_t));
    if (!g_cent || !g_clidx || !g_esccl || !g_tokcl) return 1;
    FILE *f;
    /* centroid_f16.bin: 载入后一次性转 fp32 (避免每 token 917K 次 f16 转换) */
    uint16_t *ch = malloc((size_t)LMHEAD_C * D * sizeof(uint16_t));
    if (!ch) return 1;
    if (!(f = fopen(CENT_PATH, "rb"))) return 1;
    if (fread(ch, 2, (size_t)LMHEAD_C * D, f) != (size_t)LMHEAD_C * D) return 1; fclose(f);
    for (size_t i = 0; i < (size_t)LMHEAD_C * D; i++) g_cent[i] = f16_to_f32(ch[i]);
    free(ch);
    if (!(f = fopen(CLIDX_PATH, "rb"))) return 1;
    if (fread(g_clidx, 4, (size_t)LMHEAD_C * 2, f) != (size_t)LMHEAD_C * 2) return 1; fclose(f);
    if (!(f = fopen(ESC_CL_PATH, "rb"))) return 1;
    if (fread(g_esccl, 4, V, f) != V) return 1; fclose(f);
    if (!(f = fopen(TOK_CL_PATH, "rb"))) return 1;
    if (fread(g_tokcl, 4, V, f) != V) return 1; fclose(f);
    /* embed_i8_cl.bin: 保持 fd, Stage2 用 pread 读整 span (避免 mmap 每页 fault) */
    g_embcl_fd = open(EMBED_CL_PATH, O_RDONLY);
    if (g_embcl_fd < 0) return 1;
    /* 关键: 仅 mlock g_cent (stage1 每次 token 全量读 3.67MB).
     * Linux 侧仅 ~28MB (ION 占 ~36MB), 24 层运行产生内存压力会把匿名页换出,
     * g_cent 被 swap -> stage1 每次 swap-in ~500ms. mlock 常驻后 ~9ms.
     * 注意: 不要 mlock 过多 (会把 swap 压力转移到 bias/frms/esc, 拖慢 24 层),
     * 实测全量 mlock 4.67MB 使 prefill +103s; 只 mlock g_cent 3.67MB 最优. */
    size_t lk = 0;
    /* NOMLOCK=1 时跳过, 用于对照: 区分 mlock 内存压力 vs SD/热漂移 */
    if (!getenv("NOMLOCK"))
        if (mlock(g_cent, (size_t)LMHEAD_C * D * sizeof(float)) == 0) lk += (size_t)LMHEAD_C * D * sizeof(float);
    printf("  [lmhead2_load] mlock'd %.2f MB (g_cent only%s)\n", lk / 1048576.0,
           getenv("NOMLOCK") ? ", DISABLED" : "");
    return 0;
}

/* 诊断: 读 /proc/self/status VmSwap (确认 LM head 权重是否被换出) */
static long vmswap_kb(void) {
    FILE *f = fopen("/proc/self/status", "r"); if (!f) return -1;
    char line[128]; long v = -1;
    while (fgets(line, sizeof line, f)) if (!strncmp(line, "VmSwap:", 7)) { v = atol(line + 7); break; }
    fclose(f); return v;
}

/* Stage1: score[c]=h·centroid[c]; fill sel[0..Kc) with top-Kc cluster ids.
 * float32 累加: host 验证与 float64 逐位一致 (top-128 集合/rank 全同). */
static double g_t_dot = 0, g_t_sel = 0;   /* stage1 dot/selection 分解 (调试) */
static void lmhead_stage1(const float *h, int Kc, int *sel) {
    float sc[LMHEAD_C];
    double t0 = now();
    for (int c = 0; c < LMHEAD_C; c++) {
        const float *cd = g_cent + (size_t)c * D;
        float s = 0;
        for (int j = 0; j < D; j++) s += h[j] * cd[j];
        sc[c] = s;
    }
    g_t_dot = now() - t0;
    t0 = now();
    for (int i = 0; i < Kc; i++) {
        int best = -1; float bv = -1e30f;
        for (int c = 0; c < LMHEAD_C; c++) if (sc[c] > bv) { bv = sc[c]; best = c; }
        sel[i] = best; sc[best] = -1e30f;
    }
    g_t_sel = now() - t0;
}

/* Stage2: pread top-Kc spans (offset 升序, 提升 SD 顺序性), 精确 logits, running top-5.
 * 关键: 避免 mmap 每 4KB 页 fault (实测 3.1MB/s); 用 pread 整 span (~17-35MB/s). */
typedef struct { long off; long len; int base; } LmSpan;
static double g_t_sd = 0, g_t_cpu = 0;   /* stage2 SD/compute 分解 (调试) */
static void lmhead_stage2(const float *h, const int *sel, int Kc,
                          int *top, double *tv, size_t *cand_rows, double *gap) {
    for (int i = 0; i < 5; i++) { top[i] = -1; tv[i] = -1e300; }
    /* 收集候选 span (row offset/len -> byte) */
    LmSpan sp[LMHEAD_KC]; int ns = 0;
    for (int i = 0; i < Kc; i++) {
        int c = sel[i], o = g_clidx[c * 2 + 0], cnt = g_clidx[c * 2 + 1];
        if (cnt <= 0) continue;
        sp[ns].off = (long)o * D; sp[ns].len = (long)cnt * D; sp[ns].base = o; ns++;
    }
    /* offset 升序 (简单插入排序, ns<=128) */
    for (int i = 1; i < ns; i++) { LmSpan key = sp[i]; int j = i - 1;
        while (j >= 0 && sp[j].off > key.off) { sp[j + 1] = sp[j]; j--; } sp[j + 1] = key; }
    /* pread 每 span + 计算 logits (每 span 读后即算, sbuf 复用) */
    static int8_t sbuf[1048576];       /* 最大 span < 1MB (实测簇 ~133KB) */
    size_t n = 0;
    double t0, tsd = 0, tcpu = 0;
    for (int i = 0; i < ns; i++) {
        t0 = now();
        if (pread(g_embcl_fd, sbuf, sp[i].len, sp[i].off) != sp[i].len) { fprintf(stderr, "pread span\n"); exit(2); }
        tsd += now() - t0;
        t0 = now();
        int rows = (int)(sp[i].len / D);
        float esc_orow = 0;
        for (int r = 0; r < rows; r++) {
            const int8_t *row = sbuf + (size_t)r * D;
            int orow = sp[i].base + r;
            esc_orow = g_esccl[orow];       /* float32; host 验证与 f64 一致 */
            float s = 0;
            for (int j = 0; j < D; j++) s += h[j] * (float)row[j] * esc_orow;
            int t = g_tokcl[orow];
            n++;
            if (s > tv[0]) { for (int k = 4; k > 0; k--) { top[k] = top[k - 1]; tv[k] = tv[k - 1]; } top[0] = t; tv[0] = s; }
            else if (s > tv[1]) { for (int k = 4; k > 1; k--) { top[k] = top[k - 1]; tv[k] = tv[k - 1]; } top[1] = t; tv[1] = s; }
            else if (s > tv[2]) { for (int k = 4; k > 2; k--) { top[k] = top[k - 1]; tv[k] = tv[k - 1]; } top[2] = t; tv[2] = s; }
            else if (s > tv[3]) { for (int k = 4; k > 3; k--) { top[k] = top[k - 1]; tv[k] = tv[k - 1]; } top[3] = t; tv[3] = s; }
            else if (s > tv[4]) { top[4] = t; tv[4] = s; }
        }
        tcpu += now() - t0;
    }
    g_t_sd = tsd; g_t_cpu = tcpu;
    *cand_rows = n;
    *gap = tv[0] - tv[1];
}

/* ---------------- one prompt forward ---------------- */
static void run_prompt(CVI_RT_HANDLE rt, CVI_RT_MEM mem, uint64_t pa, uint8_t *va,
                       const int *toks, int seq, int pid,
                       const float *esc, const float *frms,
                       const float (*bias_all)[D + DKV + DKV],
                       PoolSet *ps, size_t lsz,
                       int *bad1, int *bad2, int *rbad,
                       double *t_layers, double *t_head, int *next_token,
                       int gpos, int do_head) {
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
        wd_kick();   /* watchdog 心跳: 每层打点, 挂死时强制退出防 ION 孤儿泄漏 */
        g_cur_layer = l; g_cur_dchunk = 0;   /* Phase 7c: rsafe 查表定位 (layer, matrix) */
        /* 层权重读路径: mmap / mmap+readahead / pread / ion_db (Phase 7, LW_READ) */
        LayerIO lio;
        if (layer_io_begin(l, lsz, &lio) != 0) { fprintf(stderr, "layer_io_begin %d\n", l); exit(2); }
        LayerRef lr;
        if (g_lw_mode == LW_ION_DB) layer_io_lr(&lr, l);          /* B-2: rms/gsc 缓存 + nib 逐矩阵 */
        else { parse_layer(lio.src, lsz, &lr); gsc_ion_apply(&lr, l); }
        const float *bias = bias_all[l];
        const float *bq = bias, *bk = bias + D, *bv = bias + D + DKV;

        /* ---- QKV ---- */
        rms_norm(x, seq, lr.rms_attn, D, h);
        per_row_quant(h, seq, D, xi, scr);
        lr.Wq_nib = mat_nib(&lr, l, 0, lr.Wq_nib);
        eng_matmul(rt, mem, pa, va, "q", xi, seq, lr.Wq_nib, lr.Wq_gsc, D, D, scr, qbuf, ps, bad1, bad2, rbad);
        lr.Wk_nib = mat_nib(&lr, l, 1, lr.Wk_nib);
        eng_matmul(rt, mem, pa, va, "k", xi, seq, lr.Wk_nib, lr.Wk_gsc, D, DKV, scr, kbuf, ps, bad1, bad2, rbad);
        lr.Wv_nib = mat_nib(&lr, l, 2, lr.Wv_nib);
        eng_matmul(rt, mem, pa, va, "v", xi, seq, lr.Wv_nib, lr.Wv_gsc, D, DKV, scr, vbuf, ps, bad1, bad2, rbad);
        for (int m = 0; m < seq; m++) {
            for (int j = 0; j < D; j++) qbuf[(size_t)m * D + j] += bq[j];
            for (int j = 0; j < DKV; j++) { kbuf[(size_t)m * DKV + j] += bk[j]; vbuf[(size_t)m * DKV + j] += bv[j]; }
        }
        for (int m = 0; m < seq; m++) {
            for (int hh = 0; hh < H; hh++) rope_inplace(qbuf + (size_t)m * D + (size_t)hh * HD, gpos + m, cosb, sinb);
            for (int hh = 0; hh < KVH; hh++) rope_inplace(kbuf + (size_t)m * DKV + (size_t)hh * HD, gpos + m, cosb, sinb);
        }
        /* ---- KV cache capture (CHAT: 写绝对位置 gpos+m) ---- */
        for (int m = 0; m < seq; m++) {
            memcpy(g_kvk + (size_t)(l * KV_CAP + gpos + m) * DKV, kbuf + (size_t)m * DKV, DKV * sizeof(float));
            memcpy(g_kvv + (size_t)(l * KV_CAP + gpos + m) * DKV, vbuf + (size_t)m * DKV, DKV * sizeof(float));
        }
        g_kv_len = gpos + seq;
        if (gpos == 0)
            attention(seq, qbuf, kbuf, vbuf, attn);
        else
            attention_ext(seq, gpos, qbuf, kbuf, vbuf,
                          g_kvk + (size_t)l * KV_CAP * DKV,
                          g_kvv + (size_t)l * KV_CAP * DKV, attn);

        /* ---- wo ---- */
        per_row_quant(attn, seq, D, ai, sca);
        lr.Wo_nib = mat_nib(&lr, l, 3, lr.Wo_nib);
        eng_matmul(rt, mem, pa, va, "wo", ai, seq, lr.Wo_nib, lr.Wo_gsc, D, D, sca, oout, ps, bad1, bad2, rbad);
        for (int m = 0; m < seq; m++) for (int j = 0; j < D; j++) x[(size_t)m * D + j] += oout[(size_t)m * D + j];

        /* ---- ffn: up / gate ---- */
        rms_norm(x, seq, lr.rms_ffn, D, h);
        per_row_quant(h, seq, D, xi, scr);
        lr.up_nib = mat_nib(&lr, l, 4, lr.up_nib);
        eng_matmul(rt, mem, pa, va, "up", xi, seq, lr.up_nib, lr.up_gsc, D, F, scr, upb, ps, bad1, bad2, rbad);
        lr.gate_nib = mat_nib(&lr, l, 5, lr.gate_nib);
        eng_matmul(rt, mem, pa, va, "gate", xi, seq, lr.gate_nib, lr.gate_gsc, D, F, scr, gateb, ps, bad1, bad2, rbad);
        for (int m = 0; m < seq; m++) for (int j = 0; j < F; j++)
            mid[(size_t)m * F + j] = upb[(size_t)m * F + j] * silu(gateb[(size_t)m * F + j]);

        /* ---- ffn: down (K-chunk 1024, per-chunk per-row quant) ---- */
        memset(oout, 0, (size_t)seq * D * sizeof(float));
        lr.down_nib = mat_nib(&lr, l, 6, lr.down_nib);
        for (int kc = 0; kc < F; kc += 1024) {
            int kcn = (F - kc < 1024) ? F - kc : 1024;
            for (int m = 0; m < seq; m++) memcpy(mch + (size_t)m * kcn, mid + (size_t)m * F + kc, (size_t)kcn * sizeof(float));
            per_row_quant(mch, seq, kcn, mch_i8, mch_sc);
            const uint8_t  *dnib = lr.down_nib + (size_t)(kc / G) * D * 16;
            const uint16_t *dgsc = lr.down_gsc + (size_t)(kc / G) * D;   /* halfword units */
            g_cur_dchunk = kc / 1024;   /* Phase 7c: down K-chunk 索引 (0..4) */
            eng_matmul(rt, mem, pa, va, "down", mch_i8, seq, dnib, dgsc, kcn, D, mch_sc, sub, ps, bad1, bad2, rbad);
            for (int m = 0; m < seq; m++) for (int j = 0; j < D; j++) oout[(size_t)m * D + j] += sub[(size_t)m * D + j];
        }
        for (int m = 0; m < seq; m++) for (int j = 0; j < D; j++) x[(size_t)m * D + j] += oout[(size_t)m * D + j];
        layer_io_end(&lio);
    }
    *t_layers = now() - t0;
    /* ion_db: LM head 前排空预读流水线 (SD 线程空闲, 避免 embed_cl pread 并发读);
     * pass 基址前移 — 下一 forward 首矩阵 q(0) 由冷槽同步补载, ~18ms/step, 可忽略. */
    if (g_lw_mode == LW_ION_DB) { ion_drain(); g_pass_base += L * NMAT; }

    /* ---- LM head: final rms -> two-stage (Stage1 h·centroid, Stage2 spans) ----
     * CHAT chunked prefill: 仅最后一个 chunk 需要 head (do_head=1); 中间 chunk 跳过. */
    if (do_head) {
        double t1 = now();
        rms_norm(x + (size_t)(seq - 1) * D, 1, frms, D, h);
        double t_rms = now() - t1;
        int top[5]; double tv[5]; size_t cand_rows; double gap;
        int sel[LMHEAD_KC];
        t1 = now();
        lmhead_stage1(h, LMHEAD_KC, sel);
        double t_s1 = now() - t1;
        t1 = now();
        lmhead_stage2(h, sel, LMHEAD_KC, top, tv, &cand_rows, &gap);
        double t_s2 = now() - t1;
        *t_head = t_rms + t_s1 + t_s2;
        *next_token = top[0];

        printf("LMHEAD2 total=%.3fs (s1=%.3fs [dot=%.3fs] s2=%.3fs [sd=%.3fs cpu=%.3fs] cand=%zu VmSwap=%ldkB)\n",
               *t_head, t_s1, g_t_dot, t_s2, g_t_sd, g_t_cpu, cand_rows, vmswap_kb());
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
    } else {
        *t_head = 0; *next_token = -1;
        printf("  CHUNK gpos=%d seq=%d done (kv_len=%d)\n", gpos, seq, g_kv_len);
    }
}

/* ---------------- decode 实测: 单 token 前向 (KV cache + M=1) ----------------
 * pos = g_kv_len (新 token 的位置). 逐层:
 *   embed -> rms -> per_row_quant -> q/k/v (M=1 TIU) + bias + rope(pos)
 *   -> 追加 KV 到 cache[pos] -> attention_cached(pos, cache, len=pos+1)
 *   -> wo + residual -> ffn(up/gate/down) -> residual
 * 最后: rms_norm(last) -> two-stage LM head -> next token.
 * 返回本步墙钟. M=1 pools 走 p128/p384/p896 (max_tile_for_m(1)=896). */
static double run_decode_step(CVI_RT_HANDLE rt, CVI_RT_MEM mem, uint64_t pa, uint8_t *va,
                              int tok, const float *esc, const float *frms,
                              const float (*bias_all)[D + DKV + DKV],
                              PoolSet *ps, MergedPool *mp, size_t lsz,
                              int *bad1, int *bad2, int *rbad, int *next_token) {
    int pos = g_kv_len;
    struct rusage ru0, ru1; long dmf = 0;   /* 步内 major fault 计数 (SD 页读诊断) */
    getrusage(RUSAGE_SELF, &ru0);
    /* ---- embed[tok]*esc[tok] (单行) ---- */
    {
        FILE *ef = fopen(EMBED_PATH, "rb");
        if (!ef) { fprintf(stderr, "cannot open %s\n", EMBED_PATH); exit(2); }
        int8_t er[D];
        if (fseek(ef, (long)tok * D, SEEK_SET)) { fprintf(stderr, "seek embed\n"); exit(2); }
        if (fread(er, 1, D, ef) != D) { fprintf(stderr, "short embed read\n"); exit(2); }
        fclose(ef);
        float es = esc[tok];
        for (int j = 0; j < D; j++) x[j] = (float)er[j] * es;
    }

    double t0 = now();
    for (int l = 0; l < L; l++) {
        wd_kick();   /* watchdog 心跳: 每层打点, 挂死时强制退出防 ION 孤儿泄漏 */
        g_cur_layer = l; g_cur_dchunk = 0;   /* Phase 7c: rsafe 查表定位 (layer, matrix) */
        /* 层权重读路径: mmap / mmap+readahead / pread / ion_db (Phase 7, LW_READ) */
        LayerIO lio;
        if (layer_io_begin(l, lsz, &lio) != 0) { fprintf(stderr, "layer_io_begin %d\n", l); exit(2); }
        LayerRef lr;
        if (g_lw_mode == LW_ION_DB) layer_io_lr(&lr, l);          /* B-2: rms/gsc 缓存 + nib 逐矩阵 */
        else { parse_layer(lio.src, lsz, &lr); gsc_ion_apply(&lr, l); }
        const float *bias = bias_all[l];
        const float *bq = bias, *bk = bias + D, *bv = bias + D + DKV;

        /* ---- QKV (M=1) ---- */
        rms_norm(x, 1, lr.rms_attn, D, h);
        per_row_quant(h, 1, D, xi, scr);
        lr.Wq_nib = mat_nib(&lr, l, 0, lr.Wq_nib);
        eng_matmul(rt, mem, pa, va, "q", xi, 1, lr.Wq_nib, lr.Wq_gsc, D, D, scr, qbuf, ps, bad1, bad2, rbad);
        lr.Wk_nib = mat_nib(&lr, l, 1, lr.Wk_nib);
        eng_matmul(rt, mem, pa, va, "k", xi, 1, lr.Wk_nib, lr.Wk_gsc, D, DKV, scr, kbuf, ps, bad1, bad2, rbad);
        lr.Wv_nib = mat_nib(&lr, l, 2, lr.Wv_nib);
        eng_matmul(rt, mem, pa, va, "v", xi, 1, lr.Wv_nib, lr.Wv_gsc, D, DKV, scr, vbuf, ps, bad1, bad2, rbad);
        for (int j = 0; j < D; j++) qbuf[j] += bq[j];
        for (int j = 0; j < DKV; j++) { kbuf[j] += bk[j]; vbuf[j] += bv[j]; }
        for (int hh = 0; hh < H; hh++) rope_inplace(qbuf + (size_t)hh * HD, pos, cosb, sinb);
        for (int hh = 0; hh < KVH; hh++) rope_inplace(kbuf + (size_t)hh * HD, pos, cosb, sinb);
        /* 追加 KV 到 cache */
        memcpy(g_kvk + (size_t)(l * KV_CAP + pos) * DKV, kbuf, DKV * sizeof(float));
        memcpy(g_kvv + (size_t)(l * KV_CAP + pos) * DKV, vbuf, DKV * sizeof(float));
        attention_cached(pos, qbuf,
                         g_kvk + (size_t)l * KV_CAP * DKV,
                         g_kvv + (size_t)l * KV_CAP * DKV,
                         attn, pos + 1);

        /* ---- wo + residual ---- */
        per_row_quant(attn, 1, D, ai, sca);
        lr.Wo_nib = mat_nib(&lr, l, 3, lr.Wo_nib);
        eng_matmul(rt, mem, pa, va, "wo", ai, 1, lr.Wo_nib, lr.Wo_gsc, D, D, sca, oout, ps, bad1, bad2, rbad);
        for (int j = 0; j < D; j++) x[j] += oout[j];

        /* ---- ffn: up / gate ---- */
        rms_norm(x, 1, lr.rms_ffn, D, h);
        per_row_quant(h, 1, D, xi, scr);
        lr.up_nib = mat_nib(&lr, l, 4, lr.up_nib);
        lr.gate_nib = mat_nib(&lr, l, 5, lr.gate_nib);
        if (g_use_merged && mp) {
            eng_matmul_merged(rt, mem, pa, va, "up", xi, 1, lr.up_nib, lr.up_gsc, D, F, scr, upb, mp, bad1, bad2, rbad);
            eng_matmul_merged(rt, mem, pa, va, "gate", xi, 1, lr.gate_nib, lr.gate_gsc, D, F, scr, gateb, mp, bad1, bad2, rbad);
        } else {
            eng_matmul(rt, mem, pa, va, "up", xi, 1, lr.up_nib, lr.up_gsc, D, F, scr, upb, ps, bad1, bad2, rbad);
            eng_matmul(rt, mem, pa, va, "gate", xi, 1, lr.gate_nib, lr.gate_gsc, D, F, scr, gateb, ps, bad1, bad2, rbad);
        }
        for (int j = 0; j < F; j++) mid[j] = upb[j] * silu(gateb[j]);

        /* ---- ffn: down (K-chunk 1024, M=1) ---- */
        memset(oout, 0, D * sizeof(float));
        lr.down_nib = mat_nib(&lr, l, 6, lr.down_nib);
        for (int kc = 0; kc < F; kc += 1024) {
            int kcn = (F - kc < 1024) ? F - kc : 1024;
            memcpy(mch, mid + kc, (size_t)kcn * sizeof(float));
            per_row_quant(mch, 1, kcn, mch_i8, mch_sc);
            const uint8_t  *dnib = lr.down_nib + (size_t)(kc / G) * D * 16;
            const uint16_t *dgsc = lr.down_gsc + (size_t)(kc / G) * D;
            g_cur_dchunk = kc / 1024;   /* Phase 7c: down K-chunk 索引 (0..4) */
            eng_matmul(rt, mem, pa, va, "down", mch_i8, 1, dnib, dgsc, kcn, D, mch_sc, sub, ps, bad1, bad2, rbad);
            for (int j = 0; j < D; j++) oout[j] += sub[j];
        }
        for (int j = 0; j < D; j++) x[j] += oout[j];
        layer_io_end(&lio);
    }
    double t_layers = now() - t0;
    /* ion_db: LM head 前排空预读流水线 (SD 线程空闲, 避免 embed_cl pread 并发读);
     * pass 基址前移 — 下一 forward 首矩阵 q(0) 由冷槽同步补载, ~18ms/step, 可忽略. */
    if (g_lw_mode == LW_ION_DB) { ion_drain(); g_pass_base += L * NMAT; }

    /* ---- LM head: final rms -> two-stage ---- */
    double t1 = now();
    rms_norm(x, 1, frms, D, h);
    double t_rms = now() - t1;
    int top[5]; double tv[5]; size_t cand_rows; double gap;
    int sel[LMHEAD_KC];
    t1 = now();
    lmhead_stage1(h, LMHEAD_KC, sel);
    double t_s1 = now() - t1;
    t1 = now();
    lmhead_stage2(h, sel, LMHEAD_KC, top, tv, &cand_rows, &gap);
    double t_s2 = now() - t1;
    *next_token = top[0];
    g_kv_len = pos + 1;
    getrusage(RUSAGE_SELF, &ru1);
    dmf = ru1.ru_majflt - ru0.ru_majflt;

    printf("DECODE pos=%d tok_in=%d next=%d gap=%.4f t_layers=%.2fs t_head=%.2fs (s1=%.3fs s2=%.3fs) cand=%zu total=%.2fs VmSwap=%ldkB majflt=%ld\n",
           pos, tok, top[0], (double)(tv[0] - tv[1]), t_layers, t_rms + t_s1 + t_s2, t_s1, t_s2,
           cand_rows, t_layers + t_rms + t_s1 + t_s2, vmswap_kb(), dmf);
    if (g_lw_mode == LW_ION_DB)
        printf("A' metrics: prefetch_n=%d err=%d sd=%.3fs memcpy=%.3fs sd_wait=%.3fs sd_idle=%.3fs(idle_empty=%.3fs noslot=%.3fs reord=%d) (t_layers=%.2fs)\n",
               g_pf_n, g_pf_err, g_t_sd_read, g_t_memcpy, g_t_sd_wait, g_t_sd_idle,
               g_t_sd_idle_empty, g_t_sd_idle_noslot, g_reorder_n, t_layers);
    return t_layers + t_rms + t_s1 + t_s2;
}

/* ---------------- PROFILE 汇总 (decode 段, M=1 compute 剖析) ---------------- */
static void profile_reset(void) {
    g_t_dequant = g_t_copyact = g_t_flush = g_t_runcmd = 0;
    g_t_invld = g_t_verify = g_t_matmul = g_t_other = 0;
    g_t_bm = g_t_acc = g_t_fin = 0;
    g_n_runcmd = 0;
    for (int i = 0; i < 7; i++) { g_t_mm[i] = 0; g_n_mm[i] = 0; }
    g_pf_n = 0; g_pf_err = 0; g_t_sd_read = 0; g_t_memcpy = 0; g_t_sd_wait = 0; g_t_sd_idle = 0;
    g_t_sd_idle_empty = 0; g_t_sd_idle_noslot = 0; g_reorder_n = 0;  /* A' 指标 */
}
static void profile_report(const char *phase) {
    double sum = g_t_dequant + g_t_copyact + g_t_flush + g_t_runcmd +
                 g_t_invld + g_t_verify + g_t_other + g_t_bm + g_t_acc + g_t_fin;
    double gap = g_t_matmul - sum;
    printf("\n==== PROFILE: eng_matmul breakdown (%s) ====\n", phase);
    printf("eng_matmul wall : %9.3fs (%ld calls)\n", g_t_matmul, g_n_mm[0] + g_n_mm[1] + g_n_mm[2] + g_n_mm[3] + g_n_mm[4] + g_n_mm[5] + g_n_mm[6]);
    printf("  dequant_rvv   : %9.3fs (%5.1f%%)  [CPU dequant INT4->INT8]\n", g_t_dequant, 100.0 * g_t_dequant / (g_t_matmul ? g_t_matmul : 1));
    printf("  copy_act      : %9.3fs (%5.1f%%)  [activation staging]\n", g_t_copyact, 100.0 * g_t_copyact / (g_t_matmul ? g_t_matmul : 1));
    printf("  flush(g2l DMA): %9.3fs (%5.1f%%)\n", g_t_flush, 100.0 * g_t_flush / (g_t_matmul ? g_t_matmul : 1));
    printf("  runcmdbuf(TIU): %9.3fs (%5.1f%%)  [%ld calls, %.3f ms/call, g2l+TIU+l2g]\n",
           g_t_runcmd, 100.0 * g_t_runcmd / (g_t_matmul ? g_t_matmul : 1),
           g_n_runcmd, g_n_runcmd ? 1000.0 * g_t_runcmd / g_n_runcmd : 0);
    printf("  invld(l2g DMA): %9.3fs (%5.1f%%)\n", g_t_invld, 100.0 * g_t_invld / (g_t_matmul ? g_t_matmul : 1));
    printf("  verify(host)  : %9.3fs (%5.1f%%)  [host int32 reference]\n", g_t_verify, 100.0 * g_t_verify / (g_t_matmul ? g_t_matmul : 1));
    printf("  other(accum/blockmax/fault): %9.3fs (%5.1f%%)\n", g_t_other, 100.0 * g_t_other / (g_t_matmul ? g_t_matmul : 1));
    printf("    |-blockmax   : %9.3fs\n", g_t_bm);
    printf("    |-accum      : %9.3fs\n", g_t_acc);
    printf("    |-final      : %9.3fs\n", g_t_fin);
    printf("    `-other-rest : %9.3fs  (wmax scan + gsc-fault + ION lat + misc)\n", g_t_other);
    printf("  sum_buckets   : %9.3fs | wall-sum gap = %+.4fs (计时开销/未计入)\n", sum, gap);
    printf("per-matmul:\n");
    for (int i = 0; i < 7; i++)
        if (g_n_mm[i])
            printf("  %-5s : %9.3fs  %7ld calls  %8.3f ms/call\n",
                   g_mm_names[i], g_t_mm[i], g_n_mm[i],
                   1000.0 * g_t_mm[i] / g_n_mm[i]);
}

/* ---------------- prompts (P0) ---------------- */
static const int P1_TOKS[MS] = {105538, 59975, 100132, 0, 0, 0, 0};
static const int P2_TOKS[MS] = {785, 6722, 315, 9625, 374, 0, 0};
static const int P3_TOKS[MS] = {100644, 104307, 101243, 3837, 97639, 85336, 102077};
static const int PSEQ[3] = {3, 5, 7};
static const int EXPECTED_NEXT[3] = {2130, 12095, 99366};

/* ---- ION 失败路径统一清理: 先 free 已分配 chunk 再退出 (不把 ION 留给进程存活期).
 * 仅用于 ION 分配失败等早退路径; 进程正常退出时内核也会关闭 fd 释放 ION, 这里显式
 * free 是双保险, 并避免 SDK 内部 "reopen ion dev 重试" 挂死期间占用. */
static void ion_cleanup_exit(CVI_RT_HANDLE rt, CVI_RT_MEM mem) {
    if (g_gsc_ion_mem) { CVI_RT_MemFree(rt, g_gsc_ion_mem); g_gsc_ion_mem = NULL; }
    for (int s = 0; s < SD_NSLOT; s++) {
        if (g_sd_ion[s]) { CVI_RT_MemFree(rt, g_sd_ion[s]); g_sd_ion[s] = NULL; }
    }
    if (mem) CVI_RT_MemFree(rt, mem);
    CVI_RT_DeInit(rt);
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);
    printf("===== M2 24-layer prefill + LM head (TIU Path A', 3-prompt regression) =====\n");
    ion_abort_install();   /* 捕获 CVI_RT_MemAlloc 的 SIGABRT -> rt_alloc_safe 回退, 不杀进程 */

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

    /* ---- Phase 6 two-stage LM head weights (centroid/clidx/esc_cl/tok_cl + mmap embed_cl) ---- */
    if (lmhead2_load()) { fprintf(stderr, "lmhead2_load failed (cluster files missing?)\n"); return 2; }
    printf("lmhead2 loaded (C=%d Kc=%d; centroid+idx ~%.2fMB resident, embed_cl mmap'd)\n",
           LMHEAD_C, LMHEAD_KC, ((size_t)LMHEAD_C * D * 2 + (size_t)LMHEAD_C * 8) / 1048576.0);
    printf("BUILD: stage1=float32-per-token-dot centroid=preconverted\n");

    /* ---- rope tables ---- */
    for (int pos = 0; pos < MAX_SEQ + 8; pos++)
        for (int j = 0; j < HD / 2; j++) {
            float freq = powf(ROPE_THETA, -(float)j / (HD / 2));
            cosb[(size_t)pos * (HD / 2) + j] = cosf(pos * freq);
            sinb[(size_t)pos * (HD / 2) + j] = sinf(pos * freq);
        }

    /* ---- ION + prebuilt pools (per M, exact tile widths) ---- */
    CVI_RT_HANDLE rt; CVI_RT_Init(&rt);
    ion_carveout_report("start");   /* 启动基线: used/free/peak 诊断 (OOM 排障) */
    CVI_RT_MEM mem = rt_alloc_safe(rt, NEURON_SZ, "neuron");
    if (!mem) {
        fprintf(stderr, "neuron ION alloc %u B FAILED; carveout 已满?\n", (unsigned)NEURON_SZ);
        ion_carveout_report("neuron-fail");
        CVI_RT_DeInit(rt); return 2;
    }
    uint64_t pa = CVI_RT_MemGetPAddr(mem);
    uint8_t *va = CVI_RT_MemGetVAddr(mem);
    CVI_RT_SetBaseReg(rt, 0, pa);
    memset(va, 0, NEURON_SZ);

    /* ---- Phase 7e: gsc cache (GSC_ION=0 关闭; 先于 pool cmdbuf 分配大块以最大化
     * ION 连续空间). B-2 (LW_READ=ion_db): gsc 22 层 ION + 2 层 DDR (nib-only SD_BUF
     * 不含 gsc, 必须缓存), rms 全层 DDR 缓存. GSC_ION=1 (mmap 基线): 24 层全 ION. */
    const char *gi = getenv("GSC_ION");
    const char *lw0 = getenv("LW_READ");
    int lw_ion_db = (lw0 && !strcmp(lw0, "ion_db"));
    /* ---- layer 文件大小 (提前, B-2 的 rms_cache_load/SD_BUF 初始化需要) ---- */
    struct stat st;
    if (stat(WDIR "/layer0_kal.bin", &st)) {
        fprintf(stderr, "stat layer0\n"); ion_cleanup_exit(rt, mem); return 2;
    }
    size_t lsz = (size_t)st.st_size;
    printf("layer file size = %zu B\n", lsz);
    if (lw_ion_db) {
        double tg = now();
        if (gsc_cache_load_b2(rt, lsz) != 0) {
            fprintf(stderr, "B-2 gsc cache load FAILED — carveout 余量不足或文件缺失\n");
            ion_cleanup_exit(rt, mem); return 2;
        }
        printf("B-2 gsc cache load done in %.3fs\n", now() - tg);
        /* B-2: SD_BUF 槽紧跟 gsc 分配 (先于 pool/mp 的碎片化小分配), 保证 2.18MB
         * 大块连续可分配. 2 槽全大槽 (2.18MB), per-matrix 双缓冲, 全 ION. */
        lsz_global = lsz;
        if (rms_cache_load(lsz) != 0) {
            fprintf(stderr, "rms cache load FAILED\n"); ion_cleanup_exit(rt, mem); return 2;
        }
        g_bounce_sz = SD_BOUNCE_SZ;
        if (posix_memalign((void **)&g_bounce, 4096, g_bounce_sz)) {
            fprintf(stderr, "oom bounce\n"); ion_cleanup_exit(rt, mem); return 2;
        }
        for (int s = 0; s < SD_NSLOT; s++) {
            size_t sz = SD_BUF_SZ;
            g_sd_ion[s] = rt_alloc_safe(rt, sz, "sd_buf");
            if (!g_sd_ion[s]) {
                fprintf(stderr, "B-2 SD_BUF%d ION alloc %zu B FAILED (carveout 余量不足?)\n", s, sz);
                ion_cleanup_exit(rt, mem); return 2;
            }
            g_sd_va[s] = CVI_RT_MemGetVAddr(g_sd_ion[s]);
        }
        if (pthread_create(&g_pf_thread, NULL, ion_prefetch_thread, NULL) != 0) {
            fprintf(stderr, "ion pf thread create failed\n"); ion_cleanup_exit(rt, mem); return 2;
        }
        printf("  [LW_ION_DB] SD_NSLOT=%d SD_BUF big=%d B x%d (ION, per-matrix 双缓冲), queue=%d, bounce=%zu B, gsc ION=%d/DDR=%d\n",
               SD_NSLOT, SD_BUF_SZ, SD_NSLOT, SD_QCAP, g_bounce_sz,
               GSC_ION_LAYERS, GSC_DDR_LAYERS);
    } else if (gi && !strcmp(gi, "0")) {
        printf("GSC_ION=0: disabled (cold mmap gsc reads)\n");
    } else {
        double tg = now();
        if (gsc_ion_load(rt) == 0) printf("GSC_ION load done in %.3fs\n", now() - tg);
    }

    /* pool/mp 分配前预检: 非 gsc 部分需 ~1.9MB (实测 pools 1.70MB + mp 0.26MB).
     * 空间不足先关 merged pool (省 ~0.26MB), 再不足则干净报错退出 (不触发 SDK assert/泄漏). */
    PoolSet ps1, ps3, ps5, ps7; memset(&ps1, 0, sizeof ps1);
    memset(&ps3, 0, sizeof ps3); memset(&ps5, 0, sizeof ps5); memset(&ps7, 0, sizeof ps7);
    {
        long long free_ion = ion_carveout_free();
        long long need = (long long)ION_POOLS_EST + (g_use_merged ? (long long)ION_MP_EST : 0);
        if (free_ion >= 0 && free_ion < need) {
            if (g_use_merged) {
                fprintf(stderr, "carveout pre-check: free=%lld B < pools+mp %lld B; disable merged pool\n",
                        free_ion, need);
                g_use_merged = 0;
                need = (long long)ION_POOLS_EST;
            }
            if (free_ion < need) {
                fprintf(stderr, "carveout pre-check FAIL: free=%lld B < pools need %lld B; "
                                "run 'run_clean.sh --clean' 清理遗留后重试\n", free_ion, need);
                CVI_RT_MemFree(rt, mem); CVI_RT_DeInit(rt);
                return 2;
            }
        }
    }
    double t0 = now();
    if (pool_build(rt, pa, 1, 128, &ps1.p128) || pool_build(rt, pa, 1, 384, &ps1.p384) ||
        pool_build(rt, pa, 1, 896, &ps1.p896) || pool_build(rt, pa, 3, 128, &ps3.p128) ||
        pool_build(rt, pa, 3, 384, &ps3.p384) || pool_build(rt, pa, 3, 896, &ps3.p896) ||
        pool_build(rt, pa, 5, 128, &ps5.p128) || pool_build(rt, pa, 5, 256, &ps5.p256) ||
        pool_build(rt, pa, 5, 768, &ps5.p768) || pool_build(rt, pa, 7, 128, &ps7.p128) ||
        pool_build(rt, pa, 7, 256, &ps7.p256) || pool_build(rt, pa, 7, 768, &ps7.p768)) {
        fprintf(stderr, "pools build FAILED (ION); freeing partial + exit\n");
        pool_set_free(rt, &ps1); pool_set_free(rt, &ps3);
        pool_set_free(rt, &ps5); pool_set_free(rt, &ps7);
        CVI_RT_MemFree(rt, mem); CVI_RT_DeInit(rt);
        return 2;
    }
    printf("pools built (M=1:{128,384,896} M=3:{128,384,896} M=5/7:{128,256,768}) in %.3fs\n", now() - t0);
    MergedPool mp; memset(&mp, 0, sizeof mp);
    if (g_use_merged) {
        double tm = now();
        if (mpool_build(rt, pa, 1, MTILEW, MNT, &mp) != 0) {
            fprintf(stderr, "merged pool build FAILED; continue without mp (MERGE=0 fallback)\n");
            g_use_merged = 0;
        } else {
            printf("merged pool built (M=1 up/gate: %d tiles x %d, F=%d) in %.3fs\n", MNT, MTILEW, MNT * MTILEW, now() - tm);
        }
    }

    /* ---- KV cache (decode 实测): [L][KV_CAP][DKV] fp32 ---- */
    g_kvk = malloc((size_t)L * KV_CAP * DKV * sizeof(float));
    g_kvv = malloc((size_t)L * KV_CAP * DKV * sizeof(float));
    if (!g_kvk || !g_kvv) { fprintf(stderr, "oom kv cache\n"); return 2; }
    memset(g_kvk, 0, (size_t)L * KV_CAP * DKV * sizeof(float));
    memset(g_kvv, 0, (size_t)L * KV_CAP * DKV * sizeof(float));
    printf("KV cache = %zu B/layer x2 (L=%d KV_CAP=%d DKV=%d)\n",
           (size_t)KV_CAP * DKV * 4, L, KV_CAP, DKV);

    /* ---- Phase 7/7b: 层权重读路径模式 (LW_READ=mmap|mmap_ra|pread|mmap_db) ---- */
    const char *lw = getenv("LW_READ");
    if (lw && !strcmp(lw, "mmap_ra")) g_lw_mode = LW_MMAP_RA;
    else if (lw && !strcmp(lw, "pread")) g_lw_mode = LW_PREAD;
    else if (lw && !strcmp(lw, "mmap_db")) g_lw_mode = LW_MMAP_DB;
    else if (lw && !strcmp(lw, "mmap_th")) g_lw_mode = LW_MMAP_TH;
    else if (lw && !strcmp(lw, "ion_db")) g_lw_mode = LW_ION_DB;
    else if (lw && !strcmp(lw, "mmap")) g_lw_mode = LW_MMAP;
    const char *vv = getenv("VERIFY");
    if (vv && !strcmp(vv, "0")) g_verify = 0;
    if (getenv("PROFILE")) g_profile = 1;
    /* ---- Phase 7d: merged up/gate cmdbuf (MERGE=0 关闭, A/B 对照) ---- */
    const char *mg = getenv("MERGE");
    if (mg && !strcmp(mg, "0")) g_use_merged = 0;
    /* ---- Phase 7c: 离线 rsafe 预标定表 (RSH=1 查表跳过 wmax 预扫) ---- */
    const char *rs = getenv("RSH");
    if (rs && !strcmp(rs, "1")) {
        FILE *rf = fopen(WDIR "/rsafe.bin", "rb");
        if (rf) {
            if (fread(g_rsh, 1, sizeof g_rsh, rf) != sizeof g_rsh) {
                fprintf(stderr, "rsafe.bin short read (%zu exp)\n", sizeof g_rsh);
                fclose(rf); return 2;
            }
            fclose(rf);
            g_rsh_loaded = 1; g_rsh_skip = 1;
            printf("RSH=1: loaded rsafe.bin (%zu B, %d layer x [6 mats + %d down chunks]) -> skip wmax pre-scan\n",
                   sizeof g_rsh, L, RSH_DCHUNKS);
        } else {
            fprintf(stderr, "RSH=1 but rsafe.bin missing; fallback to runtime wmax scan\n");
        }
    }
    printf("layer read mode = %s | VERIFY=%d PROFILE=%d RSH=%s\n",
           g_lw_mode == LW_PREAD ? "pread (buffered, ~21MB/s target, +8.4MB anon)"
         : g_lw_mode == LW_MMAP_RA ? "mmap+readahead (page-cache prefetch)"
         : g_lw_mode == LW_MMAP_DB ? "mmap_db (main-thread readahead; blocked ~0.4s/layer)"
         : g_lw_mode == LW_MMAP_TH ? "mmap_th (dual-core prefetch thread, non-blocking)"
         : g_lw_mode == LW_ION_DB ? "ion_db (B-2 per-matrix ION prefetch, nib-only 2x2.18MiB + gsc/rms cache)"
         : "mmap bare (Phase 6 baseline)", g_verify, g_profile, g_rsh_skip ? "1(skip)" : "0(scan)");

    if (g_lw_mode == LW_PREAD) {
        g_layer_buf = malloc(lsz);
        if (!g_layer_buf) { fprintf(stderr, "oom layer buffer (%zu B)\n", lsz); return 2; }
        printf("  [LW_PREAD] allocated %zu B reused layer buffer\n", lsz);
    }
    if (g_lw_mode == LW_MMAP_TH) {
        g_pf_lsz = lsz;
        if (pthread_create(&g_pf_thread, NULL, prefetch_thread_main, NULL) != 0) {
            fprintf(stderr, "prefetch thread create failed\n"); return 2;
        }
        printf("  [LW_MMAP_TH] prefetch thread started (core 2)\n");
    }
    if (g_lw_mode == LW_ION_DB) {
        /* ION debugfs 摘要 (直读, 免 fork): 验证 SD_BUF+gsc+pool 分配后 carveout 余量 */
        FILE *f = fopen("/sys/kernel/debug/ion/cvi_carveout_heap_dump/summary", "r");
        if (f) {
            char line[256];
            for (int i = 0; i < 6 && fgets(line, sizeof line, f); i++) fputs(line, stdout);
            fclose(f);
        }
    }

    int bad1 = 0, bad2 = 0, rbad = 0;
    wd_start();   /* ION 分配全部完成后启 watchdog: 防 regression/prefill/decode 挂死孤儿进程 */
    /* CHAT=1 跳过 3-prompt 回归 (省 ~60s + ION/DDR 压力), 直接进 CHAT prefill/decode. */
    if (!getenv("CHAT")) {
        const int *PT[3] = {P1_TOKS, P2_TOKS, P3_TOKS};
        int ok_all = 1;
        double tot_all = now();
        for (int p = 0; p < 3; p++) {
            double tl, th; int nxt;
            PoolSet *ps = (PSEQ[p] <= 3) ? &ps3 : (PSEQ[p] <= 5) ? &ps5 : &ps7;
            run_prompt(rt, mem, pa, va, PT[p], PSEQ[p], p, esc, frms, bias_all,
                       ps, lsz, &bad1, &bad2, &rbad, &tl, &th, &nxt, 0, 1);
            int ok = (nxt == EXPECTED_NEXT[p]);
            ok_all &= ok;
            printf("  expected_next=%d  %s\n", EXPECTED_NEXT[p], ok ? "OK" : "MISMATCH");
        }
        printf("total wall = %.2fs (all 3 prompts)\n", now() - tot_all);
        printf("==== P1/P2 bit-exact: bad1=%d bad2=%d  r_opt mismatches=%d  rsh(scan-vs-table)=%ld ====\n", bad1, bad2, rbad, g_rsh_bad);
        printf("==== TIU runs: pass1=%ld pass2=%ld total=%ld ====\n", g_runs_pass1, g_runs_pass2, g_runs_pass1 + g_runs_pass2);
        printf("==== 24L regression: expected_next 3/3 %s ====\n", ok_all ? "OK" : "FAIL");
        printf("==== 24L regression: TIU internal %s ====\n", (bad1 + bad2 + rbad == 0) ? "BIT-EXACT" : "HAS MISMATCHES");
    }

    /* ---- decode 实测 (chat 循环): DECODE=1 启用; DECODE_STEPS=N 控制步数 ---- */
    if (getenv("DECODE") && !getenv("CHAT")) {
        int ndec = 5;
        const char *nd = getenv("DECODE_STEPS");
        if (nd) ndec = atoi(nd);
        if (ndec < 1) ndec = 1;
        if (ndec > KV_CAP - 7) ndec = KV_CAP - 7;   /* prefill 3 + decode <= KV_CAP */
        printf("==== DECODE 实测: prefill P1(seq=3) + %d decode steps (M=1, KV cache) ====\n", ndec);
        g_kv_len = 0;
        g_ra_err = 0; g_ra_res = 0; g_ra_tot = 0;
        double tl, th; int nxt;
        run_prompt(rt, mem, pa, va, P1_TOKS, 3, 0, esc, frms, bias_all,
                   &ps3, lsz, &bad1, &bad2, &rbad, &tl, &th, &nxt, 0, 1);
        printf("  prefill done: kv_len=%d next=%d (expected 2130)\n", g_kv_len, nxt);
        /* B-2: decode 前释放 M=5/7 prefill pools (ION 回收 ~0.8MB, 提高 carveout 余量) */
        if (g_lw_mode == LW_ION_DB) {
            pool_set_free(rt, &ps5); pool_set_free(rt, &ps7);
            printf("  [ion_db] prefill pools ps5/ps7 freed (ION reclaimed)\n");
        }
        profile_reset();          /* prefill 之后重置: 只统计 decode 段 (M=1) */
        int tok = nxt;    /* prefill 的 next 作为首个 decode 输入 */
        double sum = 0;
        for (int i = 0; i < ndec; i++) {
            double dt = run_decode_step(rt, mem, pa, va, tok, esc, frms, bias_all,
                                        &ps1, g_use_merged ? &mp : NULL, lsz, &bad1, &bad2, &rbad, &nxt);
            tok = nxt;
            sum += dt;
        }
        printf("==== decode avg per-token = %.2fs over %d steps (prefill P1 seq=3) ====\n",
               sum / ndec, ndec);
        printf("==== decode TIU runs: pass1=%ld pass2=%ld total=%ld ====\n",
               g_runs_pass1, g_runs_pass2, g_runs_pass1 + g_runs_pass2);
        printf("==== decode bit-exact: bad1=%d bad2=%d r_opt=%d rsh=%ld ====\n", bad1, bad2, rbad, g_rsh_bad);
        if (g_lw_mode == LW_MMAP_RA || g_lw_mode == LW_MMAP_DB)
            printf("==== readahead errors: %d | mincore resident snapshot: %ld/%ld (%.1f%%) ====\n",
                   g_ra_err, g_ra_res, g_ra_tot,
                   g_ra_tot ? 100.0 * g_ra_res / g_ra_tot : 0.0);
        if (g_profile) profile_report("decode M=1");
    }

    /* ---- CHAT 问答模式 (CHAT=1): 读 /data/qwen/input_tokens.bin -> 泛化 prefill -> decode ----
     * input_tokens.bin 格式: [n_tokens int32][token ids int32...] (与 qwen_tokenize.py 输出一致).
     * 文件缺失/空 -> 报错退出. N<=7 复用 ps3/ps5/ps7; N>7 chunked prefill (每 chunk<=7, 绝对位置
     * rope/KV, attention_ext 跨 chunk 因果). decode 生成 min(DECODE_STEPS, KV_CAP-N) 步.
     * CHAT: 行 = prefill next + decode 步输出 (空格分隔 token id, host 解 token 用). */
    if (getenv("CHAT")) {
        /* 默认固定路径 /data/qwen/input_tokens.bin; CHAT_INPUT 可覆盖 (避免多 agent 并发覆盖冲突) */
        static char ip_buf[160];
        const char *ip = getenv("CHAT_INPUT");
        if (!ip) ip = WDIR "/input_tokens.bin";
        else { snprintf(ip_buf, sizeof ip_buf, "%s", ip); ip = ip_buf; }
        FILE *tf = fopen(ip, "rb");
        if (!tf) { fprintf(stderr, "CHAT: cannot open %s\n", ip); return 2; }
        int32_t hdr = 0;
        if (fread(&hdr, 4, 1, tf) != 1 || hdr <= 0) {
            fprintf(stderr, "CHAT: %s missing/empty header (need [n_tokens int32][ids...])\n", ip);
            fclose(tf); return 2;
        }
        if (hdr > MAX_SEQ) { fprintf(stderr, "CHAT: n_tokens=%d > MAX_SEQ=%d\n", hdr, MAX_SEQ); fclose(tf); return 2; }
        if (hdr >= KV_CAP)  { fprintf(stderr, "CHAT: n_tokens=%d >= KV_CAP=%d (need >=1 decode slot)\n", hdr, KV_CAP); fclose(tf); return 2; }
        int ntok = (int)hdr;
        int toks[MAX_SEQ];
        if (fread(toks, 4, ntok, tf) != (size_t)ntok) { fprintf(stderr, "CHAT: short token read\n"); fclose(tf); return 2; }
        fclose(tf);

        int ndec = 5;
        const char *nd = getenv("DECODE_STEPS");
        if (nd) ndec = atoi(nd);
        if (ndec < 1) ndec = 1;
        if (ndec > KV_CAP - ntok) ndec = KV_CAP - ntok;

        printf("==== CHAT: n_tokens=%d + %d decode steps (KV_CAP=%d, 20->48 +0.67MB DDR anon) ====\n",
               ntok, ndec, KV_CAP);
        g_kv_len = 0;
        g_ra_err = 0; g_ra_res = 0; g_ra_tot = 0;
        double tl, th; int nxt;
        int bad1c = 0, bad2c = 0, rbadc = 0;
        double t_pre = now();
        if (ntok <= 7) {
            PoolSet *ps = (ntok <= 3) ? &ps3 : (ntok <= 5) ? &ps5 : &ps7;
            run_prompt(rt, mem, pa, va, toks, ntok, 0, esc, frms, bias_all,
                       ps, lsz, &bad1c, &bad2c, &rbadc, &tl, &th, &nxt, 0, 1);
        } else {
            int off = 0;
            while (off < ntok) {
                int cs = (ntok - off > 7) ? 7 : ntok - off;
                int last = (off + cs == ntok);
                PoolSet *ps = (cs <= 3) ? &ps3 : (cs <= 5) ? &ps5 : &ps7;
                printf("  CHAT prefill chunk gpos=%d seq=%d%s\n", off, cs, last ? " (last)" : "");
                run_prompt(rt, mem, pa, va, toks + off, cs, 0, esc, frms, bias_all,
                           ps, lsz, &bad1c, &bad2c, &rbadc, &tl, &th, &nxt, off, last);
                off += cs;
            }
        }
        printf("  CHAT prefill done: kv_len=%d next=%d (%.2fs)\n", g_kv_len, nxt, now() - t_pre);

        if (g_lw_mode == LW_ION_DB) {
            pool_set_free(rt, &ps5); pool_set_free(rt, &ps7);
            printf("  [ion_db] prefill pools ps5/ps7 freed (ION reclaimed)\n");
        }
        profile_reset();          /* prefill 之后重置: 只统计 decode 段 (M=1) */
        int tok = nxt;            /* prefill next 作为首个生成 token / decode 输入 */
        int gen[MAX_SEQ]; int ngen = 0;   /* 收集生成 token, 最后统一打 CHAT 行 (避免与 DECODE 诊断交错) */
        gen[ngen++] = tok;
        double sum = 0;
        for (int i = 0; i < ndec; i++) {
            double dt = run_decode_step(rt, mem, pa, va, tok, esc, frms, bias_all,
                                        &ps1, g_use_merged ? &mp : NULL, lsz,
                                        &bad1c, &bad2c, &rbadc, &nxt);
            tok = nxt;
            sum += dt;
            gen[ngen++] = tok;
        }
        printf("CHAT:");
        for (int i = 0; i < ngen; i++) printf(" %d", gen[i]);
        printf("\n");
        printf("==== CHAT decode avg per-token = %.2fs over %d steps (prefill n=%d) ====\n",
               sum / ndec, ndec, ntok);
        printf("==== CHAT TIU runs: pass1=%ld pass2=%ld total=%ld ====\n",
               g_runs_pass1, g_runs_pass2, g_runs_pass1 + g_runs_pass2);
        printf("==== CHAT bit-exact: bad1=%d bad2=%d r_opt=%d rsh=%ld ====\n", bad1c, bad2c, rbadc, g_rsh_bad);
        if (g_lw_mode == LW_MMAP_RA || g_lw_mode == LW_MMAP_DB)
            printf("==== readahead errors: %d | mincore resident snapshot: %ld/%ld (%.1f%%) ====\n",
                   g_ra_err, g_ra_res, g_ra_tot,
                   g_ra_tot ? 100.0 * g_ra_res / g_ra_tot : 0.0);
        if (g_profile) profile_report("CHAT M=1");
    }

    if (g_lw_mode == LW_MMAP_TH) {
        pthread_mutex_lock(&g_pf_mtx);
        g_pf_shutdown = 1;
        pthread_cond_signal(&g_pf_cv);
        pthread_mutex_unlock(&g_pf_mtx);
        pthread_join(g_pf_thread, NULL);
    }
    if (g_lw_mode == LW_ION_DB) {
        pthread_mutex_lock(&g_ion_mtx);
        g_ion_shutdown = 1;
        pthread_cond_broadcast(&g_ion_req_cv);
        pthread_cond_broadcast(&g_ion_done_cv);
        pthread_mutex_unlock(&g_ion_mtx);
        pthread_join(g_pf_thread, NULL);
        for (int s = 0; s < SD_NSLOT; s++)
            if (g_sd_ion[s]) CVI_RT_MemFree(rt, g_sd_ion[s]);
        free(g_bounce);
    }
    free(esc);
    if (g_gsc_ion_mem) CVI_RT_MemFree(rt, g_gsc_ion_mem);
    for (int i = 0; i < GSC_DDR_LAYERS; i++)
        if (g_gsc_ddr_map[i]) munmap(g_gsc_ddr_map[i], lsz);
    free(g_rms_all);
    CVI_RT_MemFree(rt, mem); CVI_RT_DeInit(rt);
    return 0;
}
