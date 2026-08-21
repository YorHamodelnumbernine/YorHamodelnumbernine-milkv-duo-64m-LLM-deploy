/* smollm2_pool_demo.c — SmolLM2-135M with Unified Pool + Batch Loading.
 *
 * Design (per user spec):
 *   Phase 1 — Embedding + first 3 layers loaded into unified ION+DDR pool.
 *     Embedding (28MB) split: DDR first (~15MB), remainder in ION.
 *     First 3 layers (10.1MB) in ION after embedding portion.
 *   Phase 2 — Batch 12 layers at a time (layers 3-14, 15-26, 27-29).
 *     7 layers → ION slots, 5 layers → DDR overflow.
 *     DDR overflow memcpy→ION as earlier layers finish & free their slots.
 *   Phase 3 — LM Head: reuse embedding already in pool, transpose on-the-fly.
 *
 * Build: make smollm2_pool_demo
 * Run:   ./smollm2_pool_demo <weight_dir> <token_ids.bin> <max_new_tokens>
 */

#include "common/tpu_bench.h"
#include "common/mha_descriptor.h"
#include "common/rtos_cmdqu.h"
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <errno.h>
#include <pthread.h>

#define TICK() ({ struct timespec _ts; clock_gettime(CLOCK_MONOTONIC, &_ts); \
                  _ts.tv_sec * 1e6 + _ts.tv_nsec / 1e3; })

/* ---- Task 1 step 1.1: LM_Head per-phase timing breakdown ----
 * g_lm_brk_on is enabled only around the LM_Head block so the tile-copy /
 * MemFlush instrumentation inside tpu_matmul_build() only runs there. */
typedef struct {
    uint64_t n_chunks;
    uint64_t t_total;        /* whole LM_Head chunk loop */
    uint64_t t_sd_wait;      /* ef_wait for async SD read */
    uint64_t t_xpose_kick;   /* start async mbox transpose + flush */
    uint64_t t_load;         /* LM_LOAD_CHUNK for chunk i+2 */
    uint64_t t_matmul;       /* tpu_matmul_build() call */
    uint64_t t_tile_copy;    /* tile memcpy (576xcn) inside build */
    uint64_t t_tile_flush;   /* CVI_RT_MemFlush(1MB) inside build */
    uint64_t t_submit;       /* CVI_RT_Submit + MemInvld */
    uint64_t t_dequant;      /* dequant + scale */
    uint64_t t_xpose_poll;   /* mbox_poll_desc */
} lm_brk_t;
static lm_brk_t g_lm_brk;
static int      g_lm_brk_on = 0;

/* LM_BRK=1 enables the per-phase LM_Head timing breakdown; default OFF so the
 * production path has zero TICK() (clock_gettime) overhead.  LM_BRK_TICK()
 * returns 0 when disabled, and LM_BRK_ADD() only records when enabled. */
#define LM_BRK_TICK()   (g_lm_brk_on ? TICK() : 0)
#define LM_BRK_ADD(field, t0) \
    do { if (g_lm_brk_on) g_lm_brk.field += (uint64_t)(TICK() - (t0)); } while (0)

/* ---- ION orphan watchdog (DESIGN_ION_CLEANUP.md) ----
 * A hung-but-alive process is what leaks the 24MB ION carveout pool and
 * poisons later runs (ion ioctl fail:: Out of memory).  The watchdog thread
 * monitors a main-thread heartbeat; if the main thread stalls — CVI_RT
 * reopen-ion retry loop / TPU submit timeout / SD read stall — for
 * SM_WD_TIMEOUT (default 30) seconds, it _exit()s so the kernel closes the
 * ion/dma-buf fds and releases ION.  A dead process never leaks ION. */
static volatile double g_wd_hb = 0;
static double wd_now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}
static void wd_kick(void) { g_wd_hb = wd_now(); }

static void *wd_thread(void *arg) {
    (void)arg;
    int to = 30;
    { const char *e = getenv("SM_WD_TIMEOUT");
      if (e && atoi(e) > 0) to = atoi(e); }
    wd_kick();
    for (;;) {
        double last = g_wd_hb;
        sleep(to);
        if (g_wd_hb - last < 1e-6) {       /* no heartbeat for `to` s => hung */
            fprintf(stderr, "[wd] NO HEARTBEAT >%ds — force _exit to release ION\n", to);
            _exit(1);                      /* close fds -> kernel frees ION */
        }
    }
    return NULL;
}
static void wd_start(void) {
    pthread_t wd;
    if (pthread_create(&wd, NULL, wd_thread, NULL) != 0)
        fprintf(stderr, "[wd] watchdog thread create failed (non-fatal)\n");
}

/* ================================================================
 *  Runtime config
 * ================================================================ */
typedef struct {
    int D, n_heads, n_kv_heads, head_dim, n_layers, FFN, V, max_seq;
    int d_qkv, n_groups;
} sm_cfg_t;

static float *g_scales = NULL;
static float *g_embed_scales = NULL;   /* per-row embed scales, 49152 floats */
static float *g_layer_scales = NULL;   /* per-channel layer scales */
#define EMBED_SCALE       (g_scales ? g_scales[0] : 0.01544f)
#define W_SCALE(l, idx)   (g_scales ? g_scales[1 + (l)*7 + (idx)] : 0.001f)

static int sm_read_config(const char *base, sm_cfg_t *c) {
    char path[256];
    snprintf(path, sizeof(path), "%s/config.bin", base);
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    int cfg[8];
    if (read(fd, cfg, sizeof(cfg)) != sizeof(cfg)) { close(fd); return -1; }
    close(fd);
    c->D = cfg[0]; c->n_heads = cfg[1]; c->n_kv_heads = cfg[2];
    c->head_dim = cfg[3]; c->n_layers = cfg[4]; c->FFN = cfg[5];
    c->V = cfg[6]; c->max_seq = cfg[7];
    c->d_qkv = c->n_kv_heads * c->head_dim;
    c->n_groups = c->n_heads / c->n_kv_heads;
    return 0;
}

/* ================================================================
 *  File I/O
 * ================================================================ */
static int read_file(const char *path, void *buf, int sz) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    int n = read(fd, buf, sz);
    close(fd);
    return (n == sz) ? 0 : -1;
}

/* ================================================================
 *  CPU FP32 Operators
 * ================================================================ */
static void rms_norm_f32(float *out, const float *x, const float *gamma,
                         int tokens, int dim, float eps) {
    for (int t = 0; t < tokens; t++) {
        const float *xi = x + t * dim;
        float *oi = out + t * dim;
        float sum_sq = 0;
        for (int i = 0; i < dim; i++) { float v = xi[i]; sum_sq += v * v; }
        float inv = 1.0f / sqrtf(sum_sq / (float)dim + eps);
        for (int i = 0; i < dim; i++) oi[i] = xi[i] * inv * gamma[i];
    }
}

static void silu_f32(float *x, int n) {
    for (int i = 0; i < n; i++) { float v = x[i]; x[i] = v / (1.0f + expf(-v)); }
}

static void rope_precompute(int seq_len, int head_dim,
                            float *cos_tab, float *sin_tab) {
    int half = head_dim / 2;
    for (int i = 0; i < half; i++) {
        float freq = 1.0f / powf(100000.0f, (2.0f * i) / (float)head_dim);
        for (int pos = 0; pos < seq_len; pos++) {
            float angle = (float)pos * freq;
            cos_tab[pos * half + i] = cosf(angle);
            sin_tab[pos * half + i] = sinf(angle);
        }
    }
}

static void rope_apply_single_f32(float *q_or_k, int head_dim, int pos,
                                   const float *cos_tab, const float *sin_tab) {
    int half = head_dim / 2;
    const float *c = cos_tab + pos * half, *s = sin_tab + pos * half;
    for (int i = 0; i < half; i++) {
        float x = q_or_k[i], y = q_or_k[i + half];
        q_or_k[i] = x * c[i] - y * s[i];
        q_or_k[i + half] = x * s[i] + y * c[i];
    }
}

static void softmax_f32(float *buf, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        float *row = buf + r * cols; float maxv = row[0];
        for (int c = 1; c < cols; c++) if (row[c] > maxv) maxv = row[c];
        float sum = 0;
        for (int c = 0; c < cols; c++) { row[c] = expf(row[c] - maxv); sum += row[c]; }
        float inv = 1.0f / (sum + 1e-10f);
        for (int c = 0; c < cols; c++) row[c] *= inv;
    }
}

static int get_swap_kb(void) {
    FILE *f = fopen("/proc/self/status", "r");
    if (!f) return -1;
    char line[128]; int swap = 0;
    while (fgets(line, sizeof(line), f))
        if (sscanf(line, "VmSwap: %d kB", &swap) == 1) break;
    fclose(f);
    return swap;
}

static int sample_argmax(const float *logits, int n) {
    int best = 0; float best_v = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > best_v) { best_v = logits[i]; best = i; }
    /* Print top-5 for diagnosis */
    int top5[5] = {0,0,0,0,0};
    float top5v[5] = {-1e30f, -1e30f, -1e30f, -1e30f, -1e30f};
    for (int i = 0; i < n; i++) {
        float v = logits[i];
        int pos = 5;
        for (int k = 0; k < 5; k++) {
            if (v > top5v[k]) { pos = k; break; }
        }
        if (pos < 5) {
            for (int m = 4; m > pos; m--) { top5[m] = top5[m-1]; top5v[m] = top5v[m-1]; }
            top5[pos] = i; top5v[pos] = v;
        }
    }
    fprintf(stderr, "  [logits] #1=%d(%.1f) #2=%d(%.1f) #3=%d(%.1f) #4=%d(%.1f) #5=%d(%.1f) | gap=%.1f\n",
            top5[0], top5v[0], top5[1], top5v[1], top5[2], top5v[2],
            top5[3], top5v[3], top5[4], top5v[4], top5v[0] - top5v[1]);
    return best;
}

/* xorshift32 PRNG — fast, no libc dependency */
static uint32_t xorshift_state = 42;
static uint32_t xorshift32(void) {
    uint32_t x = xorshift_state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    xorshift_state = x;
    return x;
}
static float randf(void) {
    return (float)(xorshift32() & 0xFFFFFF) / (float)(0xFFFFFF + 1);
}

/* Temperature softmax sampling: lower T = more greedy, higher T = more random.
 * Returns a token ID sampled from the softmax distribution. */
static int sample_softmax(const float *logits, int n, float temperature, int top_k) {
    /* Find top-k indices first */
    typedef struct { int id; float v; } kv_t;
    kv_t top[50]; /* enough for top_k up to 50 */
    int n_top = (top_k > 0 && top_k < 50) ? top_k : 50;
    for (int i = 0; i < n_top; i++) { top[i].id = -1; top[i].v = -1e30f; }
    for (int i = 0; i < n; i++) {
        float v = logits[i];
        int pos = n_top;
        for (int k = 0; k < n_top; k++) { if (v > top[k].v) { pos = k; break; } }
        if (pos < n_top) {
            for (int m = n_top - 1; m > pos; m--) { top[m] = top[m-1]; }
            top[pos].id = i; top[pos].v = v;
        }
    }
    /* Compute softmax over top-k with temperature */
    double sum = 0.0, probs[50];
    for (int i = 0; i < n_top; i++) {
        if (top[i].id < 0) { probs[i] = 0; continue; }
        probs[i] = exp((double)(top[i].v / temperature));
        sum += probs[i];
    }
    if (sum < 1e-30) return top[0].id; /* fallback */
    /* Sample */
    double r = randf() * sum;
    for (int i = 0; i < n_top; i++) {
        if (top[i].id < 0) continue;
        r -= probs[i];
        if (r <= 0) return top[i].id;
    }
    return top[0].id;
}

/* ================================================================
 *  INT8 Quantization
 * ================================================================ */
static float compute_scale_sym(const float *data, int n) {
    float absmax = 0;
    for (int i = 0; i < n; i++) { float v = fabsf(data[i]); if (v > absmax) absmax = v; }
    if (absmax < 1e-10f) absmax = 1.0f;
    return absmax / 127.0f;
}

static void quantize_i8_sym(int8_t *dst, const float *src, int n, float sc) {
    float inv = 1.0f / sc;
    for (int i = 0; i < n; i++) {
        int q = (int)roundf(src[i] * inv);
        if (q > 127) q = 127; else if (q < -128) q = -128;
        dst[i] = (int8_t)q;
    }
}

static void dequantize_f32(float *dst, const int8_t *src, int n, float sc, int zp) {
    for (int i = 0; i < n; i++) dst[i] = ((float)(int)src[i] - (float)zp) * sc;
}

static void dequant_i8(float *dst, const int8_t *src, int n, float sc_wt, float sc_in, int rshift) {
    float sc = sc_wt * sc_in * (float)(1 << rshift);
    for (int i = 0; i < n; i++) dst[i] = (float)src[i] * sc;
}

/* ================================================================
 *  TPU Matmul (identical to smollm2_demo.c)
 * ================================================================ */
#define NEURON_SZ       0x100000
#define SM_SCRATCH_OFF  0x000000
#define SM_SCRATCH_SZ   0x040000
#define SM_Q_I8_OFF     0x040000
#define SM_K_I8_OFF     0x050000
#define SM_KT_I8_OFF    0x054000
#define SM_V_I8_OFF     0x058000
#define SM_S_I8_OFF     0x05C000
#define SM_A_I8_OFF     0x060000
#define SM_O_I8_OFF     0x070000
#define SM_UP_I8_OFF    0x080000
#define SM_GATE_I8_OFF  0x090000



static inline int lmem_matrix_bytes(int rows, int cols) {
    int c = (rows + 1) / 2, w = (cols + 31) / 32;
    return c * w * 32;
}
static int tpu_find_tile_n(cvk_context_t *cvk, int M, int K) {
    int left = lmem_matrix_bytes(M, K);
    for (int tn = 256; tn >= 16; tn -= 16) {
        int lm_r = lmem_matrix_bytes(K, tn);
        int lm_o = lmem_matrix_bytes(M, tn);
        if (left + lm_r + lm_o <= 32768) return tn;
    }
    return -1;
}
static inline int matmul_rshift(int K) { int r = 0, md = K*127*127; while ((md>>r)>127) r++; return r; }

static int tpu_matmul(tpu_ctx *ctx, cvk_context_t *cvk,
    const void *left, int M, int K, const void *right, int N,
    void *result, uint32_t scratch_off, int rshift)
{
    const int8_t *l_i8 = (const int8_t *)left, *r_i8 = (const int8_t *)right;
    int8_t *o_i8 = (int8_t *)result;
    uint8_t *nm = ctx->neuron_vaddr;
    uint32_t off_l = scratch_off, off_r = scratch_off + M * K;
    int lm_l=lmem_matrix_bytes(M,K), lm_r=lmem_matrix_bytes(K,N), lm_o=lmem_matrix_bytes(M,N);
    int need_tile = (lm_l+lm_r+lm_o>32768);
    if (!need_tile) {
        uint32_t off_o = scratch_off + M*K + K*N;
        memcpy(nm+off_l, l_i8, M*K); memcpy(nm+off_r, r_i8, K*N);
        CVI_RT_MemFlush(ctx->rt_handle, ctx->neuron_mem);
        cvk_ml_t *ml_l = cvk->ops->lmem_alloc_matrix(cvk, cvk->ops->ml_default_shape(cvk,M,K,CVK_FMT_I8),CVK_FMT_I8,1);
        cvk_ml_t *ml_r = cvk->ops->lmem_alloc_matrix(cvk, cvk->ops->ml_default_shape(cvk,K,N,CVK_FMT_I8),CVK_FMT_I8,1);
        cvk_ml_t *ml_o = cvk->ops->lmem_alloc_matrix(cvk, cvk->ops->ml_default_shape(cvk,M,N,CVK_FMT_I8),CVK_FMT_I8,1);
        cvk->ops->tdma_g2l_matrix_copy(cvk, &(cvk_tdma_g2l_matrix_copy_param_t){
            .src=&(cvk_mg_t){0,TPU_PA(ctx,off_l),CVK_FMT_I8,{M,K},{K}}, .dst=ml_l});
        cvk->ops->tdma_g2l_matrix_copy(cvk, &(cvk_tdma_g2l_matrix_copy_param_t){
            .src=&(cvk_mg_t){0,TPU_PA(ctx,off_r),CVK_FMT_I8,{K,N},{N}}, .dst=ml_r});
        cvk->ops->tiu_matrix_multiplication(cvk, &(cvk_tiu_matrix_multiplication_param_t){
            .res=ml_o,.left=ml_l,.right=ml_r,.bias=NULL,.lshift_bits=0,.rshift_bits=rshift,
            .res_is_int8=1,.relu_enable=0,.add_result=0,.ps32_mode=0});
        cvk->ops->tdma_l2g_matrix_copy(cvk, &(cvk_tdma_l2g_matrix_copy_param_t){
            .src=ml_o,.dst=&(cvk_mg_t){0,TPU_PA(ctx,off_o),CVK_FMT_I8,{M,N},{N}}});
        cvk->ops->lmem_free_matrix(cvk,ml_o); cvk->ops->lmem_free_matrix(cvk,ml_r); cvk->ops->lmem_free_matrix(cvk,ml_l);
        CVI_RT_Submit(ctx->rt_khandle); CVI_RT_MemInvld(ctx->rt_handle, ctx->neuron_mem);
        memcpy(o_i8, nm+off_o, M*N);
        return 0;
    }
    int tile_n = tpu_find_tile_n(cvk, M, K);
    if (tile_n < 16) return -1;
    uintptr_t nm_base = (uintptr_t)nm;
    int r_is_nm = ((uintptr_t)r_i8 >= nm_base && (uintptr_t)r_i8 < nm_base + ctx->neuron_size);
    uint32_t r_nm_off = r_is_nm ? (uint32_t)((uintptr_t)r_i8 - nm_base) : 0;
    uint32_t off_o_base = scratch_off + M*K + K*tile_n;
    memcpy(nm+off_l, l_i8, M*K); CVI_RT_MemFlush(ctx->rt_handle, ctx->neuron_mem);
    for (int ns = 0; ns < N; ns += tile_n) {
        int cn = (ns+tile_n <= N) ? tile_n : N-ns;
        if (!r_is_nm) { uint8_t *td=nm+off_r; for (int r=0;r<K;r++) memcpy(td+r*cn, r_i8+r*N+ns, cn); CVI_RT_MemFlush(ctx->rt_handle,ctx->neuron_mem); }
        cvk_ml_t *ml_l=cvk->ops->lmem_alloc_matrix(cvk,cvk->ops->ml_default_shape(cvk,M,K,CVK_FMT_I8),CVK_FMT_I8,1);
        cvk_ml_t *ml_r=cvk->ops->lmem_alloc_matrix(cvk,cvk->ops->ml_default_shape(cvk,K,cn,CVK_FMT_I8),CVK_FMT_I8,1);
        cvk_ml_t *ml_o=cvk->ops->lmem_alloc_matrix(cvk,cvk->ops->ml_default_shape(cvk,M,cn,CVK_FMT_I8),CVK_FMT_I8,1);
        cvk->ops->tdma_g2l_matrix_copy(cvk, &(cvk_tdma_g2l_matrix_copy_param_t){
            .src=&(cvk_mg_t){0,TPU_PA(ctx,off_l),CVK_FMT_I8,{M,K},{K}}, .dst=ml_l});
        cvk_mg_t sr = r_is_nm ? (cvk_mg_t){0,TPU_PA(ctx,r_nm_off+ns),CVK_FMT_I8,{K,cn},{N}}
                               : (cvk_mg_t){0,TPU_PA(ctx,off_r),CVK_FMT_I8,{K,cn},{cn}};
        cvk->ops->tdma_g2l_matrix_copy(cvk, &(cvk_tdma_g2l_matrix_copy_param_t){.src=&sr, .dst=ml_r});
        cvk->ops->tiu_matrix_multiplication(cvk, &(cvk_tiu_matrix_multiplication_param_t){
            .res=ml_o,.left=ml_l,.right=ml_r,.bias=NULL,.lshift_bits=0,.rshift_bits=rshift,
            .res_is_int8=1,.relu_enable=0,.add_result=0,.ps32_mode=0});
        cvk->ops->tdma_l2g_matrix_copy(cvk, &(cvk_tdma_l2g_matrix_copy_param_t){
            .src=ml_o,.dst=&(cvk_mg_t){0,TPU_PA(ctx,off_o_base+ns),CVK_FMT_I8,{M,cn},{N}}});
        cvk->ops->lmem_free_matrix(cvk,ml_o); cvk->ops->lmem_free_matrix(cvk,ml_r); cvk->ops->lmem_free_matrix(cvk,ml_l);
    }
    CVI_RT_Submit(ctx->rt_khandle); CVI_RT_MemInvld(ctx->rt_handle, ctx->neuron_mem);
    memcpy(o_i8, nm+off_o_base, M*N);
    return 0;
}

static int tpu_matmul_build(tpu_ctx *ctx, cvk_context_t *cvk,
    const void *left, int M, int K, const void *right, int N,
    uint32_t result_off, uint32_t scratch_off, int rshift)
{
    const int8_t *l_i8=(const int8_t*)left, *r_i8=(const int8_t*)right;
    uint8_t *nm=ctx->neuron_vaddr; uint32_t off_l=scratch_off, off_r=scratch_off+M*K;
    int need_tile=(lmem_matrix_bytes(M,K)+lmem_matrix_bytes(K,N)+lmem_matrix_bytes(M,N)>32768);
    if(!need_tile){
        memcpy(nm+off_l,l_i8,M*K); memcpy(nm+off_r,r_i8,K*N);
        CVI_RT_MemFlush(ctx->rt_handle,ctx->neuron_mem);
        cvk_ml_t *ml_l=cvk->ops->lmem_alloc_matrix(cvk,cvk->ops->ml_default_shape(cvk,M,K,CVK_FMT_I8),CVK_FMT_I8,1);
        cvk_ml_t *ml_r=cvk->ops->lmem_alloc_matrix(cvk,cvk->ops->ml_default_shape(cvk,K,N,CVK_FMT_I8),CVK_FMT_I8,1);
        cvk_ml_t *ml_o=cvk->ops->lmem_alloc_matrix(cvk,cvk->ops->ml_default_shape(cvk,M,N,CVK_FMT_I8),CVK_FMT_I8,1);
        cvk->ops->tdma_g2l_matrix_copy(cvk,&(cvk_tdma_g2l_matrix_copy_param_t){
            .src=&(cvk_mg_t){0,TPU_PA(ctx,off_l),CVK_FMT_I8,{M,K},{K}},.dst=ml_l});
        cvk->ops->tdma_g2l_matrix_copy(cvk,&(cvk_tdma_g2l_matrix_copy_param_t){
            .src=&(cvk_mg_t){0,TPU_PA(ctx,off_r),CVK_FMT_I8,{K,N},{N}},.dst=ml_r});
        cvk->ops->tiu_matrix_multiplication(cvk,&(cvk_tiu_matrix_multiplication_param_t){
            .res=ml_o,.left=ml_l,.right=ml_r,.bias=NULL,.lshift_bits=0,.rshift_bits=rshift,
            .res_is_int8=1,.relu_enable=0,.add_result=0,.ps32_mode=0});
        cvk->ops->tdma_l2g_matrix_copy(cvk,&(cvk_tdma_l2g_matrix_copy_param_t){
            .src=ml_o,.dst=&(cvk_mg_t){0,TPU_PA(ctx,result_off),CVK_FMT_I8,{M,N},{N}}});
        cvk->ops->lmem_free_matrix(cvk,ml_o); cvk->ops->lmem_free_matrix(cvk,ml_r); cvk->ops->lmem_free_matrix(cvk,ml_l);
        return 0;
    }
    int tile_n = tpu_find_tile_n(cvk, M, K);
    if (tile_n < 16) return -1;
    uintptr_t nm_base=(uintptr_t)nm;
    int r_is_nm=((uintptr_t)r_i8>=nm_base && (uintptr_t)r_i8<nm_base+ctx->neuron_size);
    uint32_t r_nm_off=r_is_nm?(uint32_t)((uintptr_t)r_i8-nm_base):0;
    memcpy(nm+off_l,l_i8,M*K);
    CVI_RT_MemFlush(ctx->rt_handle, ctx->neuron_mem);
    for(int ns=0;ns<N;ns+=tile_n){
        int cn=(ns+tile_n<=N)?tile_n:N-ns;
        if(!r_is_nm){
            uint64_t _ta = g_lm_brk_on ? TICK() : 0;
            uint8_t*td=nm+off_r; for(int r=0;r<K;r++)memcpy(td+r*cn,r_i8+r*N+ns,cn);
            uint64_t _tb = g_lm_brk_on ? TICK() : 0;
            CVI_RT_MemFlush(ctx->rt_handle,ctx->neuron_mem);
            uint64_t _tc = g_lm_brk_on ? TICK() : 0;
            if (g_lm_brk_on) { g_lm_brk.t_tile_copy += _tb - _ta; g_lm_brk.t_tile_flush += _tc - _tb; }
        }
        cvk_ml_t*ml_l=cvk->ops->lmem_alloc_matrix(cvk,cvk->ops->ml_default_shape(cvk,M,K,CVK_FMT_I8),CVK_FMT_I8,1);
        cvk_ml_t*ml_r=cvk->ops->lmem_alloc_matrix(cvk,cvk->ops->ml_default_shape(cvk,K,cn,CVK_FMT_I8),CVK_FMT_I8,1);
        cvk_ml_t*ml_o=cvk->ops->lmem_alloc_matrix(cvk,cvk->ops->ml_default_shape(cvk,M,cn,CVK_FMT_I8),CVK_FMT_I8,1);
        cvk->ops->tdma_g2l_matrix_copy(cvk,&(cvk_tdma_g2l_matrix_copy_param_t){
            .src=&(cvk_mg_t){0,TPU_PA(ctx,off_l),CVK_FMT_I8,{M,K},{K}},.dst=ml_l});
        cvk_mg_t sr=r_is_nm?(cvk_mg_t){0,TPU_PA(ctx,r_nm_off+ns),CVK_FMT_I8,{K,cn},{N}}
                            :(cvk_mg_t){0,TPU_PA(ctx,off_r),CVK_FMT_I8,{K,cn},{cn}};
        cvk->ops->tdma_g2l_matrix_copy(cvk,&(cvk_tdma_g2l_matrix_copy_param_t){.src=&sr,.dst=ml_r});
        cvk->ops->tiu_matrix_multiplication(cvk,&(cvk_tiu_matrix_multiplication_param_t){
            .res=ml_o,.left=ml_l,.right=ml_r,.bias=NULL,.lshift_bits=0,.rshift_bits=rshift,
            .res_is_int8=1,.relu_enable=0,.add_result=0,.ps32_mode=0});
        cvk->ops->tdma_l2g_matrix_copy(cvk,&(cvk_tdma_l2g_matrix_copy_param_t){
            .src=ml_o,.dst=&(cvk_mg_t){0,TPU_PA(ctx,result_off+ns),CVK_FMT_I8,{M,cn},{N}}});
        cvk->ops->lmem_free_matrix(cvk,ml_o); cvk->ops->lmem_free_matrix(cvk,ml_r); cvk->ops->lmem_free_matrix(cvk,ml_l);
    }
    return 0;
}

/* ================================================================
 *  Per-layer weight layout
 * ================================================================ */
static inline int sm_layer_bytes(int D, int dkv, int F) {
    return D*4 + D*D + D*dkv + D*dkv + D*D + D*4 + D*F + D*F + F*D;
}

typedef struct {
    uint8_t *raw; int raw_sz; int is_ion;
    int8_t *Wq, *Wk, *Wv, *Wo, *ffn_up, *ffn_gate, *ffn_down;
    float  *rms_attn, *rms_ffn;
} sm_layer_w_t;

static void sm_setup_ptrs(sm_layer_w_t *w, uint8_t *base, int D, int dkv, int F) {
    int s0=D*4, s1=D*D, s2=D*dkv, s3=D*dkv, s4=D*D;
    uint8_t *p=base;
    w->rms_attn=(float*)(p); p+=s0; w->Wq=(int8_t*)(p); p+=s1;
    w->Wk=(int8_t*)(p); p+=s2; w->Wv=(int8_t*)(p); p+=s3;
    w->Wo=(int8_t*)(p); p+=s4; w->rms_ffn=(float*)(p); p+=D*4;
    w->ffn_up=(int8_t*)(p); p+=D*F; w->ffn_gate=(int8_t*)(p); p+=D*F;
    w->ffn_down=(int8_t*)(p);
}

/* ================================================================
 *  Static work buffers for sm_layer_forward — allocated ONCE,
 *  reused across all layers/steps.  Eliminates 15-20 malloc/free
 *  calls per layer invocation (~450-600 per token generated).
 *
 *  Sizes are hardcoded for CHUNK_PREFILL=10, max_kv=64, D=576,
 *  dkv=192, F=1536, d=64, groups=3.
 * ================================================================ */
#define WORK_MAX_SEQ   10
#define WORK_MAX_KV    64
#define WORK_D         576
#define WORK_DKV       192
#define WORK_F         1536
#define WORK_D_HEAD    64
#define WORK_GROUPS    3

/* Float pool layout (~224 KB):
 *   normed(23K) + Q_f32(23K) + K_f32(7.7K) + V_f32(7.7K) +
 *   Attn_out(23K) + group_buf(16K) + up_f32(61K) + gate_f32(61K)
 *   Wo_out overlaps Q_f32 (sequential). ffn_out overlaps gate_f32. */
#define WF_NORMED      0
#define WF_Q_F32       (WF_NORMED    + WORK_MAX_SEQ * WORK_D * 4)
#define WF_K_F32       (WF_Q_F32     + WORK_MAX_SEQ * WORK_D * 4)
#define WF_V_F32       (WF_K_F32     + WORK_MAX_SEQ * WORK_DKV * 4)
#define WF_ATTN_OUT    (WF_V_F32     + WORK_MAX_SEQ * WORK_DKV * 4)
#define WF_GROUP_BUF   (WF_ATTN_OUT  + WORK_MAX_SEQ * WORK_D * 4)
#define WF_UP_F32      (WF_GROUP_BUF + WORK_MAX_KV * WORK_D_HEAD * 4)
#define WF_GATE_F32    (WF_UP_F32    + WORK_MAX_SEQ * WORK_F * 4)
/* Wo_out overlaps Q_f32 (Q_f32 unused after attention-prep loop). */
#define WF_WO_OUT      WF_Q_F32
/* ffn_out overlaps gate_f32 (gate_f32 freed after silu*gate multiply). */
#define WF_FFN_OUT     WF_GATE_F32
#define WORK_F32_SZ    (WF_GATE_F32 + WORK_MAX_SEQ * WORK_F * 4)

/* INT8 pool layout (~51 KB):
 *   x_i8(5.8K) + Q_i8(5.8K) + K_i8(1.9K) + V_i8(1.9K) +
 *   Kh_i8_tmp(4.1K) + up_i8(15K) + gate_i8(15K)
 *   O_i8 overlaps Q_i8 (sequential). down_i8 overlaps x_i8. */
#define WI_X_I8        0
#define WI_Q_I8        (WI_X_I8     + WORK_MAX_SEQ * WORK_D)
#define WI_K_I8        (WI_Q_I8     + WORK_MAX_SEQ * WORK_D)
#define WI_V_I8        (WI_K_I8     + WORK_MAX_SEQ * WORK_DKV)
#define WI_KH_I8_TMP   (WI_V_I8     + WORK_MAX_SEQ * WORK_DKV)
#define WI_UP_I8       (WI_KH_I8_TMP + WORK_MAX_KV * WORK_D_HEAD)
#define WI_GATE_I8     (WI_UP_I8    + WORK_MAX_SEQ * WORK_F)
/* O_i8 overlaps Q_i8 (Q_i8 freed after dequant). */
#define WI_O_I8        WI_Q_I8
/* down_i8 overlaps x_i8 (x_i8 freed after FFN up+gate matmuls). */
#define WI_DOWN_I8     WI_X_I8
#define WORK_I8_SZ     (WI_GATE_I8 + WORK_MAX_SEQ * WORK_F)

/* ================================================================
 *  KV Cache — stored in ION (CPU-accessible via vaddr), freeing
 *  ~3 MB Linux heap and reducing swap pressure.
 *
 *  ION layout (24MB):
 *    [Weight slots (variable)] [Embed cache] [KV cache (grows from end)]
 *
 *  KV cache at top of ION means weight slots + embed cache compete
 *  for remaining space.  As max_seq grows, weight slots naturally
 *  decrease — no arbitrary kv_len thresholds needed.
 * ================================================================ */
struct pool_t;  /* forward declaration */

typedef struct {
    int8_t *K[30], *V[30];     /* per-layer KV buffers in ION (INT8, raw QKV out) */
    float  *K_s[30], *V_s[30]; /* per-layer per-token input scale (heap, small) */
} sm_kv_cache_t;

/* Implementation below, after pool_t is fully defined */

static void sm_kv_free(sm_kv_cache_t *kv, const sm_cfg_t *c, CVI_RT_HANDLE rt) {
    /* K/V buffers are inside ION — freed with pool.  Scale arrays are heap. */
    if (!kv) return;
    (void)rt;
    int n = c ? c->n_layers : 30;
    for (int l = 0; l < n; l++) { free(kv->K_s[l]); free(kv->V_s[l]); }
    free(kv);
}
/* Store the raw INT8 QKV matmul output (pre-dequant / pre-RoPE) + the input
 * scale sc_x used to dequantize it.  On read we reconstruct the exact FP32
 * K/V the baseline derives (v = i8 * sc_x * (1<<rshift) * per-channel lsc),
 * then re-apply RoPE for K.  This makes the INT8 KV cache bit-exact vs the
 * FP32 baseline — the only precision loss is the inherent INT8 QKV matmul
 * precision, which is identical in both paths.  No extra quantization noise. */
static void kv_store_i8(int8_t *cache, float *scale, const int8_t *new_data,
                        int seq, int pos, int dkv, float sc_x) {
    memcpy(cache + (size_t)pos * dkv, new_data, (size_t)seq * dkv);
    for (int s = 0; s < seq; s++) scale[pos + s] = sc_x;
}
/* ================================================================
 *  Layer forward — uses tpu_matmul_build + batch Submit pattern
 *  (matching the proven approach from smollm2_demo.c).
 *  ION weights are passed directly to tpu_matmul_build — no DDR staging.
 * ================================================================ */
static int sm_layer_forward(tpu_ctx *ctx, cvk_context_t *cvk,
    uint8_t *nm, const sm_cfg_t *c, const sm_layer_w_t *w,
    float *x, int seq, int pos, int kv_len,
    sm_kv_cache_t *kv, int layer,
    float *rope_cos, float *rope_sin,
    double *timing,
    uint8_t *work_f32, uint8_t *work_i8)
{
    int D=c->D, H=c->n_heads, Kvh=c->n_kv_heads, d=c->head_dim;
    int dkv=c->d_qkv, F=c->FFN, groups=c->n_groups;
    int total=seq*D; double ts;
    int rc;

    /* ---- Per-channel layer scale offsets ---- */
    int ls_pl = D + dkv + dkv + D + F + F + D;
    int ls_wq=0, ls_wk=D, ls_wv=D+dkv, ls_wo=D+2*dkv;
    int ls_up=D*2+2*dkv, ls_gate=ls_up+F, ls_down=ls_gate+F;
    float *lsc = g_layer_scales ? g_layer_scales + layer * ls_pl : NULL;

    /* ---- Static buffer pointers (from pre-allocated pools) ---- */
    float *normed   = (float *)(work_f32 + WF_NORMED);
    float *Q_f32    = (float *)(work_f32 + WF_Q_F32);
    float *K_f32    = (float *)(work_f32 + WF_K_F32);
    float *V_f32    = (float *)(work_f32 + WF_V_F32);
    float *Attn_out = (float *)(work_f32 + WF_ATTN_OUT);
    float *grp_buf  = (float *)(work_f32 + WF_GROUP_BUF);  /* Qg/Kh/Scores/Vh */
    float *Wo_out   = (float *)(work_f32 + WF_WO_OUT);     /* overlaps Q_f32 */
    float *up_f32   = (float *)(work_f32 + WF_UP_F32);
    float *gate_f32 = (float *)(work_f32 + WF_GATE_F32);
    float *ffn_out  = (float *)(work_f32 + WF_FFN_OUT);    /* overlaps gate_f32 */

    int8_t *x_i8      = (int8_t *)(work_i8 + WI_X_I8);
    int8_t *Q_i8      = (int8_t *)(work_i8 + WI_Q_I8);
    int8_t *K_i8      = (int8_t *)(work_i8 + WI_K_I8);
    int8_t *V_i8      = (int8_t *)(work_i8 + WI_V_I8);
    int8_t *Kh_i8_tmp = (int8_t *)(work_i8 + WI_KH_I8_TMP);
    int8_t *O_i8      = (int8_t *)(work_i8 + WI_O_I8);     /* overlaps Q_i8 */
    int8_t *up_i8     = (int8_t *)(work_i8 + WI_UP_I8);
    int8_t *gate_i8   = (int8_t *)(work_i8 + WI_GATE_I8);
    int8_t *down_i8   = (int8_t *)(work_i8 + WI_DOWN_I8);  /* overlaps x_i8 */

    /* ---- RMS attn norm ---- */
    ts=TICK();
    rms_norm_f32(normed, x, w->rms_attn, seq, D, 1e-6f);
    timing[0]+=TICK()-ts;

    float sc_x=compute_scale_sym(normed, total);
    quantize_i8_sym(x_i8, normed, total, sc_x);

    /* ---- Q, K, V matmuls ---- */
    int rshift_qkv=matmul_rshift(D) - 5; if (rshift_qkv < 8) rshift_qkv = 8;
    ts=TICK();
    rc = tpu_matmul(ctx,cvk,x_i8,seq,D,w->Wq,D, Q_i8,SM_SCRATCH_OFF,rshift_qkv);
    if(!rc) rc = tpu_matmul(ctx,cvk,x_i8,seq,D,w->Wk,dkv, K_i8,SM_SCRATCH_OFF,rshift_qkv);
    if(!rc) rc = tpu_matmul(ctx,cvk,x_i8,seq,D,w->Wv,dkv, V_i8,SM_SCRATCH_OFF,rshift_qkv);
    if(rc){fprintf(stderr,"    L%d: QKV FAIL rc=%d\n",layer,rc); return rc;}
    timing[1]+=TICK()-ts;

    /* ---- Dequant Q, K, V (per-channel) ---- */
    if (lsc) {
        float b = sc_x * (float)(1 << rshift_qkv);
        for (int i = 0; i < seq; i++) {
            for (int j = 0; j < D; j++)
                Q_f32[i*D+j] = (float)Q_i8[i*D+j] * b * lsc[ls_wq+j];
            for (int j = 0; j < dkv; j++) {
                K_f32[i*dkv+j] = (float)K_i8[i*dkv+j] * b * lsc[ls_wk+j];
                V_f32[i*dkv+j] = (float)V_i8[i*dkv+j] * b * lsc[ls_wv+j];
            }
        }
    } else {
        dequant_i8(Q_f32,Q_i8,seq*D,W_SCALE(layer,0),sc_x,rshift_qkv);
        dequant_i8(K_f32,K_i8,seq*dkv,W_SCALE(layer,1),sc_x,rshift_qkv);
        dequant_i8(V_f32,V_i8,seq*dkv,W_SCALE(layer,2),sc_x,rshift_qkv);
    }

    /* ---- RoPE ---- */
    ts=TICK();
    for(int s=0;s<seq;s++){
        for(int h=0;h<H;h++)rope_apply_single_f32(Q_f32+s*D+h*d,d,pos+s,rope_cos,rope_sin);
        for(int h=0;h<Kvh;h++)rope_apply_single_f32(K_f32+s*dkv+h*d,d,pos+s,rope_cos,rope_sin);
    }
    timing[4]+=TICK()-ts;

    /* ---- KV cache store (raw INT8 QKV output + input scale) ---- */
    ts=TICK();
    kv_store_i8(kv->K[layer], kv->K_s[layer], K_i8, seq, pos, dkv, sc_x);
    kv_store_i8(kv->V[layer], kv->V_s[layer], V_i8, seq, pos, dkv, sc_x);
    timing[5]+=TICK()-ts;

    /* ---- KV cache now in ION — read directly, no K_full/V_full copy ---- */

    /* ---- Attention: prep Qg, Kt in neuron memory ---- */
    float softmax_scale=1.0f/sqrtf((float)d);
    memset(Attn_out, 0, seq*D*sizeof(float));
    int rshift_scores=matmul_rshift(d) - 5; if (rshift_scores < 8) rshift_scores = 8;
    int Qg_sz=seq*groups*d, Kt_sz=d*kv_len, Sg_sz=seq*groups*kv_len;
    float sc_qg[9],sc_kh[9];

    for(int g=0;g<Kvh;g++){
        float *Qg_f32 = grp_buf;
        for(int s=0;s<seq;s++)for(int h=0;h<groups;h++)memcpy(Qg_f32+(s*groups+h)*d,Q_f32+s*D+(g*groups+h)*d,d*sizeof(float));
        sc_qg[g]=compute_scale_sym(Qg_f32,Qg_sz);
        quantize_i8_sym((int8_t*)(nm+SM_Q_I8_OFF+g*Qg_sz),Qg_f32,Qg_sz,sc_qg[g]);

        float *Kh_f32 = grp_buf;  /* reuse grp_buf */
        {
            float b = (float)(1 << rshift_qkv);
            const float *lsk = lsc ? lsc + ls_wk + g*d : NULL;
            float sk_flat = lsc ? 0.0f : b * W_SCALE(layer,1);
            for(int s=0;s<kv_len;s++){
                float b_s = kv->K_s[layer][s] * b;   /* per-token input scale */
                const int8_t *krow = kv->K[layer] + (size_t)s*dkv + g*d;
                if (lsc) {
                    for(int c=0;c<d;c++)Kh_f32[s*d+c]=(float)krow[c]*b_s*lsk[c];
                } else {
                    float sk = b_s * sk_flat;
                    for(int c=0;c<d;c++)Kh_f32[s*d+c]=(float)krow[c]*sk;
                }
                rope_apply_single_f32(Kh_f32+s*d, d, s, rope_cos, rope_sin);
            }
        }
        sc_kh[g]=compute_scale_sym(Kh_f32,kv_len*d);
        quantize_i8_sym(Kh_i8_tmp,Kh_f32,kv_len*d,sc_kh[g]);
        int8_t *Kt=(int8_t*)(nm+SM_KT_I8_OFF+g*Kt_sz);
        for(int r=0;r<kv_len;r++)for(int c=0;c<d;c++)Kt[c*kv_len+r]=Kh_i8_tmp[r*d+c];
    }
    /* Q_f32, K_f32, V_f32 no longer needed after this point.
     * Wo_out overlaps Q_f32 space (via WF_WO_OUT == WF_Q_F32). */

    /* ---- Scores: batch build all groups, single Submit ---- */
    ts=TICK();
    for(int g=0;g<Kvh;g++){
        int8_t *Qg_i8=(int8_t*)(nm+SM_Q_I8_OFF+g*Qg_sz),*Kt_g=(int8_t*)(nm+SM_KT_I8_OFF+g*Kt_sz);
        rc=tpu_matmul_build(ctx,cvk,Qg_i8,seq*groups,d,Kt_g,kv_len, SM_S_I8_OFF+g*Sg_sz,SM_SCRATCH_OFF,rshift_scores);
        if(rc) return rc;
    }
    CVI_RT_Submit(ctx->rt_khandle);
    CVI_RT_MemInvld(ctx->rt_handle, ctx->neuron_mem);
    timing[6]+=TICK()-ts;

    /* ---- Softmax + quantize S, prep V in neuron memory ---- */
    float sc_sg[9],sc_vh[9];
    ts=TICK();
    for(int g=0;g<Kvh;g++){
        int8_t *Sg_i8=(int8_t*)(nm+SM_S_I8_OFF+g*Sg_sz);
        float *Scores_f32 = grp_buf;
        dequant_i8(Scores_f32,Sg_i8,seq*groups*kv_len,sc_kh[g],sc_qg[g],rshift_scores);
        for(int h=0;h<groups;h++){for(int s=0;s<seq;s++){float*row=Scores_f32+(s*groups+h)*kv_len;int mask_from=pos+s+1;for(int i=0;i<kv_len;i++){if(i>=mask_from)row[i]=-1e30f;else row[i]*=softmax_scale;}softmax_f32(row,1,kv_len);}}
        sc_sg[g]=compute_scale_sym(Scores_f32,seq*groups*kv_len);
        quantize_i8_sym(Sg_i8,Scores_f32,seq*groups*kv_len,sc_sg[g]);

        float *Vh_f32 = grp_buf;  /* reuse grp_buf */
        {
            float b = (float)(1 << rshift_qkv);
            const float *lsv = lsc ? lsc + ls_wv + g*d : NULL;
            float sv_flat = lsc ? 0.0f : b * W_SCALE(layer,2);
            for(int s=0;s<kv_len;s++){
                float b_s = kv->V_s[layer][s] * b;   /* per-token input scale */
                const int8_t *vrow = kv->V[layer] + (size_t)s*dkv + g*d;
                if (lsc) {
                    for(int c=0;c<d;c++)Vh_f32[s*d+c]=(float)vrow[c]*b_s*lsv[c];
                } else {
                    float sv = b_s * sv_flat;
                    for(int c=0;c<d;c++)Vh_f32[s*d+c]=(float)vrow[c]*sv;
                }
            }
        }
        sc_vh[g]=compute_scale_sym(Vh_f32,kv_len*d);
        quantize_i8_sym((int8_t*)(nm+SM_V_I8_OFF+g*kv_len*d),Vh_f32,kv_len*d,sc_vh[g]);
    }
    timing[7]+=TICK()-ts;

    /* ---- Attn output: batch build all groups, single Submit ---- */
    ts=TICK();
    int rshift_attn=matmul_rshift(kv_len) - 5; if (rshift_attn < 8) rshift_attn = 8;
    for(int g=0;g<Kvh;g++){
        rc=tpu_matmul_build(ctx,cvk,(int8_t*)(nm+SM_S_I8_OFF+g*Sg_sz),seq*groups,kv_len,
                            (int8_t*)(nm+SM_V_I8_OFF+g*kv_len*d),d,
                            SM_A_I8_OFF+g*seq*groups*d, SM_SCRATCH_OFF, rshift_attn);
        if(rc) return rc;
    }
    CVI_RT_Submit(ctx->rt_khandle);
    CVI_RT_MemInvld(ctx->rt_handle, ctx->neuron_mem);
    timing[8]+=TICK()-ts;

    /* ---- Accumulate attn output (float) ---- */
    for(int g=0;g<Kvh;g++){
        int8_t *Ag_i8=(int8_t*)(nm+SM_A_I8_OFF+g*seq*groups*d);
        float sc_attn=sc_sg[g]*sc_vh[g]*(float)(1<<rshift_attn);
        for(int h=0;h<groups;h++){int hq=g*groups+h;for(int s=0;s<seq;s++)for(int c=0;c<d;c++)Attn_out[s*D+hq*d+c]+=(float)Ag_i8[(s*groups+h)*d+c]*sc_attn;}
    }

    /* ---- Wo projection (Wo_out in Q_f32 space) ---- */
    ts=TICK();
    float sc_attn_q=compute_scale_sym(Attn_out,seq*D);
    quantize_i8_sym((int8_t*)(nm+SM_O_I8_OFF),Attn_out,seq*D,sc_attn_q);
    rc=tpu_matmul_build(ctx,cvk,(int8_t*)(nm+SM_O_I8_OFF),seq,D,w->Wo,D, SM_Q_I8_OFF,SM_SCRATCH_OFF,rshift_qkv);
    if(!rc){CVI_RT_Submit(ctx->rt_khandle); CVI_RT_MemInvld(ctx->rt_handle, ctx->neuron_mem); memcpy(O_i8, nm+SM_Q_I8_OFF, total);}
    timing[9]+=TICK()-ts;
    if(rc) return rc;
    if (lsc) {
        float b = sc_attn_q * (float)(1 << rshift_qkv);
        for (int i = 0; i < total; i++)
            Wo_out[i] = (float)O_i8[i] * b * lsc[ls_wo + (i % D)];
    } else {
        dequant_i8(Wo_out,O_i8,total,W_SCALE(layer,3),sc_attn_q,rshift_qkv);
    }
    for(int i=0;i<total;i++)x[i]+=Wo_out[i];

    /* ---- FFN: rms_norm + quantize (reuse normed, x_i8 buffers) ---- */
    ts=TICK(); rms_norm_f32(normed,x,w->rms_ffn,seq,D,1e-6f); timing[10]+=TICK()-ts;
    sc_x=compute_scale_sym(normed,total); quantize_i8_sym(x_i8,normed,total,sc_x);

    /* ---- FFN up + gate: batch build, single Submit ---- */
    int rshift_ffn_up=matmul_rshift(D) - 5; if (rshift_ffn_up < 8) rshift_ffn_up = 8;
    ts=TICK();
    rc = tpu_matmul_build(ctx,cvk,x_i8,seq,D,w->ffn_up,F, SM_UP_I8_OFF,SM_SCRATCH_OFF,rshift_ffn_up);
    if(!rc) rc = tpu_matmul_build(ctx,cvk,x_i8,seq,D,w->ffn_gate,F, SM_GATE_I8_OFF,SM_SCRATCH_OFF+0x10000,rshift_ffn_up);
    if(rc) return rc;
    CVI_RT_Submit(ctx->rt_khandle);
    CVI_RT_MemInvld(ctx->rt_handle, ctx->neuron_mem);
    timing[11]+=TICK()-ts;

    memcpy(up_i8, nm+SM_UP_I8_OFF, seq*F);
    memcpy(gate_i8, nm+SM_GATE_I8_OFF, seq*F);

    if (lsc) {
        float b = sc_x * (float)(1 << rshift_ffn_up);
        for (int i = 0; i < seq; i++) {
            for (int j = 0; j < F; j++) {
                up_f32[i*F+j]   = (float)up_i8[i*F+j]   * b * lsc[ls_up+j];
                gate_f32[i*F+j] = (float)gate_i8[i*F+j] * b * lsc[ls_gate+j];
            }
        }
    } else {
        dequant_i8(up_f32,up_i8,seq*F,W_SCALE(layer,4),sc_x,rshift_ffn_up);
        dequant_i8(gate_f32,gate_i8,seq*F,W_SCALE(layer,5),sc_x,rshift_ffn_up);
    }
    silu_f32(gate_f32,seq*F);
    for(int i=0;i<seq*F;i++)up_f32[i]*=gate_f32[i];

    /* ---- FFN down (ffn_out overlaps gate_f32) ---- */
    float sc_mid=compute_scale_sym(up_f32,seq*F);
    quantize_i8_sym((int8_t*)(nm+SM_UP_I8_OFF),up_f32,seq*F,sc_mid);
    int rshift_ffn_down=matmul_rshift(F) - 5; if (rshift_ffn_down < 8) rshift_ffn_down = 8;
    ts=TICK();
    rc=tpu_matmul_build(ctx,cvk,(int8_t*)(nm+SM_UP_I8_OFF),seq,F,w->ffn_down,D, SM_Q_I8_OFF,SM_SCRATCH_OFF,rshift_ffn_down);
    if(!rc){CVI_RT_Submit(ctx->rt_khandle); CVI_RT_MemInvld(ctx->rt_handle, ctx->neuron_mem); memcpy(down_i8, nm+SM_Q_I8_OFF, total);}
    timing[13]+=TICK()-ts;
    if(rc) return rc;
    if (lsc) {
        float b = sc_mid * (float)(1 << rshift_ffn_down);
        for (int i = 0; i < seq; i++)
            for (int j = 0; j < D; j++)
                ffn_out[i*D+j] = (float)down_i8[i*D+j] * b * lsc[ls_down+j];
    } else {
        dequant_i8(ffn_out,down_i8,total,W_SCALE(layer,6),sc_mid,rshift_ffn_down);
    }
    for(int i=0;i<total;i++)x[i]+=ffn_out[i];

    return 0;
}

/* ================================================================
 *  PREFETCH PIPELINE
 *
 *  Design: while TPU computes layer N, a background thread reads
 *  layer N+1 from SD into a DDR staging buffer.  This hides SD
 *  latency behind TPU compute.
 *
 *  One thread is created per layer (pthread_create + join).  The
 *  Linux preemptive scheduler overlaps the SD read (thread blocks
 *  in kernel DMA) with TPU compute on the main thread.
 *
 *  Memory: staging = 2 * layer_sz (~6.8 MB).  Fits in remaining DDR.
 * ================================================================ */
typedef struct {
    uint8_t     *buf;           /* base staging buffer for the batch   */
    int          layer_id;      /* first layer to load                 */
    int          n;             /* number of consecutive layers        */
    int          sz;            /* layer_sz bytes per layer            */
    int          ready;         /* 0=loading, 1=done, -1=error         */
    const char  *weight_dir;
    pthread_t        tid;
    pthread_mutex_t  lock;
    pthread_cond_t   cond;
} pf_job_t;

/* Change C: ONE thread reads the whole batch (n consecutive layer files)
 * SEQUENTIALLY.  Measured SD sequential throughput is ~16-19MB/s vs
 * ~8-10MB/s aggregate for N concurrent readers, so merging the reads
 * into a single thread roughly halves the wall time spent pulling a
 * batch off the SD card.  The overlap with TPU compute is preserved:
 * this thread runs on a separate pthread while the main thread computes
 * the active bank. */
static void *pf_worker(void *arg) {
    pf_job_t *j = (pf_job_t *)arg;
    int ok = 1;

    for (int i = 0; i < j->n; i++) {
        char path[256];
        snprintf(path, sizeof(path), "%s/layer%d.bin",
                 j->weight_dir, j->layer_id + i);
        int fd = open(path, O_RDONLY);
        if (fd < 0) { ok = 0; break; }
        int remain = j->sz;
        uint8_t *dst = j->buf + (size_t)i * j->sz;
        while (remain > 0) {
            int n = read(fd, dst, remain);
            if (n <= 0) { ok = 0; break; }
            dst += n; remain -= n;
        }
        close(fd);
        if (!ok) break;
    }

    pthread_mutex_lock(&j->lock);
    j->ready = ok ? 1 : -1;
    pthread_cond_signal(&j->cond);
    pthread_mutex_unlock(&j->lock);
    return NULL;
}

/* Start a thread to load a whole batch (n consecutive layers) into
 * buf[0..n-1].  Returns 0 on success, -1 if pthread_create fails
 * (e.g. OOM). */
static int pf_start(pf_job_t *j, const char *weight_dir, int first_layer,
                     int n, int sz, uint8_t *buf) {
    j->weight_dir = weight_dir;
    j->layer_id   = first_layer;
    j->n          = n;
    j->sz         = sz;
    j->buf        = buf;
    j->ready      = 0;
    return pthread_create(&j->tid, NULL, pf_worker, j);
}

/* Wait for the load thread to finish.  Returns 0 on success, -1 on error. */
static int pf_wait(pf_job_t *j) {
    pthread_mutex_lock(&j->lock);
    while (j->ready == 0)
        pthread_cond_wait(&j->cond, &j->lock);
    int r = j->ready;
    pthread_mutex_unlock(&j->lock);
    void *ret; pthread_join(j->tid, &ret);
    return (r == 1) ? 0 : -1;
}

/* ---- Embed prefetch: async SD→ION read for LM Head chunks ---- */
typedef struct {
    int       embed_fd;
    off_t     file_off;
    int       len;
    uint8_t  *dst;
    int       ready;
    pthread_t tid;
    pthread_mutex_t lock;
    pthread_cond_t  cond;
} ef_job_t;

static void *ef_worker(void *arg) {
    ef_job_t *j = (ef_job_t *)arg;
    pthread_mutex_lock(&j->lock);
    lseek(j->embed_fd, j->file_off, SEEK_SET);
    int n = read(j->embed_fd, j->dst, j->len);
    j->ready = (n == j->len) ? 1 : -1;
    pthread_cond_signal(&j->cond);
    pthread_mutex_unlock(&j->lock);
    return NULL;
}

static int ef_start(ef_job_t *j, int embed_fd, off_t off, int len, uint8_t *dst) {
    memset(j, 0, sizeof(*j));
    j->embed_fd  = embed_fd;
    j->file_off  = off;
    j->len       = len;
    j->dst       = dst;
    j->ready     = 0;
    pthread_mutex_init(&j->lock, NULL);
    pthread_cond_init(&j->cond, NULL);
    return pthread_create(&j->tid, NULL, ef_worker, j);
}

static int ef_wait(ef_job_t *j) {
    pthread_mutex_lock(&j->lock);
    while (j->ready == 0)
        pthread_cond_wait(&j->cond, &j->lock);
    int r = j->ready;
    pthread_mutex_unlock(&j->lock);
    void *ret; pthread_join(j->tid, &ret);
    pthread_mutex_destroy(&j->lock);
    pthread_cond_destroy(&j->cond);
    return (r == 1) ? 0 : -1;
}

/* ---- Mailbox ioctl helpers (Linux -> secondary core) ---- */
static int mbox_fd = -1;

/* open mailbox device */
static int mbox_open(void) {
    if (mbox_fd >= 0) return 0;
    mbox_fd = open("/dev/cvi-rtos-cmdqu", O_RDWR);
    return (mbox_fd < 0) ? -1 : 0;
}
static void mbox_close(void) {
    if (mbox_fd >= 0) { close(mbox_fd); mbox_fd = -1; }
}

/* Non-blocking send: fire-and-forget to secondary core */
static int mbox_send_async(cmdqu_t *cmdq) {
    if (mbox_fd < 0) return -1;
    return ioctl(mbox_fd, RTOS_CMDQU_SEND, cmdq);
}

/* Blocking send + wait for secondary core reply */
static int mbox_send_wait(cmdqu_t *cmdq) {
    if (mbox_fd < 0) return -1;
    return ioctl(mbox_fd, RTOS_CMDQU_SEND_WAIT, cmdq);
}

/* ---- DMA descriptor helpers ---- */
/* Get a DMA descriptor from shared neuron memory at a given slot index */
static mha_dma_desc_t *mbox_desc_ptr(uint8_t *nm, uint64_t nm_pa, int slot) {
    return (mha_dma_desc_t *)(nm + MHA_OFF_DMA_DESC + slot * sizeof(mha_dma_desc_t));
}
static uint64_t mbox_desc_pa(uint64_t nm_pa, int slot) {
    return nm_pa + MHA_OFF_DMA_DESC + slot * sizeof(mha_dma_desc_t);
}

/* Poll descriptor result field until secondary core finishes.
 * FreeRTOS writes result=0 on success, result<0 on error.
 * We initialize result=-1 (not done) before sending.
 * Must invalidate cache before each read (secondary core writes to phys memory). */
static int mbox_poll_desc(tpu_ctx *ctx, mha_dma_desc_t *d, int timeout_us) {
    int waited = 0;
    while (1) {
        CVI_RT_MemInvld(ctx->rt_handle, ctx->neuron_mem);
        if (d->result != -1) break;  /* 0=success, <0=error */
        usleep(10);
        waited += 10;
        if (timeout_us > 0 && waited >= timeout_us) {
            fprintf(stderr, "  MBOX: poll timeout after %d us (result=%d)\n",
                    waited, d->result);
            return -1;
        }
    }
    int rc = d->result;
    d->result = -1;  /* reset for next use */
    return rc;
}

/* ---- Dual-core data-mover commands (async) ---- */

/* DDR->ION memcpy on secondary core (non-blocking).
 * desc must be in shared neuron memory. */
static int mbox_ddr_to_ion_async(tpu_ctx *ctx, int slot,
                                  uint64_t src_pa, uint64_t dst_pa,
                                  unsigned int size) {
    if (mbox_fd < 0) return -1;
    uint8_t *nm = ctx->neuron_vaddr;
    uint64_t nm_pa = CVI_RT_MemGetPAddr(ctx->neuron_mem);

    mha_dma_desc_t *d = mbox_desc_ptr(nm, nm_pa, slot);
    d->src_paddr = (uint32_t)src_pa;
    d->dst_paddr = (uint32_t)dst_pa;
    d->size      = size;
    d->rows      = 0;
    d->cols      = 0;
    d->scale     = 0.0f;
    d->zero_point = 0;
    d->result    = -1;  /* -1=pending, 0=success (set by sec core) */

    CVI_RT_MemFlush(ctx->rt_handle, ctx->neuron_mem);

    cmdqu_t cmdq;
    memset(&cmdq, 0, sizeof(cmdq));
    cmdq.ip_id     = IP_SYSTEM;
    cmdq.cmd_id    = CMD_MHA_DDR_TO_ION;
    cmdq.block     = 0;
    cmdq.param_ptr = (uint32_t)mbox_desc_pa(nm_pa, slot);

    int rc = mbox_send_async(&cmdq);
    if (rc < 0) {
        fprintf(stderr, "  MBOX: DDR_TO_ION async send failed rc=%d\n", rc);
    }
    return rc;
}

/* Embedding transpose on secondary core (non-blocking).
 * desc must be in shared neuron memory. */
static int mbox_embed_xpose_async(tpu_ctx *ctx, int slot,
                                   uint64_t src_pa, uint64_t dst_pa,
                                   unsigned int D, unsigned int cur_v) {
    if (mbox_fd < 0) return -1;
    uint8_t *nm = ctx->neuron_vaddr;
    uint64_t nm_pa = CVI_RT_MemGetPAddr(ctx->neuron_mem);

    mha_dma_desc_t *d = mbox_desc_ptr(nm, nm_pa, slot);
    d->src_paddr = (uint32_t)src_pa;
    d->dst_paddr = (uint32_t)dst_pa;
    d->size      = D * cur_v;
    d->rows      = D;
    d->cols      = cur_v;
    d->scale     = 0.0f;
    d->zero_point = 0;
    d->result    = -1;  /* -1=pending, 0=success (set by sec core) */

    CVI_RT_MemFlush(ctx->rt_handle, ctx->neuron_mem);

    cmdqu_t cmdq;
    memset(&cmdq, 0, sizeof(cmdq));
    cmdq.ip_id     = IP_SYSTEM;
    cmdq.cmd_id    = CMD_MHA_EMBED_XPOSE;
    cmdq.block     = 0;
    cmdq.param_ptr = (uint32_t)mbox_desc_pa(nm_pa, slot);

    int rc = mbox_send_async(&cmdq);
    if (rc < 0) {
        fprintf(stderr, "  MBOX: EMBED_XPOSE async send failed rc=%d\n", rc);
    }
    return rc;
}

/* DDR->ION memcpy (blocking wrapper). Used when we need the copy done
 * before proceeding. */
static int mbox_ddr_to_ion_wait(tpu_ctx *ctx, int slot,
                                 uint64_t src_pa, uint64_t dst_pa,
                                 unsigned int size) {
    if (mbox_fd < 0) return -1;
    int rc = mbox_ddr_to_ion_async(ctx, slot, src_pa, dst_pa, size);
    if (rc < 0) return rc;
    uint8_t *nm = ctx->neuron_vaddr;
    uint64_t nm_pa = CVI_RT_MemGetPAddr(ctx->neuron_mem);
    return mbox_poll_desc(ctx, mbox_desc_ptr(nm, nm_pa, slot), 5000000);
}

/* Cache flush on secondary core (non-blocking) */
static int mbox_cache_flush_async(tpu_ctx *ctx, int slot,
                                   uint64_t pa, unsigned int size) {
    if (mbox_fd < 0) return -1;
    uint8_t *nm = ctx->neuron_vaddr;
    uint64_t nm_pa = CVI_RT_MemGetPAddr(ctx->neuron_mem);

    mha_dma_desc_t *d = mbox_desc_ptr(nm, nm_pa, slot);
    d->src_paddr = (uint32_t)pa;
    d->size      = size;
    d->result    = -1;  /* -1=pending, 0=success (set by sec core) */
    CVI_RT_MemFlush(ctx->rt_handle, ctx->neuron_mem);

    cmdqu_t cmdq;
    memset(&cmdq, 0, sizeof(cmdq));
    cmdq.ip_id     = IP_SYSTEM;
    cmdq.cmd_id    = CMD_MHA_CACHE_FLUSH;
    cmdq.block     = 0;
    cmdq.param_ptr = (uint32_t)mbox_desc_pa(nm_pa, slot);
    return mbox_send_async(&cmdq);
}

/* ================================================================
 *  UNIFIED POOL MANAGER
 *
 *  ION (24MB) + DDR (~15MB) = ~39MB unified pool.
 *
 *  Layout:
 *    DDR: [embedding bytes 0 .. embed_ddr_bytes-1]
 *    ION: [embedding bytes embed_ddr_bytes .. embed_total-1]
 *         [layer slot 0] [layer slot 1] ... [layer slot N-1]
 *
 *  Phase 1 (init):  Load full embedding (split DDR+ION) + first 3 layers (ION).
 *  Phase 2 (batch): Load 12 layers: fill ION slots + DDR overflow.
 *                   DDR overflow recycled into freed ION slots during compute.
 *  Phase 3 (lm_head): Reuse embedding in pool, transpose chunks on-the-fly.
 * ================================================================ */
#define ION_POOL_SZ      0x1800000   /* 24 MB */
#define DDR_POOL_TRY     0xC00000    /* 12 MB — bigger embed cache */
#define DDR_POOL_MIN     0x400000    /* fallback 4 MB */
#define EMBED_DDR_DEFAULT 0x200000   /* 2 MB — Phase 4 variant (c) default DDR embed cache */
#define ION_MAX_SLOTS    7           /* max layer slots in ION */
#define DDR_MAX_OVERFLOW 5           /* max DDR overflow layers */
#define MBOX_TIMEOUT_US  500000      /* 500ms timeout: EMBED_XPOSE 1.18MB takes ~ms; fail fast */

static uint8_t io_buf[256 * 1024];   /* staging buffer in BSS — zero heap pressure */

typedef struct pool_t {
    /* ION */
    CVI_RT_HANDLE rt_handle;
    CVI_RT_MEM  ion_mem;
    uint8_t    *ion_vaddr;
    uint64_t    ion_paddr;

    /* DDR */
    uint8_t    *ddr_base;
    int         ddr_sz;

    /* Embedding split — DDR cache + ION cache (slot 6 during decode) + SD */
    int         embed_total;      /* D * V bytes */
    int         embed_ddr_bytes;  /* bytes of embedding stored in DDR */
    int         embed_ion_bytes;  /* bytes of embedding in ION slot 6 (decode) */
    int         embed_ion_offset; /* byte offset in ION where embed cache starts */

    /* ION layer slots */
    int         ion_layer_off;    /* byte offset in ION where layer slots start (=0) */
    int         ion_n_slots;      /* actual number of layer slots in ION */
    int         ion_slot_layer[ION_MAX_SLOTS]; /* which layer is in each slot, -1=free */

    int         layer_sz, D, dkv, F;
    char        weight_dir[256];
    int         embed_fd;         /* fd for streaming embed reads (LM Head) */

    /* Secondary core DDR->ION DMA offload (optional) */
    int         use_mbox;         /* 1 if secondary core mailbox is available */

    int         ion_expanded;     /* 1 after first pool_ion_expand_for_batch */
    int         pipeline_mode;    /* 3=3+3, 2=2+2, 1=1+1 — downgrade as KV grows */

    /* KV cache in ION (top of ION, grows downward) */
    int         kv_bytes;         /* total KV cache bytes in ION */
    int         kv_start;         /* byte offset in ION where KV cache starts */
    int         ion_free;         /* bytes between 0 and kv_start for weights+embed */

    /* Static work buffers — allocated once, reused across all sm_layer_forward calls */
    uint8_t    *work_f32;         /* float working pool, ~224 KB */
    uint8_t    *work_i8;          /* int8 working pool, ~51 KB */
} pool_t;

static void pool_free(pool_t *p, CVI_RT_HANDLE rt) {
    if (p->embed_fd >= 0) { close(p->embed_fd); p->embed_fd = -1; }
    if (p->ion_mem) { CVI_RT_MemFree(rt, p->ion_mem); p->ion_mem = NULL; }
    free(p->ddr_base);
    free(p->work_f32);
    free(p->work_i8);
    mbox_close();
    memset(p, 0, sizeof(*p));
}

static int pool_init(pool_t *p, CVI_RT_HANDLE rt, const char *weight_dir,
                     int D, int dkv, int F) {
    memset(p, 0, sizeof(*p));
    p->rt_handle = rt;
    p->D = D; p->dkv = dkv; p->F = F;
    p->layer_sz = sm_layer_bytes(D, dkv, F);
    p->embed_total = D * 49152;  /* vocab=49152, weight-tied with lm_head */
    p->embed_fd = -1;
    snprintf(p->weight_dir, sizeof(p->weight_dir), "%s", weight_dir);

    /* --- ION allocation --- */
    p->ion_mem = CVI_RT_MemAlloc(rt, ION_POOL_SZ);
    if (!p->ion_mem) {
        fprintf(stderr, "  POOL: ION alloc failed — check stale holders:\n");
        system("cat /sys/kernel/debug/ion/cvi_carveout_heap_dump/summary 2>/dev/null");
        return -1;
    }
    p->ion_vaddr = (uint8_t *)CVI_RT_MemGetVAddr(p->ion_mem);
    p->ion_paddr = CVI_RT_MemGetPAddr(p->ion_mem);
    for (int i = 0; i < ION_MAX_SLOTS; i++) p->ion_slot_layer[i] = -1;

    /* --- DDR allocation: try large, fall back --- */
    int ddr_try = DDR_POOL_TRY;
    while (ddr_try >= DDR_POOL_MIN) {
        p->ddr_base = (uint8_t *)malloc(ddr_try);
        if (p->ddr_base) break;
        ddr_try -= 0x100000;  /* step down 1 MB */
    }
    if (!p->ddr_base) { fprintf(stderr, "  POOL: DDR alloc failed (tried down to %d MB)\n",
            DDR_POOL_MIN/1024/1024); pool_free(p, rt); return -1; }
    p->ddr_sz = ddr_try;

    /* --- Decide embedding split: DDR only.
     *   ION is 100% for layer weights (3+3 double-buffer pipeline).
     *   Embed rows beyond DDR are streamed from SD via embed_fd.
     *
     *   Task 1 diagnosis: the malloc'd DDR embed cache gets swapped out on
     *   this 28MB-RAM box, so LM_Head chunk loads (1.125MB memcpy) page-fault
     *   at ~80-200ms each.
     *
     *   Phase 4 fix (variant c, CEO-approved): default the DDR embed cache to
     *   a small 2MB so LM_Head chunk reads come from async SD (overlapped with
     *   the current chunk's transpose+matmul) instead of sync swapped-DDR.
     *   Measured: LM_Head 1248->834/845ms (-33%), Total -8~10%, next_token
     *   stable; 2MB also relieves the swap/memory pressure behind the device
     *   reboot + ION-orphan stability issues.
     *
     *   WARNING (never default to 0): LM_EMB_DDR_KB=0 pushes ALL embed reads
     *   to SD, slamming the page cache hard enough to OOM-kill the process;
     *   a SIGKILL leaks the ION allocation and poisons every later run until
     *   the device is rebooted.  The env override is kept ONLY for diagnosis. */
    int ddr_embed_max = (int)p->ddr_sz;
    p->embed_ddr_bytes = p->embed_total;
    if (p->embed_ddr_bytes > ddr_embed_max) p->embed_ddr_bytes = ddr_embed_max;
    if (p->embed_ddr_bytes > EMBED_DDR_DEFAULT) p->embed_ddr_bytes = EMBED_DDR_DEFAULT;
    {
        const char *kb = getenv("LM_EMB_DDR_KB");
        if (kb) {
            int cap = atoi(kb) * 1024;
            if (cap >= 0 && cap < p->embed_ddr_bytes) p->embed_ddr_bytes = cap;
            if (cap <= 0)
                fprintf(stderr, "  WARNING: LM_EMB_DDR_KB=%s forces ALL embed from SD — "
                        "OOM-kill + ION-leak poison risk; diagnosis only.\n", kb);
        }
    }

    /* --- ION layer slots: entire ION for layer weights --- */
    p->ion_layer_off = 0;
    p->kv_bytes = 0;
    p->kv_start = ION_POOL_SZ;
    p->ion_free = ION_POOL_SZ;  /* full ION available until KV is allocated */
    p->ion_n_slots = p->ion_free / p->layer_sz;
    if (p->ion_n_slots > ION_MAX_SLOTS) p->ion_n_slots = ION_MAX_SLOTS;

    /* --- Try to open mailbox to secondary core --- */
    p->use_mbox = (mbox_open() == 0);
    if (p->use_mbox) {
        fprintf(stderr, "[pool] mbox OK (secondary core available)\n");
    } else {
        fprintf(stderr, "[pool] mbox not available — CPU-only fallback\n");
    }

    /* DDR holds embed cache; ION stages weights directly (no DDR staging needed) */
    if (p->ddr_sz < DDR_POOL_MIN) {
        fprintf(stderr, "  POOL: DDR too small (%d MB < min %d MB)\n",
                p->ddr_sz/1024/1024, DDR_POOL_MIN/1024/1024);
        pool_free(p, rt); return -1;
    }

    /* Open embed fd early so inode stays cached */
    {
        char epath[256];
        snprintf(epath, sizeof(epath), "%s/embed.i8", weight_dir);
        p->embed_fd = open(epath, O_RDONLY);
    }
    /* fd may be -1 if file missing; pool_get_embed_row will fallback */

    fprintf(stderr, "[pool] ION pa=0x%llx va=%p sz=%d MB\n",
            (unsigned long long)p->ion_paddr, p->ion_vaddr, ION_POOL_SZ/1024/1024);
    fprintf(stderr, "[pool] DDR va=%p sz=%d MB (embed cache)\n",
            p->ddr_base, p->ddr_sz/1024/1024);
    fprintf(stderr, "[pool] Embed: %d KB total, %d KB DDR + SD\n",
            p->embed_total/1024, p->embed_ddr_bytes/1024);
    fprintf(stderr, "[pool] ION slots: %d (pipeline mode set after KV alloc)\n",
            p->ion_n_slots);

    /* Allocate static work buffers for sm_layer_forward (~275 KB total).
     * These are reused across all 30 layers and all steps, eliminating
     * 15-20 malloc/free calls per layer invocation. */
    p->work_f32 = (uint8_t *)malloc(WORK_F32_SZ);
    p->work_i8  = (uint8_t *)malloc(WORK_I8_SZ);
    if (!p->work_f32 || !p->work_i8) {
        fprintf(stderr, "  POOL: work buffer alloc failed\n");
        free(p->work_f32); free(p->work_i8);
        pool_free(p, rt); return -1;
    }
    fprintf(stderr, "[pool] work buffers: f32=%d KB, i8=%d KB\n",
            WORK_F32_SZ/1024, WORK_I8_SZ/1024);

    return 0;
}

/* ---- KV cache allocation in ION (implementation, after pool_t defined) ---- */
static sm_kv_cache_t *sm_kv_alloc_ion(struct pool_t *pool, int max_seq, int dkv, int n_layers, CVI_RT_HANDLE rt) {
    sm_kv_cache_t *kv = (sm_kv_cache_t*)calloc(1, sizeof(sm_kv_cache_t));
    if (!kv) return NULL;
    (void)rt;

    int per_layer = max_seq * dkv * 1;   /* INT8 KV */
    int per_layer_aligned = (per_layer + 255) & ~255;
    int kv_total = n_layers * 2 * per_layer_aligned;
    int kv_start = ION_POOL_SZ - kv_total;

    if (kv_start < 0) {
        fprintf(stderr, "  KV: too large for ION (%d KB > %d MB)\n",
                kv_total/1024, ION_POOL_SZ/1024/1024);
        free(kv); return NULL;
    }

    uint8_t *kv_base = pool->ion_vaddr + kv_start;
    memset(kv_base, 0, kv_total);
    for (int l = 0; l < n_layers; l++) {
        kv->K[l] = (int8_t *)(kv_base + l * 2 * per_layer_aligned);
        kv->V[l] = (int8_t *)(kv_base + (l * 2 + 1) * per_layer_aligned);
        /* per-token input scale arrays on heap (do not consume ION budget) */
        kv->K_s[l] = (float *)calloc(max_seq, sizeof(float));
        kv->V_s[l] = (float *)calloc(max_seq, sizeof(float));
        if (!kv->K_s[l] || !kv->V_s[l]) {
            fprintf(stderr, "  KV: scale alloc failed\n");
            sm_kv_free(kv, NULL, rt);
            return NULL;
        }
    }

    pool->kv_bytes  = kv_total;
    pool->kv_start  = kv_start;
    pool->ion_free  = kv_start;

    fprintf(stderr, "[kv] ION: %d KB at off=%d (max_seq=%d), ION free=%d KB\n",
            kv_total/1024, kv_start, max_seq, pool->ion_free/1024);
    return kv;
}

/* ---- Load embedding + first 3 layers into pool ---- */
static int pool_load_embed_and_init_layers(pool_t *p) {
    char path[256];
    double ts = TICK();
    wd_kick();   /* watchdog: init-phase milestone */

    /* Load embedding: split across DDR + ION.
     * Close/reopen between DDR and ION portions to release kernel page cache,
     * avoiding OOM on the 28 MB embed file. */
    uint8_t *buf = io_buf;

    /* DDR portion: bytes [0, embed_ddr_bytes) */
    if (p->embed_ddr_bytes > 0) {
        snprintf(path, sizeof(path), "%s/embed.i8", p->weight_dir);
        int fd = open(path, O_RDONLY);
        if (fd < 0) { fprintf(stderr, "  POOL: cannot open embed.i8\n"); return -1; }
        fprintf(stderr, "  Loading embed DDR: %d KB...\n", p->embed_ddr_bytes/1024);
        int remain = p->embed_ddr_bytes;
        uint8_t *dst = p->ddr_base;
        while (remain > 0) {
            int n = (remain < 262144) ? remain : 262144;
            if (read(fd, buf, n) != n) {
                fprintf(stderr, "  POOL: embed DDR read fail\n");
                close(fd); return -1;
            }
            memcpy(dst, buf, n);
            dst += n; remain -= n;
        }
        close(fd);
        fprintf(stderr, "  Embed DDR done.\n");
    }
    fprintf(stderr, "  Embed loaded: %.0f ms (%d KB DDR)\n",
            (TICK()-ts)/1000.0, p->embed_ddr_bytes/1024);
    wd_kick();   /* watchdog: embed load milestone */

    /* Load first 3 layers into ION slots — reuse static staging buffer */
    ts = TICK();
    int n_init = 3;
    if (n_init > p->ion_n_slots) n_init = p->ion_n_slots;
    for (int i = 0; i < n_init; i++) {
        snprintf(path, sizeof(path), "%s/layer%d.bin", p->weight_dir, i);
        int fd = open(path, O_RDONLY);
        if (fd < 0) { fprintf(stderr, "  POOL: open layer%d fail\n", i); return -1; }
        int remain = p->layer_sz;
        uint8_t *dst = p->ion_vaddr + i * p->layer_sz;
        while (remain > 0) {
            int n = (remain < 262144) ? remain : 262144;
            if (read(fd, buf, n) != n) {
                fprintf(stderr, "  POOL: layer%d read fail\n", i);
                close(fd); return -1;
            }
            memcpy(dst, buf, n);
            dst += n; remain -= n;
        }
        close(fd);
        p->ion_slot_layer[i] = i;
    }
    fprintf(stderr, "  Init layers 0-%d: %.0f ms\n",
            n_init-1, (TICK()-ts)/1000.0);
    wd_kick();   /* watchdog: init-layers milestone */
    return 0;
}

/* ---- Called after embedding lookup + init layers done.
 *   ION+DDR repurposed for layer weight slots + staging buffers.
 *   Preserves ION embed in place to accelerate LM Head chunk reads. ---- */
static void pool_ion_expand_for_batch(pool_t *p) {
    if (p->ion_expanded) {
        fprintf(stderr, "  POOL reuse: %d ION weight slots, %d+%d pipe, embed hit=%.0f%%\n",
                p->ion_n_slots, p->pipeline_mode, p->pipeline_mode,
                100.0*(p->embed_ddr_bytes+p->embed_ion_bytes)/p->embed_total);
        return;
    }

    /* Weight slots + embed cache fit in ion_free (space below KV cache). */
    int weight_slots = p->pipeline_mode * 2;
    p->ion_layer_off = 0;
    p->ion_n_slots = weight_slots;
    for (int i = 0; i < p->ion_n_slots; i++) p->ion_slot_layer[i] = -1;

    /* First-time setup: keep DDR embed + load ION embed cache for rows beyond DDR. */
    int cb = p->D * 1024;
    p->embed_ddr_bytes = (p->embed_ddr_bytes / cb) * cb;  /* keep DDR, don't release */

    p->embed_ion_offset = weight_slots * p->layer_sz;
    int embed_space = p->ion_free - p->embed_ion_offset;
    p->embed_ion_bytes = p->embed_total - p->embed_ddr_bytes;
    if (p->embed_ion_bytes > embed_space) p->embed_ion_bytes = embed_space;
    p->embed_ion_bytes = (p->embed_ion_bytes / cb) * cb;
    /* DDR stays alive — embed_ddr_bytes preserved, not zeroed */

    if (p->embed_ion_bytes > 0 && p->embed_fd >= 0) {
        uint8_t *dst = p->ion_vaddr + p->embed_ion_offset;
        int remain = p->embed_ion_bytes;
        lseek(p->embed_fd, p->embed_ddr_bytes, SEEK_SET);  /* start where DDR ends */
        while (remain > 0) {
            int n = (remain < 262144) ? remain : 262144;
            if (read(p->embed_fd, io_buf, n) != n) break;
            memcpy(dst, io_buf, n);
            dst += n; remain -= n;
        }
        CVI_RT_MemFlush(p->rt_handle, p->ion_mem);
    }
    p->ion_expanded = 1;

    fprintf(stderr, "  POOL expand: %d+%d pipe, %dW slots + embed DDR=%dKB ION=%dKB (hit=%.0f%%, KV=%dKB in ION)\n",
            p->pipeline_mode, p->pipeline_mode, p->ion_n_slots,
            p->embed_ddr_bytes/1024, p->embed_ion_bytes/1024,
            100.0*(p->embed_ddr_bytes+p->embed_ion_bytes)/p->embed_total,
            p->kv_bytes/1024);
}

/* ---- Dynamic pipeline downgrade as KV cache grows in ION.
 *   KV cache sits at top of ION.  Weight slots + embed cache share
 *   the remaining space below.  As max_seq (and thus KV) grows,
 *   fewer weight slots fit → pipeline mode drops naturally.
 *
 *   This is set ONCE at init based on ION free space after KV alloc,
 *   NOT dynamically during decode (KV space is pre-allocated).
 *
 *   Mode 3 (3+3): need 6 weight slots (20.3MB) + min 2MB embed → ion_free >= 22.3MB
 *   Mode 2 (2+2): need 4 weight slots (13.5MB) + min 2MB embed → ion_free >= 15.5MB
 *   Mode 1 (1+1): need 2 weight slots (6.8MB)  + min 2MB embed → ion_free >= 8.8MB ---- */
static int pool_calc_pipeline_mode(int ion_free, int layer_sz) {
    int min_embed = 2 * 1024 * 1024;  /* reserve at least 2MB for embed cache */
    int slots = (ion_free - min_embed) / layer_sz;
    if (slots < 0) slots = 0;

    /* Need even number of slots for double-buffering */
    int mode;
    if      (slots >= 6)  mode = 3;
    else if (slots >= 4)  mode = 2;
    else                  mode = 1;

    int weight_slots = mode * 2;
    int embed_bytes = ion_free - weight_slots * layer_sz;

    fprintf(stderr, "  [pipeline] ion_free=%dKB → %d+%d pipe, %dW slots, embed=%dKB DDR=%dKB\n",
            ion_free/1024, mode, mode, weight_slots, embed_bytes/1024,
            embed_bytes > 0 ? embed_bytes/1024 : 0);
    return mode;
}
static int pool_reconfig_for_kv(pool_t *p, int kv_len) {
    /* Pipeline mode is determined at init from ion_free (KV in ION).
     * No dynamic change needed — KV space is pre-allocated.
     * This stub exists for the call site in decode loop. */
    (void)kv_len;
    return p->pipeline_mode;
}

/* ---- Look up token embedding: DDR > ION > SD (embed_fd) ---- */
static void pool_get_embed_row(pool_t *p, int token_id, uint8_t *row_out) {
    int off = token_id * p->D;
    if (off < p->embed_ddr_bytes) {
        memcpy(row_out, p->ddr_base + off, p->D);
        return;
    }
    if (p->embed_ion_bytes > 0 &&
        off >= p->embed_ddr_bytes &&
        off < p->embed_ddr_bytes + p->embed_ion_bytes) {
        memcpy(row_out, p->ion_vaddr + p->embed_ion_offset + (off - p->embed_ddr_bytes), p->D);
        return;
    }
    /* SD fallback: try cached fd first, re-open if needed */
    if (p->embed_fd >= 0) {
        lseek(p->embed_fd, (off_t)off, SEEK_SET);
        int n = read(p->embed_fd, row_out, p->D);
        if (n == p->D) return;
    }
    /* Last resort: open fresh fd (cached fd may have failed) */
    {
        char epath[256];
        snprintf(epath, sizeof(epath), "%s/embed.i8", p->weight_dir);
        int fd = open(epath, O_RDONLY);
        if (fd >= 0) {
            lseek(fd, (off_t)off, SEEK_SET);
            int n = read(fd, row_out, p->D);
            close(fd);
            if (n == p->D) return;
        }
    }
    memset(row_out, 0, p->D);
}

/* ---- LM Head: stream an embedding chunk from SD, transpose to column-major.
 *   Uses pool->embed_fd (pre-opened in sm_forward_pool); reads only the
 *   required rows instead of all 28 MB.  Eliminates the full embed reload. ---- */
static void pool_transpose_lm_chunk(pool_t *p, int v_start, int cur_v,
                                    int8_t *w_chunk) {
    int D = p->D;
    int total = D * cur_v;
    off_t off = (off_t)v_start * D;

    /* Allocate staging for one chunk read (up to ~1 MB).  Fall back to
     * row-by-row reading if malloc fails. */
    uint8_t *row_buf = (uint8_t *)malloc(total);
    if (row_buf) {
        lseek(p->embed_fd, off, SEEK_SET);
        int n = read(p->embed_fd, row_buf, total);
        if (n != total) { free(row_buf); memset(w_chunk, 0, D * cur_v); return; }
        for (int j = 0; j < D; j++)
            for (int v = 0; v < cur_v; v++)
                w_chunk[j * cur_v + v] = (int8_t)row_buf[v * D + j];
        free(row_buf);
    } else {
        /* OOM fallback: read one row at a time */
        uint8_t row[576];  /* D=576 */
        for (int v = 0; v < cur_v; v++) {
            lseek(p->embed_fd, off + (off_t)v * D, SEEK_SET);
            if (read(p->embed_fd, row, D) != D) {
                memset(w_chunk, 0, D * cur_v); return;
            }
            for (int j = 0; j < D; j++)
                w_chunk[j * cur_v + v] = (int8_t)row[j];
        }
    }
}

/* ================================================================
 *  Forward pass with pool-managed batched loading
 * ================================================================ */
typedef struct {
    double t_embed, t_rms_attn, t_q, t_rope, t_kv;
    double t_scores, t_softmax, t_attn, t_wo, t_rms_ffn;
    double t_ffn_up, t_ffn_down, t_final_rms, t_lm_head;
    double t_weight_load; int n_steps;
} sm_timing_t;

static int sm_forward_pool(tpu_ctx *ctx, cvk_context_t *cvk, uint8_t *nm,
    const sm_cfg_t *c, const char *weight_dir,
    const int *token_ids, int n_tokens, int kv_start,
    sm_kv_cache_t *kv, float *logits_out,
    pool_t *pool, float *rope_cos, float *rope_sin, sm_timing_t *t,
    int need_lm_head)
{
    int D = c->D, V = c->V, L = c->n_layers;
    double ts;
    wd_kick();   /* watchdog: forward entry (covers prefill chunk / decode step) */

    /* ---- Embedding: from pool (ION + DDR, opened in pool_init) ---- */
    ts = TICK();
    float *x = (float *)malloc(n_tokens * D * sizeof(float));
    if (!x) { return -1; }
    int8_t row_i8[576];
    for (int i = 0; i < n_tokens; i++) {
        int tid = token_ids[i];
        if (tid < 0 || tid >= V) tid = 0;
        pool_get_embed_row(pool, tid, (uint8_t *)row_i8);
        dequantize_f32(x + i * D, row_i8, D, g_embed_scales ? g_embed_scales[tid] : EMBED_SCALE, 0);
    }
    t->t_embed += TICK() - ts;

    int kv_len = kv_start + n_tokens;
    sm_layer_w_t layer_buf; memset(&layer_buf, 0, sizeof(layer_buf));

    /* Repurpose ION for layer slots after first call */
    pool_ion_expand_for_batch(pool);

    int ion_n_slots = pool->ion_n_slots;
    uint8_t *ion_va = pool->ion_vaddr;
    int layer_sz = pool->layer_sz;

    /* ================================================================
     *  ION 3+3 DOUBLE-BUFFER BATCH PIPELINE
     *
     *  ION split into 2 banks (3 slots each = 10.1MB/bank):
     *    Bank A (slots 0-2): active — TPU computes from these
     *    Bank B (slots 3-5): prefetch — SD reads next batch here
     *
     *  Pipeline:  compute Bank A | SD read → Bank B  (overlapped)
     *             swap banks
     *             compute Bank B | SD read → Bank A  (overlapped)
     *
     *  Key: SD reads directly into ION — no DDR staging, no memcpy.
     *  Falls back to serial mode if ION slots < 6.
     * ================================================================ */
    int batch_slots = pool->pipeline_mode;  /* 3, 2, or 1 */
    int bank_a = 0, bank_b = batch_slots;
    int use_pipeline = (ion_n_slots >= 2 * batch_slots);

    /* Change C: single prefetch thread reads the whole next batch
     * sequentially.  Overlap with TPU compute on Bank A is preserved:
     * the thread runs while the main thread computes the active bank. */
    pf_job_t pf_batch;
    int pf_inflight = 0;
    memset(&pf_batch, 0, sizeof(pf_batch));
    pthread_mutex_init(&pf_batch.lock, NULL);
    pthread_cond_init(&pf_batch.cond, NULL);

    if (use_pipeline) {
        fprintf(stderr, "  [pipeline] ION %d+%d batch: compute Bank A | SD→Bank B\n",
                batch_slots, batch_slots);

        /* ---- Load first batch (layers 0-2) into Bank A (sync) ---- */
        ts = TICK();
        for (int i = 0; i < batch_slots; i++) {
            char lpath[256];
            snprintf(lpath, sizeof(lpath), "%s/layer%d.bin", weight_dir, i);
            int fd = open(lpath, O_RDONLY);
            if (fd < 0) { use_pipeline = 0; break; }
            uint8_t *dst = ion_va + (bank_a + i) * layer_sz;
            int remain = layer_sz;
            while (remain > 0) {
                int n = (remain < 262144) ? remain : 262144;
                if (read(fd, io_buf, n) != n) { close(fd); use_pipeline = 0; break; }
                memcpy(dst, io_buf, n); dst += n; remain -= n;
            }
            close(fd);
            if (!use_pipeline) break;
        }
        if (use_pipeline) {
            CVI_RT_MemFlush(ctx->rt_handle, pool->ion_mem);
            t->t_weight_load += TICK() - ts;

            /* Start prefetching second batch (layers 3-5) into Bank B —
             * one thread, sequential file reads (Change C). */
            if (pf_start(&pf_batch, weight_dir, batch_slots, batch_slots,
                         layer_sz, ion_va + bank_b * layer_sz) != 0) {
                use_pipeline = 0;
            }
            if (use_pipeline) pf_inflight = 1;
        }

        if (use_pipeline) {

            for (int batch = 0; batch < (L + batch_slots - 1) / batch_slots; batch++) {
                int batch_start = batch * batch_slots;
                int is_last = (batch_start + batch_slots >= L);

                /* ---- Compute active bank ---- */
                for (int i = 0; i < batch_slots && (batch_start + i) < L; i++) {
                    uint8_t *base = ion_va + (bank_a + i) * layer_sz;
                    sm_setup_ptrs(&layer_buf, base, D, pool->dkv, pool->F);
                    double lt[14] = {0};
                    int rc = sm_layer_forward(ctx, cvk, nm, c, &layer_buf,
                                              x, n_tokens, kv_start, kv_len,
                                              kv, batch_start + i,
                                              rope_cos, rope_sin, lt,
                                              pool->work_f32, pool->work_i8);
                    if (rc) {
                        if (pf_inflight) { pf_wait(&pf_batch); pf_inflight = 0; }
                        pthread_mutex_destroy(&pf_batch.lock);
                        pthread_cond_destroy(&pf_batch.cond);
                        free(x); return rc;
                    }
                    t->t_rms_attn += lt[0];  t->t_q      += lt[1];
                    t->t_rope     += lt[4];  t->t_kv     += lt[5];
                    t->t_scores   += lt[6];  t->t_softmax+= lt[7];
                    t->t_attn     += lt[8];  t->t_wo     += lt[9];
                    t->t_rms_ffn  += lt[10]; t->t_ffn_up += lt[11];
                    t->t_ffn_down += lt[13];
                }

                /* Last batch done — compute complete, break */
                if (is_last) break;

                /* ---- Wait for prefetch into Bank B (single thread) ---- */
                ts = TICK();
                if (pf_wait(&pf_batch) != 0) { use_pipeline = 0; }
                pf_inflight = 0;
                wd_kick();   /* watchdog: per weight-batch prefetch done */
                if (!use_pipeline) break;
                CVI_RT_MemFlush(ctx->rt_handle, pool->ion_mem);
                t->t_weight_load += TICK() - ts;

                /* ---- Swap banks ---- */
                int tmp = bank_a; bank_a = bank_b; bank_b = tmp;

                /* ---- Start prefetching next batch into (new) Bank B ---- */
                int next_start = batch_start + 2 * batch_slots;
                int n_pf = (next_start < L) ? (L - next_start) : 0;
                if (n_pf > batch_slots) n_pf = batch_slots;
                if (n_pf > 0) {
                    if (pf_start(&pf_batch, weight_dir, next_start, n_pf,
                                 layer_sz, ion_va + bank_b * layer_sz) != 0) {
                        use_pipeline = 0; break;
                    }
                    pf_inflight = 1;
                }
            }

        }
    }

    if (!use_pipeline) {
fallback:
        fprintf(stderr, "  [fallback] serial SD read + TPU (no pipeline)\n");
        /* Fallback: single ION slot, serial SD read + TPU */
        for (int i = 0; i < L; i++) {
            wd_kick();   /* watchdog: per-layer in serial fallback */
            ts = TICK();
            char lpath[256];
            snprintf(lpath, sizeof(lpath), "%s/layer%d.bin", weight_dir, i);
            int fd = open(lpath, O_RDONLY);
            if (fd < 0) { free(x); return -1; }
            int remain = layer_sz;
            uint8_t *dst = ion_va;
            while (remain > 0) {
                int n = (remain < 262144) ? remain : 262144;
                if (read(fd, io_buf, n) != n) { close(fd); free(x); return -1; }
                memcpy(dst, io_buf, n); dst += n; remain -= n;
            }
            close(fd);
            CVI_RT_MemFlush(ctx->rt_handle, pool->ion_mem);
            t->t_weight_load += TICK() - ts;

            sm_setup_ptrs(&layer_buf, ion_va, D, pool->dkv, pool->F);
            double lt[14] = {0};
            int rc = sm_layer_forward(ctx, cvk, nm, c, &layer_buf,
                                      x, n_tokens, kv_start, kv_len,
                                      kv, i, rope_cos, rope_sin, lt,
                                      pool->work_f32, pool->work_i8);
            if (rc) { free(x); return rc; }
            t->t_rms_attn += lt[0];  t->t_q      += lt[1];
            t->t_rope     += lt[4];  t->t_kv     += lt[5];
            t->t_scores   += lt[6];  t->t_softmax+= lt[7];
            t->t_attn     += lt[8];  t->t_wo     += lt[9];
            t->t_rms_ffn  += lt[10]; t->t_ffn_up += lt[11];
            t->t_ffn_down += lt[13];
        }
    }

    if (pf_inflight) { pf_wait(&pf_batch); pf_inflight = 0; }
    wd_kick();   /* watchdog: layer loop complete, entering LM_Head */
    pthread_mutex_destroy(&pf_batch.lock);
    pthread_cond_destroy(&pf_batch.cond);

    /* ---- LM Head: stream embed chunks, transpose via mbox or CPU.
     *   Double-buffered in ION pool memory.
     *   Skipped for intermediate chunks in chunked prefill. ---- */
    if (need_lm_head) {
    ts = TICK();
    memset(&g_lm_brk, 0, sizeof(g_lm_brk));
    const char *lm_brk = getenv("LM_BRK");   /* LM_BRK=1 enables timing; default off */
    g_lm_brk_on = (lm_brk && atoi(lm_brk)) ? 1 : 0;
    float *final_normed = (float *)malloc(n_tokens * D * sizeof(float));
    float *final_rms = (float *)malloc(D * sizeof(float));
    char path[256];
    snprintf(path, sizeof(path), "%s/final_rms.f32", weight_dir);
    read_file(path, final_rms, D * 4);
    rms_norm_f32(final_normed, x, final_rms, n_tokens, D, 1e-6f);
    free(final_rms);
    t->t_final_rms += TICK() - ts;

    ts = TICK();
    /* CHUNK sizing (Change A): transposed-chunk buffers (4 × D×CHUNK) must fit
     * in the ION region freed by the layer weight slots.  Bigger CHUNK = fewer
     * submits, larger sequential SD reads (better bandwidth).  Auto-downgrades
     * to 2048 in 1+1 mode, 4096 in 2+2/3+3 modes. */
    int weight_region = pool->ion_n_slots * pool->layer_sz;
    int CHUNK = 4096;
    /* Secondary-core EMBED_XPOSE rejects cur_v > 2048 (comm_main.c). */
    if (pool->use_mbox && CHUNK > 2048) CHUNK = 2048;
    while (CHUNK * D * 4 > weight_region) CHUNK /= 2;
    if (CHUNK < 512) CHUNK = 512;
    int n_lm_chunks = (V + CHUNK - 1) / CHUNK;
    fprintf(stderr, "  [LM_Head] CHUNK=%d (%d chunks, %.2f MB/chunk, weight_region=%d KB) "
            "emb_ddr=%dKB emb_ion=%dKB\n",
            CHUNK, n_lm_chunks, (double)(D * CHUNK) / 1024 / 1024, weight_region/1024,
            pool->embed_ddr_bytes/1024, pool->embed_ion_bytes/1024);
    float sc_final = compute_scale_sym(final_normed, n_tokens * D);
    int8_t *x_final_i8 = (int8_t *)malloc(n_tokens * D);
    quantize_i8_sym(x_final_i8, final_normed, n_tokens * D, sc_final);
    free(final_normed);

    /* LM Head needs smaller rshift than QKV/FFN because the final
     * hidden state has small INT8 values (~[-8,7] vs ~[-64,64] mid-network).
     * rshift=17 (safe for worst-case 576*127*127=9.3M) would zero out
     * nearly all dot products.  rshift=12 keeps typical dot products
     * while allowing max ~9.3M/4096=2270 which is clipped to 127 but
     * real dot products are far below that. */
    int rshift_lm = matmul_rshift(D) - 5;  /* 17 → 12, keep precision without overflow */
    if (rshift_lm < 10) rshift_lm = 10;

    int lm_use_mbox = pool->use_mbox;
    int buf_sz = D * CHUNK;
    uint8_t *xpose_src[2];
    uint8_t *xpose_dst[2];
    uint64_t xpose_src_pa[2], xpose_dst_pa[2];
    uint64_t xpose_ion_pa = pool->ion_paddr;
    for (int b = 0; b < 2; b++) {
        xpose_src[b] = pool->ion_vaddr + b * buf_sz;
        xpose_dst[b] = pool->ion_vaddr + (2 + b) * buf_sz;
        xpose_src_pa[b] = xpose_ion_pa + b * buf_sz;
        xpose_dst_pa[b] = xpose_ion_pa + (2 + b) * buf_sz;
    }

    int rc_lm, cur = 0;
    uint32_t lm_result_off = SM_S_I8_OFF;

    /* Cache-aware embed chunk read: DDR > ION > SD.
     * DDR covers [0, embed_ddr_bytes), ION covers
     * [embed_ddr_bytes, embed_ddr_bytes+embed_ion_bytes).
     * Cross-boundary or beyond-cache → SD fallback. */
    int emb_ddr = pool->embed_ddr_bytes;
    int emb_ion = pool->embed_ion_bytes;
    int emb_ion_off = pool->embed_ion_offset;

    /* ---- Async next-chunk prefetch (Change B).
     * Cache-aware chunk load: DDR > ION > async SD.
     * Cache hits are sync memcpy (fast, microseconds).  SD fallback is a
     * background thread (ef_job_t) that overlaps with the current chunk's
     * transpose + matmul + dequant.  This hides most of the SD latency that
     * was previously exposed synchronously on the critical path. ---- */
    static ef_job_t lm_job;
    static int      lm_job_pending = 0;
    static int      lm_sd_chunks   = 0;   /* diagnostics: chunks read from SD */
    lm_sd_chunks = 0;
    /* Change B2: async sec-core EMBED_XPOSE transpose of the next chunk
     * overlaps the current chunk's matmul+dequant.  lm_xpose_pending=1 while
     * a transpose is in flight (must be polled before the next matmul). */
    int lm_xpose_pending = 0;

    #define LM_LOAD_CHUNK(dst_va, v_start_, cur_v_) do { \
        off_t _off = (off_t)(v_start_) * D; \
        int   _len = D * (cur_v_); \
        if (_off + _len <= emb_ddr) { \
            memcpy((dst_va), pool->ddr_base + _off, _len); \
            lm_job_pending = 0; \
        } else if (emb_ion > 0 && _off >= emb_ddr && \
                   _off + _len <= emb_ddr + emb_ion) { \
            memcpy((dst_va), pool->ion_vaddr + emb_ion_off + (_off - emb_ddr), _len); \
            lm_job_pending = 0; \
        } else { \
            lm_sd_chunks++; \
            if (ef_start(&lm_job, pool->embed_fd, _off, _len, (dst_va)) == 0) \
                lm_job_pending = 1; \
            else { /* thread create failed: fall back to sync read */ \
                lseek(pool->embed_fd, _off, SEEK_SET); \
                read(pool->embed_fd, (dst_va), _len); \
                lm_job_pending = 0; \
            } \
        } \
    } while(0)

    /* ---- Change B2: overlap sec-core transpose with matmul ----
     * Software-pipeline across chunks (steady state, iter i):
     *   (a) confirm SD read(i+1) done → start async mbox transpose(i+1)
     *   (b) kick off SD read(i+2)
     *   (c) matmul(i) + dequant(i) using xpose_dst[cur]
     *   (d) poll transpose(i+1)
     * TRANSPOSE(i+1) therefore overlaps MATMUL(i)+DEQUANT(i), hiding the
     * ~100ms/chunk mbox EMBED_XPOSE latency behind the matmul+dequant.
     * Safe: LM_Head matmul scratch stays below MHA_OFF_DMA_DESC (0x1F800),
     * so the in-flight descriptor is never clobbered. */
    #define LM_CPU_TRANSPOSE(dst_buf, src_buf, v_cnt) do { \
        for (int _j = 0; _j < D; _j++) \
            for (int _v = 0; _v < (v_cnt); _v++) \
                ((int8_t *)(dst_buf))[_j * (v_cnt) + _v] = \
                    (int8_t)((src_buf)[_v * D + _j]); \
        CVI_RT_MemFlush(ctx->rt_handle, pool->ion_mem); \
    } while (0)

    /* Pre-load chunk 0 (sync — memcpy or SD read), kick off chunk 1's async
     * SD read (overlaps transpose(0) below), then transpose chunk 0 (sync). */
    {
        int v_start = 0;
        int cur_v = (v_start + CHUNK <= V) ? CHUNK : V - v_start;
        LM_LOAD_CHUNK(xpose_src[cur], v_start, cur_v);
        if (lm_job_pending) { ef_wait(&lm_job); lm_job_pending = 0; }
        CVI_RT_MemFlush(ctx->rt_handle, pool->ion_mem);

        if (V > CHUNK) {
            int nxt_v_start = CHUNK;
            int nxt_cur_v = (nxt_v_start + CHUNK <= V) ? CHUNK : V - nxt_v_start;
            LM_LOAD_CHUNK(xpose_src[1 - cur], nxt_v_start, nxt_cur_v);
        }

        if (lm_use_mbox) {
            int mbox_slot = 0;
            int rc = mbox_embed_xpose_async(ctx, mbox_slot,
                                             xpose_src_pa[cur], xpose_dst_pa[cur],
                                             D, cur_v);
            if (rc == 0) {
                uint8_t *nm_ptr = ctx->neuron_vaddr;
                uint64_t nm_pa = CVI_RT_MemGetPAddr(ctx->neuron_mem);
                mha_dma_desc_t *desc = mbox_desc_ptr(nm_ptr, nm_pa, mbox_slot);
                rc = mbox_poll_desc(ctx, desc, MBOX_TIMEOUT_US);
            }
            if (rc != 0) { lm_use_mbox = 0; LM_CPU_TRANSPOSE(xpose_dst[cur], xpose_src[cur], cur_v); }
        } else {
            LM_CPU_TRANSPOSE(xpose_dst[cur], xpose_src[cur], cur_v);
        }
    }

    for (int v_start = 0; v_start < V; v_start += CHUNK) {
        wd_kick();   /* watchdog: LM_Head chunk */
        double _loop0 = LM_BRK_TICK();
        int cur_v = (v_start + CHUNK <= V) ? CHUNK : V - v_start;
        int nxt = 1 - cur;
        int nxt_v_start = v_start + CHUNK;
        int has_nxt = nxt_v_start < V;

        /* (a) Confirm SD read of chunk i+1 (kicked off in the previous
         * iteration) and start the async mbox transpose of chunk i+1 into
         * xpose_dst[nxt].  Cache-hit chunks skip the wait. */
        if (has_nxt) {
            int nxt_cur_v = (nxt_v_start + CHUNK <= V) ? CHUNK : V - nxt_v_start;
            if (lm_job_pending) {
                double _a0 = LM_BRK_TICK();
                ef_wait(&lm_job);
                lm_job_pending = 0;
                CVI_RT_MemFlush(ctx->rt_handle, pool->ion_mem);
                LM_BRK_ADD(t_sd_wait, _a0);
            }
            if (lm_use_mbox) {
                int mbox_slot = 0;
                double _a1 = LM_BRK_TICK();
                if (mbox_embed_xpose_async(ctx, mbox_slot,
                                           xpose_src_pa[nxt], xpose_dst_pa[nxt],
                                           D, nxt_cur_v) == 0) {
                    lm_xpose_pending = 1;
                } else {
                    lm_use_mbox = 0;   /* fall through to CPU transpose */
                }
                LM_BRK_ADD(t_xpose_kick, _a1);
            }
            if (!lm_use_mbox) {
                double _a2 = LM_BRK_TICK();
                LM_CPU_TRANSPOSE(xpose_dst[nxt], xpose_src[nxt], nxt_cur_v);
                lm_xpose_pending = 0;
                LM_BRK_ADD(t_xpose_poll, _a2);
            }
        }

        /* (b) Kick off SD read of chunk i+2 into xpose_src[cur], freed by
         * TRANSPOSE(i) which was confirmed done in the previous iteration. */
        double _b0 = LM_BRK_TICK();
        if (nxt_v_start + CHUNK < V) {
            int v2 = nxt_v_start + CHUNK;   /* chunk i+2 */
            int c2 = (v2 + CHUNK <= V) ? CHUNK : V - v2;
            LM_LOAD_CHUNK(xpose_src[cur], v2, c2);
        }
        LM_BRK_ADD(t_load, _b0);

        /* (c) matmul + dequant of current chunk using xpose_dst[cur], which
         * was fully transposed before this iteration began. */
        double _c0 = LM_BRK_TICK();
        rc_lm = tpu_matmul_build(ctx, cvk, x_final_i8, n_tokens, D,
                                  xpose_dst[cur], cur_v,
                                  lm_result_off, SM_SCRATCH_OFF, rshift_lm);
        LM_BRK_ADD(t_matmul, _c0);
        if (!rc_lm) {
            double _c1 = LM_BRK_TICK();
            CVI_RT_Submit(ctx->rt_khandle);
            CVI_RT_MemInvld(ctx->rt_handle, ctx->neuron_mem);
            LM_BRK_ADD(t_submit, _c1);
        }
        if (rc_lm) { free(x_final_i8); free(x); free(logits_out); return -1; }

        int8_t *logits_i8 = (int8_t *)(nm + lm_result_off);
        double _d0 = LM_BRK_TICK();
        for (int t = 0; t < n_tokens; t++) {
            float *lt_out = logits_out + t * V + v_start;
            int8_t *lt_i8  = logits_i8 + t * cur_v;
            if (g_embed_scales) {
                float s = sc_final * (1 << rshift_lm);
                for (int v = 0; v < cur_v; v++)
                    lt_out[v] = (float)lt_i8[v] * s * g_embed_scales[v_start + v];
            } else {
                dequant_i8(lt_out, lt_i8, cur_v,
                           EMBED_SCALE, sc_final, rshift_lm);
            }
        }
        LM_BRK_ADD(t_dequant, _d0);

        /* (d) Poll the async transpose of chunk i+1 (ran during the matmul+
         * dequant above).  Fall back to CPU transpose on mbox failure. */
        if (has_nxt && lm_xpose_pending) {
            int nxt_cur_v = (nxt_v_start + CHUNK <= V) ? CHUNK : V - nxt_v_start;
            uint8_t *nm_ptr = ctx->neuron_vaddr;
            uint64_t nm_pa = CVI_RT_MemGetPAddr(ctx->neuron_mem);
            mha_dma_desc_t *desc = mbox_desc_ptr(nm_ptr, nm_pa, 0);
            double _d1 = LM_BRK_TICK();
            if (mbox_poll_desc(ctx, desc, MBOX_TIMEOUT_US) != 0) {
                lm_use_mbox = 0;
                double _d2 = LM_BRK_TICK();
                LM_CPU_TRANSPOSE(xpose_dst[nxt], xpose_src[nxt], nxt_cur_v);
                LM_BRK_ADD(t_xpose_poll, _d2);
            }
            LM_BRK_ADD(t_xpose_poll, _d1);
            lm_xpose_pending = 0;
        }
        cur = nxt;
        if (g_lm_brk_on) {
            g_lm_brk.n_chunks++;
            g_lm_brk.t_total += (uint64_t)(TICK() - _loop0);
        }
    }
    fprintf(stderr, "  [LM_Head] %d/%d chunks from SD\n",
            lm_sd_chunks, n_lm_chunks);
    if (g_lm_brk_on && g_lm_brk.n_chunks) {
        uint64_t u = g_lm_brk.n_chunks;
        uint64_t sum_ph = g_lm_brk.t_sd_wait + g_lm_brk.t_xpose_kick +
                          g_lm_brk.t_load + g_lm_brk.t_matmul +
                          g_lm_brk.t_submit + g_lm_brk.t_dequant +
                          g_lm_brk.t_xpose_poll;
        int64_t other = (int64_t)g_lm_brk.t_total - (int64_t)sum_ph;
        fprintf(stderr,
            "  [LM_Head brk] chunks=%llu avg_total=%.2fms "
            "tile_copy=%.2f(%.1f%%) flush=%.2f(%.1f%%) matmul=%.2f(%.1f%%) "
            "submit=%.2f(%.1f%%) dequant=%.2f(%.1f%%) sd_wait=%.2f(%.1f%%) "
            "xpose_kick=%.2f(%.1f%%) load=%.2f(%.1f%%) poll=%.2f(%.1f%%) "
            "OTHER=%.2f(%.1f%%)\n",
            (unsigned long long)u,
            g_lm_brk.t_total/1000.0/u,
            g_lm_brk.t_tile_copy/1000.0/u, 100.0*g_lm_brk.t_tile_copy/g_lm_brk.t_total,
            g_lm_brk.t_tile_flush/1000.0/u, 100.0*g_lm_brk.t_tile_flush/g_lm_brk.t_total,
            g_lm_brk.t_matmul/1000.0/u,    100.0*g_lm_brk.t_matmul/g_lm_brk.t_total,
            g_lm_brk.t_submit/1000.0/u,    100.0*g_lm_brk.t_submit/g_lm_brk.t_total,
            g_lm_brk.t_dequant/1000.0/u,   100.0*g_lm_brk.t_dequant/g_lm_brk.t_total,
            g_lm_brk.t_sd_wait/1000.0/u,   100.0*g_lm_brk.t_sd_wait/g_lm_brk.t_total,
            g_lm_brk.t_xpose_kick/1000.0/u,100.0*g_lm_brk.t_xpose_kick/g_lm_brk.t_total,
            g_lm_brk.t_load/1000.0/u,      100.0*g_lm_brk.t_load/g_lm_brk.t_total,
            g_lm_brk.t_xpose_poll/1000.0/u,100.0*g_lm_brk.t_xpose_poll/g_lm_brk.t_total,
            other/1000.0/u, 100.0*other/g_lm_brk.t_total);
        /* note: tile_copy+flush are a subset of matmul (host-side prep only) */
    }
    g_lm_brk_on = 0;
    free(x_final_i8);
    t->t_lm_head += TICK() - ts;
    } /* need_lm_head */

    t->n_steps++;
    free(x);
    return 0;
}

/* ================================================================
 *  Main
 * ================================================================ */
int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <weight_dir> <token_ids.bin> <max_new_tokens> [force_mode:0=auto,1-3=force] [eos_id:0]\n", argv[0]);
        fprintf(stderr, "  Output: one token_id per line on stdout, diagnostics on stderr\n");
        return 1;
    }
    const char *weight_dir = argv[1], *token_file = argv[2];
    int max_new = atoi(argv[3]);
    int force_mode = (argc >= 5) ? atoi(argv[4]) : 0;
    int eos_id    = (argc >= 6) ? atoi(argv[5]) : 0;
    fprintf(stderr, "force_mode=%d (0=dynamic, 1-3=fixed), eos_id=%d\n", force_mode, eos_id);
    /* Start ION-orphan watchdog BEFORE any CVI_RT / tpu_init call, so even a
     * hang inside the library (reopen-ion retry loop) is force-killed and the
     * kernel frees ION.  SM_WD_TIMEOUT env overrides the 30s default. */
    wd_start();
    /* TEST-ONLY: SM_HANG_TEST=1 spins main() forever to exercise the watchdog
     * _exit path (normal runs never enter this). */
    { const char *ht = getenv("SM_HANG_TEST");
      if (ht && atoi(ht) > 0) {
          fprintf(stderr, "[wd] SM_HANG_TEST set — spinning main; watchdog should _exit\n");
          for (;;) { }
      } }
    { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
      xorshift_state = (uint32_t)(ts.tv_sec ^ ts.tv_nsec); }
    setvbuf(stderr, NULL, _IONBF, 0);
    fprintf(stderr, "=== smollm2_pool_demo START ===\n");

    sm_cfg_t c;
    fprintf(stderr, "Reading config...\n");
    if (sm_read_config(weight_dir, &c) != 0) {
        fprintf(stderr, "ERROR: cannot read config.bin\n"); return 1;
    }
    int D=c.D, d=c.head_dim, F=c.FFN, V=c.V, max_seq=c.max_seq;
    /* SM_MAX_SEQ: optional env override to raise max_seq (INT8 KV long-context test). */
    { const char *ms = getenv("SM_MAX_SEQ");
      if (ms && atoi(ms) > 0) { max_seq = atoi(ms); c.max_seq = max_seq;
        fprintf(stderr, "[Init] SM_MAX_SEQ override -> %d\n", max_seq); } }

    fprintf(stderr, "SmolLM2-135M POOL: D=%d H=%d Kvh=%d d=%d L=%d F=%d V=%d max_seq=%d\n",
            D, c.n_heads, c.n_kv_heads, d, c.n_layers, F, V, max_seq);

    { char path[256]; snprintf(path,sizeof(path),"%s/scales.bin",weight_dir);
      g_scales=(float*)malloc(212*sizeof(float));
      if(read_file(path,g_scales,212*4)!=0){free(g_scales);g_scales=NULL;} }
    /* Load per-row embed scales (generated by convert_smollm2.py --per-row) */
    { char path[256]; snprintf(path,sizeof(path),"%s/embed_scales.f32",weight_dir);
      g_embed_scales=(float*)malloc(V*sizeof(float));
      if(read_file(path,g_embed_scales,V*4)!=0){free(g_embed_scales);g_embed_scales=NULL;} }
    /* Load per-channel layer scales (generated by convert_smollm2.py with per-channel quant) */
    { int ls_per_layer = D + c.d_qkv + c.d_qkv + D + F + F + D;
      int ls_total = c.n_layers * ls_per_layer;
      char path[256]; snprintf(path,sizeof(path),"%s/layer_scales.bin",weight_dir);
      g_layer_scales=(float*)malloc(ls_total*sizeof(float));
      if(read_file(path,g_layer_scales,ls_total*4)!=0){free(g_layer_scales);g_layer_scales=NULL;}
      else fprintf(stderr,"  [layer_scales] %d scales (%d per layer, %d KB)\n",
                   ls_total, ls_per_layer, ls_total*4/1024); }

    struct stat st; stat(token_file, &st);
    int prompt_len=st.st_size/4;
    int max_prompt = max_seq - max_new;
    if (prompt_len > max_prompt) {
        fprintf(stderr, "[Init] Truncating prompt %d -> %d (max_seq=%d, max_new=%d)\n",
                prompt_len, max_prompt, max_seq, max_new);
        prompt_len = max_prompt;
    }
    if (prompt_len < 1) prompt_len = 1;
    int *token_ids=(int*)malloc(prompt_len*sizeof(int));
    if(read_file(token_file,token_ids,prompt_len*4)!=0){fprintf(stderr,"ERROR: cannot read %s\n",token_file);return 1;}
    fprintf(stderr,"\n[Init] Prompt: %d tokens, max_new: %d\n",prompt_len,max_new);

    tpu_ctx ctx;
    if(tpu_init(&ctx,NEURON_SZ)!=0){fprintf(stderr,"TPU init failed!\n");return 1;}
    wd_kick();   /* watchdog: tpu_init done */
    cvk_context_t *cvk=ctx.cvk_ctx; uint8_t *nm=ctx.neuron_vaddr;

    /* Test MemFlush right after tpu_init */
    fprintf(stderr, "[dbg] Testing MemFlush after tpu_init...\n");
    CVI_RT_MemFlush(ctx.rt_handle, ctx.neuron_mem);
    fprintf(stderr, "[dbg] MemFlush after tpu_init OK\n");

    /* Test 1: tpu_matmul with small non-tiled params */
    {
        int tM=3, tK=256, tN=128;
        int8_t *tL=(int8_t*)calloc(tM*tK,1);
        int8_t *tR=(int8_t*)calloc(tK*tN,1);
        int8_t *tO=(int8_t*)calloc(tM*tN,1);
        for(int i=0;i<tM*tK;i++) tL[i]=1;
        for(int i=0;i<tK*tN;i++) tR[i]=1;
        fprintf(stderr,"[dbg] Test1: non-tiled tpu_matmul M=%d K=%d N=%d...\n",tM,tK,tN);
        int trc=tpu_matmul(&ctx,cvk,tL,tM,tK,tR,tN,tO,SM_SCRATCH_OFF,0);
        fprintf(stderr,"[dbg] Test1 rc=%d\n",trc);
        free(tL);free(tR);free(tO);
        if(trc){fprintf(stderr,"Test1 FAIL!\n");return 1;}
    }
    /* Test 2: tpu_matmul with tiled params (same as QKV) */
    {
        int tM=3, tK=576, tN=576;
        int8_t *tL=(int8_t*)calloc(tM*tK,1);
        int8_t *tR=(int8_t*)calloc(tK*tN,1);
        int8_t *tO=(int8_t*)calloc(tM*tN,1);
        for(int i=0;i<tM*tK;i++) tL[i]=1;
        for(int i=0;i<tK*tN;i++) tR[i]=1;
        fprintf(stderr,"[dbg] Test2: TILED tpu_matmul M=%d K=%d N=%d...\n",tM,tK,tN);
        int trc=tpu_matmul(&ctx,cvk,tL,tM,tK,tR,tN,tO,SM_SCRATCH_OFF,17);
        fprintf(stderr,"[dbg] Test2 rc=%d\n",trc);
        free(tL);free(tR);free(tO);
        if(trc){fprintf(stderr,"Test2 FAIL!\n");return 1;}
    }

    /* Init pool + load embedding & first 3 layers */
    pool_t pool;
    if(pool_init(&pool,ctx.rt_handle,weight_dir,D,c.d_qkv,F)!=0){
        fprintf(stderr,"Pool init failed!\n");return 1;}
    wd_kick();   /* watchdog: pool_init done */
    /* TEST-ONLY: SM_HANG_AFTER_INIT=1 spins AFTER ION is allocated, so the
     * watchdog has to kill a live ION holder and the kernel must free ION. */
    { const char *ha = getenv("SM_HANG_AFTER_INIT");
      if (ha && atoi(ha) > 0) {
          fprintf(stderr, "[wd] SM_HANG_AFTER_INIT set — spinning with ION held; "
                          "watchdog should _exit and release ION\n");
          for (;;) { }
      } }
    fprintf(stderr, "[swap] after pool_init: VmSwap=%d KB\n", get_swap_kb());

    /* Test MemFlush after pool_init */
    fprintf(stderr, "[dbg] Testing MemFlush after pool_init...\n");
    CVI_RT_MemFlush(ctx.rt_handle, ctx.neuron_mem);
    fprintf(stderr, "[dbg] MemFlush after pool_init OK\n");
    /* Test 3: tiled tpu_matmul after pool_init (right matrix in pool ION) */
    {
        int tM=3, tK=576, tN=576;
        int8_t *tL=(int8_t*)calloc(tM*tK,1);
        int8_t *tO=(int8_t*)calloc(tM*tN,1);
        for(int i=0;i<tM*tK;i++) tL[i]=1;
        for(int i=0;i<tK*tN;i++) pool.ion_vaddr[i]=1; // write to ION
        fprintf(stderr,"[dbg] Test3: TILED with ION right matrix...\n");
        int trc=tpu_matmul(&ctx,cvk,tL,tM,tK,(int8_t*)pool.ion_vaddr,tN,tO,SM_SCRATCH_OFF,17);
        fprintf(stderr,"[dbg] Test3 rc=%d\n",trc);
        free(tL);free(tO);
        if(trc){fprintf(stderr,"Test3 FAIL!\n");return 1;}
    }

    if(pool_load_embed_and_init_layers(&pool)!=0){
        fprintf(stderr,"Embed+init layers load failed!\n");return 1;}
    wd_kick();   /* watchdog: embed+init layers done */
    fprintf(stderr, "[swap] after embed+init load: VmSwap=%d KB\n", get_swap_kb());

    /* Test MemFlush after embed load */
    fprintf(stderr, "[dbg] Testing MemFlush after embed load...\n");
    CVI_RT_MemFlush(ctx.rt_handle, ctx.neuron_mem);
    fprintf(stderr, "[dbg] MemFlush after embed load OK\n");
    /* Test 4: tiled tpu_matmul after embed load */
    {
        int tM=3, tK=576, tN=576;
        int8_t *tL=(int8_t*)calloc(tM*tK,1);
        int8_t *tO=(int8_t*)calloc(tM*tN,1);
        int8_t *tR=(int8_t*)calloc(tK*tN,1);
        for(int i=0;i<tM*tK;i++) tL[i]=1;
        for(int i=0;i<tK*tN;i++) tR[i]=1;
        fprintf(stderr,"[dbg] Test4: TILED with DDR right matrix after embed...\n");
        int trc=tpu_matmul(&ctx,cvk,tL,tM,tK,tR,tN,tO,SM_SCRATCH_OFF,17);
        fprintf(stderr,"[dbg] Test4 rc=%d\n",trc);
        free(tL);free(tR);free(tO);
        if(trc){fprintf(stderr,"Test4 FAIL!\n");return 1;}
    }

    /* Shrink max_seq to actual needed length — saves KV cache space in ION */
    int kv_needed = prompt_len + max_new;
    if (c.max_seq > kv_needed) { c.max_seq = kv_needed; max_seq = kv_needed; }

    float *rope_cos=(float*)malloc(max_seq*(d/2)*sizeof(float));
    float *rope_sin=(float*)malloc(max_seq*(d/2)*sizeof(float));
    rope_precompute(max_seq,d,rope_cos,rope_sin);

    /* KV cache in ION — frees Linux heap, reduces swap */
    sm_kv_cache_t *kv=sm_kv_alloc_ion(&pool, max_seq, c.d_qkv, c.n_layers, ctx.rt_handle);
    if(!kv){fprintf(stderr,"KV cache alloc failed!\n");return 1;}
    fprintf(stderr, "[swap] after kv_alloc_ion: VmSwap=%d KB\n", get_swap_kb());

    /* Pipeline mode from actual ION free space after KV allocation */
    pool.pipeline_mode = pool_calc_pipeline_mode(pool.ion_free, pool.layer_sz);
    if (force_mode) pool.pipeline_mode = force_mode;
    pool.ion_n_slots = pool.pipeline_mode * 2;

    /* ---- Chunked Prefill: TPU can handle at most 10 tokens/batch.
     *   Split long prompts into chunks, accumulate KV cache across chunks,
     *   only compute LM Head on the last chunk. ---- */
    #define CHUNK_PREFILL 10
    fprintf(stderr,"\n[Prefill] %d tokens (chunked, max %d/chunk)...\n",prompt_len,CHUNK_PREFILL);
    sm_timing_t t; memset(&t,0,sizeof(t));
    double t_prefill=TICK();
    float *chunk_logits=(float*)malloc(CHUNK_PREFILL * V * sizeof(float));
    if(!chunk_logits){fprintf(stderr,"OOM for chunk logits\n");return 1;}

    int next_token = 0;
    for (int start = 0; start < prompt_len; start += CHUNK_PREFILL) {
        int chunk = (start + CHUNK_PREFILL <= prompt_len) ? CHUNK_PREFILL : prompt_len - start;
        int is_last = (start + chunk >= prompt_len);
        float *logits_ptr = is_last ? chunk_logits : NULL;
        int rc = sm_forward_pool(&ctx, cvk, nm, &c, weight_dir,
                                  &token_ids[start], chunk, start,
                                  kv, logits_ptr,
                                  &pool, rope_cos, rope_sin, &t, is_last);
        if (rc) {
            fprintf(stderr,"Prefill chunk start=%d failed rc=%d\n", start, rc);
            free(chunk_logits); return 1;
        }
        if (is_last) {
            next_token = sample_argmax(chunk_logits + (chunk - 1) * V, V);
        }
    }
    free(chunk_logits);
    t_prefill = TICK() - t_prefill;
    fprintf(stderr,"  Prefill: %.0f ms, next_token=%d\n",t_prefill/1000.0,next_token);
    wd_kick();   /* watchdog: prefill done */

    /* ---- Decode ---- */
    fprintf(stderr,"\n[Decode] %d tokens...\n",max_new);
    fprintf(stderr,"[swap] before decode: VmSwap=%d KB\n", get_swap_kb());
    int generated[256],n_gen=0;
    printf("%d\n", next_token); fflush(stdout);  /* stream first token (from prefill) */
    generated[n_gen++]=next_token;
    double t_decode_total=0; int kv_len=prompt_len;

    for(int step=0;step<max_new;step++){
        int sw_now = get_swap_kb();
        int tid[1]={next_token};
        float *step_logits=(float*)malloc(V*sizeof(float));
        double t_step=TICK();
        int rc=sm_forward_pool(&ctx,cvk,nm,&c,weight_dir,tid,1,kv_len,kv,step_logits,&pool,rope_cos,rope_sin,&t,1);
        t_step=TICK()-t_step; t_decode_total+=t_step; kv_len++;
        if(rc){fprintf(stderr,"Decode step %d failed rc=%d\n",step,rc);break;}
        next_token=sample_argmax(step_logits,V);  /* log top-5 */
        next_token=sample_softmax(step_logits,V,1.5f,40);  /* temperature sampling */
        free(step_logits);
        if(next_token<0||next_token>=V)break;
        printf("%d\n", next_token); fflush(stdout);  /* stream token */
        generated[n_gen++]=next_token;
        if(next_token == eos_id) {
            fprintf(stderr,"  step %d: EOS (tok=%d), stopping\n", step+1, next_token);
            break;
        }
        if((step+1)%5==0)fprintf(stderr,"  step %d: tok=%d, %.0f ms (avg %.0f ms/tok) swap=%dK\n",
            step+1,next_token,t_step/1000.0,t_decode_total/1000.0/(step+1), sw_now);
    }

    /* ---- Report ---- */
    double t_total=t_prefill+t_decode_total; int ns=t.n_steps>0?t.n_steps:1;
    fprintf(stderr,"\n===== RESULTS =====\n");
    fprintf(stderr,"Total:     %8.0f ms\n",t_total/1000.0);
    fprintf(stderr,"Prefill:   %8.0f ms\n",t_prefill/1000.0);
    fprintf(stderr,"Decode:    %8.0f ms (%d tok, %.0f ms/tok, %.2f tok/s)\n",
            t_decode_total/1000.0,n_gen,t_decode_total/1000.0/n_gen,1000.0*n_gen/t_decode_total);
    fprintf(stderr,"\n--- Per-step avg (n=%d) ---\n",ns);
    fprintf(stderr,"Wt load:%8.0f ms  Embed:%8.0f ms  RMS_attn:%8.0f ms\n",t.t_weight_load/1000.0/ns,t.t_embed/1000.0/ns,t.t_rms_attn/1000.0/ns);
    fprintf(stderr,"QKV:   %8.0f ms  RoPE: %8.0f ms  KV:     %8.0f ms\n",t.t_q/1000.0/ns,t.t_rope/1000.0/ns,t.t_kv/1000.0/ns);
    fprintf(stderr,"Scores:%8.0f ms  SM:   %8.0f ms  Attn:   %8.0f ms\n",t.t_scores/1000.0/ns,t.t_softmax/1000.0/ns,t.t_attn/1000.0/ns);
    fprintf(stderr,"Wo:    %8.0f ms  RMS_f:%8.0f ms  FFN_up: %8.0f ms\n",t.t_wo/1000.0/ns,t.t_rms_ffn/1000.0/ns,t.t_ffn_up/1000.0/ns);
    fprintf(stderr,"FFN_dn:%8.0f ms  Final:%8.0f ms  LM_Head:%8.0f ms\n",t.t_ffn_down/1000.0/ns,t.t_final_rms/1000.0/ns,t.t_lm_head/1000.0/ns);
    fprintf(stderr,"\n--- Tokens (%d) ---\n", n_gen);
    for(int i=0;i<n_gen;i++)fprintf(stderr,"%d ",generated[i]); fprintf(stderr,"\n");

    free(rope_cos);free(rope_sin);free(token_ids);
    if(g_scales)free(g_scales);
    if(g_embed_scales)free(g_embed_scales);
    if(g_layer_scales)free(g_layer_scales);
    sm_kv_free(kv,&c,ctx.rt_handle); pool_free(&pool,ctx.rt_handle);
    tpu_cleanup(&ctx);
    return 0;
}
