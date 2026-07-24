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

/* ================================================================
 *  Runtime config
 * ================================================================ */
typedef struct {
    int D, n_heads, n_kv_heads, head_dim, n_layers, FFN, V, max_seq;
    int d_qkv, n_groups;
} sm_cfg_t;

static float *g_scales = NULL;
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

static int sample_argmax(const float *logits, int n) {
    int best = 0; float best_v = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > best_v) { best_v = logits[i]; best = i; }
    return best;
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
        if(!r_is_nm){uint8_t*td=nm+off_r; for(int r=0;r<K;r++)memcpy(td+r*cn,r_i8+r*N+ns,cn); CVI_RT_MemFlush(ctx->rt_handle,ctx->neuron_mem);}
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
 *  KV Cache — stored in ION (CPU-accessible via vaddr), freeing
 *  ~3 MB DDR for the weight pool.
 * ================================================================ */
typedef struct {
    float *K[30], *V[30]; /* per-layer KV buffers */
} sm_kv_cache_t;

static sm_kv_cache_t *sm_kv_alloc(CVI_RT_HANDLE rt, const sm_cfg_t *c) {
    sm_kv_cache_t *kv = (sm_kv_cache_t*)calloc(1, sizeof(sm_kv_cache_t));
    if (!kv) return NULL;
    int per_layer = c->max_seq * c->d_qkv * sizeof(float);
    for (int l = 0; l < c->n_layers; l++) {
        kv->K[l] = (float *)calloc(per_layer, 1);
        kv->V[l] = (float *)calloc(per_layer, 1);
        if (!kv->K[l] || !kv->V[l]) {
            for (int j = 0; j <= l; j++) { free(kv->K[j]); free(kv->V[j]); }
            free(kv); return NULL;
        }
    }
    return kv;
}

static void sm_kv_free(sm_kv_cache_t *kv, const sm_cfg_t *c, CVI_RT_HANDLE rt) {
    if (!kv) return;
    (void)rt;
    for (int l = 0; l < c->n_layers; l++) { free(kv->K[l]); free(kv->V[l]); }
    free(kv);
}
static void sm_kv_store_contig(float *cache, const float *new_data, int seq, int pos, int dkv) {
    memcpy(cache+pos*dkv, new_data, seq*dkv*sizeof(float));
}
static void sm_kv_load_contig(float *dst, const float *cache, int kv_len, int dkv) {
    memcpy(dst, cache, kv_len*dkv*sizeof(float));
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
    double *timing)
{
    int D=c->D, H=c->n_heads, Kvh=c->n_kv_heads, d=c->head_dim;
    int dkv=c->d_qkv, F=c->FFN, groups=c->n_groups;
    int total=seq*D; double ts;
    int rc;

    ts=TICK();
    float *normed=(float*)malloc(total*sizeof(float));
    if(!normed){fprintf(stderr,"    L%d: normed OOM\n",layer);return-1;}
    rms_norm_f32(normed, x, w->rms_attn, seq, D, 1e-6f);
    timing[0]+=TICK()-ts;

    float sc_x=compute_scale_sym(normed, total);
    int8_t *x_i8=(int8_t*)malloc(total);
    if(!x_i8){fprintf(stderr,"    L%d: x_i8 OOM\n",layer);free(normed);return-1;}
    quantize_i8_sym(x_i8, normed, total, sc_x);

    /* ---- Q, K, V matmuls: use tpu_matmul (proven with M=3 in Tests 1-4) ---- */
    int rshift_qkv=matmul_rshift(D);
    int wq_sz=seq*D, wk_sz=seq*dkv, wv_sz=seq*dkv;
    int8_t *Q_i8=(int8_t*)malloc(wq_sz),*K_i8=(int8_t*)malloc(wk_sz),*V_i8=(int8_t*)malloc(wv_sz);
    if(!Q_i8||!K_i8||!V_i8){fprintf(stderr,"    L%d: QKV OOM\n",layer);free(Q_i8);free(K_i8);free(V_i8);free(x_i8);free(normed);return-1;}
    ts=TICK();
    rc = tpu_matmul(ctx,cvk,x_i8,seq,D,w->Wq,D, Q_i8,SM_SCRATCH_OFF,rshift_qkv);
    if(!rc) rc = tpu_matmul(ctx,cvk,x_i8,seq,D,w->Wk,dkv, K_i8,SM_SCRATCH_OFF,rshift_qkv);
    if(!rc) rc = tpu_matmul(ctx,cvk,x_i8,seq,D,w->Wv,dkv, V_i8,SM_SCRATCH_OFF,rshift_qkv);
    if(rc){fprintf(stderr,"    L%d: QKV FAIL rc=%d\n",layer,rc); free(Q_i8);free(K_i8);free(V_i8);free(x_i8);free(normed);return rc;}
    timing[1]+=TICK()-ts; free(x_i8);

    /* ---- Dequant Q, K, V ---- */
    float *Q_f32=(float*)malloc(seq*D*sizeof(float)),*K_f32=(float*)malloc(seq*dkv*sizeof(float)),*V_f32=(float*)malloc(seq*dkv*sizeof(float));
    if(!Q_f32||!K_f32||!V_f32){fprintf(stderr,"    L%d: dequant OOM\n",layer);free(Q_i8);free(K_i8);free(V_i8);free(normed);return-1;}
    dequant_i8(Q_f32,Q_i8,seq*D,W_SCALE(layer,0),sc_x,rshift_qkv);
    dequant_i8(K_f32,K_i8,seq*dkv,W_SCALE(layer,1),sc_x,rshift_qkv);
    dequant_i8(V_f32,V_i8,seq*dkv,W_SCALE(layer,2),sc_x,rshift_qkv);
    free(Q_i8);free(K_i8);free(V_i8);

    /* ---- RoPE ---- */
    ts=TICK();
    for(int s=0;s<seq;s++){
        for(int h=0;h<H;h++)rope_apply_single_f32(Q_f32+s*D+h*d,d,pos+s,rope_cos,rope_sin);
        for(int h=0;h<Kvh;h++)rope_apply_single_f32(K_f32+s*dkv+h*d,d,pos+s,rope_cos,rope_sin);
    }
    timing[4]+=TICK()-ts;

    /* ---- KV cache store + load ---- */
    ts=TICK();
    sm_kv_store_contig(kv->K[layer],K_f32,seq,pos,dkv);
    sm_kv_store_contig(kv->V[layer],V_f32,seq,pos,dkv);
    timing[5]+=TICK()-ts;

    float *K_full=(float*)malloc(kv_len*dkv*sizeof(float)),*V_full=(float*)malloc(kv_len*dkv*sizeof(float));
    if(!K_full||!V_full){fprintf(stderr,"    L%d: K_full OOM\n",layer);free(Q_f32);free(K_f32);free(V_f32);free(normed);return-1;}
    sm_kv_load_contig(K_full,kv->K[layer],kv_len,dkv);
    sm_kv_load_contig(V_full,kv->V[layer],kv_len,dkv);

    /* ---- Attention: prep Qg, Kt in neuron memory ---- */
    float softmax_scale=1.0f/sqrtf((float)d);
    float *Attn_out=(float*)calloc(seq*D,sizeof(float));
    int rshift_scores=matmul_rshift(d);
    int Qg_sz=seq*groups*d, Kt_sz=d*kv_len, Sg_sz=seq*groups*kv_len;
    float sc_qg[9],sc_kh[9];

    for(int g=0;g<Kvh;g++){
        float *Qg_f32=(float*)malloc(Qg_sz*sizeof(float));
        for(int s=0;s<seq;s++)for(int h=0;h<groups;h++)memcpy(Qg_f32+(s*groups+h)*d,Q_f32+s*D+(g*groups+h)*d,d*sizeof(float));
        sc_qg[g]=compute_scale_sym(Qg_f32,Qg_sz);
        quantize_i8_sym((int8_t*)(nm+SM_Q_I8_OFF+g*Qg_sz),Qg_f32,Qg_sz,sc_qg[g]); free(Qg_f32);
        float *Kh_f32=(float*)malloc(kv_len*d*sizeof(float));
        for(int s=0;s<kv_len;s++)memcpy(Kh_f32+s*d,K_full+s*dkv+g*d,d*sizeof(float));
        sc_kh[g]=compute_scale_sym(Kh_f32,kv_len*d);
        int8_t *Kh_i8_tmp=(int8_t*)malloc(kv_len*d);
        quantize_i8_sym(Kh_i8_tmp,Kh_f32,kv_len*d,sc_kh[g]); free(Kh_f32);
        int8_t *Kt=(int8_t*)(nm+SM_KT_I8_OFF+g*Kt_sz);
        for(int r=0;r<kv_len;r++)for(int c=0;c<d;c++)Kt[c*kv_len+r]=Kh_i8_tmp[r*d+c];
        free(Kh_i8_tmp);
    }

    /* ---- Scores: batch build all groups, single Submit ---- */
    ts=TICK();
    for(int g=0;g<Kvh;g++){
        int8_t *Qg_i8=(int8_t*)(nm+SM_Q_I8_OFF+g*Qg_sz),*Kt_g=(int8_t*)(nm+SM_KT_I8_OFF+g*Kt_sz);
        rc=tpu_matmul_build(ctx,cvk,Qg_i8,seq*groups,d,Kt_g,kv_len, SM_S_I8_OFF+g*Sg_sz,SM_SCRATCH_OFF,rshift_scores);
        if(rc){free(K_full);free(V_full);free(Attn_out);free(Q_f32);free(K_f32);free(V_f32);free(normed);return rc;}
    }
    CVI_RT_Submit(ctx->rt_khandle);
    CVI_RT_MemInvld(ctx->rt_handle, ctx->neuron_mem);
    timing[6]+=TICK()-ts;

    /* ---- Softmax + quantize S, prep V in neuron memory ---- */
    float sc_sg[9],sc_vh[9];
    ts=TICK();
    for(int g=0;g<Kvh;g++){
        int8_t *Sg_i8=(int8_t*)(nm+SM_S_I8_OFF+g*Sg_sz);
        float *Scores_f32=(float*)malloc(seq*groups*kv_len*sizeof(float));
        dequant_i8(Scores_f32,Sg_i8,seq*groups*kv_len,sc_kh[g],sc_qg[g],rshift_scores);
        for(int h=0;h<groups;h++){for(int s=0;s<seq;s++){float*row=Scores_f32+(s*groups+h)*kv_len;int mask_from=pos+s+1;for(int i=0;i<kv_len;i++){if(i>=mask_from)row[i]=-1e30f;else row[i]*=softmax_scale;}softmax_f32(row,1,kv_len);}}
        sc_sg[g]=compute_scale_sym(Scores_f32,seq*groups*kv_len);
        quantize_i8_sym(Sg_i8,Scores_f32,seq*groups*kv_len,sc_sg[g]); free(Scores_f32);
        float *Vh_f32=(float*)malloc(kv_len*d*sizeof(float));
        for(int s=0;s<kv_len;s++)memcpy(Vh_f32+s*d,V_full+s*dkv+g*d,d*sizeof(float));
        sc_vh[g]=compute_scale_sym(Vh_f32,kv_len*d);
        quantize_i8_sym((int8_t*)(nm+SM_V_I8_OFF+g*kv_len*d),Vh_f32,kv_len*d,sc_vh[g]); free(Vh_f32);
    }
    timing[7]+=TICK()-ts;

    /* ---- Attn output: batch build all groups, single Submit ---- */
    ts=TICK();
    int rshift_attn=matmul_rshift(kv_len);
    for(int g=0;g<Kvh;g++){
        rc=tpu_matmul_build(ctx,cvk,(int8_t*)(nm+SM_S_I8_OFF+g*Sg_sz),seq*groups,kv_len,
                            (int8_t*)(nm+SM_V_I8_OFF+g*kv_len*d),d,
                            SM_A_I8_OFF+g*seq*groups*d, SM_SCRATCH_OFF, rshift_attn);
        if(rc){free(K_full);free(V_full);free(Attn_out);free(Q_f32);free(K_f32);free(V_f32);free(normed);return rc;}
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
    free(K_full);free(V_full);

    /* ---- Wo projection ---- */
    ts=TICK();
    float sc_attn_q=compute_scale_sym(Attn_out,seq*D);
    quantize_i8_sym((int8_t*)(nm+SM_O_I8_OFF),Attn_out,seq*D,sc_attn_q);
    int8_t *O_i8=(int8_t*)malloc(total);
    if(!O_i8){free(Attn_out);free(Q_f32);free(K_f32);free(V_f32);free(normed);return-1;}
    rc=tpu_matmul_build(ctx,cvk,(int8_t*)(nm+SM_O_I8_OFF),seq,D,w->Wo,D, SM_Q_I8_OFF,SM_SCRATCH_OFF,rshift_qkv);
    if(!rc){CVI_RT_Submit(ctx->rt_khandle); CVI_RT_MemInvld(ctx->rt_handle, ctx->neuron_mem); memcpy(O_i8, nm+SM_Q_I8_OFF, total);}
    timing[9]+=TICK()-ts;
    if(rc){free(O_i8);free(Attn_out);free(Q_f32);free(K_f32);free(V_f32);free(normed);return rc;}
    float *Wo_out=(float*)malloc(total*sizeof(float));
    dequant_i8(Wo_out,O_i8,total,W_SCALE(layer,3),sc_attn_q,rshift_qkv);
    for(int i=0;i<total;i++)x[i]+=Wo_out[i];
    free(Wo_out);free(O_i8);free(Attn_out);free(Q_f32);free(K_f32);free(V_f32);

    /* ---- FFN: rms_norm + quantize ---- */
    ts=TICK(); rms_norm_f32(normed,x,w->rms_ffn,seq,D,1e-6f); timing[10]+=TICK()-ts;
    sc_x=compute_scale_sym(normed,total); x_i8=(int8_t*)malloc(total); quantize_i8_sym(x_i8,normed,total,sc_x); free(normed);

    /* ---- FFN up + gate: batch build, single Submit ---- */
    int rshift_ffn_up=matmul_rshift(D);
    ts=TICK();
    rc = tpu_matmul_build(ctx,cvk,x_i8,seq,D,w->ffn_up,F, SM_UP_I8_OFF,SM_SCRATCH_OFF,rshift_ffn_up);
    if(!rc) rc = tpu_matmul_build(ctx,cvk,x_i8,seq,D,w->ffn_gate,F, SM_GATE_I8_OFF,SM_SCRATCH_OFF+0x10000,rshift_ffn_up);
    if(rc){free(x_i8);return rc;}
    CVI_RT_Submit(ctx->rt_khandle);
    CVI_RT_MemInvld(ctx->rt_handle, ctx->neuron_mem);
    timing[11]+=TICK()-ts; free(x_i8);

    int8_t *up_i8=(int8_t*)malloc(seq*F),*gate_i8=(int8_t*)malloc(seq*F);
    if(!up_i8||!gate_i8){free(up_i8);free(gate_i8);return-1;}
    memcpy(up_i8, nm+SM_UP_I8_OFF, seq*F);
    memcpy(gate_i8, nm+SM_GATE_I8_OFF, seq*F);

    float *up_f32=(float*)malloc(seq*F*sizeof(float)),*gate_f32=(float*)malloc(seq*F*sizeof(float));
    dequant_i8(up_f32,up_i8,seq*F,W_SCALE(layer,4),sc_x,rshift_ffn_up);
    dequant_i8(gate_f32,gate_i8,seq*F,W_SCALE(layer,5),sc_x,rshift_ffn_up);
    free(up_i8);free(gate_i8); silu_f32(gate_f32,seq*F);
    for(int i=0;i<seq*F;i++)up_f32[i]*=gate_f32[i]; free(gate_f32);

    /* ---- FFN down ---- */
    float sc_mid=compute_scale_sym(up_f32,seq*F);
    quantize_i8_sym((int8_t*)(nm+SM_UP_I8_OFF),up_f32,seq*F,sc_mid); free(up_f32);
    int rshift_ffn_down=matmul_rshift(F);
    ts=TICK();
    int8_t *down_i8=(int8_t*)malloc(total);
    if(!down_i8) return -1;
    rc=tpu_matmul_build(ctx,cvk,(int8_t*)(nm+SM_UP_I8_OFF),seq,F,w->ffn_down,D, SM_Q_I8_OFF,SM_SCRATCH_OFF,rshift_ffn_down);
    if(!rc){CVI_RT_Submit(ctx->rt_khandle); CVI_RT_MemInvld(ctx->rt_handle, ctx->neuron_mem); memcpy(down_i8, nm+SM_Q_I8_OFF, total);}
    timing[13]+=TICK()-ts;
    if(rc){free(down_i8);return rc;}
    float *ffn_out=(float*)malloc(total*sizeof(float));
    dequant_i8(ffn_out,down_i8,total,W_SCALE(layer,6),sc_mid,rshift_ffn_down);
    for(int i=0;i<total;i++)x[i]+=ffn_out[i];
    free(ffn_out);free(down_i8);
    return 0;
}

/* ================================================================
 *  PREFETCH PIPELINE
 *
 *  Design: while TPU computes layer N, a background thread reads
 *  layer N+1 from SD into a DDR staging buffer.  This hides SD
 *  latency behind TPU compute.
 *
 *  Memory: staging = 2 * layer_sz (~6.8 MB).  Fits in remaining DDR.
 * ================================================================ */
typedef struct {
    uint8_t     *buf;           /* staging buffer for one layer        */
    int          layer_id;      /* which layer to load                */
    int          sz;            /* layer_sz bytes                     */
    int          ready;         /* 0=loading, 1=done, -1=error       */
    const char  *weight_dir;
    pthread_t        tid;
    pthread_mutex_t  lock;
    pthread_cond_t   cond;
} pf_job_t;

static void *pf_worker(void *arg) {
    pf_job_t *j = (pf_job_t *)arg;
    char path[256];

    snprintf(path, sizeof(path), "%s/layer%d.bin", j->weight_dir, j->layer_id);
    int fd = open(path, O_RDONLY);
    if (fd < 0) { j->ready = -1; return NULL; }
    int remain = j->sz;
    uint8_t *dst = j->buf;
    while (remain > 0) {
        int n = read(fd, dst, remain);
        if (n <= 0) { close(fd); j->ready = -1; return NULL; }
        dst += n; remain -= n;
    }
    close(fd);

    pthread_mutex_lock(&j->lock);
    j->ready = 1;
    pthread_cond_signal(&j->cond);
    pthread_mutex_unlock(&j->lock);
    return NULL;
}

static void pf_start(pf_job_t *j, const char *weight_dir, int layer_id, int sz) {
    j->weight_dir = weight_dir;
    j->layer_id   = layer_id;
    j->sz         = sz;
    j->ready      = 0;
    pthread_create(&j->tid, NULL, pf_worker, j);
}

static int pf_wait(pf_job_t *j) {
    pthread_mutex_lock(&j->lock);
    while (j->ready == 0)
        pthread_cond_wait(&j->cond, &j->lock);
    int r = j->ready;
    pthread_mutex_unlock(&j->lock);
    void *ret; pthread_join(j->tid, &ret);
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
#define DDR_POOL_TRY     0x800000    /* 8 MB — need 2x layer_sz (~6.8MB) for dual buf */
#define DDR_POOL_MIN     0x400000    /* fallback 4 MB */
#define ION_MAX_SLOTS    7           /* max layer slots in ION */
#define DDR_MAX_OVERFLOW 5           /* max DDR overflow layers */
#define MBOX_TIMEOUT_US  2000000     /* 2 sec timeout for secondary core commands */

static uint8_t io_buf[256 * 1024];   /* staging buffer in BSS — zero heap pressure */

typedef struct {
    /* ION */
    CVI_RT_MEM  ion_mem;
    uint8_t    *ion_vaddr;
    uint64_t    ion_paddr;

    /* DDR */
    uint8_t    *ddr_base;
    int         ddr_sz;

    /* Embedding split */
    int         embed_total;      /* D * V bytes */
    int         embed_ddr_bytes;  /* bytes of embedding stored in DDR */
    int         embed_ion_bytes;  /* bytes of embedding stored in ION */

    /* ION layer slots (after embedding portion) */
    int         ion_layer_off;    /* byte offset in ION where layer slots start */
    int         ion_n_slots;      /* actual number of layer slots in ION */
    int         ion_slot_layer[ION_MAX_SLOTS]; /* which layer is in each slot, -1=free */

    /* Current batch state */
    int         batch_start, batch_end;
    int         ddr_layer_ids[DDR_MAX_OVERFLOW];
    int         ddr_n_layers;     /* actual number of DDR overflow layers this batch */
    int         ddr_max_overflow; /* computed: ddr_sz / layer_sz */
    int         batch_target;     /* computed: ion_n_slots + ddr_max_overflow */

    int         layer_sz, D, dkv, F;
    char        weight_dir[256];
    int         embed_fd;         /* fd for streaming embed reads (LM Head) */

    /* Dual DDR staging buffers for pipelined SD→DDR→ION */
    uint8_t    *ddr_buf[2];       /* DDR staging, each layer_sz bytes */
    uint64_t    ddr_buf_pa[2];    /* phys addr (0 if unresolved, local memcpy only) */
    int         ddr_buf_loaded[2]; /* which buffer has valid data */
    int         ddr_cur_buf;      /* current buffer index for next SD read */
    int         use_mbox;         /* 1 if secondary core mailbox is available */
    uint64_t    nm_paddr;         /* cached neuron memory phys addr for mbox */
    int         ddr_pa_done;      /* phys addr resolution attempted */
} pool_t;

static void pool_free(pool_t *p, CVI_RT_HANDLE rt) {
    if (p->embed_fd >= 0) { close(p->embed_fd); p->embed_fd = -1; }
    if (p->ion_mem) { CVI_RT_MemFree(rt, p->ion_mem); p->ion_mem = NULL; }
    free(p->ddr_base);
    free(p->ddr_buf[0]);
    free(p->ddr_buf[1]);
    mbox_close();
    memset(p, 0, sizeof(*p));
}

static int pool_init(pool_t *p, CVI_RT_HANDLE rt, const char *weight_dir,
                     int D, int dkv, int F) {
    memset(p, 0, sizeof(*p));
    p->D = D; p->dkv = dkv; p->F = F;
    p->layer_sz = sm_layer_bytes(D, dkv, F);
    p->embed_total = D * 49152;  /* vocab=49152, weight-tied with lm_head */
    p->embed_fd = -1;
    snprintf(p->weight_dir, sizeof(p->weight_dir), "%s", weight_dir);

    /* --- ION allocation --- */
    p->ion_mem = CVI_RT_MemAlloc(rt, ION_POOL_SZ);
    if (!p->ion_mem) { fprintf(stderr, "  POOL: ION alloc failed\n"); return -1; }
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

    /* --- Decide embedding split: DDR first, rest in ION.
     *   Reserve up to 2*layer_sz for dual staging buffers (post-expand).
     *   If DDR is too small for 2 full buffers, fall back to 1 buffer. --- */
    int ddr_staging_needed = p->layer_sz * 2;
    if (ddr_staging_needed > (int)p->ddr_sz) {
        ddr_staging_needed = p->layer_sz;  /* single buffer fallback */
    }
    int ddr_embed_max = (int)p->ddr_sz - ddr_staging_needed;
    if (ddr_embed_max < 0) ddr_embed_max = 0;
    p->embed_ddr_bytes = p->embed_total;
    if (p->embed_ddr_bytes > ddr_embed_max) p->embed_ddr_bytes = ddr_embed_max;
    p->embed_ion_bytes = p->embed_total - p->embed_ddr_bytes;
    /* Cap ION embed portion to ION_POOL_SZ (must leave room for at least
     * 2 init-layer slots).  Embed rows beyond DDR+ION are streamed from SD. */
    int ion_reserve = p->layer_sz * 2;  /* at least 2 init layer slots */
    int ion_embed_max = (int)ION_POOL_SZ - ion_reserve;
    if (p->embed_ion_bytes > ion_embed_max) {
        p->embed_ion_bytes = ion_embed_max;
        /* Adjust total: everything beyond DDR+ION is SD-only */
    }

    /* --- ION layer slots: after embedding portion --- */
    p->ion_layer_off = p->embed_ion_bytes;
    int ion_remain = ION_POOL_SZ - p->ion_layer_off;
    p->ion_n_slots = ion_remain / p->layer_sz;
    if (p->ion_n_slots > ION_MAX_SLOTS) p->ion_n_slots = ION_MAX_SLOTS;

    /* --- DDR overflow: entire DDR available after embedding lookup discarded --- */
    p->ddr_max_overflow = p->ddr_sz / p->layer_sz;
    if (p->ddr_max_overflow > DDR_MAX_OVERFLOW) p->ddr_max_overflow = DDR_MAX_OVERFLOW;
    p->batch_target = p->ion_n_slots + p->ddr_max_overflow;

    /* --- Try to open mailbox to secondary core --- */
    p->use_mbox = (mbox_open() == 0);
    if (p->use_mbox) {
        fprintf(stderr, "[pool] mbox OK (secondary core available)\n");
    } else {
        fprintf(stderr, "[pool] mbox not available — fallback to local memcpy\n");
    }

    /* --- Resolve DDR buffer phys addrs via /proc/self/pagemap --- */
    p->ddr_buf_pa[0] = 0;
    p->ddr_buf_pa[1] = 0;
    p->ddr_pa_done = 1;
    if (!p->ddr_pa_done) {} /* suppress unused warning */
    /* pagemap not available on this kernel; DDR→ION always uses local memcpy.
     * LM Head transpose uses ION→ION addresses (known) and can use mbox. */

    fprintf(stderr, "[pool] ION pa=0x%llx va=%p sz=%d MB\n",
            (unsigned long long)p->ion_paddr, p->ion_vaddr, ION_POOL_SZ/1024/1024);
    fprintf(stderr, "[pool] DDR va=%p sz=%d MB (tried %d MB)\n",
            p->ddr_base, p->ddr_sz/1024/1024, DDR_POOL_TRY/1024/1024);
    fprintf(stderr, "[pool] Embed: %d KB total, %d KB DDR + %d KB ION\n",
            p->embed_total/1024, p->embed_ddr_bytes/1024, p->embed_ion_bytes/1024);
    fprintf(stderr, "[pool] ION slots: %d init / %d expanded, DDR overflow: %d, batch: %d\n",
            p->ion_n_slots, ION_MAX_SLOTS, p->ddr_max_overflow, p->batch_target);
    fprintf(stderr, "[pool] DDR staging: %d MB (%d buffers x %.1f MB each)\n",
            ddr_staging_needed/1024/1024, ddr_staging_needed/p->layer_sz,
            p->layer_sz/1024.0/1024.0);
    return 0;
}

/* ---- Load embedding + first 3 layers into pool ---- */
static int pool_load_embed_and_init_layers(pool_t *p) {
    char path[256];
    double ts = TICK();

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
    /* ION portion: bytes [embed_ddr_bytes, embed_total).
     * Re-open to seek past DDR portion + release DDR page cache. */
    if (p->embed_ion_bytes > 0) {
        snprintf(path, sizeof(path), "%s/embed.i8", p->weight_dir);
        int fd = open(path, O_RDONLY);
        if (fd < 0) { fprintf(stderr, "  POOL: cannot reopen embed.i8\n"); return -1; }
        lseek(fd, p->embed_ddr_bytes, SEEK_SET);
        fprintf(stderr, "  Loading embed ION: %d KB...\n", p->embed_ion_bytes/1024);
        int remain = p->embed_ion_bytes;
        uint8_t *dst = p->ion_vaddr;
        while (remain > 0) {
            int n = (remain < 262144) ? remain : 262144;
            if (read(fd, buf, n) != n) {
                fprintf(stderr, "  POOL: embed ION read fail at %d/%d\n",
                        p->embed_ion_bytes - remain, p->embed_ion_bytes);
                close(fd); return -1;
            }
            memcpy(dst, buf, n);
            dst += n; remain -= n;
        }
        close(fd);
        fprintf(stderr, "  Embed ION done.\n");
    }
    fprintf(stderr, "  Embed loaded: %.0f ms (%d KB DDR + %d KB ION)\n",
            (TICK()-ts)/1000.0, p->embed_ddr_bytes/1024, p->embed_ion_bytes/1024);

    /* Load first 3 layers into ION slots — reuse static staging buffer */
    ts = TICK();
    int n_init = 3;
    if (n_init > p->ion_n_slots) n_init = p->ion_n_slots;
    for (int i = 0; i < n_init; i++) {
        snprintf(path, sizeof(path), "%s/layer%d.bin", p->weight_dir, i);
        int fd = open(path, O_RDONLY);
        if (fd < 0) { fprintf(stderr, "  POOL: open layer%d fail\n", i); return -1; }
        int remain = p->layer_sz;
        uint8_t *dst = p->ion_vaddr + p->ion_layer_off + i * p->layer_sz;
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
    return 0;
}

/* ---- Called after embedding lookup + init layers done.
 *   ION+DDR repurposed for layer weight slots + staging buffers.
 *   Preserves ION embed in place to accelerate LM Head chunk reads. ---- */
static void pool_ion_expand_for_batch(pool_t *p) {
    /* Keep ION embed valid — slot count is what fits AFTER embed. */
    p->ion_layer_off = p->embed_ion_bytes;
    int ion_avail = ION_POOL_SZ - p->ion_layer_off;
    p->ion_n_slots = ion_avail / p->layer_sz;
    if (p->ion_n_slots > ION_MAX_SLOTS) p->ion_n_slots = ION_MAX_SLOTS;
    p->ddr_max_overflow = p->ddr_sz / p->layer_sz;
    if (p->ddr_max_overflow > DDR_MAX_OVERFLOW) p->ddr_max_overflow = DDR_MAX_OVERFLOW;
    p->batch_target = p->ion_n_slots + p->ddr_max_overflow;
    for (int i = 0; i < p->ion_n_slots; i++) p->ion_slot_layer[i] = -1;

    /* Repurpose DDR for dual staging buffers:
     *   ddr_buf[0] = ddr_base[0 .. layer_sz-1]
     *   ddr_buf[1] = ddr_base[layer_sz .. 2*layer_sz-1]
     * DDR embed is invalidated but ION embed stays. */
    p->ddr_buf[0] = p->ddr_base;
    p->ddr_buf[1] = p->ddr_base + p->layer_sz;
    p->ddr_buf_loaded[0] = 0;
    p->ddr_buf_loaded[1] = 0;
    p->ddr_cur_buf = 0;
    p->embed_ddr_bytes = 0;  /* DDR embed overwritten by staging */

    if (p->ddr_sz < p->layer_sz * 2) {
        p->ddr_buf[1] = NULL;
        fprintf(stderr, "  POOL: single-buffer staging (DDR %d MB < 2*layer_sz %d MB)\n",
                p->ddr_sz/1024/1024, p->layer_sz*2/1024/1024);
    }
    fprintf(stderr, "  POOL expand: %d ION slots (offset=%d KB), embed_ion=%d KB kept\n",
            p->ion_n_slots, p->ion_layer_off/1024, p->embed_ion_bytes/1024);
}

/* ---- Look up token embedding: DDR portion from memory (permanent),
 *   ION portion from embed fd (streamed to avoid 28 MB reload). ---- */
static void pool_get_embed_row(pool_t *p, int token_id, uint8_t *row_out) {
    int off = token_id * p->D;
    if (off < p->embed_ddr_bytes) {
        memcpy(row_out, p->ddr_base + off, p->D);
    } else {
        /* ION-range token: read from embed fd (seek + read 576 bytes) */
        if (p->embed_fd >= 0) {
            lseek(p->embed_fd, (off_t)off, SEEK_SET);
            int n = read(p->embed_fd, row_out, p->D);
            if (n == p->D) return;
        }
        memset(row_out, 0, p->D);  /* fallback: zero row */
    }
}

/* ---- Load a batch of layers: fill ION slots + DDR overflow ---- */
static int pool_load_batch(pool_t *p, int batch_start, int batch_n) {
    char path[256];
    int n_ion = (batch_n < p->ion_n_slots) ? batch_n : p->ion_n_slots;
    int n_ddr = batch_n - n_ion;
    if (n_ddr > p->ddr_max_overflow) {
        fprintf(stderr, "  POOL: batch too large (%d > ION%d+DDR%d)\n",
                batch_n, p->ion_n_slots, p->ddr_max_overflow);
        return -1;
    }

    for (int i = 0; i < p->ion_n_slots; i++) p->ion_slot_layer[i] = -1;
    p->ddr_n_layers = 0;

    uint8_t *buf = io_buf;

    double ts = TICK();
    for (int i = 0; i < n_ion; i++) {
        int l = batch_start + i;
        snprintf(path, sizeof(path), "%s/layer%d.bin", p->weight_dir, l);
        int fd = open(path, O_RDONLY);
        if (fd < 0) { fprintf(stderr, "  POOL: open layer%d fail\n", l); return -1; }
        int remain = p->layer_sz;
        uint8_t *dst = p->ion_vaddr + p->ion_layer_off + i * p->layer_sz;
        while (remain > 0) {
            int n = (remain < 262144) ? remain : 262144;
            if (read(fd, buf, n) != n) {
                fprintf(stderr, "  POOL: layer%d ION read fail\n", l);
                close(fd); return -1;
            }
            memcpy(dst, buf, n);
            dst += n; remain -= n;
        }
        close(fd);
        p->ion_slot_layer[i] = l;
    }
    for (int i = 0; i < n_ddr; i++) {
        int l = batch_start + n_ion + i;
        snprintf(path, sizeof(path), "%s/layer%d.bin", p->weight_dir, l);
        int fd = open(path, O_RDONLY);
        if (fd < 0) { fprintf(stderr, "  POOL: open layer%d DDR fail\n", l); return -1; }
        int remain = p->layer_sz;
        uint8_t *dst = p->ddr_base + i * p->layer_sz;
        while (remain > 0) {
            int n = (remain < 262144) ? remain : 262144;
            if (read(fd, buf, n) != n) {
                fprintf(stderr, "  POOL: layer%d DDR read fail\n", l);
                close(fd); return -1;
            }
            memcpy(dst, buf, n);
            dst += n; remain -= n;
        }
        close(fd);
        p->ddr_layer_ids[i] = l;
    }
    p->ddr_n_layers = n_ddr;
    p->batch_start = batch_start;
    p->batch_end = batch_start + batch_n;

    fprintf(stderr, "  Batch %d-%d: %d ION + %d DDR loaded, %.0f ms\n",
            batch_start, batch_start+batch_n-1, n_ion, n_ddr, (TICK()-ts)/1000.0);
    return 0;
}

/* ---- Get layer weight ptr.
 *   ION layers (offset < ion_n_slots): direct pointer.
 *   DDR overflow: memcpy into recycled ION slot (freed by earlier layer). ---- */
static void pool_get_layer(pool_t *p, int layer_id, sm_layer_w_t *w) {
    int batch_off = layer_id - p->batch_start;
    if (batch_off < p->ion_n_slots) {
        /* Direct ION access */
        uint8_t *base = p->ion_vaddr + p->ion_layer_off + batch_off * p->layer_sz;
        sm_setup_ptrs(w, base, p->D, p->dkv, p->F);
    } else {
        /* DDR overflow → recycle into ION slot */
        int ddr_idx = batch_off - p->ion_n_slots;

        /* Find a free ION slot. Earlier layers in this batch have already
         * finished computing, so the first `ddr_idx` ION slots are free. */
        int ion_slot = ddr_idx;
        if (ion_slot >= p->ion_n_slots) ion_slot = p->ion_n_slots - 1;

        uint8_t *src = p->ddr_base + ddr_idx * p->layer_sz;
        uint8_t *dst = p->ion_vaddr + p->ion_layer_off + ion_slot * p->layer_sz;
        memcpy(dst, src, p->layer_sz);
        p->ion_slot_layer[ion_slot] = layer_id;
        sm_setup_ptrs(w, dst, p->D, p->dkv, p->F);
    }
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
    pool_t *pool, float *rope_cos, float *rope_sin, sm_timing_t *t)
{
    int D = c->D, V = c->V, L = c->n_layers;
    double ts;

    /* ---- Open embed.i8 for streaming DDR-range + LM Head reads ---- */
    {
        char epath[256];
        snprintf(epath, sizeof(epath), "%s/embed.i8", pool->weight_dir);
        pool->embed_fd = open(epath, O_RDONLY);
        /* fd may be -1; pool_get_embed_row will fallback to zeros for ION-range */
    }

    /* ---- Embedding: from pool (DDR permanent + ION via fd streaming) ---- */
    //fprintf(stderr, "  [dbg]embed start, n_tokens=%d, D=%d\n", n_tokens, D);
    ts = TICK();
    float *x = (float *)malloc(n_tokens * D * sizeof(float));
    if (!x) { return -1; }
    //fprintf(stderr, "  [dbg]x=%p\n", (void*)x);
    int8_t row_i8[576];  /* D=576 fits on stack */
    for (int i = 0; i < n_tokens; i++) {
        int tid = token_ids[i];
        if (tid < 0 || tid >= V) tid = 0;
        pool_get_embed_row(pool, tid, (uint8_t *)row_i8);
        dequantize_f32(x + i * D, row_i8, D, EMBED_SCALE, 0);
    }
    //fprintf(stderr, "  [dbg]embed done\n");
    t->t_embed += TICK() - ts;

    int kv_len = kv_start + n_tokens;
    sm_layer_w_t layer_buf; memset(&layer_buf, 0, sizeof(layer_buf));

    /* Always reload ALL layers from SD each forward pass.
     * Init-layer preloading is only valid for the very first call;
     * after expand+batch, ION slots are recycled and init data is gone. */
    int cur_layer = 0;
    int n_init = 0;

    /* Init layers done (none after first pass), embedding ION portion no longer needed. */
    //fprintf(stderr, "  [dbg]before expand_for_batch\n");
    pool_ion_expand_for_batch(pool);
    //fprintf(stderr, "  [dbg]expand done, ion_n_slots=%d, layer_sz=%d\n", pool->ion_n_slots, pool->layer_sz);

    /* ---- Dual-buffer pipelined layer loading.
     *   DDR→ION uses local memcpy (pagemap unavailable for phys addr).
     *   With dual buffers: SD read next layer while DDR→ION copies current.
     *   Single buffer fallback: serial SD→DDR→ION. ---- */
    int ion_n_slots = pool->ion_n_slots;
    int ddr_has_dual = (pool->ddr_buf[1] != NULL);
    uint64_t ion_pa = pool->ion_paddr;
    uint64_t ddr_pa_buf0 = pool->ddr_buf_pa[0];
    uint64_t ddr_pa_buf1 = pool->ddr_buf_pa[1];

    /* Helper: read one layer from SD into a DDR buffer */
    #define SD_READ_LAYER(lid, dst) do { \
        char _path[256]; \
        snprintf(_path, sizeof(_path), "%s/layer%d.bin", pool->weight_dir, lid); \
        int _fd = open(_path, O_RDONLY); \
        if (_fd < 0) { free(x); return -1; } \
        int _remain = pool->layer_sz; \
        uint8_t *_d = dst; \
        while (_remain > 0) { \
            int _n = (_remain < 262144) ? _remain : 262144; \
            int _rn = read(_fd, io_buf, _n); \
            if (_rn != _n) { close(_fd); free(x); return -1; } \
            memcpy(_d, io_buf, _n); _d += _n; _remain -= _n; \
        } \
        close(_fd); \
    } while(0)

    /* Helper: copy DDR buffer -> ION slot and flush cache */
    #define DDR_TO_ION(ddr_va, ion_off) do { \
        uint8_t *_ion_dst = pool->ion_vaddr + pool->ion_layer_off + ion_off; \
        memcpy(_ion_dst, ddr_va, pool->layer_sz); \
        CVI_RT_MemFlush(ctx->rt_handle, pool->ion_mem); \
    } while(0)

    /* ---- Layer 0: no pipeline, load directly ---- */
    int prev_buf = 0;
    SD_READ_LAYER(0, pool->ddr_buf[0]);
    DDR_TO_ION(pool->ddr_buf[0], 0);
    pool->ion_slot_layer[0] = 0;

    {
        uint8_t *base = pool->ion_vaddr + pool->ion_layer_off;
        sm_setup_ptrs(&layer_buf, base, pool->D, pool->dkv, pool->F);
        double lt[14] = {0};
        int rc = sm_layer_forward(ctx, cvk, nm, c, &layer_buf,
                                  x, n_tokens, kv_start, kv_len,
                                  kv, 0, rope_cos, rope_sin, lt);
        if (rc) { free(x); return rc; }
        t->t_rms_attn += lt[0];  t->t_q      += lt[1];
        t->t_rope     += lt[4];  t->t_kv     += lt[5];
        t->t_scores   += lt[6];  t->t_softmax+= lt[7];
        t->t_attn     += lt[8];  t->t_wo     += lt[9];
        t->t_rms_ffn  += lt[10]; t->t_ffn_up += lt[11];
        t->t_ffn_down += lt[13];
    }
    cur_layer = 1;

    /* Pre-read layer 1 for pipeline start.
     * Dual-buffer: layer 1 goes to buf[1] (buf[0] holds layer 0 data).
     * Single-buffer: layer 1 goes to buf[0] (overwrites stale layer 0 data). */
    if (L > 1) {
        int first_buf = ddr_has_dual ? 1 : 0;
        SD_READ_LAYER(1, pool->ddr_buf[first_buf]);
        pool->ddr_buf_loaded[first_buf] = 1;
        if (!ddr_has_dual) prev_buf = 0;  /* single-buffer: buf[0] has layer 1 */
    }

    /* ---- Layers 1..L-1: pipelined ---- */
    while (cur_layer < L) {
        int cur_buf = ddr_has_dual ? (1 - prev_buf) : 0;  /* pre-filled buf */
        int next_buf = prev_buf;  /* prev layer's buf, now free for next SD read */
        int ion_slot = cur_layer % ion_n_slots;
        int next_layer = cur_layer + 1;

        ts = TICK();

        /* Dual-buffer: kick off SD read of NEXT layer while we DDR→ION current.
         * The SD read goes into next_buf (the buffer used for the PREVIOUS layer,
         * which is now free). This overlaps SD I/O with DDR→ION memcpy.
         * Single-buffer: SD read happens AFTER DDR→ION (below). */
        if (ddr_has_dual && next_layer < L) {
            SD_READ_LAYER(next_layer, pool->ddr_buf[next_buf]);
            pool->ddr_buf_loaded[next_buf] = 1;
        }

        /* DDR → ION for current layer.
         * Prefer secondary core async copy; fall back to local memcpy. */
        int use_async = (pool->use_mbox && ddr_pa_buf0 > 0);
        int mbox_slot = cur_layer % 4;  /* rotate through 4 desc slots */
        uint64_t ddr_pa = (cur_buf == 0) ? ddr_pa_buf0 : ddr_pa_buf1;
        if (use_async && ddr_pa > 0) {
            uint64_t ion_off_pa = ion_pa + pool->ion_layer_off + ion_slot * pool->layer_sz;
            if (mbox_ddr_to_ion_async(ctx, mbox_slot, ddr_pa, ion_off_pa,
                                       pool->layer_sz) == 0) {
                uint8_t *nm_ptr = ctx->neuron_vaddr;
                uint64_t nm_pa = CVI_RT_MemGetPAddr(ctx->neuron_mem);
                mha_dma_desc_t *desc = mbox_desc_ptr(nm_ptr, nm_pa, mbox_slot);
                if (mbox_poll_desc(ctx, desc, MBOX_TIMEOUT_US) != 0) {
                    fprintf(stderr, "  MBOX: layer %d DDR→ION timeout, local fallback\n",
                            cur_layer);
                    DDR_TO_ION(pool->ddr_buf[cur_buf],
                               ion_slot * pool->layer_sz);
                }
            } else {
                DDR_TO_ION(pool->ddr_buf[cur_buf],
                           ion_slot * pool->layer_sz);
            }
        } else {
            DDR_TO_ION(pool->ddr_buf[cur_buf],
                       ion_slot * pool->layer_sz);
        }
        pool->ion_slot_layer[ion_slot] = cur_layer;
        t->t_weight_load += TICK() - ts;

        /* Single buffer: SD read next layer after DDR→ION is done */
        if (!ddr_has_dual && next_layer < L) {
            SD_READ_LAYER(next_layer, pool->ddr_buf[0]);
        }

        /* TPU compute on current layer */
        {
            uint8_t *base = pool->ion_vaddr + pool->ion_layer_off + ion_slot * pool->layer_sz;
            sm_setup_ptrs(&layer_buf, base, pool->D, pool->dkv, pool->F);

            double lt[14] = {0};
            int rc = sm_layer_forward(ctx, cvk, nm, c, &layer_buf,
                                      x, n_tokens, kv_start, kv_len,
                                      kv, cur_layer, rope_cos, rope_sin, lt);
            if (rc) { free(x); return rc; }
            t->t_rms_attn += lt[0];  t->t_q      += lt[1];
            t->t_rope     += lt[4];  t->t_kv     += lt[5];
            t->t_scores   += lt[6];  t->t_softmax+= lt[7];
            t->t_attn     += lt[8];  t->t_wo     += lt[9];
            t->t_rms_ffn  += lt[10]; t->t_ffn_up += lt[11];
            t->t_ffn_down += lt[13];
        }
        prev_buf = cur_buf;
        cur_layer++;
    }

    #undef SD_READ_LAYER
    #undef DDR_TO_ION

    /* ---- LM Head: stream embed chunks from SD, transpose via secondary core
     *   (or CPU fallback).  Double-buffered in ION pool memory. ---- */
    /* embed_fd was already opened at start of this function. */
    ts = TICK();
    float *final_normed = (float *)malloc(n_tokens * D * sizeof(float));
    float *final_rms = (float *)malloc(D * sizeof(float));
    char path[256];
    snprintf(path, sizeof(path), "%s/final_rms.f32", weight_dir);
    read_file(path, final_rms, D * 4);
    rms_norm_f32(final_normed, x, final_rms, n_tokens, D, 1e-6f);
    free(final_rms);
    t->t_final_rms += TICK() - ts;

    ts = TICK();
    int CHUNK = 1024;
    float sc_final = compute_scale_sym(final_normed, n_tokens * D);
    int8_t *x_final_i8 = (int8_t *)malloc(n_tokens * D);
    quantize_i8_sym(x_final_i8, final_normed, n_tokens * D, sc_final);
    free(final_normed);

    int rshift_lm = matmul_rshift(D);

    /* Double-buffer pipeline: while TPU computes chunk i, pre-read chunk i+1
     * from SD (or use ION in-memory embed for cached tokens).
     * Buffers: 4 regions in ION pool (2 src + 2 dst for double-buffering). */
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

    /* Number of embed tokens already in ION memory */
    int embed_ion_tokens = pool->embed_ion_bytes / D;

    int rc_lm, cur = 0;
    uint32_t lm_result_off = SM_S_I8_OFF;

    /* Pre-load first chunk */
    {
        int v_start = 0;
        int cur_v = (v_start + CHUNK <= V) ? CHUNK : V - v_start;
        int total = D * cur_v;
        if (v_start < embed_ion_tokens) {
            int mem_v = cur_v;
            if (v_start + cur_v > embed_ion_tokens)
                mem_v = embed_ion_tokens - v_start;
            int mem_bytes = D * mem_v;
            memcpy(xpose_src[cur], pool->ion_vaddr + v_start * D, mem_bytes);
            if (mem_v < cur_v) {
                off_t sd_off = (off_t)embed_ion_tokens * D;
                int sd_bytes = D * (cur_v - mem_v);
                lseek(pool->embed_fd, sd_off, SEEK_SET);
                read(pool->embed_fd, xpose_src[cur] + mem_bytes, sd_bytes);
            }
        } else {
            lseek(pool->embed_fd, (off_t)v_start * D, SEEK_SET);
            read(pool->embed_fd, xpose_src[cur], total);
        }
        CVI_RT_MemFlush(ctx->rt_handle, pool->ion_mem);
    }

    for (int v_start = 0; v_start < V; v_start += CHUNK) {
        int cur_v = (v_start + CHUNK <= V) ? CHUNK : V - v_start;
        int nxt = 1 - cur;  /* next buffer index */
        int nxt_v_start = v_start + CHUNK;

        /* Start pre-reading next chunk from SD into nxt buffer */
        if (nxt_v_start < V) {
            int nxt_cur_v = (nxt_v_start + CHUNK <= V) ? CHUNK : V - nxt_v_start;
            int total = D * nxt_cur_v;
            if (nxt_v_start < embed_ion_tokens) {
                int mem_v = nxt_cur_v;
                if (nxt_v_start + nxt_cur_v > embed_ion_tokens)
                    mem_v = embed_ion_tokens - nxt_v_start;
                int mem_bytes = D * mem_v;
                memcpy(xpose_src[nxt], pool->ion_vaddr + nxt_v_start * D, mem_bytes);
                if (mem_v < nxt_cur_v) {
                    off_t sd_off = (off_t)embed_ion_tokens * D;
                    int sd_bytes = D * (nxt_cur_v - mem_v);
                    lseek(pool->embed_fd, sd_off, SEEK_SET);
                    read(pool->embed_fd, xpose_src[nxt] + mem_bytes, sd_bytes);
                }
            } else {
                lseek(pool->embed_fd, (off_t)nxt_v_start * D, SEEK_SET);
                read(pool->embed_fd, xpose_src[nxt], total);
            }
            /* Don't flush yet — will flush after transpose */
        }

        /* Transpose current chunk: secondary core (async) or CPU. */
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
            if (rc != 0) {
                lm_use_mbox = 0;
                for (int j = 0; j < D; j++)
                    for (int v = 0; v < cur_v; v++)
                        ((int8_t *)xpose_dst[cur])[j * cur_v + v] = (int8_t)xpose_src[cur][v * D + j];
                CVI_RT_MemFlush(ctx->rt_handle, pool->ion_mem);
            }
        } else {
            for (int j = 0; j < D; j++)
                for (int v = 0; v < cur_v; v++)
                    ((int8_t *)xpose_dst[cur])[j * cur_v + v] = (int8_t)xpose_src[cur][v * D + j];
            CVI_RT_MemFlush(ctx->rt_handle, pool->ion_mem);
        }

        /* TPU matmul */
        rc_lm = tpu_matmul_build(ctx, cvk, x_final_i8, n_tokens, D,
                                  xpose_dst[cur], cur_v,
                                  lm_result_off, SM_SCRATCH_OFF, rshift_lm);
        if (!rc_lm) {
            CVI_RT_Submit(ctx->rt_khandle);
            CVI_RT_MemInvld(ctx->rt_handle, ctx->neuron_mem);
        }
        if (rc_lm) { free(x_final_i8); free(logits_out); return -1; }

        /* Dequantize logits */
        int8_t *logits_i8 = (int8_t *)(nm + lm_result_off);
        for (int t = 0; t < n_tokens; t++)
            dequant_i8(logits_out + t * V + v_start,
                       logits_i8 + t * cur_v, cur_v,
                       EMBED_SCALE, sc_final, rshift_lm);

        /* Flush next buffer's source data before transpose starts */
        if (nxt_v_start < V) {
            CVI_RT_MemFlush(ctx->rt_handle, pool->ion_mem);
        }
        cur = nxt;
    }
    free(x_final_i8);
    t->t_lm_head += TICK() - ts;

    /* Close embed fd (opened at start of this function) */
    if (pool->embed_fd >= 0) { close(pool->embed_fd); pool->embed_fd = -1; }

    t->n_steps++;
    free(x);
    return 0;
}

/* ================================================================
 *  Main
 * ================================================================ */
int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <weight_dir> <token_ids.bin> <max_new_tokens>\n", argv[0]);
        return 1;
    }
    const char *weight_dir = argv[1], *token_file = argv[2];
    int max_new = atoi(argv[3]);
    setvbuf(stderr, NULL, _IONBF, 0);
    fprintf(stderr, "=== smollm2_pool_demo START ===\n");

    sm_cfg_t c;
    fprintf(stderr, "Reading config...\n");
    if (sm_read_config(weight_dir, &c) != 0) {
        fprintf(stderr, "ERROR: cannot read config.bin\n"); return 1;
    }
    int D=c.D, d=c.head_dim, F=c.FFN, V=c.V, max_seq=c.max_seq;

    fprintf(stderr, "SmolLM2-135M POOL: D=%d H=%d Kvh=%d d=%d L=%d F=%d V=%d max_seq=%d\n",
            D, c.n_heads, c.n_kv_heads, d, c.n_layers, F, V, max_seq);

    { char path[256]; snprintf(path,sizeof(path),"%s/scales.bin",weight_dir);
      g_scales=(float*)malloc(212*sizeof(float));
      if(read_file(path,g_scales,212*4)!=0){free(g_scales);g_scales=NULL;} }

    struct stat st; stat(token_file, &st);
    int prompt_len=st.st_size/4; if(prompt_len>max_seq)prompt_len=max_seq;
    int *token_ids=(int*)malloc(prompt_len*sizeof(int));
    if(read_file(token_file,token_ids,prompt_len*4)!=0){fprintf(stderr,"ERROR: cannot read %s\n",token_file);return 1;}
    fprintf(stderr,"\n[Init] Prompt: %d tokens, max_new: %d\n",prompt_len,max_new);

    tpu_ctx ctx;
    if(tpu_init(&ctx,NEURON_SZ)!=0){fprintf(stderr,"TPU init failed!\n");return 1;}
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

    /* Shrink max_seq to actual needed length — saves ~2.5 MB KV cache DDR */
    int kv_needed = prompt_len + max_new;
    if (c.max_seq > kv_needed) { c.max_seq = kv_needed; max_seq = kv_needed; }
    fprintf(stderr, "[kv] max_seq=%d, ~%d KB\n", max_seq,
            (int)(c.n_layers * 2 * max_seq * c.d_qkv * sizeof(float) / 1024));

    float *rope_cos=(float*)malloc(max_seq*(d/2)*sizeof(float));
    float *rope_sin=(float*)malloc(max_seq*(d/2)*sizeof(float));
    rope_precompute(max_seq,d,rope_cos,rope_sin);
    sm_kv_cache_t *kv=sm_kv_alloc(ctx.rt_handle,&c);
    if(!kv){fprintf(stderr,"KV cache alloc failed!\n");return 1;}

    /* ---- Prefill (batch all tokens in one forward pass) ---- */
    fprintf(stderr,"\n[Prefill] %d tokens (batch)...\n",prompt_len);
    sm_timing_t t; memset(&t,0,sizeof(t));
    double t_prefill=TICK();
    float *all_logits=(float*)malloc(prompt_len * V * sizeof(float));
    if(!all_logits){fprintf(stderr,"OOM for prefill logits\n");return 1;}
    int rc=sm_forward_pool(&ctx,cvk,nm,&c,weight_dir,token_ids,prompt_len,0,kv,all_logits,&pool,rope_cos,rope_sin,&t);
    if(rc){fprintf(stderr,"Prefill failed rc=%d\n",rc);free(all_logits);return 1;}
    t_prefill=TICK()-t_prefill;
    int next_token=sample_argmax(all_logits+(prompt_len-1)*V, V);
    free(all_logits);
    fprintf(stderr,"  Prefill: %.0f ms, next_token=%d\n",t_prefill/1000.0,next_token);

    /* ---- Decode ---- */
    fprintf(stderr,"\n[Decode] %d tokens...\n",max_new);
    int generated[256],n_gen=0; generated[n_gen++]=next_token;
    double t_decode_total=0; int kv_len=prompt_len;

    for(int step=0;step<max_new;step++){
        int tid[1]={next_token};
        float *step_logits=(float*)malloc(V*sizeof(float));
        double t_step=TICK();
        int rc=sm_forward_pool(&ctx,cvk,nm,&c,weight_dir,tid,1,kv_len,kv,step_logits,&pool,rope_cos,rope_sin,&t);
        t_step=TICK()-t_step; t_decode_total+=t_step; kv_len++;
        if(rc){fprintf(stderr,"Decode step %d failed rc=%d\n",step,rc);break;}
        next_token=sample_argmax(step_logits,V); free(step_logits);
        if(next_token<=0||next_token>=V)break;
        generated[n_gen++]=next_token;
        if((step+1)%5==0)fprintf(stderr,"  step %d: tok=%d, %.0f ms (avg %.0f ms/tok)\n",
            step+1,next_token,t_step/1000.0,t_decode_total/1000.0/(step+1));
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
    fprintf(stderr,"\n--- Tokens ---\n");
    for(int i=0;i<n_gen;i++)fprintf(stderr,"%d ",generated[i]); fprintf(stderr,"\n");

    free(rope_cos);free(rope_sin);free(token_ids);
    if(g_scales)free(g_scales);
    sm_kv_free(kv,&c,ctx.rt_handle); pool_free(&pool,ctx.rt_handle);
    tpu_cleanup(&ctx);
    return 0;
}
