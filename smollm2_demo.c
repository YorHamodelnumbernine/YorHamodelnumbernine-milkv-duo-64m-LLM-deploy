/* smollm2_demo.c — SmolLM2-135M TPU-accelerated inference on CV1800B.
   GQA (9Q/3KV heads), 30 layers, d_model=576, FFN=1536, vocab=49152.
   Weights streamed per-layer from SD card.  KV cache in heap.

   Build: make smollm2_demo
   Run:   ./smollm2_demo <token_ids.bin> <max_new_tokens>
          where token_ids.bin is raw int32 array from host tokenizer
*/
#include "common/tpu_bench.h"
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

#define TICK() ({ struct timespec _ts; clock_gettime(CLOCK_MONOTONIC, &_ts); \
                  _ts.tv_sec * 1e6 + _ts.tv_nsec / 1e3; })

/* ================================================================
 *  Runtime model config (read from config.bin)
 * ================================================================ */
typedef struct {
    int D, n_heads, n_kv_heads, head_dim, n_layers, FFN, V, max_seq;
    int d_qkv;    /* n_kv_heads * head_dim */
    int n_groups; /* n_heads / n_kv_heads */
} sm_cfg_t;

/* Per-tensor weight quantization scales, loaded from scales.bin */
static float *g_scales = NULL;
#define EMBED_SCALE       (g_scales ? g_scales[0] : 0.01544f)
#define W_SCALE(l, idx)   (g_scales ? g_scales[1 + (l)*7 + (idx)] : 0.001f)
/* idx: 0=Wq 1=Wk 2=Wv 3=Wo 4=ffn_up 5=ffn_gate 6=ffn_down */

static int sm_read_config(const char *base, sm_cfg_t *c) {
    char path[256];
    snprintf(path, sizeof(path), "%s/config.bin", base);
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    int cfg[8];
    if (read(fd, cfg, sizeof(cfg)) != sizeof(cfg)) { close(fd); return -1; }
    close(fd);
    c->D          = cfg[0];
    c->n_heads    = cfg[1];
    c->n_kv_heads = cfg[2];
    c->head_dim   = cfg[3];
    c->n_layers   = cfg[4];
    c->FFN        = cfg[5];
    c->V          = cfg[6];
    c->max_seq    = cfg[7];
    c->d_qkv      = c->n_kv_heads * c->head_dim;
    c->n_groups   = c->n_heads / c->n_kv_heads;
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
                         int n, float eps) {
    float sum_sq = 0;
    for (int i = 0; i < n; i++) { float v = x[i]; sum_sq += v * v; }
    float inv = 1.0f / sqrtf(sum_sq / (float)n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * inv * gamma[i];
}

static void silu_f32(float *x, int n) {
    for (int i = 0; i < n; i++) {
        float v = x[i];
        x[i] = v / (1.0f + expf(-v));
    }
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
    const float *c = cos_tab + pos * half;
    const float *s = sin_tab + pos * half;
    for (int i = 0; i < half; i++) {
        float x = q_or_k[i], y = q_or_k[i + half];
        q_or_k[i]        = x * c[i] - y * s[i];
        q_or_k[i + half] = x * s[i] + y * c[i];
    }
}

static void softmax_f32(float *buf, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        float *row = buf + r * cols;
        float maxv = row[0];
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
static float compute_scale(const float *data, int n, int *zp_out) {
    float min = data[0], max = data[0];
    for (int i = 1; i < n; i++) {
        if (data[i] < min) min = data[i];
        if (data[i] > max) max = data[i];
    }
    float s = (max - min) / 255.0f;
    if (s < 1e-10f) s = 1.0f;
    *zp_out = (int)(-128.0f - min / s);
    if (*zp_out < -128) *zp_out = -128;
    if (*zp_out > 127)  *zp_out = 127;
    return s;
}

static void quantize_i8(int8_t *dst, const float *src, int n, float sc, int zp) {
    float inv = 1.0f / sc;
    for (int i = 0; i < n; i++) {
        int q = (int)(src[i] * inv + (float)zp + 0.5f);
        if (q > 127) q = 127; else if (q < -128) q = -128;
        dst[i] = (int8_t)q;
    }
}

static void dequantize_f32(float *dst, const int8_t *src, int n, float sc, int zp) {
    for (int i = 0; i < n; i++) dst[i] = ((float)(int)src[i] - (float)zp) * sc;
}

/* Symmetric INT8 quantization (zp=0, scale = absmax / 127) */
static float compute_scale_sym(const float *data, int n) {
    float absmax = 0;
    for (int i = 0; i < n; i++) {
        float v = fabsf(data[i]);
        if (v > absmax) absmax = v;
    }
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

/* ================================================================
 *  TPU Matmul (from transformer_demo.c)
 *  M,K,N are logical dims. On Duo M,K,N each must be <= 256.
 *  For larger dims, we tile on N dimension.
 * ================================================================ */
static inline int lmem_matrix_bytes(int rows, int cols) {
    int c = (rows + 1) / 2, w = (cols + 31) / 32;
    return c * w * 32;
}

static int tpu_find_tile_n(cvk_context_t *cvk, int M, int K) {
    int left = lmem_matrix_bytes(M, K);
    for (int tn = 256; tn >= 16; tn -= 16) {
        if (left + lmem_matrix_bytes(K, tn) + lmem_matrix_bytes(M, tn) <= 32768) return tn;
    }
    return -1;
}

/* Compute rshift to avoid INT8 saturation: keep max dot product under 127.
   max_dot = K * 127 * 127; rshift = ceil(log2(max_dot / 127)) = ceil(log2(K*127))
   e.g. K=576 → rshift=17, K=1536 → rshift=18, K=64 → rshift=13.
   With rshift, dequant is: output_f32 = output_i8 * sc_in * sc_wt * 2^rshift */
static inline int matmul_rshift(int K) {
    int r = 0;
    int max_dot = K * 127 * 127;
    while ((max_dot >> r) > 127) r++;
    return r;
}

static int tpu_matmul(tpu_ctx *ctx, cvk_context_t *cvk,
    const void *left, int M, int K, const void *right, int N,
    void *result, uint32_t scratch_off, int rshift)
{
    const int8_t *l_i8 = (const int8_t *)left;
    const int8_t *r_i8 = (const int8_t *)right;
    int8_t       *o_i8 = (int8_t *)result;
    uint8_t *nm = ctx->neuron_vaddr;
    uint32_t off_l = scratch_off, off_r = scratch_off + M * K;

    int need_tile = (lmem_matrix_bytes(M, K) +
                     lmem_matrix_bytes(K, N) +
                     lmem_matrix_bytes(M, N) > 32768);
    if (!need_tile) {
        uint32_t off_o = scratch_off + M * K + K * N;
        memcpy(nm + off_l, l_i8, M * K);
        memcpy(nm + off_r, r_i8, K * N);
        CVI_RT_MemFlush(ctx->rt_handle, ctx->neuron_mem);
        cvk_ml_t *ml_l = cvk->ops->lmem_alloc_matrix(cvk,
            cvk->ops->ml_default_shape(cvk, M, K, CVK_FMT_I8), CVK_FMT_I8, 1);
        cvk_ml_t *ml_r = cvk->ops->lmem_alloc_matrix(cvk,
            cvk->ops->ml_default_shape(cvk, K, N, CVK_FMT_I8), CVK_FMT_I8, 1);
        cvk_ml_t *ml_o = cvk->ops->lmem_alloc_matrix(cvk,
            cvk->ops->ml_default_shape(cvk, M, N, CVK_FMT_I8), CVK_FMT_I8, 1);
        cvk->ops->tdma_g2l_matrix_copy(cvk, &(cvk_tdma_g2l_matrix_copy_param_t){
            .src = &(cvk_mg_t){0, TPU_PA(ctx, off_l), CVK_FMT_I8, {M, K}, {K}}, .dst = ml_l});
        cvk->ops->tdma_g2l_matrix_copy(cvk, &(cvk_tdma_g2l_matrix_copy_param_t){
            .src = &(cvk_mg_t){0, TPU_PA(ctx, off_r), CVK_FMT_I8, {K, N}, {N}}, .dst = ml_r});
        cvk->ops->tiu_matrix_multiplication(cvk, &(cvk_tiu_matrix_multiplication_param_t){
            .res = ml_o, .left = ml_l, .right = ml_r, .bias = NULL,
            .lshift_bits = 0, .rshift_bits = rshift, .res_is_int8 = 1,
            .relu_enable = 0, .add_result = 0, .ps32_mode = 0,
        });
        cvk->ops->tdma_l2g_matrix_copy(cvk, &(cvk_tdma_l2g_matrix_copy_param_t){
            .src = ml_o,
            .dst = &(cvk_mg_t){0, TPU_PA(ctx, off_o), CVK_FMT_I8, {M, N}, {N}}});
        cvk->ops->lmem_free_matrix(cvk, ml_o);
        cvk->ops->lmem_free_matrix(cvk, ml_r);
        cvk->ops->lmem_free_matrix(cvk, ml_l);
        CVI_RT_Submit(ctx->rt_khandle);
        CVI_RT_MemInvld(ctx->rt_handle, ctx->neuron_mem);
        memcpy(o_i8, nm + off_o, M * N);
        return 0;
    }

    int tile_n = tpu_find_tile_n(cvk, M, K);
    if (tile_n < 16) return -1;
    uintptr_t nm_base = (uintptr_t)nm;
    uintptr_t nm_end  = nm_base + ctx->neuron_size;
    int r_is_nm = ((uintptr_t)r_i8 >= nm_base && (uintptr_t)r_i8 < nm_end);
    uint32_t r_nm_off = r_is_nm ? (uint32_t)((uintptr_t)r_i8 - nm_base) : 0;
    uint32_t off_o_base = scratch_off + M * K + K * tile_n;

    memcpy(nm + off_l, l_i8, M * K);
    CVI_RT_MemFlush(ctx->rt_handle, ctx->neuron_mem);
    for (int n_start = 0; n_start < N; n_start += tile_n) {
        int cur_n = (n_start + tile_n <= N) ? tile_n : N - n_start;
        if (!r_is_nm) {
            uint8_t *td = nm + off_r;
            for (int row = 0; row < K; row++)
                memcpy(td + row * cur_n, r_i8 + row * N + n_start, cur_n);
            CVI_RT_MemFlush(ctx->rt_handle, ctx->neuron_mem);
        }
        cvk_ml_t *ml_l = cvk->ops->lmem_alloc_matrix(cvk,
            cvk->ops->ml_default_shape(cvk, M, K, CVK_FMT_I8), CVK_FMT_I8, 1);
        cvk_ml_t *ml_r = cvk->ops->lmem_alloc_matrix(cvk,
            cvk->ops->ml_default_shape(cvk, K, cur_n, CVK_FMT_I8), CVK_FMT_I8, 1);
        cvk_ml_t *ml_o = cvk->ops->lmem_alloc_matrix(cvk,
            cvk->ops->ml_default_shape(cvk, M, cur_n, CVK_FMT_I8), CVK_FMT_I8, 1);
        cvk->ops->tdma_g2l_matrix_copy(cvk, &(cvk_tdma_g2l_matrix_copy_param_t){
            .src = &(cvk_mg_t){0, TPU_PA(ctx, off_l), CVK_FMT_I8, {M, K}, {K}}, .dst = ml_l});
        cvk_mg_t src_r = r_is_nm
            ? (cvk_mg_t){0, TPU_PA(ctx, r_nm_off + n_start), CVK_FMT_I8, {K, cur_n}, {N}}
            : (cvk_mg_t){0, TPU_PA(ctx, off_r), CVK_FMT_I8, {K, cur_n}, {cur_n}};
        cvk->ops->tdma_g2l_matrix_copy(cvk, &(cvk_tdma_g2l_matrix_copy_param_t){
            .src = &src_r, .dst = ml_r});
        cvk->ops->tiu_matrix_multiplication(cvk, &(cvk_tiu_matrix_multiplication_param_t){
            .res = ml_o, .left = ml_l, .right = ml_r, .bias = NULL,
            .lshift_bits = 0, .rshift_bits = rshift, .res_is_int8 = 1,
            .relu_enable = 0, .add_result = 0, .ps32_mode = 0,
        });
        cvk->ops->tdma_l2g_matrix_copy(cvk, &(cvk_tdma_l2g_matrix_copy_param_t){
            .src = ml_o,
            .dst = &(cvk_mg_t){0, TPU_PA(ctx, off_o_base + n_start),
                               CVK_FMT_I8, {M, cur_n}, {N}}});
        cvk->ops->lmem_free_matrix(cvk, ml_o);
        cvk->ops->lmem_free_matrix(cvk, ml_r);
        cvk->ops->lmem_free_matrix(cvk, ml_l);
    }
    CVI_RT_Submit(ctx->rt_khandle);
    CVI_RT_MemInvld(ctx->rt_handle, ctx->neuron_mem);
    memcpy(o_i8, nm + off_o_base, M * N);
    return 0;
}

/* tpu_matmul_build: build matmul cmds into cmdbuf WITHOUT Submit/Invld/memcpy.
   Result stays in neuron memory at result_off. Caller batches multiple builds
   then calls CVI_RT_Submit + CVI_RT_MemInvld + memcpy once. */
static int tpu_matmul_build(tpu_ctx *ctx, cvk_context_t *cvk,
    const void *left, int M, int K, const void *right, int N,
    uint32_t result_off, uint32_t scratch_off, int rshift)
{
    const int8_t *l_i8 = (const int8_t *)left;
    const int8_t *r_i8 = (const int8_t *)right;
    uint8_t *nm = ctx->neuron_vaddr;
    uint32_t off_l = scratch_off, off_r = scratch_off + M * K;

    int need_tile = (lmem_matrix_bytes(M, K) +
                     lmem_matrix_bytes(K, N) +
                     lmem_matrix_bytes(M, N) > 32768);
    if (!need_tile) {
        memcpy(nm + off_l, l_i8, M * K);
        memcpy(nm + off_r, r_i8, K * N);
        CVI_RT_MemFlush(ctx->rt_handle, ctx->neuron_mem);
        cvk_ml_t *ml_l = cvk->ops->lmem_alloc_matrix(cvk,
            cvk->ops->ml_default_shape(cvk, M, K, CVK_FMT_I8), CVK_FMT_I8, 1);
        cvk_ml_t *ml_r = cvk->ops->lmem_alloc_matrix(cvk,
            cvk->ops->ml_default_shape(cvk, K, N, CVK_FMT_I8), CVK_FMT_I8, 1);
        cvk_ml_t *ml_o = cvk->ops->lmem_alloc_matrix(cvk,
            cvk->ops->ml_default_shape(cvk, M, N, CVK_FMT_I8), CVK_FMT_I8, 1);
        cvk->ops->tdma_g2l_matrix_copy(cvk, &(cvk_tdma_g2l_matrix_copy_param_t){
            .src = &(cvk_mg_t){0, TPU_PA(ctx, off_l), CVK_FMT_I8, {M, K}, {K}}, .dst = ml_l});
        cvk->ops->tdma_g2l_matrix_copy(cvk, &(cvk_tdma_g2l_matrix_copy_param_t){
            .src = &(cvk_mg_t){0, TPU_PA(ctx, off_r), CVK_FMT_I8, {K, N}, {N}}, .dst = ml_r});
        cvk->ops->tiu_matrix_multiplication(cvk, &(cvk_tiu_matrix_multiplication_param_t){
            .res = ml_o, .left = ml_l, .right = ml_r, .bias = NULL,
            .lshift_bits = 0, .rshift_bits = rshift, .res_is_int8 = 1,
            .relu_enable = 0, .add_result = 0, .ps32_mode = 0,
        });
        cvk->ops->tdma_l2g_matrix_copy(cvk, &(cvk_tdma_l2g_matrix_copy_param_t){
            .src = ml_o,
            .dst = &(cvk_mg_t){0, TPU_PA(ctx, result_off), CVK_FMT_I8, {M, N}, {N}}});
        cvk->ops->lmem_free_matrix(cvk, ml_o);
        cvk->ops->lmem_free_matrix(cvk, ml_r);
        cvk->ops->lmem_free_matrix(cvk, ml_l);
        return 0;
    }

    int tile_n = tpu_find_tile_n(cvk, M, K);
    if (tile_n < 16) return -1;
    uintptr_t nm_base = (uintptr_t)nm;
    uintptr_t nm_end  = nm_base + ctx->neuron_size;
    int r_is_nm = ((uintptr_t)r_i8 >= nm_base && (uintptr_t)r_i8 < nm_end);
    uint32_t r_nm_off = r_is_nm ? (uint32_t)((uintptr_t)r_i8 - nm_base) : 0;

    memcpy(nm + off_l, l_i8, M * K);
    CVI_RT_MemFlush(ctx->rt_handle, ctx->neuron_mem);
    for (int n_start = 0; n_start < N; n_start += tile_n) {
        int cur_n = (n_start + tile_n <= N) ? tile_n : N - n_start;
        if (!r_is_nm) {
            uint8_t *td = nm + off_r;
            for (int row = 0; row < K; row++)
                memcpy(td + row * cur_n, r_i8 + row * N + n_start, cur_n);
            CVI_RT_MemFlush(ctx->rt_handle, ctx->neuron_mem);
        }
        cvk_ml_t *ml_l = cvk->ops->lmem_alloc_matrix(cvk,
            cvk->ops->ml_default_shape(cvk, M, K, CVK_FMT_I8), CVK_FMT_I8, 1);
        cvk_ml_t *ml_r = cvk->ops->lmem_alloc_matrix(cvk,
            cvk->ops->ml_default_shape(cvk, K, cur_n, CVK_FMT_I8), CVK_FMT_I8, 1);
        cvk_ml_t *ml_o = cvk->ops->lmem_alloc_matrix(cvk,
            cvk->ops->ml_default_shape(cvk, M, cur_n, CVK_FMT_I8), CVK_FMT_I8, 1);
        cvk->ops->tdma_g2l_matrix_copy(cvk, &(cvk_tdma_g2l_matrix_copy_param_t){
            .src = &(cvk_mg_t){0, TPU_PA(ctx, off_l), CVK_FMT_I8, {M, K}, {K}}, .dst = ml_l});
        cvk_mg_t src_r = r_is_nm
            ? (cvk_mg_t){0, TPU_PA(ctx, r_nm_off + n_start), CVK_FMT_I8, {K, cur_n}, {N}}
            : (cvk_mg_t){0, TPU_PA(ctx, off_r), CVK_FMT_I8, {K, cur_n}, {cur_n}};
        cvk->ops->tdma_g2l_matrix_copy(cvk, &(cvk_tdma_g2l_matrix_copy_param_t){
            .src = &src_r, .dst = ml_r});
        cvk->ops->tiu_matrix_multiplication(cvk, &(cvk_tiu_matrix_multiplication_param_t){
            .res = ml_o, .left = ml_l, .right = ml_r, .bias = NULL,
            .lshift_bits = 0, .rshift_bits = rshift, .res_is_int8 = 1,
            .relu_enable = 0, .add_result = 0, .ps32_mode = 0,
        });
        cvk->ops->tdma_l2g_matrix_copy(cvk, &(cvk_tdma_l2g_matrix_copy_param_t){
            .src = ml_o,
            .dst = &(cvk_mg_t){0, TPU_PA(ctx, result_off + n_start),
                               CVK_FMT_I8, {M, cur_n}, {N}}});
        cvk->ops->lmem_free_matrix(cvk, ml_o);
        cvk->ops->lmem_free_matrix(cvk, ml_r);
        cvk->ops->lmem_free_matrix(cvk, ml_l);
    }
    return 0;
}

/* Submit all batched cmds, invalidate, read results from nm offsets.
   Each result is specified by (nm_off, dst, bytes) triple. */
static void tpu_batch_submit_read(tpu_ctx *ctx, cvk_context_t *cvk,
    uint32_t *nm_offs, void **dsts, int *bytes, int n_results)
{
    CVI_RT_Submit(ctx->rt_khandle);
    CVI_RT_MemInvld(ctx->rt_handle, ctx->neuron_mem);
    uint8_t *nm = ctx->neuron_vaddr;
    for (int i = 0; i < n_results; i++)
        memcpy(dsts[i], nm + nm_offs[i], bytes[i]);
}

/* Dequant INT8 output to FP32: output_f32 = output_i8 * sc_in * sc_wt * 2^rshift */
static void dequant_i8(float *dst, const int8_t *src, int n, float sc_wt, float sc_in, int rshift) {
    float sc = sc_wt * sc_in * (float)(1 << rshift);
    for (int i = 0; i < n; i++) dst[i] = (float)src[i] * sc;
}

/* ================================================================
 *  Neuron memory: small (1MB) for TPU scratch + INT8 buffers only.
 *  Weight cache: separate large ION allocation (24MB) avoids MemFlush overhead.
 * ================================================================ */
#define NEURON_SZ       0x100000   /* 1 MB — TPU scratch + buffers */
#define WCACHE_SZ       0x1800000  /* 24 MB — weight cache (separate ION alloc) */
#define WCACHE_LAYERS   7          /* 24MB / 3.38MB = 7.1 */

/* TPU matmul scratch: M*K + K*tile_n + M*N, ~200KB for prefill */
#define SM_SCRATCH_OFF  0x000000
#define SM_SCRATCH_SZ   0x040000  /* 256 KB */

/* INT8 buffers. Offsets relative to start, sizes vary by prefill vs decode.
   Maximum sizes for seq=64 prefill shown:
   Q_i8: 64*576=36KB  K_i8: 64*192=12KB  Kt_i8: 12KB  V_i8: 12KB
   S_i8: 9*64*64=36KB  A_i8: 36KB  O_i8: 36KB
   FFN up: 64*1536=96KB  FFN gate: 96KB  FFN down: 36KB
   Total: ~372KB */
#define SM_Q_I8_OFF     0x040000
#define SM_K_I8_OFF     0x050000
#define SM_KT_I8_OFF    0x054000
#define SM_V_I8_OFF     0x058000
#define SM_S_I8_OFF     0x05C000
#define SM_A_I8_OFF     0x060000
#define SM_O_I8_OFF     0x070000
#define SM_UP_I8_OFF    0x080000
#define SM_GATE_I8_OFF  0x090000

/* Weight cache: separate ION allocation, managed independently. */
typedef struct {
    CVI_RT_MEM mem;
    uint8_t   *vaddr;
    size_t     size;
    int        n_layers;
} wcache_t;

static void wcache_free(wcache_t *wc, CVI_RT_HANDLE rt) {
    if (wc->mem) { CVI_RT_MemFree(rt, wc->mem); wc->mem = NULL; }
    wc->vaddr = NULL;
    wc->size = 0;
    wc->n_layers = 0;
}

static int wcache_alloc(wcache_t *wc, CVI_RT_HANDLE rt, size_t sz) {
    memset(wc, 0, sizeof(*wc));
    wc->mem = CVI_RT_MemAlloc(rt, sz);
    if (!wc->mem) return -1;
    wc->vaddr = CVI_RT_MemGetVAddr(wc->mem);
    wc->size  = CVI_RT_MemGetSize(wc->mem);
    return 0;
}

/* ================================================================
 *  Per-layer weight loading from SD or ION cache
 * ================================================================ */
static inline int sm_layer_bytes(int D, int dkv, int F) {
    return D*4 + D*D + D*dkv + D*dkv + D*D + D*4 + D*F + D*F + F*D;
}

typedef struct {
    uint8_t *raw;
    int raw_sz;
    int is_ion;   /* 1 if pointers reference ION memory (don't free) */
    int8_t *Wq, *Wk, *Wv, *Wo;        /* QKV weights [in, out] */
    int8_t *ffn_up, *ffn_gate, *ffn_down;
    float  *rms_attn, *rms_ffn;
} sm_layer_w_t;

static void sm_free_layer(sm_layer_w_t *w);

static void sm_setup_ptrs(sm_layer_w_t *w, uint8_t *base, int D, int dkv, int F) {
    int s0 = D*4, s1 = D*D, s2 = D*dkv, s3 = D*dkv, s4 = D*D;
    int s5 = D*4, s6 = D*F, s7 = D*F, s8 = F*D;
    uint8_t *p = base;
    w->rms_attn = (float *)(p); p += s0;
    w->Wq       = (int8_t *)(p); p += s1;
    w->Wk       = (int8_t *)(p); p += s2;
    w->Wv       = (int8_t *)(p); p += s3;
    w->Wo       = (int8_t *)(p); p += s4;
    w->rms_ffn  = (float *)(p); p += s5;
    w->ffn_up   = (int8_t *)(p); p += s6;
    w->ffn_gate = (int8_t *)(p); p += s7;
    w->ffn_down = (int8_t *)(p);
}

static int sm_alloc_layer(sm_layer_w_t *w, const sm_cfg_t *c) {
    memset(w, 0, sizeof(*w));
    int total = sm_layer_bytes(c->D, c->d_qkv, c->FFN);
    w->raw = (uint8_t *)malloc(total);
    if (!w->raw) return -1;
    w->raw_sz = total;
    w->is_ion = 0;
    sm_setup_ptrs(w, w->raw, c->D, c->d_qkv, c->FFN);
    return 0;
}

/* Point layer_w_t into weight cache ION memory (no heap alloc) */
static void sm_layer_ion(sm_layer_w_t *w, uint8_t *wc_base, int layer,
                         int D, int dkv, int F) {
    w->is_ion = 1;
    int sz = sm_layer_bytes(D, dkv, F);
    w->raw_sz = sz;
    sm_setup_ptrs(w, wc_base + layer * sz, D, dkv, F);
}

/* Preload n layers from SD into weight cache ION. */
static int sm_preload_ion(const char *weight_dir, uint8_t *wc_base,
                          int start, int n, int layer_sz) {
    char path[256];
    for (int l = 0; l < n; l++) {
        snprintf(path, sizeof(path), "%s/layer%d.bin", weight_dir, start + l);
        int fd = open(path, O_RDONLY);
        if (fd < 0) { fprintf(stderr, "  WCache preload fail: %s\n", path); return -1; }
        uint8_t *dst = wc_base + l * layer_sz;
        int remain = layer_sz;
        while (remain > 0) {
            int nr = read(fd, dst, remain);
            if (nr <= 0) { close(fd); return -1; }
            dst += nr; remain -= nr;
        }
        close(fd);
    }
    return 0;
}

static int sm_load_layer(const char *base, int l, sm_layer_w_t *w,
                         const sm_cfg_t *c) {
    char path[256];
    snprintf(path, sizeof(path), "%s/layer%d.bin", base, l);
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    int n = read(fd, w->raw, w->raw_sz);
    close(fd);
    if (n != w->raw_sz) return -1;
    /* Re-point to heap buffer (may have been overridden by ION pointers) */
    w->is_ion = 0;
    sm_setup_ptrs(w, w->raw, c->D, c->d_qkv, c->FFN);
    return 0;
}

static void sm_free_layer(sm_layer_w_t *w) {
    if (!w->is_ion) free(w->raw);
    memset(w, 0, sizeof(*w));
}

/* ================================================================
 *  KV Cache (heap, contiguous layout: [max_seq, dkv] per layer)
 *  dkv = n_kv_heads * head_dim.  Simple memcpy store/load.
 * ================================================================ */
typedef struct {
    float *K;   /* [max_seq * dkv] contiguous */
    float *V;
} sm_kv_cache_t;

static sm_kv_cache_t *sm_kv_alloc(const sm_cfg_t *c) {
    sm_kv_cache_t *kv = (sm_kv_cache_t *)calloc(c->n_layers, sizeof(sm_kv_cache_t));
    int per_layer = c->max_seq * c->d_qkv * sizeof(float);
    for (int l = 0; l < c->n_layers; l++) {
        kv[l].K = (float *)calloc(per_layer, 1);
        kv[l].V = (float *)calloc(per_layer, 1);
    }
    return kv;
}

static void sm_kv_free(sm_kv_cache_t *kv, const sm_cfg_t *c) {
    for (int l = 0; l < c->n_layers; l++) { free(kv[l].K); free(kv[l].V); }
    free(kv);
}

static void sm_kv_store_contig(float *cache, const float *new_data, int seq,
                                int pos, int dkv) {
    memcpy(cache + pos * dkv, new_data, seq * dkv * sizeof(float));
}

static void sm_kv_load_contig(float *dst, const float *cache, int kv_len, int dkv) {
    memcpy(dst, cache, kv_len * dkv * sizeof(float));
}

/* ================================================================
 *  Layer forward (GQA attention + SwiGLU FFN) — OPTIMIZED
 *
 *  Optimizations:
 *  1. QKV merged into 1 TPU submit (3→1)
 *  2. GQA heads batched by KV group (9 heads → 3 groups for scores/attn)
 *  3. FFN up+gate merged into 1 submit (2→1)
 *  4. KV cache contiguous layout (single memcpy store/load)
 *
 *  Results in ~10 submits/layer instead of 25.
 * ================================================================ */
static int sm_layer_forward(tpu_ctx *ctx, cvk_context_t *cvk,
    uint8_t *nm, const sm_cfg_t *c, const sm_layer_w_t *w,
    float *x, int seq, int pos, int kv_len,
    sm_kv_cache_t *kv, int layer,
    float *rope_cos, float *rope_sin,
    double *timing)
{
    int D = c->D, H = c->n_heads, Kvh = c->n_kv_heads, d = c->head_dim;
    int dkv = c->d_qkv, F = c->FFN, groups = c->n_groups;
    int total = seq * D;
    double ts;

    /* ---- Pre-attn RMS Norm ---- */
    ts = TICK();
    float *normed = (float *)malloc(total * sizeof(float));
    rms_norm_f32(normed, x, w->rms_attn, total, 1e-6f);
    timing[0] += TICK() - ts;

    /* ---- Quantize input (symmetric) ---- */
    float sc_x = compute_scale_sym(normed, total);
    int8_t *x_i8 = (int8_t *)malloc(total);
    quantize_i8_sym(x_i8, normed, total, sc_x);

    /* === QKV MERGED: build Q,K,V matmuls, then 1 submit === */
    int rshift_qkv = matmul_rshift(D);
    /* 3 separate scratch regions to avoid overwrite within batch */
    uint32_t scrQ = SM_SCRATCH_OFF;
    uint32_t scrK = SM_SCRATCH_OFF + 0x10000;
    uint32_t scrV = SM_SCRATCH_OFF + 0x20000;

    ts = TICK();
    int rc = tpu_matmul_build(ctx, cvk, x_i8, seq, D, w->Wq, D,
                               SM_Q_I8_OFF, scrQ, rshift_qkv);
    if (!rc) rc = tpu_matmul_build(ctx, cvk, x_i8, seq, D, w->Wk, dkv,
                                    SM_K_I8_OFF, scrK, rshift_qkv);
    if (!rc) rc = tpu_matmul_build(ctx, cvk, x_i8, seq, D, w->Wv, dkv,
                                    SM_V_I8_OFF, scrV, rshift_qkv);
    if (rc) { free(x_i8); free(normed); return rc; }
    /* Submit all 3 matmuls at once, then read results */
    CVI_RT_Submit(ctx->rt_khandle);
    CVI_RT_MemInvld(ctx->rt_handle, ctx->neuron_mem);
    int8_t *Q_i8 = (int8_t *)malloc(seq * D);
    int8_t *K_i8 = (int8_t *)malloc(seq * dkv);
    int8_t *V_i8 = (int8_t *)malloc(seq * dkv);
    memcpy(Q_i8, nm + SM_Q_I8_OFF, seq * D);
    memcpy(K_i8, nm + SM_K_I8_OFF, seq * dkv);
    memcpy(V_i8, nm + SM_V_I8_OFF, seq * dkv);
    timing[1] += TICK() - ts;  /* t_q = merged QKV time */
    free(x_i8);

    /* ---- Dequant Q, K, V from INT8 → FP32 ---- */
    float *Q_f32 = (float *)malloc(seq * D * sizeof(float));
    float *K_f32 = (float *)malloc(seq * dkv * sizeof(float));
    float *V_f32 = (float *)malloc(seq * dkv * sizeof(float));
    dequant_i8(Q_f32, Q_i8, seq * D, W_SCALE(layer, 0), sc_x, rshift_qkv);
    dequant_i8(K_f32, K_i8, seq * dkv, W_SCALE(layer, 1), sc_x, rshift_qkv);
    dequant_i8(V_f32, V_i8, seq * dkv, W_SCALE(layer, 2), sc_x, rshift_qkv);
    free(Q_i8); free(K_i8); free(V_i8);

    /* ---- RoPE (Q heads and KV heads separately) ---- */
    ts = TICK();
    for (int s = 0; s < seq; s++) {
        for (int h = 0; h < H; h++)
            rope_apply_single_f32(Q_f32 + s * D + h * d, d, pos + s,
                                   rope_cos, rope_sin);
        for (int h = 0; h < Kvh; h++)
            rope_apply_single_f32(K_f32 + s * dkv + h * d, d, pos + s,
                                   rope_cos, rope_sin);
    }
    timing[4] += TICK() - ts;

    /* ---- KV cache store (contiguous, single memcpy) ---- */
    ts = TICK();
    sm_kv_store_contig(kv[layer].K, K_f32, seq, pos, dkv);
    sm_kv_store_contig(kv[layer].V, V_f32, seq, pos, dkv);
    timing[5] += TICK() - ts;

    /* ---- Load full KV for attention (contiguous, single memcpy) ---- */
    float *K_full = (float *)malloc(kv_len * dkv * sizeof(float));
    float *V_full = (float *)malloc(kv_len * dkv * sizeof(float));
    sm_kv_load_contig(K_full, kv[layer].K, kv_len, dkv);
    sm_kv_load_contig(V_full, kv[layer].V, kv_len, dkv);

    /* === GQA Attention: batch by KV group (3 groups instead of 9 heads) === */
    float softmax_scale = 1.0f / sqrtf((float)d);
    float *Attn_out = (float *)calloc(seq * D, sizeof(float));
    int rshift_scores = matmul_rshift(d);

    /* -- Scores: 3 KV groups, each with 3 Q heads, batched into 1 submit -- */
    int Qg_sz = seq * groups * d;   /* bytes per Q group */
    int Kt_sz = d * kv_len;          /* bytes per K^T */
    int Sg_sz = seq * groups * kv_len; /* bytes per Scores group */

    /* Prepare Q groups and K transposes in nm for batched submit.
       Save per-group scales for proper dequantization later. */
    float sc_qg[9], sc_kh[9];  /* max 9 KV heads, we use 3 */
    for (int g = 0; g < Kvh; g++) {
        int hkv = g;
        float *Qg_f32 = (float *)malloc(Qg_sz * sizeof(float));
        for (int s = 0; s < seq; s++)
            for (int h = 0; h < groups; h++)
                memcpy(Qg_f32 + (s * groups + h) * d,
                       Q_f32 + s * D + (g * groups + h) * d, d * sizeof(float));
        sc_qg[g] = compute_scale_sym(Qg_f32, Qg_sz);
        quantize_i8_sym((int8_t *)(nm + SM_Q_I8_OFF + g * Qg_sz), Qg_f32, Qg_sz, sc_qg[g]);
        free(Qg_f32);

        float *Kh_f32 = (float *)malloc(kv_len * d * sizeof(float));
        for (int s = 0; s < kv_len; s++)
            memcpy(Kh_f32 + s * d, K_full + s * dkv + hkv * d, d * sizeof(float));
        sc_kh[g] = compute_scale_sym(Kh_f32, kv_len * d);
        int8_t *Kh_i8_tmp = (int8_t *)malloc(kv_len * d);
        quantize_i8_sym(Kh_i8_tmp, Kh_f32, kv_len * d, sc_kh[g]);
        free(Kh_f32);
        int8_t *Kt = (int8_t *)(nm + SM_KT_I8_OFF + g * Kt_sz);
        for (int r = 0; r < kv_len; r++)
            for (int c = 0; c < d; c++)
                Kt[c * kv_len + r] = Kh_i8_tmp[r * d + c];
        free(Kh_i8_tmp);
    }

    /* Build all 3 scores matmuls into cmdbuf */
    ts = TICK();
    for (int g = 0; g < Kvh; g++) {
        int8_t *Qg_i8 = (int8_t *)(nm + SM_Q_I8_OFF + g * Qg_sz);
        int8_t *Kt_g = (int8_t *)(nm + SM_KT_I8_OFF + g * Kt_sz);
        uint32_t Sg_off = SM_S_I8_OFF + g * Sg_sz;
        uint32_t scr = SM_SCRATCH_OFF + g * 0x4000;
        rc = tpu_matmul_build(ctx, cvk, Qg_i8, seq * groups, d,
                               Kt_g, kv_len, Sg_off, scr, rshift_scores);
        if (rc) { free(K_full); free(V_full); free(Attn_out); free(Q_f32); free(K_f32); free(V_f32); free(normed); return rc; }
    }
    CVI_RT_Submit(ctx->rt_khandle);
    CVI_RT_MemInvld(ctx->rt_handle, ctx->neuron_mem);
    timing[6] += TICK() - ts;

    /* Softmax per group with proper scales, then prepare for Attn batch */
    float sc_sg[9], sc_vh[9];
    ts = TICK();
    for (int g = 0; g < Kvh; g++) {
        int hkv = g;
        int8_t *Sg_i8 = (int8_t *)(nm + SM_S_I8_OFF + g * Sg_sz);
        float *Scores_f32 = (float *)malloc(seq * groups * kv_len * sizeof(float));
        dequant_i8(Scores_f32, Sg_i8, seq * groups * kv_len, sc_kh[g], sc_qg[g], rshift_scores);
        for (int h = 0; h < groups; h++) {
            float *row = Scores_f32 + h * (seq * kv_len);
            for (int i = 0; i < seq * kv_len; i++) row[i] *= softmax_scale;
            softmax_f32(row, seq, kv_len);
        }
        sc_sg[g] = compute_scale_sym(Scores_f32, seq * groups * kv_len);
        quantize_i8_sym(Sg_i8, Scores_f32, seq * groups * kv_len, sc_sg[g]);
        free(Scores_f32);

        float *Vh_f32 = (float *)malloc(kv_len * d * sizeof(float));
        for (int s = 0; s < kv_len; s++)
            memcpy(Vh_f32 + s * d, V_full + s * dkv + hkv * d, d * sizeof(float));
        sc_vh[g] = compute_scale_sym(Vh_f32, kv_len * d);
        quantize_i8_sym((int8_t *)(nm + SM_V_I8_OFF + g * kv_len * d), Vh_f32, kv_len * d, sc_vh[g]);
        free(Vh_f32);
    }
    timing[7] += TICK() - ts;

    /* -- Attn: 3 KV groups batched into 1 submit -- */
    ts = TICK();
    int rshift_attn = matmul_rshift(kv_len);
    for (int g = 0; g < Kvh; g++) {
        int8_t *Sg_i8 = (int8_t *)(nm + SM_S_I8_OFF + g * Sg_sz);
        int8_t *Vg_i8 = (int8_t *)(nm + SM_V_I8_OFF + g * kv_len * d);
        uint32_t Ag_off = SM_A_I8_OFF + g * seq * groups * d;
        uint32_t scr = SM_SCRATCH_OFF + g * 0x4000;
        rc = tpu_matmul_build(ctx, cvk, Sg_i8, seq * groups, kv_len,
                               Vg_i8, d, Ag_off, scr, rshift_attn);
        if (rc) { free(K_full); free(V_full); free(Attn_out); free(Q_f32); free(K_f32); free(V_f32); free(normed); return rc; }
    }
    CVI_RT_Submit(ctx->rt_khandle);
    CVI_RT_MemInvld(ctx->rt_handle, ctx->neuron_mem);
    timing[8] += TICK() - ts;

    /* Accumulate Attn results into Attn_out with proper per-group scales */
    for (int g = 0; g < Kvh; g++) {
        int8_t *Ag_i8 = (int8_t *)(nm + SM_A_I8_OFF + g * seq * groups * d);
        float sc_attn = sc_sg[g] * sc_vh[g] * (float)(1 << rshift_attn);
        for (int h = 0; h < groups; h++) {
            int hq = g * groups + h;
            for (int s = 0; s < seq; s++)
                for (int c = 0; c < d; c++)
                    Attn_out[s * D + hq * d + c] +=
                        (float)Ag_i8[(s * groups + h) * d + c] * sc_attn;
        }
    }
    free(K_full); free(V_full);

    /* ---- Wo projection: [seq, D] × Wo[D, D] (1 submit) ---- */
    ts = TICK();
    float sc_attn_q = compute_scale_sym(Attn_out, seq * D);
    quantize_i8_sym((int8_t *)(nm + SM_O_I8_OFF), Attn_out, seq * D, sc_attn_q);
    int8_t *O_in = (int8_t *)(nm + SM_O_I8_OFF);
    int8_t *O_i8 = (int8_t *)malloc(total);
    rc = tpu_matmul(ctx, cvk, O_in, seq, D, w->Wo, D, O_i8, SM_SCRATCH_OFF, rshift_qkv);
    timing[9] += TICK() - ts;
    if (rc) { free(O_i8); free(Attn_out); free(Q_f32); free(K_f32); free(V_f32); free(normed); return rc; }

    /* Residual: x = x + Wo_out */
    float *Wo_out = (float *)malloc(total * sizeof(float));
    dequant_i8(Wo_out, O_i8, total, W_SCALE(layer, 3), sc_attn_q, rshift_qkv);
    for (int i = 0; i < total; i++) x[i] += Wo_out[i];
    free(Wo_out); free(O_i8); free(Attn_out);
    free(Q_f32); free(K_f32); free(V_f32);

    /* ---- Pre-FFN RMS Norm ---- */
    ts = TICK();
    rms_norm_f32(normed, x, w->rms_ffn, total, 1e-6f);
    timing[10] += TICK() - ts;

    /* ---- Quantize for FFN ---- */
    sc_x = compute_scale_sym(normed, total);
    x_i8 = (int8_t *)malloc(total);
    quantize_i8_sym(x_i8, normed, total, sc_x);
    free(normed);

    /* === FFN up+gate MERGED: build both matmuls, 1 submit === */
    int rshift_ffn_up = matmul_rshift(D);
    ts = TICK();
    rc = tpu_matmul_build(ctx, cvk, x_i8, seq, D, w->ffn_up, F,
                           SM_UP_I8_OFF, scrQ, rshift_ffn_up);
    if (!rc) rc = tpu_matmul_build(ctx, cvk, x_i8, seq, D, w->ffn_gate, F,
                                    SM_GATE_I8_OFF, scrK, rshift_ffn_up);
    if (rc) { free(x_i8); return rc; }
    CVI_RT_Submit(ctx->rt_khandle);
    CVI_RT_MemInvld(ctx->rt_handle, ctx->neuron_mem);
    int8_t *up_i8 = (int8_t *)malloc(seq * F);
    int8_t *gate_i8 = (int8_t *)malloc(seq * F);
    memcpy(up_i8, nm + SM_UP_I8_OFF, seq * F);
    memcpy(gate_i8, nm + SM_GATE_I8_OFF, seq * F);
    timing[11] += TICK() - ts;  /* t_ffn_up = merged up+gate time */
    free(x_i8);

    /* ---- SiLU(gate) * up (CPU FP32) ---- */
    float *up_f32   = (float *)malloc(seq * F * sizeof(float));
    float *gate_f32 = (float *)malloc(seq * F * sizeof(float));
    dequant_i8(up_f32, up_i8, seq * F, W_SCALE(layer, 4), sc_x, rshift_ffn_up);
    dequant_i8(gate_f32, gate_i8, seq * F, W_SCALE(layer, 5), sc_x, rshift_ffn_up);
    free(up_i8); free(gate_i8);
    silu_f32(gate_f32, seq * F);
    for (int i = 0; i < seq * F; i++) up_f32[i] *= gate_f32[i];
    free(gate_f32);

    /* ---- FFN down: [seq, F] × down[F, D] (1 submit) ---- */
    float sc_mid = compute_scale_sym(up_f32, seq * F);
    quantize_i8_sym((int8_t *)(nm + SM_UP_I8_OFF), up_f32, seq * F, sc_mid);
    free(up_f32);

    int rshift_ffn_down = matmul_rshift(F);
    ts = TICK();
    int8_t *down_in = (int8_t *)(nm + SM_UP_I8_OFF);
    int8_t *down_i8 = (int8_t *)malloc(total);
    rc = tpu_matmul(ctx, cvk, down_in, seq, F, w->ffn_down, D,
                    down_i8, SM_SCRATCH_OFF, rshift_ffn_down);
    timing[13] += TICK() - ts;
    if (rc) { free(down_i8); return rc; }

    /* Residual: x = x + FFN_down_out */
    float *ffn_out = (float *)malloc(total * sizeof(float));
    dequant_i8(ffn_out, down_i8, total, W_SCALE(layer, 6), sc_mid, rshift_ffn_down);
    for (int i = 0; i < total; i++) x[i] += ffn_out[i];
    free(ffn_out); free(down_i8);

    return 0;
}

/* ================================================================
 *  Full forward pass
 * ================================================================ */
typedef struct {
    double t_embed, t_rms_attn, t_q, t_k, t_v, t_rope, t_kv;
    double t_scores, t_softmax, t_attn, t_wo, t_rms_ffn;
    double t_ffn_up, t_ffn_gate, t_ffn_down, t_final_rms, t_lm_head;
    double t_weight_load;
    int n_steps;
} sm_timing_t;

static int sm_forward(tpu_ctx *ctx, cvk_context_t *cvk, uint8_t *nm,
    const sm_cfg_t *c, const char *weight_dir,
    const int *token_ids, int n_tokens, int kv_start,
    sm_kv_cache_t *kv, float *logits_out,
    sm_layer_w_t *bufA,
    wcache_t *wc,
    float *rope_cos, float *rope_sin,
    sm_timing_t *t)
{
    int D = c->D, V = c->V;

    /* ---- Embedding: read one row at a time from SD ---- */
    double ts = TICK();
    float *x = (float *)malloc(n_tokens * D * sizeof(float));
    if (!x) { fprintf(stderr, "  FAIL: malloc x\n"); return -1; }
    char path[256];
    snprintf(path, sizeof(path), "%s/embed.i8", weight_dir);
    int fd = open(path, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "  FAIL: open embed.i8\n"); free(x); return -1; }

    /* Dequantize scale for embedding (stored as INT8) */
    for (int i = 0; i < n_tokens; i++) {
        int tid = token_ids[i];
        if (tid < 0 || tid >= V) tid = 0;
        /* Read one row: V×D INT8, seek to tid*D, read D bytes */
        /* But we read as float for dequantized result */
        int8_t *row_i8 = (int8_t *)malloc(D);
        if (!row_i8) { fprintf(stderr, "  FAIL: malloc row_i8\n"); close(fd); free(x); return -1; }
        lseek(fd, tid * D, SEEK_SET);
        if (read(fd, row_i8, D) != D) { fprintf(stderr, "  FAIL: read embed row tid=%d\n", tid); free(row_i8); close(fd); free(x); return -1; }
        dequantize_f32(x + i * D, row_i8, D, EMBED_SCALE, 0);
        free(row_i8);
    }
    close(fd);
    t->t_embed += TICK() - ts;

    int kv_len = kv_start + n_tokens;

    /* ---- Process layers one at a time (single buffer to save memory) ----
       ION weight-cached layers point into wcache; others loaded from SD into heap buffer. */
    int n_ion = wc ? wc->n_layers : 0;
    uint8_t *wc_base = wc ? wc->vaddr : NULL;
    int L = c->n_layers;
    for (int l = 0; l < L; l++) {
        ts = TICK();
        if (l < n_ion) {
            sm_layer_ion(bufA, wc_base, l, D, c->d_qkv, c->FFN);
        } else {
            if (sm_load_layer(weight_dir, l, bufA, c)) { free(x); return -1; }
        }
        t->t_weight_load += TICK() - ts;

        double lt[14] = {0};
        int rc = sm_layer_forward(ctx, cvk, nm, c, bufA,
                                  x, n_tokens, kv_start, kv_len,
                                  kv, l, rope_cos, rope_sin, lt);
        if (rc) { free(x); return rc; }
        t->t_rms_attn  += lt[0];  t->t_q         += lt[1];  /* merged QKV */
        t->t_rope      += lt[4];  t->t_kv        += lt[5];
        t->t_scores    += lt[6];  t->t_softmax   += lt[7];
        t->t_attn      += lt[8];  t->t_wo        += lt[9];
        t->t_rms_ffn   += lt[10]; t->t_ffn_up    += lt[11]; /* merged up+gate */
        t->t_ffn_down  += lt[13];
    }

    /* ---- Final RMS Norm ---- */
    ts = TICK();
    float *final_normed = (float *)malloc(n_tokens * D * sizeof(float));
    float *final_rms = (float *)malloc(D * sizeof(float));
    snprintf(path, sizeof(path), "%s/final_rms.f32", weight_dir);
    read_file(path, final_rms, D * 4);
    rms_norm_f32(final_normed, x, final_rms, n_tokens * D, 1e-6f);
    free(final_rms);
    t->t_final_rms += TICK() - ts;

    /* ---- LM Head: chunked matmul x[seq, D] × W[D, V] ----
       Sequential read of embed.i8 [V, D], transpose each chunk to [D, cur_v].
       No lseek needed — file position advances naturally with read(). */
    ts = TICK();
    int CHUNK = 1024; /* 1024*576=0.59MB per chunk */

    snprintf(path, sizeof(path), "%s/embed.i8", weight_dir);
    fd = open(path, O_RDONLY);
    if (fd < 0) { free(final_normed); free(x); return -1; }

    /* Quantize final_normed once (symmetric) */
    float sc_final = compute_scale_sym(final_normed, n_tokens * D);
    int8_t *x_final_i8 = (int8_t *)malloc(n_tokens * D);
    quantize_i8_sym(x_final_i8, final_normed, n_tokens * D, sc_final);
    free(final_normed);

    int max_chunk_sz = CHUNK * D;
    int8_t *chunk_raw = (int8_t *)malloc(max_chunk_sz);

    for (int v_start = 0; v_start < V; v_start += CHUNK) {
        int cur_v = (v_start + CHUNK <= V) ? CHUNK : V - v_start;
        /* Sequential read — no lseek, file position is already correct */
        if (read(fd, chunk_raw, cur_v * D) != cur_v * D) {
            free(chunk_raw); close(fd); free(x_final_i8); free(x); return -1;
        }

        int8_t *w_chunk = (int8_t *)malloc(D * cur_v);
        for (int v = 0; v < cur_v; v++)
            for (int j = 0; j < D; j++)
                w_chunk[j * cur_v + v] = chunk_raw[v * D + j];

        int rshift_lm = matmul_rshift(D);
        int8_t *logits_i8 = (int8_t *)(nm + SM_S_I8_OFF);
        int rc = tpu_matmul(ctx, cvk, x_final_i8, n_tokens, D,
                            w_chunk, cur_v, logits_i8, SM_SCRATCH_OFF, rshift_lm);
        free(w_chunk);
        if (rc) { free(chunk_raw); close(fd); free(x_final_i8); free(x); return -1; }

        dequant_i8(logits_out + v_start, logits_i8, n_tokens * cur_v,
                   EMBED_SCALE, sc_final, rshift_lm);
    }
    free(chunk_raw);
    close(fd);
    free(x_final_i8);
    t->t_lm_head += TICK() - ts;

    t->n_steps++;
    free(x);

    return 0;
}

/* ================================================================
 *  Main
 * ================================================================ */
int main(int argc, char **argv) {
    setvbuf(stderr, NULL, _IONBF, 0);
    const char *weight_dir = "/tmp/smollm2";
    const char *token_file = NULL;
    int max_new = 16;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--help")) {
            fprintf(stderr, "Usage: smollm2_demo <token_ids.bin> [max_new_tokens] [--weights dir]\n");
            fprintf(stderr, "  token_ids.bin: raw int32 tokens from host tokenizer\n");
            return 0;
        } else if (!strcmp(argv[i], "--weights") && i + 1 < argc) {
            weight_dir = argv[++i];
        } else if (argv[i][0] >= '0' && argv[i][0] <= '9') {
            max_new = atoi(argv[i]);
        } else {
            token_file = argv[i];
        }
    }

    if (!token_file) {
        fprintf(stderr, "ERROR: token file required\n");
        return 1;
    }

    /* Read config */
    sm_cfg_t c;
    if (sm_read_config(weight_dir, &c) != 0) {
        fprintf(stderr, "ERROR: cannot read config.bin from %s\n", weight_dir);
        return 1;
    }

    int D = c.D, H = c.n_heads, Kvh = c.n_kv_heads, d = c.head_dim;
    int F = c.FFN, V = c.V, L = c.n_layers, max_seq = c.max_seq;

    fprintf(stderr, "\n============================================================\n");
    fprintf(stderr, "  SmolLM2-135M TPU Inference on CV1800B\n");
    fprintf(stderr, "  D=%d H=%d KV_H=%d d=%d layers=%d FFN=%d vocab=%d\n",
            D, H, Kvh, d, L, F, V);
    fprintf(stderr, "============================================================\n");

    /* Load per-tensor weight quantization scales */
    {
        char spath[256];
        snprintf(spath, sizeof(spath), "%s/scales.bin", weight_dir);
        g_scales = (float *)malloc(212 * sizeof(float));
        if (read_file(spath, g_scales, 212 * 4) != 0) {
            fprintf(stderr, "WARNING: no scales.bin, using default scales\n");
            free(g_scales);
            g_scales = NULL;
        } else {
            fprintf(stderr, "  Scales loaded: embed=%.6f Wq0=%.6f\n",
                    EMBED_SCALE, W_SCALE(0, 0));
        }
    }

    /* Read token IDs from file */
    struct stat st;
    if (stat(token_file, &st) != 0) {
        fprintf(stderr, "ERROR: cannot stat %s\n", token_file);
        return 1;
    }
    int prompt_len = st.st_size / 4;
    if (prompt_len > max_seq) prompt_len = max_seq;
    int *token_ids = (int *)malloc(prompt_len * sizeof(int));
    if (read_file(token_file, token_ids, prompt_len * 4) != 0) {
        fprintf(stderr, "ERROR: cannot read %s\n", token_file);
        return 1;
    }

    fprintf(stderr, "\n[Init] Prompt: %d tokens\n", prompt_len);

    /* Init TPU: small neuron memory (1MB) for scratch + buffers.
       Separate large ION weight cache (24MB) avoids MemFlush overhead. */
    tpu_ctx ctx;
    if (tpu_init(&ctx, NEURON_SZ) != 0) {
        fprintf(stderr, "TPU init failed!\n");
        return 1;
    }
    cvk_context_t *cvk = ctx.cvk_ctx;
    uint8_t *nm = ctx.neuron_vaddr;

    /* Allocate separate ION weight cache */
    wcache_t wc;
    int layer_sz = sm_layer_bytes(D, c.d_qkv, F);
    int max_wcache_layers = WCACHE_SZ / layer_sz;
    {
        if (wcache_alloc(&wc, ctx.rt_handle, WCACHE_SZ) != 0) {
            fprintf(stderr, "Weight cache alloc failed, falling back to SD-only\n");
            memset(&wc, 0, sizeof(wc));
        } else {
            fprintf(stderr, "[wcache] pa=0x%llx va=%p sz=%zu\n",
                    (unsigned long long)CVI_RT_MemGetPAddr(wc.mem),
                    wc.vaddr, wc.size);
            /* Preload layers from SD into ION weight cache (one-time) */
            double ts = TICK();
            wc.n_layers = max_wcache_layers;
            if (sm_preload_ion(weight_dir, wc.vaddr, 0, wc.n_layers, layer_sz) != 0) {
                fprintf(stderr, "WARNING: WCache preload failed, falling back to SD\n");
                wcache_free(&wc, ctx.rt_handle);
                memset(&wc, 0, sizeof(wc));
            } else {
                fprintf(stderr, "  WCache: %d layers (%.1f MB) in %.0f ms\n",
                        wc.n_layers, (double)wc.n_layers * layer_sz / 1024 / 1024,
                        (TICK() - ts) / 1000.0);
            }
        }
    }

    /* Precompute RoPE tables */
    float *rope_cos = (float *)malloc(max_seq * (d / 2) * sizeof(float));
    float *rope_sin = (float *)malloc(max_seq * (d / 2) * sizeof(float));
    rope_precompute(max_seq, d, rope_cos, rope_sin);

    /* Allocate KV cache (~2.8MB, reserved in Linux heap) */
    sm_kv_cache_t *kv = sm_kv_alloc(&c);
    if (!kv) { fprintf(stderr, "KV cache alloc failed!\n"); return 1; }

    /* Single heap buffer for SD-streamed layers — allocated once, reused. */
    sm_layer_w_t layer_bufA;
    if (sm_alloc_layer(&layer_bufA, &c)) {
        fprintf(stderr, "ERROR: failed to allocate layer buffer\n");
        return 1;
    }

    /* ---- Prefill (token-by-token, seq>1 can't fit LMEM for K=1536 FFN) ---- */
    fprintf(stderr, "\n[Prefill] Processing %d tokens (token-by-token)...\n", prompt_len);
    sm_timing_t t; memset(&t, 0, sizeof(t));

    double t_prefill = TICK();
    float *prefill_logits = NULL;
    int next_token = 0;

    wcache_t *wp = wc.mem ? &wc : NULL;

    for (int p = 0; p < prompt_len; p++) {
        int tid = token_ids[p];
        float *step_logits = (float *)malloc(V * sizeof(float));
        int rc = sm_forward(&ctx, cvk, nm, &c, weight_dir,
                            &tid, 1, p, kv, step_logits,
                            &layer_bufA,
                            wp,
                            rope_cos, rope_sin, &t);
        if (rc) { fprintf(stderr, "Prefill token %d failed rc=%d\n", p, rc); return 1; }

        if (p == prompt_len - 1) {
            prefill_logits = step_logits;
        } else {
            next_token = sample_argmax(step_logits, V);
            free(step_logits);
        }
        if ((p + 1) % 10 == 0)
            fprintf(stderr, "  prefill %d/%d tokens done\n", p + 1, prompt_len);
    }
    t_prefill = TICK() - t_prefill;

    next_token = sample_argmax(prefill_logits, V);
    free(prefill_logits);

    fprintf(stderr, "  Prefill: %.1f ms, next_token=%d\n",
            t_prefill / 1000.0, next_token);

    /* ---- Decode ---- */
    fprintf(stderr, "\n[Decode] Generating %d tokens...\n", max_new);
    int generated[256]; int n_gen = 0;
    generated[n_gen++] = next_token;

    double t_decode_total = 0;
    int kv_len = prompt_len;

    for (int step = 0; step < max_new; step++) {
        int tid[1] = { next_token };
        float *step_logits = (float *)malloc(V * sizeof(float));

        double t_step = TICK();
        int rc2 = sm_forward(&ctx, cvk, nm, &c, weight_dir,
                        tid, 1, kv_len, kv, step_logits,
                        &layer_bufA,
                        wp,
                        rope_cos, rope_sin, &t);
        t_step = TICK() - t_step;
        t_decode_total += t_step;
        kv_len++;
        if (rc2) { fprintf(stderr, "Decode step %d failed rc=%d\n", step, rc2); break; }

        next_token = sample_argmax(step_logits, V);
        free(step_logits);

        if (next_token <= 0 || next_token >= V) break;
        generated[n_gen++] = next_token;

        if ((step + 1) % 5 == 0)
            fprintf(stderr, "  step %d: token=%d, %.0f ms\n",
                    step + 1, next_token, t_step / 1000.0);
    }

    fprintf(stderr, "  Generated %d tokens in %.1f ms (%.1f ms/tok, %.2f tok/s)\n",
            n_gen, t_decode_total / 1000.0,
            n_gen > 0 ? t_decode_total / (1000.0 * n_gen) : 0,
            n_gen > 0 ? 1000000.0 * n_gen / t_decode_total : 0);

    /* Output token IDs for host-side decoding */
    printf("TOKENS:");
    for (int i = 0; i < prompt_len; i++) printf(" %d", token_ids[i]);
    for (int i = 0; i < n_gen; i++) printf(" %d", generated[i]);
    printf("\n");

    /* ---- Timing breakdown ---- */
    fprintf(stderr, "\n============================================================\n");
    fprintf(stderr, "  TIMING BREAKDOWN (avg over %d steps)\n", t.n_steps);
    fprintf(stderr, "============================================================\n");

    double t_total = t.t_embed + t.t_weight_load + t.t_rms_attn
        + t.t_q + t.t_rope + t.t_kv
        + t.t_scores + t.t_softmax + t.t_attn + t.t_wo
        + t.t_rms_ffn + t.t_ffn_up + t.t_ffn_down
        + t.t_final_rms + t.t_lm_head;
    if (t_total < 1) t_total = 1;

#define TPRINT(name, val) fprintf(stderr, "  %-25s %10.0f us %8.1f%%\n", \
        name, val, 100.0 * (val) / t_total)
    TPRINT("Embedding",           t.t_embed);
    TPRINT("Weight load (SD)",    t.t_weight_load);
    TPRINT("RMS Norm (attn)",     t.t_rms_attn);
    TPRINT("QKV proj (TPU,merged)", t.t_q);
    TPRINT("RoPE",                t.t_rope);
    TPRINT("KV Cache store/load", t.t_kv);
    TPRINT("Scores = QK^T (TPU)", t.t_scores);
    TPRINT("Softmax",             t.t_softmax);
    TPRINT("Attn = SV (TPU)",     t.t_attn);
    TPRINT("Wo (TPU)",            t.t_wo);
    TPRINT("RMS Norm (FFN)",      t.t_rms_ffn);
    TPRINT("FFN up+gate (TPU,merged)", t.t_ffn_up);
    TPRINT("FFN down (TPU)",      t.t_ffn_down);
    TPRINT("Final RMS Norm",      t.t_final_rms);
    TPRINT("LM Head (TPU)",       t.t_lm_head);
    TPRINT("--- TOTAL ---",       t_total);
#undef TPRINT

    fprintf(stderr, "\n  Prefill:  %.1f ms\n", t_prefill / 1000.0);
    fprintf(stderr, "  Decode:   %.1f ms/token (%.2f tok/s)\n",
            n_gen > 0 ? t_decode_total / (1000.0 * n_gen) : 0,
            n_gen > 0 ? 1000000.0 * n_gen / t_decode_total : 0);

    /* Cleanup */
    sm_kv_free(kv, &c);
    sm_free_layer(&layer_bufA);
    if (wc.mem) wcache_free(&wc, ctx.rt_handle);
    free(rope_cos); free(rope_sin);
    free(token_ids);
    if (g_scales) free(g_scales);
    tpu_cleanup(&ctx);

    fprintf(stderr, "\nDone.\n");
    return 0;
}
