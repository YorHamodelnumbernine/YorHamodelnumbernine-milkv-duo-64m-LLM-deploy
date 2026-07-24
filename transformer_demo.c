/* transformer_demo.c — Complete Transformer inference on CV1800B.
   Mini Llama-style decoder: RMSNorm + RoPE + SwiGLU FFN + KV-cache.
   TPU matmul + CPU FP32 math. Weights from SD card.

   Build: make transformer_demo
   Run:   ./transformer_demo [prompt_text] [max_new_tokens]
*/
#include "common/tpu_bench.h"
#include "transformer.h"
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

#define TICK() ({ struct timespec _ts; clock_gettime(CLOCK_MONOTONIC, &_ts); \
                  _ts.tv_sec * 1e6 + _ts.tv_nsec / 1e3; })

/* ================================================================
 *  File I/O
 * ================================================================ */
static int read_file(const char *path, void *buf, int sz) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "  open(%s) failed\n", path); return -1; }
    int n = read(fd, buf, sz);
    close(fd);
    if (n != sz) { fprintf(stderr, "  read(%s) %d != %d\n", path, n, sz); return -1; }
    return 0;
}
static int write_file(const char *path, const void *buf, int sz) {
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) { fprintf(stderr, "  create(%s) failed\n", path); return -1; }
    int n = write(fd, buf, sz);
    close(fd);
    if (n != sz) { fprintf(stderr, "  write(%s) %d != %d\n", path, n, sz); return -1; }
    return 0;
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

static void swiglu_gate_f32(float *gate, float *up, int n) {
    silu_f32(gate, n);
    for (int i = 0; i < n; i++) up[i] *= gate[i];
}

static void rope_precompute(int seq_len, int head_dim,
                            float *cos_tab, float *sin_tab) {
    int half = head_dim / 2;
    for (int i = 0; i < half; i++) {
        float freq = 1.0f / powf(10000.0f, (2.0f * i) / (float)head_dim);
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

static void embedding_lookup_f32(float *out, const int *token_ids, int n,
                                  const float *embed_table, int d_model) {
    for (int i = 0; i < n; i++) {
        int tid = token_ids[i];
        if (tid < 0 || tid >= TR_VOCAB_SIZE) tid = 0;
        memcpy(out + i * d_model, embed_table + tid * d_model,
               d_model * sizeof(float));
    }
}

static int sample_argmax(const float *logits, int n) {
    int best = 0; float best_v = logits[0];
    for (int i = 1; i < n; i++) {
        if (logits[i] > best_v) { best_v = logits[i]; best = i; }
    }
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

/* ================================================================
 *  TPU Matmul (from mha_sdcard_full.c)
 * ================================================================ */
static int tpu_find_tile_n(cvk_context_t *cvk, int M, int K) {
    int left = tr_lmem_matrix_bytes(M, K);
    for (int tn = 128; tn >= 16; tn -= 16) {
        if (left + tr_lmem_matrix_bytes(K, tn) +
            tr_lmem_matrix_bytes(M, tn) <= 32768) return tn;
    }
    return -1;
}

static int tpu_matmul(tpu_ctx *ctx, cvk_context_t *cvk,
    const void *left, int M, int K, const void *right, int N,
    void *result, uint32_t scratch_off)
{
    const int8_t *l_i8 = (const int8_t *)left;
    const int8_t *r_i8 = (const int8_t *)right;
    int8_t       *o_i8 = (int8_t *)result;
    uint32_t off_l = scratch_off, off_r = scratch_off + M * K;

    int need_tile = (tr_lmem_matrix_bytes(M, K) +
                     tr_lmem_matrix_bytes(K, N) +
                     tr_lmem_matrix_bytes(M, N) > 32768);
    if (!need_tile) {
        uint32_t off_o = scratch_off + M * K + K * N;
        memcpy(ctx->neuron_vaddr + off_l, l_i8, M * K);
        memcpy(ctx->neuron_vaddr + off_r, r_i8, K * N);
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
            .lshift_bits = 0, .rshift_bits = 0, .res_is_int8 = 1,
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
        memcpy(o_i8, ctx->neuron_vaddr + off_o, M * N);
        return 0;
    }

    int tile_n = tpu_find_tile_n(cvk, M, K);
    if (tile_n < 16) { fprintf(stderr, "LMEM fail M=%d K=%d N=%d\n", M, K, N); return -1; }
    uintptr_t nm_base = (uintptr_t)ctx->neuron_vaddr;
    uintptr_t nm_end  = nm_base + ctx->neuron_size;
    int r_is_nm = ((uintptr_t)r_i8 >= nm_base && (uintptr_t)r_i8 < nm_end);
    uint32_t r_nm_off = r_is_nm ? (uint32_t)((uintptr_t)r_i8 - nm_base) : 0;
    uint32_t off_o_base = scratch_off + M * K + K * tile_n;

    memcpy(ctx->neuron_vaddr + off_l, l_i8, M * K);
    CVI_RT_MemFlush(ctx->rt_handle, ctx->neuron_mem);
    for (int n_start = 0; n_start < N; n_start += tile_n) {
        int cur_n = (n_start + tile_n <= N) ? tile_n : N - n_start;
        if (!r_is_nm) {
            uint8_t *td = ctx->neuron_vaddr + off_r;
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
            .lshift_bits = 0, .rshift_bits = 0, .res_is_int8 = 1,
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
    memcpy(o_i8, ctx->neuron_vaddr + off_o_base, M * N);
    return 0;
}

/* ================================================================
 *  Model struct
 * ================================================================ */
typedef struct {
    tpu_ctx *ctx;
    cvk_context_t *cvk;

    /* Weights (INT8 for matmul layers, FP32 for norms/embedding) */
    float  *embed_f32;
    int8_t *W_qkv[TR_N_LAYERS]; /* fused QKV [D, 3*D] for single-submit QKV */
    int8_t *Wo[TR_N_LAYERS];
    int8_t *ffn_up[TR_N_LAYERS], *ffn_gate[TR_N_LAYERS], *ffn_down[TR_N_LAYERS];
    int8_t *lm_head;
    float  *rms_attn[TR_N_LAYERS], *rms_ffn[TR_N_LAYERS], *final_rms;

    /* RoPE tables */
    float *rope_cos, *rope_sin;

    /* Neuron memory */
    uint8_t *nm;

    /* Timing */
    double t_embed, t_rms_attn, t_qkv, t_rope, t_kv_cache;
    double t_scores, t_softmax, t_attn, t_wo;
    double t_rms_ffn, t_ffn[3], t_final_rms, t_lm_head;
    int    n_steps;
} tr_model_t;

/* ================================================================
 *  MHA: QKV projection (FUSED single submit) + RoPE + KV cache
 *  W_qkv = [Wq | Wk | Wv]  shape [D, 3*D], result [seq, 3*D]
 * ================================================================ */
static int tr_mha_qkv(tr_model_t *m, float *x, int seq, int layer,
                       float *Q_out, float *K_out, float *V_out,
                       int8_t *qkv_i8, float sc_x, int zp_x, int pos_offset)
{
    int D = TR_D_MODEL, d = TR_HEAD_DIM, H = TR_N_HEADS;
    int total = seq * D;
    tpu_ctx *ctx = m->ctx;
    cvk_context_t *cvk = m->cvk;

    /* Quantize input once */
    int8_t *x_i8 = (int8_t *)malloc(total);
    quantize_i8(x_i8, x, total, sc_x, zp_x);

    /* Fused QKV: x[seq,D] × W_qkv[D,3*D] → qkv_i8[seq, 3*D] */
    double ts = TICK();
    int rc = tpu_matmul(ctx, cvk, x_i8, seq, D, m->W_qkv[layer], 3 * D,
                        qkv_i8, TR_MATMUL_SCR);
    m->t_qkv += TICK() - ts;
    free(x_i8);
    if (rc) return rc;

    /* Dequant & split Q,K,V from fused result */
    float sc_out = sc_x * 0.001f;
    for (int s = 0; s < seq; s++) {
        for (int i = 0; i < D; i++) {
            int base = s * 3 * D;
            Q_out[s * D + i] = (float)(int)qkv_i8[base + i] * sc_out;
            K_out[s * D + i] = (float)(int)qkv_i8[base + D + i] * sc_out;
            V_out[s * D + i] = (float)(int)qkv_i8[base + 2 * D + i] * sc_out;
        }
    }

    /* RoPE per head, per position */
    ts = TICK();
    for (int h = 0; h < H; h++) {
        for (int s = 0; s < seq; s++) {
            rope_apply_single_f32(Q_out + s * D + h * d, d, pos_offset + s,
                                   m->rope_cos, m->rope_sin);
            rope_apply_single_f32(K_out + s * D + h * d, d, pos_offset + s,
                                   m->rope_cos, m->rope_sin);
        }
    }
    m->t_rope += TICK() - ts;

    /* KV cache store (interleaved layout [pos*H+h, d]) */
    ts = TICK();
    for (int h = 0; h < H; h++) {
        for (int s = 0; s < seq; s++) {
            int gp = pos_offset + s;
            if (gp >= TR_MAX_SEQ) break;
            int off = (gp * H + h) * d;
            memcpy((float *)(m->nm + TR_OFF_K_CACHE(layer)) + off,
                   K_out + s * D + h * d, d * sizeof(float));
            memcpy((float *)(m->nm + TR_OFF_V_CACHE(layer)) + off,
                   V_out + s * D + h * d, d * sizeof(float));
        }
    }
    m->t_kv_cache += TICK() - ts;
    return 0;
}

/* ================================================================
 *  MHA: Attention (Scores + Softmax + Attn*V + Wo) — OPTIMIZED
 *  Pre-extracts all heads, quantizes upfront, bursts TPU submits.
 * ================================================================ */
static int tr_mha_attn(tr_model_t *m, float *Q_f32, float *K_f32, float *V_f32,
                        float *Attn_out, int seq, int kv_len, int layer,
                        int8_t *S_i8, int8_t *A_i8, int8_t *O_i8)
{
    int D = TR_D_MODEL, d = TR_HEAD_DIM, H = TR_N_HEADS;
    tpu_ctx *ctx = m->ctx;
    cvk_context_t *cvk = m->cvk;
    float softmax_scale = 1.0f / sqrtf((float)d);
    float *Scores_f32 = (float *)(m->nm + TR_OFF_SCORES_F32);

    /* Neuron memory pools for per-head INT8 data.
       K pool: first half untransposed [kv_len,d], second half transposed [d,kv_len] */
    #define HSTRIDE_Q  128
    #define HSTRIDE_K  (TR_MAX_SEQ * d)   /* 2048 bytes per head */
    #define HSTRIDE_V  (TR_MAX_SEQ * d)
    int8_t *Q_i8_pool = (int8_t *)(m->nm + TR_OFF_Q_I8);
    int8_t *K_i8_pool = (int8_t *)(m->nm + TR_OFF_K_I8);  /* [0..8K) untransposed */
    int8_t *Kt_pool   = K_i8_pool + H * HSTRIDE_K;         /* [8K..16K) transposed */
    int8_t *V_i8_pool = (int8_t *)(m->nm + TR_OFF_V_I8);

    /* Phase 1: Pre-extract & quantize all heads (CPU burst, no TPU) */
    double ts = TICK();
    for (int h = 0; h < H; h++) {
        float qh_buf[TR_MAX_SEQ * d], kh_buf[TR_MAX_SEQ * d], vh_buf[TR_MAX_SEQ * d];
        for (int s = 0; s < seq; s++)
            memcpy(qh_buf + s * d, Q_f32 + s * D + h * d, d * sizeof(float));
        for (int s = 0; s < kv_len; s++)
            memcpy(kh_buf + s * d, K_f32 + s * D + h * d, d * sizeof(float));
        for (int s = 0; s < kv_len; s++)
            memcpy(vh_buf + s * d, V_f32 + s * D + h * d, d * sizeof(float));

        /* Fixed scales (skip compute_scale for small tensors) */
        quantize_i8(Q_i8_pool + h * HSTRIDE_Q, qh_buf, seq * d, 0.01f, -128);
        quantize_i8(K_i8_pool + h * HSTRIDE_K, kh_buf, kv_len * d, 0.01f, -128);
        quantize_i8(V_i8_pool + h * HSTRIDE_V, vh_buf, kv_len * d, 0.01f, -128);

        /* Out-of-place transpose: Kh[kv_len,d] → Kt[d,kv_len] */
        int8_t *Kh = K_i8_pool + h * HSTRIDE_K;
        int8_t *Kt = Kt_pool   + h * HSTRIDE_K;
        for (int r = 0; r < kv_len; r++)
            for (int c = 0; c < d; c++)
                Kt[c * kv_len + r] = Kh[r * d + c];
    }
    m->t_kv_cache += TICK() - ts; /* reuse timing slot for data prep */

    /* Phase 2: Scores = Q × K^T — 4 TPU submits in burst */
    ts = TICK();
    for (int h = 0; h < H; h++) {
        int8_t *Qh = Q_i8_pool + h * HSTRIDE_Q;
        int8_t *Kt = Kt_pool   + h * HSTRIDE_K;
        int rc = tpu_matmul(ctx, cvk, Qh, seq, d, Kt, kv_len,
                            S_i8 + h * seq * kv_len, TR_MATMUL_SCR);
        if (rc) return rc;
    }
    m->t_scores += TICK() - ts;

    /* Phase 3: Softmax (CPU FP32) */
    ts = TICK();
    dequantize_f32(Scores_f32, S_i8, H * seq * kv_len, 0.01f, -128);
    for (int i = 0; i < H * seq * kv_len; i++) Scores_f32[i] *= softmax_scale;
    for (int h = 0; h < H; h++)
        softmax_f32(Scores_f32 + h * seq * kv_len, seq, kv_len);
    m->t_softmax += TICK() - ts;

    /* Phase 4: Attn = Softmax × V — re-quant + 4 TPU submits in burst */
    ts = TICK();
    for (int h = 0; h < H; h++) {
        /* Re-quantize softmax scores (fixed scale) */
        quantize_i8(S_i8, Scores_f32 + h * seq * kv_len, seq * kv_len, 0.01f, -128);
        int8_t *Vh = V_i8_pool + h * HSTRIDE_V;
        int rc = tpu_matmul(ctx, cvk, S_i8, seq, kv_len, Vh, d,
                            A_i8 + h * seq * d, TR_MATMUL_SCR);
        if (rc) return rc;
    }
    /* Dequant & interleave: [4][seq,d] → [seq, H*d] */
    for (int h = 0; h < H; h++)
        for (int s = 0; s < seq; s++)
            for (int c = 0; c < d; c++)
                Attn_out[s * D + h * d + c] =
                    (float)(int)A_i8[h * seq * d + s * d + c] * 0.001f;
    m->t_attn += TICK() - ts;

    /* Phase 5: Output projection Wo (1 TPU submit) */
    ts = TICK();
    quantize_i8(A_i8, Attn_out, seq * D, 0.01f, -128);
    int rc = tpu_matmul(ctx, cvk, A_i8, seq, D, m->Wo[layer], D, O_i8, TR_MATMUL_SCR);
    if (rc) return rc;
    dequantize_f32(Attn_out, O_i8, seq * D, 0.001f, 0);
    m->t_wo += TICK() - ts;
    return 0;
}

/* ================================================================
 *  FFN: SwiGLU (up, gate) + down
 * ================================================================ */
static int tr_ffn(tr_model_t *m, float *x, int seq, int layer,
                   float sc_x, int zp_x)
{
    int D = TR_D_MODEL, F = TR_FFN_HIDDEN;
    int total = seq * D, ffn_total = seq * F;
    tpu_ctx *ctx = m->ctx;
    cvk_context_t *cvk = m->cvk;
    float *ffn_f32 = (float *)(m->nm + TR_OFF_FFN_F32);
    int8_t *up_i8  = (int8_t *)(m->nm + TR_OFF_FFN_UP_I8);
    int8_t *gate_i8 = (int8_t *)(m->nm + TR_OFF_FFN_GT_I8);
    int8_t *down_i8 = (int8_t *)(m->nm + TR_OFF_FFN_DN_I8);

    int8_t *x_i8 = (int8_t *)malloc(total);
    quantize_i8(x_i8, x, total, sc_x, zp_x);

    /* FFN up */
    double ts = TICK();
    int rc = tpu_matmul(ctx, cvk, x_i8, seq, D, m->ffn_up[layer], F, up_i8, TR_MATMUL_SCR);
    m->t_ffn[0] += TICK() - ts;
    if (rc) { free(x_i8); return rc; }

    /* FFN gate */
    ts = TICK();
    rc = tpu_matmul(ctx, cvk, x_i8, seq, D, m->ffn_gate[layer], F, gate_i8, TR_MATMUL_SCR);
    m->t_ffn[1] += TICK() - ts;
    free(x_i8);
    if (rc) return rc;

    /* Dequant → SiLU(gate) * up */
    float sc_mid = sc_x * 0.001f;
    dequantize_f32(ffn_f32, up_i8, ffn_total, sc_mid, 0);
    float *gate_f32 = (float *)malloc(ffn_total * sizeof(float));
    dequantize_f32(gate_f32, gate_i8, ffn_total, sc_mid, 0);
    swiglu_gate_f32(gate_f32, ffn_f32, ffn_total);
    free(gate_f32);

    /* Quantize → FFN down */
    float sc_mid2; int zp_mid2;
    sc_mid2 = compute_scale(ffn_f32, ffn_total, &zp_mid2);
    quantize_i8(down_i8, ffn_f32, ffn_total, sc_mid2, zp_mid2);

    /* FFN down: merged [seq, F] * down [F, D] */
    ts = TICK();
    rc = tpu_matmul(ctx, cvk, down_i8, seq, F, m->ffn_down[layer], D,
                    (int8_t *)(m->nm + TR_OFF_O_I8), TR_MATMUL_SCR);
    m->t_ffn[2] += TICK() - ts;
    if (rc) return rc;

    dequantize_f32(x, (int8_t *)(m->nm + TR_OFF_O_I8), total,
                   sc_mid2 * 0.001f, 0);
    return 0;
}

/* ================================================================
 *  Single layer forward
 * ================================================================ */
static int tr_layer_forward(tr_model_t *m, float *x, int seq, int layer,
                             int pos_offset, int kv_len)
{
    int D = TR_D_MODEL, d = TR_HEAD_DIM, H = TR_N_HEADS;
    int total = seq * D;

    float *Q_f32   = (float *)(m->nm + TR_OFF_Q_F32);
    float *K_f32   = (float *)(m->nm + TR_OFF_K_F32);
    float *V_f32   = (float *)(m->nm + TR_OFF_V_F32);
    float *Attn_f32 = (float *)(m->nm + TR_OFF_ATTN_F32);
    float *Residual = (float *)(m->nm + TR_OFF_OUT_F32);
    int8_t *qkv_i8 = (int8_t *)(m->nm + TR_OFF_Q_I8);  /* fused QKV output [seq, 3*D] */
    int8_t *S_i8   = (int8_t *)(m->nm + TR_OFF_S_I8);
    int8_t *A_i8   = (int8_t *)(m->nm + TR_OFF_A_I8);
    int8_t *O_i8   = (int8_t *)(m->nm + TR_OFF_O_I8);

    /* Save residual */
    memcpy(Residual, x, total * sizeof(float));

    /* Pre-attn RMS Norm */
    double ts = TICK();
    rms_norm_f32(x, x, m->rms_attn[layer], total, 1e-6f);
    m->t_rms_attn += TICK() - ts;

    /* Quantize for QKV */
    float sc_x; int zp_x;
    sc_x = compute_scale(x, total, &zp_x);

    /* Fused QKV + RoPE + KV cache store (single TPU submit) */
    int rc = tr_mha_qkv(m, x, seq, layer, Q_f32, K_f32, V_f32,
                         qkv_i8, sc_x, zp_x, pos_offset);
    if (rc) return rc;

    /* For decode with KV cache: load full cache into K_f32, V_f32 */
    if (kv_len > seq) {
        ts = TICK();
        for (int s = 0; s < kv_len; s++) {
            for (int h = 0; h < H; h++) {
                int off = (s * H + h) * d;
                memcpy(K_f32 + s * D + h * d,
                       (float *)(m->nm + TR_OFF_K_CACHE(layer)) + off,
                       d * sizeof(float));
                memcpy(V_f32 + s * D + h * d,
                       (float *)(m->nm + TR_OFF_V_CACHE(layer)) + off,
                       d * sizeof(float));
            }
        }
        m->t_kv_cache += TICK() - ts;
    }

    /* Attention (optimized: pre-extract, burst TPU, fixed scales) */
    rc = tr_mha_attn(m, Q_f32, K_f32, V_f32, Attn_f32, seq, kv_len, layer,
                     S_i8, A_i8, O_i8);
    if (rc) return rc;

    /* Residual: x = residual + Attn_out */
    for (int i = 0; i < total; i++) x[i] = Residual[i] + Attn_f32[i];

    /* Pre-FFN RMS Norm */
    ts = TICK();
    rms_norm_f32(x, x, m->rms_ffn[layer], total, 1e-6f);
    m->t_rms_ffn += TICK() - ts;

    /* Save residual for FFN */
    memcpy(Residual, x, total * sizeof(float));

    /* Quantize for FFN */
    sc_x = compute_scale(x, total, &zp_x);

    /* FFN */
    rc = tr_ffn(m, x, seq, layer, sc_x, zp_x);
    if (rc) return rc;

    /* Residual: x = residual + FFN_out */
    for (int i = 0; i < total; i++) x[i] = Residual[i] + x[i];

    return 0;
}

/* ================================================================
 *  Full forward pass
 *   kv_start: starting position in KV cache (0 for prefill)
 *   Returns: logits_out [n_tokens * V]
 * ================================================================ */
static int tr_forward(tr_model_t *m, const int *token_ids, int n_tokens,
                       int kv_start, float *logits_out)
{
    int D = TR_D_MODEL, V = TR_VOCAB_SIZE;
    int total = n_tokens * D;

    /* Embedding */
    double ts = TICK();
    float *x = (float *)(m->nm + TR_OFF_X_F32);
    embedding_lookup_f32(x, token_ids, n_tokens, m->embed_f32, D);
    m->t_embed += TICK() - ts;

    /* KV cache total length after this step */
    int kv_len = kv_start + n_tokens;

    /* Layers */
    for (int l = 0; l < TR_N_LAYERS; l++) {
        int rc = tr_layer_forward(m, x, n_tokens, l, kv_start, kv_len);
        if (rc) return rc;
    }

    /* Final RMS Norm */
    ts = TICK();
    rms_norm_f32(x, x, m->final_rms, total, 1e-6f);
    m->t_final_rms += TICK() - ts;

    /* LM Head */
    ts = TICK();
    float sc_final; int zp_final;
    sc_final = compute_scale(x, total, &zp_final);
    int8_t *x_i8 = (int8_t *)malloc(total);
    quantize_i8(x_i8, x, total, sc_final, zp_final);

    int8_t *logits_i8 = (int8_t *)(m->nm + TR_OFF_LOGITS_I8);
    int rc = tpu_matmul(m->ctx, m->cvk, x_i8, n_tokens, D,
                        m->lm_head, V, logits_i8, TR_MATMUL_SCR);
    free(x_i8);
    if (rc) return rc;

    dequantize_f32(logits_out, logits_i8, n_tokens * V, sc_final * 0.001f, 0);
    m->t_lm_head += TICK() - ts;

    m->n_steps++;
    return 0;
}

/* ================================================================
 *  CPU FP32 Reference (for accuracy verification)
 * ================================================================ */
static void cpu_matmul_f32(float *C, const float *A, const float *B,
                           int M, int K, int N) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float s = 0;
            for (int kk = 0; kk < K; kk++)
                s += A[i * K + kk] * B[kk * N + j];
            C[i * N + j] = s;
        }
}

typedef struct {
    float *embed;
    float *rms_attn[TR_N_LAYERS], *rms_ffn[TR_N_LAYERS], *final_rms;
    float *Wq[TR_N_LAYERS], *Wk[TR_N_LAYERS], *Wv[TR_N_LAYERS], *Wo[TR_N_LAYERS];
    float *ffn_up[TR_N_LAYERS], *ffn_gate[TR_N_LAYERS], *ffn_down[TR_N_LAYERS];
    float *lm_head;
    float *rope_cos, *rope_sin;
} tr_ref_t;

static void tr_ref_layer(tr_ref_t *r, float *x, int seq, int layer,
                          int pos_offset, int kv_len,
                          float *K_cache, float *V_cache) {
    int D = TR_D_MODEL, d = TR_HEAD_DIM, H = TR_N_HEADS, F = TR_FFN_HIDDEN;
    int total = seq * D;

    float *residual = (float *)malloc(total * sizeof(float));
    memcpy(residual, x, total * sizeof(float));

    /* RMS Norm (attn pre-norm) */
    rms_norm_f32(x, x, r->rms_attn[layer], total, 1e-6f);

    /* QKV projection */
    float *Q = (float *)malloc(total * sizeof(float));
    float *K = (float *)malloc(total * sizeof(float));
    float *V = (float *)malloc(total * sizeof(float));
    cpu_matmul_f32(Q, x, r->Wq[layer], seq, D, D);
    cpu_matmul_f32(K, x, r->Wk[layer], seq, D, D);
    cpu_matmul_f32(V, x, r->Wv[layer], seq, D, D);

    /* RoPE per head per position */
    for (int h = 0; h < H; h++) {
        for (int s = 0; s < seq; s++) {
            rope_apply_single_f32(Q + s * D + h * d, d, pos_offset + s,
                                   r->rope_cos, r->rope_sin);
            rope_apply_single_f32(K + s * D + h * d, d, pos_offset + s,
                                   r->rope_cos, r->rope_sin);
        }
    }

    /* KV cache store + load full history */
    for (int h = 0; h < H; h++) {
        for (int s = 0; s < seq; s++) {
            int gp = pos_offset + s;
            memcpy(K_cache + (gp * H + h) * d, K + s * D + h * d, d * sizeof(float));
            memcpy(V_cache + (gp * H + h) * d, V + s * D + h * d, d * sizeof(float));
        }
    }
    /* Reload full KV for attention */
    float *K_full = (float *)malloc(kv_len * D * sizeof(float));
    float *V_full = (float *)malloc(kv_len * D * sizeof(float));
    for (int h = 0; h < H; h++) {
        for (int s = 0; s < kv_len; s++) {
            memcpy(K_full + s * D + h * d, K_cache + (s * H + h) * d, d * sizeof(float));
            memcpy(V_full + s * D + h * d, V_cache + (s * H + h) * d, d * sizeof(float));
        }
    }

    float softmax_scale = 1.0f / sqrtf((float)d);
    float *Attn_out = (float *)calloc(total, sizeof(float));

    /* Per-head attention */
    for (int h = 0; h < H; h++) {
        float *Scores_h = (float *)malloc(seq * kv_len * sizeof(float));
        /* Q[seq,d] × K_full^T[d,kv_len] */
        for (int i = 0; i < seq; i++) {
            for (int j = 0; j < kv_len; j++) {
                float s = 0;
                for (int kk = 0; kk < d; kk++)
                    s += Q[i * D + h * d + kk] * K_full[j * D + h * d + kk];
                Scores_h[i * kv_len + j] = s * softmax_scale;
            }
        }
        softmax_f32(Scores_h, seq, kv_len);
        /* Scores × V_full */
        for (int i = 0; i < seq; i++) {
            for (int j = 0; j < d; j++) {
                float s = 0;
                for (int kk = 0; kk < kv_len; kk++)
                    s += Scores_h[i * kv_len + kk] * V_full[kk * D + h * d + j];
                Attn_out[i * D + h * d + j] += s;
            }
        }
        free(Scores_h);
    }
    free(K_full); free(V_full);

    /* Wo projection */
    float *Wo_out = (float *)malloc(total * sizeof(float));
    cpu_matmul_f32(Wo_out, Attn_out, r->Wo[layer], seq, D, D);

    /* Residual */
    for (int i = 0; i < total; i++) x[i] = residual[i] + Wo_out[i];
    free(Attn_out); free(Wo_out);

    /* RMS Norm (FFN pre-norm) */
    memcpy(residual, x, total * sizeof(float));
    rms_norm_f32(x, x, r->rms_ffn[layer], total, 1e-6f);

    /* SwiGLU FFN */
    float *up   = (float *)malloc(seq * F * sizeof(float));
    float *gate = (float *)malloc(seq * F * sizeof(float));
    cpu_matmul_f32(up,   x, r->ffn_up[layer],   seq, D, F);
    cpu_matmul_f32(gate, x, r->ffn_gate[layer], seq, D, F);
    swiglu_gate_f32(gate, up, seq * F);
    float *ffn_out = (float *)malloc(total * sizeof(float));
    cpu_matmul_f32(ffn_out, up, r->ffn_down[layer], seq, F, D);
    free(up); free(gate);

    /* Residual */
    for (int i = 0; i < total; i++) x[i] = residual[i] + ffn_out[i];
    free(residual); free(ffn_out);
    free(Q); free(K); free(V);
}

static void tr_ref_forward(tr_ref_t *r, const int *token_ids, int n_tokens,
                            float *logits_out) {
    int D = TR_D_MODEL, V = TR_VOCAB_SIZE;
    int total = n_tokens * D;

    float *x = (float *)malloc(total * sizeof(float));
    embedding_lookup_f32(x, token_ids, n_tokens, r->embed, D);

    /* KV cache for reference */
    int max_kv = TR_MAX_SEQ;
    float *K_cache = (float *)calloc(max_kv * TR_N_HEADS * TR_HEAD_DIM, sizeof(float));
    float *V_cache = (float *)calloc(max_kv * TR_N_HEADS * TR_HEAD_DIM, sizeof(float));

    for (int l = 0; l < TR_N_LAYERS; l++) {
        tr_ref_layer(r, x, n_tokens, l, 0, n_tokens, K_cache, V_cache);
    }

    rms_norm_f32(x, x, r->final_rms, total, 1e-6f);
    cpu_matmul_f32(logits_out, x, r->lm_head, n_tokens, D, V);

    free(x); free(K_cache); free(V_cache);
}

/* ================================================================
 *  Weight generation helpers
 * ================================================================ */
static float *gen_f32(int n) {
    float *d = (float *)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) d[i] = (float)(rand() % 256 - 128) / 1000.0f;
    return d;
}

static int8_t *gen_i8(int n, float *sc_out, int *zp_out) {
    float *tmp = gen_f32(n);
    *sc_out = compute_scale(tmp, n, zp_out);
    int8_t *q = (int8_t *)malloc(n);
    quantize_i8(q, tmp, n, *sc_out, *zp_out);
    free(tmp);
    return q;
}

static float *gen_rms(int n) {
    float *d = (float *)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) d[i] = 1.0f;
    return d;
}

/* ================================================================
 *  Main
 * ================================================================ */
int main(int argc, char **argv) {
    setvbuf(stderr, NULL, _IONBF, 0);
    const char *prompt = "Hello";
    int max_new = 16, do_ref = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--ref")) do_ref = 1;
        else if (!strcmp(argv[i], "--help")) {
            fprintf(stderr, "Usage: transformer_demo [prompt] [max_tokens] [--ref]\n");
            return 0;
        } else if (argv[i][0] >= '0' && argv[i][0] <= '9')
            max_new = atoi(argv[i]);
        else
            prompt = argv[i];
    }

    int D = TR_D_MODEL, H = TR_N_HEADS, d = TR_HEAD_DIM;
    int F = TR_FFN_HIDDEN, V = TR_VOCAB_SIZE;
    int w_sz_DD = D * D, w_sz_DF = D * F, w_sz_FD = F * D;

    fprintf(stderr, "\n============================================================\n");
    fprintf(stderr, "  Transformer Demo - Mini Llama-style Decoder\n");
    fprintf(stderr, "  D=%d H=%d d=%d layers=%d ffn=%d vocab=%d max_seq=%d\n",
            D, H, d, TR_N_LAYERS, F, V, TR_MAX_SEQ);
    fprintf(stderr, "============================================================\n");

    /* ---- Phase 1: Generate random weights & write to SD ---- */
    fprintf(stderr, "\n[Phase 1] Generate weights & write SD cards...\n");
    const char *base = "/tmp/tr_bench";
    mkdir(base, 0755);
    srand(42);
    double t_start = TICK();
    char path[256];
    int zp_dummy;

    /* Embedding (FP32) */
    float *embed = gen_f32(V * D);
    snprintf(path, sizeof(path), "%s/embed.f32", base);
    write_file(path, embed, V * D * 4);

    for (int l = 0; l < TR_N_LAYERS; l++) {
        /* RMS attn gamma */
        float *r = gen_rms(D);
        snprintf(path, sizeof(path), "%s/layer%d_rms_attn.f32", base, l);
        write_file(path, r, D * 4); free(r);

        /* Fused QKV weight [D, 3*D] (single submit) + Wo [D, D] */
        int w_sz_D3D = D * 3 * D;
        int8_t *qkv = gen_i8(w_sz_D3D, &(float){0}, &zp_dummy);
        snprintf(path, sizeof(path), "%s/layer%d_Wqkv.i8", base, l); write_file(path, qkv, w_sz_D3D); free(qkv);
        int8_t *qo = gen_i8(w_sz_DD, &(float){0}, &zp_dummy);
        snprintf(path, sizeof(path), "%s/layer%d_Wo.i8", base, l); write_file(path, qo, w_sz_DD); free(qo);

        /* RMS ffn gamma */
        r = gen_rms(D);
        snprintf(path, sizeof(path), "%s/layer%d_rms_ffn.f32", base, l);
        write_file(path, r, D * 4); free(r);

        /* FFN weights (INT8) */
        int8_t *fq = gen_i8(w_sz_DF, &(float){0}, &zp_dummy);
        snprintf(path, sizeof(path), "%s/layer%d_ffn_up.i8", base, l); write_file(path, fq, w_sz_DF); free(fq);
        fq = gen_i8(w_sz_DF, &(float){0}, &zp_dummy);
        snprintf(path, sizeof(path), "%s/layer%d_ffn_gate.i8", base, l); write_file(path, fq, w_sz_DF); free(fq);
        fq = gen_i8(w_sz_FD, &(float){0}, &zp_dummy);
        snprintf(path, sizeof(path), "%s/layer%d_ffn_down.i8", base, l); write_file(path, fq, w_sz_FD); free(fq);
    }

    /* Final RMS */
    float *fr = gen_rms(D);
    snprintf(path, sizeof(path), "%s/final_rms.f32", base); write_file(path, fr, D * 4); free(fr);

    /* LM head (INT8) */
    int8_t *lm = gen_i8(V * D, &(float){0}, &zp_dummy);
    snprintf(path, sizeof(path), "%s/lm_head.i8", base); write_file(path, lm, V * D); free(lm);

    sync();
    fprintf(stderr, "  Weights generated: %.1f ms\n", (TICK() - t_start) / 1000.0);

    /* ---- Phase 2: Load weights from SD ---- */
    fprintf(stderr, "\n[Phase 2] Load weights from SD...\n");
    t_start = TICK();

    tr_model_t m; memset(&m, 0, sizeof(m));
    m.embed_f32 = (float *)malloc(V * D * sizeof(float));
    snprintf(path, sizeof(path), "%s/embed.f32", base);
    read_file(path, m.embed_f32, V * D * 4);

    for (int l = 0; l < TR_N_LAYERS; l++) {
        m.rms_attn[l] = (float *)malloc(D * sizeof(float));
        snprintf(path, sizeof(path), "%s/layer%d_rms_attn.f32", base, l);
        read_file(path, m.rms_attn[l], D * 4);

        m.W_qkv[l] = (int8_t *)malloc(D * 3 * D);
        snprintf(path, sizeof(path), "%s/layer%d_Wqkv.i8", base, l);
        read_file(path, m.W_qkv[l], D * 3 * D);
        m.Wo[l] = (int8_t *)malloc(w_sz_DD);
        snprintf(path, sizeof(path), "%s/layer%d_Wo.i8", base, l);
        read_file(path, m.Wo[l], w_sz_DD);

        m.rms_ffn[l] = (float *)malloc(D * sizeof(float));
        snprintf(path, sizeof(path), "%s/layer%d_rms_ffn.f32", base, l);
        read_file(path, m.rms_ffn[l], D * 4);

        m.ffn_up[l]   = (int8_t *)malloc(w_sz_DF);
        snprintf(path, sizeof(path), "%s/layer%d_ffn_up.i8", base, l); read_file(path, m.ffn_up[l], w_sz_DF);
        m.ffn_gate[l] = (int8_t *)malloc(w_sz_DF);
        snprintf(path, sizeof(path), "%s/layer%d_ffn_gate.i8", base, l); read_file(path, m.ffn_gate[l], w_sz_DF);
        m.ffn_down[l] = (int8_t *)malloc(w_sz_FD);
        snprintf(path, sizeof(path), "%s/layer%d_ffn_down.i8", base, l); read_file(path, m.ffn_down[l], w_sz_FD);
    }

    m.final_rms = (float *)malloc(D * sizeof(float));
    snprintf(path, sizeof(path), "%s/final_rms.f32", base);
    read_file(path, m.final_rms, D * 4);

    m.lm_head = (int8_t *)malloc(V * D);
    snprintf(path, sizeof(path), "%s/lm_head.i8", base);
    read_file(path, m.lm_head, V * D);

    /* RoPE tables */
    m.rope_cos = (float *)malloc(TR_MAX_SEQ * (d / 2) * sizeof(float));
    m.rope_sin = (float *)malloc(TR_MAX_SEQ * (d / 2) * sizeof(float));
    rope_precompute(TR_MAX_SEQ, d, m.rope_cos, m.rope_sin);

    fprintf(stderr, "  Weights loaded: %.1f ms\n", (TICK() - t_start) / 1000.0);

    /* ---- Phase 3: Init TPU ---- */
    fprintf(stderr, "\n[Phase 3] Init TPU...\n");
    tpu_ctx ctx;
    if (tpu_init(&ctx, TR_NEURON_SIZE) != 0) {
        fprintf(stderr, "TPU init failed!\n"); return 1;
    }
    m.ctx = &ctx;
    m.cvk = ctx.cvk_ctx;
    m.nm  = ctx.neuron_vaddr;

    /* ---- Phase 4: Prefill ---- */
    int prompt_len = strlen(prompt);
    if (prompt_len > TR_MAX_SEQ) prompt_len = TR_MAX_SEQ;
    int *token_ids = (int *)malloc(prompt_len * sizeof(int));
    for (int i = 0; i < prompt_len; i++)
        token_ids[i] = (int)(unsigned char)prompt[i] % V;

    fprintf(stderr, "\n[Phase 4] Prefill: \"%s\" (%d tokens)\n", prompt, prompt_len);

    double t_prefill = TICK();
    float *logits = (float *)malloc(prompt_len * V * sizeof(float));
    int rc = tr_forward(&m, token_ids, prompt_len, 0, logits);
    if (rc) { fprintf(stderr, "Prefill failed rc=%d\n", rc); return 1; }
    t_prefill = TICK() - t_prefill;

    int kv_len = prompt_len;
    int next_token = sample_argmax(logits + (prompt_len - 1) * V, V);

    fprintf(stderr, "  Prefill: %.1f ms, next_token=%d ('%c')\n",
            t_prefill / 1000.0, next_token,
            next_token >= 32 && next_token < 127 ? (char)next_token : '?');

    /* ---- Phase 5: Decode ---- */
    fprintf(stderr, "\n[Phase 5] Decode (%d tokens max)...\n", max_new);
    int generated[128]; int n_gen = 0;
    generated[n_gen++] = next_token;

    double t_decode_total = 0;

    for (int step = 0; step < max_new; step++) {
        int tid[1] = { next_token };
        float *step_logits = (float *)malloc(V * sizeof(float));

        double t_step = TICK();
        rc = tr_forward(&m, tid, 1, kv_len, step_logits);
        t_step = TICK() - t_step;
        t_decode_total += t_step;
        kv_len++;

        next_token = sample_argmax(step_logits, V);
        free(step_logits);

        if (next_token <= 0 || next_token >= 127) break;
        generated[n_gen++] = next_token;
    }

    fprintf(stderr, "  Generated %d tokens in %.1f ms (%.1f ms/tok, %.0f tok/s)\n",
            n_gen, t_decode_total / 1000.0,
            n_gen > 0 ? t_decode_total / (1000.0 * n_gen) : 0,
            n_gen > 0 ? 1000000.0 * n_gen / t_decode_total : 0);

    /* Print generated text */
    fprintf(stderr, "  Output: \"%s", prompt);
    for (int i = 0; i < n_gen; i++)
        fprintf(stderr, "%c", generated[i]);
    fprintf(stderr, "\"\n");

    /* ---- Phase 6: Timing breakdown ---- */
    fprintf(stderr, "\n============================================================\n");
    fprintf(stderr, "  TIMING BREAKDOWN (avg over %d steps)\n", m.n_steps);
    fprintf(stderr, "============================================================\n");
    fprintf(stderr, "  %-25s %10s %10s\n", "Component", "us", "%");

    double t_total = m.t_embed + m.t_rms_attn + m.t_qkv
        + m.t_rope + m.t_kv_cache + m.t_scores + m.t_softmax + m.t_attn + m.t_wo
        + m.t_rms_ffn + m.t_ffn[0] + m.t_ffn[1] + m.t_ffn[2]
        + m.t_final_rms + m.t_lm_head;
    if (t_total < 1) t_total = 1;

#define TPRINT(name, val) fprintf(stderr, "  %-25s %10.0f %9.1f%%\n", \
        name, val, 100.0 * (val) / t_total)
    TPRINT("Embedding",           m.t_embed);
    TPRINT("RMS Norm (attn)",     m.t_rms_attn);
    TPRINT("QKV fused (TPU)",     m.t_qkv);
    TPRINT("RoPE",                m.t_rope);
    TPRINT("KV Cache store/load", m.t_kv_cache);
    TPRINT("Scores = QK^T (TPU)", m.t_scores);
    TPRINT("Softmax",             m.t_softmax);
    TPRINT("Attn = SV (TPU)",     m.t_attn);
    TPRINT("Wo (TPU)",            m.t_wo);
    TPRINT("RMS Norm (FFN)",      m.t_rms_ffn);
    TPRINT("FFN up (TPU)",        m.t_ffn[0]);
    TPRINT("FFN gate (TPU)",      m.t_ffn[1]);
    TPRINT("FFN down (TPU)",      m.t_ffn[2]);
    TPRINT("Final RMS Norm",      m.t_final_rms);
    TPRINT("LM Head (TPU)",       m.t_lm_head);
    TPRINT("--- TOTAL ---",       t_total);
#undef TPRINT

    fprintf(stderr, "\n  Prefill:  %.1f ms\n", t_prefill / 1000.0);
    fprintf(stderr, "  Decode:   %.1f ms/token\n",
            n_gen > 0 ? t_decode_total / (1000.0 * n_gen) : 0);

    /* ---- Phase 7: CPU FP32 Reference comparison (--ref flag) ---- */
    if (do_ref) {
    fprintf(stderr, "\n============================================================\n");
    fprintf(stderr, "  ACCURACY vs CPU FP32 Reference\n");
    fprintf(stderr, "============================================================\n");

    /* Build reference model from the same SD weight files */
    tr_ref_t ref; memset(&ref, 0, sizeof(ref));

    ref.embed = (float *)malloc(V * D * sizeof(float));
    snprintf(path, sizeof(path), "%s/embed.f32", base);
    read_file(path, ref.embed, V * D * 4);

    for (int l = 0; l < TR_N_LAYERS; l++) {
        ref.rms_attn[l] = (float *)malloc(D * sizeof(float));
        snprintf(path, sizeof(path), "%s/layer%d_rms_attn.f32", base, l);
        read_file(path, ref.rms_attn[l], D * 4);
        ref.rms_ffn[l] = (float *)malloc(D * sizeof(float));
        snprintf(path, sizeof(path), "%s/layer%d_rms_ffn.f32", base, l);
        read_file(path, ref.rms_ffn[l], D * 4);

        /* Load INT8 weights and dequant for FP32 ref */
        int8_t *tmp_i8 = (int8_t *)malloc(w_sz_DD);
        ref.Wq[l] = (float *)malloc(w_sz_DD * sizeof(float));
        snprintf(path, sizeof(path), "%s/layer%d_Wq.i8", base, l);
        read_file(path, tmp_i8, w_sz_DD);
        for (int i = 0; i < w_sz_DD; i++) ref.Wq[l][i] = (float)(int)tmp_i8[i] * 0.001f;
        ref.Wk[l] = (float *)malloc(w_sz_DD * sizeof(float));
        snprintf(path, sizeof(path), "%s/layer%d_Wk.i8", base, l);
        read_file(path, tmp_i8, w_sz_DD);
        for (int i = 0; i < w_sz_DD; i++) ref.Wk[l][i] = (float)(int)tmp_i8[i] * 0.001f;
        ref.Wv[l] = (float *)malloc(w_sz_DD * sizeof(float));
        snprintf(path, sizeof(path), "%s/layer%d_Wv.i8", base, l);
        read_file(path, tmp_i8, w_sz_DD);
        for (int i = 0; i < w_sz_DD; i++) ref.Wv[l][i] = (float)(int)tmp_i8[i] * 0.001f;
        ref.Wo[l] = (float *)malloc(w_sz_DD * sizeof(float));
        snprintf(path, sizeof(path), "%s/layer%d_Wo.i8", base, l);
        read_file(path, tmp_i8, w_sz_DD);
        for (int i = 0; i < w_sz_DD; i++) ref.Wo[l][i] = (float)(int)tmp_i8[i] * 0.001f;
        free(tmp_i8);

        tmp_i8 = (int8_t *)malloc(w_sz_DF);
        ref.ffn_up[l] = (float *)malloc(w_sz_DF * sizeof(float));
        snprintf(path, sizeof(path), "%s/layer%d_ffn_up.i8", base, l);
        read_file(path, tmp_i8, w_sz_DF);
        for (int i = 0; i < w_sz_DF; i++) ref.ffn_up[l][i] = (float)(int)tmp_i8[i] * 0.001f;
        ref.ffn_gate[l] = (float *)malloc(w_sz_DF * sizeof(float));
        snprintf(path, sizeof(path), "%s/layer%d_ffn_gate.i8", base, l);
        read_file(path, tmp_i8, w_sz_DF);
        for (int i = 0; i < w_sz_DF; i++) ref.ffn_gate[l][i] = (float)(int)tmp_i8[i] * 0.001f;
        free(tmp_i8);

        tmp_i8 = (int8_t *)malloc(w_sz_FD);
        ref.ffn_down[l] = (float *)malloc(w_sz_FD * sizeof(float));
        snprintf(path, sizeof(path), "%s/layer%d_ffn_down.i8", base, l);
        read_file(path, tmp_i8, w_sz_FD);
        for (int i = 0; i < w_sz_FD; i++) ref.ffn_down[l][i] = (float)(int)tmp_i8[i] * 0.001f;
        free(tmp_i8);
    }

    ref.final_rms = (float *)malloc(D * sizeof(float));
    snprintf(path, sizeof(path), "%s/final_rms.f32", base);
    read_file(path, ref.final_rms, D * 4);

    int8_t *lm_tmp = (int8_t *)malloc(V * D);
    ref.lm_head = (float *)malloc(V * D * sizeof(float));
    snprintf(path, sizeof(path), "%s/lm_head.i8", base);
    read_file(path, lm_tmp, V * D);
    for (int i = 0; i < V * D; i++) ref.lm_head[i] = (float)(int)lm_tmp[i] * 0.001f;
    free(lm_tmp);

    ref.rope_cos = (float *)malloc(TR_MAX_SEQ * (d / 2) * sizeof(float));
    ref.rope_sin = (float *)malloc(TR_MAX_SEQ * (d / 2) * sizeof(float));
    rope_precompute(TR_MAX_SEQ, d, ref.rope_cos, ref.rope_sin);

    /* Run reference */
    float *ref_logits = (float *)malloc(prompt_len * V * sizeof(float));
    tr_ref_forward(&ref, token_ids, prompt_len, ref_logits);

    /* Compare */
    float max_err = 0, mse = 0;
    for (int i = 0; i < prompt_len * V; i++) {
        float err = fabsf(logits[i] - ref_logits[i]);
        if (err > max_err) max_err = err;
        mse += err * err;
    }
    mse /= (float)(prompt_len * V);
    fprintf(stderr, "  Logits vs FP32 ref: max_err=%.6f  MSE=%.6f\n", max_err, mse);
    fprintf(stderr, "  Ref argmax: %d  TPU argmax: %d  %s\n",
            sample_argmax(ref_logits + (prompt_len - 1) * V, V),
            sample_argmax(logits + (prompt_len - 1) * V, V),
            sample_argmax(ref_logits + (prompt_len - 1) * V, V) ==
            sample_argmax(logits + (prompt_len - 1) * V, V) ? "MATCH" : "MISMATCH");

    /* Ref cleanup */
    free(ref.embed); free(ref.final_rms); free(ref.rope_cos); free(ref.rope_sin); free(ref.lm_head);
    for (int l = 0; l < TR_N_LAYERS; l++) {
        free(ref.rms_attn[l]); free(ref.rms_ffn[l]);
        free(ref.Wq[l]); free(ref.Wk[l]); free(ref.Wv[l]); free(ref.Wo[l]);
        free(ref.ffn_up[l]); free(ref.ffn_gate[l]); free(ref.ffn_down[l]);
    }
    free(ref_logits);
    } /* if (do_ref) */

    /* Cleanup TPU model */
    free(m.embed_f32); free(m.final_rms); free(m.rope_cos); free(m.rope_sin); free(m.lm_head);
    for (int l = 0; l < TR_N_LAYERS; l++) {
        free(m.rms_attn[l]); free(m.rms_ffn[l]);
        free(m.W_qkv[l]); free(m.Wo[l]);
        free(m.ffn_up[l]); free(m.ffn_gate[l]); free(m.ffn_down[l]);
    }
    free(token_ids); free(logits);
    tpu_cleanup(&ctx);

    fprintf(stderr, "\nDone.\n");
    return 0;
}
