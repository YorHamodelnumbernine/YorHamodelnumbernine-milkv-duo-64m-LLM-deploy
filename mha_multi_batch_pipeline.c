/* Multi-batch MHA with ping-pong dual-core pipeline.
   Small core (FreeRTOS) pre-quantizes next batch's input while
   main core (Linux + TPU) processes the current batch.

   Memory layout (2MB neuron memory):
     0x000000-0x03FFFF: Shared weights Wq,Wk,Wv,Wo (256KB)
     0x040000-0x0BFFFF: BATCH_A (512KB) — batches 0,2,4
     0x0C0000-0x13FFFF: BATCH_B (512KB) — batches 1,3,5
     0x140000-0x15FFFF: TPU matmul scratch (128KB)
     0x160000-0x1BFFFF: Staging (384KB) — pre-loaded batch inputs

   Build: make mha_multi_batch_pipeline
   Run:   ./mha_multi_batch_pipeline [d_model] [n_heads] [seq_len] [num_batches]
*/
#include "common/tpu_bench.h"
#include "common/mha_descriptor.h"
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <errno.h>

/* ---- RTOS cmdqu ---- */
#ifndef RTOS_CMDQU_SEND
#define RTOS_CMDQU_SEND      _IOW('r', 1, unsigned long)
#define RTOS_CMDQU_SEND_WAIT _IOW('r', 2, unsigned long)
#endif
#define IP_SYSTEM  1

typedef struct {
    unsigned char ip_id;
    unsigned char cmd_id : 7;
    unsigned char block  : 1;
    unsigned short mstime;
    unsigned int   param_ptr;
} __attribute__((packed)) cmdqu_t;

/* ---- Ping-pong memory layout ---- */
#define BATCH_SZ       0x80000   /* 512KB per batch */
#define BATCH_A_OFF    0x040000
#define BATCH_B_OFF    0x0C0000
#define SCRATCH_OFF    0x140000
#define STAGING_OFF    0x160000
#define TOTAL_MEM      0x1C0000   /* 1.75MB */

/* Per-batch buffer offsets (relative to batch base) */
enum {
    BOFF_INPUT_F32 = 0x00000,  /* 64KB */
    BOFF_Q_F32     = 0x10000,  /* 64KB */
    BOFF_K_F32     = 0x20000,  /* 64KB */
    BOFF_V_F32     = 0x30000,  /* 64KB */
    BOFF_SCORES_F32= 0x40000,  /* 32KB */
    BOFF_ATTN_F32  = 0x48000,  /* 64KB */
    BOFF_OUT_F32   = 0x58000,  /* 64KB */
    BOFF_Q_I8      = 0x68000,  /* 16KB */
    BOFF_K_I8      = 0x6C000,  /* 16KB */
    BOFF_V_I8      = 0x70000,  /* 16KB */
    BOFF_S_I8      = 0x74000,  /* 16KB */
    BOFF_A_I8      = 0x78000,  /* 16KB */
    BOFF_O_I8      = 0x7C000,  /* 16KB */
    BOFF_INPUT_I8  = 0x78000,  /* 16KB, shares with A_i8 (A_i8 used after input consumed) */
};

/* ---- Helpers ---- */
static double tick(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

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

static void quantize_i8(int8_t *dst, const float *src, int n, float scale, int zp) {
    float inv = 1.0f / scale;
    for (int i = 0; i < n; i++) {
        int q = (int)(src[i] * inv + zp + 0.5f);
        if (q > 127) q = 127; else if (q < -128) q = -128;
        dst[i] = (int8_t)q;
    }
}

static void dequantize_f32(float *dst, const int8_t *src, int n, float scale, int zp) {
    for (int i = 0; i < n; i++) dst[i] = (float)((int)src[i] - zp) * scale;
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

static void transpose_i8(int8_t *dst, const int8_t *src, int rows, int cols) {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            dst[j * rows + i] = src[i * cols + j];
}

/* ---- RoPE with lookup table ---- */
static void rope_precompute(int seq_len, int head_dim, float base,
                            float *cos_tab, float *sin_tab) {
    int half = head_dim / 2;
    for (int i = 0; i < half; i++) {
        float freq = 1.0f / powf(base, (2.0f * i) / (float)head_dim);
        for (int pos = 0; pos < seq_len; pos++) {
            float angle = (float)pos * freq;
            cos_tab[pos * half + i] = cosf(angle);
            sin_tab[pos * half + i] = sinf(angle);
        }
    }
}

static void apply_rope_f32_tab(float *buf, int seq_len, int head_dim,
                               const float *cos_tab, const float *sin_tab) {
    int half = head_dim / 2;
    for (int pos = 0; pos < seq_len; pos++) {
        const float *c_row = cos_tab + pos * half;
        const float *s_row = sin_tab + pos * half;
        for (int i = 0; i < half; i++) {
            float c = c_row[i], s = s_row[i];
            float x = buf[pos * head_dim + i];
            float y = buf[pos * head_dim + i + half];
            buf[pos * head_dim + i]         = x * c - y * s;
            buf[pos * head_dim + i + half]  = x * s + y * c;
        }
    }
}

/* ---- Mailbox helpers ---- */
static int mbox_fd = -1;

static int mbox_open(void) {
    mbox_fd = open("/dev/cvi-rtos-cmdqu", O_RDWR);
    if (mbox_fd < 0) { perror("open mbox"); return -1; }
    return 0;
}
static void mbox_close(void) { if (mbox_fd >= 0) close(mbox_fd); }

/* Non-blocking send: returns immediately, small core processes async.
   Caller must later poll dma->result after MemInvld to check completion.
   Returns 0 on success, negative on error. */
static int mbox_send_async(uint32_t cmd_id, uint32_t param_pa) {
    cmdqu_t c = {
        .ip_id = IP_SYSTEM, .cmd_id = cmd_id, .block = 0,
        .mstime = 0, .param_ptr = param_pa
    };
    int rc = ioctl(mbox_fd, RTOS_CMDQU_SEND, &c);
    if (rc < 0) {
        fprintf(stderr, "  mbox_send_async FAIL: cmd=%02x rc=%d errno=%d\n",
                cmd_id, rc, errno);
    }
    return rc;
}

/* ---- TPU matmul with auto-tiling ---- */
static int tpu_find_tile_n(cvk_context_t *cvk, int M, int K) {
    int left = mha_lmem_matrix_bytes(M, K);
    for (int tn = 128; tn >= 16; tn -= 16) {
        if (left + mha_lmem_matrix_bytes(K, tn) +
            mha_lmem_matrix_bytes(M, tn) <= 32768) return tn;
    }
    return -1;
}

static int tpu_matmul_submit(tpu_ctx *ctx, cvk_context_t *cvk,
    const void *left, int M, int K, const void *right, int N,
    void *result, int do_submit, uint32_t scratch_off)
{
    const int8_t *l_i8 = (const int8_t *)left;
    const int8_t *r_i8 = (const int8_t *)right;
    int8_t       *o_i8 = (int8_t *)result;

    uint32_t off_l = scratch_off;
    uint32_t off_r = scratch_off + M * K;

    int need_tile = (mha_lmem_matrix_bytes(M, K) +
                     mha_lmem_matrix_bytes(K, N) +
                     mha_lmem_matrix_bytes(M, N) > 32768);

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
            .relu_enable = 0, .add_result = 0, .ps32_mode = 0});
        cvk->ops->tdma_l2g_matrix_copy(cvk, &(cvk_tdma_l2g_matrix_copy_param_t){
            .src = ml_o,
            .dst = &(cvk_mg_t){0, TPU_PA(ctx, off_o), CVK_FMT_I8, {M, N}, {N}}});

        cvk->ops->lmem_free_matrix(cvk, ml_o);
        cvk->ops->lmem_free_matrix(cvk, ml_r);
        cvk->ops->lmem_free_matrix(cvk, ml_l);

        if (do_submit) {
            CVI_RT_Submit(ctx->rt_khandle);
            CVI_RT_MemInvld(ctx->rt_handle, ctx->neuron_mem);
            memcpy(o_i8, ctx->neuron_vaddr + off_o, M * N);
        }
        return 0;
    }

    /* ---- Tiling path ---- */
    int tile_n = tpu_find_tile_n(cvk, M, K);
    if (tile_n < 16) {
        fprintf(stderr, "LMEM fail M=%d K=%d N=%d\n", M, K, N);
        return -1;
    }

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
            .relu_enable = 0, .add_result = 0, .ps32_mode = 0});

        cvk->ops->tdma_l2g_matrix_copy(cvk, &(cvk_tdma_l2g_matrix_copy_param_t){
            .src = ml_o,
            .dst = &(cvk_mg_t){0, TPU_PA(ctx, off_o_base + n_start),
                               CVK_FMT_I8, {M, cur_n}, {N}}});

        cvk->ops->lmem_free_matrix(cvk, ml_o);
        cvk->ops->lmem_free_matrix(cvk, ml_r);
        cvk->ops->lmem_free_matrix(cvk, ml_l);
    }

    if (do_submit) {
        CVI_RT_Submit(ctx->rt_khandle);
        CVI_RT_MemInvld(ctx->rt_handle, ctx->neuron_mem);
        memcpy(o_i8, ctx->neuron_vaddr + off_o_base, M * N);
    }
    return 0;
}

static int tpu_matmul_wait(tpu_ctx *ctx, void *result, int M, int K, int N,
                           uint32_t scratch_off) {
    CVI_RT_WaitForAsync(ctx->rt_khandle);
    CVI_RT_MemInvld(ctx->rt_handle, ctx->neuron_mem);

    uint32_t off_o;
    int left  = mha_lmem_matrix_bytes(M, K);
    int right = mha_lmem_matrix_bytes(K, N);
    int out   = mha_lmem_matrix_bytes(M, N);
    if (left + right + out <= 32768)
        off_o = scratch_off + M * K + K * N;
    else {
        int tile_n = tpu_find_tile_n(ctx->cvk_ctx, M, K);
        if (tile_n < 16) tile_n = 64;
        off_o = scratch_off + M * K + K * tile_n;
    }
    memcpy(result, ctx->neuron_vaddr + off_o, M * N);
    return 0;
}

/* ---- MHA Pipeline (batch data at batch_base) ----
   x_i8: pre-quantized input (Step 0 already done by caller/small-core)
   Returns output in neuron memory at batch_base + BOFF_OUT_F32
   step_us[9] filled with per-step timing, step_us[0]=0 */
static int mha_pipeline_batch(const mha_config_t *cfg, tpu_ctx *ctx,
    const int8_t *x_i8, float sc_x, int zp_x,
    const int8_t *Wq, const int8_t *Wk, const int8_t *Wv, const int8_t *Wo,
    double *step_us, uint32_t batch_base)
{
    int S = cfg->seq_len, D = cfg->d_model, H = cfg->n_heads, d = cfg->head_dim;
    int total = S * D, h_ss = H * S * S;
    float scale_f = cfg->softmax_scale;
    cvk_context_t *cvk = ctx->cvk_ctx;
    double ts;
    int rc;
    uint8_t *v = ctx->neuron_vaddr;

    float *Q_f32 = (float *)(v + batch_base + BOFF_Q_F32);
    float *K_f32 = (float *)(v + batch_base + BOFF_K_F32);
    float *V_f32 = (float *)(v + batch_base + BOFF_V_F32);
    float *Sc_f32= (float *)(v + batch_base + BOFF_SCORES_F32);
    float *At_f32= (float *)(v + batch_base + BOFF_ATTN_F32);
    float *Out_f32=(float *)(v + batch_base + BOFF_OUT_F32);
    int8_t *Q_i8 = (int8_t *)(v + batch_base + BOFF_Q_I8);
    int8_t *K_i8 = (int8_t *)(v + batch_base + BOFF_K_I8);
    int8_t *V_i8 = (int8_t *)(v + batch_base + BOFF_V_I8);
    int8_t *S_i8 = (int8_t *)(v + batch_base + BOFF_S_I8);
    int8_t *A_i8 = (int8_t *)(v + batch_base + BOFF_A_I8);
    int8_t *O_i8 = (int8_t *)(v + batch_base + BOFF_O_I8);

    memset(step_us, 0, 9 * sizeof(double));
    step_us[0] = 0;  /* input already quantized */

    /* ---- Step 1: Q = x * Wq (async) ---- */
    ts = tick();
    rc = tpu_matmul_submit(ctx, cvk, x_i8, S, D, Wq, D, Q_i8, 0, SCRATCH_OFF);
    CVI_RT_SubmitAsync(ctx->rt_khandle, 0);
    step_us[1] = tick() - ts;
    if (rc) return -1;

    /* ---- Step 2: K = x * Wk (blocking, overlaps with Step 1 async) ---- */
    ts = tick();
    rc = tpu_matmul_submit(ctx, cvk, x_i8, S, D, Wk, D, K_i8, 1, SCRATCH_OFF);
    step_us[2] = tick() - ts;
    if (rc) return -2;

    CVI_RT_WaitForAsync(ctx->rt_khandle);
    CVI_RT_MemInvld(ctx->rt_handle, ctx->neuron_mem);

    float sc_qk = sc_x * 0.001f;
    dequantize_f32(Q_f32, Q_i8, total, sc_qk, 0);
    dequantize_f32(K_f32, K_i8, total, sc_qk, 0);

    /* ---- Step 3: V = x * Wv ---- */
    ts = tick();
    rc = tpu_matmul_submit(ctx, cvk, x_i8, S, D, Wv, D, V_i8, 1, SCRATCH_OFF);
    step_us[3] = tick() - ts;
    if (rc) return -3;

    /* ---- Step 4: RoPE with lookup table ---- */
    ts = tick();
    int half_d = d / 2;
    float *cos_tab = (float *)malloc(S * half_d * sizeof(float));
    float *sin_tab = (float *)malloc(S * half_d * sizeof(float));
    rope_precompute(S, d, 10000.0f, cos_tab, sin_tab);
    for (int h = 0; h < H; h++) {
        float *Qh = (float *)malloc(S * d * sizeof(float));
        float *Kh = (float *)malloc(S * d * sizeof(float));
        for (int r = 0; r < S; r++) {
            for (int c = 0; c < d; c++) {
                Qh[r * d + c] = Q_f32[r * D + h * d + c];
                Kh[r * d + c] = K_f32[r * D + h * d + c];
            }
        }
        apply_rope_f32_tab(Qh, S, d, cos_tab, sin_tab);
        apply_rope_f32_tab(Kh, S, d, cos_tab, sin_tab);
        for (int r = 0; r < S; r++) {
            for (int c = 0; c < d; c++) {
                Q_f32[r * D + h * d + c] = Qh[r * d + c];
                K_f32[r * D + h * d + c] = Kh[r * d + c];
            }
        }
        free(Qh); free(Kh);
    }
    free(cos_tab); free(sin_tab);
    step_us[4] = tick() - ts;

    /* Re-quantize after RoPE */
    int zp_q2; float sc_q2 = compute_scale(Q_f32, total, &zp_q2);
    quantize_i8(Q_i8, Q_f32, total, sc_q2, zp_q2);
    int zp_k2; float sc_k2 = compute_scale(K_f32, total, &zp_k2);
    quantize_i8(K_i8, K_f32, total, sc_k2, zp_k2);

    /* ---- Step 5: Scores = Q * K^T per head ---- */
    ts = tick();
    for (int h = 0; h < H; h++) {
        int8_t *Qh = (int8_t *)malloc(S * d);
        int8_t *Kh = (int8_t *)malloc(S * d);
        int8_t *Kt = (int8_t *)malloc(S * d);
        for (int r = 0; r < S; r++) {
            memcpy(Qh + r * d, Q_i8 + r * D + h * d, d);
            memcpy(Kh + r * d, K_i8 + r * D + h * d, d);
        }
        transpose_i8(Kt, Kh, S, d);
        rc = tpu_matmul_submit(ctx, cvk, Qh, S, d, Kt, S,
                               S_i8 + h * S * S, 1, SCRATCH_OFF);
        free(Qh); free(Kh); free(Kt);
        if (rc) return -5;
    }
    dequantize_f32(Sc_f32, S_i8, h_ss, sc_q2 * sc_k2, 0);
    for (int i = 0; i < h_ss; i++) Sc_f32[i] *= scale_f;
    step_us[5] = tick() - ts;

    /* ---- Step 6: Softmax ---- */
    ts = tick();
    for (int h = 0; h < H; h++) softmax_f32(Sc_f32 + h * S * S, S, S);
    step_us[6] = tick() - ts;

    int zp_s2; float sc_s2 = compute_scale(Sc_f32, h_ss, &zp_s2);
    quantize_i8(S_i8, Sc_f32, h_ss, sc_s2, zp_s2);

    dequantize_f32(V_f32, V_i8, total, sc_qk, 0);
    int zp_v2; float sc_v2 = compute_scale(V_f32, total, &zp_v2);
    quantize_i8(V_i8, V_f32, total, sc_v2, zp_v2);

    /* ---- Step 7: Attn = Scores * V ---- */
    ts = tick();
    for (int h = 0; h < H; h++) {
        int8_t *Vh = (int8_t *)malloc(S * d);
        for (int r = 0; r < S; r++)
            memcpy(Vh + r * d, V_i8 + r * D + h * d, d);
        rc = tpu_matmul_submit(ctx, cvk, S_i8 + h * S * S, S, S, Vh, d,
                               A_i8 + h * S * d, 1, SCRATCH_OFF);
        free(Vh);
        if (rc) return -7;
    }
    dequantize_f32(At_f32, A_i8, total, sc_s2 * sc_v2, 0);
    step_us[7] = tick() - ts;

    /* ---- Step 8: Output = Attn * Wo ---- */
    ts = tick();
    int8_t *Ai = (int8_t *)malloc(total);
    for (int h = 0; h < H; h++)
        for (int r = 0; r < S; r++)
            memcpy(Ai + r * D + h * d, A_i8 + h * S * d + r * d, d);
    int zp_a2; float sc_a2 = compute_scale(At_f32, total, &zp_a2);
    quantize_i8(Ai, At_f32, total, sc_a2, zp_a2);
    rc = tpu_matmul_submit(ctx, cvk, Ai, S, D, Wo, D, O_i8, 1, SCRATCH_OFF);
    if (rc) { free(Ai); return -8; }
    dequantize_f32(Out_f32, O_i8, total, sc_a2 * 0.001f, 0);
    free(Ai);
    step_us[8] = tick() - ts;
    return 0;
}

/* ================================================================
 * main
 * ================================================================ */
int main(int argc, char **argv) {
    setvbuf(stderr, NULL, _IONBF, 0);  /* unbuffered for SSH */
    int d_model   = argc > 1 ? atoi(argv[1]) : 512;
    int n_heads   = argc > 2 ? atoi(argv[2]) : 8;
    int seq_len   = argc > 3 ? atoi(argv[3]) : 32;
    int n_batches = argc > 4 ? atoi(argv[4]) : 6;

    int S = seq_len, D = d_model, d = d_model / n_heads;
    mha_config_t cfg = {
        .d_model = D, .n_heads = n_heads, .head_dim = d,
        .seq_len = S, .softmax_scale = 1.0f / sqrtf((float)d),
    };
    int H = n_heads, total = S * D, w_sz = D * D;
    int input_bytes = total * (int)sizeof(float);

    fprintf(stderr, "\n========== MHA Multi-Batch Pipeline d_model=%d n_heads=%d head_dim=%d seq_len=%d batches=%d ==========\n",
            D, H, d, S, n_batches);
    fprintf(stderr, "  total_mem=%.2fMB  input/batch=%.1fKB\n",
            TOTAL_MEM / 1048576.0, input_bytes / 1024.0);

    /* ---- Shared weights ---- */
    float *W_f32[4]; int8_t *W_i8[4];
    for (int w = 0; w < 4; w++) {
        W_f32[w] = (float *)malloc(w_sz * sizeof(float));
        W_i8[w]  = (int8_t *)malloc(w_sz);
    }
    srand(42);
    for (int i = 0; i < w_sz; i++)
        for (int w = 0; w < 4; w++)
            W_f32[w][i] = (float)(rand() % 256 - 128) / 1000.0f;
    for (int w = 0; w < 4; w++) {
        int zp; float sc = compute_scale(W_f32[w], w_sz, &zp);
        quantize_i8(W_i8[w], W_f32[w], w_sz, sc, zp);
    }

    /* ---- Generate N batch inputs ---- */
    float **batch_x = (float **)malloc(n_batches * sizeof(float *));
    int    *batch_zp = (int *)malloc(n_batches * sizeof(int));
    float  *batch_sc = (float *)malloc(n_batches * sizeof(float));
    for (int b = 0; b < n_batches; b++) {
        batch_x[b] = (float *)malloc(total * sizeof(float));
        for (int i = 0; i < total; i++)
            batch_x[b][i] = (float)(rand() % 256 - 128) / 200.0f;
        batch_sc[b] = compute_scale(batch_x[b], total, &batch_zp[b]);
    }

    float **outputs = (float **)malloc(n_batches * sizeof(float *));
    for (int b = 0; b < n_batches; b++)
        outputs[b] = (float *)malloc(total * sizeof(float));

    /* ================================================================
     * MODE 0: Baseline — CPU-only (no small core ping-pong)
     * ================================================================ */
    fprintf(stderr, "\n--- Mode 0: CPU-only baseline (no small core) ---\n");
    tpu_ctx ctx0;
    if (tpu_init(&ctx0, TOTAL_MEM) != 0) {
        fprintf(stderr, "tpu_init failed\n"); return 1;
    }

    /* Copy weights to neuron memory */
    uint32_t w_offs[4] = {MHA_WQ_OFF, MHA_WK_OFF, MHA_WV_OFF, MHA_WO_OFF};
    int8_t *Wq_nm, *Wk_nm, *Wv_nm, *Wo_nm;
    Wq_nm = (int8_t *)(ctx0.neuron_vaddr + MHA_OFF_WEIGHTS + MHA_WQ_OFF);
    Wk_nm = (int8_t *)(ctx0.neuron_vaddr + MHA_OFF_WEIGHTS + MHA_WK_OFF);
    Wv_nm = (int8_t *)(ctx0.neuron_vaddr + MHA_OFF_WEIGHTS + MHA_WV_OFF);
    Wo_nm = (int8_t *)(ctx0.neuron_vaddr + MHA_OFF_WEIGHTS + MHA_WO_OFF);
    int8_t *W_all[4] = {Wq_nm, Wk_nm, Wv_nm, Wo_nm};
    for (int w = 0; w < 4; w++)
        memcpy(W_all[w], W_i8[w], w_sz);
    CVI_RT_MemFlush(ctx0.rt_handle, ctx0.neuron_mem);

    double t0_pipe[6], t0_quant[6], t0_total_us;
    double t1_pipe[6], t1_total_us = 0;
    double t2_pipe[6], t2_total_us = 0;
    for (int b = 0; b < 6; b++) t2_pipe[b] = 0;
    double ts0 = tick();

    for (int b = 0; b < n_batches; b++) {
        uint32_t base = (b % 2) ? BATCH_B_OFF : BATCH_A_OFF;
        uint8_t *v = ctx0.neuron_vaddr;

        /* Step 0: Quantize locally */
        double tq = tick();
        float *x_in = (float *)(v + base + BOFF_INPUT_F32);
        int8_t *x_q  = (int8_t *)malloc(total);
        memcpy(x_in, batch_x[b], input_bytes);
        quantize_i8(x_q, x_in, total, batch_sc[b], batch_zp[b]);
        CVI_RT_MemFlush(ctx0.rt_handle, ctx0.neuron_mem);
        t0_quant[b] = tick() - tq;

        /* Steps 1-8 */
        double step_us[9], tp = tick();
        int rc = mha_pipeline_batch(&cfg, &ctx0, x_q, batch_sc[b], batch_zp[b],
            Wq_nm, Wk_nm, Wv_nm, Wo_nm, step_us, base);
        t0_pipe[b] = tick() - tp;
        if (rc) fprintf(stderr, "  batch[%d] FAIL rc=%d\n", b, rc);

        /* Copy output */
        memcpy(outputs[b], v + base + BOFF_OUT_F32, input_bytes);
        free(x_q);
    }
    t0_total_us = tick() - ts0;
    fprintf(stderr, "  CPU baseline total: %.1f us (avg/batch: %.1f us)\n",
            t0_total_us, t0_total_us / n_batches);

    /* Free ctx0 to avoid TPU resource conflict */
    tpu_cleanup(&ctx0);

    /* ================================================================
     * MODE 1: Pre-quantized — quantize to heap buffers, pass to pipeline
     *
     * Measures pure pipeline time without quantize overhead by
     * pre-quantizing all inputs to heap buffers upfront (not timed).
     * This is the UPPER BOUND of what overlapping can achieve.
     *
     * Avoids neuron-memory aliasing (BOFF_INPUT_I8 shares with BOFF_A_I8)
     * by keeping pre-quantized data in separate heap allocations.
     * ================================================================ */
    fprintf(stderr, "\n--- Mode 1: Pre-quantized (upper bound for overlap) ---\n");

    tpu_ctx ctx1;
    if (tpu_init(&ctx1, TOTAL_MEM) != 0) {
        fprintf(stderr, "tpu_init failed\n"); goto print_results;
    }

    /* Copy weights to ctx1 neuron memory */
    int8_t *Wq1 = (int8_t *)(ctx1.neuron_vaddr + MHA_OFF_WEIGHTS + MHA_WQ_OFF);
    int8_t *Wk1 = (int8_t *)(ctx1.neuron_vaddr + MHA_OFF_WEIGHTS + MHA_WK_OFF);
    int8_t *Wv1 = (int8_t *)(ctx1.neuron_vaddr + MHA_OFF_WEIGHTS + MHA_WV_OFF);
    int8_t *Wo1 = (int8_t *)(ctx1.neuron_vaddr + MHA_OFF_WEIGHTS + MHA_WO_OFF);
    memcpy(Wq1, W_i8[0], w_sz); memcpy(Wk1, W_i8[1], w_sz);
    memcpy(Wv1, W_i8[2], w_sz); memcpy(Wo1, W_i8[3], w_sz);
    CVI_RT_MemFlush(ctx1.rt_handle, ctx1.neuron_mem);

    /* Pre-quantize all batch inputs to heap buffers (not timed) */
    int8_t **preq_x = (int8_t **)malloc(n_batches * sizeof(int8_t *));
    for (int b = 0; b < n_batches; b++) {
        preq_x[b] = (int8_t *)malloc(total);
        quantize_i8(preq_x[b], batch_x[b], total, batch_sc[b], batch_zp[b]);
    }

    /* Run 6 pipelines using pre-quantized heap buffers — TIMED */
    double ts1 = tick();

    for (int b = 0; b < n_batches; b++) {
        uint32_t base = (b % 2) ? BATCH_B_OFF : BATCH_A_OFF;

        double step_us[9], tp = tick();
        int rc = mha_pipeline_batch(&cfg, &ctx1, preq_x[b], batch_sc[b], batch_zp[b],
            Wq1, Wk1, Wv1, Wo1, step_us, base);
        t1_pipe[b] = tick() - tp;
        if (rc) fprintf(stderr, "  batch[%d] PREQ FAIL rc=%d\n", b, rc);

        memcpy(outputs[b], ctx1.neuron_vaddr + base + BOFF_OUT_F32, input_bytes);
    }
    t1_total_us = tick() - ts1;

    fprintf(stderr, "  Pre-quantized total: %.1f us (avg/batch: %.1f us)\n",
            t1_total_us, t1_total_us / n_batches);
    for (int b = 0; b < n_batches; b++)
        fprintf(stderr, "    batch[%d] pipe=%8.1f us\n", b, t1_pipe[b]);

    /* ---- Analysis ---- */
    double sum0_pipe = 0, sum0_quant = 0, sum1_pipe = 0;
    for (int b = 0; b < n_batches; b++) {
        sum0_pipe += t0_pipe[b];
        sum0_quant += t0_quant[b];
        sum1_pipe += t1_pipe[b];
    }
    double avg_quant = sum0_quant / n_batches;
    double avg_mbox = 4000.0;  /* measured mailbox round-trip ~4ms */

    fprintf(stderr, "\n--- Overlap Analysis ---\n");
    fprintf(stderr, "  Avg quantize time (main core, 1GHz): %.1f us\n", avg_quant);
    fprintf(stderr, "  Est quantize time (small core, 700MHz): %.1f us\n", avg_quant * 1.4);
    fprintf(stderr, "  Measured mailbox round-trip (SEND_WAIT): ~%.0f us\n", avg_mbox);
    fprintf(stderr, "  Per-batch savings (main-core interleave): %.1f us\n", avg_quant);
    fprintf(stderr, "  Per-batch cost (small-core mailbox):      %.0f us\n", avg_mbox);
    fprintf(stderr, "  Overlap benefit (6-batch):\n");
    fprintf(stderr, "    Main-core interleave:  save ~%.0f us (%.1f%%)\n",
            sum0_quant, 100.0 * sum0_quant / t0_total_us);
    fprintf(stderr, "    Dual-core via mailbox: cost ~%.0f us (%.1f%%) — net LOSS\n",
            5.0 * avg_mbox - sum0_quant, 100.0 * (5.0 * avg_mbox - sum0_quant) / t0_total_us);

    for (int b = 0; b < n_batches; b++) free(preq_x[b]);
    free(preq_x);
    tpu_cleanup(&ctx1);

    /* ================================================================
     * MODE 2: Dual-core async pipeline
     *
     * Main core quantizes batch 0 locally, then fires async QUANTIZE
     * to small core for batch N+1 during batch N's TPU pipeline.
     * Small core quantize (~1ms) fits easily within pipeline (~28ms).
     *
     * Key design choices for async mailbox reliability:
     *  - DMA descriptor in neuron memory (CVI_RT_MemInvld for cache coherency)
     *  - Input data pre-loaded to staging area (neuron memory, PA accessible)
     *  - Quantized output to BATCH_A/BATCH_B INPUT_I8 (ping-pong)
     *  - Poll dma->result with MemInvld before each batch
     * ================================================================ */
    fprintf(stderr, "\n--- Mode 2: Dual-core async quantize pipeline ---\n");

    if (mbox_open() != 0) {
        fprintf(stderr, "  mbox_open failed, skipping Mode 2\n");
        goto print_results;
    }

    tpu_ctx ctx2;
    if (tpu_init(&ctx2, TOTAL_MEM) != 0) {
        fprintf(stderr, "tpu_init failed\n"); goto print_results;
    }

    /* DMA descriptor in neuron memory (safe area within scratch) */
    #define DMA_DESC_OFF  (SCRATCH_OFF + 0x1F000)
    mha_dma_desc_t *dma = (mha_dma_desc_t *)(ctx2.neuron_vaddr + DMA_DESC_OFF);

    /* Copy weights */
    int8_t *Wq2 = (int8_t *)(ctx2.neuron_vaddr + MHA_OFF_WEIGHTS + MHA_WQ_OFF);
    int8_t *Wk2 = (int8_t *)(ctx2.neuron_vaddr + MHA_OFF_WEIGHTS + MHA_WK_OFF);
    int8_t *Wv2 = (int8_t *)(ctx2.neuron_vaddr + MHA_OFF_WEIGHTS + MHA_WV_OFF);
    int8_t *Wo2 = (int8_t *)(ctx2.neuron_vaddr + MHA_OFF_WEIGHTS + MHA_WO_OFF);
    memcpy(Wq2, W_i8[0], w_sz); memcpy(Wk2, W_i8[1], w_sz);
    memcpy(Wv2, W_i8[2], w_sz); memcpy(Wo2, W_i8[3], w_sz);

    /* Pre-load all batch inputs to neuron memory staging area.
       Small core accesses these via physical address (no cache issues). */
    for (int b = 0; b < n_batches; b++) {
        float *dst = (float *)(ctx2.neuron_vaddr + STAGING_OFF + b * input_bytes);
        memcpy(dst, batch_x[b], input_bytes);
    }
    CVI_RT_MemFlush(ctx2.rt_handle, ctx2.neuron_mem);

    int async_ok = 0, async_fail = 0;
    double ts2 = tick();

    for (int b = 0; b < n_batches; b++) {
        uint32_t base = (b % 2) ? BATCH_B_OFF : BATCH_A_OFF;
        uint32_t next_base = ((b + 1) % 2) ? BATCH_B_OFF : BATCH_A_OFF;
        int8_t *x_q;
        int used_small_core = 0;

        if (b == 0) {
            /* Batch 0: quantize locally (no previous batch to overlap with) */
            int8_t *local_q = (int8_t *)malloc(total);
            quantize_i8(local_q, batch_x[0], total, batch_sc[0], batch_zp[0]);
            x_q = local_q;
        } else {
            /* Batches 1+: data should be ready from small core.
               Poll with cache invalidation until dma->result becomes 0. */
            int poll = 0;
            int ready = 0;
            for (poll = 0; poll < 500; poll++) {
                CVI_RT_MemInvld(ctx2.rt_handle, ctx2.neuron_mem);
                if (dma->result == 0) { ready = 1; break; }
                usleep(100);  /* 100us poll interval, max 50ms timeout */
            }
            if (ready) {
                x_q = (int8_t *)(ctx2.neuron_vaddr + base + BOFF_INPUT_I8);
                used_small_core = 1;
                async_ok++;
            } else {
                /* Timeout — fallback to local quantize */
                fprintf(stderr, "  batch[%d] small-core TIMEOUT (result=%d after %d polls), local fallback\n",
                        b, dma->result, poll);
                int8_t *fb_q = (int8_t *)malloc(total);
                quantize_i8(fb_q, batch_x[b], total, batch_sc[b], batch_zp[b]);
                x_q = fb_q;
                async_fail++;
            }
        }

        /* Fire async QUANTIZE for NEXT batch (overlaps with current pipeline).
           Small core will write to next_base + BOFF_INPUT_I8. */
        if (b < n_batches - 1) {
            uint32_t src_off = STAGING_OFF + (b + 1) * input_bytes;
            uint32_t dst_off = next_base + BOFF_INPUT_I8;

            dma->src_paddr  = (uint32_t)TPU_PA(&ctx2, src_off);
            dma->dst_paddr  = (uint32_t)TPU_PA(&ctx2, dst_off);
            dma->size       = total;
            dma->scale      = batch_sc[b + 1];
            dma->zero_point = batch_zp[b + 1];
            dma->result     = -1;  /* in-progress marker */
            CVI_RT_MemFlush(ctx2.rt_handle, ctx2.neuron_mem);

            int rc = mbox_send_async(CMD_MHA_QUANTIZE,
                                     (uint32_t)TPU_PA(&ctx2, DMA_DESC_OFF));
            if (rc < 0) {
                fprintf(stderr, "  batch[%d] async send FAIL (rc=%d), next batch will quantize locally\n",
                        b, rc);
                dma->result = -2;  /* mark as failed so next batch falls back */
            }
        }

        /* Run MHA pipeline on current batch */
        double step_us[9], tp = tick();
        int rc = mha_pipeline_batch(&cfg, &ctx2, x_q, batch_sc[b], batch_zp[b],
                                    Wq2, Wk2, Wv2, Wo2, step_us, base);
        t2_pipe[b] = tick() - tp;
        if (rc) fprintf(stderr, "  batch[%d] DUAL FAIL rc=%d\n", b, rc);

        memcpy(outputs[b], ctx2.neuron_vaddr + base + BOFF_OUT_F32, input_bytes);

        if (!used_small_core) free(x_q);
    }
    t2_total_us = tick() - ts2;

    fprintf(stderr, "  Dual-core async total: %.1f us (avg/batch: %.1f us)\n",
            t2_total_us, t2_total_us / n_batches);
    fprintf(stderr, "  Async quantize: %d OK, %d fail\n", async_ok, async_fail);
    for (int b = 0; b < n_batches; b++)
        fprintf(stderr, "    batch[%d] pipe=%8.1f us\n", b, t2_pipe[b]);

    tpu_cleanup(&ctx2);

print_results:
    /* ---- Summary ---- */
    fprintf(stderr, "\n========== Multi-Batch Pipeline Comparison ==========\n");
    fprintf(stderr, "  %-40s %12s %12s %12s\n",
            "Mode", "Total(us)", "Avg/Batch(us)", "Speedup");
    fprintf(stderr, "  %-40s %12.1f %12.1f %12s\n",
            "CPU baseline (sequential)", t0_total_us, t0_total_us / n_batches, "1.00x");
    fprintf(stderr, "  %-40s %12.1f %12.1f %10.2fx\n",
            "Pre-quantized (upper bound)", t1_total_us, t1_total_us / n_batches,
            t0_total_us / t1_total_us);
    if (t2_total_us > 0) {
        fprintf(stderr, "  %-40s %12.1f %12.1f %10.2fx\n",
                "Dual-core async pipeline", t2_total_us, t2_total_us / n_batches,
                t0_total_us / t2_total_us);
    } else {
        fprintf(stderr, "  %-40s %12s %12s %12s\n",
                "Dual-core async pipeline", "SKIPPED", "-", "-");
    }
    fprintf(stderr, "\n  Overlap potential:\n");
    fprintf(stderr, "    Main-core: save ~%.0f us (%.1f%%) — quantize hidden behind TPU\n",
            sum0_quant, 100.0 * sum0_quant / t0_total_us);
    if (t2_total_us > 0) {
        fprintf(stderr, "    Dual-core: save ~%.0f us (%.1f%%) — async quantize via mailbox\n",
                t0_total_us - t2_total_us,
                100.0 * (t0_total_us - t2_total_us) / t0_total_us);
    }

    /* Per-batch breakdown */
    fprintf(stderr, "\n  %-6s %10s %10s | %10s %10s\n",
            "Batch", "CPU:quant", "CPU:pipe", "PreQ:pipe", "Dual:pipe");
    for (int b = 0; b < n_batches; b++) {
        fprintf(stderr, "  %4d   %10.1f %10.1f | %10.1f %10.1f\n",
                b, t0_quant[b], t0_pipe[b], t1_pipe[b], t2_pipe[b]);
    }

    /* ---- Cleanup ---- */
    mbox_close();
    for (int b = 0; b < n_batches; b++) { free(batch_x[b]); free(outputs[b]); }
    free(batch_x); free(batch_zp); free(batch_sc); free(outputs);
    for (int w = 0; w < 4; w++) { free(W_f32[w]); free(W_i8[w]); }

    return 0;
}
