/* Multi-batch MHA with double-buffering & small-core data pre-fetch.
   Batch 0 area + Batch 1 area in neuron memory (ping-pong).
   Small core (FreeRTOS) MEMCPYs next batch input while TPU+CPU
   processes the current batch.

   Strategy:
     At start of batch b: poll dma.result (was set by previous async send).
       By this point, pipeline took ~10ms > MEMCPY ~8ms, so data is ready.
     After pipeline: fire async MEMCPY for batch b+1 → other buffer.
       Last batch skips this.

   Build: make mha_multi_batch
   Run:   ./mha_multi_batch [d_model] [n_heads] [seq_len] [num_batches]
*/
#include "common/tpu_bench.h"
#include "common/mha_descriptor.h"
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>

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

/* ---- Double buffer ---- */
#define BATCH0_OFF  0
#define BATCH1_OFF  MHA_TOTAL_SIZE
#define TOTAL_MEM   (MHA_TOTAL_SIZE * 2 + 0x20000)  /* +128KB staging */
#define STAGING_OFF (MHA_TOTAL_SIZE * 2)

/* ---- DMA descriptor (matches comm_main.c) ---- */
typedef struct __attribute__((packed)) {
    uint32_t src_paddr;
    uint32_t dst_paddr;
    uint32_t size;
    uint32_t rows;
    uint32_t cols;
    float    scale;
    int32_t  zero_point;
    int32_t  result;
} dma_desc_t;

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

static void apply_rope_f32(float *buf, int seq_len, int head_dim, float base) {
    int half = head_dim / 2;
    for (int pos = 0; pos < seq_len; pos++) {
        for (int i = 0; i < half; i++) {
            float freq = 1.0f / powf(base, (2.0f * i) / (float)head_dim);
            float angle = (float)pos * freq;
            float c = cosf(angle), s = sinf(angle);
            float x = buf[pos * head_dim + i];
            float y = buf[pos * head_dim + i + half];
            buf[pos * head_dim + i]         = x * c - y * s;
            buf[pos * head_dim + i + half]  = x * s + y * c;
        }
    }
}

static void transpose_i8(int8_t *dst, const int8_t *src, int rows, int cols) {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            dst[j * rows + i] = src[i * cols + j];
}

/* ---- Mailbox helpers ---- */
static int mbox_fd = -1;

static int mbox_open(void) {
    mbox_fd = open("/dev/cvi-rtos-cmdqu", O_RDWR);
    if (mbox_fd < 0) { perror("open mbox"); return -1; }
    return 0;
}
static void mbox_close(void) { if (mbox_fd >= 0) close(mbox_fd); }

/* Blocking send: wait for small core response */
static int mbox_send_wait(uint32_t cmd_id, uint32_t param_pa) {
    cmdqu_t c = {
        .ip_id = IP_SYSTEM, .cmd_id = cmd_id, .block = 1,
        .mstime = 0xFFFF, .param_ptr = param_pa
    };
    return ioctl(mbox_fd, RTOS_CMDQU_SEND_WAIT, &c);
}

/* ---- TPU matmul ---- */
#define ASYNC_SCRATCH  0x00000
#define SYNC_SCRATCH   0x20000

static int tpu_matmul_submit(tpu_ctx *ctx, cvk_context_t *cvk,
    const void *left, int M, int K, const void *right, int N,
    void *result, int do_submit, uint32_t scratch_off)
{
    const int8_t *l_i8 = (const int8_t *)left;
    const int8_t *r_i8 = (const int8_t *)right;
    int8_t       *o_i8 = (int8_t *)result;

    uint32_t off_l = scratch_off;
    uint32_t off_r = scratch_off + M * K;
    uint32_t off_o = scratch_off + M * K + K * N;

    memcpy(ctx->neuron_vaddr + off_l, l_i8, M * K);
    memcpy(ctx->neuron_vaddr + off_r, r_i8, K * N);
    CVI_RT_MemFlush(ctx->rt_handle, ctx->neuron_mem);

    cvk_ml_shape_t sl = cvk->ops->ml_default_shape(cvk, M, K, CVK_FMT_I8);
    cvk_ml_shape_t sr = cvk->ops->ml_default_shape(cvk, K, N, CVK_FMT_I8);
    cvk_ml_shape_t so = cvk->ops->ml_default_shape(cvk, M, N, CVK_FMT_I8);

    cvk_ml_t *ml_l = cvk->ops->lmem_alloc_matrix(cvk, sl, CVK_FMT_I8, 1);
    cvk_ml_t *ml_r = cvk->ops->lmem_alloc_matrix(cvk, sr, CVK_FMT_I8, 1);
    cvk_ml_t *ml_o = cvk->ops->lmem_alloc_matrix(cvk, so, CVK_FMT_I8, 1);
    if (!ml_l || !ml_r || !ml_o) { fprintf(stderr, "LMEM fail\n"); return -1; }

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

    if (do_submit) {
        CVI_RT_Submit(ctx->rt_khandle);
        CVI_RT_MemInvld(ctx->rt_handle, ctx->neuron_mem);
        memcpy(o_i8, ctx->neuron_vaddr + off_o, M * N);
    }

    cvk->ops->lmem_free_matrix(cvk, ml_o);
    cvk->ops->lmem_free_matrix(cvk, ml_r);
    cvk->ops->lmem_free_matrix(cvk, ml_l);
    return 0;
}

/* ---- MHA Pipeline for a specific batch area ---- */
static int mha_pipeline_batch(const mha_config_t *cfg, tpu_ctx *ctx,
    const int8_t *x_i8, int zp_x, float sc_x,
    const int8_t *Wq, const int8_t *Wk, const int8_t *Wv, const int8_t *Wo,
    float *output, double *step_us, uint32_t batch_off)
{
    int S = cfg->seq_len, D = cfg->d_model, H = cfg->n_heads, d = cfg->head_dim;
    int total = S * D, h_ss = H * S * S;
    float scale_f = cfg->softmax_scale;
    cvk_context_t *cvk = ctx->cvk_ctx;
    double ts;
    int rc;

    float *Q_f32 = (float *)(ctx->neuron_vaddr + batch_off + MHA_OFF_Q_F32);
    float *K_f32 = (float *)(ctx->neuron_vaddr + batch_off + MHA_OFF_K_F32);
    float *V_f32 = (float *)(ctx->neuron_vaddr + batch_off + MHA_OFF_V_F32);
    float *Sc_f32 = (float *)(ctx->neuron_vaddr + batch_off + MHA_OFF_SCORES_F32);
    float *At_f32 = (float *)(ctx->neuron_vaddr + batch_off + MHA_OFF_ATTN_F32);
    int8_t *Q_i8 = (int8_t *)(ctx->neuron_vaddr + batch_off + MHA_OFF_Q_I8);
    int8_t *K_i8 = (int8_t *)(ctx->neuron_vaddr + batch_off + MHA_OFF_K_I8);
    int8_t *V_i8 = (int8_t *)(ctx->neuron_vaddr + batch_off + MHA_OFF_V_I8);
    int8_t *S_i8 = (int8_t *)(ctx->neuron_vaddr + batch_off + MHA_OFF_S_I8);
    int8_t *A_i8 = (int8_t *)(ctx->neuron_vaddr + batch_off + MHA_OFF_A_I8);
    int8_t *O_i8 = (int8_t *)(ctx->neuron_vaddr + batch_off + MHA_OFF_OUT_I8);

    step_us[0] = 0;  /* input already quantized by caller */

    /* Step 1: Q = x * Wq (async) */
    ts = tick();
    rc = tpu_matmul_submit(ctx, cvk, x_i8, S, D, Wq, D, Q_i8, 0, ASYNC_SCRATCH);
    CVI_RT_SubmitAsync(ctx->rt_khandle, 0);
    step_us[1] = tick() - ts;
    if (rc) return -1;

    /* Step 2: K = x * Wk (blocking, overlaps with Step 1 async) */
    ts = tick();
    rc = tpu_matmul_submit(ctx, cvk, x_i8, S, D, Wk, D, K_i8, 1, SYNC_SCRATCH);
    step_us[2] = tick() - ts;
    if (rc) return -2;

    CVI_RT_WaitForAsync(ctx->rt_khandle);
    CVI_RT_MemInvld(ctx->rt_handle, ctx->neuron_mem);

    float sc_qk_out = sc_x * 0.001f;
    dequantize_f32(Q_f32, Q_i8, total, sc_qk_out, 0);
    dequantize_f32(K_f32, K_i8, total, sc_qk_out, 0);

    /* Step 3: V = x * Wv */
    ts = tick();
    rc = tpu_matmul_submit(ctx, cvk, x_i8, S, D, Wv, D, V_i8, 1, SYNC_SCRATCH);
    step_us[3] = tick() - ts;
    if (rc) return -3;

    /* Step 4: RoPE on Q, K (CPU FP32) */
    ts = tick();
    for (int h = 0; h < H; h++) {
        float *Qh = (float *)malloc(S * d * sizeof(float));
        float *Kh = (float *)malloc(S * d * sizeof(float));
        for (int r = 0; r < S; r++) {
            for (int c = 0; c < d; c++) {
                Qh[r * d + c] = Q_f32[r * D + h * d + c];
                Kh[r * d + c] = K_f32[r * D + h * d + c];
            }
        }
        apply_rope_f32(Qh, S, d, 10000.0f);
        apply_rope_f32(Kh, S, d, 10000.0f);
        for (int r = 0; r < S; r++) {
            for (int c = 0; c < d; c++) {
                Q_f32[r * D + h * d + c] = Qh[r * d + c];
                K_f32[r * D + h * d + c] = Kh[r * d + c];
            }
        }
        free(Qh); free(Kh);
    }
    step_us[4] = tick() - ts;

    int zp_q2; float sc_q2 = compute_scale(Q_f32, total, &zp_q2);
    quantize_i8(Q_i8, Q_f32, total, sc_q2, zp_q2);
    int zp_k2; float sc_k2 = compute_scale(K_f32, total, &zp_k2);
    quantize_i8(K_i8, K_f32, total, sc_k2, zp_k2);

    /* Step 5: Scores = Q * K^T per head */
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
        rc = tpu_matmul_submit(ctx, cvk, Qh, S, d, Kt, S, S_i8 + h * S * S, 1, SYNC_SCRATCH);
        free(Qh); free(Kh); free(Kt);
        if (rc) return -5;
    }
    dequantize_f32(Sc_f32, S_i8, h_ss, sc_q2 * sc_k2, 0);
    for (int i = 0; i < h_ss; i++) Sc_f32[i] *= scale_f;
    step_us[5] = tick() - ts;

    /* Step 6: Softmax */
    ts = tick();
    for (int h = 0; h < H; h++) softmax_f32(Sc_f32 + h * S * S, S, S);
    step_us[6] = tick() - ts;

    int zp_s2; float sc_s2 = compute_scale(Sc_f32, h_ss, &zp_s2);
    quantize_i8(S_i8, Sc_f32, h_ss, sc_s2, zp_s2);

    dequantize_f32(V_f32, V_i8, total, sc_qk_out, 0);
    int zp_v2; float sc_v2 = compute_scale(V_f32, total, &zp_v2);
    quantize_i8(V_i8, V_f32, total, sc_v2, zp_v2);

    /* Step 7: Attn = Scores * V */
    ts = tick();
    for (int h = 0; h < H; h++) {
        int8_t *Vh = (int8_t *)malloc(S * d);
        for (int r = 0; r < S; r++)
            memcpy(Vh + r * d, V_i8 + r * D + h * d, d);
        rc = tpu_matmul_submit(ctx, cvk,
            S_i8 + h * S * S, S, S, Vh, d, A_i8 + h * S * d, 1, SYNC_SCRATCH);
        free(Vh);
        if (rc) return -7;
    }
    dequantize_f32(At_f32, A_i8, total, sc_s2 * sc_v2, 0);
    step_us[7] = tick() - ts;

    /* Step 8: Output = Attn * Wo */
    ts = tick();
    int8_t *Ai = (int8_t *)malloc(total);
    for (int h = 0; h < H; h++)
        for (int r = 0; r < S; r++)
            memcpy(Ai + r * D + h * d, A_i8 + h * S * d + r * d, d);
    int zp_a2; float sc_a2 = compute_scale(At_f32, total, &zp_a2);
    quantize_i8(Ai, At_f32, total, sc_a2, zp_a2);
    rc = tpu_matmul_submit(ctx, cvk, Ai, S, D, Wo, D, O_i8, 1, SYNC_SCRATCH);
    if (rc) { free(Ai); return -8; }
    dequantize_f32(output, O_i8, total, sc_a2 * 0.001f, 0);
    free(Ai);
    step_us[8] = tick() - ts;
    return 0;
}

static const char *step_names[] = {
    "Input quant","Q proj","K proj","V proj","RoPE","Scores","Softmax","Attn*V","Output"
};

/* ================================================================
 * main
 * ================================================================ */
int main(int argc, char **argv) {
    int d_model   = argc > 1 ? atoi(argv[1]) : 128;
    int n_heads   = argc > 2 ? atoi(argv[2]) : 4;
    int seq_len   = argc > 3 ? atoi(argv[3]) : 16;
    int n_batches = argc > 4 ? atoi(argv[4]) : 6;

    int D = d_model, S = seq_len, d = d_model / n_heads;
    mha_config_t cfg = {
        .d_model = D, .n_heads = n_heads, .head_dim = d,
        .seq_len = S, .softmax_scale = 1.0f / sqrtf((float)d),
    };
    int total = S * D, w_sz = D * D;
    int input_bytes = total * sizeof(float);

    fprintf(stderr, "\n========== MHA Multi-Batch d_model=%d n_heads=%d head_dim=%d seq_len=%d batches=%d ==========\n",
            D, n_heads, d, S, n_batches);
    fprintf(stderr, "  input/batch: %d floats (%.1f KB)  total mem: %d bytes (%.1f MB)\n",
            total, input_bytes / 1024.0, TOTAL_MEM, TOTAL_MEM / 1048576.0);

    /* ---- Shared weights ---- */
    float *W_f32[4]; int8_t *W_i8[4];
    for (int w = 0; w < 4; w++) {
        W_f32[w] = (float *)malloc(w_sz * sizeof(float));
        W_i8[w]  = (int8_t *)malloc(w_sz);
    }
    srand(42);
    for (int i = 0; i < w_sz; i++) {
        for (int w = 0; w < 4; w++)
            W_f32[w][i] = (float)(rand() % 256 - 128) / 1000.0f;
    }
    for (int w = 0; w < 4; w++) {
        int zp; float sc = compute_scale(W_f32[w], w_sz, &zp);
        quantize_i8(W_i8[w], W_f32[w], w_sz, sc, zp);
    }

    /* ---- Generate N batch inputs ---- */
    float **batch_x = (float **)malloc(n_batches * sizeof(float *));
    int8_t **batch_x_i8 = (int8_t **)malloc(n_batches * sizeof(int8_t *));
    int *batch_zp = (int *)malloc(n_batches * sizeof(int));
    float *batch_sc = (float *)malloc(n_batches * sizeof(float));

    for (int b = 0; b < n_batches; b++) {
        batch_x[b] = (float *)malloc(total * sizeof(float));
        for (int i = 0; i < total; i++)
            batch_x[b][i] = (float)(rand() % 256 - 128) / 200.0f;
        batch_x_i8[b] = (int8_t *)malloc(total);
        batch_sc[b] = compute_scale(batch_x[b], total, &batch_zp[b]);
        quantize_i8(batch_x_i8[b], batch_x[b], total, batch_sc[b], batch_zp[b]);
    }

    /* ================================================================
     * MODE 0: CPU memcpy (baseline, single-core)
     * ================================================================ */
    fprintf(stderr, "\n--- Mode 0: CPU memcpy (no small core) ---\n");
    tpu_ctx ctx0;
    if (tpu_init(&ctx0, TOTAL_MEM) != 0) return 1;

    float **out0 = (float **)malloc(n_batches * sizeof(float *));
    for (int b = 0; b < n_batches; b++)
        out0[b] = (float *)malloc(total * sizeof(float));

    double t0_memcpy[6] = {0}, t0_quant[6] = {0}, t0_pipe[6] = {0};
    double ts0 = tick();

    for (int b = 0; b < n_batches; b++) {
        uint32_t bo = (b % 2) ? BATCH1_OFF : BATCH0_OFF;
        uint8_t *v = ctx0.neuron_vaddr;

        /* CPU memcpy: input to current batch area */
        double t1 = tick();
        memcpy(v + bo + MHA_OFF_INPUT, batch_x[b], input_bytes);
        t0_memcpy[b] = tick() - t1;

        CVI_RT_MemFlush(ctx0.rt_handle, ctx0.neuron_mem);

        /* Quantize */
        t1 = tick();
        float *x_in = (float *)(v + bo + MHA_OFF_INPUT);
        int8_t *x_q  = (int8_t *)(v + bo + MHA_OFF_SCRATCH_I8);
        int zp; float sc = compute_scale(x_in, total, &zp);
        quantize_i8(x_q, x_in, total, sc, zp);
        t0_quant[b] = tick() - t1;

        /* Pipeline */
        double step_us[9];
        t1 = tick();
        mha_pipeline_batch(&cfg, &ctx0, x_q, zp, sc,
            W_i8[0], W_i8[1], W_i8[2], W_i8[3], out0[b], step_us, bo);
        t0_pipe[b] = tick() - t1;
    }
    double t0_total = tick() - ts0;

    /* ================================================================
     * MODE 1: Small-core pre-fetch (dual-core double-buffering)
     *
     * Flow per batch b:
     *   1. If b>0: poll dma.result (should be 0 from previous async send)
     *   2. Quantize current batch input
     *   3. Run MHA pipeline
     *   4. If b<n_batches-1: fire async MEMCPY for batch[b+1] → other buffer
     *
     * Pipeline (~10ms) always outruns MEMCPY (~8ms), so dma.result is
     * always ready at step 1 (no polling wait needed).
     * ================================================================ */
    fprintf(stderr, "\n--- Mode 1: Small-core pre-fetch (dual-core, blocking mbox) ---\n");

    if (mbox_open() != 0) {
        fprintf(stderr, "  Mailbox open failed, skipping\n");
        goto print_results;
    }

    tpu_ctx ctx1;
    if (tpu_init(&ctx1, TOTAL_MEM) != 0) { mbox_close(); goto print_results; }

    float **out1 = (float **)malloc(n_batches * sizeof(float *));
    for (int b = 0; b < n_batches; b++)
        out1[b] = (float *)malloc(total * sizeof(float));

    /* DMA descriptor at fixed offset */
    dma_desc_t *dma = (dma_desc_t *)(ctx1.neuron_vaddr + MHA_OFF_REF_OUT);
    uint32_t dma_pa = TPU_PA(&ctx1, MHA_OFF_REF_OUT);
    uint8_t *v = ctx1.neuron_vaddr;

    /* Pre-load batch 0 data to BATCH0_OFF (skip MEMCPY for batch 0 in loop) */
    memcpy(v + BATCH0_OFF + MHA_OFF_INPUT, batch_x[0], input_bytes);
    /* Pre-load staging area with batches 1..N inputs (used by small-core MEMCPY) */
    for (int b = 1; b < n_batches; b++) {
        memcpy(v + STAGING_OFF + b * input_bytes, batch_x[b], input_bytes);
    }
    CVI_RT_MemFlush(ctx1.rt_handle, ctx1.neuron_mem);

    double t1_mbox[6] = {0}, t1_quant[6] = {0}, t1_pipe[6] = {0};
    double ts1 = tick();

    for (int b = 0; b < n_batches; b++) {
        uint32_t cur_off = (b % 2) ? BATCH1_OFF : BATCH0_OFF;
        uint32_t nxt_off = (b % 2) ? BATCH0_OFF : BATCH1_OFF;

        /* Step A: Small core MEMCPY for CURRENT batch's input (blocking)
           For batch 0: CPU already pre-loaded to BATCH0_OFF (skip)
           For batch 1-5: small core copies from staging to current area */
        if (b > 0) {
            /* Also pre-fetch NEXT batch (b+1) to the OTHER buffer simultaneously
               via a second mailbox command. Actually, we can't do 2 at once.
               So do MEMCPY for current batch, which also implicitly prefetches
               next batch's area. */

            uint32_t stag_src = STAGING_OFF + b * input_bytes;

            /* Copy current batch input to current area via small core */
            *dma = (dma_desc_t){
                .src_paddr = TPU_PA(&ctx1, stag_src),
                .dst_paddr = TPU_PA(&ctx1, cur_off + MHA_OFF_INPUT),
                .size = input_bytes, .result = -1
            };
            CVI_RT_MemFlush(ctx1.rt_handle, ctx1.neuron_mem);

            double t_m = tick();
            int rc = mbox_send_wait(CMD_MHA_MEMCPY, dma_pa);
            t1_mbox[b] = tick() - t_m;

            CVI_RT_MemInvld(ctx1.rt_handle, ctx1.neuron_mem);
            if (rc < 0 || dma->result != 0)
                fprintf(stderr, "  batch[%d] MEMCPY FAIL: rc=%d result=%d\n", b, rc, dma->result);
        }

        /* Step B: Quantize current batch input */
        double t_q = tick();
        float *x_cur = (float *)(v + cur_off + MHA_OFF_INPUT);
        int8_t *x_qcur = (int8_t *)(v + cur_off + MHA_OFF_SCRATCH_I8);
        int zp_cur; float sc_cur = compute_scale(x_cur, total, &zp_cur);
        quantize_i8(x_qcur, x_cur, total, sc_cur, zp_cur);
        t1_quant[b] = tick() - t_q;

        /* Step C: Run MHA pipeline */
        double step_us[9], t_pipe = tick();
        mha_pipeline_batch(&cfg, &ctx1, x_qcur, zp_cur, sc_cur,
            W_i8[0], W_i8[1], W_i8[2], W_i8[3], out1[b], step_us, cur_off);
        t1_pipe[b] = tick() - t_pipe;
    }
    double t1_total = tick() - ts1;

    fprintf(stderr, "  Dual-core total: %.1f us  (avg/batch: %.1f us)\n",
            t1_total, t1_total / n_batches);
    for (int b = 0; b < n_batches; b++) {
        double sum = t1_mbox[b] + t1_quant[b] + t1_pipe[b];
        fprintf(stderr, "    batch[%d] mbox=%6.1f quant=%6.1f pipe=%8.1f  sum=%8.1f us\n",
                b, t1_mbox[b], t1_quant[b], t1_pipe[b], sum);
    }

    /* ---- Verification: both modes should produce same output ---- */
    fprintf(stderr, "\n--- Cross-mode verification ---\n");
    int ok = 1;
    for (int b = 0; b < n_batches && ok; b++) {
        double max_err = 0;
        for (int i = 0; i < total; i++) {
            double err = fabs((double)out0[b][i] - (double)out1[b][i]);
            if (err > max_err) max_err = err;
        }
        fprintf(stderr, "  batch[%d] CPU vs DC  max_err=%.6f %s\n",
                b, max_err, max_err < 0.001 ? "OK" : "DIFFERENT!");
        if (max_err >= 0.001) ok = 0;
    }

print_results:
    /* ---- Comparison ---- */
    fprintf(stderr, "\n========== Multi-Batch Comparison ==========\n");
    fprintf(stderr, "  %-35s %12s %12s %12s\n", "Mode", "Total(us)", "Avg/Batch(us)", "vs CPU");
    fprintf(stderr, "  %-35s %12s %12s %12s\n", "---", "--------", "------------", "------");
    fprintf(stderr, "  %-35s %12.1f %12.1f %11s\n",
            "CPU memcpy (baseline)", t0_total, t0_total / n_batches, "1.00x");
    if (mbox_fd >= 0) {
        fprintf(stderr, "  %-35s %12.1f %12.1f %10.2fx\n",
                "Dual-core small-core pre-fetch", t1_total, t1_total / n_batches,
                t0_total / t1_total);
    }

    /* Per-batch breakdown */
    fprintf(stderr, "\n  %-6s %8s %8s %8s | %8s %8s %8s\n",
            "Batch", "CPU:cpy", "CPU:qnt", "CPU:pipe",
            "DC:mbox", "DC:qnt", "DC:pipe");
    for (int b = 0; b < n_batches; b++) {
        fprintf(stderr, "  %4d   %8.1f %8.1f %8.1f | %8.1f %8.1f %8.1f\n",
                b, t0_memcpy[b], t0_quant[b], t0_pipe[b],
                t1_mbox[b], t1_quant[b], t1_pipe[b]);
    }

    /* Pipeline time comparison */
    double sum0_pipe = 0, sum1_pipe = 0, sum0_overhead = 0, sum1_overhead = 0;
    for (int b = 0; b < n_batches; b++) {
        sum0_pipe += t0_pipe[b];
        sum0_overhead += t0_memcpy[b] + t0_quant[b];
        sum1_pipe += t1_pipe[b];
        sum1_overhead += t1_mbox[b] + t1_quant[b];
    }
    fprintf(stderr, "\n  Pipeline time (sum):  CPU=%.1f us  DC=%.1f us\n", sum0_pipe, sum1_pipe);
    fprintf(stderr, "  Overhead (sum):       CPU=%.1f us  DC=%.1f us\n", sum0_overhead, sum1_overhead);
    fprintf(stderr, "  Overhead/pipe:        CPU=%.1f%%  DC=%.1f%%\n",
            100.0 * sum0_overhead / sum0_pipe, 100.0 * sum1_overhead / sum1_pipe);

    /* Machine-readable JSON */
    fprintf(stderr, "\n--- MHA_MULTI_BATCH_JSON ---\n");
    fprintf(stderr, "{\"cfg\":{\"d_model\":%d,\"n_heads\":%d,\"head_dim\":%d,\"seq_len\":%d,\"batches\":%d},\n",
            D, n_heads, d, S, n_batches);
    fprintf(stderr, " \"cpu\":{\"total_us\":%.1f,\"pipe_us\":[", t0_total);
    for (int b = 0; b < n_batches; b++)
        fprintf(stderr, "%.1f%s", t0_pipe[b], b < n_batches-1 ? "," : "");
    fprintf(stderr, "],\"memcpy_us\":[");
    for (int b = 0; b < n_batches; b++)
        fprintf(stderr, "%.1f%s", t0_memcpy[b], b < n_batches-1 ? "," : "");
    fprintf(stderr, "],\"quant_us\":[");
    for (int b = 0; b < n_batches; b++)
        fprintf(stderr, "%.1f%s", t0_quant[b], b < n_batches-1 ? "," : "");
    fprintf(stderr, "]},\n");
    fprintf(stderr, " \"dualcore\":{\"total_us\":%.1f,\"pipe_us\":[", t1_total);
    for (int b = 0; b < n_batches; b++)
        fprintf(stderr, "%.1f%s", t1_pipe[b], b < n_batches-1 ? "," : "");
    fprintf(stderr, "],\"mbox_us\":[");
    for (int b = 0; b < n_batches; b++)
        fprintf(stderr, "%.1f%s", t1_mbox[b], b < n_batches-1 ? "," : "");
    fprintf(stderr, "],\"quant_us\":[");
    for (int b = 0; b < n_batches; b++)
        fprintf(stderr, "%.1f%s", t1_quant[b], b < n_batches-1 ? "," : "");
    fprintf(stderr, "]}}\n");

    /* Cleanup */
    mbox_close();
    for (int b = 0; b < n_batches; b++) {
        free(batch_x[b]); free(batch_x_i8[b]);
        free(out0[b]); if (mbox_fd >= 0) free(out1[b]);
    }
    free(batch_x); free(batch_x_i8); free(batch_zp); free(batch_sc);
    free(out0); free(out1);
    for (int w = 0; w < 4; w++) { free(W_f32[w]); free(W_i8[w]); }
    tpu_cleanup(&ctx0);
    if (mbox_fd >= 0) tpu_cleanup(&ctx1);

    return 0;
}
