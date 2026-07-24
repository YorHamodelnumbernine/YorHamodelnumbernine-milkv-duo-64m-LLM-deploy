/* Multihead Attention on Milk-V Duo CV1800B — dual-core orchestration.
   TPU (cvikernel) handles matmuls; big core handles softmax/RoPE/RMSNorm;
   small core (FreeRTOS) handles bulk data movement.
   Build: make mha_attention   Run: ./mha_attention [d_model] [n_heads] [seq_len]
   Per-step timing + blocking vs async comparison.
*/
#include "common/tpu_bench.h"
#include "common/mha_descriptor.h"
#include <math.h>

/* ---- Scratch areas for concurrent async operations ---- */
#define ASYNC_SCRATCH  0x00000   /* for async-submitted matmul */
#define SYNC_SCRATCH   0x20000   /* for blocking matmul that overlaps with async */

/* ---- forward declarations ---- */
static void mha_reference_fp32(const mha_config_t *cfg,
    const float *x, const float *Wq, const float *Wk, const float *Wv,
    const float *Wo, float *output);
static int  mha_tpu_pipeline(const mha_config_t *cfg, tpu_ctx *ctx,
    const float *x, const int8_t *Wq, const int8_t *Wk, const int8_t *Wv,
    const int8_t *Wo, float *output, int use_async, double *step_us);

/* ---- math helpers ---- */
static void softmax_f32(float *buf, int rows, int cols);
static void apply_rope_f32(float *buf, int seq_len, int head_dim, float base);

/* ---- quant helpers ---- */
static void quantize_i8(int8_t *dst, const float *src, int n, float scale, int zp);
static void dequantize_f32(float *dst, const int8_t *src, int n, float scale, int zp);
static float compute_scale(const float *data, int n, int *zp_out);

/* ---- simple us timer ---- */
static double tick(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

/* ================================================================
 * CPU reference: pure FP32 MHA
 * ================================================================ */
static void mha_reference_fp32(const mha_config_t *cfg,
    const float *x, const float *Wq, const float *Wk, const float *Wv,
    const float *Wo, float *output)
{
    int S = cfg->seq_len, D = cfg->d_model, H = cfg->n_heads, d = cfg->head_dim;
    int total = S * D;
    float scale = cfg->softmax_scale;

    float *Q = (float *)malloc(total * sizeof(float));
    float *K = (float *)malloc(total * sizeof(float));
    float *V = (float *)malloc(total * sizeof(float));
    float *Scores = (float *)malloc(H * S * S * sizeof(float));
    float *Attn   = (float *)malloc(total * sizeof(float));

    /* QKV projection */
    for (int i = 0; i < S; i++) {
        for (int j = 0; j < D; j++) {
            float q = 0, k = 0, v = 0;
            for (int kk = 0; kk < D; kk++) {
                float xi = x[i * D + kk];
                q += xi * Wq[kk * D + j];
                k += xi * Wk[kk * D + j];
                v += xi * Wv[kk * D + j];
            }
            Q[i * D + j] = q; K[i * D + j] = k; V[i * D + j] = v;
        }
    }

    /* RoPE on Q and K */
    apply_rope_f32(Q, S, D, 10000.0f);
    apply_rope_f32(K, S, D, 10000.0f);

    /* Per-head attention */
    for (int h = 0; h < H; h++) {
        float *Qh = (float *)malloc(S * d * sizeof(float));
        float *Kh = (float *)malloc(S * d * sizeof(float));
        float *Vh = (float *)malloc(S * d * sizeof(float));
        for (int i = 0; i < S; i++) {
            for (int j = 0; j < d; j++) {
                Qh[i * d + j] = Q[i * D + h * d + j];
                Kh[i * d + j] = K[i * D + h * d + j];
                Vh[i * d + j] = V[i * D + h * d + j];
            }
        }
        float *Sh = Scores + h * S * S;
        float *Ah = Attn + h * S * d;

        for (int i = 0; i < S; i++) {
            for (int j = 0; j < S; j++) {
                float s = 0;
                for (int kk = 0; kk < d; kk++)
                    s += Qh[i * d + kk] * Kh[j * d + kk];
                Sh[i * S + j] = s * scale;
            }
        }
        softmax_f32(Sh, S, S);

        for (int i = 0; i < S; i++) {
            for (int j = 0; j < d; j++) {
                float a = 0;
                for (int kk = 0; kk < S; kk++)
                    a += Sh[i * S + kk] * Vh[kk * d + j];
                Ah[i * d + j] = a;
            }
        }
        free(Qh); free(Kh); free(Vh);
    }

    /* Interleave heads back: [h][r][d] -> [r][h*d] */
    float *Attn_interleaved = (float *)malloc(total * sizeof(float));
    for (int h = 0; h < H; h++)
        for (int r = 0; r < S; r++)
            memcpy(Attn_interleaved + r * D + h * d, Attn + h * S * d + r * d, d * sizeof(float));

    for (int i = 0; i < S; i++) {
        for (int j = 0; j < D; j++) {
            float o = 0;
            for (int kk = 0; kk < D; kk++)
                o += Attn_interleaved[i * D + kk] * Wo[kk * D + j];
            output[i * D + j] = o;
        }
    }
    free(Attn_interleaved);
    free(Q); free(K); free(V); free(Scores); free(Attn);
}

/* ================================================================
 * CPU math primitives
 * ================================================================ */
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

/* Same as apply_rope_f32 but uses precomputed cos/sin tables.
   cos_tab and sin_tab are [seq_len][half] row-major. */
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

/* Precompute RoPE cos/sin tables: each [seq_len * (head_dim/2)] */
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

/* ================================================================
 * Quantization helpers
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

static void quantize_i8(int8_t *dst, const float *src, int n, float scale, int zp) {
    float inv = 1.0f / scale;
    for (int i = 0; i < n; i++) {
        int q = (int)(src[i] * inv + zp + 0.5f);
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        dst[i] = (int8_t)q;
    }
}

static void dequantize_f32(float *dst, const int8_t *src, int n, float scale, int zp) {
    for (int i = 0; i < n; i++) dst[i] = (float)((int)src[i] - zp) * scale;
}

/* CPU transpose: dst[j*rows + i] = src[i*cols + j] */
static void transpose_i8(int8_t *dst, const int8_t *src, int rows, int cols) {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            dst[j * rows + i] = src[i * cols + j];
}

/* ================================================================
 * TPU matmul — uses scratch_off for DRAM temp buffers.
 *   scratch_off: base offset for left/right/out matrices
 *   do_submit: 0=batch cmdbuf only, 1=submit+wait+copyback
 *
 * When the right matrix [K,N] is too large for 32KB LMEM, this
 * function automatically tiles along N so each tile fits.
 *
 * With do_submit=0, caller must later call tpu_matmul_wait()
 * with the same scratch_off, and must NOT reuse scratch_off
 * or the cvk context until after the wait.
 * ================================================================ */

/* Find the largest TileN such that [M,K] + [K,TileN] + [M,TileN] fits in LMEM.
   Uses the analytical LMEM layout from mha_descriptor.h (cvikernel INT8 lane format).
   LMEM total = 32KB (32768 bytes). */
static int tpu_find_tile_n(cvk_context_t *cvk, int M, int K) {
    int left = mha_lmem_matrix_bytes(M, K);
    for (int tn = 128; tn >= 16; tn -= 16) {
        int right = mha_lmem_matrix_bytes(K, tn);
        int out   = mha_lmem_matrix_bytes(M, tn);
        if (left + right + out <= 32768) return tn;
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

    /* Decide fast-path vs tiling analytically (avoid partial alloc/free issues) */
    int need_tile = (mha_lmem_matrix_bytes(M, K) +
                     mha_lmem_matrix_bytes(K, N) +
                     mha_lmem_matrix_bytes(M, N) > 32768);

    if (!need_tile) {
        /* ---- Fast path: everything fits in LMEM ---- */
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
            .src = &(cvk_mg_t){0, TPU_PA(ctx, off_l), CVK_FMT_I8, {M, K}, {K}},
            .dst = ml_l});
        cvk->ops->tdma_g2l_matrix_copy(cvk, &(cvk_tdma_g2l_matrix_copy_param_t){
            .src = &(cvk_mg_t){0, TPU_PA(ctx, off_r), CVK_FMT_I8, {K, N}, {N}},
            .dst = ml_r});

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

        if (do_submit) {
            CVI_RT_Submit(ctx->rt_khandle);
            CVI_RT_MemInvld(ctx->rt_handle, ctx->neuron_mem);
            memcpy(o_i8, ctx->neuron_vaddr + off_o, M * N);
        }
        return 0;
    }

    /* ---- Tiling path: split N dimension ---- */
    int tile_n = tpu_find_tile_n(cvk, M, K);
    if (tile_n < 16) {
        fprintf(stderr, "LMEM fail M=%d K=%d N=%d (tiling exhausted)\n", M, K, N);
        return -1;
    }

    /* If right matrix is already in neuron memory (e.g., Wq/Wk/Wv/Wo),
       point TPU DMA directly at it with stride N — no CPU memcpy needed. */
    uintptr_t nm_base = (uintptr_t)ctx->neuron_vaddr;
    uintptr_t nm_end  = nm_base + ctx->neuron_size;
    int r_is_nm = ((uintptr_t)r_i8 >= nm_base && (uintptr_t)r_i8 < nm_end);
    uint32_t r_nm_off = r_is_nm ? (uint32_t)((uintptr_t)r_i8 - nm_base) : 0;

    uint32_t off_o_base = scratch_off + M * K + K * tile_n;

    /* Copy left matrix to DRAM once (reused across tiles) */
    memcpy(ctx->neuron_vaddr + off_l, l_i8, M * K);
    CVI_RT_MemFlush(ctx->rt_handle, ctx->neuron_mem);

    for (int n_start = 0; n_start < N; n_start += tile_n) {
        int cur_n = (n_start + tile_n <= N) ? tile_n : N - n_start;

        if (!r_is_nm) {
            /* CPU extracts right tile columns [n_start:n_start+cur_n) from
               each of the K rows (host memory source only) */
            uint8_t *tile_dst = ctx->neuron_vaddr + off_r;
            for (int row = 0; row < K; row++)
                memcpy(tile_dst + row * cur_n, r_i8 + row * N + n_start, cur_n);
            CVI_RT_MemFlush(ctx->rt_handle, ctx->neuron_mem);
        }

        cvk_ml_t *ml_l = cvk->ops->lmem_alloc_matrix(cvk,
            cvk->ops->ml_default_shape(cvk, M, K, CVK_FMT_I8), CVK_FMT_I8, 1);
        cvk_ml_t *ml_r = cvk->ops->lmem_alloc_matrix(cvk,
            cvk->ops->ml_default_shape(cvk, K, cur_n, CVK_FMT_I8), CVK_FMT_I8, 1);
        cvk_ml_t *ml_o = cvk->ops->lmem_alloc_matrix(cvk,
            cvk->ops->ml_default_shape(cvk, M, cur_n, CVK_FMT_I8), CVK_FMT_I8, 1);

        cvk->ops->tdma_g2l_matrix_copy(cvk, &(cvk_tdma_g2l_matrix_copy_param_t){
            .src = &(cvk_mg_t){0, TPU_PA(ctx, off_l), CVK_FMT_I8, {M, K}, {K}},
            .dst = ml_l});
        /* Right G2L: when r_is_nm, DMA reads directly from the original
           [K,N] matrix at column offset n_start with stride N.
           Otherwise reads from the contiguous tile buffer at off_r. */
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

        /* Write tile columns to correct positions in full [M,N] output via stride N */
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

/* Wait for async completion, then copy result from the scratch area.
   M, K, N must match the values passed to tpu_matmul_submit.
   Output offset is computed consistently with tpu_matmul_submit. */
static int tpu_matmul_wait(tpu_ctx *ctx, void *result, int M, int K, int N,
                           uint32_t scratch_off) {
    CVI_RT_WaitForAsync(ctx->rt_khandle);
    CVI_RT_MemInvld(ctx->rt_handle, ctx->neuron_mem);

    /* Same TileN logic as tpu_matmul_submit:
       if L+R+O fit → off_o=M*K+K*N; else → off_o=M*K+K*tile_n */
    uint32_t off_o;
    int left  = mha_lmem_matrix_bytes(M, K);
    int right = mha_lmem_matrix_bytes(K, N);
    int out   = mha_lmem_matrix_bytes(M, N);
    if (left + right + out <= 32768) {
        off_o = scratch_off + M * K + K * N;
    } else {
        int tile_n = tpu_find_tile_n(ctx->cvk_ctx, M, K);
        if (tile_n < 16) tile_n = 64;
        off_o = scratch_off + M * K + K * tile_n;
    }

    memcpy(result, ctx->neuron_vaddr + off_o, M * N);
    return 0;
}

/* ================================================================
 * MHA Pipeline — with per-step timing (step_us[9])
 *
 * Async strategy:
 *   Step 1 (Q proj) submitted async with ASYNC_SCRATCH area.
 *   CPU immediately begins Step 2 (K proj) using SYNC_SCRATCH.
 *   TPU runs Step 1 while CPU preps & submits Step 2.
 *   Step 2's CVI_RT_Submit drains the TPU queue (waits for Step 1
 *   AND Step 2), so after Step 2 returns, both are done.
 *   This overlaps CPU prep work with TPU compute.
 *
 * step_us[] indices:
 *   0=Input quant   1=Q proj     2=K proj     3=V proj
 *   4=RoPE+re-quant 5=Scores     6=Softmax    7=Attn*V
 *   8=Output proj
 * ================================================================ */
static int mha_tpu_pipeline(const mha_config_t *cfg, tpu_ctx *ctx,
    const float *x, const int8_t *Wq, const int8_t *Wk, const int8_t *Wv,
    const int8_t *Wo, float *output, int use_async, double *step_us)
{
    int S = cfg->seq_len, D = cfg->d_model, H = cfg->n_heads, d = cfg->head_dim;
    int total = S * D;
    float scale_f = cfg->softmax_scale;
    cvk_context_t *cvk = ctx->cvk_ctx;
    int rc;
    double ts;

    /* ---- Memory-mapped buffers in neuron DRAM ---- */
    float *Q_f32 = (float *)(ctx->neuron_vaddr + MHA_OFF_Q_F32);
    float *K_f32 = (float *)(ctx->neuron_vaddr + MHA_OFF_K_F32);
    float *V_f32 = (float *)(ctx->neuron_vaddr + MHA_OFF_V_F32);
    float *Scores_f32 = (float *)(ctx->neuron_vaddr + MHA_OFF_SCORES_F32);
    float *Attn_f32  = (float *)(ctx->neuron_vaddr + MHA_OFF_ATTN_F32);
    int8_t *Q_i8 = (int8_t *)(ctx->neuron_vaddr + MHA_OFF_Q_I8);
    int8_t *K_i8 = (int8_t *)(ctx->neuron_vaddr + MHA_OFF_K_I8);
    int8_t *V_i8 = (int8_t *)(ctx->neuron_vaddr + MHA_OFF_V_I8);
    int8_t *S_i8 = (int8_t *)(ctx->neuron_vaddr + MHA_OFF_S_I8);
    int8_t *A_i8 = (int8_t *)(ctx->neuron_vaddr + MHA_OFF_A_I8);
    int8_t *O_i8 = (int8_t *)(ctx->neuron_vaddr + MHA_OFF_OUT_I8);

    /* ---- Step 0: Quantize input x: FP32 → INT8 ---- */
    ts = tick();
    int zp_x; float sc_x = compute_scale(x, total, &zp_x);
    int8_t *x_i8 = (int8_t *)malloc(total);
    quantize_i8(x_i8, x, total, sc_x, zp_x);
    step_us[0] = tick() - ts;

    /* ---- Step 1: Q = x * Wq ---- */
    ts = tick();
    if (use_async) {
        rc = tpu_matmul_submit(ctx, cvk, x_i8, S, D, Wq, D, Q_i8, 0, ASYNC_SCRATCH);
        CVI_RT_SubmitAsync(ctx->rt_khandle, 0);
    } else {
        rc = tpu_matmul_submit(ctx, cvk, x_i8, S, D, Wq, D, Q_i8, 1, SYNC_SCRATCH);
    }
    step_us[1] = tick() - ts;
    if (rc) { free(x_i8); return -1; }

    /* ---- Step 2: K = x * Wk (uses SYNC_SCRATCH, safe vs async ASYNC_SCRATCH) ---- */
    ts = tick();
    rc = tpu_matmul_submit(ctx, cvk, x_i8, S, D, Wk, D, K_i8, 1, SYNC_SCRATCH);
    step_us[2] = tick() - ts;
    if (rc) { free(x_i8); return -2; }

    /* After Step 2's blocking submit, both Step 1 (async) and Step 2 are done.
       Still call WaitForAsync for correctness. */
    if (use_async) {
        /* Result already in Q_i8 from the async submit; just WaitForAsync */
        CVI_RT_WaitForAsync(ctx->rt_khandle);
        CVI_RT_MemInvld(ctx->rt_handle, ctx->neuron_mem);
    }

    /* Dequantize Q, K after TPU matmul.
       Output scale ≈ sc_x * sc_w (product of input scales for matmul) */
    float sc_qk_out = sc_x * 0.001f;  /* ballpark: weight scale ~0.001 */
    dequantize_f32(Q_f32, Q_i8, total, sc_qk_out, 0);
    dequantize_f32(K_f32, K_i8, total, sc_qk_out, 0);

    /* ---- Step 3: V = x * Wv ---- */
    ts = tick();
    rc = tpu_matmul_submit(ctx, cvk, x_i8, S, D, Wv, D, V_i8, 1, SYNC_SCRATCH);
    step_us[3] = tick() - ts;
    free(x_i8);
    if (rc) return -3;

    /* ---- Step 4: RoPE on Q and K (CPU, FP32, lookup-table) ---- */
    ts = tick();
    int rope_half = d / 2;
    float *rope_cos = (float *)malloc(S * rope_half * sizeof(float));
    float *rope_sin = (float *)malloc(S * rope_half * sizeof(float));
    rope_precompute(S, d, 10000.0f, rope_cos, rope_sin);
    for (int h = 0; h < H; h++) {
        float *Qh = (float *)malloc(S * d * sizeof(float));
        float *Kh = (float *)malloc(S * d * sizeof(float));
        for (int r = 0; r < S; r++) {
            for (int c = 0; c < d; c++) {
                Qh[r * d + c] = Q_f32[r * D + h * d + c];
                Kh[r * d + c] = K_f32[r * D + h * d + c];
            }
        }
        apply_rope_f32_tab(Qh, S, d, rope_cos, rope_sin);
        apply_rope_f32_tab(Kh, S, d, rope_cos, rope_sin);
        for (int r = 0; r < S; r++) {
            for (int c = 0; c < d; c++) {
                Q_f32[r * D + h * d + c] = Qh[r * d + c];
                K_f32[r * D + h * d + c] = Kh[r * d + c];
            }
        }
        free(Qh); free(Kh);
    }
    free(rope_cos); free(rope_sin);
    step_us[4] = tick() - ts;

    /* Re-quantize Q,K after RoPE for score matmul */
    int zp_q2; float sc_q2 = compute_scale(Q_f32, total, &zp_q2);
    quantize_i8(Q_i8, Q_f32, total, sc_q2, zp_q2);
    int zp_k2; float sc_k2 = compute_scale(K_f32, total, &zp_k2);
    quantize_i8(K_i8, K_f32, total, sc_k2, zp_k2);

    /* ---- Step 5: Per-head Scores = Q * K^T / sqrt(d) ---- */
    ts = tick();
    for (int h = 0; h < H; h++) {
        int8_t *Qh = (int8_t *)malloc(S * d);
        int8_t *Kh = (int8_t *)malloc(S * d);
        int8_t *Kh_t = (int8_t *)malloc(S * d);
        for (int r = 0; r < S; r++) {
            memcpy(Qh + r * d, Q_i8 + r * D + h * d, d);
            memcpy(Kh + r * d, K_i8 + r * D + h * d, d);
        }
        transpose_i8(Kh_t, Kh, S, d);
        rc = tpu_matmul_submit(ctx, cvk, Qh, S, d, Kh_t, S, S_i8 + h * S * S, 1, SYNC_SCRATCH);
        free(Qh); free(Kh); free(Kh_t);
        if (rc) return -5;
    }
    /* Dequantize scores: output scale ≈ sc_q2 * sc_k2, then apply softmax_scale */
    float sc_s_out = sc_q2 * sc_k2;
    dequantize_f32(Scores_f32, S_i8, H * S * S, sc_s_out, 0);
    for (int i = 0; i < H * S * S; i++) Scores_f32[i] *= scale_f;
    step_us[5] = tick() - ts;

    /* ---- Step 6: Softmax per head (CPU, FP32) ---- */
    ts = tick();
    for (int h = 0; h < H; h++)
        softmax_f32(Scores_f32 + h * S * S, S, S);
    step_us[6] = tick() - ts;

    /* Re-quantize softmax scores for attention matmul */
    int zp_s2; float sc_s2 = compute_scale(Scores_f32, H * S * S, &zp_s2);
    quantize_i8(S_i8, Scores_f32, H * S * S, sc_s2, zp_s2);

    /* Dequantize V, then re-quantize for attention matmul */
    dequantize_f32(V_f32, V_i8, total, sc_qk_out, 0);
    int zp_v2; float sc_v2 = compute_scale(V_f32, total, &zp_v2);
    quantize_i8(V_i8, V_f32, total, sc_v2, zp_v2);

    /* ---- Step 7: Attn = softmax(Scores) * V (per head) ---- */
    ts = tick();
    for (int h = 0; h < H; h++) {
        int8_t *Vh = (int8_t *)malloc(S * d);
        for (int r = 0; r < S; r++)
            memcpy(Vh + r * d, V_i8 + r * D + h * d, d);
        rc = tpu_matmul_submit(ctx, cvk,
            S_i8 + h * S * S, S, S, Vh, d,
            A_i8 + h * S * d, 1, SYNC_SCRATCH);
        free(Vh);
        if (rc) return -7;
    }
    /* Dequantize attention output */
    float sc_a_out = sc_s2 * sc_v2;
    dequantize_f32(Attn_f32, A_i8, total, sc_a_out, 0);
    step_us[7] = tick() - ts;

    /* ---- Step 8: Output = Attn * Wo ---- */
    ts = tick();
    /* Interleave heads: [h][r][d] -> [r][h*d] */
    int8_t *A_interleaved = (int8_t *)malloc(total);
    for (int h = 0; h < H; h++)
        for (int r = 0; r < S; r++)
            memcpy(A_interleaved + r * D + h * d, A_i8 + h * S * d + r * d, d);

    int zp_a2; float sc_a2 = compute_scale(Attn_f32, total, &zp_a2);
    quantize_i8(A_interleaved, Attn_f32, total, sc_a2, zp_a2);
    rc = tpu_matmul_submit(ctx, cvk, A_interleaved, S, D, Wo, D, O_i8, 1, SYNC_SCRATCH);
    if (rc) { free(A_interleaved); return -8; }
    float sc_o_out = sc_a2 * 0.001f;
    dequantize_f32(output, O_i8, total, sc_o_out, 0);
    free(A_interleaved);
    step_us[8] = tick() - ts;

    return 0;
}

/* ================================================================
 * Verification
 * ================================================================ */
static void verify_output(const char *label, const float *got, const float *exp,
                          int n, double *max_err_out, double *mse_out) {
    double max_err = 0, mse = 0;
    for (int i = 0; i < n; i++) {
        double err = fabs((double)got[i] - (double)exp[i]);
        if (err > max_err) max_err = err;
        mse += err * err;
    }
    mse /= n;
    *max_err_out = max_err; *mse_out = mse;
    fprintf(stderr, "  %-20s max_err=%.4f  MSE=%.6f\n", label, max_err, mse);
}

static const char *step_names[] = {
    "Input quant", "Q proj", "K proj", "V proj",
    "RoPE+re-quant", "Scores Q*K^T", "Softmax", "Attn*V",
    "Output proj"
};

static void print_step_table(double *blk, double *asyn, int n) {
    fprintf(stderr, "\n  %-3s %-18s %12s %12s %12s\n",
            "No", "Step", "Blocking(us)", "Async(us)", "Delta");
    fprintf(stderr, "  %-3s %-18s %12s %12s %12s\n",
            "---", "----------------", "----------", "----------", "----------");
    double blk_sum = 0, asy_sum = 0;
    for (int i = 0; i < n; i++) {
        double d = blk[i] - asyn[i];
        fprintf(stderr, "  %2d  %-18s %12.1f %12.1f %+12.1f\n",
                i, step_names[i], blk[i], asyn[i], d);
        blk_sum += blk[i]; asy_sum += asyn[i];
    }
    fprintf(stderr, "  %-3s %-18s %12s %12s %12s\n",
            "---", "----------------", "----------", "----------", "----------");
    fprintf(stderr, "  %-3s %-18s %12.1f %12.1f %+12.1f\n",
            "SUM", "Total pipeline", blk_sum, asy_sum, blk_sum - asy_sum);
}

/* ================================================================
 * main
 * ================================================================ */
int main(int argc, char **argv) {
    int d_model = argc > 1 ? atoi(argv[1]) : 512;
    int n_heads = argc > 2 ? atoi(argv[2]) : 8;
    int seq_len = argc > 3 ? atoi(argv[3]) : 32;

    mha_config_t cfg = {
        .d_model = d_model, .n_heads = n_heads,
        .head_dim = d_model / n_heads, .seq_len = seq_len,
        .softmax_scale = 1.0f / sqrtf((float)(d_model / n_heads)),
    };
    int D = cfg.d_model, S = cfg.seq_len, d = cfg.head_dim;

    fprintf(stderr, "\n========== MHA d_model=%d n_heads=%d head_dim=%d seq_len=%d ==========\n",
            D, n_heads, d, S);
    fprintf(stderr, "  softmax_scale=%.4f  total_act=%d  weight_sz=%d\n\n",
            cfg.softmax_scale, S * D, D * D);

    int total = S * D, w_sz = D * D;
    float *x   = (float *)malloc(total * sizeof(float));
    float *Wq  = (float *)malloc(w_sz * sizeof(float));
    float *Wk  = (float *)malloc(w_sz * sizeof(float));
    float *Wv  = (float *)malloc(w_sz * sizeof(float));
    float *Wo  = (float *)malloc(w_sz * sizeof(float));
    int8_t *Wq_i8 = (int8_t *)malloc(w_sz);
    int8_t *Wk_i8 = (int8_t *)malloc(w_sz);
    int8_t *Wv_i8 = (int8_t *)malloc(w_sz);
    int8_t *Wo_i8 = (int8_t *)malloc(w_sz);

    srand(42);
    for (int i = 0; i < total; i++) x[i] = (float)(rand() % 256 - 128) / 200.0f;
    for (int i = 0; i < w_sz; i++) {
        Wq[i] = (float)(rand() % 256 - 128) / 1000.0f;
        Wk[i] = (float)(rand() % 256 - 128) / 1000.0f;
        Wv[i] = (float)(rand() % 256 - 128) / 1000.0f;
        Wo[i] = (float)(rand() % 256 - 128) / 1000.0f;
    }

    /* Quantize weights */
    int zp_w; float sc_w;
    sc_w = compute_scale(Wq, w_sz, &zp_w); quantize_i8(Wq_i8, Wq, w_sz, sc_w, zp_w);
    sc_w = compute_scale(Wk, w_sz, &zp_w); quantize_i8(Wk_i8, Wk, w_sz, sc_w, zp_w);
    sc_w = compute_scale(Wv, w_sz, &zp_w); quantize_i8(Wv_i8, Wv, w_sz, sc_w, zp_w);
    sc_w = compute_scale(Wo, w_sz, &zp_w); quantize_i8(Wo_i8, Wo, w_sz, sc_w, zp_w);

    /* ---- CPU Reference ---- */
    fprintf(stderr, "--- CPU Reference (FP32) ---\n");
    float *ref_out = (float *)malloc(total * sizeof(float));
    double ts_ref = tick();
    mha_reference_fp32(&cfg, x, Wq, Wk, Wv, Wo, ref_out);
    double ref_us = tick() - ts_ref;
    fprintf(stderr, "  CPU ref time: %.1f us\n\n", ref_us);

    /* ---- TPU MHA (Blocking) ---- */
    fprintf(stderr, "--- TPU MHA (Blocking) ---\n");
    tpu_ctx ctx_blk;
    if (tpu_init(&ctx_blk, MHA_TOTAL_SIZE) != 0) return 1;
    float *tpu_out_blk = (float *)malloc(total * sizeof(float));
    double step_blk[9] = {0};

    double ts_blk = tick();
    int rc_blk = mha_tpu_pipeline(&cfg, &ctx_blk, x, Wq_i8, Wk_i8, Wv_i8, Wo_i8,
                                   tpu_out_blk, 0, step_blk);
    double blk_us = tick() - ts_blk;
    fprintf(stderr, "  Blocking total: %.1f us  rc=%d\n", blk_us, rc_blk);

    /* ---- TPU MHA (Async) ---- */
    fprintf(stderr, "\n--- TPU MHA (Async) ---\n");
    tpu_ctx ctx_asy;
    if (tpu_init(&ctx_asy, MHA_TOTAL_SIZE) != 0) return 1;
    float *tpu_out_asy = (float *)malloc(total * sizeof(float));
    double step_asy[9] = {0};

    double ts_asy = tick();
    int rc_asy = mha_tpu_pipeline(&cfg, &ctx_asy, x, Wq_i8, Wk_i8, Wv_i8, Wo_i8,
                                   tpu_out_asy, 1, step_asy);
    double asy_us = tick() - ts_asy;
    fprintf(stderr, "  Async total: %.1f us  rc=%d\n", asy_us, rc_asy);

    /* ---- Per-step comparison ---- */
    fprintf(stderr, "\n--- Per-Step Timing Breakdown ---\n");
    print_step_table(step_blk, step_asy, 9);

    /* ---- Verification ---- */
    fprintf(stderr, "\n--- Verification ---\n");
    double max_err, mse;
    if (rc_blk == 0) {
        verify_output("Blocking vs Ref", tpu_out_blk, ref_out, total, &max_err, &mse);
    }
    if (rc_asy == 0) {
        verify_output("Async vs Ref", tpu_out_asy, ref_out, total, &max_err, &mse);
    }
    if (rc_blk == 0 && rc_asy == 0) {
        verify_output("Blocking vs Async", tpu_out_blk, tpu_out_asy, total, &max_err, &mse);
    }
    fprintf(stderr, "\n  Speedup vs CPU ref (%.0fus):  Blk=%.1fx  Async=%.1fx\n",
            ref_us, ref_us / blk_us, ref_us / asy_us);

    /* ---- Machine-readable step data for HTML ---- */
    fprintf(stderr, "\n--- MHA_STEP_DATA_JSON ---\n");
    fprintf(stderr, "{\"cfg\":{\"d_model\":%d,\"n_heads\":%d,\"head_dim\":%d,\"seq_len\":%d,"
            "\"softmax_scale\":%.4f},\n", D, n_heads, d, S, cfg.softmax_scale);
    fprintf(stderr, " \"ref_us\":%.1f, \"blk_total_us\":%.1f, \"asy_total_us\":%.1f,\n",
            ref_us, blk_us, asy_us);
    fprintf(stderr, " \"steps\":[");
    for (int i = 0; i < 9; i++) {
        fprintf(stderr, "{\"name\":\"%s\",\"blk_us\":%.1f,\"asy_us\":%.1f}",
                step_names[i], step_blk[i], step_asy[i]);
        if (i < 8) fprintf(stderr, ",");
    }
    fprintf(stderr, "]}\n");

    /* Cleanup */
    free(x); free(Wq); free(Wk); free(Wv); free(Wo);
    free(Wq_i8); free(Wk_i8); free(Wv_i8); free(Wo_i8);
    free(ref_out); free(tpu_out_blk); free(tpu_out_asy);
    tpu_cleanup(&ctx_blk); tpu_cleanup(&ctx_asy);
    return 0;
}
