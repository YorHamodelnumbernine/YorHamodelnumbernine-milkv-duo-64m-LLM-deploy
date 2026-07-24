/* Complete MHA from SD card: read weights + input from SD card, quantize,
   then run full MHA pipeline on TPU. Per-step timing throughout.
   Build: make mha_sdcard_full
   Run:   ./mha_sdcard_full [d_model] [n_heads] [seq_len]
*/
#include "common/tpu_bench.h"
#include "common/mha_descriptor.h"
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

/* ---- Scratch for TPU matmul ---- */
#define MATMUL_SCRATCH 0x00000

/* ---- Timer ---- */
static double tick(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

/* ---- File I/O ---- */
static int read_file(const char *path, void *buf, int expected_sz) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "  open(%s) failed\n", path); return -1; }
    int n = read(fd, buf, expected_sz);
    close(fd);
    if (n != expected_sz) {
        fprintf(stderr, "  read(%s) got %d, expected %d\n", path, n, expected_sz);
        return -1;
    }
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

/* ---- Quantization ---- */
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

/* ---- CPU Math ---- */
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

static void transpose_i8(int8_t *dst, const int8_t *src, int rows, int cols) {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            dst[j * rows + i] = src[i * cols + j];
}

/* ---- CPU Reference: pure FP32 MHA ---- */
static void mha_ref_fp32(int S, int D, int H, int d, float scale,
    const float *x, const float *Wq, const float *Wk, const float *Wv,
    const float *Wo, float *output)
{
    int total = S * D;
    float *Q = (float *)malloc(total * sizeof(float));
    float *K = (float *)malloc(total * sizeof(float));
    float *V = (float *)malloc(total * sizeof(float));

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

    int half = d / 2;
    float *cos_tab = (float *)malloc(S * half * sizeof(float));
    float *sin_tab = (float *)malloc(S * half * sizeof(float));
    rope_precompute(S, d, 10000.0f, cos_tab, sin_tab);
    for (int h = 0; h < H; h++) {
        float *Qh = (float *)malloc(S * d * sizeof(float));
        float *Kh = (float *)malloc(S * d * sizeof(float));
        for (int r = 0; r < S; r++) {
            for (int c = 0; c < d; c++) {
                Qh[r * d + c] = Q[r * D + h * d + c];
                Kh[r * d + c] = K[r * D + h * d + c];
            }
        }
        apply_rope_f32_tab(Qh, S, d, cos_tab, sin_tab);
        apply_rope_f32_tab(Kh, S, d, cos_tab, sin_tab);
        for (int r = 0; r < S; r++) {
            for (int c = 0; c < d; c++) {
                Q[r * D + h * d + c] = Qh[r * d + c];
                K[r * D + h * d + c] = Kh[r * d + c];
            }
        }
        free(Qh); free(Kh);
    }
    free(cos_tab); free(sin_tab);

    /* V stays as-is, already computed */

    float *Scores = (float *)malloc(H * S * S * sizeof(float));
    float *Attn   = (float *)malloc(total * sizeof(float));
    for (int h = 0; h < H; h++) {
        float *Qh = (float *)malloc(S * d * sizeof(float));
        float *Kh = (float *)malloc(S * d * sizeof(float));
        float *Vh = (float *)malloc(S * d * sizeof(float));
        for (int r = 0; r < S; r++) {
            for (int c = 0; c < d; c++) {
                Qh[r * d + c] = Q[r * D + h * d + c];
                Kh[r * d + c] = K[r * D + h * d + c];
                Vh[r * d + c] = V[r * D + h * d + c];
            }
        }
        float *Sh = Scores + h * S * S;
        for (int i = 0; i < S; i++) {
            for (int j = 0; j < S; j++) {
                float s = 0;
                for (int kk = 0; kk < d; kk++)
                    s += Qh[i * d + kk] * Kh[j * d + kk];
                Sh[i * S + j] = s * scale;
            }
        }
        softmax_f32(Sh, S, S);
        float *Ah = Attn + h * S * d;
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

    float *A_inter = (float *)malloc(total * sizeof(float));
    for (int h = 0; h < H; h++)
        for (int r = 0; r < S; r++)
            memcpy(A_inter + r * D + h * d, Attn + h * S * d + r * d, d * sizeof(float));

    for (int i = 0; i < S; i++) {
        for (int j = 0; j < D; j++) {
            float o = 0;
            for (int kk = 0; kk < D; kk++)
                o += A_inter[i * D + kk] * Wo[kk * D + j];
            output[i * D + j] = o;
        }
    }
    free(A_inter); free(Q); free(K); free(V); free(Scores); free(Attn);
}

/* ---- TPU matmul helpers (from mha_attention.c) ---- */
static int tpu_find_tile_n(cvk_context_t *cvk, int M, int K) {
    int left = mha_lmem_matrix_bytes(M, K);
    for (int tn = 128; tn >= 16; tn -= 16) {
        int right = mha_lmem_matrix_bytes(K, tn);
        int out   = mha_lmem_matrix_bytes(M, tn);
        if (left + right + out <= 32768) return tn;
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

        CVI_RT_Submit(ctx->rt_khandle);
        CVI_RT_MemInvld(ctx->rt_handle, ctx->neuron_mem);
        memcpy(o_i8, ctx->neuron_vaddr + off_o, M * N);
        return 0;
    }

    /* Tiling path */
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
 * MHA Pipeline - blocking mode, per-step timing
 * step_us[0..11] = 12 steps total
 *   0=Read W from SD   1=Quant W
 *   2=Read input from SD  3=Quant input
 *   4=Q proj (TPU)  5=K proj (TPU)  6=V proj (TPU)
 *   7=RoPE+re-quant (CPU)  8=Scores Q*K^T (TPU)
 *   9=Softmax (CPU)  10=Attn*V (TPU)  11=Output proj (TPU)
 * ================================================================ */
static int mha_sdcard_pipeline(int S, int D, int H, int d, float softmax_scale,
    tpu_ctx *ctx, const char *sd_path,
    const int8_t *Wq_i8, const int8_t *Wk_i8, const int8_t *Wv_i8, const int8_t *Wo_i8,
    float *output, double *step_us)
{
    int total = S * D;
    cvk_context_t *cvk = ctx->cvk_ctx;
    int rc;

    /* Neuron memory buffers */
    float *Q_f32 = (float *)(ctx->neuron_vaddr + MHA_OFF_Q_F32);
    float *K_f32 = (float *)(ctx->neuron_vaddr + MHA_OFF_K_F32);
    float *Scores_f32 = (float *)(ctx->neuron_vaddr + MHA_OFF_SCORES_F32);
    float *Attn_f32  = (float *)(ctx->neuron_vaddr + MHA_OFF_ATTN_F32);
    int8_t *Q_i8 = (int8_t *)(ctx->neuron_vaddr + MHA_OFF_Q_I8);
    int8_t *K_i8 = (int8_t *)(ctx->neuron_vaddr + MHA_OFF_K_I8);
    int8_t *V_i8 = (int8_t *)(ctx->neuron_vaddr + MHA_OFF_V_I8);
    int8_t *S_i8 = (int8_t *)(ctx->neuron_vaddr + MHA_OFF_S_I8);
    int8_t *A_i8 = (int8_t *)(ctx->neuron_vaddr + MHA_OFF_A_I8);
    int8_t *O_i8 = (int8_t *)(ctx->neuron_vaddr + MHA_OFF_OUT_I8);

    /* ---- Step 2: Read input from SD ---- */
    char path[256];
    snprintf(path, sizeof(path), "%s/input.f32", sd_path);
    int input_bytes = total * (int)sizeof(float);
    float *x_rd = (float *)malloc(input_bytes);
    if (!x_rd) { fprintf(stderr, "OOM x_rd\n"); return -1; }
    double ts = tick();
    if (read_file(path, x_rd, input_bytes) != 0) { free(x_rd); return -1; }
    step_us[2] = tick() - ts;

    /* ---- Step 3: Quantize input ---- */
    ts = tick();
    int8_t *x_i8 = (int8_t *)malloc(total);
    if (!x_i8) { free(x_rd); return -1; }
    int zp_x; float sc_x = compute_scale(x_rd, total, &zp_x);
    quantize_i8(x_i8, x_rd, total, sc_x, zp_x);
    step_us[3] = tick() - ts;
    free(x_rd);

    /* ---- Step 4: Q = x * Wq ---- */
    ts = tick();
    rc = tpu_matmul(ctx, cvk, x_i8, S, D, Wq_i8, D, Q_i8, MATMUL_SCRATCH);
    step_us[4] = tick() - ts;
    if (rc) { free(x_i8); return -4; }

    /* ---- Step 5: K = x * Wk ---- */
    ts = tick();
    rc = tpu_matmul(ctx, cvk, x_i8, S, D, Wk_i8, D, K_i8, MATMUL_SCRATCH);
    step_us[5] = tick() - ts;
    if (rc) { free(x_i8); return -5; }

    /* Dequant Q, K */
    float sc_qk_out = sc_x * 0.001f;
    dequantize_f32(Q_f32, Q_i8, total, sc_qk_out, 0);
    dequantize_f32(K_f32, K_i8, total, sc_qk_out, 0);

    /* ---- Step 6: V = x * Wv ---- */
    ts = tick();
    rc = tpu_matmul(ctx, cvk, x_i8, S, D, Wv_i8, D, V_i8, MATMUL_SCRATCH);
    step_us[6] = tick() - ts;
    free(x_i8);
    if (rc) return -6;

    /* ---- Step 7: RoPE on Q and K (CPU, lookup-table) ---- */
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

    /* Re-quantize Q,K after RoPE */
    int zp_q2; float sc_q2 = compute_scale(Q_f32, total, &zp_q2);
    quantize_i8(Q_i8, Q_f32, total, sc_q2, zp_q2);
    int zp_k2; float sc_k2 = compute_scale(K_f32, total, &zp_k2);
    quantize_i8(K_i8, K_f32, total, sc_k2, zp_k2);
    step_us[7] = tick() - ts;

    /* ---- Step 8: Per-head Scores = Q * K^T ---- */
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
        rc = tpu_matmul(ctx, cvk, Qh, S, d, Kh_t, S, S_i8 + h * S * S, MATMUL_SCRATCH);
        free(Qh); free(Kh); free(Kh_t);
        if (rc) return -8;
    }
    float sc_s_out = sc_q2 * sc_k2;
    dequantize_f32(Scores_f32, S_i8, H * S * S, sc_s_out, 0);
    for (int i = 0; i < H * S * S; i++) Scores_f32[i] *= softmax_scale;
    step_us[8] = tick() - ts;

    /* ---- Step 9: Softmax (CPU) ---- */
    ts = tick();
    for (int h = 0; h < H; h++)
        softmax_f32(Scores_f32 + h * S * S, S, S);
    step_us[9] = tick() - ts;

    /* Re-quantize softmax scores + dequant/re-quant V */
    int zp_s2; float sc_s2 = compute_scale(Scores_f32, H * S * S, &zp_s2);
    quantize_i8(S_i8, Scores_f32, H * S * S, sc_s2, zp_s2);
    float *V_f32 = (float *)(ctx->neuron_vaddr + MHA_OFF_V_F32);
    dequantize_f32(V_f32, V_i8, total, sc_qk_out, 0);
    int zp_v2; float sc_v2 = compute_scale(V_f32, total, &zp_v2);
    quantize_i8(V_i8, V_f32, total, sc_v2, zp_v2);

    /* ---- Step 10: Attn = Softmax * V (per head) ---- */
    ts = tick();
    for (int h = 0; h < H; h++) {
        int8_t *Vh = (int8_t *)malloc(S * d);
        for (int r = 0; r < S; r++)
            memcpy(Vh + r * d, V_i8 + r * D + h * d, d);
        rc = tpu_matmul(ctx, cvk, S_i8 + h * S * S, S, S, Vh, d, A_i8 + h * S * d, MATMUL_SCRATCH);
        free(Vh);
        if (rc) return -10;
    }
    float sc_a_out = sc_s2 * sc_v2;
    dequantize_f32(Attn_f32, A_i8, total, sc_a_out, 0);
    step_us[10] = tick() - ts;

    /* ---- Step 11: Output = Attn * Wo ---- */
    ts = tick();
    int8_t *A_interleaved = (int8_t *)malloc(total);
    for (int h = 0; h < H; h++)
        for (int r = 0; r < S; r++)
            memcpy(A_interleaved + r * D + h * d, A_i8 + h * S * d + r * d, d);

    int zp_a2; float sc_a2 = compute_scale(Attn_f32, total, &zp_a2);
    quantize_i8(A_interleaved, Attn_f32, total, sc_a2, zp_a2);
    rc = tpu_matmul(ctx, cvk, A_interleaved, S, D, Wo_i8, D, O_i8, MATMUL_SCRATCH);
    if (rc) { free(A_interleaved); return -11; }
    float sc_o_out = sc_a2 * 0.001f;
    dequantize_f32(output, O_i8, total, sc_o_out, 0);
    free(A_interleaved);
    step_us[11] = tick() - ts;

    return 0;
}

/* ================================================================
 * main
 * ================================================================ */
int main(int argc, char **argv) {
    setvbuf(stderr, NULL, _IONBF, 0);
    int d_model = argc > 1 ? atoi(argv[1]) : 256;
    int n_heads = argc > 2 ? atoi(argv[2]) : 8;
    int seq_len = argc > 3 ? atoi(argv[3]) : 32;

    int D = d_model, H = n_heads, S = seq_len, d = D / H;
    int total = S * D, w_sz = D * D;
    int input_bytes = total * (int)sizeof(float);
    int weight_bytes = w_sz * (int)sizeof(float);
    float softmax_scale = 1.0f / sqrtf((float)d);

    fprintf(stderr, "\n============================================================\n");
    fprintf(stderr, "  MHA from SD Card — D=%d H=%d d=%d S=%d\n", D, H, d, S);
    fprintf(stderr, "  Weights: %d KB x4  |  Input: %d KB\n",
            weight_bytes / 1024, input_bytes / 1024);
    fprintf(stderr, "============================================================\n");

    /* ---- Phase 1: Generate data & write to SD card ---- */
    fprintf(stderr, "\n[Phase 1] Generate FP32 data & write to SD card...\n");
    double t_gen = tick();

    float *W_f32[4];
    for (int w = 0; w < 4; w++) {
        W_f32[w] = (float *)malloc(weight_bytes);
        if (!W_f32[w]) { fprintf(stderr, "OOM W_f32[%d]\n", w); return 1; }
        for (int i = 0; i < w_sz; i++)
            W_f32[w][i] = (float)(rand() % 256 - 128) / 1000.0f;
    }
    float *x_f32 = (float *)malloc(input_bytes);
    if (!x_f32) { fprintf(stderr, "OOM x_f32\n"); return 1; }
    for (int i = 0; i < total; i++)
        x_f32[i] = (float)(rand() % 256 - 128) / 200.0f;

    const char *base = "/tmp/mha_bench";
    mkdir(base, 0755);
    char path[256];
    double t_wr = tick();
    snprintf(path, sizeof(path), "%s/Wq.f32", base); write_file(path, W_f32[0], weight_bytes);
    snprintf(path, sizeof(path), "%s/Wk.f32", base); write_file(path, W_f32[1], weight_bytes);
    snprintf(path, sizeof(path), "%s/Wv.f32", base); write_file(path, W_f32[2], weight_bytes);
    snprintf(path, sizeof(path), "%s/Wo.f32", base); write_file(path, W_f32[3], weight_bytes);
    snprintf(path, sizeof(path), "%s/input.f32", base); write_file(path, x_f32, input_bytes);
    double t_wr_done = tick();

    fprintf(stderr, "  Generate: %.1f ms  |  Write SD: %.1f ms\n",
            (t_wr - t_gen) / 1000.0, (t_wr_done - t_wr) / 1000.0);

    /* ---- Phase 1b: CPU FP32 Reference (while FP32 data still in memory) ---- */
    fprintf(stderr, "\n[Phase 1b] CPU FP32 Reference (before freeing FP32 buffers)...\n");
    float *ref_out = (float *)malloc(input_bytes);
    if (!ref_out) { fprintf(stderr, "OOM ref_out\n"); return 1; }
    double t_ref = tick();
    mha_ref_fp32(S, D, H, d, softmax_scale,
        x_f32, W_f32[0], W_f32[1], W_f32[2], W_f32[3], ref_out);
    double t_ref_done = tick() - t_ref;
    fprintf(stderr, "  CPU ref: %.1f ms\n", t_ref_done / 1000.0);

    /* Free write buffers (ref already computed, no copies needed) */
    for (int w = 0; w < 4; w++) free(W_f32[w]);
    free(x_f32);
    sync();

    /* ---- Phase 2: Read weights from SD + quantize (one at a time) ---- */
    fprintf(stderr, "\n[Phase 2] Read weights from SD & quantize (one at a time)...\n");
    double t_read_w[4], t_quant_w[4];
    int8_t *W_i8[4];
    int wp_zp[4]; float wp_sc[4];
    double t_rw_start = tick();

    for (int w = 0; w < 4; w++) {
        float *wbuf = (float *)malloc(weight_bytes);
        if (!wbuf) { fprintf(stderr, "OOM wbuf[%d]\n", w); return 1; }
        W_i8[w] = (int8_t *)malloc(w_sz);
        if (!W_i8[w]) { fprintf(stderr, "OOM W_i8[%d]\n", w); return 1; }

        snprintf(path, sizeof(path), "%s/W%c.f32", base, "qkvo"[w]);
        double t1 = tick();
        if (read_file(path, wbuf, weight_bytes) != 0) return 1;
        double t2 = tick();
        t_read_w[w] = t2 - t1;

        wp_sc[w] = compute_scale(wbuf, w_sz, &wp_zp[w]);
        quantize_i8(W_i8[w], wbuf, w_sz, wp_sc[w], wp_zp[w]);
        double t3 = tick();
        t_quant_w[w] = t3 - t2;

        free(wbuf);
        fprintf(stderr, "  W%c: read %7.0f us  quant %7.0f us\n",
                "qkvo"[w], t_read_w[w], t_quant_w[w]);
    }
    double t_rw_total = tick() - t_rw_start;

    fprintf(stderr, "  Weight I/O total: %.1f ms\n", t_rw_total / 1000.0);

    /* ---- Phase 3: Init TPU ---- */
    fprintf(stderr, "\n[Phase 3] Init TPU & run MHA pipeline...\n");
    tpu_ctx ctx;
    if (tpu_init(&ctx, MHA_TOTAL_SIZE) != 0) return 1;
    float *tpu_out = (float *)malloc(input_bytes);
    if (!tpu_out) { fprintf(stderr, "OOM tpu_out\n"); return 1; }

    double step_us[12] = {0};
    step_us[0] = t_read_w[0]+t_read_w[1]+t_read_w[2]+t_read_w[3];
    step_us[1] = t_quant_w[0]+t_quant_w[1]+t_quant_w[2]+t_quant_w[3];

    double t_pipe = tick();
    int rc = mha_sdcard_pipeline(S, D, H, d, softmax_scale,
        &ctx, base, W_i8[0], W_i8[1], W_i8[2], W_i8[3], tpu_out, step_us);
    double t_pipe_done = tick() - t_pipe;

    if (rc != 0) {
        fprintf(stderr, "  MHA pipeline FAILED at step %d\n", -rc);
    } else {
        fprintf(stderr, "  MHA pipeline OK (%.1f ms)\n", t_pipe_done / 1000.0);
    }

    /* ---- Phase 4: Verify TPU vs CPU Reference ---- */
    fprintf(stderr, "\n[Phase 4] Verification...\n");
    double max_err = 0, mse = 0;
    for (int i = 0; i < total; i++) {
        double err = fabs((double)tpu_out[i] - (double)ref_out[i]);
        if (err > max_err) max_err = err;
        mse += err * err;
    }
    mse /= total;
    fprintf(stderr, "  TPU vs CPU Ref:  max_err=%.4f  MSE=%.6f\n", max_err, mse);

    /* ---- Per-step timing table ---- */
    const char *step_names[12] = {
        "SD读权重(4文件)", "量化权重(4矩阵)",
        "SD读输入",       "量化输入",
        "Q=输入×Wq(TPU)", "K=输入×Wk(TPU)", "V=输入×Wv(TPU)",
        "RoPE+重量化(CPU)","Scores=Q×Kᵀ(TPU)",
        "Softmax(CPU)",   "Attn=Softmax×V(TPU)",
        "Output=Attn×Wo(TPU)"
    };

    fprintf(stderr, "\n============================================================\n");
    fprintf(stderr, "  %-30s %12s %12s\n", "Step", "Time(us)", "占比");
    fprintf(stderr, "  %-30s %12s %12s\n",
            "------------------------------", "----------", "----------");

    double total_all = step_us[0]+step_us[1]+t_pipe_done;
    /* step_us[4..11] = pipeline steps, step_us[0..3] = SD+quant */
    for (int i = 0; i < 12; i++) {
        fprintf(stderr, "  %2d.%-28s %12.1f %11.1f%%\n",
                i, step_names[i], step_us[i],
                step_us[i] / total_all * 100.0);
    }
    fprintf(stderr, "  %-30s %12s %12s\n",
            "------------------------------", "----------", "----------");
    fprintf(stderr, "  %-30s %12.1f %11.1f%%\n",
            "SD+Quant 小计 (Step 0)", step_us[0]+step_us[1]+step_us[2]+step_us[3],
            (step_us[0]+step_us[1]+step_us[2]+step_us[3]) / total_all * 100.0);
    fprintf(stderr, "  %-30s %12.1f %11.1f%%\n",
            "TPU Pipeline 小计", t_pipe_done,
            t_pipe_done / total_all * 100.0);
    fprintf(stderr, "  %-30s %12.1f %12s\n",
            "MHA 总计", total_all, "100.0%");

    /* ---- Phase 6: Machine-readable JSON for email ---- */
    fprintf(stderr, "\n--- MHA_SDCARD_JSON ---\n");
    fprintf(stderr, "{\"cfg\":{\"d_model\":%d,\"n_heads\":%d,\"head_dim\":%d,\"seq_len\":%d,"
            "\"softmax_scale\":%.4f},\n", D, H, d, S, softmax_scale);
    fprintf(stderr, " \"timing\":{\n");
    for (int i = 0; i < 12; i++) {
        fprintf(stderr, "  \"s%d_%s\":%.1f", i, step_names[i], step_us[i]);
        fprintf(stderr, "%s\n", i < 11 ? "," : "");
    }
    fprintf(stderr, " },\n");
    fprintf(stderr, " \"totals\":{\"sd_quant_us\":%.1f,\"pipeline_us\":%.1f,\"grand_us\":%.1f},\n",
            step_us[0]+step_us[1]+step_us[2]+step_us[3], t_pipe_done, total_all);
    fprintf(stderr, " \"ref_us\":%.1f,\"max_err\":%.4f,\"mse\":%.6f,\n",
            t_ref_done, max_err, mse);
    fprintf(stderr, " \"speedup_vs_cpu\":%.2f}\n", t_ref_done / total_all);

    /* Cleanup */
    for (int w = 0; w < 4; w++) free(W_i8[w]);
    free(tpu_out); free(ref_out);
    tpu_cleanup(&ctx);

    fprintf(stderr, "\nDone.\n");
    return 0;
}
