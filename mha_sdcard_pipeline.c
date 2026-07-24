/* MHA from SD card with block-level pipeline: Load → Quant → Compute.
   Weights stored column-major on SD so column-block reads are contiguous.
   Pipeline: while block N loads from SD, block N-1 quantizes, block N-2 runs on TPU.
   Build: make mha_sdcard_pipeline
   Run:   ./mha_sdcard_pipeline [d_model] [n_heads] [seq_len] [num_blocks]
*/
#include "common/tpu_bench.h"
#include "common/mha_descriptor.h"
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

#define MAX_INFLIGHT 3

static double tick(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

/* ---- File I/O ---- */
static int read_file(const char *path, void *buf, int sz) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "  open(%s) failed\n", path); return -1; }
    int n = read(fd, buf, sz);
    close(fd);
    if (n != sz) { fprintf(stderr, "  read(%s) %d != %d\n", path, n, sz); return -1; }
    return 0;
}

static int read_file_at(const char *path, void *buf, int sz, int offset) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "  open(%s) failed\n", path); return -1; }
    if (lseek(fd, offset, SEEK_SET) < 0) { close(fd); return -1; }
    int n = read(fd, buf, sz);
    close(fd);
    if (n != sz) { fprintf(stderr, "  read_at(%s,+%d) %d != %d\n", path, offset, n, sz); return -1; }
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

/* Transpose one block: column-major [D, B] → row-major [D, B] */
static void transpose_block_c2r(int8_t *dst, const int8_t *src, int D, int B) {
    for (int col = 0; col < B; col++)
        for (int row = 0; row < D; row++)
            dst[row * B + col] = src[col * D + row];
}

/* ---- CPU Reference (row-major input, same as mha_sdcard_full.c) ---- */
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

static int tpu_matmul_accum(tpu_ctx *ctx, cvk_context_t *cvk,
    const void *left, int M, int K, const void *right, int N,
    void *result, uint32_t scratch_off)
{
    const int8_t *l_i8 = (const int8_t *)left;
    const int8_t *r_i8 = (const int8_t *)right;
    int8_t       *o_i8 = (int8_t *)result;

    uint32_t off_l = scratch_off;
    uint32_t off_r = scratch_off + M * K;

    /* For small matmuls (N≤64), single tile fits in LMEM */
    int left_bytes  = mha_lmem_matrix_bytes(M, K);
    int right_bytes = mha_lmem_matrix_bytes(K, N);
    int out_bytes   = mha_lmem_matrix_bytes(M, N);
    int need_tile = (left_bytes + right_bytes + out_bytes > 32768);

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

    uint32_t off_o_base = scratch_off + M * K + K * tile_n;
    memcpy(ctx->neuron_vaddr + off_l, l_i8, M * K);
    CVI_RT_MemFlush(ctx->rt_handle, ctx->neuron_mem);

    for (int n_start = 0; n_start < N; n_start += tile_n) {
        int cur_n = (n_start + tile_n <= N) ? tile_n : N - n_start;

        uint8_t *tile_dst = ctx->neuron_vaddr + off_r;
        for (int row = 0; row < K; row++)
            memcpy(tile_dst + row * cur_n, r_i8 + row * N + n_start, cur_n);
        CVI_RT_MemFlush(ctx->rt_handle, ctx->neuron_mem);

        cvk_ml_t *ml_l = cvk->ops->lmem_alloc_matrix(cvk,
            cvk->ops->ml_default_shape(cvk, M, K, CVK_FMT_I8), CVK_FMT_I8, 1);
        cvk_ml_t *ml_r = cvk->ops->lmem_alloc_matrix(cvk,
            cvk->ops->ml_default_shape(cvk, K, cur_n, CVK_FMT_I8), CVK_FMT_I8, 1);
        cvk_ml_t *ml_o = cvk->ops->lmem_alloc_matrix(cvk,
            cvk->ops->ml_default_shape(cvk, M, cur_n, CVK_FMT_I8), CVK_FMT_I8, 1);

        cvk->ops->tdma_g2l_matrix_copy(cvk, &(cvk_tdma_g2l_matrix_copy_param_t){
            .src = &(cvk_mg_t){0, TPU_PA(ctx, off_l), CVK_FMT_I8, {M, K}, {K}},
            .dst = ml_l});
        cvk->ops->tdma_g2l_matrix_copy(cvk, &(cvk_tdma_g2l_matrix_copy_param_t){
            .src = &(cvk_mg_t){0, TPU_PA(ctx, off_r), CVK_FMT_I8, {K, cur_n}, {cur_n}},
            .dst = ml_r});

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
 * Pipelined weight load + quant + TPU compute for one projection
 *   Reads FP32 column-major weight from file, splits into B column blocks.
 *   Pipeline: Load(N) | Quant(N-1) | Compute(N-2) using 3-stage overlap.
 *   result_i8[S, D] gets columns from each block's TPU output.
 *
 * Returns per-cycle timing arrays (caller allocates B+2 entries each):
 *   cyc_load_us, cyc_quant_us, cyc_comp_us
 *   cyc_load_us[i] = time spent in SD read during cycle i
 *   cyc_quant_us[i] = time spent in quantize+transpose during cycle i
 *   cyc_comp_us[i] = time spent in TPU matmul+submit during cycle i
 * ================================================================ */
static int pipeline_weight(tpu_ctx *ctx, cvk_context_t *cvk,
    const char *filepath, int D, int B,
    const int8_t *x_i8, int S, int8_t *result_i8,
    uint32_t scratch_off,
    double *cyc_load_us, double *cyc_quant_us, double *cyc_comp_us,
    double *total_us)
{
    int block_cols = D / B;
    int block_fp32  = block_cols * D * (int)sizeof(float);
    int block_total = block_cols * D;  /* elements per block */

    /* Ring buffers for 3-stage pipeline */
    float  *fp32_ring[MAX_INFLIGHT];
    int8_t *i8_ring[MAX_INFLIGHT];
    int8_t *tp_ring[MAX_INFLIGHT];  /* transposed: row-major [D, block_cols] */
    double  load_start[MAX_INFLIGHT] = {0};
    int     load_done[MAX_INFLIGHT] = {0};

    for (int i = 0; i < MAX_INFLIGHT; i++) {
        fp32_ring[i] = (float *)malloc(block_fp32);
        i8_ring[i]   = (int8_t *)malloc(block_total);
        tp_ring[i]   = (int8_t *)malloc(block_total);
        if (!fp32_ring[i] || !i8_ring[i] || !tp_ring[i]) {
            fprintf(stderr, "  OOM ring[%d]\n", i); return -1;
        }
        load_done[i] = 0;
    }

    int total_cycles = B + 2;  /* includes pipeline drain */
    double t_start = tick();

    for (int cycle = 0; cycle < total_cycles; cycle++) {
        int i_load  = cycle % MAX_INFLIGHT;
        int i_quant = (cycle - 1) % MAX_INFLIGHT;
        int i_comp  = (cycle - 2) % MAX_INFLIGHT;

        double t_cyc = tick();
        cyc_load_us[cycle] = 0;
        cyc_quant_us[cycle] = 0;
        cyc_comp_us[cycle] = 0;

        /* Stage 3: TPU compute block (cycle-2) */
        if (cycle >= 2 && (cycle - 2) < B) {
            double t0 = tick();
            int col_start = (cycle - 2) * block_cols;
            int rc = tpu_matmul_accum(ctx, cvk,
                x_i8, S, D, tp_ring[i_comp], block_cols,
                result_i8 + col_start, scratch_off);
            if (rc) { fprintf(stderr, "  TPU fail blk %d\n", cycle-2); return -1; }
            cyc_comp_us[cycle] = tick() - t0;
            load_done[i_comp] = 0; /* free slot */
        }

        /* Stage 2: Quantize + transpose block (cycle-1) */
        if (cycle >= 1 && (cycle - 1) < B) {
            double t0 = tick();
            int zp; float sc = compute_scale(fp32_ring[i_quant], block_total, &zp);
            quantize_i8(i8_ring[i_quant], fp32_ring[i_quant], block_total, sc, zp);
            /* Transpose: column-major [D, block_cols] → row-major [D, block_cols] */
            transpose_block_c2r(tp_ring[i_quant], i8_ring[i_quant], D, block_cols);
            cyc_quant_us[cycle] = tick() - t0;
            load_done[i_quant] = 0; /* consumed by quant */
        }

        /* Stage 1: SD read block (cycle) */
        if (cycle < B) {
            double t0 = tick();
            int offset = cycle * block_fp32;
            read_file_at(filepath, fp32_ring[i_load], block_fp32, offset);
            cyc_load_us[cycle] = tick() - t0;
            load_done[i_load] = 1;
        }
    }

    *total_us = tick() - t_start;

    for (int i = 0; i < MAX_INFLIGHT; i++) {
        free(fp32_ring[i]); free(i8_ring[i]); free(tp_ring[i]);
    }
    return 0;
}

/* ================================================================
 * Full MHA with pipelined weight loading
 * ================================================================ */
int main(int argc, char **argv) {
    setvbuf(stderr, NULL, _IONBF, 0);
    int d_model = argc > 1 ? atoi(argv[1]) : 256;
    int n_heads = argc > 2 ? atoi(argv[2]) : 8;
    int seq_len = argc > 3 ? atoi(argv[3]) : 32;
    int n_blocks = argc > 4 ? atoi(argv[4]) : 8;

    int D = d_model, H = n_heads, S = seq_len, d = D / H;
    int total = S * D, w_sz = D * D;
    int input_bytes = total * (int)sizeof(float);
    int weight_bytes = w_sz * (int)sizeof(float);
    float softmax_scale = 1.0f / sqrtf((float)d);
    int block_cols = D / n_blocks;

    fprintf(stderr, "\n============================================================\n");
    fprintf(stderr, "  MHA SD-Card Pipeline — D=%d H=%d d=%d S=%d  B=%d blocks (cols=%d)\n",
            D, H, d, S, n_blocks, block_cols);
    fprintf(stderr, "  Weights: %d KB x4 (column-major) | Input: %d KB\n",
            weight_bytes / 1024, input_bytes / 1024);
    fprintf(stderr, "  Pipeline: Load → Quant → Compute  (3-stage overlap)\n");
    fprintf(stderr, "============================================================\n");

    /* ---- Phase 1: Generate column-major FP32 data & write to SD ---- */
    fprintf(stderr, "\n[Phase 1] Generate column-major FP32 & write to SD...\n");
    double t_gen = tick();

    float *W_f32[4];
    for (int w = 0; w < 4; w++) {
        W_f32[w] = (float *)malloc(weight_bytes);
        if (!W_f32[w]) { fprintf(stderr, "OOM W_f32[%d]\n", w); return 1; }
    }
    float *x_f32 = (float *)malloc(input_bytes);
    if (!x_f32) { fprintf(stderr, "OOM x_f32\n"); return 1; }

    /* Generate: W in column-major (W[j][i] at file[j*D + i]), x in row-major */
    srand(42);
    for (int w = 0; w < 4; w++)
        for (int j = 0; j < D; j++)
            for (int i = 0; i < D; i++)
                W_f32[w][j * D + i] = (float)(rand() % 256 - 128) / 1000.0f;
    for (int i = 0; i < total; i++)
        x_f32[i] = (float)(rand() % 256 - 128) / 200.0f;

    /* Also make row-major copies for CPU reference */
    float *Wq_rm = (float *)malloc(weight_bytes);
    float *Wk_rm = (float *)malloc(weight_bytes);
    float *Wv_rm = (float *)malloc(weight_bytes);
    float *Wo_rm = (float *)malloc(weight_bytes);
    for (int i = 0; i < D; i++)
        for (int j = 0; j < D; j++) {
            /* W_f32[w][j*D + i] = column-major: col j, row i */
            Wq_rm[i * D + j] = W_f32[0][j * D + i];
            Wk_rm[i * D + j] = W_f32[1][j * D + i];
            Wv_rm[i * D + j] = W_f32[2][j * D + i];
            Wo_rm[i * D + j] = W_f32[3][j * D + i];
        }

    /* CPU Reference */
    fprintf(stderr, "  Running CPU FP32 reference...\n");
    float *ref_out = (float *)malloc(input_bytes);
    if (!ref_out) { fprintf(stderr, "OOM ref_out\n"); return 1; }
    double t_ref = tick();
    mha_ref_fp32(S, D, H, d, softmax_scale,
        x_f32, Wq_rm, Wk_rm, Wv_rm, Wo_rm, ref_out);
    double ref_us = tick() - t_ref;
    fprintf(stderr, "  CPU ref: %.1f ms\n", ref_us / 1000.0);

    /* Free row-major ref copies immediately after ref to save RAM */
    free(Wq_rm); free(Wk_rm); free(Wv_rm); free(Wo_rm);

    /* Write column-major files to SD (free each W_f32 after write to save RAM) */
    const char *base = "/tmp/mha_bench";
    mkdir(base, 0755);
    char path[256];
    snprintf(path, sizeof(path), "%s/Wq.f32", base); write_file(path, W_f32[0], weight_bytes); free(W_f32[0]);
    snprintf(path, sizeof(path), "%s/Wk.f32", base); write_file(path, W_f32[1], weight_bytes); free(W_f32[1]);
    snprintf(path, sizeof(path), "%s/Wv.f32", base); write_file(path, W_f32[2], weight_bytes); free(W_f32[2]);
    snprintf(path, sizeof(path), "%s/Wo.f32", base); write_file(path, W_f32[3], weight_bytes); free(W_f32[3]);
    snprintf(path, sizeof(path), "%s/input.f32", base); write_file(path, x_f32, input_bytes); free(x_f32);
    double t_wr = tick();
    fprintf(stderr, "  Generate+Write: %.1f ms\n", (t_wr - t_gen) / 1000.0);

    sync();

    /* ---- Phase 2: Read input + Quantize (input is small, no pipeline needed) ---- */
    fprintf(stderr, "\n[Phase 2] Read & quantize input from SD...\n");
    float *x_rd = (float *)malloc(input_bytes);
    int8_t *x_i8 = (int8_t *)malloc(total);
    if (!x_rd || !x_i8) { fprintf(stderr, "OOM input\n"); return 1; }
    snprintf(path, sizeof(path), "%s/input.f32", base);
    double t_ri = tick();
    if (read_file(path, x_rd, input_bytes) != 0) return 1;
    double t_ri_done = tick();
    int zp_x; float sc_x = compute_scale(x_rd, total, &zp_x);
    quantize_i8(x_i8, x_rd, total, sc_x, zp_x);
    double t_qi = tick();
    fprintf(stderr, "  Read: %.0f us  Quant: %.0f us\n",
            t_ri_done - t_ri, t_qi - t_ri_done);
    free(x_rd);

    /* ---- Phase 3: Init TPU ---- */
    fprintf(stderr, "\n[Phase 3] Init TPU & run pipelined MHA...\n");
    tpu_ctx ctx;
    if (tpu_init(&ctx, MHA_TOTAL_SIZE) != 0) return 1;

    cvk_context_t *cvk = ctx.cvk_ctx;
    int8_t *Q_i8 = (int8_t *)(ctx.neuron_vaddr + MHA_OFF_Q_I8);
    int8_t *K_i8 = (int8_t *)(ctx.neuron_vaddr + MHA_OFF_K_I8);
    int8_t *V_i8 = (int8_t *)(ctx.neuron_vaddr + MHA_OFF_V_I8);
    float *Q_f32 = (float *)(ctx.neuron_vaddr + MHA_OFF_Q_F32);
    float *K_f32 = (float *)(ctx.neuron_vaddr + MHA_OFF_K_F32);
    float *V_f32 = (float *)(ctx.neuron_vaddr + MHA_OFF_V_F32);
    float *Scores_f32 = (float *)(ctx.neuron_vaddr + MHA_OFF_SCORES_F32);
    float *Attn_f32  = (float *)(ctx.neuron_vaddr + MHA_OFF_ATTN_F32);
    int8_t *S_i8 = (int8_t *)(ctx.neuron_vaddr + MHA_OFF_S_I8);
    int8_t *A_i8 = (int8_t *)(ctx.neuron_vaddr + MHA_OFF_A_I8);
    int8_t *O_i8 = (int8_t *)(ctx.neuron_vaddr + MHA_OFF_OUT_I8);

#define MATMUL_SCR 0x00000

    /* ---- Pipeline Q, K, V projections ---- */
    double w_total[4], w_load_sum[4], w_quant_sum[4], w_comp_sum[4];
    int max_cycles = n_blocks + 2;
    double load_us[max_cycles], quant_us[max_cycles], comp_us[max_cycles];

    fprintf(stderr, "\n--- Pipelined Q = x * Wq ---\n");
    int rc = pipeline_weight(&ctx, cvk, "/tmp/mha_bench/Wq.f32", D, n_blocks,
        x_i8, S, Q_i8, MATMUL_SCR, load_us, quant_us, comp_us, &w_total[0]);
    if (rc) return -1;
    w_load_sum[0] = 0; w_quant_sum[0] = 0; w_comp_sum[0] = 0;
    for (int c = 0; c < max_cycles; c++) {
        w_load_sum[0] += load_us[c];
        w_quant_sum[0] += quant_us[c];
        w_comp_sum[0] += comp_us[c];
        if (load_us[c] > 0 || quant_us[c] > 0 || comp_us[c] > 0)
            fprintf(stderr, "  Cycle%2d: Load=%7.0f  Quant=%7.0f  Compute=%7.0f us\n",
                    c, load_us[c], quant_us[c], comp_us[c]);
    }
    fprintf(stderr, "  Wq Total=%.0f us (Load=%.0f Quant=%.0f Compute=%.0f)\n\n",
            w_total[0], w_load_sum[0], w_quant_sum[0], w_comp_sum[0]);

    fprintf(stderr, "--- Pipelined K = x * Wk ---\n");
    rc = pipeline_weight(&ctx, cvk, "/tmp/mha_bench/Wk.f32", D, n_blocks,
        x_i8, S, K_i8, MATMUL_SCR, load_us, quant_us, comp_us, &w_total[1]);
    if (rc) return -1;
    w_load_sum[1] = 0; w_quant_sum[1] = 0; w_comp_sum[1] = 0;
    for (int c = 0; c < max_cycles; c++) {
        w_load_sum[1] += load_us[c];
        w_quant_sum[1] += quant_us[c];
        w_comp_sum[1] += comp_us[c];
        if (load_us[c] > 0 || quant_us[c] > 0 || comp_us[c] > 0)
            fprintf(stderr, "  Cycle%2d: Load=%7.0f  Quant=%7.0f  Compute=%7.0f us\n",
                    c, load_us[c], quant_us[c], comp_us[c]);
    }
    fprintf(stderr, "  Wk Total=%.0f us (Load=%.0f Quant=%.0f Compute=%.0f)\n\n",
            w_total[1], w_load_sum[1], w_quant_sum[1], w_comp_sum[1]);

    fprintf(stderr, "--- Pipelined V = x * Wv ---\n");
    rc = pipeline_weight(&ctx, cvk, "/tmp/mha_bench/Wv.f32", D, n_blocks,
        x_i8, S, V_i8, MATMUL_SCR, load_us, quant_us, comp_us, &w_total[2]);
    if (rc) return -1;
    w_load_sum[2] = 0; w_quant_sum[2] = 0; w_comp_sum[2] = 0;
    for (int c = 0; c < max_cycles; c++) {
        w_load_sum[2] += load_us[c];
        w_quant_sum[2] += quant_us[c];
        w_comp_sum[2] += comp_us[c];
        if (load_us[c] > 0 || quant_us[c] > 0 || comp_us[c] > 0)
            fprintf(stderr, "  Cycle%2d: Load=%7.0f  Quant=%7.0f  Compute=%7.0f us\n",
                    c, load_us[c], quant_us[c], comp_us[c]);
    }
    fprintf(stderr, "  Wv Total=%.0f us (Load=%.0f Quant=%.0f Compute=%.0f)\n\n",
            w_total[2], w_load_sum[2], w_quant_sum[2], w_comp_sum[2]);

    free(x_i8);

    /* ---- Dequantize Q, K ---- */
    float sc_qk_out = sc_x * 0.001f;
    dequantize_f32(Q_f32, Q_i8, total, sc_qk_out, 0);
    dequantize_f32(K_f32, K_i8, total, sc_qk_out, 0);

    /* ---- Step 7: RoPE on Q and K (CPU) ---- */
    double t_rope = tick();
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

    int zp_q2; float sc_q2 = compute_scale(Q_f32, total, &zp_q2);
    quantize_i8(Q_i8, Q_f32, total, sc_q2, zp_q2);
    int zp_k2; float sc_k2 = compute_scale(K_f32, total, &zp_k2);
    quantize_i8(K_i8, K_f32, total, sc_k2, zp_k2);
    double rope_us = tick() - t_rope;

    /* ---- Scores = Q * K^T ---- */
    double t_scores = tick();
    for (int h = 0; h < H; h++) {
        int8_t *Qh = (int8_t *)malloc(S * d);
        int8_t *Kh = (int8_t *)malloc(S * d);
        int8_t *Kh_t = (int8_t *)malloc(S * d);
        for (int r = 0; r < S; r++) {
            memcpy(Qh + r * d, Q_i8 + r * D + h * d, d);
            memcpy(Kh + r * d, K_i8 + r * D + h * d, d);
        }
        transpose_i8(Kh_t, Kh, S, d);
        tpu_matmul_accum(&ctx, cvk, Qh, S, d, Kh_t, S, S_i8 + h * S * S, MATMUL_SCR);
        free(Qh); free(Kh); free(Kh_t);
    }
    float sc_s_out = sc_q2 * sc_k2;
    dequantize_f32(Scores_f32, S_i8, H * S * S, sc_s_out, 0);
    for (int i = 0; i < H * S * S; i++) Scores_f32[i] *= softmax_scale;
    double scores_us = tick() - t_scores;

    /* ---- Softmax ---- */
    double t_sm = tick();
    for (int h = 0; h < H; h++)
        softmax_f32(Scores_f32 + h * S * S, S, S);
    double sm_us = tick() - t_sm;

    /* Re-quantize */
    int zp_s2; float sc_s2 = compute_scale(Scores_f32, H * S * S, &zp_s2);
    quantize_i8(S_i8, Scores_f32, H * S * S, sc_s2, zp_s2);
    dequantize_f32(V_f32, V_i8, total, sc_qk_out, 0);
    int zp_v2; float sc_v2 = compute_scale(V_f32, total, &zp_v2);
    quantize_i8(V_i8, V_f32, total, sc_v2, zp_v2);

    /* ---- Attn = Softmax * V ---- */
    double t_attn = tick();
    for (int h = 0; h < H; h++) {
        int8_t *Vh = (int8_t *)malloc(S * d);
        for (int r = 0; r < S; r++)
            memcpy(Vh + r * d, V_i8 + r * D + h * d, d);
        tpu_matmul_accum(&ctx, cvk, S_i8 + h * S * S, S, S, Vh, d, A_i8 + h * S * d, MATMUL_SCR);
        free(Vh);
    }
    float sc_a_out = sc_s2 * sc_v2;
    dequantize_f32(Attn_f32, A_i8, total, sc_a_out, 0);
    double attn_us = tick() - t_attn;

    /* ---- Output = Attn * Wo (pipelined) ---- */
    /* First, interleave and quantize Attn for Wo projection */
    int8_t *A_interleaved = (int8_t *)malloc(total);
    for (int h = 0; h < H; h++)
        for (int r = 0; r < S; r++)
            memcpy(A_interleaved + r * D + h * d, A_i8 + h * S * d + r * d, d);
    int zp_a2; float sc_a2 = compute_scale(Attn_f32, total, &zp_a2);
    quantize_i8(A_interleaved, Attn_f32, total, sc_a2, zp_a2);

    fprintf(stderr, "--- Pipelined Wo projection ---\n");
    rc = pipeline_weight(&ctx, cvk, "/tmp/mha_bench/Wo.f32", D, n_blocks,
        A_interleaved, S, O_i8, MATMUL_SCR, load_us, quant_us, comp_us, &w_total[3]);
    if (rc) { free(A_interleaved); return -1; }
    w_load_sum[3] = 0; w_quant_sum[3] = 0; w_comp_sum[3] = 0;
    for (int c = 0; c < max_cycles; c++) {
        w_load_sum[3] += load_us[c];
        w_quant_sum[3] += quant_us[c];
        w_comp_sum[3] += comp_us[c];
        if (load_us[c] > 0 || quant_us[c] > 0 || comp_us[c] > 0)
            fprintf(stderr, "  Cycle%2d: Load=%7.0f  Quant=%7.0f  Compute=%7.0f us\n",
                    c, load_us[c], quant_us[c], comp_us[c]);
    }
    fprintf(stderr, "  Wo Total=%.0f us (Load=%.0f Quant=%.0f Compute=%.0f)\n\n",
            w_total[3], w_load_sum[3], w_quant_sum[3], w_comp_sum[3]);

    float *tpu_out = (float *)malloc(input_bytes);
    float sc_o_out = sc_a2 * 0.001f;
    dequantize_f32(tpu_out, O_i8, total, sc_o_out, 0);
    free(A_interleaved);

    /* ---- Verify ---- */
    fprintf(stderr, "\n[Verify] TPU vs CPU Reference...\n");
    double max_err = 0, mse = 0;
    for (int i = 0; i < total; i++) {
        double err = fabs((double)tpu_out[i] - (double)ref_out[i]);
        if (err > max_err) max_err = err;
        mse += err * err;
    }
    mse /= total;
    fprintf(stderr, "  max_err=%.4f  MSE=%.6f\n", max_err, mse);

    /* ---- Summary ---- */
    double pipeline_total = w_total[0]+w_total[1]+w_total[2]+rope_us+
                            scores_us+sm_us+attn_us+w_total[3];
    double load_sum = w_load_sum[0]+w_load_sum[1]+w_load_sum[2]+w_load_sum[3];
    double quant_sum = w_quant_sum[0]+w_quant_sum[1]+w_quant_sum[2]+w_quant_sum[3];
    double comp_sum = w_comp_sum[0]+w_comp_sum[1]+w_comp_sum[2]+w_comp_sum[3];

    fprintf(stderr, "\n============================================================\n");
    fprintf(stderr, "  %-28s %12s\n", "Component", "Time(us)");
    fprintf(stderr, "  %-28s %12s\n", "----------------------------", "----------");
    fprintf(stderr, "  %-28s %12.1f\n", "Wq pipelined", w_total[0]);
    fprintf(stderr, "  %-28s %12.1f\n", "Wk pipelined", w_total[1]);
    fprintf(stderr, "  %-28s %12.1f\n", "Wv pipelined", w_total[2]);
    fprintf(stderr, "  %-28s %12.1f\n", "RoPE + re-quant (CPU)", rope_us);
    fprintf(stderr, "  %-28s %12.1f\n", "Scores = Q*K^T (TPU)", scores_us);
    fprintf(stderr, "  %-28s %12.1f\n", "Softmax (CPU)", sm_us);
    fprintf(stderr, "  %-28s %12.1f\n", "Attn = Softmax*V (TPU)", attn_us);
    fprintf(stderr, "  %-28s %12.1f\n", "Wo pipelined", w_total[3]);
    fprintf(stderr, "  %-28s %12s\n", "----------------------------", "----------");
    fprintf(stderr, "  %-28s %12.1f\n", "MHA Pipeline TOTAL", pipeline_total);
    fprintf(stderr, "  %-28s %12.1f\n", "  of which: SD Load", load_sum);
    fprintf(stderr, "  %-28s %12.1f\n", "  of which: Quantize", quant_sum);
    fprintf(stderr, "  %-28s %12.1f\n", "  of which: TPU Compute", comp_sum);
    fprintf(stderr, "  %-28s %12.1f\n", "CPU Ref", ref_us);
    fprintf(stderr, "  %-28s %11.2fx\n", "Speedup vs CPU", ref_us / pipeline_total);

    fprintf(stderr, "\n  Pipeline efficiency:\n");
    double sequential_est = load_sum + quant_sum + comp_sum;
    fprintf(stderr, "    Sequential estimate:  %.0f us (Load+Quant+Comp)\n", sequential_est);
    fprintf(stderr, "    Pipelined actual:     %.0f us (only Wq+Wk+Wv+Wo)\n",
            w_total[0]+w_total[1]+w_total[2]+w_total[3]);
    fprintf(stderr, "    Overlap saved:        %.0f us (%.1f%%)\n",
            sequential_est - (w_total[0]+w_total[1]+w_total[2]+w_total[3]),
            (1.0 - (w_total[0]+w_total[1]+w_total[2]+w_total[3])/sequential_est) * 100.0);

    /* Cleanup */
    free(tpu_out); free(ref_out);
    tpu_cleanup(&ctx);

    fprintf(stderr, "\nDone.\n");
    return 0;
}
