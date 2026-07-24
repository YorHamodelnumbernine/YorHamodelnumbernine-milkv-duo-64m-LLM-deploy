/* cvikernel matmul v2 — test matrix multiplication parameter combinations.
   Tries: different sizes (alignment), add_result vs not, bias vs NULL. */
#include "../common/tpu_bench.h"

static void fill_identity(int8_t *m, int size) {
  memset(m, 0, size*size);
  for (int i = 0; i < size; i++) m[i*size + i] = 1;
}

static int try_matmul(tpu_ctx *ctx, cvk_context_t *cvk,
                       int M, int K, int N,
                       int add_result,    /* 1=add bias to result */
                       int use_bias)      /* 1=load bias tensor, 0=NULL */
{
  int left_sz  = M * K;
  int right_sz = K * N;
  int res_sz   = M * N;
  int8_t left[M*K], right[K*N];
  memset(left, 0, sizeof(left)); memset(right, 0, sizeof(right));

  /* Simple test: left=identity(K), right=ones → result per element = sum of col */
  if (M == K) fill_identity(left, M);  // identity
  else for (int i=0;i<left_sz;i++) left[i]=1;
  for (int i=0;i<right_sz;i++) right[i]=1;

  #define GM_MAT_L  0
  #define GM_MAT_R  256
  #define GM_MAT_B  512
  #define GM_MAT_O  768

  memcpy(ctx->neuron_vaddr + GM_MAT_L, left, left_sz);
  memcpy(ctx->neuron_vaddr + GM_MAT_R, right, right_sz);
  memset(ctx->neuron_vaddr + GM_MAT_B, 0, res_sz);
  if (use_bias) {
    for (int i = 0; i < res_sz; i++) ctx->neuron_vaddr[GM_MAT_B + i] = 10;
  }
  CVI_RT_MemFlush(ctx->rt_handle, ctx->neuron_mem);

  cvk_ml_shape_t sl = cvk->ops->ml_default_shape(cvk, M, K, CVK_FMT_I8);
  cvk_ml_shape_t sr = cvk->ops->ml_default_shape(cvk, K, N, CVK_FMT_I8);
  cvk_ml_shape_t so = cvk->ops->ml_default_shape(cvk, M, N, CVK_FMT_I8);

  cvk_ml_t *ml_l = cvk->ops->lmem_alloc_matrix(cvk, sl, CVK_FMT_I8, 1);
  cvk_ml_t *ml_r = cvk->ops->lmem_alloc_matrix(cvk, sr, CVK_FMT_I8, 1);
  cvk_ml_t *ml_o = cvk->ops->lmem_alloc_matrix(cvk, so, CVK_FMT_I8, 1);
  cvk_ml_t *ml_b = NULL;
  if (use_bias) {
    ml_b = cvk->ops->lmem_alloc_matrix(cvk, so, CVK_FMT_I8, 1);
    if (!ml_b) { fprintf(stderr,"  bias alloc fail\n"); goto fail; }
  }
  if (!ml_l||!ml_r||!ml_o) { fprintf(stderr,"  alloc fail\n"); goto fail; }

  cvk_mg_t mg_l = {0, TPU_PA(ctx,GM_MAT_L), CVK_FMT_I8, {M,K}, {K}};
  cvk_mg_t mg_r = {0, TPU_PA(ctx,GM_MAT_R), CVK_FMT_I8, {K,N}, {N}};
  cvk_mg_t mg_o = {0, TPU_PA(ctx,GM_MAT_O), CVK_FMT_I8, {M,N}, {N}};

  cvk->ops->tdma_g2l_matrix_copy(cvk, &(cvk_tdma_g2l_matrix_copy_param_t){&mg_l, ml_l});
  cvk->ops->tdma_g2l_matrix_copy(cvk, &(cvk_tdma_g2l_matrix_copy_param_t){&mg_r, ml_r});

  cvk_mg_t mg_b;
  if (use_bias && ml_b) {
    mg_b = (cvk_mg_t){0, TPU_PA(ctx,GM_MAT_B), CVK_FMT_I8, {M,N}, {N}};
    cvk->ops->tdma_g2l_matrix_copy(cvk, &(cvk_tdma_g2l_matrix_copy_param_t){&mg_b, ml_b});
  }

  cvk_tiu_matrix_multiplication_param_t p = {
    .res = ml_o, .left = ml_l, .right = ml_r,
    .bias = ml_b,
    .lshift_bits = 0, .rshift_bits = 0,
    .res_is_int8 = 1, .relu_enable = 0,
    .add_result = add_result,
    .ps32_mode = 0,
  };

  fprintf(stderr,"  [try] (%dx%d)*(%dx%d) add=%d bias=%d (l.shape col=%d) ... ",
    M,K,K,N, add_result, use_bias, sl.col);
  cvk->ops->tiu_matrix_multiplication(cvk, &p);
  fprintf(stderr,"tiu_mm OK, ");

  cvk->ops->tdma_l2g_matrix_copy(cvk, &(cvk_tdma_l2g_matrix_copy_param_t){ml_o, &mg_o});

  int64_t t_ns = tpu_submit(ctx);
  if (t_ns < 0) {
    fprintf(stderr,"Submit fail!\n");
    goto fail;
  }

  CVI_RT_MemInvld(ctx->rt_handle, ctx->neuron_mem);
  int8_t *v = (int8_t*)ctx->neuron_vaddr + GM_MAT_O;

  /* Compute expected — use heap to avoid VLA goto issue */
  int8_t *exp = (int8_t*)alloca(M * N);
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      int sum = 0;
      for (int k = 0; k < K; k++) {
        sum += (int)left[i*K + k] * (int)right[k*N + j];
      }
      if (add_result && use_bias) sum += 10;
      exp[i*N + j] = (int8_t)sum;
    }
  }

  int ok = 1;
  for (int i = 0; i < res_sz; i++)
    if (v[i] != exp[i]) { ok = 0; break; }
  fprintf(stderr,"%s (%.1f us)\n", ok ? "OK" : "MISMATCH", t_ns/1000.0);
  if (!ok) {
    fprintf(stderr,"    got:"); for(int i=0;i<res_sz;i++) fprintf(stderr,"%d,",v[i]);
    fprintf(stderr,"  exp:"); for(int i=0;i<res_sz;i++) fprintf(stderr,"%d,",exp[i]);
    fprintf(stderr,"\n");
  }

  cvk->ops->lmem_free_matrix(cvk, ml_o);
  cvk->ops->lmem_free_matrix(cvk, ml_r);
  cvk->ops->lmem_free_matrix(cvk, ml_l);
  if (ml_b) cvk->ops->lmem_free_matrix(cvk, ml_b);
  return ok ? 0 : 1;

fail:
  if (ml_o) cvk->ops->lmem_free_matrix(cvk, ml_o);
  if (ml_r) cvk->ops->lmem_free_matrix(cvk, ml_r);
  if (ml_l) cvk->ops->lmem_free_matrix(cvk, ml_l);
  if (ml_b) cvk->ops->lmem_free_matrix(cvk, ml_b);
  return -1;
}

int main() {
  tpu_ctx ctx;
  if (tpu_init(&ctx, 4096) != 0) return 1;
  cvk_context_t *cvk = ctx.cvk_ctx;

  printf("  [matmul_cvk_v2] testing matrix sizes & bias options...\n\n");

  /* 2x2: minimum test */
  printf("  --- 2x2 ---\n");
  try_matmul(&ctx, cvk, 2,2,2, 0, 0);
  try_matmul(&ctx, cvk, 2,2,2, 1, 1);

  /* 4x4 */
  printf("  --- 4x4 ---\n");
  try_matmul(&ctx, cvk, 4,4,4, 0, 0);
  try_matmul(&ctx, cvk, 4,4,4, 1, 1);

  /* 8x8: should be aligned */
  printf("  --- 8x8 ---\n");
  try_matmul(&ctx, cvk, 8,8,8, 0, 0);
  try_matmul(&ctx, cvk, 8,8,8, 1, 1);

  /* 16x16 */
  printf("  --- 16x16 ---\n");
  try_matmul(&ctx, cvk, 16,16,16, 0, 0);
  try_matmul(&ctx, cvk, 16,16,16, 1, 1);

  tpu_cleanup(&ctx);
  return 0;
}
