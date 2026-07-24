/* TPU matrix_multiplication_qm: INT8 matmul with quantization.
   left[2x2] * right[2x2] + bias. lshift=0, rshift=0, quan_m=1.
   left=[1,2;3,4], right=[1,2;3,4], bias=[1,1;1,1].
   result = left*right + bias = [7,10;15,22] + [1,1;1,1] = [8,11;16,23] */
#include "../common/tpu_bench.h"

#define M 2
#define K 2
#define N 2

static const int8_t left[M*K]      = {1, 2, 3, 4};
static const int8_t right[K*N]     = {1, 2, 3, 4};
static const int8_t bias_data[M*N] = {1, 1, 1, 1};
static const int8_t exp[M*N]       = {8, 11, 16, 23};

int main() {
  tpu_ctx ctx;
  if (tpu_init(&ctx, 4096) != 0) return 1;
  cvk_context_t *cvk = ctx.cvk_ctx;

  memcpy(ctx.neuron_vaddr, left, M*K);
  memcpy(ctx.neuron_vaddr+16, right, K*N);
  memcpy(ctx.neuron_vaddr+32, bias_data, M*N);
  CVI_RT_MemFlush(ctx.rt_handle, ctx.neuron_mem);

  cvk_ml_shape_t s_left  = cvk->ops->ml_default_shape(cvk, M, K, CVK_FMT_I8);
  cvk_ml_shape_t s_right = cvk->ops->ml_default_shape(cvk, K, N, CVK_FMT_I8);
  cvk_ml_shape_t s_res   = cvk->ops->ml_default_shape(cvk, M, N, CVK_FMT_I8);

  cvk_ml_t *ml_l = cvk->ops->lmem_alloc_matrix(cvk, s_left, CVK_FMT_I8, 1);
  cvk_ml_t *ml_r = cvk->ops->lmem_alloc_matrix(cvk, s_right, CVK_FMT_I8, 1);
  cvk_ml_t *ml_res = cvk->ops->lmem_alloc_matrix(cvk, s_res, CVK_FMT_I8, 1);
  cvk_ml_t *ml_bias = cvk->ops->lmem_alloc_matrix(cvk, s_res, CVK_FMT_I8, 1);
  if (!ml_l||!ml_r||!ml_res||!ml_bias) {
    fprintf(stderr,"alloc fail\n"); tpu_cleanup(&ctx); return 1;
  }

  cvk_mg_t mg_l    = {0, TPU_PA(&ctx,0),  CVK_FMT_I8, {M,K}, {K}};
  cvk_mg_t mg_r    = {0, TPU_PA(&ctx,16), CVK_FMT_I8, {K,N}, {N}};
  cvk_mg_t mg_b    = {0, TPU_PA(&ctx,32), CVK_FMT_I8, {M,N}, {N}};
  cvk_mg_t mg_o    = {0, TPU_PA(&ctx,48), CVK_FMT_I8, {M,N}, {N}};

  cvk->ops->tdma_g2l_matrix_copy(cvk, &(cvk_tdma_g2l_matrix_copy_param_t){&mg_l, ml_l});
  cvk->ops->tdma_g2l_matrix_copy(cvk, &(cvk_tdma_g2l_matrix_copy_param_t){&mg_r, ml_r});
  cvk->ops->tdma_g2l_matrix_copy(cvk, &(cvk_tdma_g2l_matrix_copy_param_t){&mg_b, ml_bias});

  cvk_tiu_matrix_multiplication_qm_param_t p = {
    .res = ml_res, .left = ml_l, .right = ml_r,
    .bias = ml_bias,
    .lshift_bits = 0, .rshift_bits = 0,
    .res_is_int8 = 1, .relu_enable = 0,
    .add_result = 1, .ps32_mode = 0,
    .quan_m = 1,
  };
  cvk->ops->tiu_matrix_multiplication_qm(cvk, &p);

  cvk->ops->tdma_l2g_matrix_copy(cvk, &(cvk_tdma_l2g_matrix_copy_param_t){ml_res, &mg_o});

  int64_t t_ns = tpu_submit(&ctx);
  if (t_ns < 0) { fprintf(stderr,"submit fail\n"); goto out; }

  CVI_RT_MemInvld(ctx.rt_handle, ctx.neuron_mem);
  int8_t *v = (int8_t*)ctx.neuron_vaddr;
  fprintf(stderr,"[dbg] matmul_qm:"); for(int i=0;i<M*N;i++) fprintf(stderr,"%d,",v[48+i]);
  fprintf(stderr,"\n");
  int errs = tpu_check_i8("matmul_qm", v+48, exp, M*N, M*N);
  printf("  time: %.2f us\n", t_ns/1000.0);

out:
  cvk->ops->lmem_free_matrix(cvk, ml_bias);
  cvk->ops->lmem_free_matrix(cvk, ml_res);
  cvk->ops->lmem_free_matrix(cvk, ml_r);
  cvk->ops->lmem_free_matrix(cvk, ml_l);
  tpu_cleanup(&ctx);
  return errs?1:0;
}
