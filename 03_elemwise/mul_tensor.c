/* TPU mul_tensor: element-wise multiply two INT8 tensors */
#include "../common/tpu_bench.h"
#define N 16
static const int8_t a[N] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
static const int8_t b[N] = {2,2,2,2,2,2,2,2,2, 2, 2, 2, 2, 2, 2, 2};
static const int8_t exp[N]={2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32};

int main() {
  tpu_ctx ctx;
  if (tpu_init(&ctx, 4096) != 0) return 1;
  cvk_context_t *cvk = ctx.cvk_ctx;

  memcpy(ctx.neuron_vaddr, a, N);
  memcpy(ctx.neuron_vaddr + 16, b, N);
  CVI_RT_MemFlush(ctx.rt_handle, ctx.neuron_mem);

  cvk_tl_shape_t s8 = {1,1,1,N};
  cvk_tl_t *tl_a = cvk->ops->lmem_alloc_tensor(cvk, s8, CVK_FMT_I8, 1);
  cvk_tl_t *tl_b = cvk->ops->lmem_alloc_tensor(cvk, s8, CVK_FMT_I8, 1);
  cvk_tl_t *tl_r = cvk->ops->lmem_alloc_tensor(cvk, s8, CVK_FMT_I8, 1);
  if (!tl_a||!tl_b||!tl_r) { fprintf(stderr,"alloc fail\n"); tpu_cleanup(&ctx); return 1; }

  cvk_tg_shape_t gs = {1,1,1,N};
  cvk_tg_stride_t st = cvk->ops->tg_default_stride(cvk, gs, CVK_FMT_I8);
  cvk_tg_t tg_a = {0, TPU_PA(&ctx,0),   CVK_FMT_I8, gs, st};
  cvk_tg_t tg_b = {0, TPU_PA(&ctx,16),  CVK_FMT_I8, gs, st};
  cvk_tg_t tg_d = {0, TPU_PA(&ctx,32),  CVK_FMT_I8, gs, st};

  cvk_tdma_g2l_tensor_copy_param_t g2l_a = {&tg_a, tl_a};
  cvk_tdma_g2l_tensor_copy_param_t g2l_b = {&tg_b, tl_b};
  cvk->ops->tdma_g2l_tensor_copy(cvk, &g2l_a);
  cvk->ops->tdma_g2l_tensor_copy(cvk, &g2l_b);

  cvk_tiu_mul_param_t mul = {
    .res_high = NULL, .res_low = tl_r, .a = tl_a,
    .b_is_const = 0, .b = tl_b,
    .rshift_bits = 0, .relu_enable = 0,
  };
  cvk->ops->tiu_mul(cvk, &mul);

  cvk_tdma_l2g_tensor_copy_param_t l2g = {tl_r, &tg_d};
  cvk->ops->tdma_l2g_tensor_copy(cvk, &l2g);

  int64_t t_ns = tpu_submit(&ctx);
  if (t_ns < 0) { fprintf(stderr,"submit fail\n"); goto out; }

  CVI_RT_MemInvld(ctx.rt_handle, ctx.neuron_mem);
  int errs = tpu_check_i8("mul_tensor", (int8_t*)ctx.neuron_vaddr+32, exp, N, N);
  printf("  time: %.2f us\n", t_ns/1000.0);

out:
  cvk->ops->lmem_free_tensor(cvk, tl_r);
  cvk->ops->lmem_free_tensor(cvk, tl_b);
  cvk->ops->lmem_free_tensor(cvk, tl_a);
  tpu_cleanup(&ctx);
  return errs?1:0;
}
