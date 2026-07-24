/* TPU GE: greater-than-or-equal compared to constant (INT8) */
#include "../common/tpu_bench.h"
#define N 16
static const int8_t in[N]  = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
/* GE(a, 8): where a >= 8, result = 1 (true), else 0 */
static const int8_t exp[N] = {0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1};

int main() {
  tpu_ctx ctx;
  if (tpu_init(&ctx, 4096) != 0) return 1;
  cvk_context_t *cvk = ctx.cvk_ctx;

  memcpy(ctx.neuron_vaddr, in, N);
  CVI_RT_MemFlush(ctx.rt_handle, ctx.neuron_mem);

  cvk_tl_shape_t s8 = {1,1,1,N};
  cvk_tl_t *tl_a = cvk->ops->lmem_alloc_tensor(cvk, s8, CVK_FMT_I8, 1);
  cvk_tl_t *tl_r = cvk->ops->lmem_alloc_tensor(cvk, s8, CVK_FMT_I8, 1);
  if (!tl_a||!tl_r) { fprintf(stderr,"alloc fail\n"); tpu_cleanup(&ctx); return 1; }

  cvk_tg_shape_t gs = {1,1,1,N};
  cvk_tg_stride_t st = cvk->ops->tg_default_stride(cvk, gs, CVK_FMT_I8);
  cvk_tg_t tg_a = {0, TPU_PA(&ctx,0),  CVK_FMT_I8, gs, st};
  cvk_tg_t tg_d = {0, TPU_PA(&ctx,16), CVK_FMT_I8, gs, st};

  cvk_tdma_g2l_tensor_copy_param_t g2l = {&tg_a, tl_a};
  cvk->ops->tdma_g2l_tensor_copy(cvk, &g2l);

  cvk_tiu_ge_param_t p = {
    .ge = tl_r, .a = tl_a,
    .b_is_const = 1, .b_const = { .val = 8, .is_signed = 1 },
  };
  cvk->ops->tiu_ge(cvk, &p);

  cvk_tdma_l2g_tensor_copy_param_t l2g = {tl_r, &tg_d};
  cvk->ops->tdma_l2g_tensor_copy(cvk, &l2g);

  int64_t t_ns = tpu_submit(&ctx);
  if (t_ns < 0) { fprintf(stderr,"submit fail\n"); goto out; }

  CVI_RT_MemInvld(ctx.rt_handle, ctx.neuron_mem);
  int errs = tpu_check_i8("ge8", (int8_t*)ctx.neuron_vaddr+16, exp, N, N);
  printf("  time: %.2f us\n", t_ns/1000.0);

out:
  cvk->ops->lmem_free_tensor(cvk, tl_r);
  cvk->ops->lmem_free_tensor(cvk, tl_a);
  tpu_cleanup(&ctx);
  return errs?1:0;
}
