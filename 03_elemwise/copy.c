/* TPU copy: TIU tensor copy (src → dst) */
#include "../common/tpu_bench.h"
#define N 16
static const int8_t in[N]  = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};

int main() {
  tpu_ctx ctx;
  if (tpu_init(&ctx, 4096) != 0) return 1;
  cvk_context_t *cvk = ctx.cvk_ctx;

  memcpy(ctx.neuron_vaddr, in, N);
  CVI_RT_MemFlush(ctx.rt_handle, ctx.neuron_mem);

  cvk_tl_shape_t s8 = {1,1,1,N};
  cvk_tl_t *tl_s = cvk->ops->lmem_alloc_tensor(cvk, s8, CVK_FMT_I8, 1);
  cvk_tl_t *tl_d = cvk->ops->lmem_alloc_tensor(cvk, s8, CVK_FMT_I8, 1);
  if (!tl_s||!tl_d) { fprintf(stderr,"alloc fail\n"); tpu_cleanup(&ctx); return 1; }

  cvk_tg_shape_t gs = {1,1,1,N};
  cvk_tg_stride_t st = cvk->ops->tg_default_stride(cvk, gs, CVK_FMT_I8);
  cvk_tg_t tg_a = {0, TPU_PA(&ctx,0),  CVK_FMT_I8, gs, st};
  cvk_tg_t tg_b = {0, TPU_PA(&ctx,16), CVK_FMT_I8, gs, st};

  cvk_tdma_g2l_tensor_copy_param_t g2l = {&tg_a, tl_s};
  cvk->ops->tdma_g2l_tensor_copy(cvk, &g2l);

  cvk_tiu_copy_param_t cp = { .src = tl_s, .dst = tl_d };
  cvk->ops->tiu_copy(cvk, &cp);

  cvk_tdma_l2g_tensor_copy_param_t l2g = {tl_d, &tg_b};
  cvk->ops->tdma_l2g_tensor_copy(cvk, &l2g);

  int64_t t_ns = tpu_submit(&ctx);
  if (t_ns < 0) { fprintf(stderr,"submit fail\n"); goto out; }

  CVI_RT_MemInvld(ctx.rt_handle, ctx.neuron_mem);
  int errs = tpu_check_i8("copy", (int8_t*)ctx.neuron_vaddr+16, in, N, N);
  printf("  time: %.2f us\n", t_ns/1000.0);

out:
  cvk->ops->lmem_free_tensor(cvk, tl_d);
  cvk->ops->lmem_free_tensor(cvk, tl_s);
  tpu_cleanup(&ctx);
  return errs?1:0;
}
