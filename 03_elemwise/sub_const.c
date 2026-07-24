/* TPU sub: element-wise SUB of two INT16 tensors (a - b).
   a_lo = [10..25], b_lo = [3,3,...]. Result: [7..22] */
#include "../common/tpu_bench.h"
#define N 16
static const int8_t a_lo[N] = {10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25};
static const int8_t b_lo[N] = {3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3};
static const int8_t exp_lo[N]={7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22};

int main() {
  tpu_ctx ctx;
  if (tpu_init(&ctx, 4096) != 0) return 1;
  cvk_context_t *cvk = ctx.cvk_ctx;

  memcpy(ctx.neuron_vaddr, a_lo, N);
  memset(ctx.neuron_vaddr + 16, 0, N);  /* a_hi = 0 */
  memcpy(ctx.neuron_vaddr + 32, b_lo, N);
  memset(ctx.neuron_vaddr + 48, 0, N);  /* b_hi = 0 */
  CVI_RT_MemFlush(ctx.rt_handle, ctx.neuron_mem);

  cvk_tl_shape_t s8 = {1,1,1,N};
  cvk_tl_t *tl_al = cvk->ops->lmem_alloc_tensor(cvk, s8, CVK_FMT_I8, 1);
  cvk_tl_t *tl_ah = cvk->ops->lmem_alloc_tensor(cvk, s8, CVK_FMT_I8, 1);
  cvk_tl_t *tl_bl = cvk->ops->lmem_alloc_tensor(cvk, s8, CVK_FMT_I8, 1);
  cvk_tl_t *tl_bh = cvk->ops->lmem_alloc_tensor(cvk, s8, CVK_FMT_I8, 1);
  cvk_tl_t *tl_rl = cvk->ops->lmem_alloc_tensor(cvk, s8, CVK_FMT_I8, 1);
  cvk_tl_t *tl_rh = cvk->ops->lmem_alloc_tensor(cvk, s8, CVK_FMT_I8, 1);
  if (!tl_al||!tl_ah||!tl_bl||!tl_bh||!tl_rl||!tl_rh) {
    fprintf(stderr,"alloc fail\n"); tpu_cleanup(&ctx); return 1;
  }

  cvk_tg_shape_t gs = {1,1,1,N};
  cvk_tg_stride_t st = cvk->ops->tg_default_stride(cvk, gs, CVK_FMT_I8);
  cvk_tg_t tg_al = {0, TPU_PA(&ctx,0),  CVK_FMT_I8, gs, st};
  cvk_tg_t tg_ah = {0, TPU_PA(&ctx,16), CVK_FMT_I8, gs, st};
  cvk_tg_t tg_bl = {0, TPU_PA(&ctx,32), CVK_FMT_I8, gs, st};
  cvk_tg_t tg_bh = {0, TPU_PA(&ctx,48), CVK_FMT_I8, gs, st};
  cvk_tg_t tg_rl = {0, TPU_PA(&ctx,64), CVK_FMT_I8, gs, st};
  cvk_tg_t tg_rh = {0, TPU_PA(&ctx,80), CVK_FMT_I8, gs, st};

  /* G2L: load all inputs */
  cvk->ops->tdma_g2l_tensor_copy(cvk, &(cvk_tdma_g2l_tensor_copy_param_t){&tg_al, tl_al});
  cvk->ops->tdma_g2l_tensor_copy(cvk, &(cvk_tdma_g2l_tensor_copy_param_t){&tg_ah, tl_ah});
  cvk->ops->tdma_g2l_tensor_copy(cvk, &(cvk_tdma_g2l_tensor_copy_param_t){&tg_bl, tl_bl});
  cvk->ops->tdma_g2l_tensor_copy(cvk, &(cvk_tdma_g2l_tensor_copy_param_t){&tg_bh, tl_bh});

  /* TIU SUB: res = a - b */
  cvk_tiu_sub_param_t p = {
    .res_high = tl_rh, .res_low = tl_rl,
    .a_high = tl_ah, .a_low = tl_al,
    .b_high = tl_bh, .b_low = tl_bl,
    .rshift_bits = 0,
  };
  cvk->ops->tiu_sub(cvk, &p);

  /* L2G */
  cvk->ops->tdma_l2g_tensor_copy(cvk, &(cvk_tdma_l2g_tensor_copy_param_t){tl_rl, &tg_rl});
  cvk->ops->tdma_l2g_tensor_copy(cvk, &(cvk_tdma_l2g_tensor_copy_param_t){tl_rh, &tg_rh});

  int64_t t_ns = tpu_submit(&ctx);
  if (t_ns < 0) { fprintf(stderr,"submit fail\n"); return 1; }

  CVI_RT_MemInvld(ctx.rt_handle, ctx.neuron_mem);
  int8_t *v = (int8_t*)ctx.neuron_vaddr;
  int errs = tpu_check_i8("sub_lo", v+64, exp_lo, N, N);
  printf("  time: %.2f us\n", t_ns/1000.0);

  cvk->ops->lmem_free_tensor(cvk, tl_rh);
  cvk->ops->lmem_free_tensor(cvk, tl_rl);
  cvk->ops->lmem_free_tensor(cvk, tl_bh);
  cvk->ops->lmem_free_tensor(cvk, tl_bl);
  cvk->ops->lmem_free_tensor(cvk, tl_ah);
  cvk->ops->lmem_free_tensor(cvk, tl_al);
  tpu_cleanup(&ctx);
  return errs?1:0;
}
