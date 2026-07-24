/* TPU add_const: ADD each INT8 element with constant 3 using TIU ADD (16-bit).
   INT8 data is split into high/low bytes for the 16-bit ADD operation. */
#include "../common/tpu_bench.h"
#define N 16
static const int8_t in[N]  = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
/* a_lo = lower byte, a_hi = 0 for positive. Result = a + 3 */
static const int8_t exp_lo[N] = {4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19};

int main() {
  tpu_ctx ctx;
  if (tpu_init(&ctx, 4096) != 0) return 1;
  cvk_context_t *cvk = ctx.cvk_ctx;

  /* Write a_lo and a_hi to global memory (separate arrays) */
  memcpy(ctx.neuron_vaddr, in, N);           /* a_lo at offset 0 */
  memset(ctx.neuron_vaddr + 16, 0, N);       /* a_hi at offset 16 (all zeros for positive values) */
  CVI_RT_MemFlush(ctx.rt_handle, ctx.neuron_mem);

  cvk_tl_shape_t s8 = {1,1,1,N};
  cvk_tl_t *tl_al = cvk->ops->lmem_alloc_tensor(cvk, s8, CVK_FMT_I8, 1);
  cvk_tl_t *tl_ah = cvk->ops->lmem_alloc_tensor(cvk, s8, CVK_FMT_I8, 1);
  cvk_tl_t *tl_rl = cvk->ops->lmem_alloc_tensor(cvk, s8, CVK_FMT_I8, 1);
  cvk_tl_t *tl_rh = cvk->ops->lmem_alloc_tensor(cvk, s8, CVK_FMT_I8, 1);
  if (!tl_al||!tl_ah||!tl_rl||!tl_rh) { fprintf(stderr,"alloc fail\n"); tpu_cleanup(&ctx); return 1; }

  cvk_tg_shape_t gs = {1,1,1,N};
  cvk_tg_stride_t st = cvk->ops->tg_default_stride(cvk, gs, CVK_FMT_I8);
  cvk_tg_t tg_al = {0, TPU_PA(&ctx,0),  CVK_FMT_I8, gs, st};
  cvk_tg_t tg_ah = {0, TPU_PA(&ctx,16), CVK_FMT_I8, gs, st};
  cvk_tg_t tg_rl = {0, TPU_PA(&ctx,32), CVK_FMT_I8, gs, st};
  cvk_tg_t tg_rh = {0, TPU_PA(&ctx,48), CVK_FMT_I8, gs, st};

  /* G2L */
  cvk_tdma_g2l_tensor_copy_param_t g2l_al = {&tg_al, tl_al};
  cvk_tdma_g2l_tensor_copy_param_t g2l_ah = {&tg_ah, tl_ah};
  cvk->ops->tdma_g2l_tensor_copy(cvk, &g2l_al);
  cvk->ops->tdma_g2l_tensor_copy(cvk, &g2l_ah);

  /* TIU ADD: res = a + 3. a_high/a_low are INT16 halves, b_const is INT16 */
  cvk_tiu_add_param_t p = {
    .res_high = tl_rh, .res_low = tl_rl,
    .a_high   = tl_ah, .a_low   = tl_al,
    .b_is_const = 1,
    .b_const = { .val = 3, .is_signed = 1 },
    .rshift_bits = 0,
    .relu_enable = 0,
  };
  cvk->ops->tiu_add(cvk, &p);

  /* L2G */
  cvk_tdma_l2g_tensor_copy_param_t l2g_l = {tl_rl, &tg_rl};
  cvk_tdma_l2g_tensor_copy_param_t l2g_h = {tl_rh, &tg_rh};
  cvk->ops->tdma_l2g_tensor_copy(cvk, &l2g_l);
  cvk->ops->tdma_l2g_tensor_copy(cvk, &l2g_h);

  int64_t t_ns = tpu_submit(&ctx);
  if (t_ns < 0) { fprintf(stderr,"submit fail\n"); return 1; }

  CVI_RT_MemInvld(ctx.rt_handle, ctx.neuron_mem);
  int8_t *v = (int8_t*)ctx.neuron_vaddr;
  int errs = tpu_check_i8("add_lo", v+32, exp_lo, N, N);
  /* res_hi should be all 0 for small positive results */
  static const int8_t zero16[N];
  errs += tpu_check_i8("add_hi", v+48, zero16, N, N);
  printf("  time: %.2f us\n", t_ns/1000.0);

  cvk->ops->lmem_free_tensor(cvk, tl_rh);
  cvk->ops->lmem_free_tensor(cvk, tl_rl);
  cvk->ops->lmem_free_tensor(cvk, tl_ah);
  cvk->ops->lmem_free_tensor(cvk, tl_al);
  tpu_cleanup(&ctx);
  return errs?1:0;
}
