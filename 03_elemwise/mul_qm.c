/* TPU mul_qm: MUL with quantization.
   a=[1..8], b_const=1. multiplier=(1<<rshift)=4, rshift=2.
   Result: (a * 1 * 4) >> 2 = a */
#include "../common/tpu_bench.h"
#define N 8

static const int8_t a_lo[N] = {1,2,3,4,5,6,7,8};
static const int8_t exp_lo[N]={1,2,3,4,5,6,7,8};
static const int8_t exp_hi[N]={0,0,0,0,0,0,0,0};

int main() {
  tpu_ctx ctx;
  if (tpu_init(&ctx, 4096) != 0) return 1;
  cvk_context_t *cvk = ctx.cvk_ctx;

  memcpy(ctx.neuron_vaddr, a_lo, N);
  CVI_RT_MemFlush(ctx.rt_handle, ctx.neuron_mem);

  cvk_tl_shape_t s8 = {1,1,1,N};
  cvk_tl_t *tl_a  = cvk->ops->lmem_alloc_tensor(cvk, s8, CVK_FMT_I8, 1);
  cvk_tl_t *tl_rl = cvk->ops->lmem_alloc_tensor(cvk, s8, CVK_FMT_I8, 1);
  cvk_tl_t *tl_rh = cvk->ops->lmem_alloc_tensor(cvk, s8, CVK_FMT_I8, 1);
  if (!tl_a||!tl_rl||!tl_rh) { fprintf(stderr,"alloc fail\n"); tpu_cleanup(&ctx); return 1; }

  cvk_tg_shape_t gs = {1,1,1,N};
  cvk_tg_stride_t st = cvk->ops->tg_default_stride(cvk, gs, CVK_FMT_I8);
  cvk_tg_t tg_a  = {0, TPU_PA(&ctx,0),  CVK_FMT_I8, gs, st};
  cvk_tg_t tg_rl = {0, TPU_PA(&ctx,32), CVK_FMT_I8, gs, st};
  cvk_tg_t tg_rh = {0, TPU_PA(&ctx,48), CVK_FMT_I8, gs, st};

  cvk->ops->tdma_g2l_tensor_copy(cvk, &(cvk_tdma_g2l_tensor_copy_param_t){&tg_a, tl_a});

  /* res = (a * b_const * multiplier) >> rshift.
     b_const=1, multiplier=1<<2=4, rshift=2 => (a*1*4)>>2 = a */
  cvk_tiu_mul_qm_param_t p = {
    .res_high = tl_rh, .res_low = tl_rl,
    .a = tl_a,
    .b_is_const = 1,
    .b_const = { .val = 1, .is_signed = 1 },
    .rshift_bits = 2,
    .relu_enable = 0,
    .multiplier = 4,
  };
  cvk->ops->tiu_mul_qm(cvk, &p);

  cvk->ops->tdma_l2g_tensor_copy(cvk, &(cvk_tdma_l2g_tensor_copy_param_t){tl_rl, &tg_rl});
  cvk->ops->tdma_l2g_tensor_copy(cvk, &(cvk_tdma_l2g_tensor_copy_param_t){tl_rh, &tg_rh});

  int64_t t_ns = tpu_submit(&ctx);
  if (t_ns < 0) { fprintf(stderr,"submit fail\n"); goto out; }

  CVI_RT_MemInvld(ctx.rt_handle, ctx.neuron_mem);
  int8_t *v = (int8_t*)ctx.neuron_vaddr;
  int errs = tpu_check_i8("mul_qm_lo", v+32, exp_lo, N, N);
  errs +=   tpu_check_i8("mul_qm_hi", v+48, exp_hi, N, N);
  printf("  time: %.2f us\n", t_ns/1000.0);

out:
  cvk->ops->lmem_free_tensor(cvk, tl_rh);
  cvk->ops->lmem_free_tensor(cvk, tl_rl);
  cvk->ops->lmem_free_tensor(cvk, tl_a);
  tpu_cleanup(&ctx);
  return errs?1:0;
}
