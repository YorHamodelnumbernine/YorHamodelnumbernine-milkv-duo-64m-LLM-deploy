/* TPU arith_shift: arithmetic right shift INT16 value by constant scalar amount.
   Each element shifted right by the same number of bits (scalar broadcast). */
#include "../common/tpu_bench.h"
#define N 8

static const int8_t a_lo[N]   = {0,0,-128,-128,64,-64,32,-32};
static const int8_t a_hi[N]   = {1,-1,0,-1,0,-1,0,-1};
/* Shift all elements right by 2 */
/* bits range [-16,16]. Positive=left shift, negative=right shift */
static const int8_t shift[N]  = {-2,-2,-2,-2,-2,-2,-2,-2};
/* Expected: INT16 [256,-256,128,-128,64,-64,32,-32] >> 2 */
static const int8_t exp_lo[N] = {64,-64,32,-32,16,-16,8,-8};
static const int8_t exp_hi[N] = {0,-1,0,-1,0,-1,0,-1};

int main() {
  tpu_ctx ctx;
  if (tpu_init(&ctx, 4096) != 0) return 1;
  cvk_context_t *cvk = ctx.cvk_ctx;

  memcpy(ctx.neuron_vaddr,      a_lo,  N);
  memcpy(ctx.neuron_vaddr + 16, a_hi,  N);
  memcpy(ctx.neuron_vaddr + 32, shift, N);
  CVI_RT_MemFlush(ctx.rt_handle, ctx.neuron_mem);

  cvk_tl_shape_t s8 = {1,1,1,N};
  cvk_tl_t *tl_al = cvk->ops->lmem_alloc_tensor(cvk, s8, CVK_FMT_I8, 1);
  cvk_tl_t *tl_ah = cvk->ops->lmem_alloc_tensor(cvk, s8, CVK_FMT_I8, 1);
  cvk_tl_t *tl_rl = cvk->ops->lmem_alloc_tensor(cvk, s8, CVK_FMT_I8, 1);
  cvk_tl_t *tl_rh = cvk->ops->lmem_alloc_tensor(cvk, s8, CVK_FMT_I8, 1);
  cvk_tl_t *tl_bi = cvk->ops->lmem_alloc_tensor(cvk, s8, CVK_FMT_I8, 1);
  if (!tl_al||!tl_ah||!tl_rl||!tl_rh||!tl_bi) {
    fprintf(stderr,"alloc fail\n"); tpu_cleanup(&ctx); return 1;
  }

  cvk_tg_shape_t gs = {1,1,1,N};
  cvk_tg_stride_t st = cvk->ops->tg_default_stride(cvk, gs, CVK_FMT_I8);
  cvk_tg_t tg_al = {0, TPU_PA(&ctx,0),  CVK_FMT_I8, gs, st};
  cvk_tg_t tg_ah = {0, TPU_PA(&ctx,16), CVK_FMT_I8, gs, st};
  cvk_tg_t tg_bi = {0, TPU_PA(&ctx,32), CVK_FMT_I8, gs, st};
  cvk_tg_t tg_rl = {0, TPU_PA(&ctx,48), CVK_FMT_I8, gs, st};
  cvk_tg_t tg_rh = {0, TPU_PA(&ctx,64), CVK_FMT_I8, gs, st};

  /* G2L */
  cvk_tdma_g2l_tensor_copy_param_t g2l_al = {&tg_al, tl_al};
  cvk_tdma_g2l_tensor_copy_param_t g2l_ah = {&tg_ah, tl_ah};
  cvk_tdma_g2l_tensor_copy_param_t g2l_bi = {&tg_bi, tl_bi};
  cvk->ops->tdma_g2l_tensor_copy(cvk, &g2l_al);
  cvk->ops->tdma_g2l_tensor_copy(cvk, &g2l_ah);
  cvk->ops->tdma_g2l_tensor_copy(cvk, &g2l_bi);

  /* TIU arith_shift: res = a >> bits (all elements shifted by 2) */
  cvk_tiu_arith_shift_param_t p = {
    .res_high = tl_rh, .res_low = tl_rl,
    .a_high   = tl_ah, .a_low   = tl_al,
    .bits     = tl_bi,
  };
  cvk->ops->tiu_arith_shift(cvk, &p);

  /* L2G */
  cvk_tdma_l2g_tensor_copy_param_t l2g_l = {tl_rl, &tg_rl};
  cvk_tdma_l2g_tensor_copy_param_t l2g_h = {tl_rh, &tg_rh};
  cvk->ops->tdma_l2g_tensor_copy(cvk, &l2g_l);
  cvk->ops->tdma_l2g_tensor_copy(cvk, &l2g_h);

  int64_t t_ns = tpu_submit(&ctx);
  if (t_ns < 0) { fprintf(stderr,"submit fail\n"); return 1; }

  CVI_RT_MemInvld(ctx.rt_handle, ctx.neuron_mem);
  int8_t *v = (int8_t*)ctx.neuron_vaddr;
  fprintf(stderr, "[dbg] got lo:"); for(int i=0;i<N;i++) fprintf(stderr,"%d,",v[48+i]);
  fprintf(stderr,"\n[dbg] got hi:"); for(int i=0;i<N;i++) fprintf(stderr,"%d,",v[64+i]);
  fprintf(stderr,"\n");

  int errs = tpu_check_i8("sh_lo", v+48, exp_lo, N, N);
  errs +=   tpu_check_i8("sh_hi", v+64, exp_hi, N, N);
  printf("  time: %.2f us\n", t_ns/1000.0);

  cvk->ops->lmem_free_tensor(cvk, tl_bi);
  cvk->ops->lmem_free_tensor(cvk, tl_rh);
  cvk->ops->lmem_free_tensor(cvk, tl_rl);
  cvk->ops->lmem_free_tensor(cvk, tl_ah);
  cvk->ops->lmem_free_tensor(cvk, tl_al);
  tpu_cleanup(&ctx);
  return errs?1:0;
}
