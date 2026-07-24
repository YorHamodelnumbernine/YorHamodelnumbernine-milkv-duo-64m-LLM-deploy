/* TPU avg_pooling: 2x2 average pool on 4x4 INT8 input, stride 2, valid padding */
#include "../common/tpu_bench.h"
#define H 4
#define W 4
static const int8_t in[H*W] = {
  1, 3, 2, 4,
  5, 7, 6, 8,
  9,11,10,12,
  13,15,14,16,
};
/* 2x2 avg pool: (1+3+5+7)/4=4, (2+4+6+8)/4=5, (9+11+13+15)/4=12, (10+12+14+16)/4=13 */
static const int8_t exp[4] = {4, 5, 12, 13};

int main() {
  tpu_ctx ctx;
  if (tpu_init(&ctx, 4096) != 0) return 1;
  cvk_context_t *cvk = ctx.cvk_ctx;

  memcpy(ctx.neuron_vaddr, in, H*W);
  CVI_RT_MemFlush(ctx.rt_handle, ctx.neuron_mem);

  cvk_tl_shape_t s_if = {1,1,H,W};
  cvk_tl_shape_t s_of = {1,1,2,2};
  cvk_tl_t *tl_if = cvk->ops->lmem_alloc_tensor(cvk, s_if, CVK_FMT_I8, 1);
  cvk_tl_t *tl_of = cvk->ops->lmem_alloc_tensor(cvk, s_of, CVK_FMT_I8, 1);
  if (!tl_if||!tl_of) { fprintf(stderr,"alloc fail\n"); tpu_cleanup(&ctx); return 1; }

  cvk_tg_shape_t gs_if = {1,1,H,W};
  cvk_tg_shape_t gs_of = {1,1,2,2};
  cvk_tg_stride_t st_if = cvk->ops->tg_default_stride(cvk, gs_if, CVK_FMT_I8);
  cvk_tg_stride_t st_of = cvk->ops->tg_default_stride(cvk, gs_of, CVK_FMT_I8);

  cvk_tg_t tg_if = {0, TPU_PA(&ctx,0),  CVK_FMT_I8, gs_if, st_if};
  cvk_tg_t tg_of = {0, TPU_PA(&ctx,16), CVK_FMT_I8, gs_of, st_of};

  cvk_tdma_g2l_tensor_copy_param_t g2l = {&tg_if, tl_if};
  cvk->ops->tdma_g2l_tensor_copy(cvk, &g2l);

  /* avg_pooling_const = (1 << rshift_bits) / (kh * kw) in fixed point.
     For 2x2 pool with rshift=0, avg_pooling_const = 1/4 in fixed point.
     But rshift bits need to convert back. Using rshift=2 for >>2 = /4.
     With rshift_bits=2, avg_pooling_const should be 1 << rshift_bits = 4.
     Formula: result = sum * avg_pooling_const >> rshift_bits = sum * 4 >> 2 = sum. */
  cvk_tiu_average_pooling_param_t p = {
    .ofmap = tl_of, .ifmap = tl_if,
    .kh = 2, .kw = 2,
    .pad_top = 0, .pad_bottom = 0,
    .pad_left = 0, .pad_right = 0,
    .stride_h = 2, .stride_w = 2,
    .avg_pooling_const = 1,  /* multiply by 1 */
    .rshift_bits = 2,        /* then shift right by 2 (divide by 4) */
  };
  cvk->ops->tiu_average_pooling(cvk, &p);

  cvk_tdma_l2g_tensor_copy_param_t l2g = {tl_of, &tg_of};
  cvk->ops->tdma_l2g_tensor_copy(cvk, &l2g);

  int64_t t_ns = tpu_submit(&ctx);
  if (t_ns < 0) { fprintf(stderr,"submit fail\n"); goto out; }

  CVI_RT_MemInvld(ctx.rt_handle, ctx.neuron_mem);
  int8_t *v = (int8_t*)ctx.neuron_vaddr;
  fprintf(stderr,"[dbg] avgpool:"); for(int i=0;i<4;i++) fprintf(stderr,"%d,",v[16+i]);
  fprintf(stderr,"\n");
  int errs = tpu_check_i8("avgpool", v+16, exp, 4, 4);
  printf("  time: %.2f us\n", t_ns/1000.0);

out:
  cvk->ops->lmem_free_tensor(cvk, tl_of);
  cvk->ops->lmem_free_tensor(cvk, tl_if);
  tpu_cleanup(&ctx);
  return errs?1:0;
}
