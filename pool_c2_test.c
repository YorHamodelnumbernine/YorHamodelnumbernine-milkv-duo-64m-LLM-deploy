/* Test pooling with IC=2 to isolate channel count issue */
#include "../common/tpu_bench.h"

#define IC 2
#define IH 4
#define IW 4
#define OH 2
#define OW 2

/* ifmap ch0: 4x4 [1..16], ch1: 4x4 [17..32] */
static const int8_t ifmap[IC*IH*IW] = {
  1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16,
  17,18,19,20, 21,22,23,24, 25,26,27,28, 29,30,31,32,
};
/* 2x2 max pool, stride 2: ch0=[6,8,14,16], ch1=[22,24,30,32] */
static const int8_t expected[IC*OH*OW] = {
  6,8, 14,16,
  22,24, 30,32,
};

int main() {
  tpu_ctx ctx;
  if (tpu_init(&ctx, 4096) != 0) return 1;
  cvk_context_t *cvk = ctx.cvk_ctx;

  memcpy(ctx.neuron_vaddr, ifmap, sizeof(ifmap));
  CVI_RT_MemFlush(ctx.rt_handle, ctx.neuron_mem);

  cvk_tl_t *tl_if = cvk->ops->lmem_alloc_tensor(cvk, (cvk_tl_shape_t){1,IC,IH,IW}, CVK_FMT_I8, 1);
  cvk_tl_t *tl_of = cvk->ops->lmem_alloc_tensor(cvk, (cvk_tl_shape_t){1,IC,OH,OW}, CVK_FMT_I8, 1);
  if (!tl_if||!tl_of) {
    fprintf(stderr,"alloc fail\n"); tpu_cleanup(&ctx); return 1;
  }

  fprintf(stderr,"[diag] ifmap: stride=(%d,%d,%d,%d)\n", tl_if->stride.n,tl_if->stride.c,tl_if->stride.h,tl_if->stride.w);
  fprintf(stderr,"[diag] ofmap: stride=(%d,%d,%d,%d)\n", tl_of->stride.n,tl_of->stride.c,tl_of->stride.h,tl_of->stride.w);

  cvk_tg_t tg_if = {0, TPU_PA(&ctx,0),  CVK_FMT_I8, {1,IC,IH,IW},
    cvk->ops->tg_default_stride(cvk, (cvk_tg_shape_t){1,IC,IH,IW}, CVK_FMT_I8)};
  cvk_tg_t tg_of = {0, TPU_PA(&ctx,0x100), CVK_FMT_I8, {1,IC,OH,OW},
    cvk->ops->tg_default_stride(cvk, (cvk_tg_shape_t){1,IC,OH,OW}, CVK_FMT_I8)};

  cvk->ops->tdma_g2l_tensor_copy(cvk, &(cvk_tdma_g2l_tensor_copy_param_t){&tg_if, tl_if});

  cvk_tiu_max_pooling_param_t p = {
    .ofmap = tl_of, .ifmap = tl_if,
    .kh = 2, .kw = 2,
    .pad_top = 0, .pad_bottom = 0, .pad_left = 0, .pad_right = 0,
    .stride_h = 2, .stride_w = 2,
  };
  cvk->ops->tiu_max_pooling(cvk, &p);

  cvk->ops->tdma_l2g_tensor_copy(cvk, &(cvk_tdma_l2g_tensor_copy_param_t){tl_of, &tg_of});

  int64_t t_ns = tpu_submit(&ctx);
  if (t_ns < 0) { fprintf(stderr,"submit fail\n"); goto out; }

  CVI_RT_MemInvld(ctx.rt_handle, ctx.neuron_mem);
  int8_t *v = (int8_t*)ctx.neuron_vaddr;
  int el = IC*OH*OW;
  fprintf(stderr,"[dbg] out:"); for(int i=0;i<el;i++) fprintf(stderr,"%d,",v[0x100+i]);
  fprintf(stderr,"\n");
  int errs = tpu_check_i8("pool_c2", v+0x100, expected, el, el);
  printf("  time: %.2f us\n", t_ns/1000.0);

out:
  cvk->ops->lmem_free_tensor(cvk, tl_of);
  cvk->ops->lmem_free_tensor(cvk, tl_if);
  tpu_cleanup(&ctx);
  return errs?1:0;
}
