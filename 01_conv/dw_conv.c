/* Test depthwise conv with kh=kw=3, larger spatial dims */
#include "../common/tpu_bench.h"

#define IC 2
#define OC 2   /* depthwise: IC == OC */
#define KH 3
#define KW 3
#define IH 4
#define IW 4
#define OH 2
#define OW 2

static const int8_t ifmap[IC*IH*IW] = {
  /* ch0: 4x4 */
  1,2,3,4,
  5,6,7,8,
  9,10,11,12,
  13,14,15,16,
  /* ch1: 4x4 */
  17,18,19,20,
  21,22,23,24,
  25,26,27,28,
  29,30,31,32,
};
/* Depthwise weight: (1, OC, KH*KW, 1) = (1, 2, 9, 1)
   Identity: kernel[0] = 1 at center, 0 elsewhere; kernel[1] = 1 at center, 0 elsewhere
   For 3x3 kernel center is position 4 (0-indexed in row-major)
   But identity in depthwise: weight = [1] for each channel (1x1 equivalent in kh*kw dim) */
static const int8_t weight[OC*KH*KW*1] = {
  /* ch0 3x3 filter: center=1, rest=0 */
  0,0,0, 0,1,0, 0,0,0,
  /* ch1 3x3 filter */
  0,0,0, 0,1,0, 0,0,0,
};
/* Expected: for 3x3 identity filter with stride=1, pad=0: OH=IH-3+1=2, OW=IW-3+1=2
   Center 2x2 of each 4x4 input: ch0[6,7,10,11], ch1[22,23,26,27] */
static const int8_t expected[OC*OH*OW] = {
  6,7, 10,11,
  22,23, 26,27,
};

int main() {
  tpu_ctx ctx;
  if (tpu_init(&ctx, 4096) != 0) return 1;
  cvk_context_t *cvk = ctx.cvk_ctx;

  memcpy(ctx.neuron_vaddr, ifmap, sizeof(ifmap));
  memcpy(ctx.neuron_vaddr+64, weight, sizeof(weight));
  CVI_RT_MemFlush(ctx.rt_handle, ctx.neuron_mem);

  cvk_tl_t *tl_if = cvk->ops->lmem_alloc_tensor(cvk, (cvk_tl_shape_t){1,IC,IH,IW}, CVK_FMT_I8, 1);
  cvk_tl_t *tl_of = cvk->ops->lmem_alloc_tensor(cvk, (cvk_tl_shape_t){1,OC,OH,OW}, CVK_FMT_I8, 1);
  cvk_tl_t *tl_w  = cvk->ops->lmem_alloc_tensor(cvk, (cvk_tl_shape_t){1,OC,KH,KW}, CVK_FMT_I8, 1);
  tl_w->stride.n = 1;
  tl_w->cmprs_fmt = CVK_FMT_I8;
  if (!tl_if||!tl_of||!tl_w) {
    fprintf(stderr,"alloc fail\n"); tpu_cleanup(&ctx); return 1;
  }

  fprintf(stderr,"[diag] if shape=(%d,%d,%d,%d) stride=(%d,%d,%d,%d) addr=0x%x\n",
    tl_if->shape.n,tl_if->shape.c,tl_if->shape.h,tl_if->shape.w,
    tl_if->stride.n,tl_if->stride.c,tl_if->stride.h,tl_if->stride.w, tl_if->start_address);
  fprintf(stderr,"[diag] of shape=(%d,%d,%d,%d) stride=(%d,%d,%d,%d) addr=0x%x\n",
    tl_of->shape.n,tl_of->shape.c,tl_of->shape.h,tl_of->shape.w,
    tl_of->stride.n,tl_of->stride.c,tl_of->stride.h,tl_of->stride.w, tl_of->start_address);
  fprintf(stderr,"[diag] w  shape=(%d,%d,%d,%d) stride=(%d,%d,%d,%d) addr=0x%x\n",
    tl_w->shape.n,tl_w->shape.c,tl_w->shape.h,tl_w->shape.w,
    tl_w->stride.n,tl_w->stride.c,tl_w->stride.h,tl_w->stride.w, tl_w->start_address);

  cvk_tg_t tg_if = {0, TPU_PA(&ctx,0),  CVK_FMT_I8, {1,IC,IH,IW},
    cvk->ops->tg_default_stride(cvk, (cvk_tg_shape_t){1,IC,IH,IW}, CVK_FMT_I8)};
  cvk_tg_t tg_w  = {0, TPU_PA(&ctx,64), CVK_FMT_I8, {1,OC,KH,KW},
    cvk->ops->tg_default_stride(cvk, (cvk_tg_shape_t){1,OC,KH,KW}, CVK_FMT_I8)};
  cvk_tg_t tg_of = {0, TPU_PA(&ctx,128), CVK_FMT_I8, {1,OC,OH,OW},
    cvk->ops->tg_default_stride(cvk, (cvk_tg_shape_t){1,OC,OH,OW}, CVK_FMT_I8)};

  cvk->ops->tdma_g2l_tensor_copy(cvk, &(cvk_tdma_g2l_tensor_copy_param_t){&tg_if, tl_if});
  cvk->ops->tdma_g2l_tensor_copy(cvk, &(cvk_tdma_g2l_tensor_copy_param_t){&tg_w,  tl_w});

  cvk_tiu_depthwise_convolution_param_t p = {
    .ofmap = tl_of, .ifmap = tl_if, .weight = tl_w,
    .chl_quan_param = NULL,
    .weight_is_const = 0,
    .ins_h = 0, .ins_last_h = 0, .ins_w = 0, .ins_last_w = 0,
    .dilation_h = 1, .dilation_w = 1,
    .pad_top = 0, .pad_bottom = 0, .pad_left = 0, .pad_right = 0,
    .stride_h = 1, .stride_w = 1,
    .has_bias = 0,
    .relu_enable = 0,
    .cmd_pre_exe_typ = 0, .cmd_pre_exe = 0,
    .layer_id = 1,
  };

  fprintf(stderr,"[diag] calling tiu_depthwise_convolution...\n");
  cvk->ops->tiu_depthwise_convolution(cvk, &p);

  cvk->ops->tdma_l2g_tensor_copy(cvk, &(cvk_tdma_l2g_tensor_copy_param_t){tl_of, &tg_of});

  int64_t t_ns = tpu_submit(&ctx);
  if (t_ns < 0) { fprintf(stderr,"submit fail\n"); goto out; }

  CVI_RT_MemInvld(ctx.rt_handle, ctx.neuron_mem);
  int8_t *v = (int8_t*)ctx.neuron_vaddr;
  int el = OC*OH*OW;
  fprintf(stderr,"[dbg] dw_conv out:"); for(int i=0;i<el;i++) fprintf(stderr,"%d,",v[128+i]);
  fprintf(stderr,"\n");
  int errs = tpu_check_i8("dw_conv", v+128, expected, el, el);
  printf("  time: %.2f us\n", t_ns/1000.0);

out:
  cvk->ops->lmem_free_tensor(cvk, tl_w);
  cvk->ops->lmem_free_tensor(cvk, tl_of);
  cvk->ops->lmem_free_tensor(cvk, tl_if);
  tpu_cleanup(&ctx);
  return errs?1:0;
}
