/* TPU depthwise_convolution: 3x3 dw conv on 3x3 input. IC=OC=1.
   ifmap [1..9], weight [1,1,1, 1,2,1, 1,1,1], no padding, stride=1.
   Result: sum(ifmap[i] * weight[i]) = 1+2+3+4+10+6+7+8+9 = 50. Too big for INT8!
   Use small weight: [1,0,0, 0,1,0, 0,0,1].
   Result = 1*1 + 5*1 + 9*1 = 15 */
#include "../common/tpu_bench.h"

#define IC  1
#define OC  1
#define IH  3
#define IW  3
#define KH  3
#define KW  3
#define OH  1
#define OW  1

static const int8_t ifmap[IC*IH*IW] = {1,2,3,4,5,6,7,8,9};
/* depthwise weight: (1, OC, KH*KW, 1) = (1, 1, 9, 1) */
static const int8_t weight[OC*KH*KW*1] = {1,0,0, 0,1,0, 0,0,1};
static const int8_t expected[OC*OH*OW] = {15};

int main() {
  tpu_ctx ctx;
  if (tpu_init(&ctx, 4096) != 0) return 1;
  cvk_context_t *cvk = ctx.cvk_ctx;

  memcpy(ctx.neuron_vaddr, ifmap, sizeof(ifmap));
  memcpy(ctx.neuron_vaddr+16, weight, sizeof(weight));
  CVI_RT_MemFlush(ctx.rt_handle, ctx.neuron_mem);

  cvk_tl_t *tl_if = cvk->ops->lmem_alloc_tensor(cvk, (cvk_tl_shape_t){1,IC,IH,IW}, CVK_FMT_I8, 1);
  cvk_tl_t *tl_of = cvk->ops->lmem_alloc_tensor(cvk, (cvk_tl_shape_t){1,OC,OH,OW}, CVK_FMT_I8, 1);
  cvk_tl_t *tl_w  = cvk->ops->lmem_alloc_tensor(cvk, (cvk_tl_shape_t){1,OC,KH,KW}, CVK_FMT_I8, 1);
  tl_w->stride.n = 1;
  tl_w->cmprs_fmt = CVK_FMT_I8;
  cvk_tl_t *tl_q  = cvk->ops->lmem_alloc_tensor(cvk, (cvk_tl_shape_t){1,OC,1,1}, CVK_FMT_I8, 1);
  if (!tl_if||!tl_of||!tl_w||!tl_q) {
    fprintf(stderr,"alloc fail\n"); tpu_cleanup(&ctx); return 1;
  }

  cvk_tg_t tg_if = {0, TPU_PA(&ctx,0),  CVK_FMT_I8, {1,IC,IH,IW},
    cvk->ops->tg_default_stride(cvk, (cvk_tg_shape_t){1,IC,IH,IW}, CVK_FMT_I8)};
  cvk_tg_t tg_w  = {0, TPU_PA(&ctx,16), CVK_FMT_I8, {1,OC,KH,KW},
    cvk->ops->tg_default_stride(cvk, (cvk_tg_shape_t){1,OC,KH,KW}, CVK_FMT_I8)};
  cvk_tg_t tg_of = {0, TPU_PA(&ctx,32), CVK_FMT_I8, {1,OC,OH,OW},
    cvk->ops->tg_default_stride(cvk, (cvk_tg_shape_t){1,OC,OH,OW}, CVK_FMT_I8)};

  cvk->ops->tdma_g2l_tensor_copy(cvk, &(cvk_tdma_g2l_tensor_copy_param_t){&tg_if, tl_if});
  cvk->ops->tdma_g2l_tensor_copy(cvk, &(cvk_tdma_g2l_tensor_copy_param_t){&tg_w,  tl_w});

  cvk_tiu_depthwise_convolution_param_t p = {
    .ofmap = tl_of, .ifmap = tl_if, .weight = tl_w,
    .chl_quan_param = tl_q,
    .ins_h = 0, .ins_last_h = 0, .ins_w = 0, .ins_last_w = 0,
    .pad_top = 0, .pad_bottom = 0, .pad_left = 0, .pad_right = 0,
    .stride_h = 1, .stride_w = 1,
    .dilation_h = 1, .dilation_w = 1,
    .has_bias = 0, .relu_enable = 0,
    .weight_is_const = 0,
  };
  cvk->ops->tiu_depthwise_convolution(cvk, &p);

  cvk->ops->tdma_l2g_tensor_copy(cvk, &(cvk_tdma_l2g_tensor_copy_param_t){tl_of, &tg_of});

  int64_t t_ns = tpu_submit(&ctx);
  if (t_ns < 0) { fprintf(stderr,"submit fail\n"); goto out; }

  CVI_RT_MemInvld(ctx.rt_handle, ctx.neuron_mem);
  int8_t *v = (int8_t*)ctx.neuron_vaddr;
  fprintf(stderr,"[dbg] dw_conv out:"); for(int i=0;i<OC*OH*OW;i++) fprintf(stderr,"%d,",v[32+i]);
  fprintf(stderr,"\n");
  int errs = tpu_check_i8("dw_conv", v+32, expected, OC*OH*OW, OC*OH*OW);
  printf("  time: %.2f us\n", t_ns/1000.0);

out:
  cvk->ops->lmem_free_tensor(cvk, tl_q);
  cvk->ops->lmem_free_tensor(cvk, tl_w);
  cvk->ops->lmem_free_tensor(cvk, tl_of);
  cvk->ops->lmem_free_tensor(cvk, tl_if);
  tpu_cleanup(&ctx);
  return errs?1:0;
}
