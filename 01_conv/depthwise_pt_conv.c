/* TPU depthwise_pt_convolution: 1x1 depthwise point-wise conv.
   IC=OC=2, IH=IW=1. Like a per-channel linear transform.
   ifmap: [c0=1, c1=2]. weight per-channel: [2, 3]. bias: [1, -1].
   out: c0=1*2+1=3, c1=2*3-1=5 */
#include "../common/tpu_bench.h"

#define IC  2
#define OC  2
#define IH  1
#define IW  1

static const int8_t ifmap[IC*IH*IW]      = {1, 2};
static const int8_t weight[OC*1*1]        = {2, 3}; /* (1,OC,KH*KW,1) = (1,2,1,1) */
static const int8_t bias_data[2*OC*1*1]   = {1, -1, 0,0}; /* (2,OC,1,1) */
static const int8_t expected[OC*IH*IW]    = {3, 5};

int main() {
  tpu_ctx ctx;
  if (tpu_init(&ctx, 4096) != 0) return 1;
  cvk_context_t *cvk = ctx.cvk_ctx;

  memcpy(ctx.neuron_vaddr, ifmap, sizeof(ifmap));
  memcpy(ctx.neuron_vaddr+16, weight, sizeof(weight));
  memcpy(ctx.neuron_vaddr+32, bias_data, sizeof(bias_data));
  CVI_RT_MemFlush(ctx.rt_handle, ctx.neuron_mem);

  cvk_tl_t *tl_if = cvk->ops->lmem_alloc_tensor(cvk, (cvk_tl_shape_t){1,IC,IH,IW}, CVK_FMT_I8, 1);
  cvk_tl_t *tl_of = cvk->ops->lmem_alloc_tensor(cvk, (cvk_tl_shape_t){1,OC,IH,IW}, CVK_FMT_I8, 1);
  cvk_tl_t *tl_w  = cvk->ops->lmem_alloc_tensor(cvk, (cvk_tl_shape_t){1,OC,1,1}, CVK_FMT_I8, 1);
  cvk_tl_t *tl_b  = cvk->ops->lmem_alloc_tensor(cvk, (cvk_tl_shape_t){2,OC,1,1}, CVK_FMT_I8, 1);
  if (!tl_if||!tl_of||!tl_w||!tl_b) {
    fprintf(stderr,"alloc fail\n"); tpu_cleanup(&ctx); return 1;
  }

  cvk_tg_t tg_if = {0, TPU_PA(&ctx,0),  CVK_FMT_I8, {1,IC,IH,IW},
    cvk->ops->tg_default_stride(cvk, (cvk_tg_shape_t){1,IC,IH,IW}, CVK_FMT_I8)};
  cvk_tg_t tg_w  = {0, TPU_PA(&ctx,16), CVK_FMT_I8, {1,OC,1,1},
    cvk->ops->tg_default_stride(cvk, (cvk_tg_shape_t){1,OC,1,1}, CVK_FMT_I8)};
  cvk_tg_t tg_b  = {0, TPU_PA(&ctx,32), CVK_FMT_I8, {2,OC,1,1},
    cvk->ops->tg_default_stride(cvk, (cvk_tg_shape_t){2,OC,1,1}, CVK_FMT_I8)};
  cvk_tg_t tg_of = {0, TPU_PA(&ctx,48), CVK_FMT_I8, {1,OC,IH,IW},
    cvk->ops->tg_default_stride(cvk, (cvk_tg_shape_t){1,OC,IH,IW}, CVK_FMT_I8)};

  cvk->ops->tdma_g2l_tensor_copy(cvk, &(cvk_tdma_g2l_tensor_copy_param_t){&tg_if, tl_if});
  cvk->ops->tdma_g2l_tensor_copy(cvk, &(cvk_tdma_g2l_tensor_copy_param_t){&tg_w,  tl_w});
  cvk->ops->tdma_g2l_tensor_copy(cvk, &(cvk_tdma_g2l_tensor_copy_param_t){&tg_b,  tl_b});

  cvk_tiu_depthwise_pt_convolution_param_t p = {
    .ofmap = tl_of, .ifmap = tl_if, .weight = tl_w, .bias = tl_b,
    .ins_h = 0, .ins_last_h = 0, .ins_w = 0, .ins_last_w = 0,
    .dilation_h = 1, .dilation_w = 1,
    .pad_top = 0, .pad_bottom = 0, .pad_left = 0, .pad_right = 0,
    .stride_h = 1, .stride_w = 1,
    .rshift_bits = 0, .relu_enable = 0,
  };
  cvk->ops->tiu_pt_depthwise_convolution(cvk, &p);

  cvk->ops->tdma_l2g_tensor_copy(cvk, &(cvk_tdma_l2g_tensor_copy_param_t){tl_of, &tg_of});

  int64_t t_ns = tpu_submit(&ctx);
  if (t_ns < 0) { fprintf(stderr,"submit fail\n"); goto out; }

  CVI_RT_MemInvld(ctx.rt_handle, ctx.neuron_mem);
  int8_t *v = (int8_t*)ctx.neuron_vaddr;
  fprintf(stderr,"[dbg] dw_pt out:"); for(int i=0;i<OC*IH*IW;i++) fprintf(stderr,"%d,",v[48+i]);
  fprintf(stderr,"\n");
  int errs = tpu_check_i8("dw_pt", v+48, expected, OC*IH*IW, OC*IH*IW);
  printf("  time: %.2f us\n", t_ns/1000.0);

out:
  cvk->ops->lmem_free_tensor(cvk, tl_b);
  cvk->ops->lmem_free_tensor(cvk, tl_w);
  cvk->ops->lmem_free_tensor(cvk, tl_of);
  cvk->ops->lmem_free_tensor(cvk, tl_if);
  tpu_cleanup(&ctx);
  return errs?1:0;
}
