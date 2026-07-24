/* Test depthwise pt_conv — shares TIU type 1 (DEPTHWISE_POOL_FIX8B) with max_pooling */
#include "../common/tpu_bench.h"

#define IC 2
#define OC 2   /* depthwise: IC == OC */
#define IH 2
#define IW 2
#define OH 2
#define OW 2

static const int8_t ifmap[IC*IH*IW] = {
  1,2, 3,4,   /* ch0 */
  5,6, 7,8,   /* ch1 */
};
/* Depthwise weight: (1, OC, kh*kw, 1) = (1, 2, 1, 1), identity per channel */
static const int8_t weight[OC*1*1] = {1, 1};
static const int8_t bias_data[2*OC*1*1] = {0,0, 0,0};
static const int8_t expected[OC*OH*OW] = {
  1,2, 3,4,   /* oc0 = ch0 */
  5,6, 7,8,   /* oc1 = ch1 */
};

int main() {
  tpu_ctx ctx;
  if (tpu_init(&ctx, 4096) != 0) return 1;
  cvk_context_t *cvk = ctx.cvk_ctx;

  memcpy(ctx.neuron_vaddr, ifmap, sizeof(ifmap));
  memcpy(ctx.neuron_vaddr+32, weight, sizeof(weight));
  memcpy(ctx.neuron_vaddr+48, bias_data, sizeof(bias_data));
  CVI_RT_MemFlush(ctx.rt_handle, ctx.neuron_mem);

  cvk_tl_t *tl_if = cvk->ops->lmem_alloc_tensor(cvk, (cvk_tl_shape_t){1,IC,IH,IW}, CVK_FMT_I8, 1);
  cvk_tl_t *tl_of = cvk->ops->lmem_alloc_tensor(cvk, (cvk_tl_shape_t){1,OC,OH,OW}, CVK_FMT_I8, 1);
  cvk_tl_t *tl_w  = cvk->ops->lmem_alloc_tensor(cvk, (cvk_tl_shape_t){1,OC,1,1}, CVK_FMT_I8, 1);
  cvk_tl_t *tl_b  = cvk->ops->lmem_alloc_tensor(cvk, (cvk_tl_shape_t){2,OC,1,1}, CVK_FMT_I8, 1);
  if (!tl_if||!tl_of||!tl_w||!tl_b) {
    fprintf(stderr,"alloc fail\n"); tpu_cleanup(&ctx); return 1;
  }

  fprintf(stderr,"[diag] if stride=(%d,%d,%d,%d) addr=0x%x\n",
    tl_if->stride.n,tl_if->stride.c,tl_if->stride.h,tl_if->stride.w, tl_if->start_address);
  fprintf(stderr,"[diag] of stride=(%d,%d,%d,%d) addr=0x%x\n",
    tl_of->stride.n,tl_of->stride.c,tl_of->stride.h,tl_of->stride.w, tl_of->start_address);
  fprintf(stderr,"[diag] w  stride=(%d,%d,%d,%d) addr=0x%x\n",
    tl_w->stride.n,tl_w->stride.c,tl_w->stride.h,tl_w->stride.w, tl_w->start_address);
  fprintf(stderr,"[diag] b  stride=(%d,%d,%d,%d) addr=0x%x\n",
    tl_b->stride.n,tl_b->stride.c,tl_b->stride.h,tl_b->stride.w, tl_b->start_address);

  cvk_tg_t tg_if = {0, TPU_PA(&ctx,0),  CVK_FMT_I8, {1,IC,IH,IW},
    cvk->ops->tg_default_stride(cvk, (cvk_tg_shape_t){1,IC,IH,IW}, CVK_FMT_I8)};
  cvk_tg_t tg_w  = {0, TPU_PA(&ctx,32), CVK_FMT_I8, {1,OC,1,1},
    cvk->ops->tg_default_stride(cvk, (cvk_tg_shape_t){1,OC,1,1}, CVK_FMT_I8)};
  cvk_tg_t tg_b  = {0, TPU_PA(&ctx,48), CVK_FMT_I8, {2,OC,1,1},
    cvk->ops->tg_default_stride(cvk, (cvk_tg_shape_t){2,OC,1,1}, CVK_FMT_I8)};
  cvk_tg_t tg_of = {0, TPU_PA(&ctx,64), CVK_FMT_I8, {1,OC,OH,OW},
    cvk->ops->tg_default_stride(cvk, (cvk_tg_shape_t){1,OC,OH,OW}, CVK_FMT_I8)};

  cvk->ops->tdma_g2l_tensor_copy(cvk, &(cvk_tdma_g2l_tensor_copy_param_t){&tg_if, tl_if});
  cvk->ops->tdma_g2l_tensor_copy(cvk, &(cvk_tdma_g2l_tensor_copy_param_t){&tg_w,  tl_w});
  cvk->ops->tdma_g2l_tensor_copy(cvk, &(cvk_tdma_g2l_tensor_copy_param_t){&tg_b,  tl_b});

  cvk_tiu_depthwise_pt_convolution_param_t p = {
    .ofmap = tl_of, .ifmap = tl_if, .weight = tl_w, .bias = tl_b,
    .weight_is_const = 0,
    .ins_h = 0, .ins_last_h = 0, .ins_w = 0, .ins_last_w = 0,
    .pad_top = 0, .pad_bottom = 0, .pad_left = 0, .pad_right = 0,
    .stride_h = 1, .stride_w = 1,
    .dilation_h = 1, .dilation_w = 1,
    .relu_enable = 0, .rshift_bits = 0,
    .ps32_mode = 0,
    .cmd_pre_exe_typ = 0, .cmd_pre_exe = 0,
    .layer_id = 1,
  };

  fprintf(stderr,"[diag] calling tiu_pt_depthwise_convolution...\n");
  cvk->ops->tiu_pt_depthwise_convolution(cvk, &p);

  cvk->ops->tdma_l2g_tensor_copy(cvk, &(cvk_tdma_l2g_tensor_copy_param_t){tl_of, &tg_of});

  int64_t t_ns = tpu_submit(&ctx);
  if (t_ns < 0) { fprintf(stderr,"submit fail\n"); goto out; }

  CVI_RT_MemInvld(ctx.rt_handle, ctx.neuron_mem);
  int8_t *v = (int8_t*)ctx.neuron_vaddr;
  int el = OC*OH*OW;
  fprintf(stderr,"[dbg] dw_pt_conv out:"); for(int i=0;i<el;i++) fprintf(stderr,"%d,",v[64+i]);
  fprintf(stderr,"\n");
  int errs = tpu_check_i8("dw_pt_conv", v+64, expected, el, el);
  printf("  time: %.2f us\n", t_ns/1000.0);

out:
  cvk->ops->lmem_free_tensor(cvk, tl_b);
  cvk->ops->lmem_free_tensor(cvk, tl_w);
  cvk->ops->lmem_free_tensor(cvk, tl_of);
  cvk->ops->lmem_free_tensor(cvk, tl_if);
  tpu_cleanup(&ctx);
  return errs?1:0;
}
