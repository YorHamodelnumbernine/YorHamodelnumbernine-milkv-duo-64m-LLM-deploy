/* TPU conv1x1 v4: tiu_conv with kh=1,kw=1, larger spatial dims.
   IC=2, OC=2, IH=IW=4, OH=OW=4. Identity test. */
#include "../common/tpu_bench.h"

#define IC 2
#define OC 2
#define IH 4
#define IW 4
#define OH 4
#define OW 4
#define N  (IC*IH*IW)

static const int8_t ifmap[N] = {
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
/* weight (1,OC,KH*KW,IC) = (1,2,1,2): identity */
static const int8_t weight[OC*1*IC] = {1,0, 0,1};
/* chl_quan (1,OC,1,1): zeros for no quant */
static const int8_t chl_quan[OC] = {0,0};

/* expected = ifmap (identity conv) */
static const int8_t expected[OC*OH*OW] = {
  /* matches ifmap exactly */
  1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16,
  17,18,19,20, 21,22,23,24, 25,26,27,28, 29,30,31,32,
};

int main() {
  tpu_ctx ctx;
  if (tpu_init(&ctx, 4096) != 0) return 1;
  cvk_context_t *cvk = ctx.cvk_ctx;

  memcpy(ctx.neuron_vaddr, ifmap, sizeof(ifmap));
  memcpy(ctx.neuron_vaddr+64, weight, sizeof(weight));
  memcpy(ctx.neuron_vaddr+96, chl_quan, sizeof(chl_quan));
  CVI_RT_MemFlush(ctx.rt_handle, ctx.neuron_mem);

  cvk_tl_t *tl_if = cvk->ops->lmem_alloc_tensor(cvk, (cvk_tl_shape_t){1,IC,IH,IW}, CVK_FMT_I8, 1);
  cvk_tl_t *tl_of = cvk->ops->lmem_alloc_tensor(cvk, (cvk_tl_shape_t){1,OC,OH,OW}, CVK_FMT_I8, 1);
  cvk_tl_t *tl_w  = cvk->ops->lmem_alloc_tensor(cvk, (cvk_tl_shape_t){1,OC,1,IC}, CVK_FMT_I8, 1);
  cvk_tl_t *tl_q  = cvk->ops->lmem_alloc_tensor(cvk, (cvk_tl_shape_t){1,OC,1,1}, CVK_FMT_I8, 1);
  if (!tl_if||!tl_of||!tl_w||!tl_q) {
    fprintf(stderr,"alloc fail\n"); tpu_cleanup(&ctx); return 1;
  }

  fprintf(stderr,"[diag] ifmap: addr=0x%x shape=(%d,%d,%d,%d) stride=(%d,%d,%d,%d)\n",
    tl_if->start_address, tl_if->shape.n,tl_if->shape.c,tl_if->shape.h,tl_if->shape.w,
    tl_if->stride.n,tl_if->stride.c,tl_if->stride.h,tl_if->stride.w);
  fprintf(stderr,"[diag] weight: addr=0x%x shape=(%d,%d,%d,%d) stride=(%d,%d,%d,%d)\n",
    tl_w->start_address, tl_w->shape.n,tl_w->shape.c,tl_w->shape.h,tl_w->shape.w,
    tl_w->stride.n,tl_w->stride.c,tl_w->stride.h,tl_w->stride.w);
  fprintf(stderr,"[diag] ofmap: addr=0x%x shape=(%d,%d,%d,%d) stride=(%d,%d,%d,%d)\n",
    tl_of->start_address, tl_of->shape.n,tl_of->shape.c,tl_of->shape.h,tl_of->shape.w,
    tl_of->stride.n,tl_of->stride.c,tl_of->stride.h,tl_of->stride.w);
  fprintf(stderr,"[diag] chl_q: addr=0x%x shape=(%d,%d,%d,%d) stride=(%d,%d,%d,%d)\n",
    tl_q->start_address, tl_q->shape.n,tl_q->shape.c,tl_q->shape.h,tl_q->shape.w,
    tl_q->stride.n,tl_q->stride.c,tl_q->stride.h,tl_q->stride.w);

  cvk_tg_t tg_if = {0, TPU_PA(&ctx,0),  CVK_FMT_I8, {1,IC,IH,IW},
    cvk->ops->tg_default_stride(cvk, (cvk_tg_shape_t){1,IC,IH,IW}, CVK_FMT_I8)};
  cvk_tg_t tg_w  = {0, TPU_PA(&ctx,64), CVK_FMT_I8, {1,OC,1,IC},
    cvk->ops->tg_default_stride(cvk, (cvk_tg_shape_t){1,OC,1,IC}, CVK_FMT_I8)};
  cvk_tg_t tg_q  = {0, TPU_PA(&ctx,96), CVK_FMT_I8, {1,OC,1,1},
    cvk->ops->tg_default_stride(cvk, (cvk_tg_shape_t){1,OC,1,1}, CVK_FMT_I8)};
  cvk_tg_t tg_of = {0, TPU_PA(&ctx,128), CVK_FMT_I8, {1,OC,OH,OW},
    cvk->ops->tg_default_stride(cvk, (cvk_tg_shape_t){1,OC,OH,OW}, CVK_FMT_I8)};

  cvk->ops->tdma_g2l_tensor_copy(cvk, &(cvk_tdma_g2l_tensor_copy_param_t){&tg_if, tl_if});
  cvk->ops->tdma_g2l_tensor_copy(cvk, &(cvk_tdma_g2l_tensor_copy_param_t){&tg_w,  tl_w});
  cvk->ops->tdma_g2l_tensor_copy(cvk, &(cvk_tdma_g2l_tensor_copy_param_t){&tg_q,  tl_q});

  fprintf(stderr,"[diag] tiu_convolution fnptr=0x%llx\n",
    (unsigned long long)(uintptr_t)cvk->ops->tiu_convolution);

  cvk_tiu_convolution_param_t p = {
    .ofmap = tl_of, .ifmap = tl_if, .weight = tl_w,
    .chl_quan_param = tl_q,
    .ins_h = 0, .ins_last_h = 0, .ins_w = 0, .ins_last_w = 0,
    .pad_top = 0, .pad_bottom = 0, .pad_left = 0, .pad_right = 0,
    .stride_h = 1, .stride_w = 1,
    .dilation_h = 1, .dilation_w = 1,
    .has_bias = 0,
    .relu_enable = 0,
    .ps32_mode = 0, .w_is_const = 0,
    .cmd_pre_exe_typ = 0, .cmd_pre_exe = 0,
    .layer_id = 1,
  };
  cvk->ops->tiu_convolution(cvk, &p);

  cvk->ops->tdma_l2g_tensor_copy(cvk, &(cvk_tdma_l2g_tensor_copy_param_t){tl_of, &tg_of});

  int64_t t_ns = tpu_submit(&ctx);
  if (t_ns < 0) { fprintf(stderr,"submit fail\n"); goto out; }

  CVI_RT_MemInvld(ctx.rt_handle, ctx.neuron_mem);
  int8_t *v = (int8_t*)ctx.neuron_vaddr;
  int el = OC*OH*OW;
  fprintf(stderr,"[dbg] conv out first 16:"); for(int i=0;i<16;i++) fprintf(stderr,"%d,",v[128+i]);
  fprintf(stderr,"\n");
  int errs = tpu_check_i8("conv1x1", v+128, expected, el, el);
  printf("  time: %.2f us\n", t_ns/1000.0);

out:
  cvk->ops->lmem_free_tensor(cvk, tl_q);
  cvk->ops->lmem_free_tensor(cvk, tl_w);
  cvk->ops->lmem_free_tensor(cvk, tl_of);
  cvk->ops->lmem_free_tensor(cvk, tl_if);
  tpu_cleanup(&ctx);
  return errs?1:0;
}
