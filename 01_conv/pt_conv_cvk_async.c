/* cvikernel pt_convolution with SubmitAsync — working path!
   Uses tiu_pt_convolution (plain, no chl_quan) with bias=NULL.
   Measures blocking vs async timing. */
#include "../common/tpu_bench.h"

#define IC  1
#define OC  1
#define IH  3
#define IW  3
#define KH  3
#define KW  3
#define OH  1
#define OW  1
#define N_RUNS 5

static const int8_t ifmap[9] = {1,2,3,4,5,6,7,8,9};
static const int8_t wgt[9]    = {1,0,1, 0,0,0, 1,0,1};
static const int8_t exp_val    = 20;

static double run_one(tpu_ctx *ctx, cvk_context_t *cvk, int use_async) {
  memcpy(ctx->neuron_vaddr, ifmap, 9);
  memcpy(ctx->neuron_vaddr+16, wgt, 9);
  CVI_RT_MemFlush(ctx->rt_handle, ctx->neuron_mem);

  cvk_tl_t *tl_if = cvk->ops->lmem_alloc_tensor(cvk, (cvk_tl_shape_t){1,1,3,3}, CVK_FMT_I8, 1);
  cvk_tl_t *tl_of = cvk->ops->lmem_alloc_tensor(cvk, (cvk_tl_shape_t){1,1,1,1}, CVK_FMT_I8, 1);
  cvk_tl_t *tl_w  = cvk->ops->lmem_alloc_tensor(cvk, (cvk_tl_shape_t){1,1,3,3}, CVK_FMT_I8, 1);
  tl_w->stride.n = 1; tl_w->cmprs_fmt = CVK_FMT_I8;
  if (!tl_if||!tl_of||!tl_w) return -1;

  cvk_tg_shape_t g={1,1,3,3}, g1={1,1,1,1};
  cvk->ops->tdma_g2l_tensor_copy(cvk, &(cvk_tdma_g2l_tensor_copy_param_t){
    &(cvk_tg_t){0,TPU_PA(ctx,0), CVK_FMT_I8,g, cvk->ops->tg_default_stride(cvk,g,CVK_FMT_I8)},tl_if});
  cvk->ops->tdma_g2l_tensor_copy(cvk, &(cvk_tdma_g2l_tensor_copy_param_t){
    &(cvk_tg_t){0,TPU_PA(ctx,16),CVK_FMT_I8,g, cvk->ops->tg_default_stride(cvk,g,CVK_FMT_I8)},tl_w});

  cvk->ops->tiu_pt_convolution(cvk, &(cvk_tiu_pt_convolution_param_t){
    .ofmap=tl_of, .ifmap=tl_if, .weight=tl_w, .bias=NULL,
    .ins_h=0, .ins_last_h=0, .ins_w=0, .ins_last_w=0,
    .pad_top=0, .pad_bottom=0, .pad_left=0, .pad_right=0,
    .stride_h=1, .stride_w=1, .dilation_h=1, .dilation_w=1,
    .relu_enable=0, .rshift_bits=0, .ps32_mode=0, .w_is_const=0,
    .layer_id=1,
  });

  cvk->ops->tdma_l2g_tensor_copy(cvk, &(cvk_tdma_l2g_tensor_copy_param_t){tl_of,
    &(cvk_tg_t){0,TPU_PA(ctx,32),CVK_FMT_I8,g1,cvk->ops->tg_default_stride(cvk,g1,CVK_FMT_I8)}});

  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);
  if (use_async) {
    CVI_RT_SubmitAsync(ctx->rt_khandle, 0);
    volatile int d=0;
    for (int k=0;k<1000;k++) d+=(ifmap[k%9]*wgt[k%9])^(k&0xff);
    CVI_RT_WaitForAsync(ctx->rt_khandle);
  } else {
    CVI_RT_Submit(ctx->rt_khandle);
  }
  clock_gettime(CLOCK_MONOTONIC, &t1);
  double t = ((t1.tv_sec-t0.tv_sec)*1e9+(t1.tv_nsec-t0.tv_nsec))/1000.0;

  CVI_RT_MemInvld(ctx->rt_handle, ctx->neuron_mem);

  cvk->ops->lmem_free_tensor(cvk, tl_w);
  cvk->ops->lmem_free_tensor(cvk, tl_of);
  cvk->ops->lmem_free_tensor(cvk, tl_if);
  return t;
}

int main() {
  tpu_ctx ctx;
  if (tpu_init(&ctx, 4096) != 0) return 1;
  cvk_context_t *cvk = ctx.cvk_ctx;

  fprintf(stderr, "\n=== pt_conv_cvk_async: 3x3 pt_conv, blocking vs SubmitAsync ===\n");
  fprintf(stderr, "  %-20s %8s %8s %8s\n", "method", "avg_us", "min_us", "max_us");

  /* Blocking */
  {
    double sum=0, tmin=1e9, tmax=0;
    for (int r=0;r<N_RUNS;r++) {
      double t = run_one(&ctx, cvk, 0);
      if (t<0) { fprintf(stderr,"blocking fail\n"); goto out; }
      sum+=t; if(t<tmin)tmin=t; if(t>tmax)tmax=t;
      int8_t v = ctx.neuron_vaddr[32];
      if (v != exp_val) fprintf(stderr,"  WARN: run %d got=%d exp=%d\n", r, v, exp_val);
    }
    fprintf(stderr,"  %-20s %8.1f %8.1f %8.1f\n", "pt_conv(block)", sum/N_RUNS, tmin, tmax);
  }

  /* SubmitAsync + CPU work + Wait */
  {
    double sum=0, tmin=1e9, tmax=0;
    for (int r=0;r<N_RUNS;r++) {
      double t = run_one(&ctx, cvk, 1);
      if (t<0) { fprintf(stderr,"async fail\n"); goto out; }
      sum+=t; if(t<tmin)tmin=t; if(t>tmax)tmax=t;
      int8_t v = ctx.neuron_vaddr[32];
      if (v != exp_val) fprintf(stderr,"  WARN: run %d got=%d exp=%d\n", r, v, exp_val);
    }
    fprintf(stderr,"  %-20s %8.1f %8.1f %8.1f\n", "pt_conv(async)", sum/N_RUNS, tmin, tmax);
  }

out:
  fprintf(stderr, "=== DONE pt_conv_cvk_async ===\n\n");
  tpu_cleanup(&ctx);
  return 0;
}
