/* cvikernel conv v3 — diagnose result mismatch with simple weight patterns.
   3x3 input [1..9], 3x3 weight, stride=1, pad=0 → 1x1 output.
   Also tests pt_convolution (plain, no chl_quan). */
#include "../common/tpu_bench.h"

static const int8_t ifmap[9] = {1,2,3,4,5,6,7,8,9};

static void run_conv(tpu_ctx *ctx, cvk_context_t *cvk,
                     const char *label, const int8_t *wgt, int8_t exp,
                     int use_pt)  /* 0=tiu_convolution, 1=tiu_pt_convolution */
{
  memcpy(ctx->neuron_vaddr, ifmap, 9);
  memcpy(ctx->neuron_vaddr + 16, wgt, 9);
  if (!use_pt) memset(ctx->neuron_vaddr + 32, 0, 2); /* chl_quan rshift=0 */
  CVI_RT_MemFlush(ctx->rt_handle, ctx->neuron_mem);

  cvk_tl_t *tl_if = cvk->ops->lmem_alloc_tensor(cvk, (cvk_tl_shape_t){1,1,3,3}, CVK_FMT_I8, 1);
  cvk_tl_t *tl_of = cvk->ops->lmem_alloc_tensor(cvk, (cvk_tl_shape_t){1,1,1,1}, CVK_FMT_I8, 1);
  cvk_tl_t *tl_w  = cvk->ops->lmem_alloc_tensor(cvk, (cvk_tl_shape_t){1,1,3,3}, CVK_FMT_I8, 1);
  tl_w->stride.n = 1; tl_w->cmprs_fmt = CVK_FMT_I8;

  cvk_tl_t *tl_chlq = NULL, *tl_bias = NULL;
  if (!use_pt) {
    tl_chlq = cvk->ops->lmem_alloc_tensor(cvk, (cvk_tl_shape_t){1,1,1,1}, CVK_FMT_I8, 1);
  }
  /* PT conv: try bias=NULL (no bias) */
  if (!tl_if||!tl_of||!tl_w||(!use_pt&&!tl_chlq)) {
    fprintf(stderr,"  alloc fail\n"); return;
  }

  cvk_tg_shape_t g_if={1,1,3,3}, g_w={1,1,3,3}, g_of={1,1,1,1};
  cvk->ops->tdma_g2l_tensor_copy(cvk, &(cvk_tdma_g2l_tensor_copy_param_t){
    &(cvk_tg_t){0, TPU_PA(ctx,0),  CVK_FMT_I8, g_if, cvk->ops->tg_default_stride(cvk, g_if, CVK_FMT_I8)}, tl_if});
  cvk->ops->tdma_g2l_tensor_copy(cvk, &(cvk_tdma_g2l_tensor_copy_param_t){
    &(cvk_tg_t){0, TPU_PA(ctx,16), CVK_FMT_I8, g_w,  cvk->ops->tg_default_stride(cvk, g_w,  CVK_FMT_I8)}, tl_w});

  if (tl_chlq) {
    cvk_tg_shape_t g_q={1,1,1,1};
    cvk->ops->tdma_g2l_tensor_copy(cvk, &(cvk_tdma_g2l_tensor_copy_param_t){
      &(cvk_tg_t){0, TPU_PA(ctx,32), CVK_FMT_I8, g_q, cvk->ops->tg_default_stride(cvk, g_q, CVK_FMT_I8)}, tl_chlq});
  }

  if (!use_pt) {
    cvk->ops->tiu_convolution(cvk, &(cvk_tiu_convolution_param_t){
      .ofmap=tl_of, .ifmap=tl_if, .weight=tl_w, .chl_quan_param=tl_chlq,
      .ins_h=0, .ins_last_h=0, .ins_w=0, .ins_last_w=0,
      .pad_top=0, .pad_bottom=0, .pad_left=0, .pad_right=0,
      .stride_h=1, .stride_w=1, .dilation_h=1, .dilation_w=1,
      .has_bias=0, .relu_enable=0, .ps32_mode=0, .w_is_const=0, .layer_id=1,
    });
  } else {
    cvk->ops->tiu_pt_convolution(cvk, &(cvk_tiu_pt_convolution_param_t){
      .ofmap=tl_of, .ifmap=tl_if, .weight=tl_w, .bias=NULL,
      .ins_h=0, .ins_last_h=0, .ins_w=0, .ins_last_w=0,
      .pad_top=0, .pad_bottom=0, .pad_left=0, .pad_right=0,
      .stride_h=1, .stride_w=1, .dilation_h=1, .dilation_w=1,
      .relu_enable=0, .rshift_bits=0, .ps32_mode=0, .w_is_const=0,
      .layer_id=1,
    });
  }

  cvk->ops->tdma_l2g_tensor_copy(cvk, &(cvk_tdma_l2g_tensor_copy_param_t){tl_of,
    &(cvk_tg_t){0, TPU_PA(ctx,48), CVK_FMT_I8, g_of, cvk->ops->tg_default_stride(cvk, g_of, CVK_FMT_I8)}});

  int64_t t_ns = tpu_submit(ctx);
  CVI_RT_MemInvld(ctx->rt_handle, ctx->neuron_mem);
  int8_t out = ctx->neuron_vaddr[48];
  fprintf(stderr, "  %-28s got=%3d exp=%3d %s (%.0f us)\n",
    label, out, exp, out==exp?"OK":"MISMATCH", t_ns/1000.0);

  cvk->ops->lmem_free_tensor(cvk, tl_of);
  cvk->ops->lmem_free_tensor(cvk, tl_w);
  cvk->ops->lmem_free_tensor(cvk, tl_if);
  if (tl_chlq) cvk->ops->lmem_free_tensor(cvk, tl_chlq);
}

int main() {
  tpu_ctx ctx;
  if (tpu_init(&ctx, 4096) != 0) return 1;
  cvk_context_t *cvk = ctx.cvk_ctx;

  /* weight patterns:
     center1: [0,0,0, 0,1,0, 0,0,0] → exp = ifmap[1,1] = 5
     all_ones: [1,1,1, 1,1,1, 1,1,1] → exp = sum(ifmap) = 45
     corners:  [1,0,1, 0,0,0, 1,0,1] → exp = 1+3+7+9 = 20
     top_left: [1,0,0, 0,0,0, 0,0,0] → exp = ifmap[0,0] = 1
  */
  const int8_t w_center[]  = {0,0,0, 0,1,0, 0,0,0};
  const int8_t w_ones[]    = {1,1,1, 1,1,1, 1,1,1};
  const int8_t w_corners[] = {1,0,1, 0,0,0, 1,0,1};
  const int8_t w_topleft[] = {1,0,0, 0,0,0, 0,0,0};

  fprintf(stderr, "\n=== conv3x3_cvk_v3: comparing tiu_conv vs tiu_pt_conv ===\n");

  fprintf(stderr, "--- tiu_convolution (fused, chl_quan) ---\n");
  run_conv(&ctx, cvk, "center_1",     w_center,  5,  0);
  run_conv(&ctx, cvk, "all_ones",     w_ones,    45, 0);
  run_conv(&ctx, cvk, "corners",      w_corners, 20, 0);
  run_conv(&ctx, cvk, "top_left",     w_topleft, 1,  0);

  fprintf(stderr, "--- tiu_pt_convolution (plain, explicit bias+rshift) ---\n");
  run_conv(&ctx, cvk, "center_1",     w_center,  5,  1);
  run_conv(&ctx, cvk, "all_ones",     w_ones,    45, 1);
  run_conv(&ctx, cvk, "corners",      w_corners, 20, 1);
  run_conv(&ctx, cvk, "top_left",     w_topleft, 1,  1);

  fprintf(stderr, "=== DONE ===\n\n");
  tpu_cleanup(&ctx);
  return 0;
}
