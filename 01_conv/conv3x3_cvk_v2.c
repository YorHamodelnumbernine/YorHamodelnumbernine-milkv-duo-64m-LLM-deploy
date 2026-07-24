/* cvikernel conv3x3 v2 — minimal diagnostic version.
   All output to stderr to avoid stdio buffering confusion. */
#include "../common/tpu_bench.h"

#define IC 1
#define OC 1
#define IH 3
#define IW 3
#define KH 3
#define KW 3
#define OH 1
#define OW 1

static const int8_t ifmap[9] = {1,2,3,4,5,6,7,8,9};
static const int8_t wgt[9]    = {1,0,1, 0,0,0, 1,0,1};
static const int8_t expected   = 20;

int main() {
  fprintf(stderr, "\n=== RUN conv3x3_cvk_v2 PID=%d ===\n", getpid());

  tpu_ctx ctx;
  if (tpu_init(&ctx, 4096) != 0) return 1;
  cvk_context_t *cvk = ctx.cvk_ctx;

  memcpy(ctx.neuron_vaddr, ifmap, 9);
  memcpy(ctx.neuron_vaddr + 16, wgt, 9);
  memset(ctx.neuron_vaddr + 32, 0, 4);  /* chl_quan: rshift=0 */
  CVI_RT_MemFlush(ctx.rt_handle, ctx.neuron_mem);

  /* Test 1: flat weight (1,1,9,1) + chl_quan loaded */
  fprintf(stderr, "--- Test1: w=(1,1,9,1) chl_quan=loaded ---\n");
  {
    cvk_tl_t *tl_if = cvk->ops->lmem_alloc_tensor(cvk, (cvk_tl_shape_t){1,1,3,3}, CVK_FMT_I8, 1);
    cvk_tl_t *tl_of = cvk->ops->lmem_alloc_tensor(cvk, (cvk_tl_shape_t){1,1,1,1}, CVK_FMT_I8, 1);
    cvk_tl_t *tl_w  = cvk->ops->lmem_alloc_tensor(cvk, (cvk_tl_shape_t){1,1,9,1}, CVK_FMT_I8, 1);
    tl_w->stride.n = 1; tl_w->cmprs_fmt = CVK_FMT_I8;
    cvk_tl_t *tl_q  = cvk->ops->lmem_alloc_tensor(cvk, (cvk_tl_shape_t){1,1,1,1}, CVK_FMT_I8, 1);

    if (!tl_if||!tl_of||!tl_w||!tl_q) { fprintf(stderr,"alloc fail\n"); return 1; }

    cvk_tg_shape_t g_if = {1,1,3,3}, g_w = {1,1,9,1}, g_of = {1,1,1,1}, g_q = {1,1,1,1};

    cvk->ops->tdma_g2l_tensor_copy(cvk, &(cvk_tdma_g2l_tensor_copy_param_t){
      &(cvk_tg_t){0, TPU_PA(&ctx,0),  CVK_FMT_I8, g_if, cvk->ops->tg_default_stride(cvk, g_if, CVK_FMT_I8)}, tl_if});
    cvk->ops->tdma_g2l_tensor_copy(cvk, &(cvk_tdma_g2l_tensor_copy_param_t){
      &(cvk_tg_t){0, TPU_PA(&ctx,16), CVK_FMT_I8, g_w,  cvk->ops->tg_default_stride(cvk, g_w,  CVK_FMT_I8)}, tl_w});
    cvk->ops->tdma_g2l_tensor_copy(cvk, &(cvk_tdma_g2l_tensor_copy_param_t){
      &(cvk_tg_t){0, TPU_PA(&ctx,32), CVK_FMT_I8, g_q,  cvk->ops->tg_default_stride(cvk, g_q,  CVK_FMT_I8)}, tl_q});

    fprintf(stderr, "  calling tiu_convolution...\n");
    cvk->ops->tiu_convolution(cvk, &(cvk_tiu_convolution_param_t){
      .ofmap = tl_of, .ifmap = tl_if, .weight = tl_w, .chl_quan_param = tl_q,
      .ins_h = 0, .ins_last_h = 0, .ins_w = 0, .ins_last_w = 0,
      .pad_top = 0, .pad_bottom = 0, .pad_left = 0, .pad_right = 0,
      .stride_h = 1, .stride_w = 1, .dilation_h = 1, .dilation_w = 1,
      .has_bias = 0, .relu_enable = 0, .ps32_mode = 0, .w_is_const = 0,
    });
    fprintf(stderr, "  tiu_conv OK\n");

    cvk->ops->tdma_l2g_tensor_copy(cvk, &(cvk_tdma_l2g_tensor_copy_param_t){tl_of,
      &(cvk_tg_t){0, TPU_PA(&ctx,48), CVK_FMT_I8, g_of, cvk->ops->tg_default_stride(cvk, g_of, CVK_FMT_I8)}});

    int64_t t_ns = tpu_submit(&ctx);
    fprintf(stderr, "  submit rc=OK time=%.1f us\n", t_ns/1000.0);

    CVI_RT_MemInvld(ctx.rt_handle, ctx.neuron_mem);
    int8_t out = ctx.neuron_vaddr[48];
    fprintf(stderr, "  result: got=%d exp=%d %s\n", out, expected, out==expected?"OK":"MISMATCH");

    cvk->ops->lmem_free_tensor(cvk, tl_q);
    cvk->ops->lmem_free_tensor(cvk, tl_w);
    cvk->ops->lmem_free_tensor(cvk, tl_of);
    cvk->ops->lmem_free_tensor(cvk, tl_if);
  }

  /* Test 2: 4D weight (1,1,3,3) + chl_quan loaded */
  fprintf(stderr, "--- Test2: w=(1,1,3,3) chl_quan=loaded ---\n");
  {
    cvk_tl_t *tl_if = cvk->ops->lmem_alloc_tensor(cvk, (cvk_tl_shape_t){1,1,3,3}, CVK_FMT_I8, 1);
    cvk_tl_t *tl_of = cvk->ops->lmem_alloc_tensor(cvk, (cvk_tl_shape_t){1,1,1,1}, CVK_FMT_I8, 1);
    cvk_tl_t *tl_w  = cvk->ops->lmem_alloc_tensor(cvk, (cvk_tl_shape_t){1,1,3,3}, CVK_FMT_I8, 1);
    tl_w->stride.n = 1; tl_w->cmprs_fmt = CVK_FMT_I8;
    cvk_tl_t *tl_q  = cvk->ops->lmem_alloc_tensor(cvk, (cvk_tl_shape_t){1,1,1,1}, CVK_FMT_I8, 1);

    if (!tl_if||!tl_of||!tl_w||!tl_q) { fprintf(stderr,"alloc fail\n"); return 1; }

    cvk_tg_shape_t g_if = {1,1,3,3}, g_w = {1,1,3,3}, g_of = {1,1,1,1}, g_q = {1,1,1,1};

    cvk->ops->tdma_g2l_tensor_copy(cvk, &(cvk_tdma_g2l_tensor_copy_param_t){
      &(cvk_tg_t){0, TPU_PA(&ctx,0),  CVK_FMT_I8, g_if, cvk->ops->tg_default_stride(cvk, g_if, CVK_FMT_I8)}, tl_if});
    cvk->ops->tdma_g2l_tensor_copy(cvk, &(cvk_tdma_g2l_tensor_copy_param_t){
      &(cvk_tg_t){0, TPU_PA(&ctx,16), CVK_FMT_I8, g_w,  cvk->ops->tg_default_stride(cvk, g_w,  CVK_FMT_I8)}, tl_w});
    cvk->ops->tdma_g2l_tensor_copy(cvk, &(cvk_tdma_g2l_tensor_copy_param_t){
      &(cvk_tg_t){0, TPU_PA(&ctx,32), CVK_FMT_I8, g_q,  cvk->ops->tg_default_stride(cvk, g_q,  CVK_FMT_I8)}, tl_q});

    fprintf(stderr, "  calling tiu_convolution...\n");
    cvk->ops->tiu_convolution(cvk, &(cvk_tiu_convolution_param_t){
      .ofmap = tl_of, .ifmap = tl_if, .weight = tl_w, .chl_quan_param = tl_q,
      .ins_h = 0, .ins_last_h = 0, .ins_w = 0, .ins_last_w = 0,
      .pad_top = 0, .pad_bottom = 0, .pad_left = 0, .pad_right = 0,
      .stride_h = 1, .stride_w = 1, .dilation_h = 1, .dilation_w = 1,
      .has_bias = 0, .relu_enable = 0, .ps32_mode = 0, .w_is_const = 0,
    });
    fprintf(stderr, "  tiu_conv OK\n");

    cvk->ops->tdma_l2g_tensor_copy(cvk, &(cvk_tdma_l2g_tensor_copy_param_t){tl_of,
      &(cvk_tg_t){0, TPU_PA(&ctx,48), CVK_FMT_I8, g_of, cvk->ops->tg_default_stride(cvk, g_of, CVK_FMT_I8)}});

    int64_t t_ns = tpu_submit(&ctx);
    fprintf(stderr, "  submit rc=OK time=%.1f us\n", t_ns/1000.0);

    CVI_RT_MemInvld(ctx.rt_handle, ctx.neuron_mem);
    int8_t out = ctx.neuron_vaddr[48];
    fprintf(stderr, "  result: got=%d exp=%d %s\n", out, expected, out==expected?"OK":"MISMATCH");

    cvk->ops->lmem_free_tensor(cvk, tl_q);
    cvk->ops->lmem_free_tensor(cvk, tl_w);
    cvk->ops->lmem_free_tensor(cvk, tl_of);
    cvk->ops->lmem_free_tensor(cvk, tl_if);
  }

  fprintf(stderr, "=== DONE conv3x3_cvk_v2 ===\n\n");
  tpu_cleanup(&ctx);
  return 0;
}
