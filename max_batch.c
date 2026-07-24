/* Upward probe to find exact max TIU ops per submit (256KB cmdbuf).
   Start from known-safe 2048, step by 16 until crash/fail. */
#include "common/tpu_bench.h"

static tpu_ctx g_ctx;

static bool try_n(cvk_context_t *cvk, int n) {
  cvk_tl_shape_t s = {1, 1, 16, 16};
  cvk_tl_t *tl_al = cvk->ops->lmem_alloc_tensor(cvk, s, CVK_FMT_I8, 1);
  cvk_tl_t *tl_ah = cvk->ops->lmem_alloc_tensor(cvk, s, CVK_FMT_I8, 1);
  cvk_tl_t *tl_rl = cvk->ops->lmem_alloc_tensor(cvk, s, CVK_FMT_I8, 1);
  cvk_tl_t *tl_rh = cvk->ops->lmem_alloc_tensor(cvk, s, CVK_FMT_I8, 1);
  if (!tl_al || !tl_ah || !tl_rl || !tl_rh) return false;

  cvk_tg_shape_t gs = {1, 1, 16, 16};
  cvk_tg_stride_t st = cvk->ops->tg_default_stride(cvk, gs, CVK_FMT_I8);
  cvk->ops->tdma_g2l_tensor_copy(cvk, &(cvk_tdma_g2l_tensor_copy_param_t){
      &(cvk_tg_t){0, TPU_PA(&g_ctx, 0), CVK_FMT_I8, gs, st}, tl_al});
  cvk->ops->tdma_g2l_tensor_copy(cvk, &(cvk_tdma_g2l_tensor_copy_param_t){
      &(cvk_tg_t){0, TPU_PA(&g_ctx, 256), CVK_FMT_I8, gs, st}, tl_ah});

  for (int i = 0; i < n; i++) {
    cvk->ops->tiu_add(cvk, &(cvk_tiu_add_param_t){
        .res_high = tl_rh, .res_low = tl_rl, .a_high = tl_ah, .a_low = tl_al,
        .b_is_const = 1, .b_const.val = 5, .b_const.is_signed = 1,
        .rshift_bits = 0, .relu_enable = 0, .layer_id = 1,
    });
  }

  cvk->ops->tdma_l2g_tensor_copy(cvk, &(cvk_tdma_l2g_tensor_copy_param_t){
      tl_rl, &(cvk_tg_t){0, TPU_PA(&g_ctx, 512), CVK_FMT_I8, gs, st}});

  uint32_t sz = 0;
  uint8_t *cmd = cvk->ops->acquire_cmdbuf(cvk, &sz);
  return (cmd != NULL && sz > 0);
}

int main(void) {
  fprintf(stderr, "\n========== Max batch probe (256KB cmdbuf) ==========\n");
  if (tpu_init(&g_ctx, 65536) != 0) return 1;

  int max_n = 0;

  for (int probe = 2181; probe <= 2181; probe += 1) {
    CVI_RT_UnRegisterKernel(g_ctx.rt_khandle);
    g_ctx.rt_khandle = CVI_RT_RegisterKernel(g_ctx.rt_handle, 0x40000);
    if (!g_ctx.rt_khandle) { fprintf(stderr, "  re-register fail at %d\n", probe); break; }
    cvk_context_t *cvk = (cvk_context_t *)g_ctx.rt_khandle;
    if (!cvk || !cvk->ops) { fprintf(stderr, "  cvk null at %d\n", probe); break; }

    if (try_n(cvk, probe)) {
      max_n = probe;
      fprintf(stderr, "  n=%d OK\n", probe); fflush(stderr);
    } else {
      fprintf(stderr, "  n=%d FAIL (cmdbuf full)\n", probe); fflush(stderr);
      break;
    }
  }

  fprintf(stderr, "\n========== Result ==========\n");
  fprintf(stderr, "  Max tiu_add in one submit: %d\n", max_n);
  printf("MAX_BATCH|tiu_add|%d\n", max_n);

  tpu_cleanup(&g_ctx);
  return 0;
}
