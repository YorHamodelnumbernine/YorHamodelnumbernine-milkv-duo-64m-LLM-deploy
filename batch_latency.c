/* Measure submit latency vs batch size: does batching reduce per-op time? */
#include "common/tpu_bench.h"

static tpu_ctx g_ctx;

static double bench_n(int n, int rounds) {
  double total = 0;
  for (int r = 0; r < rounds; r++) {
    CVI_RT_UnRegisterKernel(g_ctx.rt_khandle);
    g_ctx.rt_khandle = CVI_RT_RegisterKernel(g_ctx.rt_handle, 0x40000);
    cvk_context_t *cvk = (cvk_context_t *)g_ctx.rt_khandle;

    cvk_tl_shape_t s = {1, 1, 16, 16};
    cvk_tl_t *al = cvk->ops->lmem_alloc_tensor(cvk, s, CVK_FMT_I8, 1);
    cvk_tl_t *ah = cvk->ops->lmem_alloc_tensor(cvk, s, CVK_FMT_I8, 1);
    cvk_tl_t *rl = cvk->ops->lmem_alloc_tensor(cvk, s, CVK_FMT_I8, 1);
    cvk_tl_t *rh = cvk->ops->lmem_alloc_tensor(cvk, s, CVK_FMT_I8, 1);
    cvk_tg_shape_t gs = {1, 1, 16, 16};
    cvk_tg_stride_t st = cvk->ops->tg_default_stride(cvk, gs, CVK_FMT_I8);
    cvk->ops->tdma_g2l_tensor_copy(cvk, &(cvk_tdma_g2l_tensor_copy_param_t){
        &(cvk_tg_t){0, TPU_PA(&g_ctx, 0), CVK_FMT_I8, gs, st}, al});
    cvk->ops->tdma_g2l_tensor_copy(cvk, &(cvk_tdma_g2l_tensor_copy_param_t){
        &(cvk_tg_t){0, TPU_PA(&g_ctx, 256), CVK_FMT_I8, gs, st}, ah});
    for (int i = 0; i < n; i++) {
      cvk->ops->tiu_add(cvk, &(cvk_tiu_add_param_t){
          .res_high = rh, .res_low = rl, .a_high = ah, .a_low = al,
          .b_is_const = 1, .b_const.val = 5, .b_const.is_signed = 1,
          .rshift_bits = 0, .relu_enable = 0, .layer_id = 1,
      });
    }
    cvk->ops->tdma_l2g_tensor_copy(cvk, &(cvk_tdma_l2g_tensor_copy_param_t){
        rl, &(cvk_tg_t){0, TPU_PA(&g_ctx, 512), CVK_FMT_I8, gs, st}});

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    CVI_RT_Submit(g_ctx.rt_khandle);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    total += (t1.tv_sec - t0.tv_sec) * 1e6 + (t1.tv_nsec - t0.tv_nsec) / 1e3;
  }
  return total / rounds;
}

int main(void) {
  fprintf(stderr, "\n========== Batch size vs Submit latency ==========\n");
  if (tpu_init(&g_ctx, 65536) != 0) return 1;

  int batches[] = {1, 10, 50, 100, 500, 1000, 1500, 2000, 2181};
  int n_batches = sizeof(batches) / sizeof(batches[0]);

  fprintf(stderr, "%-10s %10s %12s\n", "batch", "avg(us)", "us_per_op");
  fprintf(stderr, "------------------------------------\n");

  for (int i = 0; i < n_batches; i++) {
    int n = batches[i];
    int rounds = (n <= 100) ? 20 : 5; /* fewer rounds for large batches */
    double avg = bench_n(n, rounds);
    fprintf(stderr, "%-10d %10.1f %12.3f\n", n, avg, avg / n);
    printf("BATCH_LATENCY|%d|%.1f|%.3f\n", n, avg, avg / n);
  }

  tpu_cleanup(&g_ctx);
  return 0;
}
