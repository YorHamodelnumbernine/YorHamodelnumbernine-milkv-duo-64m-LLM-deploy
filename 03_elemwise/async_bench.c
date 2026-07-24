/* TPU async submit benchmark — compares blocking vs non-blocking execution.
   Uses cvikernel ADD (known reliable) to compare:
   1. CVI_RT_Submit (blocking) — baseline
   2. CVI_RT_SubmitAsync + immediate CVI_RT_WaitForAsync — no CPU work
   3. CVI_RT_SubmitAsync + CPU busywork + CVI_RT_WaitForAsync — overlap */
#include "../common/tpu_bench.h"

#define N (16 * 16)  // 256 elements

int main() {
  static const int8_t in[N];
  static const int8_t exp_lo[N];

  // Fill test data
  int8_t din[N], dexp[N];
  for (int i = 0; i < N; i++) {
    din[i] = (int8_t)(i & 0x7f);
    dexp[i] = din[i] + 3;  // add 3
  }

  tpu_ctx ctx;
  if (tpu_init(&ctx, 4096) != 0) return 1;
  cvk_context_t *cvk = ctx.cvk_ctx;

  printf("  [async_bench] cvikernel ADD (%d elements)\n", N);
  printf("  %-20s %8s %8s %8s %8s\n", "method", "avg_us", "min_us", "max_us", "note");

  // === Method 1: Blocking Submit ===
  {
    int n_runs = 10;
    double sum = 0, sum2 = 0, tmin = 1e9, tmax = 0;
    for (int r = 0; r < n_runs; r++) {
      /* Build cvk commands */
      cvk_tl_shape_t s8 = {1,1,1,N};
      cvk_tl_t *tl_al = cvk->ops->lmem_alloc_tensor(cvk, s8, CVK_FMT_I8, 1);
      cvk_tl_t *tl_ah = cvk->ops->lmem_alloc_tensor(cvk, s8, CVK_FMT_I8, 1);
      cvk_tl_t *tl_rl = cvk->ops->lmem_alloc_tensor(cvk, s8, CVK_FMT_I8, 1);
      cvk_tl_t *tl_rh = cvk->ops->lmem_alloc_tensor(cvk, s8, CVK_FMT_I8, 1);

      memcpy(ctx.neuron_vaddr, din, N);
      memset(ctx.neuron_vaddr + 256, 0, N);
      CVI_RT_MemFlush(ctx.rt_handle, ctx.neuron_mem);

      cvk_tg_shape_t gs = {1,1,1,N};
      cvk_tg_stride_t st = cvk->ops->tg_default_stride(cvk, gs, CVK_FMT_I8);
      cvk_tg_t tg_al = {0, TPU_PA(&ctx,0),  CVK_FMT_I8, gs, st};
      cvk_tg_t tg_ah = {0, TPU_PA(&ctx,256), CVK_FMT_I8, gs, st};

      cvk_tdma_g2l_tensor_copy_param_t g2l_al = {&tg_al, tl_al};
      cvk_tdma_g2l_tensor_copy_param_t g2l_ah = {&tg_ah, tl_ah};
      cvk->ops->tdma_g2l_tensor_copy(cvk, &g2l_al);
      cvk->ops->tdma_g2l_tensor_copy(cvk, &g2l_ah);

      cvk_tiu_add_param_t p = {
        .res_high = tl_rh, .res_low = tl_rl,
        .a_high   = tl_ah, .a_low   = tl_al,
        .b_is_const = 1, .b_const = { .val = 3, .is_signed = 1 },
        .rshift_bits = 0, .relu_enable = 0,
      };
      cvk->ops->tiu_add(cvk, &p);

      cvk_tg_t tg_rl = {0, TPU_PA(&ctx,512), CVK_FMT_I8, gs, st};
      cvk_tg_t tg_rh = {0, TPU_PA(&ctx,768), CVK_FMT_I8, gs, st};
      cvk_tdma_l2g_tensor_copy_param_t l2g_l = {tl_rl, &tg_rl};
      cvk_tdma_l2g_tensor_copy_param_t l2g_h = {tl_rh, &tg_rh};
      cvk->ops->tdma_l2g_tensor_copy(cvk, &l2g_l);
      cvk->ops->tdma_l2g_tensor_copy(cvk, &l2g_h);

      struct timespec t0, t1;
      clock_gettime(CLOCK_MONOTONIC, &t0);
      int rc = CVI_RT_Submit(ctx.rt_khandle);
      clock_gettime(CLOCK_MONOTONIC, &t1);
      if (rc != 0) { printf("  Submit fail rc=%d\n", rc); goto cleanup; }
      double t = ((t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec)) / 1000.0;

      sum += t; sum2 += t*t;
      if (t < tmin) tmin = t;
      if (t > tmax) tmax = t;

      // Verify
      CVI_RT_MemInvld(ctx.rt_handle, ctx.neuron_mem);
      int8_t *v = (int8_t*)ctx.neuron_vaddr;
      int ok = 1;
      for (int i = 0; i < N && ok; i++)
        if (v[512+i] != dexp[i]) ok = 0;
      if (!ok) printf("  WARN: run %d mismatch\n", r);

      cvk->ops->lmem_free_tensor(cvk, tl_rh);
      cvk->ops->lmem_free_tensor(cvk, tl_rl);
      cvk->ops->lmem_free_tensor(cvk, tl_ah);
      cvk->ops->lmem_free_tensor(cvk, tl_al);
    }
    double avg = sum / n_runs;
    double std = sqrt(sum2/n_runs - avg*avg);
    printf("  %-20s %8.2f %8.2f %8.2f   block-wait\n", "Submit(blocking)", avg, tmin, tmax);
  }

  // === Method 2: SubmitAsync + immediate Wait ===
  {
    int n_runs = 10;
    double total_sum = 0, async_sum = 0, wait_sum = 0;
    double total_min = 1e9;
    for (int r = 0; r < n_runs; r++) {
      cvk_tl_shape_t s8 = {1,1,1,N};
      cvk_tl_t *tl_al = cvk->ops->lmem_alloc_tensor(cvk, s8, CVK_FMT_I8, 1);
      cvk_tl_t *tl_ah = cvk->ops->lmem_alloc_tensor(cvk, s8, CVK_FMT_I8, 1);
      cvk_tl_t *tl_rl = cvk->ops->lmem_alloc_tensor(cvk, s8, CVK_FMT_I8, 1);
      cvk_tl_t *tl_rh = cvk->ops->lmem_alloc_tensor(cvk, s8, CVK_FMT_I8, 1);

      memcpy(ctx.neuron_vaddr, din, N);
      memset(ctx.neuron_vaddr + 256, 0, N);
      CVI_RT_MemFlush(ctx.rt_handle, ctx.neuron_mem);

      cvk_tg_shape_t gs = {1,1,1,N};
      cvk_tg_stride_t st = cvk->ops->tg_default_stride(cvk, gs, CVK_FMT_I8);
      cvk_tg_t tg_al = {0, TPU_PA(&ctx,0),  CVK_FMT_I8, gs, st};
      cvk_tg_t tg_ah = {0, TPU_PA(&ctx,256), CVK_FMT_I8, gs, st};

      cvk_tdma_g2l_tensor_copy_param_t g2l_al = {&tg_al, tl_al};
      cvk_tdma_g2l_tensor_copy_param_t g2l_ah = {&tg_ah, tl_ah};
      cvk->ops->tdma_g2l_tensor_copy(cvk, &g2l_al);
      cvk->ops->tdma_g2l_tensor_copy(cvk, &g2l_ah);

      cvk_tiu_add_param_t p = {
        .res_high = tl_rh, .res_low = tl_rl,
        .a_high   = tl_ah, .a_low   = tl_al,
        .b_is_const = 1, .b_const = { .val = 3, .is_signed = 1 },
        .rshift_bits = 0, .relu_enable = 0,
      };
      cvk->ops->tiu_add(cvk, &p);

      cvk_tg_t tg_rl = {0, TPU_PA(&ctx,512), CVK_FMT_I8, gs, st};
      cvk_tg_t tg_rh = {0, TPU_PA(&ctx,768), CVK_FMT_I8, gs, st};
      cvk_tdma_l2g_tensor_copy_param_t l2g_l = {tl_rl, &tg_rl};
      cvk_tdma_l2g_tensor_copy_param_t l2g_h = {tl_rh, &tg_rh};
      cvk->ops->tdma_l2g_tensor_copy(cvk, &l2g_l);
      cvk->ops->tdma_l2g_tensor_copy(cvk, &l2g_h);

      struct timespec t0, t1, t2;
      clock_gettime(CLOCK_MONOTONIC, &t0);
      int rc = CVI_RT_SubmitAsync(ctx.rt_khandle, 0);
      clock_gettime(CLOCK_MONOTONIC, &t1);
      if (rc != 0) { printf("  SubmitAsync fail rc=%d\n", rc); goto cleanup; }

      // No work between async submit and wait
      rc = CVI_RT_WaitForAsync(ctx.rt_khandle);
      clock_gettime(CLOCK_MONOTONIC, &t2);
      if (rc != 0) { printf("  WaitForAsync fail rc=%d\n", rc); goto cleanup; }

      double t_async = ((t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec)) / 1000.0;
      double t_wait  = ((t2.tv_sec - t1.tv_sec) * 1e9 + (t2.tv_nsec - t1.tv_nsec)) / 1000.0;
      double t_total = t_async + t_wait;
      total_sum += t_total; async_sum += t_async; wait_sum += t_wait;
      if (t_total < total_min) total_min = t_total;

      CVI_RT_MemInvld(ctx.rt_handle, ctx.neuron_mem);
      int8_t *v = (int8_t*)ctx.neuron_vaddr;
      int ok = 1;
      for (int i = 0; i < N && ok; i++)
        if (v[512+i] != dexp[i]) ok = 0;
      if (!ok) printf("  WARN: run %d async mismatch\n", r);

      cvk->ops->lmem_free_tensor(cvk, tl_rh);
      cvk->ops->lmem_free_tensor(cvk, tl_rl);
      cvk->ops->lmem_free_tensor(cvk, tl_ah);
      cvk->ops->lmem_free_tensor(cvk, tl_al);
    }
    double avg_t = total_sum / n_runs;
    double avg_a = async_sum / n_runs;
    double avg_w = wait_sum / n_runs;
    printf("  %-20s %8.2f                submit=%.2f wait=%.2f\n",
      "Async+immed.wait", avg_t, avg_a, avg_w);
  }

  // === Method 3: SubmitAsync + CPU busywork + Wait ===
  {
    int n_runs = 10;
    double total_sum = 0, async_sum = 0, work_sum = 0, wait_sum = 0;
    for (int r = 0; r < n_runs; r++) {
      cvk_tl_shape_t s8 = {1,1,1,N};
      cvk_tl_t *tl_al = cvk->ops->lmem_alloc_tensor(cvk, s8, CVK_FMT_I8, 1);
      cvk_tl_t *tl_ah = cvk->ops->lmem_alloc_tensor(cvk, s8, CVK_FMT_I8, 1);
      cvk_tl_t *tl_rl = cvk->ops->lmem_alloc_tensor(cvk, s8, CVK_FMT_I8, 1);
      cvk_tl_t *tl_rh = cvk->ops->lmem_alloc_tensor(cvk, s8, CVK_FMT_I8, 1);

      memcpy(ctx.neuron_vaddr, din, N);
      memset(ctx.neuron_vaddr + 256, 0, N);
      CVI_RT_MemFlush(ctx.rt_handle, ctx.neuron_mem);

      cvk_tg_shape_t gs = {1,1,1,N};
      cvk_tg_stride_t st = cvk->ops->tg_default_stride(cvk, gs, CVK_FMT_I8);
      cvk_tg_t tg_al = {0, TPU_PA(&ctx,0),  CVK_FMT_I8, gs, st};
      cvk_tg_t tg_ah = {0, TPU_PA(&ctx,256), CVK_FMT_I8, gs, st};

      cvk_tdma_g2l_tensor_copy_param_t g2l_al = {&tg_al, tl_al};
      cvk_tdma_g2l_tensor_copy_param_t g2l_ah = {&tg_ah, tl_ah};
      cvk->ops->tdma_g2l_tensor_copy(cvk, &g2l_al);
      cvk->ops->tdma_g2l_tensor_copy(cvk, &g2l_ah);

      cvk_tiu_add_param_t p = {
        .res_high = tl_rh, .res_low = tl_rl,
        .a_high   = tl_ah, .a_low   = tl_al,
        .b_is_const = 1, .b_const = { .val = 3, .is_signed = 1 },
        .rshift_bits = 0, .relu_enable = 0,
      };
      cvk->ops->tiu_add(cvk, &p);

      cvk_tg_t tg_rl = {0, TPU_PA(&ctx,512), CVK_FMT_I8, gs, st};
      cvk_tg_t tg_rh = {0, TPU_PA(&ctx,768), CVK_FMT_I8, gs, st};
      cvk_tdma_l2g_tensor_copy_param_t l2g_l = {tl_rl, &tg_rl};
      cvk_tdma_l2g_tensor_copy_param_t l2g_h = {tl_rh, &tg_rh};
      cvk->ops->tdma_l2g_tensor_copy(cvk, &l2g_l);
      cvk->ops->tdma_l2g_tensor_copy(cvk, &l2g_h);

      struct timespec t0, t1, t2, t3;
      clock_gettime(CLOCK_MONOTONIC, &t0);
      int rc = CVI_RT_SubmitAsync(ctx.rt_khandle, 0);
      clock_gettime(CLOCK_MONOTONIC, &t1);
      if (rc != 0) { printf("  SubmitAsync fail rc=%d\n", rc); goto cleanup; }

      // CPU busywork: compute checksum of a large array (software matmul-like)
      volatile int32_t dummy = 0;
      for (int k = 0; k < 5000; k++) {
        dummy += (din[k % N] * dexp[k % N]) ^ (k & 0xff);
      }

      clock_gettime(CLOCK_MONOTONIC, &t2);
      rc = CVI_RT_WaitForAsync(ctx.rt_khandle);
      clock_gettime(CLOCK_MONOTONIC, &t3);
      if (rc != 0) { printf("  WaitForAsync fail rc=%d\n", rc); goto cleanup; }

      double t_async = ((t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec)) / 1000.0;
      double t_work  = ((t2.tv_sec - t1.tv_sec) * 1e9 + (t2.tv_nsec - t1.tv_nsec)) / 1000.0;
      double t_wait  = ((t3.tv_sec - t2.tv_sec) * 1e9 + (t3.tv_nsec - t2.tv_nsec)) / 1000.0;
      double t_total = t_async + t_work + t_wait;
      total_sum += t_total; async_sum += t_async; work_sum += t_work; wait_sum += t_wait;

      CVI_RT_MemInvld(ctx.rt_handle, ctx.neuron_mem);
      int8_t *v = (int8_t*)ctx.neuron_vaddr;
      int ok = 1;
      for (int i = 0; i < N && ok; i++)
        if (v[512+i] != dexp[i]) ok = 0;
      if (!ok) printf("  WARN: run %d busywork mismatch\n", r);

      cvk->ops->lmem_free_tensor(cvk, tl_rh);
      cvk->ops->lmem_free_tensor(cvk, tl_rl);
      cvk->ops->lmem_free_tensor(cvk, tl_ah);
      cvk->ops->lmem_free_tensor(cvk, tl_al);
    }
    double avg_t = total_sum / n_runs;
    double avg_a = async_sum / n_runs;
    double avg_wk = work_sum / n_runs;
    double avg_w = wait_sum / n_runs;
    printf("  %-20s %8.2f                sub=%.2f work=%.2f wait=%.2f\n",
      "Async+CPU work", avg_t, avg_a, avg_wk, avg_w);
  }

cleanup:
  tpu_cleanup(&ctx);
  return 0;
}
