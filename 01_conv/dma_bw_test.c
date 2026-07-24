/* TDMA bandwidth benchmark — single-shot pattern (matching debug version that got 235us).
   Each measurement: fresh bmk+dmabuf, ONE LoadDmabuf+RunCmdbufEx, NO warmup. */
#include "../common/tpu_bench.h"
#include "bmkernel/bm1822/bmkernel_1822.h"
#include <math.h>

#define N_RUNS 8

static double do_one_submit(CVI_RT_HANDLE rt, uint64_t neuron_pa, int with_tiu) {
  uint8_t cmdbuf[8192] __attribute__((aligned(16)));
  bmk_info_t info = { .chip_version = 1822, .cmdbuf_size = sizeof(cmdbuf), .cmdbuf = cmdbuf };
  bmk1822_context_t *bmk = bmk1822_register(&info);
  if (!bmk) return -1;

  bmk1822_tensor_lmem_t *tl_if = bmk1822_lmem_alloc_tensor(bmk, (bmk1822_tensor_lmem_shape_t){1,1,64,64}, FMT_I8, 1);
  bmk1822_tensor_lmem_t *tl_of = bmk1822_lmem_alloc_tensor(bmk, (bmk1822_tensor_lmem_shape_t){1,1,62,62}, FMT_I8, 1);
  bmk1822_tensor_lmem_t *tl_w  = bmk1822_lmem_alloc_tensor(bmk, (bmk1822_tensor_lmem_shape_t){1,1,3,3}, FMT_I8, 1);
  if (!tl_if||!tl_of||!tl_w) { bmk1822_cleanup(bmk); return -1; }
  tl_w->stride.n = 1; tl_w->cmprs_fmt = FMT_I8;

  bmk1822_tensor_tgmem_t tg_if = {
    .base_reg_index = 0, .start_address = 0, .fmt = FMT_I8, .shape = {1,1,64,64},
    .stride = bmk1822_tensor_tgmem_default_stride((bmk1822_tensor_tgmem_shape_t){1,1,64,64}, FMT_I8),
  };
  bmk1822_tdma_g2l_tensor_copy(bmk, &(bmk1822_tdma_tg2l_tensor_copy_param_t){&tg_if, tl_if});

  bmk1822_tensor_tgmem_t tg_w = {
    .base_reg_index = 0, .start_address = 4096, .fmt = FMT_I8, .shape = {1,1,3,3},
    .stride = bmk1822_tensor_tgmem_default_stride((bmk1822_tensor_tgmem_shape_t){1,1,3,3}, FMT_I8),
  };
  bmk1822_tdma_g2l_tensor_copy(bmk, &(bmk1822_tdma_tg2l_tensor_copy_param_t){&tg_w, tl_w});

  if (with_tiu) {
    bmk1822_tiu_depthwise_convolution_param_t p = {
      .ofmap = tl_of, .ifmap = tl_if, .weight = tl_w, .bias = NULL,
      .weight_is_const = 0,
      .ins_h = 0, .ins_last_h = 0, .ins_w = 0, .ins_last_w = 0,
      .pad_top = 0, .pad_bottom = 0, .pad_left = 0, .pad_right = 0,
      .stride_h = 1, .stride_w = 1,
      .dilation_h = 1, .dilation_w = 1,
      .relu_enable = 0, .rshift_bits = 0,
      .cmd_pre_exe_typ = 0, .cmd_pre_exe = 0, .layer_id = 1,
    };
    bmk1822_tiu_depthwise_convolution(bmk, &p);
  }

  bmk1822_tensor_tgmem_t tg_of = {
    .base_reg_index = 0, .start_address = 8192, .fmt = FMT_I8, .shape = {1,1,62,62},
    .stride = bmk1822_tensor_tgmem_default_stride((bmk1822_tensor_tgmem_shape_t){1,1,62,62}, FMT_I8),
  };
  bmk1822_tdma_l2g_tensor_copy(bmk, &(bmk1822_tdma_l2tg_tensor_copy_param_t){tl_of, &tg_of});

  uint32_t cmd_sz;
  uint8_t *cmd = bmk1822_acquire_cmdbuf(bmk, &cmd_sz);
  if (!cmd) { bmk1822_cleanup(bmk); return -1; }
  uint32_t psize, pmu_size;
  bmk1822_dmabuf_size(cmd, cmd_sz, &psize, &pmu_size);

  CVI_RT_MEM dmabuf_mem = CVI_RT_MemAlloc(rt, psize + pmu_size);
  if (!dmabuf_mem) { bmk1822_cleanup(bmk); return -1; }
  uint8_t *dmabuf = CVI_RT_MemGetVAddr(dmabuf_mem);
  bmk1822_dmabuf_convert(cmd, cmd_sz, dmabuf);
  bmk1822_arraybase_set(dmabuf, neuron_pa, 0, 0, 0);
  CVI_RT_MemFlush(rt, dmabuf_mem);
  bmk1822_cleanup(bmk);

  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);

  CVI_RT_MEM loaded;
  CVI_RT_LoadDmabuf(rt, dmabuf_mem, psize + pmu_size, neuron_pa, 0, false, &loaded);
  CVI_RT_ARRAYBASE arr = {.gaddr_base0 = neuron_pa};
  CVI_RT_RunCmdbufEx(rt, loaded, &arr);

  clock_gettime(CLOCK_MONOTONIC, &t1);
  double t = ((t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec)) / 1000.0;

  CVI_RT_MemFree(rt, dmabuf_mem);
  return t;
}

static void print_stats(const char *label, double *t, int n, int tdma_bytes) {
  double sum = 0, sum2 = 0, tmin = 1e9, tmax = 0;
  for (int i = 0; i < n; i++) {
    sum += t[i]; sum2 += t[i]*t[i];
    if (t[i] < tmin) tmin = t[i];
    if (t[i] > tmax) tmax = t[i];
  }
  double avg = sum / n;
  double std = sqrt(sum2/n - avg*avg);
  printf("  %s  (tdma=%dB)\n", label, tdma_bytes);
  printf("    avg=%.2f  min=%.2f  max=%.2f  us   σ=%.2f\n", avg, tmin, tmax, std);
  printf("    effective throughput: %.2f MB/s\n\n", tdma_bytes / avg);
}

int main() {
  CVI_RT_HANDLE rt;
  if (CVI_RT_Init(&rt) != 0) { fprintf(stderr,"RT_Init fail\n"); return 1; }

  CVI_RT_MEM neuron_mem = CVI_RT_MemAlloc(rt, 32768);
  if (!neuron_mem) { fprintf(stderr,"MemAlloc fail\n"); CVI_RT_DeInit(rt); return 1; }
  uint64_t neuron_pa = CVI_RT_MemGetPAddr(neuron_mem);
  uint8_t *neuron_va = CVI_RT_MemGetVAddr(neuron_mem);
  CVI_RT_SetBaseReg(rt, 0, neuron_pa);

  for (int i = 0; i < 64*64; i++) neuron_va[i] = (int8_t)(i & 0xff);
  for (int i = 0; i < 9; i++) neuron_va[4096+i] = (i == 4) ? 1 : 0;
  CVI_RT_MemFlush(rt, neuron_mem);

  printf("  [dma_bw] single-shot, no warmup, %d runs each\n\n", N_RUNS);

  // --- DMA-only ---
  {
    double t_dma[N_RUNS];
    for (int r = 0; r < N_RUNS; r++)
      t_dma[r] = do_one_submit(rt, neuron_pa, 0);
    print_stats("DMA-only     (3×TDMA, no TIU)", t_dma, N_RUNS, 7949);
  }

  // --- Full conv ---
  {
    double t_conv[N_RUNS];
    for (int r = 0; r < N_RUNS; r++)
      t_conv[r] = do_one_submit(rt, neuron_pa, 1);
    print_stats("Full-conv    (DMA + dw_conv)", t_conv, N_RUNS, 7949);
  }

  CVI_RT_MemFree(rt, neuron_mem);
  CVI_RT_DeInit(rt);
  return 0;
}
