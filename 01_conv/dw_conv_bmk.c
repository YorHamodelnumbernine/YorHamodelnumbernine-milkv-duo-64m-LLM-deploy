/* TPU depthwise convolution benchmark — bmk1822 API, LoadDmabuf path.
   IC=OC=1, 3x3 kernel on 3x3 input → 1x1 output, INT8.
   Weight: center=1, rest=0.  Expected: ifmap[4] = 5. */
#include "../common/tpu_bench.h"
#include "bmkernel/bm1822/bmkernel_1822.h"

int main() {
  int errs = 0;

  CVI_RT_HANDLE rt;
  if (CVI_RT_Init(&rt) != 0) { fprintf(stderr,"RT_Init fail\n"); return 1; }

  CVI_RT_MEM neuron_mem = CVI_RT_MemAlloc(rt, 4096);
  if (!neuron_mem) { fprintf(stderr,"MemAlloc fail\n"); CVI_RT_DeInit(rt); return 1; }
  uint64_t neuron_pa = CVI_RT_MemGetPAddr(neuron_mem);
  uint8_t *neuron_va = CVI_RT_MemGetVAddr(neuron_mem);
  CVI_RT_SetBaseReg(rt, 0, neuron_pa);

  static const int8_t ifmap[9]   = {1,2,3,4,5,6,7,8,9};
  static const int8_t weight[9]  = {0,0,0, 0,1,0, 0,0,0};
  static const int8_t expected[1] = {5};

  memcpy(neuron_va, ifmap, 9);
  memcpy(neuron_va+64, weight, 9);
  CVI_RT_MemFlush(rt, neuron_mem);

  uint8_t cmdbuf[8192] __attribute__((aligned(16)));
  bmk_info_t info = { .chip_version = 1822, .cmdbuf_size = sizeof(cmdbuf), .cmdbuf = cmdbuf };
  bmk1822_context_t *bmk = bmk1822_register(&info);
  if (!bmk) { fprintf(stderr,"bmk reg fail\n"); goto out_mem; }

  bmk1822_tensor_lmem_t *tl_if = bmk1822_lmem_alloc_tensor(bmk, (bmk1822_tensor_lmem_shape_t){1,1,3,3}, FMT_I8, 1);
  bmk1822_tensor_lmem_t *tl_of = bmk1822_lmem_alloc_tensor(bmk, (bmk1822_tensor_lmem_shape_t){1,1,1,1}, FMT_I8, 1);
  bmk1822_tensor_lmem_t *tl_w  = bmk1822_lmem_alloc_tensor(bmk, (bmk1822_tensor_lmem_shape_t){1,1,3,3}, FMT_I8, 1);
  if (!tl_if||!tl_of||!tl_w) { fprintf(stderr,"alloc fail\n"); goto out_bmk; }
  tl_w->stride.n = 1;
  tl_w->cmprs_fmt = FMT_I8;

  bmk1822_tensor_tgmem_t tg_if = {
    .base_reg_index = 0, .start_address = 0, .fmt = FMT_I8, .shape = {1,1,3,3},
    .stride = bmk1822_tensor_tgmem_default_stride((bmk1822_tensor_tgmem_shape_t){1,1,3,3}, FMT_I8),
  };
  bmk1822_tensor_tgmem_t tg_w = {
    .base_reg_index = 0, .start_address = 64, .fmt = FMT_I8, .shape = {1,1,3,3},
    .stride = bmk1822_tensor_tgmem_default_stride((bmk1822_tensor_tgmem_shape_t){1,1,3,3}, FMT_I8),
  };
  bmk1822_tensor_tgmem_t tg_of = {
    .base_reg_index = 0, .start_address = 128, .fmt = FMT_I8, .shape = {1,1,1,1},
    .stride = bmk1822_tensor_tgmem_default_stride((bmk1822_tensor_tgmem_shape_t){1,1,1,1}, FMT_I8),
  };

  bmk1822_tdma_g2l_tensor_copy(bmk, &(bmk1822_tdma_tg2l_tensor_copy_param_t){&tg_if, tl_if});
  bmk1822_tdma_g2l_tensor_copy(bmk, &(bmk1822_tdma_tg2l_tensor_copy_param_t){&tg_w,  tl_w});

  bmk1822_tiu_depthwise_convolution_param_t p = {
    .ofmap = tl_of, .ifmap = tl_if, .weight = tl_w, .bias = NULL,
    .weight_is_const = 0,
    .ins_h = 0, .ins_last_h = 0, .ins_w = 0, .ins_last_w = 0,
    .dilation_h = 1, .dilation_w = 1,
    .pad_top = 0, .pad_bottom = 0, .pad_left = 0, .pad_right = 0,
    .stride_h = 1, .stride_w = 1,
    .rshift_bits = 0, .relu_enable = 0,
    .cmd_pre_exe_typ = 0, .cmd_pre_exe = 0, .layer_id = 1,
  };
  if (!bmk1822_tiu_depthwise_convolution(bmk, &p)) { fprintf(stderr,"dw_conv rejected\n"); goto out_bmk; }

  bmk1822_tdma_l2g_tensor_copy(bmk, &(bmk1822_tdma_l2tg_tensor_copy_param_t){tl_of, &tg_of});

  uint32_t cmd_sz;
  uint8_t *cmd = bmk1822_acquire_cmdbuf(bmk, &cmd_sz);
  uint32_t psize, pmu_size;
  bmk1822_dmabuf_size(cmd, cmd_sz, &psize, &pmu_size);

  CVI_RT_MEM dmabuf_mem = CVI_RT_MemAlloc(rt, psize + pmu_size);
  if (!dmabuf_mem) { fprintf(stderr,"dmabuf alloc fail\n"); goto out_bmk; }
  uint8_t *dmabuf = CVI_RT_MemGetVAddr(dmabuf_mem);
  bmk1822_dmabuf_convert(cmd, cmd_sz, dmabuf);
  bmk1822_arraybase_set(dmabuf, neuron_pa, 0, 0, 0);
  CVI_RT_MemFlush(rt, dmabuf_mem);

  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);

  CVI_RT_MEM loaded;
  int rc = CVI_RT_LoadDmabuf(rt, dmabuf_mem, psize + pmu_size, neuron_pa, 0, false, &loaded);
  if (rc != 0) { fprintf(stderr,"Load fail rc=%d\n", rc); goto out_dmabuf; }

  CVI_RT_ARRAYBASE arr = { .gaddr_base0 = neuron_pa };
  rc = CVI_RT_RunCmdbufEx(rt, loaded, &arr);
  if (rc != 0) { fprintf(stderr,"Run fail rc=%d\n", rc); goto out_dmabuf; }

  clock_gettime(CLOCK_MONOTONIC, &t1);
  int64_t t_ns = (t1.tv_sec - t0.tv_sec) * 1000000000LL + (t1.tv_nsec - t0.tv_nsec);

  CVI_RT_MemInvld(rt, neuron_mem);
  int8_t *v = (int8_t*)neuron_va;
  errs = (v[128] != expected[0]);
  printf("  [dw_conv_bmk] got=%d exp=%d  %s  time: %.2f us\n",
    v[128], expected[0], errs?"FAIL":"OK", t_ns/1000.0);

out_dmabuf:
  CVI_RT_MemFree(rt, dmabuf_mem);
out_bmk:
  bmk1822_cleanup(bmk);
out_mem:
  CVI_RT_MemFree(rt, neuron_mem);
  CVI_RT_DeInit(rt);
  return errs?1:0;
}
