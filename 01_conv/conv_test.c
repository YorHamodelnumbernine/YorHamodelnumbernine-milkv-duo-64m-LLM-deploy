/* Test regular bmk1822 convolution (not depthwise) with IC=OC=1, KH=KW=3 */
#include "../common/tpu_bench.h"
#include "bmkernel/bm1822/bmkernel_1822.h"

int main() {
  int errs = 0;

  CVI_RT_HANDLE rt_handle;
  if (CVI_RT_Init(&rt_handle) != 0) { fprintf(stderr,"RT_Init fail\n"); return 1; }

  CVI_RT_MEM neuron_mem = CVI_RT_MemAlloc(rt_handle, 4096);
  if (!neuron_mem) { fprintf(stderr,"MemAlloc fail\n"); CVI_RT_DeInit(rt_handle); return 1; }
  uint64_t neuron_pa = CVI_RT_MemGetPAddr(neuron_mem);
  uint8_t *neuron_va = CVI_RT_MemGetVAddr(neuron_mem);
  CVI_RT_SetBaseReg(rt_handle, 0, neuron_pa);

  /* Simple 3x3 identity filter conv: ifmap=1..9, weight={0,0,0,0,1,0,0,0,0}, expect=5 (center) */
  static const int8_t ifmap[9] = {1,2,3,4,5,6,7,8,9};
  /* Weight shape: (1, OC, KH*KW, IC) = (1, 1, 9, 1) for IC=OC=1 */
  static const int8_t weight[9] = {0,0,0, 0,1,0, 0,0,0}; /* center=1 */
  static const int8_t expected[1] = {5}; /* center pixel passes through */

  memcpy(neuron_va, ifmap, 9);
  memcpy(neuron_va+64, weight, 9);
  CVI_RT_MemFlush(rt_handle, neuron_mem);

  uint8_t cmdbuf[8192] __attribute__((aligned(16)));
  bmk_info_t info = { .chip_version = 1822, .cmdbuf_size = sizeof(cmdbuf), .cmdbuf = cmdbuf };
  bmk1822_context_t *bmk = bmk1822_register(&info);
  if (!bmk) { fprintf(stderr,"bmk reg fail\n"); goto out_mem; }

  /* ifmap: (1, IC, IH, IW) = (1,1,3,3), ofmap: (1, OC, OH, OW) = (1,1,1,1) */
  bmk1822_tensor_lmem_t *tl_if = bmk1822_lmem_alloc_tensor(bmk, (bmk1822_tensor_lmem_shape_t){1,1,3,3}, FMT_I8, 1);
  bmk1822_tensor_lmem_t *tl_of = bmk1822_lmem_alloc_tensor(bmk, (bmk1822_tensor_lmem_shape_t){1,1,1,1}, FMT_I8, 1);
  bmk1822_tensor_lmem_t *tl_w  = bmk1822_lmem_alloc_tensor(bmk, (bmk1822_tensor_lmem_shape_t){1,1,9,1}, FMT_I8, 1);
  if (!tl_if||!tl_of||!tl_w) { fprintf(stderr,"alloc fail\n"); goto out_bmk; }

  fprintf(stderr,"[diag] strides if:(%u,%u,%u,%u) of:(%u,%u,%u,%u) w:(%u,%u,%u,%u)\n",
    tl_if->stride.n,tl_if->stride.c,tl_if->stride.h,tl_if->stride.w,
    tl_of->stride.n,tl_of->stride.c,tl_of->stride.h,tl_of->stride.w,
    tl_w->stride.n,tl_w->stride.c,tl_w->stride.h,tl_w->stride.w);

  /* Fix weight strides for stride_type_2 */
  tl_w->stride.n = 1;
  tl_w->cmprs_fmt = FMT_I8;

  bmk1822_tensor_tgmem_t tg_if = {
    .base_reg_index = 0, .start_address = 0, .fmt = FMT_I8,
    .shape = {1,1,3,3},
    .stride = bmk1822_tensor_tgmem_default_stride((bmk1822_tensor_tgmem_shape_t){1,1,3,3}, FMT_I8),
  };
  bmk1822_tensor_tgmem_t tg_w = {
    .base_reg_index = 0, .start_address = 64, .fmt = FMT_I8,
    .shape = {1,1,9,1},
    .stride = bmk1822_tensor_tgmem_default_stride((bmk1822_tensor_tgmem_shape_t){1,1,9,1}, FMT_I8),
  };
  bmk1822_tensor_tgmem_t tg_of = {
    .base_reg_index = 0, .start_address = 128, .fmt = FMT_I8,
    .shape = {1,1,1,1},
    .stride = bmk1822_tensor_tgmem_default_stride((bmk1822_tensor_tgmem_shape_t){1,1,1,1}, FMT_I8),
  };

  bmk1822_tdma_g2l_tensor_copy(bmk, &(bmk1822_tdma_tg2l_tensor_copy_param_t){&tg_if, tl_if});
  bmk1822_tdma_g2l_tensor_copy(bmk, &(bmk1822_tdma_tg2l_tensor_copy_param_t){&tg_w,  tl_w});

  fprintf(stderr,"[diag] calling bmk1822_tiu_convolution...\n");
  bmk1822_tiu_convolution_param_t p = {
    .ofmap = tl_of, .ifmap = tl_if, .weight = tl_w, .bias = NULL,
    .ins_h = 0, .ins_last_h = 0, .ins_w = 0, .ins_last_w = 0,
    .dilation_h = 1, .dilation_w = 1,
    .pad_top = 0, .pad_bottom = 0, .pad_left = 0, .pad_right = 0,
    .stride_h = 1, .stride_w = 1,
    .rshift_bits = 0, .relu_enable = 0,
    .ps32_mode = 0, .w_is_const = 0,
  };
  bmk1822_op_t *op = bmk1822_tiu_convolution(bmk, &p);
  fprintf(stderr,"[diag] conv op=%p\n", (void*)op);
  if (!op) { fprintf(stderr,"conv rejected\n"); goto out_bmk; }

  bmk1822_tdma_l2g_tensor_copy(bmk, &(bmk1822_tdma_l2tg_tensor_copy_param_t){tl_of, &tg_of});

  uint32_t cmd_sz;
  uint8_t *cmd = bmk1822_acquire_cmdbuf(bmk, &cmd_sz);
  uint32_t psize, pmu_size;
  bmk1822_dmabuf_size(cmd, cmd_sz, &psize, &pmu_size);

  CVI_RT_MEM dmabuf_mem = CVI_RT_MemAlloc(rt_handle, psize + pmu_size);
  if (!dmabuf_mem) { fprintf(stderr,"dmabuf alloc fail\n"); goto out_bmk; }

  uint8_t *dmabuf = CVI_RT_MemGetVAddr(dmabuf_mem);
  bmk1822_dmabuf_convert(cmd, cmd_sz, dmabuf);
  bmk1822_arraybase_set(dmabuf, neuron_pa, 0, 0, 0);
  CVI_RT_MemFlush(rt_handle, dmabuf_mem);

  CVI_RT_MEM loaded_mem;
  int rc = CVI_RT_LoadDmabuf(rt_handle, dmabuf_mem, psize + pmu_size,
                              neuron_pa, 0, false, &loaded_mem);
  fprintf(stderr,"[diag] LoadDmabuf rc=%d\n", rc);
  if (rc != 0) { fprintf(stderr,"Load fail\n"); goto out_bmk; }

  CVI_RT_ARRAYBASE arr = { .gaddr_base0 = neuron_pa };
  rc = CVI_RT_RunCmdbufEx(rt_handle, loaded_mem, &arr);
  fprintf(stderr,"[diag] RunCmdbufEx rc=%d\n", rc);
  if (rc != 0) {
    rc = CVI_RT_RunCmdbuf(rt_handle, loaded_mem, neuron_pa, 0);
    fprintf(stderr,"[diag] RunCmdbuf rc=%d\n", rc);
  }
  if (rc != 0) { fprintf(stderr,"Run fail\n"); goto out_bmk; }

  CVI_RT_MemInvld(rt_handle, neuron_mem);
  int8_t *v = (int8_t*)neuron_va;
  fprintf(stderr,"[dbg] conv out(at +128):%d (expected 5)\n", v[128]);
  errs = (v[128] != 5);

out_bmk:
  bmk1822_cleanup(bmk);
out_mem:
  CVI_RT_MemFree(rt_handle, neuron_mem);
  CVI_RT_DeInit(rt_handle);
  return errs?1:0;
}
