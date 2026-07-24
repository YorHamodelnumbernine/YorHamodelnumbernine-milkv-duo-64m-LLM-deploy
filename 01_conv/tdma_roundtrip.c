/* TDMA roundtrip test: G2L + L2G to verify weight data transfer with stride_type_2 */
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

  /* Write known weight data */
  static const int8_t weight[9] = {0,1,2, 3,4,5, 6,7,8};
  memcpy(neuron_va+64, weight, 9);
  CVI_RT_MemFlush(rt_handle, neuron_mem);

  uint8_t cmdbuf[8192] __attribute__((aligned(16)));
  bmk_info_t info = { .chip_version = 1822, .cmdbuf_size = sizeof(cmdbuf), .cmdbuf = cmdbuf };
  bmk1822_context_t *bmk = bmk1822_register(&info);
  if (!bmk) { fprintf(stderr,"bmk reg fail\n"); goto out_mem; }

  /* Alloc weight tensor: shape {1,1,9,1} */
  bmk1822_tensor_lmem_t *tl_w = bmk1822_lmem_alloc_tensor(bmk, (bmk1822_tensor_lmem_shape_t){1,1,9,1}, FMT_I8, 1);
  if (!tl_w) { fprintf(stderr,"alloc fail\n"); goto out_bmk; }

  fprintf(stderr,"[diag] w alloc strides:(%u,%u,%u,%u) addr=%x\n",
    tl_w->stride.n, tl_w->stride.c, tl_w->stride.h, tl_w->stride.w, tl_w->start_address);

  /* Apply stride_type_2 to weight BEFORE TDMA */
  tl_w->stride.n = 1;
  tl_w->cmprs_fmt = FMT_I8;
  fprintf(stderr,"[diag] w fixed  strides:(%u,%u,%u,%u) cmprs=%d\n",
    tl_w->stride.n, tl_w->stride.c, tl_w->stride.h, tl_w->stride.w, tl_w->cmprs_fmt);

  bmk1822_tensor_tgmem_t tg_w = {
    .base_reg_index = 0, .start_address = 64, .fmt = FMT_I8,
    .shape = {1,1,9,1},
    .stride = bmk1822_tensor_tgmem_default_stride((bmk1822_tensor_tgmem_shape_t){1,1,9,1}, FMT_I8),
  };
  bmk1822_tensor_tgmem_t tg_out = {
    .base_reg_index = 0, .start_address = 128, .fmt = FMT_I8,
    .shape = {1,1,9,1},
    .stride = bmk1822_tensor_tgmem_default_stride((bmk1822_tensor_tgmem_shape_t){1,1,9,1}, FMT_I8),
  };

  /* G2L: load weight from DDR to LMEM */
  bmk1822_tdma_g2l_tensor_copy(bmk, &(bmk1822_tdma_tg2l_tensor_copy_param_t){&tg_w, tl_w});
  /* L2G: store weight from LMEM back to DDR (different addr) */
  bmk1822_tdma_l2g_tensor_copy(bmk, &(bmk1822_tdma_l2tg_tensor_copy_param_t){tl_w, &tg_out});

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
  if (rc != 0) { fprintf(stderr,"Run fail\n"); goto out_bmk; }

  CVI_RT_MemInvld(rt_handle, neuron_mem);
  int8_t *v = (int8_t*)neuron_va;
  fprintf(stderr,"[dbg] weight DDR(64): ");
  for (int i=0;i<9;i++) fprintf(stderr,"%d,", v[64+i]);
  fprintf(stderr,"\n");
  fprintf(stderr,"[dbg] weight RT (128): ");
  for (int i=0;i<9;i++) fprintf(stderr,"%d,", v[128+i]);
  fprintf(stderr,"\n");

  /* Compare */
  errs = 0;
  for (int i=0;i<9;i++) {
    if (v[64+i] != v[128+i]) {
      fprintf(stderr,"  MISMATCH at %d: orig=%d rt=%d\n", i, v[64+i], v[128+i]);
      errs++;
    }
  }
  if (errs==0) fprintf(stderr,"[OK] TDMA roundtrip ALL 9 bytes match!\n");
  else fprintf(stderr,"[FAIL] %d mismatches\n", errs);

out_bmk:
  bmk1822_cleanup(bmk);
out_mem:
  CVI_RT_MemFree(rt_handle, neuron_mem);
  CVI_RT_DeInit(rt_handle);
  return errs?1:0;
}
