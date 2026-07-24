/* TDMA-only test: bmk1822 g2l then l2g to verify dmabuf address mapping works */
#include "../common/tpu_bench.h"
#include "bmkernel/bm1822/bmkernel_1822.h"

int main() {
  int errs = 0;
  /* Test data: copy 16 bytes from src offset to dst offset */
  static const int8_t src[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
  static const int8_t expected[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};

  CVI_RT_HANDLE rt_handle;
  if (CVI_RT_Init(&rt_handle) != 0) { fprintf(stderr,"RT_Init fail\n"); return 1; }

  CVI_RT_MEM neuron_mem = CVI_RT_MemAlloc(rt_handle, 4096);
  if (!neuron_mem) { fprintf(stderr,"MemAlloc fail\n"); CVI_RT_DeInit(rt_handle); return 1; }
  uint64_t neuron_pa = CVI_RT_MemGetPAddr(neuron_mem);
  uint8_t *neuron_va = CVI_RT_MemGetVAddr(neuron_mem);

  CVI_RT_SetBaseReg(rt_handle, 0, neuron_pa);

  memset(neuron_va, 0, 256);
  memcpy(neuron_va, src, 16);   /* src at +0 */
  CVI_RT_MemFlush(rt_handle, neuron_mem);

  uint8_t cmdbuf[8192] __attribute__((aligned(16)));
  bmk_info_t info = { .chip_version = 1822, .cmdbuf_size = sizeof(cmdbuf), .cmdbuf = cmdbuf };
  bmk1822_context_t *bmk = bmk1822_register(&info);
  if (!bmk) { fprintf(stderr,"bmk reg fail\n"); goto out_mem; }

  bmk1822_tensor_lmem_t *tl = bmk1822_lmem_alloc_tensor(bmk, (bmk1822_tensor_lmem_shape_t){1,1,1,16}, FMT_I8, 1);
  if (!tl) { fprintf(stderr,"alloc fail\n"); goto out_bmk; }

  fprintf(stderr,"[diag] tl addr=0x%x stride=(%u,%u,%u,%u)\n",
    tl->start_address, tl->stride.n, tl->stride.c, tl->stride.h, tl->stride.w);

  /* TG: src at +0, dst at +64 — both offsets from base_reg[0] */
  bmk1822_tensor_tgmem_t tg_src = {
    .base_reg_index = 0, .start_address = 0, .fmt = FMT_I8,
    .shape = {1,1,1,16},
    .stride = bmk1822_tensor_tgmem_default_stride((bmk1822_tensor_tgmem_shape_t){1,1,1,16}, FMT_I8),
  };
  bmk1822_tensor_tgmem_t tg_dst = {
    .base_reg_index = 0, .start_address = 64, .fmt = FMT_I8,
    .shape = {1,1,1,16},
    .stride = bmk1822_tensor_tgmem_default_stride((bmk1822_tensor_tgmem_shape_t){1,1,1,16}, FMT_I8),
  };

  bmk1822_tdma_g2l_tensor_copy(bmk, &(bmk1822_tdma_tg2l_tensor_copy_param_t){&tg_src, tl});
  bmk1822_tdma_l2g_tensor_copy(bmk, &(bmk1822_tdma_l2tg_tensor_copy_param_t){tl, &tg_dst});

  uint32_t cmd_sz;
  uint8_t *cmd = bmk1822_acquire_cmdbuf(bmk, &cmd_sz);
  uint32_t psize, pmu_size;
  bmk1822_dmabuf_size(cmd, cmd_sz, &psize, &pmu_size);
  fprintf(stderr,"[diag] cmd_sz=%u psize=%u pmu=%u\n", cmd_sz, psize, pmu_size);

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
  {
    int8_t *v = (int8_t*)neuron_va;
    fprintf(stderr,"[dbg] src(+0):"); for(int i=0;i<16;i++) fprintf(stderr,"%d,",v[i]);
    fprintf(stderr,"\n");
    fprintf(stderr,"[dbg] dst(+64):"); for(int i=0;i<16;i++) fprintf(stderr,"%d,",v[64+i]);
    fprintf(stderr,"\n");
    errs = tpu_check_i8("tdma_test", v+64, expected, 16, 16);
  }

out_bmk:
  bmk1822_cleanup(bmk);
out_mem:
  CVI_RT_MemFree(rt_handle, neuron_mem);
  CVI_RT_DeInit(rt_handle);
  return errs?1:0;
}
