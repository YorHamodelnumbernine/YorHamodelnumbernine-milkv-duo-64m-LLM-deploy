/* TPU lookup_table benchmark — bmk1822 API.
   Table: f(x)=x*2+1. 16-element table, 2 NPU copies. */
#include "../common/tpu_bench.h"
#include "bmkernel/bm1822/bmkernel_1822.h"

#define N 16
#define NPU_NUM 8   /* BM1822_HW_NPU_NUM */
#define EU_NUM  16  /* BM1822_HW_EU_NUM */
#define TBL_BYTES (NPU_NUM * EU_NUM * N)
#define TBL_OFF   16
#define OUT_OFF   (TBL_OFF + TBL_BYTES)

int main() {
  int errs = 0;

  CVI_RT_HANDLE rt;
  if (CVI_RT_Init(&rt) != 0) { fprintf(stderr,"RT_Init fail\n"); return 1; }

  CVI_RT_MEM neuron_mem = CVI_RT_MemAlloc(rt, 4096);
  if (!neuron_mem) { fprintf(stderr,"MemAlloc fail\n"); CVI_RT_DeInit(rt); return 1; }
  uint64_t neuron_pa = CVI_RT_MemGetPAddr(neuron_mem);
  uint8_t *neuron_va = CVI_RT_MemGetVAddr(neuron_mem);
  CVI_RT_SetBaseReg(rt, 0, neuron_pa);

  static const int8_t idx[N] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
  /* Fill table: shape {1,NPU_NUM,EU_NUM,N} = {1,8,16,16}, all copies identical */
  static int8_t tbl[TBL_BYTES];
  static const int8_t expected[N] = {1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31};

  __builtin_memset(tbl, 0, TBL_BYTES);
  for (int c = 0; c < NPU_NUM; c++)
    for (int h = 0; h < EU_NUM; h++)
      for (int i = 0; i < N; i++)
        tbl[c*EU_NUM*N + h*N + i] = (int8_t)(i*2+1);

  memcpy(neuron_va, idx, N);
  memcpy(neuron_va + TBL_OFF, tbl, TBL_BYTES);
  CVI_RT_MemFlush(rt, neuron_mem);

  uint8_t cmdbuf[8192] __attribute__((aligned(16)));
  bmk_info_t info = { .chip_version = 1822, .cmdbuf_size = sizeof(cmdbuf), .cmdbuf = cmdbuf };
  bmk1822_context_t *bmk = bmk1822_register(&info);
  if (!bmk) { fprintf(stderr,"bmk reg fail\n"); goto out_mem; }

  bmk1822_tensor_lmem_t *tl_if  = bmk1822_lmem_alloc_tensor(bmk, (bmk1822_tensor_lmem_shape_t){1,1,1,N}, FMT_I8, 1);
  bmk1822_tensor_lmem_t *tl_tbl = bmk1822_lmem_alloc_tensor(bmk, (bmk1822_tensor_lmem_shape_t){1,NPU_NUM,EU_NUM,N}, FMT_I8, 1);
  bmk1822_tensor_lmem_t *tl_of  = bmk1822_lmem_alloc_tensor(bmk, (bmk1822_tensor_lmem_shape_t){1,1,1,N}, FMT_I8, 1);
  if (!tl_if||!tl_tbl||!tl_of) { fprintf(stderr,"alloc fail\n"); goto out_bmk; }

  bmk1822_tensor_tgmem_t tg_if = {
    .base_reg_index = 0, .start_address = 0, .fmt = FMT_I8, .shape = {1,1,1,N},
    .stride = bmk1822_tensor_tgmem_default_stride((bmk1822_tensor_tgmem_shape_t){1,1,1,N}, FMT_I8),
  };
  bmk1822_tensor_tgmem_t tg_tbl = {
    .base_reg_index = 0, .start_address = TBL_OFF, .fmt = FMT_I8, .shape = {1,NPU_NUM,EU_NUM,N},
    .stride = bmk1822_tensor_tgmem_default_stride((bmk1822_tensor_tgmem_shape_t){1,NPU_NUM,EU_NUM,N}, FMT_I8),
  };
  bmk1822_tensor_tgmem_t tg_of = {
    .base_reg_index = 0, .start_address = OUT_OFF, .fmt = FMT_I8, .shape = {1,1,1,N},
    .stride = bmk1822_tensor_tgmem_default_stride((bmk1822_tensor_tgmem_shape_t){1,1,1,N}, FMT_I8),
  };

  bmk1822_tdma_g2l_tensor_copy(bmk, &(bmk1822_tdma_tg2l_tensor_copy_param_t){&tg_if, tl_if});
  bmk1822_tdma_g2l_tensor_copy(bmk, &(bmk1822_tdma_tg2l_tensor_copy_param_t){&tg_tbl, tl_tbl});

  if (!bmk1822_tiu_lookup_table(bmk, &(bmk1822_tiu_lookup_table_param_t){tl_of, tl_if, tl_tbl, 1})) {
    fprintf(stderr,"lookup rejected\n"); goto out_bmk;
  }

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
  errs = 0;
  for (int i = 0; i < N; i++) { if (v[OUT_OFF+i] != expected[i]) errs++; }
  printf("  [lookup_table_bmk] %s  time: %.2f us\n", errs?"FAIL":"OK", t_ns/1000.0);

out_dmabuf:
  CVI_RT_MemFree(rt, dmabuf_mem);
out_bmk:
  bmk1822_cleanup(bmk);
out_mem:
  CVI_RT_MemFree(rt, neuron_mem);
  CVI_RT_DeInit(rt);
  return errs?1:0;
}
