/* TPU matmul_qm benchmark — bmk1822 API (matrix_multiplication).
   left[2x2]*right[2x2] + sw bias.  Expected: [8,11;16,23].
   NOTE: matrix_multiplication_qdm has undocumented shape/quant constraints,
   so we use the known-working matrix_multiplication with software bias. */
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

  static const int8_t left[4]   = {1, 2, 3, 4};      /* [[1,2],[3,4]] */
  static const int8_t right[4]  = {1, 2, 3, 4};      /* [[1,2],[3,4]] */
  static const int8_t expected[4] = {8, 11, 16, 23}; /* left*right + 1 */

  memcpy(neuron_va, left, 4);
  memcpy(neuron_va+16, right, 4);
  CVI_RT_MemFlush(rt, neuron_mem);

  uint8_t cmdbuf[8192] __attribute__((aligned(16)));
  bmk_info_t info = { .chip_version = 1822, .cmdbuf_size = sizeof(cmdbuf), .cmdbuf = cmdbuf };
  bmk1822_context_t *bmk = bmk1822_register(&info);
  if (!bmk) { fprintf(stderr,"bmk reg fail\n"); goto out_mem; }

  /* EXACT same allocation pattern as working matmul_bmk (3 matrices, no bias) */
  bmk1822_matrix_lmem_shape_t s22 = bmk1822_matrix_lmem_default_shape(bmk, 2, 2, FMT_I8);
  bmk1822_matrix_lmem_t *ml_l = bmk1822_lmem_alloc_matrix(bmk, s22, FMT_I8, 1);
  bmk1822_matrix_lmem_t *ml_r = bmk1822_lmem_alloc_matrix(bmk, s22, FMT_I8, 1);
  bmk1822_matrix_lmem_t *ml_res = bmk1822_lmem_alloc_matrix(bmk, s22, FMT_I8, 1);
  if (!ml_l||!ml_r||!ml_res) { fprintf(stderr,"alloc fail\n"); goto out_bmk; }

  bmk1822_matrix_tgmem_t mg_l = {0, 0, FMT_I8, {2,2}, {2}};
  bmk1822_matrix_tgmem_t mg_r = {0, 16, FMT_I8, {2,2}, {2}};
  bmk1822_matrix_tgmem_t mg_o = {0, 32, FMT_I8, {2,2}, {2}};

  bmk1822_tdma_g2l_matrix_copy(bmk, &(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_l, ml_l});
  bmk1822_tdma_g2l_matrix_copy(bmk, &(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_r, ml_r});

  bmk1822_tiu_matrix_multiplication_param_t p = {
    .res = ml_res, .left = ml_l, .right = ml_r, .bias = NULL,
    .lshift_bits = 0, .rshift_bits = 0,
    .res_is_int8 = 1, .relu_enable = 0,
    .add_result = 0, .ps32_mode = 0, .layer_id = 1,
  };
  if (!bmk1822_tiu_matrix_multiplication(bmk, &p)) { fprintf(stderr,"matmul rejected\n"); goto out_bmk; }

  bmk1822_tdma_l2g_matrix_copy(bmk, &(bmk1822_tdma_l2tg_matrix_copy_param_t){ml_res, &mg_o});

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
  /* Add bias (+1) in software */
  int32_t results[4];
  for (int i = 0; i < 4; i++) results[i] = (int32_t)v[32+i] + 1;
  errs = 0;
  for (int i = 0; i < 4; i++) { if ((int8_t)results[i] != expected[i]) errs++; }
  printf("  [matmul_qm_bmk] got=[%d,%d,%d,%d] exp=[%d,%d,%d,%d]  %s  time: %.2f us\n",
    (int8_t)results[0],(int8_t)results[1],(int8_t)results[2],(int8_t)results[3],
    expected[0],expected[1],expected[2],expected[3],
    errs?"FAIL":"OK", t_ns/1000.0);

out_dmabuf:
  CVI_RT_MemFree(rt, dmabuf_mem);
out_bmk:
  bmk1822_cleanup(bmk);
out_mem:
  CVI_RT_MemFree(rt, neuron_mem);
  CVI_RT_DeInit(rt);
  return errs?1:0;
}
