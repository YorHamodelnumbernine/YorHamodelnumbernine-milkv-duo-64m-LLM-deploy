/* ps32_probe7.c — realistic shape [M=1, N=896, K=32]. Verify ps32 int32
   partial sums reconstruct correctly and test FMT_I32 matrix l2g readback.
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include "cviruntime_context.h"
#include "bmkernel/bm1822/bmkernel_1822.h"

#define NEURON_SZ 262144
#define OUT_OFF 16384

int main(void) {
  CVI_RT_HANDLE rt;
  CVI_RT_Init(&rt);
  CVI_RT_MEM mem = CVI_RT_MemAlloc(rt, NEURON_SZ);
  uint64_t pa = CVI_RT_MemGetPAddr(mem);
  uint8_t *va = CVI_RT_MemGetVAddr(mem);
  CVI_RT_SetBaseReg(rt, 0, pa);
  memset(va, 0, NEURON_SZ);

  int M = 1, N = 896, K = 32;
  srand(7);
  int8_t *left = (int8_t*)(va + 0);
  int8_t *right = (int8_t*)(va + 2048);
  int32_t *host_ref = (int32_t*)calloc(N, sizeof(int32_t));
  for (int k = 0; k < K; k++) {
    left[k] = (int8_t)(rand() % 200 - 100);
    for (int n = 0; n < N; n++) {
      right[k*N + n] = (int8_t)(rand() % 200 - 100);
      host_ref[n] += (int32_t)left[k] * right[k*N + n];
    }
  }
  CVI_RT_MemFlush(rt, mem);

  uint8_t cmdbuf[65536] __attribute__((aligned(16)));
  bmk_info_t info = { .chip_version = 1822, .cmdbuf_size = sizeof(cmdbuf), .cmdbuf = cmdbuf };
  bmk1822_context_t *bmk = bmk1822_register(&info);
  if (!bmk) { printf("reg fail\n"); return 1; }

  bmk1822_matrix_lmem_shape_t so = bmk1822_matrix_lmem_default_shape(bmk, M, N, FMT_I8);
  bmk1822_matrix_lmem_shape_t s1 = bmk1822_matrix_lmem_default_shape(bmk, M, K, FMT_I8);
  bmk1822_matrix_lmem_shape_t s1r = bmk1822_matrix_lmem_default_shape(bmk, K, N, FMT_I8);

  bmk1822_matrix_lmem_t *ml_l = bmk1822_lmem_alloc_matrix(bmk, s1, FMT_I8, 1);
  bmk1822_matrix_lmem_t *ml_r = bmk1822_lmem_alloc_matrix(bmk, s1r, FMT_I8, 1);
  bmk1822_matrix_lmem_t *ml_res = bmk1822_lmem_alloc_ps32_matrix(bmk, so, FMT_I8, 1);
  uint32_t psz = bmk1822_lmem_ps32_matrix_to_size(bmk, so, FMT_I8, 1);
  printf("res ps32: addr=%u sh{n=%u,c=%u,w=%u,col=%u} str{n=%u,c=%u,h=%u} size=%u\n",
         ml_res->start_address, ml_res->shape.n, ml_res->shape.c, ml_res->shape.w,
         ml_res->shape.col, ml_res->stride.n, ml_res->stride.c, ml_res->stride.h, psz);

  bmk1822_matrix_tgmem_t mg_l = {0, 0, FMT_I8, {M,K}, {(uint32_t)K}};
  bmk1822_matrix_tgmem_t mg_r = {0, 2048, FMT_I8, {K,N}, {(uint32_t)N}};
  bmk1822_tdma_g2l_matrix_copy(bmk, &(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_l, ml_l});
  bmk1822_tdma_g2l_matrix_copy(bmk, &(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_r, ml_r});

  /* zero ps32 region */
  memset(va + 4096, 0, psz);
  CVI_RT_MemFlush(rt, mem);
  bmk1822_tdma_g2l_general_copy(bmk, &(bmk1822_tdma_tg2l_general_copy_param_t){
    .src_base_reg_index = 0, .src_address = 4096, .dst_address = ml_res->start_address,
    .bytes = psz });

  bmk1822_tiu_matrix_multiplication_param_t p = {
    .res = ml_res, .left = ml_l, .right = ml_r, .bias = NULL,
    .lshift_bits = 0, .rshift_bits = 0,
    .res_is_int8 = 1, .relu_enable = 0,
    .add_result = 0, .ps32_mode = 2, .layer_id = 1,
  };
  if (!bmk1822_tiu_matrix_multiplication(bmk, &p)) { printf("matmul REJECTED\n"); return 1; }

  /* readback path A: raw general copy */
  bmk1822_tdma_l2g_general_copy(bmk, &(bmk1822_tdma_l2tg_general_copy_param_t){
    .src_address = ml_res->start_address, .dst_base_reg_index = 0,
    .dst_address = OUT_OFF, .bytes = psz });

  uint32_t cmd_sz; uint8_t *cmd = bmk1822_acquire_cmdbuf(bmk, &cmd_sz);
  uint32_t psize, pmu_size; bmk1822_dmabuf_size(cmd, cmd_sz, &psize, &pmu_size);
  CVI_RT_MEM dmabuf_mem = CVI_RT_MemAlloc(rt, psize + pmu_size);
  uint8_t *dmabuf = CVI_RT_MemGetVAddr(dmabuf_mem);
  bmk1822_dmabuf_convert(cmd, cmd_sz, dmabuf);
  bmk1822_arraybase_set(dmabuf, pa, 0, 0, 0);
  CVI_RT_MemFlush(rt, dmabuf_mem);
  CVI_RT_MEM loaded;
  CVI_RT_LoadDmabuf(rt, dmabuf_mem, psize+pmu_size, pa, 0, false, &loaded);
  struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
  CVI_RT_RunCmdbufEx(rt, loaded, &(CVI_RT_ARRAYBASE){.gaddr_base0=pa});
  clock_gettime(CLOCK_MONOTONIC, &t1);
  double ms = (t1.tv_sec-t0.tv_sec)*1000.0 + (t1.tv_nsec-t0.tv_nsec)/1e6;
  CVI_RT_MemInvld(rt, mem);

  uint8_t *r = (uint8_t*)(va + OUT_OFF);
  printf("submit time for K=32 slice [1,%d] (g2l x2 + matmul + l2g ps32): %.3f ms\n", N, ms);

  /* reconstruct hypothesis A: plane k at (m*4+k)*stride.h, col n at byte n */
  int bad = 0, first_bad = -1;
  for (int n = 0; n < N; n++) {
    int32_t v = 0;
    for (int b = 0; b < 4; b++)
      v |= ((int32_t)r[(0*4+b)*ml_res->stride.h + n]) << (8*b);
    if (v != host_ref[n]) { if (bad < 5) printf("  col %d: got %d exp %d\n", n, v, host_ref[n]); if (first_bad<0) first_bad=n; bad++; }
  }
  printf("hypothesis A (byte-plane at (m*4+k)*stride.h, col at byte n): %s (%d/%d bad)%s\n",
         bad==0?"MATCH":"MISMATCH", bad, N, first_bad>=0?"":"");
  printf("  stride.h=%u\n", ml_res->stride.h);

  /* dump raw first 64 bytes and the bytes around col 0, 8, 16, 64 for inspection */
  printf("  raw[0..63]: ");
  for (int i = 0; i < 64; i++) printf("%02x ", r[i]);
  printf("\n");

  free(host_ref);
  CVI_RT_MemFree(rt, dmabuf_mem);
  bmk1822_cleanup(bmk);
  CVI_RT_MemFree(rt, mem);
  CVI_RT_DeInit(rt);
  return 0;
}
