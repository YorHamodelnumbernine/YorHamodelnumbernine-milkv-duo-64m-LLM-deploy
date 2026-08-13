/* ps32_probe8.c — layout discovery via identity trick.
   left[0]=1, left[k>0]=0; right[0,n]=n. => host_ref[n]=n.
   Then byte0 of column n == n. Scan buffer to map column->byte0 location.
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "cviruntime_context.h"
#include "bmkernel/bm1822/bmkernel_1822.h"

#define NEURON_SZ 262144
#define OUT_OFF 16384

int main(void) {
  CVI_RT_HANDLE rt; CVI_RT_Init(&rt);
  CVI_RT_MEM mem = CVI_RT_MemAlloc(rt, NEURON_SZ);
  uint64_t pa = CVI_RT_MemGetPAddr(mem);
  uint8_t *va = CVI_RT_MemGetVAddr(mem);
  CVI_RT_SetBaseReg(rt, 0, pa);
  memset(va, 0, NEURON_SZ);

  int M = 1, N = 896, K = 32;
  int8_t *left = (int8_t*)(va + 0);
  int8_t *right = (int8_t*)(va + 2048);
  memset(left, 0, K); left[0] = 1;
  for (int k = 0; k < K; k++)
    for (int n = 0; n < N; n++)
      right[k*N + n] = (k==0) ? (int8_t)(n & 0xff) : 0;
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

  memset(va + 4096, 0, psz); CVI_RT_MemFlush(rt, mem);
  bmk1822_tdma_g2l_general_copy(bmk, &(bmk1822_tdma_tg2l_general_copy_param_t){
    .src_base_reg_index = 0, .src_address = 4096, .dst_address = ml_res->start_address, .bytes = psz });

  bmk1822_tiu_matrix_multiplication_param_t p = {
    .res = ml_res, .left = ml_l, .right = ml_r, .bias = NULL,
    .lshift_bits = 0, .rshift_bits = 0, .res_is_int8 = 1, .relu_enable = 0,
    .add_result = 0, .ps32_mode = 2, .layer_id = 1 };
  if (!bmk1822_tiu_matrix_multiplication(bmk, &p)) { printf("matmul REJECTED\n"); return 1; }

  uint32_t rd = psz > 4096 ? psz : 4096;
  bmk1822_tdma_l2g_general_copy(bmk, &(bmk1822_tdma_l2tg_general_copy_param_t){
    .src_address = ml_res->start_address, .dst_base_reg_index = 0,
    .dst_address = OUT_OFF, .bytes = rd });

  uint32_t cmd_sz; uint8_t *cmd = bmk1822_acquire_cmdbuf(bmk, &cmd_sz);
  uint32_t psize, pmu_size; bmk1822_dmabuf_size(cmd, cmd_sz, &psize, &pmu_size);
  CVI_RT_MEM dmabuf_mem = CVI_RT_MemAlloc(rt, psize + pmu_size);
  uint8_t *dmabuf = CVI_RT_MemGetVAddr(dmabuf_mem);
  bmk1822_dmabuf_convert(cmd, cmd_sz, dmabuf);
  bmk1822_arraybase_set(dmabuf, pa, 0, 0, 0);
  CVI_RT_MemFlush(rt, dmabuf_mem);
  CVI_RT_MEM loaded; CVI_RT_LoadDmabuf(rt, dmabuf_mem, psize+pmu_size, pa, 0, false, &loaded);
  CVI_RT_RunCmdbufEx(rt, loaded, &(CVI_RT_ARRAYBASE){.gaddr_base0=pa});
  CVI_RT_MemInvld(rt, mem);

  uint8_t *r = (uint8_t*)(va + OUT_OFF);
  printf("buffer read %u bytes. Expected col n -> int32 n (sign-ext). byte0 of col n = n&0xff.\n", rd);
  printf("  raw first 448 bytes as rows of 8:\n");
  for (uint32_t i = 0; i < 448; i++) {
    if (i%8==0) printf("  %03u:", i);
    printf(" %02x", r[i]);
    if (i%8==7) printf("\n");
  }
  printf("\n  byte0 value at each offset in first 448 bytes (nonzero):\n");
  for (uint32_t off = 0; off < 448; off++)
    if (r[off]) printf("    off %u = %02x (%d)\n", off, r[off], r[off]);

  CVI_RT_MemFree(rt, dmabuf_mem);
  bmk1822_cleanup(bmk);
  CVI_RT_MemFree(rt, mem);
  CVI_RT_DeInit(rt);
  return 0;
}
