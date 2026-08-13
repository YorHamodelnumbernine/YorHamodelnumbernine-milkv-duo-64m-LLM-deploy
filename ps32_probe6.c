/* ps32_probe6.c — verify ps32_mode=2 + res_is_int8=1 exports TRUE int32
   partial sums for a K=32 slice with M=1, N=8. Reconstruct lane-interleaved
   layout and compare against host-computed partial sums.
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "cviruntime_context.h"
#include "bmkernel/bm1822/bmkernel_1822.h"

#define NEURON_SZ 262144
#define OUT_OFF 8192

static int32_t reconstruct(const uint8_t *buf, int byte_stride, int idx) {
  /* 4 bytes of int32 at offsets idx*1? depends on layout. We'll test variants. */
  int32_t v = 0;
  for (int b = 0; b < 4; b++)
    v |= ((int32_t)buf[idx + b*byte_stride]) << (8*b);
  return v;
}

int main(void) {
  CVI_RT_HANDLE rt;
  CVI_RT_Init(&rt);
  CVI_RT_MEM mem = CVI_RT_MemAlloc(rt, NEURON_SZ);
  uint64_t pa = CVI_RT_MemGetPAddr(mem);
  uint8_t *va = CVI_RT_MemGetVAddr(mem);
  CVI_RT_SetBaseReg(rt, 0, pa);
  memset(va, 0, NEURON_SZ);

  int M = 1, N = 8, K = 32;
  /* deterministic pseudo-random int8 */
  srand(42);
  int8_t *left = (int8_t*)(va + 0);
  int8_t *right = (int8_t*)(va + 512);
  int32_t host_ref[8] = {0};
  for (int k = 0; k < K; k++) {
    left[k] = (int8_t)(rand() % 200 - 100);
    for (int n = 0; n < N; n++) {
      right[k*N + n] = (int8_t)(rand() % 200 - 100);
      host_ref[n] += (int32_t)left[k] * right[k*N + n];
    }
  }
  printf("host partial sums:");
  for (int n = 0; n < N; n++) printf(" %d", host_ref[n]);
  printf("\n");
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
  printf("res ps32: addr=%u sh{n=%u,c=%u,w=%u,col=%u} str{n=%u,c=%u,h=%u} size=%u\n",
         ml_res->start_address, ml_res->shape.n, ml_res->shape.c, ml_res->shape.w,
         ml_res->shape.col, ml_res->stride.n, ml_res->stride.c, ml_res->stride.h,
         bmk1822_lmem_ps32_matrix_to_size(bmk, so, FMT_I8, 1));

  bmk1822_matrix_tgmem_t mg_l = {0, 0, FMT_I8, {M,K}, {(uint32_t)K}};
  bmk1822_matrix_tgmem_t mg_r = {0, 512, FMT_I8, {K,N}, {(uint32_t)N}};
  bmk1822_tdma_g2l_matrix_copy(bmk, &(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_l, ml_l});
  bmk1822_tdma_g2l_matrix_copy(bmk, &(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_r, ml_r});

  /* zero res region: need size bytes */
  uint32_t zsz = bmk1822_lmem_ps32_matrix_to_size(bmk, so, FMT_I8, 1);
  memset(va + 1536, 0, zsz);
  CVI_RT_MemFlush(rt, mem);
  bmk1822_tdma_g2l_general_copy(bmk, &(bmk1822_tdma_tg2l_general_copy_param_t){
    .src_base_reg_index = 0, .src_address = 1536, .dst_address = ml_res->start_address,
    .bytes = zsz });

  bmk1822_tiu_matrix_multiplication_param_t p = {
    .res = ml_res, .left = ml_l, .right = ml_r, .bias = NULL,
    .lshift_bits = 0, .rshift_bits = 0,
    .res_is_int8 = 1, .relu_enable = 0,
    .add_result = 0, .ps32_mode = 2, .layer_id = 1,
  };
  if (!bmk1822_tiu_matrix_multiplication(bmk, &p)) { printf("matmul REJECTED\n"); return 1; }

  bmk1822_tdma_l2g_general_copy(bmk, &(bmk1822_tdma_l2tg_general_copy_param_t){
    .src_address = ml_res->start_address, .dst_base_reg_index = 0,
    .dst_address = OUT_OFF, .bytes = zsz });

  uint32_t cmd_sz; uint8_t *cmd = bmk1822_acquire_cmdbuf(bmk, &cmd_sz);
  uint32_t psize, pmu_size; bmk1822_dmabuf_size(cmd, cmd_sz, &psize, &pmu_size);
  CVI_RT_MEM dmabuf_mem = CVI_RT_MemAlloc(rt, psize + pmu_size);
  uint8_t *dmabuf = CVI_RT_MemGetVAddr(dmabuf_mem);
  bmk1822_dmabuf_convert(cmd, cmd_sz, dmabuf);
  bmk1822_arraybase_set(dmabuf, pa, 0, 0, 0);
  CVI_RT_MemFlush(rt, dmabuf_mem);
  CVI_RT_MEM loaded;
  CVI_RT_LoadDmabuf(rt, dmabuf_mem, psize+pmu_size, pa, 0, false, &loaded);
  CVI_RT_RunCmdbufEx(rt, loaded, &(CVI_RT_ARRAYBASE){.gaddr_base0=pa});
  CVI_RT_MemInvld(rt, mem);

  uint8_t *r = (uint8_t*)(va + OUT_OFF);
  printf("raw %u bytes:", zsz);
  for (uint32_t i = 0; i < zsz; i++) { if (i%8==0) printf("\n  "); printf("%02x ", r[i]); }
  printf("\n");

  /* reconstruct: byte_stride 8 (lane-per-byte) */
  for (int s = 1; s <= 16; s++) {
    int bad = 0;
    for (int n = 0; n < N; n++) {
      int32_t v = reconstruct(r, s, n*4);  /* element n at byte n*4? guess */
      if (v != host_ref[n]) bad++;
    }
    printf("  stride=%d bytes/elem=4 off=n*4: %s\n", s, bad==0?"MATCH":"mismatch");
  }
  /* try element at n*8 */
  for (int s = 1; s <= 16; s++) {
    int bad = 0;
    for (int n = 0; n < N; n++) {
      int32_t v = reconstruct(r, s, n*8);
      if (v != host_ref[n]) bad++;
    }
    printf("  stride=%d bytes/elem=8 off=n*8: %s\n", s, bad==0?"MATCH":"mismatch");
  }

  CVI_RT_MemFree(rt, dmabuf_mem);
  bmk1822_cleanup(bmk);
  CVI_RT_MemFree(rt, mem);
  CVI_RT_DeInit(rt);
  return 0;
}
