/* ps32_probe23.c — can c>1 (multiple w=192 planes) extend the ps32 output
   beyond the N=192 single-plane cap?  Tests res {1,C,192,C*192}:
   - g2l right [K, N] completeness for the SAME c>1 shape (probe15 suggested
     c>1 g2l truncates to first w — verify).
   - matmul output reconstruction: col n -> plane n/W, byte b at
     start + plane*stride.c + b*W + (n%W).
   If c>1 works, wide matrices (N=896/4864) need far fewer calls.
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "cviruntime_context.h"
#include "bmkernel/bm1822/bmkernel_1822.h"

#define NEURON_SZ 262144
#define OUT_OFF 65536

static void run_cp(int K, int C, int W) {
  int N = C * W;
  CVI_RT_HANDLE rt; CVI_RT_Init(&rt);
  CVI_RT_MEM mem = CVI_RT_MemAlloc(rt, NEURON_SZ);
  uint64_t pa = CVI_RT_MemGetPAddr(mem);
  uint8_t *va = CVI_RT_MemGetVAddr(mem);
  CVI_RT_SetBaseReg(rt, 0, pa);
  memset(va, 0, NEURON_SZ);
  int M = 1;
  srand(23);
  int8_t *left = (int8_t*)(va + 0);
  int8_t *right = (int8_t*)(va + 2048);
  int32_t *host = (int32_t*)calloc(N, sizeof(int32_t));
  for (int k = 0; k < K; k++) {
    left[k] = (int8_t)(rand()%200-100);
    for (int n = 0; n < N; n++) {
      right[k*N + n] = (int8_t)(rand()%200-100);
      host[n] += (int32_t)left[k]*right[k*N+n];
    }
  }
  CVI_RT_MemFlush(rt, mem);

  uint8_t cmdbuf[262144] __attribute__((aligned(16)));
  bmk_info_t info = { .chip_version = 1822, .cmdbuf_size = sizeof(cmdbuf), .cmdbuf = cmdbuf };
  bmk1822_context_t *bmk = bmk1822_register(&info);

  bmk1822_matrix_lmem_shape_t sl = { .n=1, .c=1, .w=(uint32_t)K, .col=(uint32_t)K };
  bmk1822_matrix_lmem_shape_t sr = { .n=(uint32_t)K, .c=(uint32_t)C, .w=(uint32_t)W, .col=(uint32_t)N };
  bmk1822_matrix_lmem_shape_t so = { .n=1, .c=(uint32_t)C, .w=(uint32_t)W, .col=(uint32_t)N };
  bmk1822_matrix_lmem_t *ml_l = bmk1822_lmem_alloc_matrix(bmk, sl, FMT_I8, 1);
  bmk1822_matrix_lmem_t *ml_r = bmk1822_lmem_alloc_matrix(bmk, sr, FMT_I8, 1);
  bmk1822_matrix_lmem_t *ml_res = bmk1822_lmem_alloc_ps32_matrix(bmk, so, FMT_I8, 1);
  if (!ml_res) { printf("[C%d W%d] alloc fail\n", C, W); return; }
  uint32_t psz = bmk1822_lmem_ps32_matrix_to_size(bmk, so, FMT_I8, 1);
  uint32_t rsz = bmk1822_lmem_matrix_to_size(bmk, sr, FMT_I8, 1);
  printf("[C%d N%d] res{c=%u,w=%u,col=%u}@%u psz=%u str.c=%u | right@%u rsz=%u\n",
         C, N, so.c, so.w, so.col, ml_res->start_address, psz, ml_res->stride.c,
         ml_r->start_address, rsz);

  bmk1822_matrix_tgmem_t mg_l = {0, 0, FMT_I8, {M,K}, {(uint32_t)K}};
  bmk1822_matrix_tgmem_t mg_r = {0, 2048, FMT_I8, {K,N}, {(uint32_t)N}};
  bmk1822_tdma_g2l_matrix_copy(bmk, &(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_l, ml_l});
  bmk1822_tdma_g2l_matrix_copy(bmk, &(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_r, ml_r});

  bmk1822_tdma_l2g_general_copy(bmk, &(bmk1822_tdma_l2tg_general_copy_param_t){
    .src_address = ml_r->start_address, .dst_base_reg_index = 0,
    .dst_address = OUT_OFF, .bytes = rsz });

  memset(va + 8192, 0, 65536); CVI_RT_MemFlush(rt, mem);
  bmk1822_tdma_g2l_general_copy(bmk, &(bmk1822_tdma_tg2l_general_copy_param_t){
    .src_base_reg_index = 0, .src_address = 8192, .dst_address = ml_res->start_address, .bytes = 32768 });

  bmk1822_tiu_matrix_multiplication_param_t p = {
    .res = ml_res, .left = ml_l, .right = ml_r, .bias = NULL,
    .lshift_bits = 0, .rshift_bits = 0, .res_is_int8 = 1, .relu_enable = 0,
    .add_result = 0, .ps32_mode = 2, .layer_id = 1 };
  if (!bmk1822_tiu_matrix_multiplication(bmk, &p)) { printf("[C%d] REJECTED\n", C); return; }

  bmk1822_tdma_l2g_general_copy(bmk, &(bmk1822_tdma_l2tg_general_copy_param_t){
    .src_address = ml_res->start_address, .dst_base_reg_index = 0,
    .dst_address = OUT_OFF + 65536, .bytes = 32768 });

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

  /* (A) right g2l completeness */
  int8_t *rb = (int8_t*)(va + OUT_OFF);
  int rbad = 0;
  for (int k = 0; k < K; k++)
    for (int n = 0; n < N; n++)
      if (rb[k*N + n] != right[k*N + n]) rbad++;
  printf("  g2l right: %d/%d bad (%s)\n", rbad, K*N, rbad == 0 ? "COMPLETE" : "TRUNCATED");

  /* (B) matmul reconstruction: plane=n/W, byte at plane*stride.c + b*W + n%W */
  uint8_t *r = (uint8_t*)(va + OUT_OFF + 65536);
  uint32_t sc = ml_res->stride.c;
  int ok = 0, bad = 0;
  for (int n = 0; n < N; n++) {
    int32_t v = 0;
    uint32_t plane = (uint32_t)n / W, off = (uint32_t)n % W;
    for (int b = 0; b < 4; b++) {
      uint64_t addr = (uint64_t)plane * sc + (uint64_t)b * W + off;
      if (addr >= 32768) { bad++; continue; }
      v |= ((int32_t)r[addr]) << (8*b);
    }
    if (v == host[n]) ok++; else bad++;
  }
  printf("  matmul plane-layout: ok=%d/%d bad=%d | col0=%d col191=%d col192=%d (host %d/%d/%d)\n",
         ok, N, bad,
         (int)(((int32_t)r[0])|((int32_t)r[W]<<8)|((int32_t)r[2*W]<<16)|((int32_t)r[3*W]<<24)),
         (int)(((int32_t)r[W-1])|((int32_t)r[2*W-1]<<8)|((int32_t)r[3*W-1]<<16)|((int32_t)r[4*W-1]<<24)),
         (int)(((int32_t)r[sc])|((int32_t)r[sc+W]<<8)|((int32_t)r[sc+2*W]<<16)|((int32_t)r[sc+3*W]<<24)),
         host[0], host[W-1], host[W]);
  free(host);
  bmk1822_cleanup(bmk); CVI_RT_MemFree(rt, mem); CVI_RT_DeInit(rt);
}

int main(void) {
  printf("== probe23: c>1 ps32 multi-plane width ==\n");
  run_cp(32, 1, 192);   /* baseline c=1 w=192 */
  run_cp(32, 2, 192);   /* N=384 */
  run_cp(32, 4, 192);   /* N=768 */
  return 0;
}
