/* ps32_probe14.c — decisive: is ps32 output width == w a hardware cap, or is
   the right matrix g2l copy truncating?  Also maps default_shape w for N sweep.

   A) N sweep: print bmk1822_matrix_lmem_default_shape(M=1, N) for N list.
   B) For N=112: load right[32,112] with right[k][n]=n (distinct), read back the
      right lmem region and count how many distinct columns are present per row.
      -> proves g2l load is complete or truncated.
   C) For N=112: run ps32 matmul, read back 2KB around res, report all byte-plane
      offsets and which columns have nonzero byte0 (identity: all cols=1).
   D) For N=896: run ps32 matmul but FORCE res shape {1,1,112,112} (w=112,c=1),
      right [32,112] chunk.  Does one call produce 112 ps32 columns?
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "cviruntime_context.h"
#include "bmkernel/bm1822/bmkernel_1822.h"

#define NEURON_SZ 262144
#define OUT_OFF 65536

static void hexdump_nz(const char *tag, const uint8_t *b, int off, int n) {
  printf("  [%s]", tag);
  int cnt = 0;
  for (int i = 0; i < n; i++) {
    if (b[off+i]) { if (cnt < 32) printf(" +%d=%02x", i, b[off+i]); cnt++; }
  }
  printf("  nz_total=%d\n", cnt);
}

/* ---- B/C: N=112 default shape ---- */
static void run_N112(void) {
  CVI_RT_HANDLE rt; CVI_RT_Init(&rt);
  CVI_RT_MEM mem = CVI_RT_MemAlloc(rt, NEURON_SZ);
  uint64_t pa = CVI_RT_MemGetPAddr(mem);
  uint8_t *va = CVI_RT_MemGetVAddr(mem);
  CVI_RT_SetBaseReg(rt, 0, pa);
  memset(va, 0, NEURON_SZ);

  int M = 1, K = 32, N = 112;
  int8_t *left = (int8_t*)(va + 0);
  int8_t *right = (int8_t*)(va + 2048);
  memset(left, 0, K); left[0] = 1;
  for (int k = 0; k < K; k++)
    for (int n = 0; n < N; n++)
      right[k*N + n] = (int8_t)(k == 0 ? n : 0);
  CVI_RT_MemFlush(rt, mem);

  uint8_t cmdbuf[65536] __attribute__((aligned(16)));
  bmk_info_t info = { .chip_version = 1822, .cmdbuf_size = sizeof(cmdbuf), .cmdbuf = cmdbuf };
  bmk1822_context_t *bmk = bmk1822_register(&info);

  bmk1822_matrix_lmem_shape_t so = bmk1822_matrix_lmem_default_shape(bmk, M, N, FMT_I8);
  bmk1822_matrix_lmem_shape_t sl = bmk1822_matrix_lmem_default_shape(bmk, M, K, FMT_I8);
  bmk1822_matrix_lmem_shape_t sr = bmk1822_matrix_lmem_default_shape(bmk, K, N, FMT_I8);
  bmk1822_matrix_lmem_t *ml_l = bmk1822_lmem_alloc_matrix(bmk, sl, FMT_I8, 1);
  bmk1822_matrix_lmem_t *ml_r = bmk1822_lmem_alloc_matrix(bmk, sr, FMT_I8, 1);
  bmk1822_matrix_lmem_t *ml_res = bmk1822_lmem_alloc_ps32_matrix(bmk, so, FMT_I8, 1);
  uint32_t psz = bmk1822_lmem_ps32_matrix_to_size(bmk, so, FMT_I8, 1);
  printf("[N112] res sh{n=%u,c=%u,w=%u,col=%u} addr=%u psz=%u | right sh{n=%u,c=%u,w=%u,col=%u} addr=%u\n",
         so.n, so.c, so.w, so.col, ml_res->start_address, psz,
         sr.n, sr.c, sr.w, sr.col, ml_r->start_address);

  bmk1822_matrix_tgmem_t mg_l = {0, 0, FMT_I8, {M,K}, {(uint32_t)K}};
  bmk1822_matrix_tgmem_t mg_r = {0, 2048, FMT_I8, {K,N}, {(uint32_t)N}};
  bmk1822_tdma_g2l_matrix_copy(bmk, &(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_l, ml_l});
  bmk1822_tdma_g2l_matrix_copy(bmk, &(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_r, ml_r});

  /* read back right lmem to check column loading */
  uint32_t rsz = bmk1822_lmem_matrix_to_size(bmk, sr, FMT_I8, 1);
  bmk1822_tdma_l2g_general_copy(bmk, &(bmk1822_tdma_l2tg_general_copy_param_t){
    .src_address = ml_r->start_address, .dst_base_reg_index = 0,
    .dst_address = OUT_OFF, .bytes = rsz });
  printf("[N112] right matrix lmem size=%u\n", rsz);

  /* zero res region generously (2KB) */
  memset(va + 4096, 0, 2048); CVI_RT_MemFlush(rt, mem);
  bmk1822_tdma_g2l_general_copy(bmk, &(bmk1822_tdma_tg2l_general_copy_param_t){
    .src_base_reg_index = 0, .src_address = 4096, .dst_address = ml_res->start_address, .bytes = 2048 });

  bmk1822_tiu_matrix_multiplication_param_t p = {
    .res = ml_res, .left = ml_l, .right = ml_r, .bias = NULL,
    .lshift_bits = 0, .rshift_bits = 0, .res_is_int8 = 1, .relu_enable = 0,
    .add_result = 0, .ps32_mode = 2, .layer_id = 1 };
  if (!bmk1822_tiu_matrix_multiplication(bmk, &p)) { printf("[N112] REJECTED\n"); goto out; }

  /* read back res region 2KB */
  bmk1822_tdma_l2g_general_copy(bmk, &(bmk1822_tdma_l2tg_general_copy_param_t){
    .src_address = ml_res->start_address, .dst_base_reg_index = 0,
    .dst_address = OUT_OFF + 16384, .bytes = 2048 });

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

  printf("[N112] expect each col n -> byte0 = n (0..111) in right readback\n");
  hexdump_nz("right", (uint8_t*)(va + OUT_OFF), 0, rsz);
  printf("[N112] ps32 res readback (2KB): expect cols 0..111 byte0=0x01 present\n");
  hexdump_nz("res", (uint8_t*)(va + OUT_OFF + 16384), 0, 2048);
out:
  bmk1822_cleanup(bmk); CVI_RT_MemFree(rt, mem); CVI_RT_DeInit(rt);
}

/* ---- D: force res w=112, c=1 for a 112-wide chunk ---- */
static void run_forced(void) {
  CVI_RT_HANDLE rt; CVI_RT_Init(&rt);
  CVI_RT_MEM mem = CVI_RT_MemAlloc(rt, NEURON_SZ);
  uint64_t pa = CVI_RT_MemGetPAddr(mem);
  uint8_t *va = CVI_RT_MemGetVAddr(mem);
  CVI_RT_SetBaseReg(rt, 0, pa);
  memset(va, 0, NEURON_SZ);

  int M = 1, K = 32, N = 112;
  int8_t *left = (int8_t*)(va + 0);
  int8_t *right = (int8_t*)(va + 2048);
  memset(left, 0, K); left[0] = 1;
  for (int k = 0; k < K; k++)
    for (int n = 0; n < N; n++)
      right[k*N + n] = 1;   /* all 1 -> acc = 32 for every col */
  CVI_RT_MemFlush(rt, mem);

  uint8_t cmdbuf[65536] __attribute__((aligned(16)));
  bmk_info_t info = { .chip_version = 1822, .cmdbuf_size = sizeof(cmdbuf), .cmdbuf = cmdbuf };
  bmk1822_context_t *bmk = bmk1822_register(&info);

  bmk1822_matrix_lmem_shape_t sl = bmk1822_matrix_lmem_default_shape(bmk, M, K, FMT_I8);
  bmk1822_matrix_lmem_shape_t sr = bmk1822_matrix_lmem_default_shape(bmk, K, N, FMT_I8);
  /* forced res: c=1, w=N, col=N */
  bmk1822_matrix_lmem_shape_t so = { .n = 1, .c = 1, .w = (uint32_t)N, .col = (uint32_t)N };
  bmk1822_matrix_lmem_t *ml_l = bmk1822_lmem_alloc_matrix(bmk, sl, FMT_I8, 1);
  bmk1822_matrix_lmem_t *ml_r = bmk1822_lmem_alloc_matrix(bmk, sr, FMT_I8, 1);
  bmk1822_matrix_lmem_t *ml_res = bmk1822_lmem_alloc_ps32_matrix(bmk, so, FMT_I8, 1);
  if (!ml_res) { printf("[FORCE] ps32 alloc fail for {1,1,%d,%d}\n", N, N); goto out2; }
  uint32_t psz = bmk1822_lmem_ps32_matrix_to_size(bmk, so, FMT_I8, 1);
  printf("[FORCE] res sh{n=%u,c=%u,w=%u,col=%u} addr=%u psz=%u\n",
         so.n, so.c, so.w, so.col, ml_res->start_address, psz);

  bmk1822_matrix_tgmem_t mg_l = {0, 0, FMT_I8, {M,K}, {(uint32_t)K}};
  bmk1822_matrix_tgmem_t mg_r = {0, 2048, FMT_I8, {K,N}, {(uint32_t)N}};
  bmk1822_tdma_g2l_matrix_copy(bmk, &(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_l, ml_l});
  bmk1822_tdma_g2l_matrix_copy(bmk, &(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_r, ml_r});

  memset(va + 4096, 0, 4096); CVI_RT_MemFlush(rt, mem);
  bmk1822_tdma_g2l_general_copy(bmk, &(bmk1822_tdma_tg2l_general_copy_param_t){
    .src_base_reg_index = 0, .src_address = 4096, .dst_address = ml_res->start_address, .bytes = 4096 });

  bmk1822_tiu_matrix_multiplication_param_t p = {
    .res = ml_res, .left = ml_l, .right = ml_r, .bias = NULL,
    .lshift_bits = 0, .rshift_bits = 0, .res_is_int8 = 1, .relu_enable = 0,
    .add_result = 0, .ps32_mode = 2, .layer_id = 1 };
  if (!bmk1822_tiu_matrix_multiplication(bmk, &p)) { printf("[FORCE] REJECTED\n"); goto out2; }

  bmk1822_tdma_l2g_general_copy(bmk, &(bmk1822_tdma_l2tg_general_copy_param_t){
    .src_address = ml_res->start_address, .dst_base_reg_index = 0,
    .dst_address = OUT_OFF, .bytes = 4096 });

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
  printf("[FORCE] expect acc=32 (0x20) for all 112 cols. byte0 planes:\n");
  int found = 0;
  for (uint32_t off = 0; off < 4096; off++)
    if (r[off] == 0x20) { if (found < 16) printf("  off=%u\n", off); found++; }
  printf("[FORCE] total 0x20 bytes in 4KB=%d\n", found);
out2:
  bmk1822_cleanup(bmk); CVI_RT_MemFree(rt, mem); CVI_RT_DeInit(rt);
}

/* ---- A: N sweep of default_shape ---- */
static void sweep(void) {
  CVI_RT_HANDLE rt; CVI_RT_Init(&rt);
  CVI_RT_MEM mem = CVI_RT_MemAlloc(rt, 8192);
  uint64_t pa = CVI_RT_MemGetPAddr(mem);
  uint8_t *va = CVI_RT_MemGetVAddr(mem);
  CVI_RT_SetBaseReg(rt, 0, pa);
  uint8_t cmdbuf[4096] __attribute__((aligned(16)));
  bmk_info_t info = { .chip_version = 1822, .cmdbuf_size = sizeof(cmdbuf), .cmdbuf = cmdbuf };
  bmk1822_context_t *bmk = bmk1822_register(&info);
  int ns[] = {16, 32, 48, 64, 96, 112, 128, 160, 224, 256, 448, 512, 576, 896, 1024, 1536};
  printf("[SWEEP] M=1 default_shape (I8):\n");
  for (unsigned i = 0; i < sizeof(ns)/sizeof(int); i++) {
    bmk1822_matrix_lmem_shape_t s = bmk1822_matrix_lmem_default_shape(bmk, 1, ns[i], FMT_I8);
    uint32_t sz = bmk1822_lmem_ps32_matrix_to_size(bmk, s, FMT_I8, 1);
    printf("  N=%-5d -> {n=%u,c=%u,w=%u,col=%u} ps32size=%u\n",
           ns[i], s.n, s.c, s.w, s.col, sz);
  }
  bmk1822_cleanup(bmk); CVI_RT_MemFree(rt, mem); CVI_RT_DeInit(rt);
}

int main(void) {
  printf("== probe14: ps32 width cap / g2l truncation / shape sweep ==\n");
  sweep();
  run_N112();
  run_forced();
  return 0;
}
