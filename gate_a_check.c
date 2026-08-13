/* gate_a_check.c — Path A (two-pass INT8 matmul) gate sign-off, on-board.
 *
 * GATE 1 (pass1 int8 readback): standard INT8 matmul (ps32-free,
 *   ps32_mode=0, res_is_int8=1) with rshift, read back via standard
 *   l2g matrix/general copy + MemInvld, verify == sat8(acc>>rshift).
 *   This is the pass1 read of the two-pass method (CPU computes per-chunk max).
 *
 * GATE 4b (N-tile width): sweep N for standard INT8 matmul at K=32 (G-group)
 *   and K=128 (KG=128) — the two-pass Path A tiling.  Determine max N-tile
 *   that is BOTH correct (TIU width) and fits lmem (right[K,N] <= 32KB).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "cviruntime_context.h"
#include "bmkernel/bm1822/bmkernel_1822.h"

#define NEURON_SZ 262144
#define OUT_OFF   65536

static inline int sat8(int v){ return v > 127 ? 127 : (v < -128 ? -128 : v); }

/* standard INT8 matmul, M=1, left[K] x right[K,N], rshift -> int8 out at OUT_OFF */
static int run_std(int K, int N, int rshift, int verify_host) {
  CVI_RT_HANDLE rt; CVI_RT_Init(&rt);
  CVI_RT_MEM mem = CVI_RT_MemAlloc(rt, NEURON_SZ);
  uint64_t pa = CVI_RT_MemGetPAddr(mem);
  uint8_t *va = CVI_RT_MemGetVAddr(mem);
  CVI_RT_SetBaseReg(rt, 0, pa);
  memset(va, 0, NEURON_SZ);
  srand(K*1000+N);
  int8_t *left = (int8_t*)(va + 0);
  int8_t *right = (int8_t*)(va + 2048);
  int8_t *host = (int8_t*)malloc(N);
  for (int k = 0; k < K; k++) left[k] = (int8_t)(rand()%200-100);
  for (int n = 0; n < N; n++) {
    int32_t acc = 0;
    for (int k = 0; k < K; k++) { int8_t rv = (int8_t)(rand()%200-100); right[k*N+n]=rv; acc += (int32_t)left[k]*rv; }
    /* TIU uses round-half-up when rshift>0 (verified empirically: 11440>>8 -> 45 not 44) */
    int32_t biased = acc + ((rshift > 0) ? (1 << (rshift-1)) : 0);
    host[n] = sat8(biased >> rshift);
  }
  CVI_RT_MemFlush(rt, mem);

  uint8_t cmdbuf[65536] __attribute__((aligned(16)));
  bmk_info_t info = { .chip_version = 1822, .cmdbuf_size = sizeof(cmdbuf), .cmdbuf = cmdbuf };
  bmk1822_context_t *bmk = bmk1822_register(&info);

  bmk1822_matrix_lmem_shape_t sl = { .n=1, .c=1, .w=(uint32_t)K, .col=(uint32_t)K };
  bmk1822_matrix_lmem_shape_t sr = { .n=(uint32_t)K, .c=1, .w=(uint32_t)N, .col=(uint32_t)N };
  bmk1822_matrix_lmem_shape_t so = { .n=1, .c=1, .w=(uint32_t)N, .col=(uint32_t)N };
  bmk1822_matrix_lmem_t *ml_l = bmk1822_lmem_alloc_matrix(bmk, sl, FMT_I8, 1);
  bmk1822_matrix_lmem_t *ml_r = bmk1822_lmem_alloc_matrix(bmk, sr, FMT_I8, 1);
  bmk1822_matrix_lmem_t *ml_res = bmk1822_lmem_alloc_matrix(bmk, so, FMT_I8, 1);
  if (!ml_l || !ml_r || !ml_res) { printf("  alloc fail\n"); free(host); CVI_RT_DeInit(rt); return -1; }
  bmk1822_matrix_tgmem_t mg_l = {0, 0, FMT_I8, {1,(uint32_t)K}, {(uint32_t)K}};
  bmk1822_matrix_tgmem_t mg_r = {0, 2048, FMT_I8, {(uint32_t)K,(uint32_t)N}, {(uint32_t)N}};
  bmk1822_tdma_g2l_matrix_copy(bmk, &(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_l, ml_l});
  bmk1822_tdma_g2l_matrix_copy(bmk, &(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_r, ml_r});
  bmk1822_tiu_matrix_multiplication_param_t p = {
    .res = ml_res, .left = ml_l, .right = ml_r, .bias = NULL,
    .lshift_bits = 0, .rshift_bits = (uint8_t)rshift, .res_is_int8 = 1, .relu_enable = 0,
    .add_result = 0, .ps32_mode = 0, .layer_id = 1 };
  if (!bmk1822_tiu_matrix_multiplication(bmk, &p)) { printf("  REJECTED\n"); free(host); CVI_RT_DeInit(rt); return -1; }
  /* int8 res is lane-interleaved in lmem -> use matrix copy (de-interleaves) */
  bmk1822_matrix_tgmem_t mg_o = {0, OUT_OFF, FMT_I8, {1,(uint32_t)N}, {(uint32_t)N}};
  bmk1822_tdma_l2g_matrix_copy(bmk, &(bmk1822_tdma_l2tg_matrix_copy_param_t){ml_res, &mg_o});
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
  int8_t *r = (int8_t*)(va + OUT_OFF);
  int bad = 0;
  if (verify_host) for (int n = 0; n < N; n++) if (r[n] != host[n]) bad++;
  printf("  std K=%d N=%d rshift=%d: res_addr=%u lmem_r=%u | ok=%d/%d bad=%d | first8=%d %d %d %d %d %d %d %d\n",
         K, N, rshift, ml_res->start_address,
         bmk1822_lmem_matrix_to_size(bmk, sr, FMT_I8, 1), N, N-bad, bad,
         r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]);
  int ret = bad;
  free(host);
  CVI_RT_MemFree(rt, dmabuf_mem); bmk1822_cleanup(bmk); CVI_RT_MemFree(rt, mem); CVI_RT_DeInit(rt);
  return ret;
}

int main(void) {
  printf("===== Path A gate check (two-pass INT8 matmul) =====\n");
  printf("--- GATE 1: pass1 int8 readback (standard INT8 matmul, rshift) ---\n");
  run_std(32,  192, 8, 1);   /* KG=32 group, safe rshift */
  run_std(128, 192, 8, 1);   /* KG=128 chunk, safe rshift */
  run_std(128, 192, 5, 1);   /* tighter rshift, some saturate — still deterministic */
  run_std(256, 192, 8, 1);   /* KG=256 */
  printf("--- GATE 4b: N-tile width sweep for standard INT8 matmul ---\n");
  /* K=32: lmem right[32,N] <= 32KB -> N<=1024; test up to 1024 */
  run_std(32,  192, 8, 1);
  run_std(32,  256, 8, 1);
  run_std(32,  384, 8, 1);
  run_std(32,  512, 8, 1);
  run_std(32,  896, 8, 1);
  run_std(32,  1024, 8, 1);
  /* K=128: lmem right[128,N] <= 32KB -> N<=256 */
  run_std(128, 192, 8, 1);
  run_std(128, 256, 8, 1);
  /* K=128, N=384 would need 48KB lmem -> expect alloc fail (documented) */
  run_std(128, 384, 8, 0);
  return 0;
}
