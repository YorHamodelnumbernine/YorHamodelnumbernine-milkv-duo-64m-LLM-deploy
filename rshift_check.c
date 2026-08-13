/* rshift_check.c — confirm TIU round-half-up semantics hold for SMALL rshift
 * (1..4) and NEGATIVE accumulation (signed half-way cases).
 *
 * CEO follow-up after Gate sign-off: inference engine needs to confirm the CPU
 * reference formula `sat8((acc + (1<<(rshift-1))) >> rshift)` matches TIU at
 * small rshift. K=32, crafted + pseudo-random data to hit odd acc (half-way
 * at rshift=1), acc≡2 mod 4 (rshift=2), acc≡4 mod 8 (rshift=3), etc., both signs.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "cviruntime_context.h"
#include "bmkernel/bm1822/bmkernel_1822.h"

#define NEURON_SZ 131072
#define OUT_OFF   65536

static inline int sat8(int v){ return v > 127 ? 127 : (v < -128 ? -128 : v); }

static int run(int K, int N, int rshift) {
  CVI_RT_HANDLE rt; CVI_RT_Init(&rt);
  CVI_RT_MEM mem = CVI_RT_MemAlloc(rt, NEURON_SZ);
  uint64_t pa = CVI_RT_MemGetPAddr(mem);
  uint8_t *va = CVI_RT_MemGetVAddr(mem);
  CVI_RT_SetBaseReg(rt, 0, pa);
  memset(va, 0, NEURON_SZ);
  srand(9000 + rshift);
  int8_t *left = (int8_t*)(va + 0);
  int8_t *right = (int8_t*)(va + 4096);
  int8_t *host = (int8_t*)malloc(N);
  int half_neg = 0, half_pos = 0;
  for (int k = 0; k < K; k++) left[k] = (int8_t)(rand()%9 - 4);          /* -4..4 */
  for (int n = 0; n < N; n++) {
    int32_t acc = 0;
    for (int k = 0; k < K; k++) { int8_t rv = (int8_t)(rand()%9 - 4); right[k*N+n]=rv; acc += (int32_t)left[k]*rv; }
    /* half-way: dropped bits are exactly 2^(rshift-1), i.e. acc mod 2^rshift == 2^(rshift-1) */
    if ((acc & ((1<<rshift)-1)) == (1<<(rshift-1))) { if (acc < 0) half_neg++; else half_pos++; }
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
  bmk1822_matrix_tgmem_t mg_r = {0, 4096, FMT_I8, {(uint32_t)K,(uint32_t)N}, {(uint32_t)N}};
  bmk1822_tdma_g2l_matrix_copy(bmk, &(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_l, ml_l});
  bmk1822_tdma_g2l_matrix_copy(bmk, &(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_r, ml_r});
  bmk1822_tiu_matrix_multiplication_param_t p = {
    .res = ml_res, .left = ml_l, .right = ml_r, .bias = NULL,
    .lshift_bits = 0, .rshift_bits = (uint8_t)rshift, .res_is_int8 = 1, .relu_enable = 0,
    .add_result = 0, .ps32_mode = 0, .layer_id = 1 };
  if (!bmk1822_tiu_matrix_multiplication(bmk, &p)) { printf("  REJECTED\n"); free(host); CVI_RT_DeInit(rt); return -1; }
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
  int bad = 0, first_bad = -1;
  for (int n = 0; n < N; n++) if (r[n] != host[n]) { bad++; if (first_bad<0) first_bad=n; }
  printf("  rshift=%d K=%d N=%d: ok=%d/%d bad=%d first_bad@%d | half(neg/pos)=%d/%d\n",
         rshift, K, N, N-bad, N, bad, first_bad, half_neg, half_pos);
  free(host);
  CVI_RT_MemFree(rt, dmabuf_mem); bmk1822_cleanup(bmk); CVI_RT_MemFree(rt, mem); CVI_RT_DeInit(rt);
  return bad;
}

int main(void) {
  printf("===== TIU round-half-up: small rshift + signed half-way check =====\n");
  int total = 0;
  total += run(32, 128, 1);
  total += run(32, 128, 2);
  total += run(32, 128, 3);
  total += run(32, 128, 4);
  /* K=128 sanity at small rshift (values -4..4 -> max|acc|=2048, rshift=4 saturates some) */
  total += run(128, 64, 2);
  total += run(128, 64, 3);
  printf("===== TOTAL bad=%d =====\n", total);
  return total;
}
