/* ps32_probe17.c — cvikernel ps32 matmul test.
   Does cvikernel tiu_matrix_multiplication with ps32_mode=2 produce readable
   int32 partial sums, and how wide (using cvikernel shapes w=N/2)?
   Test N=16, 32, 64, 112.
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "cviruntime_context.h"
#include "cvikernel/cvikernel.h"
#include "bmkernel/bm1822/bmkernel_1822.h"

#define NEURON_SZ 262144
#define OUT_OFF 65536

static void run_cvk(int K, int N) {
  CVI_RT_HANDLE rt;
  if (CVI_RT_Init(&rt) != 0) { printf("[cN=%d] RT_Init fail\n", N); return; }
  CVI_RT_MEM mem = CVI_RT_MemAlloc(rt, NEURON_SZ);
  if (!mem) { printf("[cN=%d] alloc fail\n", N); return; }
  uint64_t pa = CVI_RT_MemGetPAddr(mem);
  uint8_t *va = CVI_RT_MemGetVAddr(mem);
  CVI_RT_SetBaseReg(rt, 0, pa);
  memset(va, 0, NEURON_SZ);

  CVI_RT_KHANDLE kh = CVI_RT_RegisterKernel(rt, 0x40000);
  if (!kh) { printf("[cN=%d] RegisterKernel fail\n", N); return; }
  cvk_context_t *cvk = (cvk_context_t *)kh;
  if (!cvk->ops) { printf("[cN=%d] ops NULL\n", N); return; }

  int M = 1;
  srand(17);
  int8_t *left = (int8_t*)(va + 0);
  int8_t *right = (int8_t*)(va + 2048);
  int32_t *host = (int32_t*)calloc(N, sizeof(int32_t));
  for (int k = 0; k < K; k++) {
    left[k] = (int8_t)(rand()%200-100);
    for (int n = 0; n < N; n++) {
      right[k*N + n] = (int8_t)(rand()%200-100);
      host[n] += (int32_t)left[k] * right[k*N + n];
    }
  }
  CVI_RT_MemFlush(rt, mem);

  cvk_ml_shape_t sl = cvk->ops->ml_default_shape(cvk, M, K, CVK_FMT_I8);
  cvk_ml_shape_t sr = cvk->ops->ml_default_shape(cvk, K, N, CVK_FMT_I8);
  cvk_ml_shape_t so = cvk->ops->ml_default_shape(cvk, M, N, CVK_FMT_I8);
  cvk_ml_t *ml_l = cvk->ops->lmem_alloc_matrix(cvk, sl, CVK_FMT_I8, 1);
  cvk_ml_t *ml_r = cvk->ops->lmem_alloc_matrix(cvk, sr, CVK_FMT_I8, 1);
  cvk_ml_t *ml_res = cvk->ops->lmem_alloc_ps32_matrix(cvk, so, CVK_FMT_I8, 1);
  if (!ml_res) { printf("[cN=%d] ps32 alloc FAIL\n", N); goto out; }
  uint32_t psz = cvk->ops->lmem_ps32_matrix_to_size(cvk, so, CVK_FMT_I8, 1);
  printf("[cN=%d] res sh{n=%u,c=%u,w=%u,col=%u} addr=%u psz=%u str{n=%u,c=%u,h=%u}\n",
         N, so.n, so.c, so.w, so.col, ml_res->start_address, psz,
         ml_res->stride.n, ml_res->stride.c, ml_res->stride.h);

  cvk_mg_t mg_l = {0, pa, CVK_FMT_I8, {M,K}, {K}};
  cvk_mg_t mg_r = {0, pa + 2048, CVK_FMT_I8, {K,N}, {N}};
  cvk->ops->tdma_g2l_matrix_copy(cvk, &(cvk_tdma_g2l_matrix_copy_param_t){&mg_l, ml_l});
  cvk->ops->tdma_g2l_matrix_copy(cvk, &(cvk_tdma_g2l_matrix_copy_param_t){&mg_r, ml_r});

  memset(va + 8192, 0, 16384); CVI_RT_MemFlush(rt, mem);
  cvk->ops->tdma_g2l_general_copy(cvk, &(cvk_tdma_g2l_general_copy_param_t){
    .src_address = 8192, .dst_address = ml_res->start_address, .bytes = 16384 });

  cvk_tiu_matrix_multiplication_param_t p = {
    .res = ml_res, .left = ml_l, .right = ml_r, .bias = NULL,
    .lshift_bits = 0, .rshift_bits = 0, .res_is_int8 = 1, .relu_enable = 0,
    .add_result = 0, .ps32_mode = 2, .layer_id = 1 };
  cvk->ops->tiu_matrix_multiplication(cvk, &p);

  cvk->ops->tdma_l2g_general_copy(cvk, &(cvk_tdma_l2g_general_copy_param_t){
    .src_address = ml_res->start_address, .dst_address = OUT_OFF, .bytes = 16384 });

  int rc = CVI_RT_Submit(kh);
  if (rc != 0) { printf("[cN=%d] submit fail rc=%d\n", N, rc); goto out; }
  CVI_RT_MemInvld(rt, mem);

  uint8_t *r = (uint8_t*)(va + OUT_OFF);
  const uint32_t strides[] = {16, (uint32_t)N, 8, 32, 64};
  int best = -1; uint32_t bst = 0;
  for (unsigned i = 0; i < sizeof(strides)/sizeof(uint32_t); i++) {
    uint32_t st = strides[i]; int ok = 0, bad = 0;
    for (int n = 0; n < N; n++) {
      int32_t v = 0; int valid = 1;
      for (int b = 0; b < 4; b++) { uint32_t o = b*st + n; if (o >= 16384) { valid=0; break; } v |= ((int32_t)r[o])<<(8*b); }
      if (!valid) { bad++; continue; }
      if (v == host[n]) ok++; else bad++;
    }
    printf("  stride=%u: ok=%d bad=%d\n", st, ok, bad);
    if (ok > best) { best = ok; bst = st; }
  }
  printf("  -> best stride=%u ok=%d/%d\n", bst, best, N);
  printf("  col0 stride16 bytes: %02x %02x %02x %02x (host=%d)\n", r[0], r[16], r[32], r[48], host[0]);
out:
  free(host);
  CVI_RT_MemFree(rt, mem); CVI_RT_DeInit(rt);
}

int main(void) {
  printf("== probe17: cvikernel ps32 matmul ==\n");
  run_cvk(32, 16);
  run_cvk(32, 32);
  run_cvk(32, 64);
  run_cvk(32, 112);
  return 0;
}
