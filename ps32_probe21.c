/* ps32_probe21.c — batched throughput for the CORRECT-width (N=192) forced c=1
   ps32 matmul.  Times cmdbuf BUILD vs RUN separately for B in {200,1000,5000}.
   Verifies output correctness after the batch (all identical since inputs fixed).
   Gives definitive per-op cost for the ~85k-submits/token estimate.
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include "cviruntime_context.h"
#include "bmkernel/bm1822/bmkernel_1822.h"

#define NEURON_SZ 262144
#define OUT_OFF 65536

static void run_batch(int K, int N, int B) {
  CVI_RT_HANDLE rt; CVI_RT_Init(&rt);
  CVI_RT_MEM mem = CVI_RT_MemAlloc(rt, NEURON_SZ);
  uint64_t pa = CVI_RT_MemGetPAddr(mem);
  uint8_t *va = CVI_RT_MemGetVAddr(mem);
  CVI_RT_SetBaseReg(rt, 0, pa);
  memset(va, 0, NEURON_SZ);
  int M = 1;
  srand(21);
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
  bmk1822_matrix_lmem_shape_t sr = { .n=(uint32_t)K, .c=1, .w=(uint32_t)N, .col=(uint32_t)N };
  bmk1822_matrix_lmem_shape_t so = { .n=1, .c=1, .w=(uint32_t)N, .col=(uint32_t)N };
  bmk1822_matrix_lmem_t *ml_l = bmk1822_lmem_alloc_matrix(bmk, sl, FMT_I8, 1);
  bmk1822_matrix_lmem_t *ml_r = bmk1822_lmem_alloc_matrix(bmk, sr, FMT_I8, 1);
  bmk1822_matrix_lmem_t *ml_res = bmk1822_lmem_alloc_ps32_matrix(bmk, so, FMT_I8, 1);
  if (!ml_res) { printf("[B%d] alloc fail\n", B); return; }
  uint32_t psz = bmk1822_lmem_ps32_matrix_to_size(bmk, so, FMT_I8, 1);

  bmk1822_matrix_tgmem_t mg_l = {0, 0, FMT_I8, {M,K}, {(uint32_t)K}};
  bmk1822_matrix_tgmem_t mg_r = {0, 2048, FMT_I8, {K,N}, {(uint32_t)N}};
  bmk1822_tdma_g2l_matrix_copy(bmk, &(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_l, ml_l});
  bmk1822_tdma_g2l_matrix_copy(bmk, &(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_r, ml_r});
  memset(va + 8192, 0, 32768); CVI_RT_MemFlush(rt, mem);
  bmk1822_tdma_g2l_general_copy(bmk, &(bmk1822_tdma_tg2l_general_copy_param_t){
    .src_base_reg_index = 0, .src_address = 8192, .dst_address = ml_res->start_address, .bytes = 32768 });

  /* build B matmuls back-to-back */
  struct timespec b0, b1; clock_gettime(CLOCK_MONOTONIC, &b0);
  for (int r = 0; r < B; r++) {
    bmk1822_tiu_matrix_multiplication_param_t p = {
      .res = ml_res, .left = ml_l, .right = ml_r, .bias = NULL,
      .lshift_bits = 0, .rshift_bits = 0, .res_is_int8 = 1, .relu_enable = 0,
      .add_result = 0, .ps32_mode = 2, .layer_id = 1 };
    if (!bmk1822_tiu_matrix_multiplication(bmk, &p)) { printf("[B%d] REJECTED\n", B); return; }
  }
  clock_gettime(CLOCK_MONOTONIC, &b1);
  double build_ms = (b1.tv_sec-b0.tv_sec)*1000.0 + (b1.tv_nsec-b0.tv_nsec)/1e6;

  bmk1822_tdma_l2g_general_copy(bmk, &(bmk1822_tdma_l2tg_general_copy_param_t){
    .src_address = ml_res->start_address, .dst_base_reg_index = 0,
    .dst_address = OUT_OFF, .bytes = 32768 });

  uint32_t cmd_sz; uint8_t *cmd = bmk1822_acquire_cmdbuf(bmk, &cmd_sz);
  uint32_t psize, pmu_size; bmk1822_dmabuf_size(cmd, cmd_sz, &psize, &pmu_size);
  CVI_RT_MEM dmabuf_mem = CVI_RT_MemAlloc(rt, psize + pmu_size);
  uint8_t *dmabuf = CVI_RT_MemGetVAddr(dmabuf_mem);
  bmk1822_dmabuf_convert(cmd, cmd_sz, dmabuf);
  bmk1822_arraybase_set(dmabuf, pa, 0, 0, 0);
  CVI_RT_MemFlush(rt, dmabuf_mem);
  CVI_RT_MEM loaded; CVI_RT_LoadDmabuf(rt, dmabuf_mem, psize+pmu_size, pa, 0, false, &loaded);
  struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
  CVI_RT_RunCmdbufEx(rt, loaded, &(CVI_RT_ARRAYBASE){.gaddr_base0=pa});
  clock_gettime(CLOCK_MONOTONIC, &t1);
  double run_ms = (t1.tv_sec-t0.tv_sec)*1000.0 + (t1.tv_nsec-t0.tv_nsec)/1e6;
  CVI_RT_MemInvld(rt, mem);

  uint8_t *r = (uint8_t*)(va + OUT_OFF);
  int ok = 0, bad = 0;
  for (int n = 0; n < N; n++) {
    int32_t v = 0;
    for (int b = 0; b < 4; b++) v |= ((int32_t)r[b*N+n])<<(8*b);
    if (v == host[n]) ok++; else bad++;
  }
  printf("[B%5d] build=%.1fms(%.3fus/op) run=%.2fms(%.3fus/op) cmdbuf=%uB ok=%d/%d\n",
         B, build_ms, build_ms/B*1000, run_ms, run_ms/B*1000, cmd_sz, ok, N);
  free(host);
  CVI_RT_MemFree(rt, dmabuf_mem); bmk1822_cleanup(bmk); CVI_RT_MemFree(rt, mem); CVI_RT_DeInit(rt);
}

int main(void) {
  printf("== probe21: batched N=192 forced-c1 ps32 throughput ==\n");
  run_batch(32, 192, 100);
  run_batch(32, 192, 500);
  run_batch(32, 192, 1000);
  run_batch(32, 192, 2000);
  return 0;
}
