/* ps32_probe12.c — full-32KB lmem scan for ps32 layout.
   zero entire lmem, run identity matmul (all cols = val), scan all of lmem.
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "cviruntime_context.h"
#include "bmkernel/bm1822/bmkernel_1822.h"

#define NEURON_SZ 262144
#define OUT_OFF 65536

static void run_all(const char *tag, int val) {
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
      right[k*N + n] = (k==0) ? (int8_t)val : 0;
  CVI_RT_MemFlush(rt, mem);

  uint8_t cmdbuf[65536] __attribute__((aligned(16)));
  bmk_info_t info = { .chip_version = 1822, .cmdbuf_size = sizeof(cmdbuf), .cmdbuf = cmdbuf };
  bmk1822_context_t *bmk = bmk1822_register(&info);
  bmk1822_matrix_lmem_shape_t so = bmk1822_matrix_lmem_default_shape(bmk, M, N, FMT_I8);
  bmk1822_matrix_lmem_shape_t s1 = bmk1822_matrix_lmem_default_shape(bmk, M, K, FMT_I8);
  bmk1822_matrix_lmem_shape_t s1r = bmk1822_matrix_lmem_default_shape(bmk, K, N, FMT_I8);
  bmk1822_matrix_lmem_t *ml_l = bmk1822_lmem_alloc_matrix(bmk, s1, FMT_I8, 1);
  bmk1822_matrix_lmem_t *ml_r = bmk1822_lmem_alloc_matrix(bmk, s1r, FMT_I8, 1);
  /* zero ENTIRE 32KB lmem FIRST */
  memset(va + 8192, 0, 32768); CVI_RT_MemFlush(rt, mem);
  bmk1822_tdma_g2l_general_copy(bmk, &(bmk1822_tdma_tg2l_general_copy_param_t){
    .src_base_reg_index = 0, .src_address = 8192, .dst_address = 0, .bytes = 32768 });
  /* allocate res AFTER (not overlapped by right matrix) */
  bmk1822_matrix_lmem_t *ml_res = bmk1822_lmem_alloc_ps32_matrix(bmk, so, FMT_I8, 1);
  uint32_t psz = bmk1822_lmem_ps32_matrix_to_size(bmk, so, FMT_I8, 1);
  printf("[%s] res addr=%u psz=%u sh{n=%u,c=%u,w=%u,col=%u} str{n=%u,c=%u,h=%u}\n",
         tag, ml_res->start_address, psz, ml_res->shape.n, ml_res->shape.c,
         ml_res->shape.w, ml_res->shape.col, ml_res->stride.n, ml_res->stride.c, ml_res->stride.h);
  printf("[%s] left addr=%u right addr=%u (right size~%u)\n", tag, ml_l->start_address, ml_r->start_address,
         (uint32_t)(ml_res->start_address - ml_r->start_address));

  bmk1822_matrix_tgmem_t mg_l = {0, 0, FMT_I8, {M,K}, {(uint32_t)K}};
  bmk1822_matrix_tgmem_t mg_r = {0, 2048, FMT_I8, {K,N}, {(uint32_t)N}};
  bmk1822_tdma_g2l_matrix_copy(bmk, &(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_l, ml_l});
  bmk1822_tdma_g2l_matrix_copy(bmk, &(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_r, ml_r});

  bmk1822_tiu_matrix_multiplication_param_t p = {
    .res = ml_res, .left = ml_l, .right = ml_r, .bias = NULL,
    .lshift_bits = 0, .rshift_bits = 0, .res_is_int8 = 1, .relu_enable = 0,
    .add_result = 0, .ps32_mode = 2, .layer_id = 1 };
  if (!bmk1822_tiu_matrix_multiplication(bmk, &p)) { printf("[%s] REJECTED\n", tag); goto out; }

  bmk1822_tdma_l2g_general_copy(bmk, &(bmk1822_tdma_l2tg_general_copy_param_t){
    .src_address = 0, .dst_base_reg_index = 0, .dst_address = OUT_OFF, .bytes = 32768 });

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
  uint8_t pat = (val == 1) ? 0x01 : 0xFF;
  printf("[%s] 0x%02x byte positions (runs of consecutive):\n", tag, pat);
  int cnt = 0; uint32_t rs = 0; int in_run = 0;
  for (uint32_t i = 0; i < 32768; i++) {
    if (r[i] == pat) {
      if (!in_run) { in_run = 1; rs = i; }
      cnt++;
    } else {
      if (in_run) { printf("    [%u..%u] len=%u\n", rs, i-1, i-rs); in_run = 0; }
    }
  }
  if (in_run) printf("    [%u..%u] len=%u\n", rs, 32767, 32768-rs);
  printf("    total bytes=0x%02x: %d\n", pat, cnt);
out:
  bmk1822_cleanup(bmk); CVI_RT_MemFree(rt, mem); CVI_RT_DeInit(rt);
}

int main(void) {
  printf("== full-lmem ps32 layout scan ==\n");
  run_all("all+1", 1);
  run_all("all-1", -1);
  return 0;
}
