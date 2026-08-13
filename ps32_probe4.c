/* ps32_probe4.c — decisive zero-region test.
   Pre-write zeros into the ps32 res matrix region via g2l copy, run the
   ps32 matmul, read back. If region stays zero -> TIU does NOT write ps32
   output for int8 matrix_multiplication. Also try qdm matmul (quan_m).
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "cviruntime_context.h"
#include "bmkernel/bm1822/bmkernel_1822.h"

#define NEURON_SZ 262144
#define OUT_OFF 8192

static void hexdump_region(const char *tag, const uint8_t *b, int off, int n) {
  printf("  [%s] +%03d:", tag, off);
  for (int i = 0; i < n; i++) printf(" %02x", b[off+i]);
  printf("\n");
}

static void run_zero(const char *name, int ps32_mode, int res_is_int8,
                     fmt_t alloc_fmt, int use_qdm) {
  CVI_RT_HANDLE rt;
  if (CVI_RT_Init(&rt) != 0) { printf("[%s] RT_Init fail\n", name); return; }
  CVI_RT_MEM mem = CVI_RT_MemAlloc(rt, NEURON_SZ);
  if (!mem) { printf("[%s] MemAlloc fail\n", name); CVI_RT_DeInit(rt); return; }
  uint64_t pa = CVI_RT_MemGetPAddr(mem);
  uint8_t *va = CVI_RT_MemGetVAddr(mem);
  CVI_RT_SetBaseReg(rt, 0, pa);
  memset(va, 0, NEURON_SZ);

  int k = 32, M = 1, N = 1, L = 100, R = 100;
  int8_t *left = (int8_t*)(va + 0);
  int8_t *right = (int8_t*)(va + 512);
  uint8_t *zeroreg = (uint8_t*)(va + 1024);   /* zero source for ps32 region */
  for (int i = 0; i < k; i++) { left[i] = L; right[i] = R; }
  CVI_RT_MemFlush(rt, mem);

  uint8_t cmdbuf[65536] __attribute__((aligned(16)));
  bmk_info_t info = { .chip_version = 1822, .cmdbuf_size = sizeof(cmdbuf), .cmdbuf = cmdbuf };
  bmk1822_context_t *bmk = bmk1822_register(&info);
  if (!bmk) { CVI_RT_MemFree(rt, mem); CVI_RT_DeInit(rt); return; }

  bmk1822_matrix_lmem_shape_t so = bmk1822_matrix_lmem_default_shape(bmk, M, N, FMT_I8);
  bmk1822_matrix_lmem_shape_t s1 = bmk1822_matrix_lmem_default_shape(bmk, M, k, FMT_I8);
  bmk1822_matrix_lmem_shape_t s1r = bmk1822_matrix_lmem_default_shape(bmk, k, N, FMT_I8);

  bmk1822_matrix_lmem_t *ml_l = bmk1822_lmem_alloc_matrix(bmk, s1, FMT_I8, 1);
  bmk1822_matrix_lmem_t *ml_r = bmk1822_lmem_alloc_matrix(bmk, s1r, FMT_I8, 1);
  bmk1822_matrix_lmem_t *ml_res = bmk1822_lmem_alloc_ps32_matrix(bmk, so, alloc_fmt, 1);

  printf("== %s == ps32_mode=%d res_is_int8=%d qdm=%d res_addr=%u\n",
         name, ps32_mode, res_is_int8, use_qdm, ml_res->start_address);

  bmk1822_matrix_tgmem_t mg_l = {0, 0, FMT_I8, {M,k}, {(uint32_t)k}};
  bmk1822_matrix_tgmem_t mg_r = {0, 512, FMT_I8, {k,N}, {(uint32_t)N}};
  /* zero source matrix for res region: point at zeroreg (all zeros), shape {1,1} */
  bmk1822_matrix_tgmem_t mg_z = {0, 1024, FMT_I8, {M,N}, {(uint32_t)N}};
  bmk1822_tdma_g2l_matrix_copy(bmk, &(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_l, ml_l});
  bmk1822_tdma_g2l_matrix_copy(bmk, &(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_r, ml_r});
  bmk1822_tdma_g2l_matrix_copy(bmk, &(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_z, ml_res}); /* zero res region */

  int ok;
  if (use_qdm) {
    bmk1822_tiu_matrix_multiplication_qdm_param_t p = {
      .res = ml_res, .left = ml_l, .right = ml_r, .bias = NULL,
      .lshift_bits = 0, .rshift_bits = 0,
      .res_is_int8 = res_is_int8, .relu_enable = 0,
      .add_result = 0, .ps32_mode = (uint8_t)ps32_mode, .quan_m = 1, .layer_id = 1,
    };
    ok = bmk1822_tiu_matrix_multiplication_qdm(bmk, &p);
    printf("  qdm matmul api return: %d\n", ok);
  } else {
    bmk1822_tiu_matrix_multiplication_param_t p = {
      .res = ml_res, .left = ml_l, .right = ml_r, .bias = NULL,
      .lshift_bits = 0, .rshift_bits = 0,
      .res_is_int8 = res_is_int8, .relu_enable = 0,
      .add_result = 0, .ps32_mode = (uint8_t)ps32_mode, .layer_id = 1,
    };
    ok = bmk1822_tiu_matrix_multiplication(bmk, &p);
    printf("  matmul api return: %d\n", ok);
  }
  if (!ok) { printf("  REJECTED\n"); goto out; }

  bmk1822_tdma_l2g_general_copy(bmk, &(bmk1822_tdma_l2tg_general_copy_param_t){
    .src_address = ml_res->start_address, .dst_base_reg_index = 0,
    .dst_address = OUT_OFF, .bytes = 64 });

  uint32_t cmd_sz; uint8_t *cmd = bmk1822_acquire_cmdbuf(bmk, &cmd_sz);
  uint32_t psize, pmu_size; bmk1822_dmabuf_size(cmd, cmd_sz, &psize, &pmu_size);
  CVI_RT_MEM dmabuf_mem = CVI_RT_MemAlloc(rt, psize + pmu_size);
  if (!dmabuf_mem) { printf("  dmabuf alloc fail\n"); goto out; }
  uint8_t *dmabuf = CVI_RT_MemGetVAddr(dmabuf_mem);
  bmk1822_dmabuf_convert(cmd, cmd_sz, dmabuf);
  bmk1822_arraybase_set(dmabuf, pa, 0, 0, 0);
  CVI_RT_MemFlush(rt, dmabuf_mem);
  CVI_RT_MEM loaded;
  if (CVI_RT_LoadDmabuf(rt, dmabuf_mem, psize+pmu_size, pa, 0, false, &loaded)!=0 ||
      CVI_RT_RunCmdbufEx(rt, loaded, &(CVI_RT_ARRAYBASE){.gaddr_base0=pa})!=0) {
    printf("  submit fail\n"); CVI_RT_MemFree(rt,dmabuf_mem); goto out; }
  CVI_RT_MemInvld(rt, mem);

  printf("  expect int32 acc=%d (0x%08x)\n", k*L*R, k*L*R);
  hexdump_region("res-region", (uint8_t*)(va + OUT_OFF), 0, 64);

  CVI_RT_MemFree(rt, dmabuf_mem);
out:
  bmk1822_cleanup(bmk);
  CVI_RT_MemFree(rt, mem);
  CVI_RT_DeInit(rt);
}

int main(void) {
  printf("== ps32 zero-region decisive probe ==\n");
  /* control: normal int8 matmul, res region zeroed first -> expect 127 at byte0 */
  run_zero("ctrl_nrm_i8", 0, 1, FMT_I8, 0);
  /* ps32_mode=1, zeroed region -> does TIU write int32 partial sum? */
  run_zero("ps32m1_i8_zeroed", 1, 1, FMT_I8, 0);
  /* ps32_mode=2, zeroed region */
  run_zero("ps32m2_bf16_zeroed", 2, 0, FMT_BF16, 0);
  /* qdm matmul with quan_m, ps32_mode=1 */
  run_zero("qdm_ps32m1_i8", 1, 1, FMT_I8, 1);
  return 0;
}
