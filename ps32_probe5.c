/* ps32_probe5.c — cleanest ps32 export test.
   - zero 64-byte res region via g2l general_copy
   - run matmul with small values (acc=288 = 0x120): int8 path -> 127 (0x7f)
     int32 partial sum -> low byte 0x20
   - read back full 64 bytes.
   Cases: ps32=0/1/2, res_is_int8 1/0, alloc I8/BF16.
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "cviruntime_context.h"
#include "bmkernel/bm1822/bmkernel_1822.h"

#define NEURON_SZ 262144
#define OUT_OFF 8192

static void hexdump(const char *tag, const uint8_t *b, int off, int n) {
  printf("  [%s]", tag);
  for (int i = 0; i < n; i++) { printf(" %02x", b[off+i]); if (i%8==7) printf(" |"); }
  printf("\n");
}

static void run_case(const char *name, int ps32_mode, int res_is_int8,
                     fmt_t alloc_fmt, int k, int L, int R) {
  CVI_RT_HANDLE rt;
  if (CVI_RT_Init(&rt) != 0) { printf("[%s] RT_Init fail\n", name); return; }
  CVI_RT_MEM mem = CVI_RT_MemAlloc(rt, NEURON_SZ);
  if (!mem) { printf("[%s] MemAlloc fail\n", name); CVI_RT_DeInit(rt); return; }
  uint64_t pa = CVI_RT_MemGetPAddr(mem);
  uint8_t *va = CVI_RT_MemGetVAddr(mem);
  CVI_RT_SetBaseReg(rt, 0, pa);
  memset(va, 0, NEURON_SZ);
  int8_t *left = (int8_t*)(va + 0);
  int8_t *right = (int8_t*)(va + 512);
  for (int i = 0; i < k; i++) { left[i] = L; right[i] = R; }
  CVI_RT_MemFlush(rt, mem);

  uint8_t cmdbuf[65536] __attribute__((aligned(16)));
  bmk_info_t info = { .chip_version = 1822, .cmdbuf_size = sizeof(cmdbuf), .cmdbuf = cmdbuf };
  bmk1822_context_t *bmk = bmk1822_register(&info);
  if (!bmk) { CVI_RT_MemFree(rt, mem); CVI_RT_DeInit(rt); return; }

  int M = 1, N = 1;
  bmk1822_matrix_lmem_shape_t so = bmk1822_matrix_lmem_default_shape(bmk, M, N, FMT_I8);
  bmk1822_matrix_lmem_shape_t s1 = bmk1822_matrix_lmem_default_shape(bmk, M, k, FMT_I8);
  bmk1822_matrix_lmem_shape_t s1r = bmk1822_matrix_lmem_default_shape(bmk, k, N, FMT_I8);

  bmk1822_matrix_lmem_t *ml_l = bmk1822_lmem_alloc_matrix(bmk, s1, FMT_I8, 1);
  bmk1822_matrix_lmem_t *ml_r = bmk1822_lmem_alloc_matrix(bmk, s1r, FMT_I8, 1);
  bmk1822_matrix_lmem_t *ml_res = bmk1822_lmem_alloc_ps32_matrix(bmk, so, alloc_fmt, 1);

  printf("== %s == ps32=%d res_i8=%d fmt=%s res_addr=%u\n", name, ps32_mode,
         res_is_int8, alloc_fmt==FMT_I8?"I8":"BF16", ml_res->start_address);

  bmk1822_matrix_tgmem_t mg_l = {0, 0, FMT_I8, {M,k}, {(uint32_t)k}};
  bmk1822_matrix_tgmem_t mg_r = {0, 512, FMT_I8, {k,N}, {(uint32_t)N}};
  bmk1822_tdma_g2l_matrix_copy(bmk, &(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_l, ml_l});
  bmk1822_tdma_g2l_matrix_copy(bmk, &(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_r, ml_r});

  /* zero 64-byte res region */
  bmk1822_tdma_g2l_general_copy(bmk, &(bmk1822_tdma_tg2l_general_copy_param_t){
    .src_base_reg_index = 0, .src_address = 1536, .dst_address = ml_res->start_address,
    .bytes = 64 });

  bmk1822_tiu_matrix_multiplication_param_t p = {
    .res = ml_res, .left = ml_l, .right = ml_r, .bias = NULL,
    .lshift_bits = 0, .rshift_bits = 0,
    .res_is_int8 = res_is_int8, .relu_enable = 0,
    .add_result = 0, .ps32_mode = (uint8_t)ps32_mode, .layer_id = 1,
  };
  bmk1822_op_t *op = bmk1822_tiu_matrix_multiplication(bmk, &p);
  if (!op) { printf("  matmul REJECTED\n"); goto out; }

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

  int acc = k*L*R;
  printf("  expect acc=%d (0x%08x): int8->0x%02x  int32_lobyte->0x%02x\n",
         acc, acc, (uint8_t)(acc>127?127:acc), (uint8_t)(acc & 0xff));
  hexdump("res", (uint8_t*)(va + OUT_OFF), 0, 64);

  CVI_RT_MemFree(rt, dmabuf_mem);
out:
  bmk1822_cleanup(bmk);
  CVI_RT_MemFree(rt, mem);
  CVI_RT_DeInit(rt);
}

int main(void) {
  printf("== ps32 zeroed-region decisive probe round5 ==\n");
  /* acc=288: int8 result would be 127 (0x7f), int32 low byte 0x20 */
  run_case("nrm_ctrl", 0, 1, FMT_I8, 32, 3, 3);
  run_case("ps32m1_i8", 1, 1, FMT_I8, 32, 3, 3);
  run_case("ps32m1_bf16", 1, 0, FMT_BF16, 32, 3, 3);
  run_case("ps32m2_bf16", 2, 0, FMT_BF16, 32, 3, 3);
  run_case("ps32m2_i8", 2, 1, FMT_I8, 32, 3, 3);
  return 0;
}
