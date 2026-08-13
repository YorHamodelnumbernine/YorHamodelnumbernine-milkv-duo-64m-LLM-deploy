/* ps32_test.c — bmk1822 matrix_multiplication capability probe (round 2).
   Questions:
     A) Does int8 accumulate correctly? (verify via rshift + K scan)
     B) res_is_int8=0 with FMT_BF16 res: what value/format is emitted?
     C) ps32_mode=1 (int32 partial) / ps32_mode=2 (fp32): readable?
   Math: left[1xK]=L, right[Kx1]=R -> acc = K*L*R (int32).
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "cviruntime_context.h"
#include "bmkernel/bm1822/bmkernel_1822.h"

#define NEURON_SZ 65536

static void run_one(const char *name, int ps32_mode, int res_is_int8,
                    int use_ps32_matrix, int k, int rshift, int L, int R) {
  CVI_RT_HANDLE rt;
  if (CVI_RT_Init(&rt) != 0) { fprintf(stderr, "[%s] RT_Init fail\n", name); return; }
  CVI_RT_MEM mem = CVI_RT_MemAlloc(rt, NEURON_SZ);
  if (!mem) { fprintf(stderr, "[%s] MemAlloc fail\n", name); CVI_RT_DeInit(rt); return; }
  uint64_t pa = CVI_RT_MemGetPAddr(mem);
  uint8_t *va = CVI_RT_MemGetVAddr(mem);
  CVI_RT_SetBaseReg(rt, 0, pa);
  memset(va, 0, NEURON_SZ);

  int8_t *left = (int8_t*)(va + 0);
  int8_t *right = (int8_t*)(va + 256);
  for (int i = 0; i < k; i++) { left[i] = L; right[i] = R; }
  CVI_RT_MemFlush(rt, mem);

  uint8_t cmdbuf[16384] __attribute__((aligned(16)));
  bmk_info_t info = { .chip_version = 1822, .cmdbuf_size = sizeof(cmdbuf), .cmdbuf = cmdbuf };
  bmk1822_context_t *bmk = bmk1822_register(&info);
  if (!bmk) { CVI_RT_MemFree(rt, mem); CVI_RT_DeInit(rt); return; }

  int M = 1, N = 1;
  bmk1822_matrix_lmem_shape_t sl = bmk1822_matrix_lmem_default_shape(bmk, M, k, FMT_I8);
  bmk1822_matrix_lmem_shape_t sr = bmk1822_matrix_lmem_default_shape(bmk, k, N, FMT_I8);
  bmk1822_matrix_lmem_shape_t so = bmk1822_matrix_lmem_default_shape(bmk, M, N, FMT_I8);

  bmk1822_matrix_lmem_t *ml_l = bmk1822_lmem_alloc_matrix(bmk, sl, FMT_I8, 1);
  bmk1822_matrix_lmem_t *ml_r = bmk1822_lmem_alloc_matrix(bmk, sr, FMT_I8, 1);
  fmt_t res_fmt = res_is_int8 ? FMT_I8 : FMT_BF16;
  bmk1822_matrix_lmem_t *ml_res = use_ps32_matrix
    ? bmk1822_lmem_alloc_ps32_matrix(bmk, so, res_fmt, 1)
    : bmk1822_lmem_alloc_matrix(bmk, so, res_fmt, 1);
  if (!ml_l || !ml_r || !ml_res) { fprintf(stderr, "[%s] alloc fail\n", name); goto out; }

  bmk1822_matrix_tgmem_t mg_l = {0, 0, FMT_I8, {M,k}, {(uint32_t)k}};
  bmk1822_matrix_tgmem_t mg_r = {0, 256, FMT_I8, {k,N}, {(uint32_t)N}};
  bmk1822_matrix_tgmem_t mg_o = {0, 512, res_fmt, {M,N}, {res_is_int8 ? (uint32_t)N : (uint32_t)N*2}};
  if (use_ps32_matrix) { mg_o.fmt = FMT_BF16; mg_o.stride.row = (uint32_t)N*4; }

  bmk1822_tdma_g2l_matrix_copy(bmk, &(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_l, ml_l});
  bmk1822_tdma_g2l_matrix_copy(bmk, &(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_r, ml_r});

  bmk1822_tiu_matrix_multiplication_param_t p = {
    .res = ml_res, .left = ml_l, .right = ml_r, .bias = NULL,
    .lshift_bits = 0, .rshift_bits = (uint8_t)rshift,
    .res_is_int8 = res_is_int8, .relu_enable = 0,
    .add_result = 0, .ps32_mode = (uint8_t)ps32_mode, .layer_id = 1,
  };
  if (!bmk1822_tiu_matrix_multiplication(bmk, &p)) {
    fprintf(stderr, "[%s] matmul rejected\n", name); goto out; }
  bmk1822_tdma_l2g_matrix_copy(bmk, &(bmk1822_tdma_l2tg_matrix_copy_param_t){ml_res, &mg_o});

  uint32_t cmd_sz; uint8_t *cmd = bmk1822_acquire_cmdbuf(bmk, &cmd_sz);
  uint32_t psize, pmu_size; bmk1822_dmabuf_size(cmd, cmd_sz, &psize, &pmu_size);
  CVI_RT_MEM dmabuf_mem = CVI_RT_MemAlloc(rt, psize + pmu_size);
  if (!dmabuf_mem) { fprintf(stderr, "[%s] dmabuf alloc fail\n", name); goto out; }
  uint8_t *dmabuf = CVI_RT_MemGetVAddr(dmabuf_mem);
  bmk1822_dmabuf_convert(cmd, cmd_sz, dmabuf);
  bmk1822_arraybase_set(dmabuf, pa, 0, 0, 0);
  CVI_RT_MemFlush(rt, dmabuf_mem);
  CVI_RT_MEM loaded;
  if (CVI_RT_LoadDmabuf(rt, dmabuf_mem, psize+pmu_size, pa, 0, false, &loaded)!=0 ||
      CVI_RT_RunCmdbufEx(rt, loaded, &(CVI_RT_ARRAYBASE){.gaddr_base0=pa})!=0) {
    fprintf(stderr,"[%s] submit fail\n", name); CVI_RT_MemFree(rt,dmabuf_mem); goto out; }
  CVI_RT_MemInvld(rt, mem);

  uint8_t *r8 = (uint8_t*)(va+512);
  int16_t *r16 = (int16_t*)r8; int32_t *r32 = (int32_t*)r8; float *rf=(float*)r8;
  printf("PS32_TEST|%s|ps32=%d res_i8=%d ps32mat=%d k=%d rshift=%d\n",
         name, ps32_mode, res_is_int8, use_ps32_matrix, k, rshift);
  printf("  in left[0..3]=%d,%d,%d,%d right[0..3]=%d,%d,%d,%d\n",
         left[0],left[1],left[2],left[3], right[0],right[1],right[2],right[3]);
  printf("  out bytes: %02x %02x %02x %02x %02x %02x %02x %02x\n",
         r8[0],r8[1],r8[2],r8[3],r8[4],r8[5],r8[6],r8[7]);
  printf("  as_i16=%d  as_i32=%d  as_f32=%f  (exact int32 acc=%d, >>rshift=%d)\n",
         r16[0], r32[0], rf[0], k*L*R, (k*L*R)>>rshift);
  CVI_RT_MemFree(rt, dmabuf_mem);
out:
  bmk1822_cleanup(bmk);
  CVI_RT_MemFree(rt, mem);
  CVI_RT_DeInit(rt);
}

int main(void) {
  printf("== ps32 capability probe round2 ==\n");
  /* A: int8 accumulate correctness via rshift */
  run_one("A1_i8_k32_r0",   0, 1, 0, 32, 0,  100, 100);  /* expect sat 127 */
  run_one("A2_i8_k32_r12",  0, 1, 0, 32, 12, 100, 100);  /* expect 78 */
  run_one("A3_i8_k32_r4",   0, 1, 0, 32, 4,  100, 100);  /* expect 20000 -> sat? */
  run_one("A4_i8_k4_r0",    0, 1, 0, 4,  0,  100, 100);  /* 40000 -> sat 127 */
  run_one("A5_i8_k2_r0",    0, 1, 0, 2,  0,  10,  10);   /* 200 -> sat 127 */
  run_one("A6_i8_k2_r1",    0, 1, 0, 2,  1,  10,  10);   /* 100 */
  run_one("A7_i8_k2_r0_v",  0, 1, 0, 2,  0,  5,   5);    /* 50 */
  /* B: int16 (BF16 fmt) res */
  run_one("B1_bf16_k2_r0",  0, 0, 0, 2,  0,  5,   5);    /* expect 50 */
  run_one("B2_bf16_k32_r0", 0, 0, 0, 32, 0,  100, 100); /* expect 320000? */
  run_one("B3_bf16_k32_r12",0, 0, 0, 32, 12, 100, 100); /* expect 78 */
  /* C: ps32 modes */
  run_one("C1_ps32m1_k32",  1, 0, 1, 32, 0,  100, 100); /* expect int32 320000 */
  run_one("C2_ps32m2_k32",  2, 0, 1, 32, 0,  100, 100); /* expect fp32 320000 */
  run_one("C3_ps32m2_k2",   2, 0, 1, 2,  0,  5,   5);    /* expect fp32 50 */
  return 0;
}
