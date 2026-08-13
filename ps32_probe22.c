/* ps32_probe22.c — two questions from reasoning-engine engineer:
   (A) element_wise_mul F32/BF16 support: alloc a,b,res as I8 / BF16 / F32,
       run mul, read back.  If TIU has no fp/bf16 datapath -> garbage/zero.
   (B) ps32 N>=224: are the FIRST 192 columns correct (stride=N) and 192+ zero?
       -> confirms TIU ps32 width cap is exactly 192, or total garbage.
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "cviruntime_context.h"
#include "bmkernel/bm1822/bmkernel_1822.h"

#define NEURON_SZ 262144
#define OUT_OFF 65536

/* ---------- (A) element_wise_mul format test ---------- */
static void test_mul_fmt(int fmt, const char *tag) {
  CVI_RT_HANDLE rt; CVI_RT_Init(&rt);
  CVI_RT_MEM mem = CVI_RT_MemAlloc(rt, 8192);
  uint64_t pa = CVI_RT_MemGetPAddr(mem);
  uint8_t *va = CVI_RT_MemGetVAddr(mem);
  CVI_RT_SetBaseReg(rt, 0, pa);
  memset(va, 0, 8192);

  /* a = {1,2,3,4,5,6,7,8} as 8-bit lanes; b = {2,...}; res expected = a*b */
  for (int i = 0; i < 8; i++) { va[i] = i+1; va[32+i] = 2; }
  /* also fill 32-bit lanes: a32[i] = 1.5f (bf16 0x3FC0 / f32 0x3FC00000) */
  uint32_t a32[8], b32[8];
  for (int i = 0; i < 8; i++) { a32[i] = 0x3FC00000u; b32[i] = 0x40000000u; } /* 1.5 and 2.0 */
  memcpy(va+64, a32, 32); memcpy(va+128, b32, 32);
  CVI_RT_MemFlush(rt, mem);

  uint8_t cmdbuf[16384] __attribute__((aligned(16)));
  bmk_info_t info = { .chip_version = 1822, .cmdbuf_size = sizeof(cmdbuf), .cmdbuf = cmdbuf };
  bmk1822_context_t *bmk = bmk1822_register(&info);

  bmk1822_tensor_lmem_shape_t ts = {1,1,1,8};
  bmk1822_tensor_lmem_t *a = bmk1822_lmem_alloc_tensor(bmk, ts, fmt, 1);
  bmk1822_tensor_lmem_t *b = bmk1822_lmem_alloc_tensor(bmk, ts, fmt, 1);
  bmk1822_tensor_lmem_t *res = bmk1822_lmem_alloc_tensor(bmk, ts, fmt, 1);
  if (!a || !b || !res) { printf("[mul %s] alloc fail\n", tag); return; }

  bmk1822_tensor_tgmem_t tg = {0, 0, fmt, {1,1,1,8},
    bmk1822_tensor_tgmem_default_stride((bmk1822_tensor_tgmem_shape_t){1,1,1,8}, fmt)};
  /* I8: a at 0, b at 32; F32/BF16: a at 64 (4B), b at 128 */
  uint32_t a_off = (fmt == FMT_I8) ? 0 : 64;
  uint32_t b_off = (fmt == FMT_I8) ? 32 : 128;
  tg.start_address = a_off; bmk1822_tdma_g2l_tensor_copy(bmk, &(bmk1822_tdma_tg2l_tensor_copy_param_t){&tg, a});
  tg.start_address = b_off; bmk1822_tdma_g2l_tensor_copy(bmk, &(bmk1822_tdma_tg2l_tensor_copy_param_t){&tg, b});

  bmk1822_tiu_element_wise_mul_param_t p = {
    .res_high = NULL, .res_low = res, .a = a, .b_is_const = 0, .b = b,
    .rshift_bits = 0, .relu_enable = 0, .layer_id = 1 };
  if (!bmk1822_tiu_element_wise_mul(bmk, &p)) { printf("[mul %s] REJECTED\n", tag); return; }

  tg.start_address = 256;
  bmk1822_tdma_l2g_tensor_copy(bmk, &(bmk1822_tdma_l2tg_tensor_copy_param_t){res, &tg});

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

  uint8_t *r = va + 256;
  int ok = 1;
  for (int i = 0; i < 8; i++) if (r[i] != (fmt == FMT_I8 ? (i+1)*2 : 0)) ok = 0;
  printf("[mul %-4s] out[0..7]=%02x %02x %02x %02x %02x %02x %02x %02x  %s (expect I8 {2,4,..16} else 0)\n",
         tag, r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], ok ? "OK" : "GARBAGE/UNSUPPORTED");
  CVI_RT_MemFree(rt, dmabuf_mem); bmk1822_cleanup(bmk); CVI_RT_MemFree(rt, mem); CVI_RT_DeInit(rt);
}

/* ---------- (B) ps32 first-192 hypothesis for N>=224 ---------- */
static void test_first192(int N) {
  CVI_RT_HANDLE rt; CVI_RT_Init(&rt);
  CVI_RT_MEM mem = CVI_RT_MemAlloc(rt, NEURON_SZ);
  uint64_t pa = CVI_RT_MemGetPAddr(mem);
  uint8_t *va = CVI_RT_MemGetVAddr(mem);
  CVI_RT_SetBaseReg(rt, 0, pa);
  memset(va, 0, NEURON_SZ);
  int M = 1, K = 32;
  srand(22);
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

  uint8_t cmdbuf[65536] __attribute__((aligned(16)));
  bmk_info_t info = { .chip_version = 1822, .cmdbuf_size = sizeof(cmdbuf), .cmdbuf = cmdbuf };
  bmk1822_context_t *bmk = bmk1822_register(&info);
  bmk1822_matrix_lmem_shape_t sl = { .n=1, .c=1, .w=(uint32_t)K, .col=(uint32_t)K };
  bmk1822_matrix_lmem_shape_t sr = { .n=(uint32_t)K, .c=1, .w=(uint32_t)N, .col=(uint32_t)N };
  bmk1822_matrix_lmem_shape_t so = { .n=1, .c=1, .w=(uint32_t)N, .col=(uint32_t)N };
  bmk1822_matrix_lmem_t *ml_l = bmk1822_lmem_alloc_matrix(bmk, sl, FMT_I8, 1);
  bmk1822_matrix_lmem_t *ml_r = bmk1822_lmem_alloc_matrix(bmk, sr, FMT_I8, 1);
  bmk1822_matrix_lmem_t *ml_res = bmk1822_lmem_alloc_ps32_matrix(bmk, so, FMT_I8, 1);
  if (!ml_res) { printf("[N%d] alloc fail\n", N); return; }
  uint32_t psz = bmk1822_lmem_ps32_matrix_to_size(bmk, so, FMT_I8, 1);

  bmk1822_matrix_tgmem_t mg_l = {0, 0, FMT_I8, {M,K}, {(uint32_t)K}};
  bmk1822_matrix_tgmem_t mg_r = {0, 2048, FMT_I8, {K,N}, {(uint32_t)N}};
  bmk1822_tdma_g2l_matrix_copy(bmk, &(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_l, ml_l});
  bmk1822_tdma_g2l_matrix_copy(bmk, &(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_r, ml_r});
  memset(va + 8192, 0, 32768); CVI_RT_MemFlush(rt, mem);
  bmk1822_tdma_g2l_general_copy(bmk, &(bmk1822_tdma_tg2l_general_copy_param_t){
    .src_base_reg_index = 0, .src_address = 8192, .dst_address = ml_res->start_address, .bytes = 32768 });
  bmk1822_tiu_matrix_multiplication_param_t p = {
    .res = ml_res, .left = ml_l, .right = ml_r, .bias = NULL,
    .lshift_bits = 0, .rshift_bits = 0, .res_is_int8 = 1, .relu_enable = 0,
    .add_result = 0, .ps32_mode = 2, .layer_id = 1 };
  if (!bmk1822_tiu_matrix_multiplication(bmk, &p)) { printf("[N%d] REJECTED\n", N); return; }
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
  CVI_RT_RunCmdbufEx(rt, loaded, &(CVI_RT_ARRAYBASE){.gaddr_base0=pa});
  CVI_RT_MemInvld(rt, mem);

  uint8_t *r = (uint8_t*)(va + OUT_OFF);
  int ok192 = 0, bad192 = 0, nz_after = 0;
  for (int n = 0; n < N; n++) {
    int32_t v = 0;
    for (int b = 0; b < 4; b++) v |= ((int32_t)r[b*N+n])<<(8*b);
    if (n < 192) { if (v == host[n]) ok192++; else bad192++; }
    else if (v != 0) nz_after++;
  }
  printf("[N%d] psz=%u first192: ok=%d bad=%d | cols>=192 nonzero=%d | ",
         N, psz, ok192, bad192, nz_after);
  /* also check byte-plane stride N col0..3 */
  printf("col0=%d col1=%d col191=%d col192=%d colN-1=%d (host col0=%d)\n",
         (int)(((int32_t)r[0])|((int32_t)r[N]<<8)|((int32_t)r[2*N]<<16)|((int32_t)r[3*N]<<24)),
         (int)(((int32_t)r[1])|((int32_t)r[N+1]<<8)|((int32_t)r[2*N+1]<<16)|((int32_t)r[3*N+1]<<24)),
         (int)(((int32_t)r[191])|((int32_t)r[N+191]<<8)|((int32_t)r[2*N+191]<<16)|((int32_t)r[3*N+191]<<24)),
         (int)(((int32_t)r[192])|((int32_t)r[N+192]<<8)|((int32_t)r[2*N+192]<<16)|((int32_t)r[3*N+192]<<24)),
         (int)(((int32_t)r[N-1])|((int32_t)r[2*N-1]<<8)|((int32_t)r[3*N-1]<<16)|((int32_t)r[4*N-1]<<24)),
         host[0]);
  free(host);
  bmk1822_cleanup(bmk); CVI_RT_MemFree(rt, mem); CVI_RT_DeInit(rt);
}

int main(void) {
  printf("== probe22: elemwise fmt + ps32 first-192 ==\n");
  test_mul_fmt(FMT_I8, "I8");
  test_mul_fmt(FMT_BF16, "BF16");
  /* F32: TDMA tensor copy asserts fmt==I8||U8||BF16 -> F32 transport unsupported */
  test_first192(224);
  test_first192(256);
  return 0;
}
