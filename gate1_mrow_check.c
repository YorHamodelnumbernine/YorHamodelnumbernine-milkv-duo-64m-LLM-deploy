/* gate1_mrow_check.c — Gate ① M>1 re-confirm: pass1 int8 readback + two-pass
 *   flow matching qwen_kal_ref.c chunk_matmul_twopass semantics, on-board.
 *
 * Flow per config (M rows, K=32 single chunk, N cols):
 *   host: acc[m][n] = sum_k left[m][k]*right[k][n]   (int32, exact)
 *         wmax = max|right| ; rsafe = matmul_rshift_w(32,wmax)-3 (>=4)
 *   TPU pass1: standard INT8 matmul (ps32-free, res_is_int8=1) rshift=rsafe
 *              -> l2g_matrix_copy + MemInvld -> p1_tpu[M,N]
 *   verify p1_tpu == int8_round_div(acc,rsafe) element-wise (bad1)
 *   CPU per-chunk max over p1_tpu: maxabs ; est=maxabs*(1<<rsafe)
 *         r_opt = min r s.t. est <= 127<<r        (same as ref pass2)
 *   TPU pass2: matmul rshift=r_opt -> l2g + MemInvld -> p2_tpu[M,N]
 *   verify p2_tpu == int8_round_div(acc,r_opt) element-wise (bad2)
 *   numeric bonus: out_f32[m][n] = p2_tpu*(1<<r_opt)*gsc[n]*sc_row[m]
 *              vs reference fp64 accumulate (max abs diff)
 *
 * Also re-confirms Gate ④b: [M,32]x[32,N] lmem fit + N=512 correctness.
 *
 * Build: make gate1_mrow_check ; deploy: duo_run.py gate1_mrow_check
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "cviruntime_context.h"
#include "bmkernel/bm1822/bmkernel_1822.h"

#define NEURON_SZ 262144
#define RIGHT_OFF 4096
#define OUT1_OFF  65536
#define OUT2_OFF  131072

/* ------- exact replicas of qwen_kal_ref.c helpers ------- */
static int32_t max_i8(const int8_t *a, size_t n) {
  int32_t m = 0; for (size_t i = 0; i < n; i++) { int32_t v = a[i]; if (v < 0) v = -v; if (v > m) m = v; } return m;
}
/* minimal r with K*127*wmax >> r <= 127 */
static int matmul_rshift_w(int K, int wmax) {
  int r = 0; long long md = (long long)K * 127 * wmax;
  while ((md >> r) > 127) r++;
  return r;
}
static inline int8_t int8_round_div(int32_t acc, int rshift) {
  int32_t half = 1 << (rshift - 1);
  int32_t r = (acc + half) >> rshift;
  if (r > 127) r = 127;
  if (r < -128) r = -128;
  return (int8_t)r;
}
static inline int sat8(int v){ return v > 127 ? 127 : (v < -128 ? -128 : v); }
static float fp16_to_f32(uint16_t h) {
  uint32_t sign = (h & 0x8000u) << 16, exp = (h >> 10) & 0x1f, man = h & 0x3ff, f;
  if (exp == 0) { if (man == 0) f = sign; else { exp = 127 - 15 + 1; while (!(man & 0x400)) { man <<= 1; exp--; } man &= 0x3ff; f = sign | (exp << 23) | (man << 13); } }
  else if (exp == 31) f = sign | 0x7f800000u | (man << 13);
  else f = sign | ((exp + (127 - 15)) << 23) | (man << 13);
  float out; memcpy(&out, &f, 4); return out;
}

/* one full two-pass test at (M,K,N); returns #bad (0 = PASS) */
static int run_twopass(int M, int K, int N, const char *desc) {
  CVI_RT_HANDLE rt; CVI_RT_Init(&rt);
  CVI_RT_MEM mem = CVI_RT_MemAlloc(rt, NEURON_SZ);
  uint64_t pa = CVI_RT_MemGetPAddr(mem);
  uint8_t *va = CVI_RT_MemGetVAddr(mem);
  CVI_RT_SetBaseReg(rt, 0, pa);
  memset(va, 0, NEURON_SZ);
  srand(M * 100000 + K * 1000 + N);

  int8_t *left = (int8_t*)(va + 0);
  int8_t *right = (int8_t*)(va + RIGHT_OFF);
  /* host: acc[M,N] int32, gsc[N] fp32 (per-col group scale), sc_row[M] */
  int32_t *acc = malloc((size_t)M * N * sizeof(int32_t));
  float *gsc = malloc(N * sizeof(float));
  float *sc_row = malloc(M * sizeof(float));
  for (int m = 0; m < M; m++) {
    for (int k = 0; k < K; k++) left[m * K + k] = (int8_t)(rand() % 200 - 100);
    sc_row[m] = 1e-4f + (float)(rand() % 1000) * 1e-4f;
  }
  for (int n = 0; n < N; n++) {
    /* group scale: fp16 in [2e-3, 2.5e-2] then widen to fp32 */
    uint32_t bits = (uint16_t)((rand() % 2200) + 0x1800); /* ~fp16 0.002..0.025 */
    gsc[n] = fp16_to_f32((uint16_t)bits);
    for (int k = 0; k < K; k++) right[k * N + n] = (int8_t)(rand() % 200 - 100);
  }
  for (int m = 0; m < M; m++)
    for (int n = 0; n < N; n++) {
      int32_t s = 0;
      for (int k = 0; k < K; k++) s += (int32_t)left[m * K + k] * (int32_t)right[k * N + n];
      acc[m * N + n] = s;
    }
  int wmax = max_i8(right, (size_t)K * N);
  int rsafe = matmul_rshift_w(K, wmax) - 3; if (rsafe < 4) rsafe = 4;
  CVI_RT_MemFlush(rt, mem);

  uint8_t cmdbuf[65536] __attribute__((aligned(16)));
  bmk_info_t info = { .chip_version = 1822, .cmdbuf_size = sizeof(cmdbuf), .cmdbuf = cmdbuf };
  bmk1822_context_t *bmk = bmk1822_register(&info);

  /* FORCED linear-matrix layout (same as gate_a_check): c=1, w=full width.
     default_shape returns c-split layout which bmk1822 l2g_matrix_copy does
     NOT reconstruct correctly; linear layout is the proven readback path. */
  bmk1822_matrix_lmem_shape_t sl = {.n=(uint32_t)M, .c=1, .w=(uint32_t)K, .col=(uint32_t)K};
  bmk1822_matrix_lmem_shape_t sr = {.n=(uint32_t)K, .c=1, .w=(uint32_t)N, .col=(uint32_t)N};
  bmk1822_matrix_lmem_shape_t so = {.n=(uint32_t)M, .c=1, .w=(uint32_t)N, .col=(uint32_t)N};
  printf("  [%s] shapes l{n=%u,c=%u,w=%u,col=%u} r{n=%u,c=%u,w=%u,col=%u} o{n=%u,c=%u,w=%u,col=%u}\n",
         desc, sl.n, sl.c, sl.w, sl.col, sr.n, sr.c, sr.w, sr.col, so.n, so.c, so.w, so.col);
  bmk1822_matrix_lmem_t *ml_l = bmk1822_lmem_alloc_matrix(bmk, sl, FMT_I8, 1);
  bmk1822_matrix_lmem_t *ml_r = bmk1822_lmem_alloc_matrix(bmk, sr, FMT_I8, 1);
  bmk1822_matrix_lmem_t *ml_o = bmk1822_lmem_alloc_matrix(bmk, so, FMT_I8, 1);
  uint32_t ls = bmk1822_lmem_matrix_to_size(bmk, sr, FMT_I8, 1)
              + bmk1822_lmem_matrix_to_size(bmk, so, FMT_I8, 1)
              + bmk1822_lmem_matrix_to_size(bmk, sl, FMT_I8, 1);
  if (!ml_l || !ml_r || !ml_o) {
    printf("  [%s] M=%d K=%d N=%d lmem_total=%u: ALLOC FAIL\n", desc, M, K, N, ls);
    free(acc); free(gsc); free(sc_row); CVI_RT_DeInit(rt); return 1;
  }

  bmk1822_matrix_tgmem_t mg_l = {0, 0, FMT_I8, {(uint32_t)M,(uint32_t)K}, {(uint32_t)K}};
  bmk1822_matrix_tgmem_t mg_r = {0, RIGHT_OFF, FMT_I8, {(uint32_t)K,(uint32_t)N}, {(uint32_t)N}};
  bmk1822_tdma_g2l_matrix_copy(bmk, &(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_l, ml_l});
  bmk1822_tdma_g2l_matrix_copy(bmk, &(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_r, ml_r});

  /* pass1 + pass2 in ONE cmdbuf, sharing the same left/right lmem (no reload).
     Simulates the engine template: rsafe run -> l2g P1, then r_opt run -> l2g P2. */
  bmk1822_matrix_tgmem_t mg_o1 = {0, OUT1_OFF, FMT_I8, {(uint32_t)M,(uint32_t)N}, {(uint32_t)N}};
  bmk1822_tiu_matrix_multiplication_param_t mm1 = {
    .res = ml_o, .left = ml_l, .right = ml_r, .bias = NULL,
    .lshift_bits = 0, .rshift_bits = (uint8_t)rsafe, .res_is_int8 = 1, .relu_enable = 0,
    .add_result = 0, .ps32_mode = 0, .layer_id = 1 };
  if (!bmk1822_tiu_matrix_multiplication(bmk, &mm1)) { printf("  [%s] P1 REJECTED\n", desc); goto fail; }
  bmk1822_tdma_l2g_matrix_copy(bmk, &(bmk1822_tdma_l2tg_matrix_copy_param_t){ml_o, &mg_o1});

  uint32_t cmd_sz; uint8_t *cmd = bmk1822_acquire_cmdbuf(bmk, &cmd_sz);
  uint32_t psize, pmu_size; bmk1822_dmabuf_size(cmd, cmd_sz, &psize, &pmu_size);
  CVI_RT_MEM dmabuf_mem = CVI_RT_MemAlloc(rt, psize + pmu_size);
  uint8_t *dmabuf = CVI_RT_MemGetVAddr(dmabuf_mem);
  bmk1822_dmabuf_convert(cmd, cmd_sz, dmabuf);
  bmk1822_arraybase_set(dmabuf, pa, 0, 0, 0);
  CVI_RT_MemFlush(rt, dmabuf_mem);
  CVI_RT_MEM loaded; CVI_RT_LoadDmabuf(rt, dmabuf_mem, psize + pmu_size, pa, 0, false, &loaded);
  CVI_RT_RunCmdbufEx(rt, loaded, &(CVI_RT_ARRAYBASE){.gaddr_base0 = pa});
  CVI_RT_MemInvld(rt, mem);

  /* ---- pass1 verify + per-chunk max (CPU reads real p1) ---- */
  int8_t *p1 = (int8_t*)(va + OUT1_OFF);
  int bad1 = 0; int maxabs = 0;
  for (int m = 0; m < M; m++)
    for (int n = 0; n < N; n++) {
      int8_t hv = int8_round_div(acc[m * N + n], rsafe);
      int8_t tv = p1[m * N + n];
      if (tv != hv) bad1++;
      int a = tv; if (a < 0) a = -a; if (a > maxabs) maxabs = a;
    }
  long long est = (long long)maxabs << rsafe;
  int r_opt = 0; while (est > (127LL << r_opt)) r_opt++;

  /* ---- pass2: fresh cmdbuf, same lmem layout, only rshift changes (no reload) ---- */
  uint8_t cmdbuf2[65536] __attribute__((aligned(16)));
  bmk_info_t info2 = { .chip_version = 1822, .cmdbuf_size = sizeof(cmdbuf2), .cmdbuf = cmdbuf2 };
  bmk1822_context_t *bmk2 = bmk1822_register(&info2);
  bmk1822_matrix_lmem_t *ml_l2 = bmk1822_lmem_alloc_matrix(bmk2, sl, FMT_I8, 1);
  bmk1822_matrix_lmem_t *ml_r2 = bmk1822_lmem_alloc_matrix(bmk2, sr, FMT_I8, 1);
  bmk1822_matrix_lmem_t *ml_o2 = bmk1822_lmem_alloc_matrix(bmk2, so, FMT_I8, 1);
  if (!ml_l2 || !ml_r2 || !ml_o2) { printf("  [%s] P2 alloc FAIL\n", desc); goto fail2; }
  /* right/left ALREADY in lmem from cmdbuf1 -> no g2l in cmdbuf2 */
  bmk1822_matrix_tgmem_t mg_o2 = {0, OUT2_OFF, FMT_I8, {(uint32_t)M,(uint32_t)N}, {(uint32_t)N}};
  bmk1822_tiu_matrix_multiplication_param_t mm2 = {
    .res = ml_o2, .left = ml_l2, .right = ml_r2, .bias = NULL,
    .lshift_bits = 0, .rshift_bits = (uint8_t)r_opt, .res_is_int8 = 1, .relu_enable = 0,
    .add_result = 0, .ps32_mode = 0, .layer_id = 2 };
  if (!bmk1822_tiu_matrix_multiplication(bmk2, &mm2)) { printf("  [%s] P2 REJECTED\n", desc); goto fail2; }
  bmk1822_tdma_l2g_matrix_copy(bmk2, &(bmk1822_tdma_l2tg_matrix_copy_param_t){ml_o2, &mg_o2});
  cmd = bmk1822_acquire_cmdbuf(bmk2, &cmd_sz);
  bmk1822_dmabuf_size(cmd, cmd_sz, &psize, &pmu_size);
  CVI_RT_MEM dmabuf2 = CVI_RT_MemAlloc(rt, psize + pmu_size);
  dmabuf = CVI_RT_MemGetVAddr(dmabuf2);
  bmk1822_dmabuf_convert(cmd, cmd_sz, dmabuf);
  bmk1822_arraybase_set(dmabuf, pa, 0, 0, 0);
  CVI_RT_MemFlush(rt, dmabuf2);
  CVI_RT_MEM loaded2; CVI_RT_LoadDmabuf(rt, dmabuf2, psize + pmu_size, pa, 0, false, &loaded2);
  CVI_RT_RunCmdbufEx(rt, loaded2, &(CVI_RT_ARRAYBASE){.gaddr_base0 = pa});
  CVI_RT_MemInvld(rt, mem);

  int8_t *p2 = (int8_t*)(va + OUT2_OFF);
  int bad2 = 0; int sat2 = 0;
  for (int m = 0; m < M; m++)
    for (int n = 0; n < N; n++) {
      int8_t hv = int8_round_div(acc[m * N + n], r_opt);
      if (p2[m * N + n] != hv) bad2++;
      if (p2[m * N + n] == 127 || p2[m * N + n] == -128) sat2++;
    }
  /* numeric bonus: fp32 accumulate vs fp64 reference (single chunk) */
  double maxdiff = 0.0;
  for (int m = 0; m < M; m++)
    for (int n = 0; n < N; n++) {
      double ref = (double)p2[m * N + n] * (double)(1 << r_opt) * (double)gsc[n] * (double)sc_row[m];
      float  tpu = (float)((float)(p2[m * N + n] * (1 << r_opt)) * gsc[n] * sc_row[m]);
      double d = fabs((double)tpu - ref); if (d > maxdiff) maxdiff = d;
    }
  printf("  [%s] M=%d K=%d N=%d lmem=%u | wmax=%d rsafe=%d | P1 ok=%d/%d bad=%d"
         " | maxabs=%d r_opt=%d | P2 ok=%d/%d bad=%d sat8=%d | fp32maxdiff=%.3g\n",
         desc, M, K, N, ls, wmax, rsafe, M*N, M*N-bad1, bad1, maxabs, r_opt,
         M*N, M*N-bad2, bad2, sat2, maxdiff);
  CVI_RT_MemFree(rt, dmabuf2); bmk1822_cleanup(bmk2);
  CVI_RT_MemFree(rt, dmabuf_mem); bmk1822_cleanup(bmk);
  free(acc); free(gsc); free(sc_row); CVI_RT_MemFree(rt, mem); CVI_RT_DeInit(rt);
  return bad1 + bad2;
fail:
  CVI_RT_MemFree(rt, dmabuf_mem); bmk1822_cleanup(bmk);
  free(acc); free(gsc); free(sc_row); CVI_RT_MemFree(rt, mem); CVI_RT_DeInit(rt);
  return 1;
fail2:
  CVI_RT_MemFree(rt, dmabuf_mem); bmk1822_cleanup(bmk); bmk1822_cleanup(bmk2);
  free(acc); free(gsc); free(sc_row); CVI_RT_MemFree(rt, mem); CVI_RT_DeInit(rt);
  return 1;
}

int main(void) {
  printf("===== Gate ① M>1 two-pass INT8 matmul readback (chunk_matmul_twopass semantics) =====\n");
  int rc = 0;
  rc += run_twopass(16, 32, 512, "G1");   /* [16,32]x[32,512]: res 8KB + right 16KB + left 0.5KB */
  rc += run_twopass(32, 32, 256, "G1");   /* [32,32]x[32,256]: res 8KB + right 8KB + left 1KB */
  rc += run_twopass(64, 32, 128, "G1");   /* MAX_SEQ=64 reference row count */
  rc += run_twopass(1,  32, 512, "4b");   /* ④b decode M=1 [1,32]x[32,512]: right 16KB */
  printf("===== rc=%d (%s) =====\n", rc, rc == 0 ? "ALL PASS" : "FAIL");
  return rc ? 1 : 0;
}
