/* gate2_realshape.c — Gate ① supplementary REAL-SHAPE on-board verification
 *   (TPU 底层工程师, 2026-08-13, response to 推理引擎工程师 request #3).
 *
 * Answers four supplemental questions for the two-pass Path A/B engine:
 *
 *  Q1  Full-K standard INT8 matmul (ps32-free) + l2g readback at REAL shapes:
 *        decode  [1,896]x[896,4864]  N-tiled (up/gate/down decode)
 *        prefill [4,896]x[896,896] and [8,896]x[896,896] (q/k/v/wo prefill)
 *      Verify element-exact vs host int32 ref with round-half-up (TIU semantics).
 *
 *  Q2  Gate ④ real-shape re-confirm: per-call rshift_bits change between
 *      pass1(rsafe) and pass2(r_opt) on the SAME lmem weight block, NO g2l
 *      reload, at K=896. (chunk_matmul_twopass flow on a single tile.)
 *
 *  Q3  ION allocator single-buffer limit: can CVI_RT_MemAlloc give
 *      >= 4.36MB (int8 up full [896,4864]) and 8.72MB (double-buffer)?
 *
 *  Q4  M-limit for [M,896]x[896,Nt]: max M that fits lmem with N-tiling
 *      (determines prefill batch shape).
 *
 * Also a small w-sweep at K=32 to pin down the minimum res/right width the
 * bm1822 TIU accepts for INT8 matmul (feeds the N-tile legality check).
 *
 * Build: make gate2_realshape ; run: duo_run.py gate2_realshape
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include "cviruntime_context.h"
#include "bmkernel/bm1822/bmkernel_1822.h"

#define LEFT_OFF  0
#define RIGHT_OFF 32768
#define OUT_OFF   (RIGHT_OFF + 896*4864)   /* 4,390,912 */
#define MAIN_BUF  (5u*1024u*1024u)

static inline int sat8(int v){ return v > 127 ? 127 : (v < -128 ? -128 : v); }
static inline int8_t ref_div(int32_t acc, int rs){
  if (rs <= 0) return (int8_t)sat8(acc);
  return (int8_t)sat8((acc + (1 << (rs - 1))) >> rs);
}

/* run a built cmdbuf; returns 0 ok */
static int run_cmdbuf(CVI_RT_HANDLE rt, uint64_t pa, uint8_t *cmdbuf, uint32_t cmd_sz){
  uint32_t psz, pmu; bmk1822_dmabuf_size(cmdbuf, cmd_sz, &psz, &pmu);
  CVI_RT_MEM dm = CVI_RT_MemAlloc(rt, psz + pmu);
  if (!dm) { printf("    dmabuf alloc fail (cmd_sz=%u psz=%u pmu=%u)\n", cmd_sz, psz, pmu); return -1; }
  uint8_t *dv = CVI_RT_MemGetVAddr(dm);
  bmk1822_dmabuf_convert(cmdbuf, cmd_sz, dv);
  bmk1822_arraybase_set(dv, pa, 0, 0, 0);
  CVI_RT_MemFlush(rt, dm);
  CVI_RT_MEM loaded;
  CVI_RT_LoadDmabuf(rt, dm, psz + pmu, pa, 0, false, &loaded);
  CVI_RT_RunCmdbufEx(rt, loaded, &(CVI_RT_ARRAYBASE){.gaddr_base0 = pa});
  CVI_RT_MemFree(rt, dm);
  return 0;
}

/* Q3: ION allocator scan */
static void ion_alloc_scan(void){
  CVI_RT_HANDLE rt;
  if (CVI_RT_Init(&rt)) { printf("  ION scan: rt init fail\n"); return; }
  const char *names[] = {"1MB", "2MB", "4MB", "4.36MB", "6MB", "8MB", "8.72MB", "10MB", "12MB", "16MB", "20MB", "24MB", "26MB"};
  const double mb[] = {1, 2, 4, 4.36, 6, 8, 8.72, 10, 12, 16, 20, 24, 26};
  int nmax = 0;
  printf("--- Q3: ION single-buffer allocator scan (carveout ~26.8MB) ---\n");
  for (int i = 0; i < 13; i++) {
    size_t sz = (size_t)(mb[i] * 1024 * 1024);
    CVI_RT_MEM m = CVI_RT_MemAlloc(rt, sz);
    if (m) {
      uint64_t pa = CVI_RT_MemGetPAddr(m);
      void *va = CVI_RT_MemGetVAddr(m);
      printf("  alloc %-6s = %8zu B  OK  pa=0x%llx va=%p\n", names[i], sz,
             (unsigned long long)pa, va);
      if (sz > (size_t)(nmax * 1024 * 1024)) nmax = (int)mb[i];
      CVI_RT_MemFree(rt, m);
    } else {
      printf("  alloc %-6s = %8zu B  FAIL\n", names[i], sz);
    }
  }
  printf("  >> largest single ION alloc OK = %d MB\n", nmax);
  CVI_RT_DeInit(rt);
}

/* Q1: full-K INT8 matmul [M,K]x[K,N] N-tiled, l2g readback, verify vs host.
 * strided=1 -> g2l/l2g tiles with tgmem stride=N (real full-matrix layout);
 * strided=0 -> tiles stored contiguous (fallback).  */
static int fullK_tile_verify(CVI_RT_HANDLE rt, CVI_RT_MEM mem, uint64_t pa, uint8_t *va,
                             int M, int K, int N, int Nt, int rshift, int strided,
                             const char *tag){
  int ntiles = N / Nt;
  int8_t *left = malloc((size_t)M * K);
  int8_t *right = malloc((size_t)K * N);
  int32_t *acc = malloc((size_t)M * N * sizeof(int32_t));
  if (!left || !right || !acc) { printf("  [%s] host oom\n", tag); return 1; }
  srand((unsigned)(K * 131 + N * 17 + M));
  for (int m = 0; m < M; m++) for (int k = 0; k < K; k++) left[m * K + k] = (int8_t)(rand() % 200 - 100);
  for (int k = 0; k < K; k++) for (int n = 0; n < N; n++) right[k * N + n] = (int8_t)(rand() % 200 - 100);
  for (int m = 0; m < M; m++) for (int n = 0; n < N; n++) {
    int32_t s = 0; for (int k = 0; k < K; k++) s += (int32_t)left[m * K + k] * right[k * N + n];
    acc[m * N + n] = s;
  }
  if (strided) {
    memcpy(va + LEFT_OFF, left, (size_t)M * K);
    memcpy(va + RIGHT_OFF, right, (size_t)K * N);
  } else {
    /* tile-contiguous: tile t at RIGHT_OFF + t*K*Nt */
    memcpy(va + LEFT_OFF, left, (size_t)M * K);
    for (int t = 0; t < ntiles; t++)
      for (int k = 0; k < K; k++)
        memcpy(va + RIGHT_OFF + (size_t)t * K * Nt + (size_t)k * Nt,
               right + (size_t)k * N + (size_t)t * Nt, Nt);
  }
  CVI_RT_MemFlush(rt, mem);

  uint8_t cmdbuf[262144] __attribute__((aligned(16)));
  bmk_info_t info = {.chip_version = 1822, .cmdbuf_size = sizeof(cmdbuf), .cmdbuf = cmdbuf};
  bmk1822_context_t *bmk = bmk1822_register(&info);
  bmk1822_matrix_lmem_shape_t sl = {.n=(uint32_t)M, .c=1, .w=(uint32_t)K, .col=(uint32_t)K};
  bmk1822_matrix_lmem_shape_t sr = {.n=(uint32_t)K, .c=1, .w=(uint32_t)Nt, .col=(uint32_t)Nt};
  bmk1822_matrix_lmem_shape_t so = {.n=(uint32_t)M, .c=1, .w=(uint32_t)Nt, .col=(uint32_t)Nt};
  bmk1822_matrix_lmem_t *ml_l = bmk1822_lmem_alloc_matrix(bmk, sl, FMT_I8, 1);
  bmk1822_matrix_lmem_t *ml_r = bmk1822_lmem_alloc_matrix(bmk, sr, FMT_I8, 1);
  bmk1822_matrix_lmem_t *ml_o = bmk1822_lmem_alloc_matrix(bmk, so, FMT_I8, 1);
  uint32_t lsz = bmk1822_lmem_matrix_to_size(bmk, sl, FMT_I8, 1)
               + bmk1822_lmem_matrix_to_size(bmk, sr, FMT_I8, 1)
               + bmk1822_lmem_matrix_to_size(bmk, so, FMT_I8, 1);
  if (!ml_l || !ml_r || !ml_o) {
    printf("  [%s] M=%d K=%d N=%d Nt=%d lmem=%u: ALLOC FAIL\n", tag, M, K, N, Nt, lsz);
    free(left); free(right); free(acc); bmk1822_cleanup(bmk); return 1;
  }
  bmk1822_matrix_tgmem_t mg_l = {0, LEFT_OFF, FMT_I8, {(uint32_t)M, (uint32_t)K}, {(uint32_t)K}};
  bmk1822_tdma_g2l_matrix_copy(bmk, &(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_l, ml_l});
  for (int t = 0; t < ntiles; t++) {
    uint32_t n0 = (uint32_t)t * Nt;
    bmk1822_matrix_tgmem_t mg_r;
    if (strided) mg_r = (bmk1822_matrix_tgmem_t){0, RIGHT_OFF + n0, FMT_I8, {(uint32_t)K, (uint32_t)Nt}, {(uint32_t)N}};
    else         mg_r = (bmk1822_matrix_tgmem_t){0, RIGHT_OFF + (uint32_t)t * K * Nt, FMT_I8, {(uint32_t)K, (uint32_t)Nt}, {(uint32_t)Nt}};
    bmk1822_tdma_g2l_matrix_copy(bmk, &(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_r, ml_r});
    bmk1822_tiu_matrix_multiplication_param_t mm = {
      .res = ml_o, .left = ml_l, .right = ml_r, .bias = NULL,
      .lshift_bits = 0, .rshift_bits = (uint8_t)rshift, .res_is_int8 = 1,
      .relu_enable = 0, .add_result = 0, .ps32_mode = 0, .layer_id = (uint16_t)(t + 1)};
    if (!bmk1822_tiu_matrix_multiplication(bmk, &mm)) {
      printf("  [%s] tile %d REJECTED\n", tag, t); goto fail;
    }
    bmk1822_matrix_tgmem_t mg_o;
    if (strided) mg_o = (bmk1822_matrix_tgmem_t){0, OUT_OFF + n0, FMT_I8, {(uint32_t)M, (uint32_t)Nt}, {(uint32_t)N}};
    else         mg_o = (bmk1822_matrix_tgmem_t){0, OUT_OFF + (uint32_t)t * M * Nt, FMT_I8, {(uint32_t)M, (uint32_t)Nt}, {(uint32_t)Nt}};
    bmk1822_tdma_l2g_matrix_copy(bmk, &(bmk1822_tdma_l2tg_matrix_copy_param_t){ml_o, &mg_o});
  }
  uint32_t cmd_sz; uint8_t *cmd = bmk1822_acquire_cmdbuf(bmk, &cmd_sz);
  struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
  int rc = run_cmdbuf(rt, pa, cmd, cmd_sz);
  clock_gettime(CLOCK_MONOTONIC, &t1);
  if (rc) { printf("  [%s] cmdbuf run fail\n", tag); goto fail; }
  CVI_RT_MemInvld(rt, mem);
  double ms = (double)(t1.tv_sec - t0.tv_sec) * 1000.0 + (double)(t1.tv_nsec - t0.tv_nsec) / 1e6;

  int8_t *out = (int8_t *)(va + OUT_OFF);
  long bad = 0; int first = -1;
  for (int m = 0; m < M; m++) for (int n = 0; n < N; n++) {
    int8_t hv = ref_div(acc[m * N + n], rshift);
    if (out[m * N + n] != hv) { bad++; if (first < 0) first = n; }
  }
  printf("  [%s] [%d,%d]x[%d,%d] Nt=%d ntiles=%d lmem=%u | rshift=%d | ok=%ld/%ld bad=%ld first@%d | %.2f ms (%.2f ms/tile)\n",
         tag, M, K, K, N, Nt, ntiles, lsz, rshift, (long)M * N - bad, (long)M * N, bad, first, ms, ms / ntiles);
  bmk1822_cleanup(bmk);
  free(left); free(right); free(acc);
  return bad ? 1 : 0;
fail:
  bmk1822_cleanup(bmk); free(left); free(right); free(acc); return 1;
}

/* Q2: real-shape two-pass on ONE [1,896]x[896,32] tile, gate ④ no-reload */
static int twopass_realshape(CVI_RT_HANDLE rt, CVI_RT_MEM mem, uint64_t pa, uint8_t *va){
  int M = 1, K = 896, Nt = 32;
  int8_t *left = malloc((size_t)M * K);
  int8_t *right = malloc((size_t)K * Nt);
  int32_t *acc = malloc((size_t)M * Nt * sizeof(int32_t));
  srand(20260813);
  for (int k = 0; k < K; k++) left[k] = (int8_t)(rand() % 200 - 100);
  for (int k = 0; k < K; k++) for (int n = 0; n < Nt; n++) right[k * Nt + n] = (int8_t)(rand() % 200 - 100);
  for (int n = 0; n < Nt; n++) {
    int32_t s = 0; for (int k = 0; k < K; k++) s += (int32_t)left[k] * right[k * Nt + n];
    acc[n] = s;
  }
  int wmax = 0; for (int i = 0; i < K * Nt; i++) { int a = right[i]; if (a < 0) a = -a; if (a > wmax) wmax = a; }
  int rsafe = 0; long long md = (long long)K * 127 * wmax; while ((md >> rsafe) > 127) rsafe++;
  rsafe -= 3; if (rsafe < 4) rsafe = 4;
  memcpy(va + LEFT_OFF, left, (size_t)M * K);
  memcpy(va + RIGHT_OFF, right, (size_t)K * Nt);
  CVI_RT_MemFlush(rt, mem);

  /* pass1 */
  uint8_t cmdbuf1[131072] __attribute__((aligned(16)));
  bmk_info_t info1 = {.chip_version = 1822, .cmdbuf_size = sizeof(cmdbuf1), .cmdbuf = cmdbuf1};
  bmk1822_context_t *bmk1 = bmk1822_register(&info1);
  bmk1822_matrix_lmem_shape_t sl = {.n=(uint32_t)M, .c=1, .w=(uint32_t)K, .col=(uint32_t)K};
  bmk1822_matrix_lmem_shape_t sr = {.n=(uint32_t)K, .c=1, .w=(uint32_t)Nt, .col=(uint32_t)Nt};
  bmk1822_matrix_lmem_shape_t so = {.n=(uint32_t)M, .c=1, .w=(uint32_t)Nt, .col=(uint32_t)Nt};
  bmk1822_matrix_lmem_t *ml_l = bmk1822_lmem_alloc_matrix(bmk1, sl, FMT_I8, 1);
  bmk1822_matrix_lmem_t *ml_r = bmk1822_lmem_alloc_matrix(bmk1, sr, FMT_I8, 1);
  bmk1822_matrix_lmem_t *ml_o = bmk1822_lmem_alloc_matrix(bmk1, so, FMT_I8, 1);
  if (!ml_l || !ml_r || !ml_o) { printf("  [Q2] pass1 ALLOC FAIL\n"); return 1; }
  bmk1822_matrix_tgmem_t mg_l = {0, LEFT_OFF, FMT_I8, {(uint32_t)M, (uint32_t)K}, {(uint32_t)K}};
  bmk1822_matrix_tgmem_t mg_r = {0, RIGHT_OFF, FMT_I8, {(uint32_t)K, (uint32_t)Nt}, {(uint32_t)Nt}};
  bmk1822_tdma_g2l_matrix_copy(bmk1, &(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_l, ml_l});
  bmk1822_tdma_g2l_matrix_copy(bmk1, &(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_r, ml_r});
  bmk1822_tiu_matrix_multiplication_param_t mm1 = {
    .res = ml_o, .left = ml_l, .right = ml_r, .bias = NULL, .lshift_bits = 0,
    .rshift_bits = (uint8_t)rsafe, .res_is_int8 = 1, .relu_enable = 0, .add_result = 0,
    .ps32_mode = 0, .layer_id = 1};
  if (!bmk1822_tiu_matrix_multiplication(bmk1, &mm1)) { printf("  [Q2] P1 REJECTED\n"); return 1; }
  bmk1822_matrix_tgmem_t mg_p1 = {0, OUT_OFF, FMT_I8, {(uint32_t)M, (uint32_t)Nt}, {(uint32_t)Nt}};
  bmk1822_tdma_l2g_matrix_copy(bmk1, &(bmk1822_tdma_l2tg_matrix_copy_param_t){ml_o, &mg_p1});
  uint32_t cmd_sz; uint8_t *cmd = bmk1822_acquire_cmdbuf(bmk1, &cmd_sz);
  if (run_cmdbuf(rt, pa, cmd, cmd_sz)) { printf("  [Q2] P1 run fail\n"); return 1; }
  CVI_RT_MemInvld(rt, mem);
  int8_t *p1 = (int8_t *)(va + OUT_OFF);
  int bad1 = 0, maxabs = 0;
  for (int n = 0; n < Nt; n++) {
    if (p1[n] != ref_div(acc[n], rsafe)) bad1++;
    int a = p1[n]; if (a < 0) a = -a; if (a > maxabs) maxabs = a;
  }
  long long est = (long long)maxabs << rsafe;
  int r_opt = 0; while (est > (127LL << r_opt)) r_opt++;
  bmk1822_cleanup(bmk1);

  /* pass2: fresh ctx, same lmem layout, rshift=r_opt, NO g2l right */
  uint8_t cmdbuf2[131072] __attribute__((aligned(16)));
  bmk_info_t info2 = {.chip_version = 1822, .cmdbuf_size = sizeof(cmdbuf2), .cmdbuf = cmdbuf2};
  bmk1822_context_t *bmk2 = bmk1822_register(&info2);
  bmk1822_matrix_lmem_t *l2 = bmk1822_lmem_alloc_matrix(bmk2, sl, FMT_I8, 1);
  bmk1822_matrix_lmem_t *r2 = bmk1822_lmem_alloc_matrix(bmk2, sr, FMT_I8, 1);
  bmk1822_matrix_lmem_t *o2 = bmk1822_lmem_alloc_matrix(bmk2, so, FMT_I8, 1);
  if (!l2 || !r2 || !o2) { printf("  [Q2] pass2 ALLOC FAIL\n"); return 1; }
  bmk1822_tiu_matrix_multiplication_param_t mm2 = {
    .res = o2, .left = l2, .right = r2, .bias = NULL, .lshift_bits = 0,
    .rshift_bits = (uint8_t)r_opt, .res_is_int8 = 1, .relu_enable = 0, .add_result = 0,
    .ps32_mode = 0, .layer_id = 2};
  if (!bmk1822_tiu_matrix_multiplication(bmk2, &mm2)) { printf("  [Q2] P2 REJECTED\n"); return 1; }
  bmk1822_matrix_tgmem_t mg_p2 = {0, OUT_OFF, FMT_I8, {(uint32_t)M, (uint32_t)Nt}, {(uint32_t)Nt}};
  bmk1822_tdma_l2g_matrix_copy(bmk2, &(bmk1822_tdma_l2tg_matrix_copy_param_t){o2, &mg_p2});
  cmd = bmk1822_acquire_cmdbuf(bmk2, &cmd_sz);
  if (run_cmdbuf(rt, pa, cmd, cmd_sz)) { printf("  [Q2] P2 run fail\n"); return 1; }
  CVI_RT_MemInvld(rt, mem);
  int8_t *p2 = (int8_t *)(va + OUT_OFF);
  int bad2 = 0;
  for (int n = 0; n < Nt; n++) if (p2[n] != ref_div(acc[n], r_opt)) bad2++;
  printf("  [Q2] K=896 M=1 Nt=32 | wmax=%d rsafe=%d | P1 ok=%d/%d bad=%d | maxabs=%d r_opt=%d | P2 ok=%d/%d bad=%d (no-reload) %s\n",
         wmax, rsafe, Nt, Nt - bad1, bad1, maxabs, r_opt, Nt, Nt - bad2, bad2,
         (bad1 + bad2) == 0 ? "PASS" : "FAIL");
  bmk1822_cleanup(bmk2);
  free(left); free(right); free(acc);
  return bad1 + bad2;
}

/* Q4: M-limit scan for [M,896]x[896,Nt] (prefill) — find largest M that fits */
static void mlimit_scan(void){
  printf("--- Q4: M-limit scan [M,896]x[896,Nt] (prefill) ---\n");
  CVI_RT_HANDLE rt; CVI_RT_Init(&rt);
  CVI_RT_MEM mem = CVI_RT_MemAlloc(rt, 1u << 20);
  if (!mem) { printf("  scan mem alloc fail\n"); return; }
  uint64_t pa = CVI_RT_MemGetPAddr(mem);
  (void)pa; (void)rt;
  int Ms[] = {1, 2, 4, 8, 16, 32, 64};
  for (int mi = 0; mi < 7; mi++) {
    int M = Ms[mi], bestNt = 0; uint32_t bestL = 0;
    int nts[] = {32, 16, 8, 4};
    for (int ti = 0; ti < 4 && !bestNt; ti++) {
      int Nt = nts[ti];
      uint8_t cmdbuf[65536] __attribute__((aligned(16)));
      bmk_info_t info = {.chip_version = 1822, .cmdbuf_size = sizeof(cmdbuf), .cmdbuf = cmdbuf};
      bmk1822_context_t *bmk = bmk1822_register(&info);
      bmk1822_matrix_lmem_shape_t sl = {.n=(uint32_t)M, .c=1, .w=896, .col=896};
      bmk1822_matrix_lmem_shape_t sr = {.n=896, .c=1, .w=(uint32_t)Nt, .col=(uint32_t)Nt};
      bmk1822_matrix_lmem_shape_t so = {.n=(uint32_t)M, .c=1, .w=(uint32_t)Nt, .col=(uint32_t)Nt};
      bmk1822_matrix_lmem_t *a = bmk1822_lmem_alloc_matrix(bmk, sl, FMT_I8, 1);
      bmk1822_matrix_lmem_t *b = bmk1822_lmem_alloc_matrix(bmk, sr, FMT_I8, 1);
      bmk1822_matrix_lmem_t *c = bmk1822_lmem_alloc_matrix(bmk, so, FMT_I8, 1);
      uint32_t lsz = bmk1822_lmem_matrix_to_size(bmk, sl, FMT_I8, 1)
                   + bmk1822_lmem_matrix_to_size(bmk, sr, FMT_I8, 1)
                   + bmk1822_lmem_matrix_to_size(bmk, so, FMT_I8, 1);
      if (a && b && c) { bestNt = Nt; bestL = lsz; }
      bmk1822_cleanup(bmk);
    }
    if (bestNt) printf("  M=%2d: fits with Nt=%-2d lmem=%u\n", M, bestNt, bestL);
    else        printf("  M=%2d: NO Nt fits (need M-tile)\n", M);
  }
  CVI_RT_MemFree(rt, mem); CVI_RT_DeInit(rt);
}

int main(void){
  setvbuf(stdout, NULL, _IONBF, 0);   /* unbuffer: keep output on crash */
  printf("===== gate2_realshape: real-shape two-pass INT8 readback supplemental =====\n");

  /* Q2 first: small, isolated two-pass at real K=896 (isolate the earlier crash) */
  {
    printf("--- Q2: real-shape two-pass gate ④ (no-reload) ---\n");
    CVI_RT_HANDLE rt; if (CVI_RT_Init(&rt)) { printf("  rt init fail\n"); return 1; }
    CVI_RT_MEM mem = CVI_RT_MemAlloc(rt, MAIN_BUF);
    if (!mem) { printf("  MAIN_BUF %u alloc FAIL\n", MAIN_BUF); CVI_RT_DeInit(rt); return 1; }
    uint64_t pa = CVI_RT_MemGetPAddr(mem);
    uint8_t *va = CVI_RT_MemGetVAddr(mem);
    CVI_RT_SetBaseReg(rt, 0, pa);
    memset(va, 0, MAIN_BUF);
    int r3 = twopass_realshape(rt, mem, pa, va);
    CVI_RT_MemFree(rt, mem); CVI_RT_DeInit(rt);
    if (r3) printf("  Q2 FAIL\n");
  }

  /* Q0 + Q1 + Q4: full-K real-shape readback */
  {
    printf("--- Q0: w-sweep at K=32 M=1 (pin min res/right width) ---\n");
    CVI_RT_HANDLE rt; if (CVI_RT_Init(&rt)) { printf("  rt init fail\n"); return 1; }
    CVI_RT_MEM mem = CVI_RT_MemAlloc(rt, MAIN_BUF);
    if (!mem) { printf("  MAIN_BUF %u alloc FAIL\n", MAIN_BUF); CVI_RT_DeInit(rt); return 1; }
    uint64_t pa = CVI_RT_MemGetPAddr(mem);
    uint8_t *va = CVI_RT_MemGetVAddr(mem);
    CVI_RT_SetBaseReg(rt, 0, pa);
    memset(va, 0, MAIN_BUF);

    int ws[] = {8, 16, 32, 64, 128, 192};
    for (int i = 0; i < 6; i++)
      fullK_tile_verify(rt, mem, pa, va, 1, 32, ws[i], ws[i], 8, 1, "w-sweep");

    printf("--- Q1: full-K real-shape INT8 matmul + l2g readback ---\n");
    fullK_tile_verify(rt, mem, pa, va, 1, 896, 4864, 32, 8, 1, "decode");
    fullK_tile_verify(rt, mem, pa, va, 4, 896, 896, 16, 8, 1, "prefill4");
    fullK_tile_verify(rt, mem, pa, va, 8, 896, 896, 16, 8, 1, "prefill8");

    CVI_RT_MemFree(rt, mem); CVI_RT_DeInit(rt);
  }

  printf("--- Q4: M-limit scan ---\n");
  mlimit_scan();

  /* Q3 last: may assert at the top end of the carveout */
  printf("--- Q3: ION allocator scan (last; 24/26MB may assert) ---\n");
  ion_alloc_scan();

  printf("===== gate2_realshape done =====\n");
  return 0;
}
