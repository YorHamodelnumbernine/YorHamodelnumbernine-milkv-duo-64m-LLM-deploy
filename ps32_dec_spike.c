/* ps32_dec_spike.c — DECISIVE spike for CEO A/B ruling (2026-08-13).
 *
 * Question 1 (PART A): does ps32_mode==2 on CV1800B (bmk1822) truly emit fp32
 *   and is it readable via the SAME DMA path as int8 (tdma_l2g general_copy +
 *   CPU MemInvld)?  Use small K=32, M=1, N=32, left=right=100 -> host acc=320000.
 *   Configs C0..C5 sweep ps32_mode x res_is_int8 x alloc fmt.  Read back FULL
 *   32KB lmem region raw, then decode with every plausible layout (int32
 *   contiguous / byte-plane, fp32 contiguous / byte-plane, bf16) and also
 *   scan the whole region for the exact bit patterns of 320000 (int32/fp32/bf16).
 *
 * Question 2 (PART B): element_wise F32/BF16 support.  Run mul/mac with
 *   FMT_I8/I16 (int control), FMT_BF16, FMT_F32.  F32 transport via tensor
 *   copy is asserted off (fmt==I8||U8||BF16), so load via general_copy raw.
 *
 * Question 3 (PART C): measured submit cost for Qwen shapes (q/k/v/o N=896,
 *   up/gate N=4864, down K=4864) — batch build+run per-op cost at the
 *   K=32-group, N=192-chunk granularity that hardware enforces.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>
#include <sys/wait.h>
#include "cviruntime_context.h"
#include "bmkernel/bm1822/bmkernel_1822.h"

#define NEURON_SZ 262144
#define OUT_OFF   65536

/* ------------------------------------------------------------------ */
/* helpers                                                            */
/* ------------------------------------------------------------------ */
static uint32_t rd(const uint8_t *p){ uint32_t v=0; for(int b=0;b<4;b++) v|=((uint32_t)p[b])<<(8*b); return v; }
static uint32_t rd_bp(const uint8_t *p, uint32_t stride, int idx){
  uint32_t v=0; for(int b=0;b<4;b++) v|=((uint32_t)p[(uint32_t)b*stride+idx])<<(8*b); return v; }

static void print_ps32_md(const uint8_t *cmdbuf, uint32_t sz) {
  /* walk bmk1822 cmdbuf descriptors (magic 0xA6) and print ps32_md of TIU cmds */
  uint32_t p = 0;
  int found = 0;
  while (p + 8 <= sz) {
    uint8_t magic = cmdbuf[p];
    if (magic != 0xA6) { p++; continue; }         /* resync */
    uint8_t len = cmdbuf[p+1];
    uint8_t eng = cmdbuf[p+2] & 0xF;              /* engine_id low nibble */
    uint32_t total = (len != 0) ? (uint32_t)len : (*(uint32_t*)(cmdbuf+p+4));
    if (total < 8 || p + total > sz) { p += 8; continue; }
    if (eng == 0 && total >= 8+56) {              /* TIU, >= one 14-dword bank */
      const uint32_t *r = (const uint32_t*)(cmdbuf+p+8);
      uint32_t tsk  = (r[0] >> 5) & 0xF;
      uint32_t eu   = (r[0] >> 9) & 0x1F;
      uint32_t pmd  = (r[3] >> 11) & 0x3;
      printf("    cmdbuf TIU@+%u len=%u tsk=%u eu=%u ps32_md=%u\n", p, total, tsk, eu, pmd);
      found++;
      /* one bank per descriptor is the common case; stop scanning banks */
    }
    p += total;
  }
  if (!found) printf("    (no TIU descriptor parsed; raw cmdbuf walk yielded none)\n");
}

/* run a possibly-asserting function in a child; parent reports outcome */
static void run_fork(const char *tag, void (*fn)(void)) {
  fflush(stdout);
  pid_t pid = fork();
  if (pid == 0) { fn(); fflush(stdout); fflush(stderr); _exit(0); }
  int st = 0; waitpid(pid, &st, 0);
  if (WIFSIGNALED(st))
    printf("[fork|%s] CHILD SIGNALED sig=%d (%s)\n", tag, WTERMSIG(st),
           WTERMSIG(st)==SIGABRT ? "ABORT=allocator assert" : "?");
  else if (WIFEXITED(st))
    printf("[fork|%s] child rc=%d\n", tag, WEXITSTATUS(st));
}

/* alloc-probe wrapper: tries to alloc ps32 matrix with given fmt (may assert) */
typedef struct { int ps32_mode, res_is_int8; fmt_t fmt; } ps32_arg_t;
static ps32_arg_t g_ps32_arg;
static void alloc_probe_child(void) {
  CVI_RT_HANDLE rt; CVI_RT_Init(&rt);
  CVI_RT_MEM mem = CVI_RT_MemAlloc(rt, 65536);
  uint8_t *va = CVI_RT_MemGetVAddr(mem);
  CVI_RT_SetBaseReg(rt, 0, CVI_RT_MemGetPAddr(mem));
  memset(va, 0, 65536);
  uint8_t cmdbuf[16384] __attribute__((aligned(16)));
  bmk_info_t info = { .chip_version = 1822, .cmdbuf_size = sizeof(cmdbuf), .cmdbuf = cmdbuf };
  bmk1822_context_t *bmk = bmk1822_register(&info);
  bmk1822_matrix_lmem_shape_t so = { .n=1, .c=1, .w=32, .col=32 };
  bmk1822_matrix_lmem_t *r = bmk1822_lmem_alloc_ps32_matrix(bmk, so, g_ps32_arg.fmt, 1);
  printf("[alloc_probe fmt=%d] ps32 alloc OK addr=%u size=%u\n",
         g_ps32_arg.fmt, r ? r->start_address : 0,
         r ? bmk1822_lmem_ps32_matrix_to_size(bmk, so, g_ps32_arg.fmt, 1) : 0);
  bmk1822_cleanup(bmk);
  CVI_RT_MemFree(rt, mem);
  CVI_RT_DeInit(rt);
}

/* ------------------------------------------------------------------ */
/* PART A — ps32_mode==2 fp32 readability                             */
/* ------------------------------------------------------------------ */
static const char *g_ps32_tag;
static void ps32_case_child(void) {
  int ps32_mode = g_ps32_arg.ps32_mode;
  int res_is_int8 = g_ps32_arg.res_is_int8;
  fmt_t alloc_fmt = g_ps32_arg.fmt;
  const char *tag = g_ps32_tag;
  int N = 32;
  CVI_RT_HANDLE rt; CVI_RT_Init(&rt);
  CVI_RT_MEM mem = CVI_RT_MemAlloc(rt, NEURON_SZ);
  uint64_t pa = CVI_RT_MemGetPAddr(mem);
  uint8_t *va = CVI_RT_MemGetVAddr(mem);
  CVI_RT_SetBaseReg(rt, 0, pa);
  memset(va, 0, NEURON_SZ);
  int M = 1, K = 32;
  int8_t *left = (int8_t*)(va + 0);
  int8_t *right = (int8_t*)(va + 2048);
  for (int k = 0; k < K; k++) {
    left[k] = 100;
    for (int n = 0; n < N; n++) right[k*N + n] = 100;
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
  bmk1822_matrix_lmem_t *ml_res = bmk1822_lmem_alloc_ps32_matrix(bmk, so, alloc_fmt, 1);
  if (!ml_l || !ml_r || !ml_res) {
    printf("[A|%s] ALLOC FAIL l=%p r=%p res=%p (fmt=%d)\n", tag, (void*)ml_l,(void*)ml_r,(void*)ml_res, alloc_fmt);
    goto outA;
  }
  uint32_t psz = bmk1822_lmem_ps32_matrix_to_size(bmk, so, alloc_fmt, 1);
  printf("[A|%s] ps32=%d res_i8=%d alloc_fmt=%d res_addr=%u psz=%u\n",
         tag, ps32_mode, res_is_int8, alloc_fmt, ml_res->start_address, psz);

  bmk1822_matrix_tgmem_t mg_l = {0, 0, FMT_I8, {M,K}, {(uint32_t)K}};
  bmk1822_matrix_tgmem_t mg_r = {0, 2048, FMT_I8, {K,N}, {(uint32_t)N}};
  bmk1822_tdma_g2l_matrix_copy(bmk, &(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_l, ml_l});
  bmk1822_tdma_g2l_matrix_copy(bmk, &(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_r, ml_r});

  memset(va + 8192, 0, 32768); CVI_RT_MemFlush(rt, mem);
  bmk1822_tdma_g2l_general_copy(bmk, &(bmk1822_tdma_tg2l_general_copy_param_t){
    .src_base_reg_index = 0, .src_address = 8192, .dst_address = ml_res->start_address, .bytes = 32768 });

  bmk1822_tiu_matrix_multiplication_param_t p = {
    .res = ml_res, .left = ml_l, .right = ml_r, .bias = NULL,
    .lshift_bits = 0, .rshift_bits = 0, .res_is_int8 = res_is_int8, .relu_enable = 0,
    .add_result = 0, .ps32_mode = (uint8_t)ps32_mode, .layer_id = 1 };
  if (!bmk1822_tiu_matrix_multiplication(bmk, &p)) { printf("[A|%s] MATMUL REJECTED\n", tag); goto outA; }

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
  printf("  ps32_md reg trace:\n");
  print_ps32_md(cmd, cmd_sz);

  uint8_t *r = (uint8_t*)(va + OUT_OFF);
  /* decodes */
  int c_i32_cont=0, c_i32_bp=0, c_f32_cont=0, c_f32_bp=0, c_bf16_cont=0;
  for (int n = 0; n < N; n++) {
    uint32_t i32c = rd(r + 4*n);
    uint32_t i32b = rd_bp(r, (uint32_t)N, n);
    if ((int32_t)i32c == 320000) c_i32_cont++;
    if ((int32_t)i32b == 320000) c_i32_bp++;
    float fc; memcpy(&fc, r+4*n, 4);
    if (fc == 320000.0f) c_f32_cont++;
    uint8_t fbp[4]; for (int b=0;b<4;b++) fbp[b]=r[(uint32_t)b*N+n];
    memcpy(&fc, fbp, 4);
    if (fc == 320000.0f) c_f32_bp++;
    uint16_t bf; bf = (uint16_t)r[2*n] | ((uint16_t)r[2*n+1]<<8);
    if (bf == 0x489C) c_bf16_cont++;   /* bf16(320000.0) = 0x489C */
  }
  /* whole-region pattern scan */
  int pat_i32=0, pat_f32=0, pat_bf16=0;
  for (uint32_t i = 0; i + 4 <= 32768; i++) {
    if (r[i]==0x00 && r[i+1]==0xE2 && r[i+2]==0x04 && r[i+3]==0x00) pat_i32++;
    if (r[i]==0x00 && r[i+1]==0x40 && r[i+2]==0x9C && r[i+3]==0x48) pat_f32++;
  }
  for (uint32_t i = 0; i + 2 <= 32768; i++)
    if (r[i]==0x9C && r[i+1]==0x48) pat_bf16++;
  printf("  decode: i32_cont=%d/%d i32_bp=%d/%d f32_cont=%d/%d f32_bp=%d/%d bf16_cont=%d/%d\n",
         c_i32_cont, N, c_i32_bp, N, c_f32_cont, N, c_f32_bp, N, c_bf16_cont, N);
  printf("  pattern-scan(32KB): int32(320000)=%d fp32(320000.0)=%d bf16(0x489C)=%d\n",
         pat_i32, pat_f32, pat_bf16);
  printf("  first 64 bytes:");
  for (int i = 0; i < 64; i++) { if (i%16==0) printf("\n    "); printf("%02x ", r[i]); }
  printf("\n  col0 as int32_cont=%d int32_bp=%d fp32_cont=%f\n",
         (int32_t)rd(r), (int32_t)rd_bp(r,N,0), *(float*)r);
  printf("  col1 as int32_cont=%d int32_bp=%d fp32_cont=%f\n",
         (int32_t)rd(r+4), (int32_t)rd_bp(r,N,1), *(float*)(r+4));

  CVI_RT_MemFree(rt, dmabuf_mem);
outA:
  bmk1822_cleanup(bmk);
  CVI_RT_MemFree(rt, mem);
  CVI_RT_DeInit(rt);
}

/* ------------------------------------------------------------------ */
/* PART B — element_wise F32/BF16                                     */
/* ------------------------------------------------------------------ */
static void ew_mul_case(const char *tag, fmt_t fmt, int use_general_copy) {
  CVI_RT_HANDLE rt; CVI_RT_Init(&rt);
  CVI_RT_MEM mem = CVI_RT_MemAlloc(rt, 8192);
  uint64_t pa = CVI_RT_MemGetPAddr(mem);
  uint8_t *va = CVI_RT_MemGetVAddr(mem);
  CVI_RT_SetBaseReg(rt, 0, pa);
  memset(va, 0, 8192);
  int n = 4, esz = (fmt==FMT_F32)?4:(fmt==FMT_BF16||fmt==FMT_I16)?2:1;
  /* a = {1.5, 2.5, -1.5, 3.0}; b = {2.0, 2.0, 2.0, 2.0}; for int: a=i+1,b=2 */
  for (int i = 0; i < n; i++) {
    float af = (i==2)? -1.5f : (i==3)? 3.0f : (i+1)*1.0f + 0.5f;
    float bf = 2.0f;
    if (fmt == FMT_F32) { memcpy(va + i*4, &af, 4); memcpy(va + 32 + i*4, &bf, 4); }
    else if (fmt == FMT_BF16) {
      uint16_t ab = (uint16_t)((af>0?af:af)); /* bf16 = top16 of f32 */
      uint32_t ua; memcpy(&ua, &af, 4); ab = (uint16_t)(ua>>16);
      uint32_t ub; memcpy(&ub, &bf, 4);
      *(uint16_t*)(va + i*2) = ab; *(uint16_t*)(va + 32 + i*2) = (uint16_t)(ub>>16);
    } else { va[i] = (int8_t)(i+1); va[32+i] = 2; }
  }
  CVI_RT_MemFlush(rt, mem);

  uint8_t cmdbuf[16384] __attribute__((aligned(16)));
  bmk_info_t info = { .chip_version = 1822, .cmdbuf_size = sizeof(cmdbuf), .cmdbuf = cmdbuf };
  bmk1822_context_t *bmk = bmk1822_register(&info);

  bmk1822_tensor_lmem_shape_t ts = {1,1,1,(uint32_t)n};
  bmk1822_tensor_lmem_t *a = bmk1822_lmem_alloc_tensor(bmk, ts, fmt, 1);
  bmk1822_tensor_lmem_t *b = bmk1822_lmem_alloc_tensor(bmk, ts, fmt, 1);
  bmk1822_tensor_lmem_t *res = bmk1822_lmem_alloc_tensor(bmk, ts, fmt, 1);
  if (!a || !b || !res) {
    printf("[B|mul %s] ALLOC FAIL (fmt=%d)\n", tag, fmt); goto outB; }
  uint32_t a_off = (fmt==FMT_F32)?0 : (fmt==FMT_BF16)?0 : 0;
  uint32_t b_off = (fmt==FMT_F32)?32 : (fmt==FMT_BF16)?32 : 32;
  if (use_general_copy || fmt==FMT_F32) {
    uint32_t an = (uint32_t)(esz*n);
    bmk1822_tdma_g2l_general_copy(bmk, &(bmk1822_tdma_tg2l_general_copy_param_t){
      .src_base_reg_index=0, .src_address=a_off, .dst_address=a->start_address, .bytes=an});
    bmk1822_tdma_g2l_general_copy(bmk, &(bmk1822_tdma_tg2l_general_copy_param_t){
      .src_base_reg_index=0, .src_address=b_off, .dst_address=b->start_address, .bytes=an});
  } else {
    bmk1822_tensor_tgmem_t tg = {0,0,fmt,{1,1,1,(uint32_t)n},
      bmk1822_tensor_tgmem_default_stride((bmk1822_tensor_tgmem_shape_t){1,1,1,(uint32_t)n}, fmt)};
    tg.start_address = a_off; bmk1822_tdma_g2l_tensor_copy(bmk, &(bmk1822_tdma_tg2l_tensor_copy_param_t){&tg, a});
    tg.start_address = b_off; bmk1822_tdma_g2l_tensor_copy(bmk, &(bmk1822_tdma_tg2l_tensor_copy_param_t){&tg, b});
  }
  bmk1822_tiu_element_wise_mul_param_t p = {
    .res_high = NULL, .res_low = res, .a = a, .b_is_const = 0, .b = b,
    .rshift_bits = 0, .relu_enable = 0, .layer_id = 1 };
  if (!bmk1822_tiu_element_wise_mul(bmk, &p)) {
    printf("[B|mul %s] OP REJECTED (fmt=%d)\n", tag, fmt); goto outB; }
  /* read back raw */
  uint32_t rsz = (uint32_t)(esz*n);
  bmk1822_tdma_l2g_general_copy(bmk, &(bmk1822_tdma_l2tg_general_copy_param_t){
    .src_address = res->start_address, .dst_base_reg_index = 0, .dst_address = 256, .bytes = rsz });
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
  printf("[B|mul %s] fmt=%d out bytes:", tag, fmt);
  for (int i = 0; i < esz*n; i++) printf(" %02x", r[i]);
  printf("\n");
  /* interpret */
  if (fmt==FMT_F32) { float *f=(float*)r; printf("  as_f32[0..%d]:", n-1); for(int i=0;i<n;i++) printf(" %g", f[i]); printf(" (expect 3,5,-3,6)\n"); }
  else if (fmt==FMT_BF16) { uint16_t *h=(uint16_t*)r; printf("  as_bf16:", n); for(int i=0;i<n;i++){ uint32_t u=((uint32_t)h[i])<<16; float f; memcpy(&f,&u,4); printf(" %g",f);} printf("\n"); }
  else { int8_t *s=(int8_t*)r; printf("  as_i8[0..%d]:", n-1); for(int i=0;i<n;i++) printf(" %d", s[i]); printf(" (expect 2,4,6,8)\n"); }
  CVI_RT_MemFree(rt, dmabuf_mem);
outB:
  bmk1822_cleanup(bmk);
  CVI_RT_MemFree(rt, mem);
  CVI_RT_DeInit(rt);
}

static void ew_mac_case(const char *tag, fmt_t fmt) {
  CVI_RT_HANDLE rt; CVI_RT_Init(&rt);
  CVI_RT_MEM mem = CVI_RT_MemAlloc(rt, 8192);
  uint64_t pa = CVI_RT_MemGetPAddr(mem);
  uint8_t *va = CVI_RT_MemGetVAddr(mem);
  CVI_RT_SetBaseReg(rt, 0, pa);
  memset(va, 0, 8192);
  int n = 4, esz = (fmt==FMT_F32)?4:(fmt==FMT_BF16)?2:1;
  int res_is_int8 = (fmt==FMT_I8) ? 1 : 0;
  /* res init = {10,20,30,40}; a = {2,2,2,2}; b = {3,3,3,3}; res = a*b + res = {16,26,36,46} */
  for (int i = 0; i < n; i++) {
    float rf = (float)((i+1)*10), af = 2.0f, bf = 3.0f;
    if (fmt == FMT_F32) {
      memcpy(va + i*4, &rf,4); memcpy(va+64+i*4,&af,4); memcpy(va+96+i*4,&bf,4);
    } else if (fmt == FMT_BF16) {
      uint32_t u; memcpy(&u,&rf,4); *(uint16_t*)(va+i*2)=(uint16_t)(u>>16);
      memcpy(&u,&af,4); *(uint16_t*)(va+64+i*2)=(uint16_t)(u>>16);
      memcpy(&u,&bf,4); *(uint16_t*)(va+96+i*2)=(uint16_t)(u>>16);
    } else {
      va[i] = (uint8_t)((i+1)*10); va[64+i] = 2; va[96+i] = 3;
    }
  }
  CVI_RT_MemFlush(rt, mem);
  uint8_t cmdbuf[16384] __attribute__((aligned(16)));
  bmk_info_t info = { .chip_version = 1822, .cmdbuf_size = sizeof(cmdbuf), .cmdbuf = cmdbuf };
  bmk1822_context_t *bmk = bmk1822_register(&info);
  bmk1822_tensor_lmem_shape_t ts = {1,1,1,(uint32_t)n};
  bmk1822_tensor_lmem_t *a = bmk1822_lmem_alloc_tensor(bmk, ts, fmt, 1);
  bmk1822_tensor_lmem_t *b = bmk1822_lmem_alloc_tensor(bmk, ts, fmt, 1);
  /* check_16bit_tiu_tensor requires low->start_address < high->start_address */
  bmk1822_tensor_lmem_t *res_l = bmk1822_lmem_alloc_tensor(bmk, ts, fmt, 1);
  bmk1822_tensor_lmem_t *res_h = bmk1822_lmem_alloc_tensor(bmk, ts, fmt, 1);
  if (!a || !b || !res_h || !res_l) {
    printf("[B|mac %s] ALLOC FAIL (fmt=%d)\n", tag, fmt); goto outB2; }
  uint32_t an = (uint32_t)(esz*n);
  bmk1822_tdma_g2l_general_copy(bmk, &(bmk1822_tdma_tg2l_general_copy_param_t){.src_base_reg_index=0,.src_address=0,.dst_address=a->start_address,.bytes=an});
  bmk1822_tdma_g2l_general_copy(bmk, &(bmk1822_tdma_tg2l_general_copy_param_t){.src_base_reg_index=0,.src_address=64,.dst_address=b->start_address,.bytes=an});
  bmk1822_tdma_g2l_general_copy(bmk, &(bmk1822_tdma_tg2l_general_copy_param_t){.src_base_reg_index=0,.src_address=96,.dst_address=res_h->start_address,.bytes=an});
  bmk1822_tdma_g2l_general_copy(bmk, &(bmk1822_tdma_tg2l_general_copy_param_t){.src_base_reg_index=0,.src_address=96,.dst_address=res_l->start_address,.bytes=an});
  bmk1822_tiu_element_wise_mac_param_t p = {
    .res_high = res_h, .res_low = res_l, .a = a, .b_is_const = 0, .b = b,
    .res_is_int8 = res_is_int8, .relu_enable = 0, .lshift_bits = 0, .rshift_bits = 0, .layer_id = 1 };
  if (!bmk1822_tiu_element_wise_mac(bmk, &p)) {
    printf("[B|mac %s] OP REJECTED (fmt=%d)\n", tag, fmt); goto outB2; }
  /* read back res (for int8: res_low only is output; for bf16: res_high=16-bit result) */
  bmk1822_tdma_l2g_general_copy(bmk, &(bmk1822_tdma_l2tg_general_copy_param_t){
    .src_address = res_is_int8 ? res_l->start_address : res_h->start_address,
    .dst_base_reg_index = 0, .dst_address = 512, .bytes = an });
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
  uint8_t *r = va + 512;
  printf("[B|mac %s] fmt=%d res_i8=%d out bytes:", tag, fmt, res_is_int8);
  for (int i = 0; i < esz*n; i++) printf(" %02x", r[i]);
  printf("\n");
  if (fmt==FMT_F32) { float *f=(float*)r; printf("  as_f32[0..%d]:", n-1); for(int i=0;i<n;i++) printf(" %g", f[i]); printf(" (expect 16,26,36,46)\n"); }
  else if (fmt==FMT_BF16) { uint16_t *h=(uint16_t*)r; printf("  as_bf16:"); for(int i=0;i<n;i++){ uint32_t u=((uint32_t)h[i])<<16; float f; memcpy(&f,&u,4); printf(" %g",f);} printf("\n"); }
  else { int8_t *s=(int8_t*)r; printf("  as_i8[0..%d]:", n-1); for(int i=0;i<n;i++) printf(" %d", s[i]); printf(" (expect 16,26,36,46)\n"); }
  CVI_RT_MemFree(rt, dmabuf_mem);
outB2:
  bmk1822_cleanup(bmk);
  CVI_RT_MemFree(rt, mem);
  CVI_RT_DeInit(rt);
}

/* ------------------------------------------------------------------ */
/* PART C — Qwen batch submit cost                                    */
/* ------------------------------------------------------------------ */
static void batch_cost(const char *tag, int nops) {
  CVI_RT_HANDLE rt; CVI_RT_Init(&rt);
  CVI_RT_MEM mem = CVI_RT_MemAlloc(rt, NEURON_SZ);
  uint64_t pa = CVI_RT_MemGetPAddr(mem);
  uint8_t *va = CVI_RT_MemGetVAddr(mem);
  CVI_RT_SetBaseReg(rt, 0, pa);
  memset(va, 0, NEURON_SZ);
  int M = 1, K = 32, N = 192;
  srand(7);
  int8_t *left = (int8_t*)(va + 0);
  int8_t *right = (int8_t*)(va + 2048);
  for (int k = 0; k < K; k++) {
    left[k] = (int8_t)(rand()%200-100);
    for (int n = 0; n < N; n++) right[k*N+n] = (int8_t)(rand()%200-100);
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
  if (!ml_l || !ml_r || !ml_res) { printf("[C|%s] ALLOC FAIL\n", tag); goto outC; }
  bmk1822_matrix_tgmem_t mg_l = {0, 0, FMT_I8, {M,K}, {(uint32_t)K}};
  bmk1822_matrix_tgmem_t mg_r = {0, 2048, FMT_I8, {K,N}, {(uint32_t)N}};
  bmk1822_tdma_g2l_matrix_copy(bmk, &(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_l, ml_l});
  bmk1822_tdma_g2l_matrix_copy(bmk, &(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_r, ml_r});
  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);
  for (int i = 0; i < nops; i++) {
    bmk1822_tiu_matrix_multiplication_param_t p = {
      .res = ml_res, .left = ml_l, .right = ml_r, .bias = NULL,
      .lshift_bits = 0, .rshift_bits = 0, .res_is_int8 = 1, .relu_enable = 0,
      .add_result = 0, .ps32_mode = 2, .layer_id = 1 };
    bmk1822_tiu_matrix_multiplication(bmk, &p);
  }
  clock_gettime(CLOCK_MONOTONIC, &t1);
  double build_ms = (t1.tv_sec-t0.tv_sec)*1000.0 + (t1.tv_nsec-t0.tv_nsec)/1e6;
  uint32_t cmd_sz; uint8_t *cmd = bmk1822_acquire_cmdbuf(bmk, &cmd_sz);
  uint32_t psize, pmu_size; bmk1822_dmabuf_size(cmd, cmd_sz, &psize, &pmu_size);
  CVI_RT_MEM dmabuf_mem = CVI_RT_MemAlloc(rt, psize + pmu_size);
  uint8_t *dmabuf = CVI_RT_MemGetVAddr(dmabuf_mem);
  bmk1822_dmabuf_convert(cmd, cmd_sz, dmabuf);
  bmk1822_arraybase_set(dmabuf, pa, 0, 0, 0);
  CVI_RT_MemFlush(rt, dmabuf_mem);
  CVI_RT_MEM loaded; CVI_RT_LoadDmabuf(rt, dmabuf_mem, psize+pmu_size, pa, 0, false, &loaded);
  clock_gettime(CLOCK_MONOTONIC, &t0);
  CVI_RT_RunCmdbufEx(rt, loaded, &(CVI_RT_ARRAYBASE){.gaddr_base0=pa});
  clock_gettime(CLOCK_MONOTONIC, &t1);
  double run_ms = (t1.tv_sec-t0.tv_sec)*1000.0 + (t1.tv_nsec-t0.tv_nsec)/1e6;
  printf("[C|%s] nops=%d build=%.2fms (%.3f us/op) run=%.2fms (%.3f us/op) cmdbuf=%uB\n",
         tag, nops, build_ms, build_ms/nops*1000.0, run_ms, run_ms/nops*1000.0, cmd_sz);
  CVI_RT_MemFree(rt, dmabuf_mem);
outC:
  bmk1822_cleanup(bmk);
  CVI_RT_MemFree(rt, mem);
  CVI_RT_DeInit(rt);
}

/* Part B fork wrappers */
typedef struct { const char *tag; fmt_t fmt; int gc; } ew_mul_arg_t;
static ew_mul_arg_t g_mul_arg;
static void ew_mul_child(void) { ew_mul_case(g_mul_arg.tag, g_mul_arg.fmt, g_mul_arg.gc); }
typedef struct { const char *tag; fmt_t fmt; } ew_mac_arg_t;
static ew_mac_arg_t g_mac_arg;
static void ew_mac_child(void) { ew_mac_case(g_mac_arg.tag, g_mac_arg.fmt); }

int main(void) {
  printf("===== ps32 DECISIVE SPIKE =====\n");
  printf("host acc for K=32,L=R=100 = %d\n", 32*100*100);

  printf("\n--- allocator format check (ps32_matrix alloc, may assert) ---\n");
  g_ps32_arg.fmt = FMT_I8;   run_fork("alloc_I8",  alloc_probe_child);
  g_ps32_arg.fmt = FMT_BF16; run_fork("alloc_BF16",alloc_probe_child);
  g_ps32_arg.fmt = FMT_I32;  run_fork("alloc_I32", alloc_probe_child);
  g_ps32_arg.fmt = FMT_F32;  run_fork("alloc_F32", alloc_probe_child);

  printf("\n--- PART A: ps32_mode fp32 readability (K=32, M=1, N=32) ---\n");
  g_ps32_arg.fmt = FMT_I8;   g_ps32_arg.res_is_int8 = 1; g_ps32_arg.ps32_mode = 0; g_ps32_tag = "C0_ctrl_i8";     run_fork("A|C0", ps32_case_child);
  g_ps32_arg.fmt = FMT_BF16; g_ps32_arg.res_is_int8 = 0; g_ps32_arg.ps32_mode = 1; g_ps32_tag = "C1_ps32m1_bf16"; run_fork("A|C1", ps32_case_child);
  g_ps32_arg.fmt = FMT_BF16; g_ps32_arg.res_is_int8 = 0; g_ps32_arg.ps32_mode = 2; g_ps32_tag = "C2_ps32m2_bf16"; run_fork("A|C2", ps32_case_child);
  g_ps32_arg.fmt = FMT_I8;   g_ps32_arg.res_is_int8 = 1; g_ps32_arg.ps32_mode = 2; g_ps32_tag = "C3_ps32m2_i8";   run_fork("A|C3", ps32_case_child);
  g_ps32_arg.fmt = FMT_I8;   g_ps32_arg.res_is_int8 = 1; g_ps32_arg.ps32_mode = 1; g_ps32_tag = "C6_ps32m1_i8";   run_fork("A|C6", ps32_case_child);
  g_ps32_arg.fmt = FMT_F32;  g_ps32_arg.res_is_int8 = 0; g_ps32_arg.ps32_mode = 2; g_ps32_tag = "C7_ps32m2_f32res"; run_fork("A|C7", ps32_case_child);

  printf("\n--- PART B: element_wise F32/BF16 ---\n");
  g_mul_arg.tag="I8_ctrl"; g_mul_arg.fmt=FMT_I8;   g_mul_arg.gc=0; run_fork("B|mul_I8",  ew_mul_child);
  g_mul_arg.tag="BF16";    g_mul_arg.fmt=FMT_BF16; g_mul_arg.gc=0; run_fork("B|mul_BF16",ew_mul_child);
  g_mul_arg.tag="F32";     g_mul_arg.fmt=FMT_F32;  g_mul_arg.gc=1; run_fork("B|mul_F32", ew_mul_child);
  g_mac_arg.tag="I8_ctrl";  g_mac_arg.fmt=FMT_I8;    run_fork("B|mac_I8",  ew_mac_child);
  g_mac_arg.tag="BF16";     g_mac_arg.fmt=FMT_BF16;  run_fork("B|mac_BF16",ew_mac_child);
  g_mac_arg.tag="F32";      g_mac_arg.fmt=FMT_F32;   run_fork("B|mac_F32", ew_mac_child);

  printf("\n--- PART C: Qwen batch submit cost (K=32 group, N=192 chunk) ---\n");
  /* Wq = 140 ops, Wk/Wv = 28, Wo = 20, up/gate = 728, down = 760 */
  batch_cost("Wk(28)",    28);
  batch_cost("Wo(20)",    20);
  batch_cost("Wq(140)",   140);
  batch_cost("up(728)",   728);
  batch_cost("down(760)", 760);
  return 0;
}
