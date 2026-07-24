#ifndef TPU_BENCH_H
#define TPU_BENCH_H
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "cviruntime_context.h"
#include "cvikernel/cvikernel.h"

typedef struct {
  CVI_RT_HANDLE rt_handle;
  CVI_RT_KHANDLE rt_khandle;   // also = cvk_context_t*
  cvk_context_t *cvk_ctx;
  CVI_RT_MEM neuron_mem;
  uint64_t neuron_paddr;
  uint8_t *neuron_vaddr;
  size_t   neuron_size;
} tpu_ctx;

/* Physical address within neuron mem — use absolute PA, not base-reg offset */
#define TPU_PA(ctx, off) ((ctx)->neuron_paddr + (off))

static inline cvk_tg_stride_t tpu_tg_default_stride(cvk_tg_shape_t s, cvk_fmt_t fmt) {
  int es;
  switch (fmt) {
    case CVK_FMT_I8:  case CVK_FMT_U8:  es = 1; break;
    case CVK_FMT_I16: case CVK_FMT_U16: case CVK_FMT_BF16: es = 2; break;
    default: es = 4; break;
  }
  cvk_tg_stride_t stride = { .w = es, .h = s.w * es,
                              .c = s.h * s.w * es, .n = s.c * s.h * s.w * es };
  return stride;
}

static inline int tpu_init(tpu_ctx *ctx, size_t global_bytes) {
  memset(ctx, 0, sizeof(*ctx));
  if (CVI_RT_Init(&ctx->rt_handle) != 0) {
    fprintf(stderr, "CVI_RT_Init fail\n"); return -1;
  }
  if (global_bytes < 4096) global_bytes = 4096;

  ctx->neuron_mem = CVI_RT_MemAlloc(ctx->rt_handle, global_bytes);
  if (!ctx->neuron_mem) { fprintf(stderr, "MemAlloc fail\n"); return -1; }
  ctx->neuron_paddr = CVI_RT_MemGetPAddr(ctx->neuron_mem);
  ctx->neuron_vaddr = CVI_RT_MemGetVAddr(ctx->neuron_mem);
  ctx->neuron_size  = CVI_RT_MemGetSize(ctx->neuron_mem);

  CVI_RT_SetBaseReg(ctx->rt_handle, 0, ctx->neuron_paddr);

  ctx->rt_khandle = CVI_RT_RegisterKernel(ctx->rt_handle, 0x40000);
  if (!ctx->rt_khandle) { fprintf(stderr, "RegisterKernel fail\n"); return -1; }
  ctx->cvk_ctx = (cvk_context_t *)ctx->rt_khandle;
  if (!ctx->cvk_ctx->ops) { fprintf(stderr, "cvk_ctx->ops is NULL\n"); return -1; }

  fprintf(stderr, "[tpu_init] pa=0x%llx va=%p sz=%zu ok\n",
          (unsigned long long)ctx->neuron_paddr, ctx->neuron_vaddr, ctx->neuron_size);
  return 0;
}

// Submit accumulated TIU/TDMA commands to TPU. Returns wall time in ns.
static inline int64_t tpu_submit(tpu_ctx *ctx) {
  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);
  int rc = CVI_RT_Submit(ctx->rt_khandle);
  clock_gettime(CLOCK_MONOTONIC, &t1);
  if (rc != 0) { fprintf(stderr, "CVI_RT_Submit fail rc=%d\n", rc); return -1; }
  return (t1.tv_sec - t0.tv_sec) * 1000000000LL + (t1.tv_nsec - t0.tv_nsec);
}

static inline void tpu_cleanup(tpu_ctx *ctx) {
  if (ctx->rt_khandle) { CVI_RT_UnRegisterKernel(ctx->rt_khandle); ctx->rt_khandle = NULL; }
  if (ctx->neuron_mem) { CVI_RT_MemFree(ctx->rt_handle, ctx->neuron_mem); ctx->neuron_mem = NULL; }
  if (ctx->rt_handle) { CVI_RT_DeInit(ctx->rt_handle); ctx->rt_handle = NULL; }
}

static inline int tpu_check_i8(const char *label, int8_t *got, const int8_t *exp, int n, int nprint) {
  int errs = 0;
  for (int i = 0; i < n; i++) {
    if (got[i] != exp[i]) {
      if (errs < 5) fprintf(stderr, "  [%s] idx=%d got=%d exp=%d\n", label, i, got[i], exp[i]);
      errs++;
    }
  }
  if (errs == 0) printf("  [%s] ALL %d OK", label, n);
  else           printf("  [%s] %d/%d MISMATCH", label, errs, n);
  if (nprint) { printf("  first_%d=[", nprint);
    for (int i = 0; i < nprint && i < n; i++) printf("%d,", got[i]);
    printf("]"); }
  printf("\n");
  return errs;
}

static inline int tpu_check_i16(const char *label, int16_t *got, const int16_t *exp, int n) {
  int errs = 0;
  for (int i = 0; i < n; i++) {
    if (got[i] != exp[i]) {
      if (errs < 5) fprintf(stderr, "  [%s] idx=%d got=%d exp=%d\n", label, i, got[i], exp[i]);
      errs++;
    }
  }
  printf("  [%s] %s (%d elems)\n", label, errs==0?"ALL OK":"MISMATCH", n);
  return errs;
}

static inline int tpu_check_f32(const char *label, float *got, const float *exp, int n) {
  int errs = 0;
  for (int i = 0; i < n; i++) {
    if (got[i] != exp[i]) {
      if (errs < 5) fprintf(stderr, "  [%s] idx=%d got=%f exp=%f\n", label, i, got[i], exp[i]);
      errs++;
    }
  }
  printf("  [%s] %s (%d elems)\n", label, errs==0?"ALL OK":"MISMATCH", n);
  return errs;
}
#endif
