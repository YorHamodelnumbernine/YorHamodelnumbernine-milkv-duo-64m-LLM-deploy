/* ps32_probe16.c — compare cvikernel vs bmk1822 matrix default shape mapping.
   Prints cvk_ml_shape for [1,N] and [K,N] for N in a sweep, plus lmem size.
*/
#include <stdio.h>
#include <string.h>
#include "cviruntime_context.h"
#include "cvikernel/cvikernel.h"
#include "bmkernel/bm1822/bmkernel_1822.h"

static void pr_shape(const char *tag, cvk_ml_shape_t s, cvk_ml_stride_t st) {
  printf("  %-6s n=%u c=%u w=%u col=%u str{n=%u,c=%u,h=%u}\n",
         tag, s.n, s.c, s.w, s.col, st.n, st.c, st.h);
}

int main(void) {
  CVI_RT_HANDLE rt;
  if (CVI_RT_Init(&rt) != 0) { printf("RT_Init fail\n"); return 1; }
  CVI_RT_MEM mem = CVI_RT_MemAlloc(rt, 8192);
  uint64_t pa = CVI_RT_MemGetPAddr(mem);
  CVI_RT_SetBaseReg(rt, 0, pa);
  CVI_RT_KHANDLE kh = CVI_RT_RegisterKernel(rt, 0x40000);
  if (!kh) { printf("RegisterKernel fail\n"); return 1; }
  cvk_context_t *cvk = (cvk_context_t *)kh;
  if (!cvk->ops) { printf("cvk_ctx->ops NULL\n"); return 1; }

  int ns[] = {8, 16, 32, 48, 64, 96, 112, 128, 160, 224, 256, 448, 576, 896, 1024, 1536};
  printf("cvikernel ml_default_shape sweep (M=1, FMT_I8):\n");
  for (unsigned i = 0; i < sizeof(ns)/sizeof(int); i++) {
    cvk_ml_shape_t s = cvk->ops->ml_default_shape(cvk, 1, ns[i], CVK_FMT_I8);
    cvk_ml_stride_t st = cvk->ops->ml_default_stride(cvk, s, CVK_FMT_I8, 1);
    uint32_t sz = cvk->ops->lmem_matrix_to_size(cvk, s, CVK_FMT_I8, 1);
    printf("  N=%-5d -> ", ns[i]);
    pr_shape("ml", s, st);
    printf("          lmem_size=%u\n", sz);
  }
  printf("cvikernel ml_default_shape sweep ([K=32,N]):\n");
  for (unsigned i = 0; i < sizeof(ns)/sizeof(int); i++) {
    cvk_ml_shape_t s = cvk->ops->ml_default_shape(cvk, 32, ns[i], CVK_FMT_I8);
    cvk_ml_stride_t st = cvk->ops->ml_default_stride(cvk, s, CVK_FMT_I8, 1);
    uint32_t sz = cvk->ops->lmem_matrix_to_size(cvk, s, CVK_FMT_I8, 1);
    printf("  N=%-5d -> ", ns[i]);
    pr_shape("ml", s, st);
    printf("          lmem_size=%u\n", sz);
  }
  
  CVI_RT_DeInit(rt);
  return 0;
}
