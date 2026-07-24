/* Diagnostic: test bmk1822_tiu_depthwise_convolution directly with proper bmk context */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include "bmkernel/bm1822/bmkernel_1822.h"

int main() {
  void *h = dlopen("libcvikernel.so", RTLD_LAZY);
  if (!h) { fprintf(stderr,"dlopen fail\n"); return 1; }

  bmk1822_context_t *(*bm_register)(bmk_info_t*) = dlsym(h, "bmk1822_register");
  void (*bm_cleanup)(bmk1822_context_t*) = dlsym(h, "bmk1822_cleanup");
  bmk1822_op_t *(*bm_dw_conv)(bmk1822_context_t*, const bmk1822_tiu_depthwise_convolution_param_t*) = dlsym(h, "bmk1822_tiu_depthwise_convolution");
  bmk1822_tensor_lmem_t *(*bm_alloc)(bmk1822_context_t*, bmk1822_tensor_lmem_shape_t, fmt_t, int) = dlsym(h, "bmk1822_lmem_alloc_tensor");
  void (*bm_free)(bmk1822_context_t*, bmk1822_tensor_lmem_t*) = dlsym(h, "bmk1822_lmem_free_tensor");

  fprintf(stderr,"[diag] symbols: reg=%p dw_conv=%p alloc=%p\n",
    (void*)bm_register, (void*)bm_dw_conv, (void*)bm_alloc);
  if (!bm_register || !bm_dw_conv || !bm_alloc || !bm_free) {
    fprintf(stderr,"symbols not found\n"); return 1;
  }

  uint8_t cmdbuf[4096] __attribute__((aligned(16)));
  bmk_info_t info = { .chip_version = 1822, .cmdbuf_size = sizeof(cmdbuf), .cmdbuf = cmdbuf };
  bmk1822_context_t *bmk = bm_register(&info);
  if (!bmk) { fprintf(stderr,"bmk1822_register fail\n"); return 1; }
  fprintf(stderr,"[diag] bmk1822 context created\n");

  /* Test 1: depthwise conv with kh=3, kw=3, IC=OC=2, IH=IW=4 */
  bmk1822_tensor_lmem_t *tl_if = bm_alloc(bmk, (bmk1822_tensor_lmem_shape_t){1,2,4,4}, FMT_I8, 1);
  bmk1822_tensor_lmem_t *tl_of = bm_alloc(bmk, (bmk1822_tensor_lmem_shape_t){1,2,2,2}, FMT_I8, 1);
  bmk1822_tensor_lmem_t *tl_w  = bm_alloc(bmk, (bmk1822_tensor_lmem_shape_t){1,2,9,1}, FMT_I8, 1);

  if (!tl_if || !tl_of || !tl_w) {
    fprintf(stderr,"alloc fail\n"); bm_cleanup(bmk); return 1;
  }

  fprintf(stderr,"[diag] if: addr=0x%x stride=(%u,%u,%u,%u)\n",
    tl_if->start_address, tl_if->stride.n, tl_if->stride.c, tl_if->stride.h, tl_if->stride.w);
  fprintf(stderr,"[diag] of: addr=0x%x stride=(%u,%u,%u,%u)\n",
    tl_of->start_address, tl_of->stride.n, tl_of->stride.c, tl_of->stride.h, tl_of->stride.w);
  fprintf(stderr,"[diag] w:  addr=0x%x stride=(%u,%u,%u,%u)\n",
    tl_w->start_address, tl_w->stride.n, tl_w->stride.c, tl_w->stride.h, tl_w->stride.w);

  bmk1822_tiu_depthwise_convolution_param_t p = {
    .ofmap = tl_of, .ifmap = tl_if, .weight = tl_w,
    .bias = NULL,
    .weight_is_const = 0,
    .ins_h = 0, .ins_last_h = 0, .ins_w = 0, .ins_last_w = 0,
    .dilation_h = 1, .dilation_w = 1,
    .pad_top = 0, .pad_bottom = 0, .pad_left = 0, .pad_right = 0,
    .stride_h = 1, .stride_w = 1,
    .rshift_bits = 0, .relu_enable = 0,
    .cmd_pre_exe_typ = 0, .cmd_pre_exe = 0,
    .layer_id = 1,
  };

  fprintf(stderr,"[diag] calling bmk1822_tiu_depthwise_convolution...\n");
  bmk1822_op_t *op = bm_dw_conv(bmk, &p);
  fprintf(stderr,"[diag] returned op=%p -> %s\n",
    (void*)op, op ? "BM1822 DW-CONV ACCEPTED!" : "BM1822 DW-CONV REJECTED");

  bm_free(bmk, tl_w);
  bm_free(bmk, tl_of);
  bm_free(bmk, tl_if);
  bm_cleanup(bmk);
  dlclose(h);
  return op ? 0 : 1;
}
