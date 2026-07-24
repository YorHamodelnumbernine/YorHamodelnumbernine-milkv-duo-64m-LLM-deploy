/* Diagnostic v3: use stride type-2 (compact) as expected by bmk1822 conv assertion */
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
  bmk1822_op_t *(*bm_conv)(bmk1822_context_t*, const bmk1822_tiu_convolution_param_t*) = dlsym(h, "bmk1822_tiu_convolution");

  fprintf(stderr,"[diag] reg=%p conv=%p\n", (void*)bm_register, (void*)bm_conv);
  if (!bm_register || !bm_conv) { fprintf(stderr,"symbols not found\n"); return 1; }

  uint8_t cmdbuf[4096] __attribute__((aligned(16)));
  bmk_info_t info = { .chip_version = 1822, .cmdbuf_size = sizeof(cmdbuf), .cmdbuf = cmdbuf };
  bmk1822_context_t *bmk = bm_register(&info);
  if (!bmk) { fprintf(stderr,"bmk1822_register fail\n"); return 1; }
  fprintf(stderr,"[diag] bmk1822 context created\n");

  /* Assertion in kernel_1822.h line 217, assert_stride_type_2:
   *   t->stride.n == fmt * align_up(c, npu_num) / npu_num
   * For FMT_I8=4, c=2, npu_num=2: n_stride = 4 * 2 / 2 = 4
   * Type-2 strides are "compact" strides where:
   *   n_stride = amount of data in one channel per NPU core
   *   c_stride = stride between consecutive channels in same NPU
   *   h_stride = stride between rows = w * bytes_per_elem
   *   w_stride = stride between columns = bytes_per_elem
   */
  #define NPU_NUM 2
  #define FMT_BYTES 1  // I8 = 1 byte

  /* Compute type-2 strides per tensor:
   * n_stride(t2) = FMT_BYTES * H * W * align_up(C, NPU_NUM) / NPU_NUM
   * Wait, actually the assertion says fmt * align_up(c, npu_num) / npu_num
   * where fmt is the ENUM VALUE (FMT_I8=4), not byte size.
   * Let me try with fmt=4 (enum) first, then try other interpretations.
   *
   * Actually let me reconsider. The fmt enum values:
   * F32=0, F16=1, I32=2, I16=3, I8=4, I4=5, I2=6, I1=7, U32=8, U16=9, U8=10, BF16=11
   *
   * If fmt=4 (as used in the enum), n_stride = 4 * 2 / 2 = 4
   * If fmt is interpreted as byte size (=1), n_stride = 1 * 2 / 2 = 1
   *
   * The bmk1822 allocator gives n_stride=16 for type-1 (full eu-aligned stride).
   * Type-2 is the "compact" or "register" representation.
   *
   * Let me try: n_stride = h * w * align_up(c,npu_num) / npu_num * fmt_bytes
   *            = 2 * 2 * align_up(2,2) / 2 * 1 = 4 * 2 / 2 * 1 = 4
   *
   * c_stride = h * w * fmt_bytes = 4
   * h_stride = w * fmt_bytes = 2
   * w_stride = 1
   */

  /* Mixed stride: n=compact(type-2), c+eu-aligned(type-0) */
  #define EU_NUM 16
  #define MIX_N(c,h,w) ((h)*(w) * align_up(c, NPU_NUM) / NPU_NUM * FMT_BYTES)
  #define MIX_C(h,w)   (align_up((h)*(w) * FMT_BYTES, EU_NUM))
  #define MIX_H(w)     ((w) * FMT_BYTES)
  #define MIX_W()      (FMT_BYTES)

  bmk1822_tensor_lmem_t tl_if = {0}, tl_of = {0}, tl_w = {0}, tl_b = {0};

  tl_if.fmt = FMT_I8;
  tl_if.shape = (bmk1822_tensor_lmem_shape_t){1,2,2,2};
  tl_if.stride = (bmk1822_tensor_lmem_stride_t){MIX_N(2,2,2), MIX_C(2,2), MIX_H(2), MIX_W()};
  tl_if.eu_align = 1;

  tl_of.fmt = FMT_I8;
  tl_of.shape = (bmk1822_tensor_lmem_shape_t){1,2,2,2};
  tl_of.stride = (bmk1822_tensor_lmem_stride_t){MIX_N(2,2,2), MIX_C(2,2), MIX_H(2), MIX_W()};
  tl_of.eu_align = 1;

  tl_w.fmt = FMT_I8;
  tl_w.shape = (bmk1822_tensor_lmem_shape_t){1,2,1,2};
  tl_w.stride = (bmk1822_tensor_lmem_stride_t){MIX_N(2,1,2), MIX_C(1,2), MIX_H(2), MIX_W()};
  tl_w.eu_align = 1;

  tl_b.fmt = FMT_I8;
  tl_b.shape = (bmk1822_tensor_lmem_shape_t){2,2,1,1};
  tl_b.stride = (bmk1822_tensor_lmem_stride_t){MIX_N(2,1,1), MIX_C(1,1), MIX_H(1), MIX_W()};
  tl_b.eu_align = 1;

  /* Manual start addresses using allocator-style (c_stride per npu_group) */
  tl_if.start_address = 0;
  tl_of.start_address = 16;  /* MIX_C(2,2) = 16 */
  tl_w.start_address  = 32;  /* 16+16 */
  tl_b.start_address  = 48;  /* 32+16 */

  fprintf(stderr,"[diag] type-2 strides:\n");
  fprintf(stderr,"  if: addr=0x%x n=%u c=%u h=%u w=%u\n",
    tl_if.start_address, tl_if.stride.n, tl_if.stride.c, tl_if.stride.h, tl_if.stride.w);
  fprintf(stderr,"  of: addr=0x%x n=%u c=%u h=%u w=%u\n",
    tl_of.start_address, tl_of.stride.n, tl_of.stride.c, tl_of.stride.h, tl_of.stride.w);
  fprintf(stderr,"  w:  addr=0x%x n=%u c=%u h=%u w=%u\n",
    tl_w.start_address, tl_w.stride.n, tl_w.stride.c, tl_w.stride.h, tl_w.stride.w);
  fprintf(stderr,"  b:  addr=0x%x n=%u c=%u h=%u w=%u\n",
    tl_b.start_address, tl_b.stride.n, tl_b.stride.c, tl_b.stride.h, tl_b.stride.w);

  bmk1822_tiu_convolution_param_t p = {
    .ofmap = &tl_of, .ifmap = &tl_if, .weight = &tl_w, .bias = &tl_b,
    .ins_h = 0, .ins_last_h = 0, .ins_w = 0, .ins_last_w = 0,
    .pad_top = 0, .pad_bottom = 0, .pad_left = 0, .pad_right = 0,
    .stride_h = 1, .stride_w = 1,
    .dilation_h = 1, .dilation_w = 1,
    .relu_enable = 0, .rshift_bits = 0,
    .ps32_mode = 0, .w_is_const = 0,
  };

  fprintf(stderr,"[diag] calling bmk1822_tiu_convolution...\n");
  bmk1822_op_t *op = bm_conv(bmk, &p);
  fprintf(stderr,"[diag] returned op=%p -> %s\n",
    (void*)op, op ? "BM1822 ACCEPTED!" : "BM1822 REJECTED");

  bm_cleanup(bmk);
  dlclose(h);
  return op ? 0 : 1;
}
