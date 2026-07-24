/* TPU mul_const: multiply each INT8 element by constant 2 using TIU MUL */
#include "../common/tpu_bench.h"

#define N_ELEMS 16

static const int8_t input[N_ELEMS]  = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
static const int8_t expected[N_ELEMS] = {2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32};

int main() {
  tpu_ctx ctx;
  if (tpu_init(&ctx, 4096) != 0) return 1;

  cvk_context_t *cvk = ctx.cvk_ctx;

  /* ---- write input to neuron memory via direct virtual address ---- */
  memset(ctx.neuron_vaddr, 0, 64);
  memcpy(ctx.neuron_vaddr, input, N_ELEMS);
  CVI_RT_MemFlush(ctx.rt_handle, ctx.neuron_mem);

  /* ---- alloc local tensors ---- */
  cvk_tl_shape_t s8 = {1, 1, 1, N_ELEMS};
  cvk_tl_t *tl_src = cvk->ops->lmem_alloc_tensor(cvk, s8, CVK_FMT_I8, 1);
  cvk_tl_t *tl_dst = cvk->ops->lmem_alloc_tensor(cvk, s8, CVK_FMT_I8, 1);
  if (!tl_src || !tl_dst) {
    fprintf(stderr, "lmem_alloc_tensor fail\n");
    if (tl_src) cvk->ops->lmem_free_tensor(cvk, tl_src);
    if (tl_dst) cvk->ops->lmem_free_tensor(cvk, tl_dst);
    tpu_cleanup(&ctx); return 1;
  }

  /* ---- global tensor descriptors ---- */
  cvk_tg_shape_t gs8 = {1, 1, 1, N_ELEMS};
  cvk_tg_stride_t gstride = cvk->ops->tg_default_stride(cvk, gs8, CVK_FMT_I8);

  cvk_tg_t tg_src = { .base_reg_index = 0, .start_address = ctx.neuron_paddr,
    .fmt = CVK_FMT_I8, .shape = gs8, .stride = gstride };
  cvk_tg_t tg_dst = { .base_reg_index = 0, .start_address = ctx.neuron_paddr + 16,
    .fmt = CVK_FMT_I8, .shape = gs8, .stride = gstride };

  /* ---- TDMA: G2L copy input ---- */
  {
    cvk_tdma_g2l_tensor_copy_param_t p = { .src = &tg_src, .dst = tl_src };
    cvk->ops->tdma_g2l_tensor_copy(cvk, &p);
  }

  /* ---- TIU: multiply by constant 2 ---- */
  {
    cvk_tiu_mul_param_t p = {
      .res_high = NULL,
      .res_low  = tl_dst,
      .a        = tl_src,
      .b_is_const = 1,
      .b_const  = { .val = 2, .is_signed = 1 },
      .rshift_bits = 0,
      .relu_enable = 0,
    };
    cvk->ops->tiu_mul(cvk, &p);
  }

  /* ---- TDMA: L2G write result ---- */
  {
    cvk_tdma_l2g_tensor_copy_param_t p = { .src = tl_dst, .dst = &tg_dst };
    cvk->ops->tdma_l2g_tensor_copy(cvk, &p);
  }

  /* ---- Submit ---- */
  int64_t wall_ns = tpu_submit(&ctx);
  if (wall_ns < 0) {
    fprintf(stderr, "TPU submit failed!\n");
    cvk->ops->lmem_free_tensor(cvk, tl_dst);
    cvk->ops->lmem_free_tensor(cvk, tl_src);
    tpu_cleanup(&ctx); return 1;
  }

  /* ---- Read back result via direct virtual address ---- */
  CVI_RT_MemInvld(ctx.rt_handle, ctx.neuron_mem);
  int8_t *out = (int8_t *)ctx.neuron_vaddr;
  fprintf(stderr, "[dbg] mem[ 0-16]: ");
  for (int i=0;i<16;i++) fprintf(stderr,"%d,",out[i]);
  fprintf(stderr,"\n[dbg] mem[16-32]: ");
  for (int i=16;i<32;i++) fprintf(stderr,"%d,",out[i]);
  fprintf(stderr,"\n");

  int errs = tpu_check_i8("mul2", out + 16, expected, N_ELEMS, 16);
  printf("  time: %.2f us\n", wall_ns / 1000.0);

  cvk->ops->lmem_free_tensor(cvk, tl_dst);
  cvk->ops->lmem_free_tensor(cvk, tl_src);
  tpu_cleanup(&ctx);
  return errs ? 1 : 0;
}
