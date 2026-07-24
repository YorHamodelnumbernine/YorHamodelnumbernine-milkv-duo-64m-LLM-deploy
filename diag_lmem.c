/* Diagnostic v2: compare lmem_alloc_tensor strides vs tl_default_stride */
#include "../common/tpu_bench.h"

static void print_strides(const char *label, cvk_tl_shape_t s, cvk_tl_stride_t st, uint32_t addr) {
  fprintf(stderr,"[diag] %s: addr=0x%x shape=(%d,%d,%d,%d) stride=(%d,%d,%d,%d)\n",
    label, addr, s.n,s.c,s.h,s.w, st.n,st.c,st.h,st.w);
}

int main() {
  tpu_ctx ctx;
  if (tpu_init(&ctx, 4096) != 0) return 1;
  cvk_context_t *cvk = ctx.cvk_ctx;

  /* shapes matching pt_conv test */
  cvk_tl_shape_t shapes[] = {
    {1,2,2,2},  /* ifmap/ofmap: n=1,c=2,h=2,w=2 */
    {1,2,1,2},  /* weight: n=1,c=OC=2,h=1,w=IC=2 */
    {2,2,1,1},  /* bias: n=2,c=OC=2,h=1,w=1 */
  };
  const char *names[] = {"ifmap", "weight", "bias"};

  for (int i = 0; i < 3; i++) {
    cvk_tl_t *tl = cvk->ops->lmem_alloc_tensor(cvk, shapes[i], CVK_FMT_I8, 1);
    if (!tl) { fprintf(stderr,"alloc fail for %s\n", names[i]); continue; }

    cvk_tl_stride_t def_st = cvk->ops->tl_default_stride(cvk, shapes[i], CVK_FMT_I8, 1);

    print_strides(names[i], shapes[i], tl->stride, tl->start_address);
    fprintf(stderr,"[diag] %s tl_default_stride: (%d,%d,%d,%d)\n",
      names[i], def_st.n, def_st.c, def_st.h, def_st.w);

    /* compute expected sizes */
    uint32_t sz = cvk->ops->lmem_tensor_to_size(cvk, shapes[i], CVK_FMT_I8, 1);
    fprintf(stderr,"[diag] %s lmem_tensor_to_size=%u\n", names[i], sz);

    cvk->ops->lmem_free_tensor(cvk, tl);
  }

  /* Also test eu_align=0 */
  fprintf(stderr,"[diag] --- eu_align=0 ---\n");
  for (int i = 0; i < 3; i++) {
    cvk_tl_t *tl = cvk->ops->lmem_alloc_tensor(cvk, shapes[i], CVK_FMT_I8, 0);
    if (!tl) { fprintf(stderr,"alloc fail for %s\n", names[i]); continue; }

    cvk_tl_stride_t def_st0 = cvk->ops->tl_default_stride(cvk, shapes[i], CVK_FMT_I8, 0);
    cvk_tl_stride_t def_st1 = cvk->ops->tl_default_stride(cvk, shapes[i], CVK_FMT_I8, 1);

    print_strides(names[i], shapes[i], tl->stride, tl->start_address);
    fprintf(stderr,"[diag] %s default(eu=0): (%d,%d,%d,%d)  default(eu=1): (%d,%d,%d,%d)\n",
      names[i], def_st0.n,def_st0.c,def_st0.h,def_st0.w,
      def_st1.n,def_st1.c,def_st1.h,def_st1.w);

    uint32_t sz = cvk->ops->lmem_tensor_to_size(cvk, shapes[i], CVK_FMT_I8, 0);
    fprintf(stderr,"[diag] %s lmem_tensor_to_size(eu=0)=%u\n", names[i], sz);

    cvk->ops->lmem_free_tensor(cvk, tl);
  }

  fprintf(stderr,"[diag] eu_num=%d npu_num=%d\n", cvk->info.eu_num, cvk->info.npu_num);

  tpu_cleanup(&ctx);
  return 0;
}
