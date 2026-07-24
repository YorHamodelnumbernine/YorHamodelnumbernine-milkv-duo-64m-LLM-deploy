/* TPU lookup_table: INT8 table lookup, all I8 format.
   Table_n=16 (min valid), npu_num=2 copies. */
#include "../common/tpu_bench.h"
#define N 16
#define NPU_NUM 2
#define TBL_N 16

static const int8_t  idx[N]   = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
/* Table: f(x) = x * 2 + 1. 2 NPU copies = 32 bytes */
static const int8_t  tbl[NPU_NUM*TBL_N] = {
  1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,
  1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,
};
static const int8_t  exp[N]   = {1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31};

int main() {
  tpu_ctx ctx;
  if (tpu_init(&ctx, 4096) != 0) return 1;
  cvk_context_t *cvk = ctx.cvk_ctx;

  memcpy(ctx.neuron_vaddr, idx, N);
  memcpy(ctx.neuron_vaddr+16, tbl, sizeof(tbl));
  CVI_RT_MemFlush(ctx.rt_handle, ctx.neuron_mem);

  /* All I8. eu_align=0 to avoid alignment issues */
  cvk_tl_shape_t s_if  = {1,1,1,N};
  cvk_tl_shape_t s_tbl = {1,NPU_NUM,1,TBL_N};
  cvk_tl_shape_t s_of  = {1,1,1,N};

  cvk_tl_t *tl_if  = cvk->ops->lmem_alloc_tensor(cvk, s_if, CVK_FMT_I8, 0);
  cvk_tl_t *tl_tbl = cvk->ops->lmem_alloc_tensor(cvk, s_tbl, CVK_FMT_I8, 0);
  cvk_tl_t *tl_of  = cvk->ops->lmem_alloc_tensor(cvk, s_of, CVK_FMT_I8, 0);
  if (!tl_if||!tl_tbl||!tl_of) { fprintf(stderr,"alloc fail\n"); tpu_cleanup(&ctx); return 1; }

  cvk_tg_shape_t gs_if  = {1,1,1,N};
  cvk_tg_shape_t gs_tbl = {1,NPU_NUM,1,TBL_N};
  cvk_tg_stride_t st_if  = cvk->ops->tg_default_stride(cvk, gs_if, CVK_FMT_I8);
  cvk_tg_stride_t st_tbl = cvk->ops->tg_default_stride(cvk, gs_tbl, CVK_FMT_I8);
  cvk_tg_stride_t st_of  = cvk->ops->tg_default_stride(cvk, gs_if, CVK_FMT_I8);

  cvk_tg_t tg_if  = {0, TPU_PA(&ctx,0),  CVK_FMT_I8, gs_if,  st_if};
  cvk_tg_t tg_tbl = {0, TPU_PA(&ctx,16), CVK_FMT_I8, gs_tbl, st_tbl};
  cvk_tg_t tg_of  = {0, TPU_PA(&ctx,48), CVK_FMT_I8, gs_if,  st_of};

  cvk_tdma_g2l_tensor_copy_param_t g2l_if  = {&tg_if,  tl_if};
  cvk_tdma_g2l_tensor_copy_param_t g2l_tbl = {&tg_tbl, tl_tbl};
  cvk->ops->tdma_g2l_tensor_copy(cvk, &g2l_if);
  cvk->ops->tdma_g2l_tensor_copy(cvk, &g2l_tbl);

  cvk_tiu_lookup_table_param_t p = {tl_of, tl_if, tl_tbl};
  cvk->ops->tiu_lookup_table(cvk, &p);

  cvk_tdma_l2g_tensor_copy_param_t l2g = {tl_of, &tg_of};
  cvk->ops->tdma_l2g_tensor_copy(cvk, &l2g);

  int64_t t_ns = tpu_submit(&ctx);
  if (t_ns < 0) { fprintf(stderr,"submit fail\n"); goto out; }

  CVI_RT_MemInvld(ctx.rt_handle, ctx.neuron_mem);
  int8_t *v = (int8_t*)ctx.neuron_vaddr;
  fprintf(stderr, "[dbg] lut out:"); for(int i=0;i<N;i++) fprintf(stderr,"%d,",v[48+i]);
  fprintf(stderr,"\n");
  int errs = tpu_check_i8("lut", v+48, exp, N, N);
  printf("  time: %.2f us\n", t_ns/1000.0);

out:
  cvk->ops->lmem_free_tensor(cvk, tl_of);
  cvk->ops->lmem_free_tensor(cvk, tl_tbl);
  cvk->ops->lmem_free_tensor(cvk, tl_if);
  tpu_cleanup(&ctx);
  return errs?1:0;
}
