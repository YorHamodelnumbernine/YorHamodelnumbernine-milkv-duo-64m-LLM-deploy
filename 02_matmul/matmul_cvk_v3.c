/* cvikernel matmul v3 — working: no bias (add_result=0, bias=NULL).
   Tests different sizes + SubmitAsync vs blocking. */
#include "../common/tpu_bench.h"

static int do_matmul(tpu_ctx *ctx, cvk_context_t *cvk, int M, int K, int N,
                      int use_async)  /* 0=blocking Submit, 1=SubmitAsync+Wait */
{
  int l_sz=M*K, r_sz=K*N, o_sz=M*N;
  int8_t left[M*K], right[K*N];
  for (int i=0;i<l_sz;i++) left[i]=(int8_t)(i&3);
  for (int i=0;i<r_sz;i++) right[i]=(int8_t)((i+1)&3);

  memcpy(ctx->neuron_vaddr, left, l_sz);
  memcpy(ctx->neuron_vaddr+256, right, r_sz);
  CVI_RT_MemFlush(ctx->rt_handle, ctx->neuron_mem);

  cvk_ml_shape_t sl = cvk->ops->ml_default_shape(cvk, M, K, CVK_FMT_I8);
  cvk_ml_shape_t sr = cvk->ops->ml_default_shape(cvk, K, N, CVK_FMT_I8);
  cvk_ml_shape_t so = cvk->ops->ml_default_shape(cvk, M, N, CVK_FMT_I8);

  cvk_ml_t *ml_l = cvk->ops->lmem_alloc_matrix(cvk, sl, CVK_FMT_I8, 1);
  cvk_ml_t *ml_r = cvk->ops->lmem_alloc_matrix(cvk, sr, CVK_FMT_I8, 1);
  cvk_ml_t *ml_o = cvk->ops->lmem_alloc_matrix(cvk, so, CVK_FMT_I8, 1);
  if (!ml_l||!ml_r||!ml_o) { fprintf(stderr,"alloc fail\n"); return -1; }

  cvk->ops->tdma_g2l_matrix_copy(cvk, &(cvk_tdma_g2l_matrix_copy_param_t){
    &(cvk_mg_t){0, TPU_PA(ctx,0),   CVK_FMT_I8, {M,K}, {K}}, ml_l});
  cvk->ops->tdma_g2l_matrix_copy(cvk, &(cvk_tdma_g2l_matrix_copy_param_t){
    &(cvk_mg_t){0, TPU_PA(ctx,256), CVK_FMT_I8, {K,N}, {N}}, ml_r});

  cvk->ops->tiu_matrix_multiplication(cvk, &(cvk_tiu_matrix_multiplication_param_t){
    .res=ml_o, .left=ml_l, .right=ml_r, .bias=NULL,
    .lshift_bits=0, .rshift_bits=0, .res_is_int8=1, .relu_enable=0,
    .add_result=0, .ps32_mode=0,
  });

  cvk->ops->tdma_l2g_matrix_copy(cvk, &(cvk_tdma_l2g_matrix_copy_param_t){
    ml_o, &(cvk_mg_t){0, TPU_PA(ctx,512), CVK_FMT_I8, {M,N}, {N}}});

  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);
  if (use_async) {
    CVI_RT_SubmitAsync(ctx->rt_khandle, 0);
    /* CPU busywork during TPU exec */
    volatile int32_t dummy=0;
    for (int k=0;k<500;k++) dummy+=(left[k%l_sz]*right[k%r_sz])^(k&0xff);
    CVI_RT_WaitForAsync(ctx->rt_khandle);
  } else {
    CVI_RT_Submit(ctx->rt_khandle);
  }
  clock_gettime(CLOCK_MONOTONIC, &t1);
  double t_us = ((t1.tv_sec-t0.tv_sec)*1e9+(t1.tv_nsec-t0.tv_nsec))/1000.0;

  CVI_RT_MemInvld(ctx->rt_handle, ctx->neuron_mem);
  int8_t *v = (int8_t*)ctx->neuron_vaddr+512;

  int8_t exp[M*N];
  for (int i=0;i<M;i++)
    for (int j=0;j<N;j++) {
      int sum=0;
      for (int k=0;k<K;k++) sum+=left[i*K+k]*right[k*N+j];
      exp[i*N+j]=(int8_t)sum;
    }
  int ok=1;
  for (int i=0;i<o_sz;i++) if(v[i]!=exp[i]){ok=0;break;}
  fprintf(stderr,"  (%dx%d)*(%dx%d) col=%d %-7s %s (%.0f us)\n",
    M,K,K,N, sl.col, use_async?"async":"block", ok?"OK":"MISMATCH", t_us);
  if(!ok){fprintf(stderr,"    got:");for(int i=0;i<o_sz;i++)fprintf(stderr,"%d,",v[i]);
    fprintf(stderr," exp:");for(int i=0;i<o_sz;i++)fprintf(stderr,"%d,",exp[i]);fprintf(stderr,"\n");}

  cvk->ops->lmem_free_matrix(cvk, ml_o);
  cvk->ops->lmem_free_matrix(cvk, ml_r);
  cvk->ops->lmem_free_matrix(cvk, ml_l);
  return ok?0:1;
}

int main() {
  tpu_ctx ctx;
  if (tpu_init(&ctx, 4096) != 0) return 1;
  cvk_context_t *cvk = ctx.cvk_ctx;

  fprintf(stderr, "\n=== matmul_cvk_v3: blocking vs SubmitAsync ===\n\n");

  int sizes[][3] = {{2,2,2},{4,4,4},{8,8,8},{16,16,16}};
  for (int s=0;s<4;s++) {
    int M=sizes[s][0], K=sizes[s][1], N=sizes[s][2];
    fprintf(stderr, "--- (%dx%d)*(%dx%d) ---\n", M,K,K,N);
    do_matmul(&ctx, cvk, M, K, N, 0);  /* blocking */
    do_matmul(&ctx, cvk, M, K, N, 1);  /* async */
  }

  fprintf(stderr, "=== DONE matmul_cvk_v3 ===\n\n");
  tpu_cleanup(&ctx);
  return 0;
}
