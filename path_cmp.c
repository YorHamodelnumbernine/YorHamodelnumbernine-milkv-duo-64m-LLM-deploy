/* bmk1822 vs cvikernel path comparison — same computation, same measurement.
   Measures:
     bmk1822: dmabuf_convert + LoadDmabuf + RunCmdbufEx
     cvikernel: Submit (blocking)
   Covers: small conv (3x3→1x1), large conv (64x64→62x62 dw), matmul (2x2 to 64x64) */
#include "common/tpu_bench.h"
#include "bmkernel/bm1822/bmkernel_1822.h"
#include <math.h>

#define N_RUNS 100
#define N_WARM  3     /* warmup runs, excluded from stats */
#define CORNERS_WGT {1,0,1, 0,0,0, 1,0,1}

/* ================================================================
 *  CONV: bmk1822  (same as cvikernel pt_conv: bias=NULL, rshift=0)
 * ================================================================ */
static int64_t conv_bmk1822(CVI_RT_HANDLE rt, CVI_RT_MEM neuron_mem,
                             uint64_t pa, uint8_t *va,
                             int ic, int oc, int ih, int iw, int kh, int kw,
                             int8_t *ifmap, int8_t *weight, int8_t *exp_out)
{
  int oh = ih - kh + 1, ow = iw - kw + 1, o_sz = oc * oh * ow;
  memcpy(va, ifmap, ic*ih*iw);
  memcpy(va + 0x1000, weight, ic*oc*kh*kw);
  CVI_RT_MemFlush(rt, neuron_mem);

  uint8_t cmdbuf[16384] __attribute__((aligned(16)));
  bmk_info_t info = { .chip_version = 1822, .cmdbuf_size = sizeof(cmdbuf), .cmdbuf = cmdbuf };
  bmk1822_context_t *bmk = bmk1822_register(&info);
  if (!bmk) return -1;

  bmk1822_tensor_lmem_shape_t s_if={1,ic,ih,iw}, s_of={1,oc,oh,ow}, s_w={ic,oc,kh,kw};
  bmk1822_tensor_lmem_t *tl_if = bmk1822_lmem_alloc_tensor(bmk, s_if, FMT_I8, 1);
  bmk1822_tensor_lmem_t *tl_of = bmk1822_lmem_alloc_tensor(bmk, s_of, FMT_I8, 1);
  bmk1822_tensor_lmem_t *tl_w  = bmk1822_lmem_alloc_tensor(bmk, s_w,  FMT_I8, 1);
  if (!tl_if||!tl_of||!tl_w) { bmk1822_cleanup(bmk); return -1; }
  tl_w->stride.n = 1; tl_w->cmprs_fmt = FMT_I8;

  bmk1822_tensor_tgmem_shape_t ts_if={1,ic,ih,iw}, ts_w={ic,oc,kh,kw}, ts_of={1,oc,oh,ow};
  bmk1822_tensor_tgmem_t tg_if={
    0,0,FMT_I8,ts_if,bmk1822_tensor_tgmem_default_stride(ts_if,FMT_I8)};
  bmk1822_tensor_tgmem_t tg_w ={
    0,0x1000,FMT_I8,ts_w,bmk1822_tensor_tgmem_default_stride(ts_w,FMT_I8)};
  bmk1822_tensor_tgmem_t tg_of={
    0,0x2000,FMT_I8,ts_of,bmk1822_tensor_tgmem_default_stride(ts_of,FMT_I8)};

  bmk1822_tdma_g2l_tensor_copy(bmk,&(bmk1822_tdma_tg2l_tensor_copy_param_t){&tg_if,tl_if});
  bmk1822_tdma_g2l_tensor_copy(bmk,&(bmk1822_tdma_tg2l_tensor_copy_param_t){&tg_w,tl_w});

  bmk1822_tiu_convolution_param_t p={
    .ofmap=tl_of,.ifmap=tl_if,.weight=tl_w,.bias=NULL,
    .ins_h=0,.ins_last_h=0,.ins_w=0,.ins_last_w=0,
    .pad_top=0,.pad_bottom=0,.pad_left=0,.pad_right=0,
    .stride_h=1,.stride_w=1,.dilation_h=1,.dilation_w=1,
    .relu_enable=0,.rshift_bits=0,
    .ps32_mode=0,.w_is_const=0,.fp_round_typ=0,
    .cmd_pre_exe_typ=0,.cmd_pre_exe=0,.layer_id=1,
  };
  if (!bmk1822_tiu_convolution(bmk,&p)) { bmk1822_cleanup(bmk); return -1; }
  bmk1822_tdma_l2g_tensor_copy(bmk,&(bmk1822_tdma_l2tg_tensor_copy_param_t){tl_of,&tg_of});

  uint32_t cmd_sz; uint8_t *cmd=bmk1822_acquire_cmdbuf(bmk,&cmd_sz);
  uint32_t psize,pmu_size; bmk1822_dmabuf_size(cmd,cmd_sz,&psize,&pmu_size);

  CVI_RT_MEM dmabuf_mem = CVI_RT_MemAlloc(rt, psize+pmu_size);
  if (!dmabuf_mem) { bmk1822_cleanup(bmk); return -1; }
  uint8_t *dmabuf = CVI_RT_MemGetVAddr(dmabuf_mem);

  /* bmk1822 full-path timing: convert + LoadDmabuf + RunCmdbufEx */
  struct timespec t0,t1, tC0,tC1;
  clock_gettime(CLOCK_MONOTONIC,&tC0);
  bmk1822_dmabuf_convert(cmd,cmd_sz,dmabuf);
  bmk1822_arraybase_set(dmabuf,pa,0,0,0);
  CVI_RT_MemFlush(rt,dmabuf_mem);
  clock_gettime(CLOCK_MONOTONIC,&tC1);
  int64_t t_convert=(tC1.tv_sec-tC0.tv_sec)*1000000000LL+(tC1.tv_nsec-tC0.tv_nsec);

  clock_gettime(CLOCK_MONOTONIC,&t0);
  CVI_RT_MEM loaded;
  CVI_RT_LoadDmabuf(rt,dmabuf_mem,psize+pmu_size,pa,0,false,&loaded);
  CVI_RT_ARRAYBASE arr={.gaddr_base0=pa};
  CVI_RT_RunCmdbufEx(rt,loaded,&arr);
  clock_gettime(CLOCK_MONOTONIC,&t1);
  int64_t t_loadrun=(t1.tv_sec-t0.tv_sec)*1000000000LL+(t1.tv_nsec-t0.tv_nsec);
  int64_t t_full=t_convert+t_loadrun;  /* fair comparison vs cvikernel Submit */

  CVI_RT_MemInvld(rt,neuron_mem);
  int ok=1;
  for(int i=0;i<o_sz;i++) if(va[0x2000+i]!=exp_out[i]){ok=0;break;}
  if(!ok){
    fprintf(stderr,"  bmk conv MISMATCH:");
    for(int i=0;i<o_sz;i++)fprintf(stderr,"%d,",va[0x2000+i]);
    fprintf(stderr," exp:");for(int i=0;i<o_sz;i++)fprintf(stderr,"%d,",exp_out[i]);
    fprintf(stderr,"\n");
  }

  CVI_RT_MemFree(rt,dmabuf_mem);
  bmk1822_cleanup(bmk);
  return ok?t_full:-1;
}
static int64_t conv_cvikernel(tpu_ctx *ctx, cvk_context_t *cvk,
                               int ic, int oc, int ih, int iw, int kh, int kw,
                               int8_t *ifmap, int8_t *weight, int8_t *exp_out)
{
  int oh=ih-kh+1, ow=iw-kw+1, o_sz=oc*oh*ow;
  memcpy(ctx->neuron_vaddr, ifmap, ic*ih*iw);
  memcpy(ctx->neuron_vaddr+0x1000, weight, ic*oc*kh*kw);
  CVI_RT_MemFlush(ctx->rt_handle, ctx->neuron_mem);

  cvk_tl_t *tl_if=cvk->ops->lmem_alloc_tensor(cvk,(cvk_tl_shape_t){1,ic,ih,iw},CVK_FMT_I8,1);
  cvk_tl_t *tl_of=cvk->ops->lmem_alloc_tensor(cvk,(cvk_tl_shape_t){1,oc,oh,ow},CVK_FMT_I8,1);
  cvk_tl_t *tl_w =cvk->ops->lmem_alloc_tensor(cvk,(cvk_tl_shape_t){ic,oc,kh,kw},CVK_FMT_I8,1);
  tl_w->stride.n=1; tl_w->cmprs_fmt=CVK_FMT_I8;
  if(!tl_if||!tl_of||!tl_w)return -1;

  cvk_tg_shape_t g_if={1,ic,ih,iw}, g_w={ic,oc,kh,kw}, g_of={1,oc,oh,ow};
  cvk->ops->tdma_g2l_tensor_copy(cvk,&(cvk_tdma_g2l_tensor_copy_param_t){
    &(cvk_tg_t){0,TPU_PA(ctx,0),CVK_FMT_I8,g_if,cvk->ops->tg_default_stride(cvk,g_if,CVK_FMT_I8)},tl_if});
  cvk->ops->tdma_g2l_tensor_copy(cvk,&(cvk_tdma_g2l_tensor_copy_param_t){
    &(cvk_tg_t){0,TPU_PA(ctx,0x1000),CVK_FMT_I8,g_w,cvk->ops->tg_default_stride(cvk,g_w,CVK_FMT_I8)},tl_w});

  cvk->ops->tiu_pt_convolution(cvk,&(cvk_tiu_pt_convolution_param_t){
    .ofmap=tl_of,.ifmap=tl_if,.weight=tl_w,.bias=NULL,
    .ins_h=0,.ins_last_h=0,.ins_w=0,.ins_last_w=0,
    .pad_top=0,.pad_bottom=0,.pad_left=0,.pad_right=0,
    .stride_h=1,.stride_w=1,.dilation_h=1,.dilation_w=1,
    .relu_enable=0,.rshift_bits=0,.ps32_mode=0,.w_is_const=0,.layer_id=1,
  });

  cvk->ops->tdma_l2g_tensor_copy(cvk,&(cvk_tdma_l2g_tensor_copy_param_t){tl_of,
    &(cvk_tg_t){0,TPU_PA(ctx,0x2000),CVK_FMT_I8,g_of,cvk->ops->tg_default_stride(cvk,g_of,CVK_FMT_I8)}});

  struct timespec t0,t1;
  clock_gettime(CLOCK_MONOTONIC,&t0);
  CVI_RT_Submit(ctx->rt_khandle);
  clock_gettime(CLOCK_MONOTONIC,&t1);
  int64_t t_ns=(t1.tv_sec-t0.tv_sec)*1000000000LL+(t1.tv_nsec-t0.tv_nsec);

  CVI_RT_MemInvld(ctx->rt_handle, ctx->neuron_mem);
  int ok=1;
  for(int i=0;i<o_sz;i++)if(ctx->neuron_vaddr[0x2000+i]!=exp_out[i]){ok=0;break;}
  if(!ok){
    fprintf(stderr,"  cvk conv MISMATCH:");
    for(int i=0;i<o_sz;i++)fprintf(stderr,"%d,",ctx->neuron_vaddr[0x2000+i]);
    fprintf(stderr," exp:");for(int i=0;i<o_sz;i++)fprintf(stderr,"%d,",exp_out[i]);
    fprintf(stderr,"\n");
  }

  cvk->ops->lmem_free_tensor(cvk,tl_w);
  cvk->ops->lmem_free_tensor(cvk,tl_of);
  cvk->ops->lmem_free_tensor(cvk,tl_if);
  return ok?t_ns:-1;
}

/* ================================================================
 *  MATMUL: bmk1822
 * ================================================================ */
static int64_t matmul_bmk1822(CVI_RT_HANDLE rt, CVI_RT_MEM neuron_mem,
                               uint64_t pa, uint8_t *va, int M, int K, int N,
                               int8_t *left, int8_t *right, int8_t *exp_out)
{
  int l_sz=M*K, r_sz=K*N, o_sz=M*N;
  memcpy(va,left,l_sz);
  memcpy(va+0x1000,right,r_sz);
  CVI_RT_MemFlush(rt,neuron_mem);

  uint8_t cmdbuf[16384] __attribute__((aligned(16)));
  bmk_info_t info={.chip_version=1822,.cmdbuf_size=sizeof(cmdbuf),.cmdbuf=cmdbuf};
  bmk1822_context_t *bmk=bmk1822_register(&info);
  if(!bmk)return -1;

  bmk1822_matrix_lmem_shape_t sl=bmk1822_matrix_lmem_default_shape(bmk,M,K,FMT_I8);
  bmk1822_matrix_lmem_shape_t sr=bmk1822_matrix_lmem_default_shape(bmk,K,N,FMT_I8);
  bmk1822_matrix_lmem_shape_t so=bmk1822_matrix_lmem_default_shape(bmk,M,N,FMT_I8);
  bmk1822_matrix_lmem_t *ml_l=bmk1822_lmem_alloc_matrix(bmk,sl,FMT_I8,1);
  bmk1822_matrix_lmem_t *ml_r=bmk1822_lmem_alloc_matrix(bmk,sr,FMT_I8,1);
  bmk1822_matrix_lmem_t *ml_o=bmk1822_lmem_alloc_matrix(bmk,so,FMT_I8,1);
  if(!ml_l||!ml_r||!ml_o){bmk1822_cleanup(bmk);return -1;}

  bmk1822_matrix_tgmem_t mg_l={0,0,FMT_I8,{M,K},{K}};
  bmk1822_matrix_tgmem_t mg_r={0,0x1000,FMT_I8,{K,N},{N}};
  bmk1822_matrix_tgmem_t mg_o={0,0x2000,FMT_I8,{M,N},{N}};

  bmk1822_tdma_g2l_matrix_copy(bmk,&(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_l,ml_l});
  bmk1822_tdma_g2l_matrix_copy(bmk,&(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_r,ml_r});

  bmk1822_tiu_matrix_multiplication_param_t p={
    .res=ml_o,.left=ml_l,.right=ml_r,.bias=NULL,
    .lshift_bits=0,.rshift_bits=0,.res_is_int8=1,.relu_enable=0,
    .add_result=0,.ps32_mode=0,.layer_id=1,
  };
  if(!bmk1822_tiu_matrix_multiplication(bmk,&p)){bmk1822_cleanup(bmk);return -1;}
  bmk1822_tdma_l2g_matrix_copy(bmk,&(bmk1822_tdma_l2tg_matrix_copy_param_t){ml_o,&mg_o});

  uint32_t cmd_sz; uint8_t *cmd=bmk1822_acquire_cmdbuf(bmk,&cmd_sz);
  uint32_t psize,pmu_size; bmk1822_dmabuf_size(cmd,cmd_sz,&psize,&pmu_size);

  CVI_RT_MEM dmabuf_mem=CVI_RT_MemAlloc(rt,psize+pmu_size);
  if(!dmabuf_mem){bmk1822_cleanup(bmk);return -1;}
  uint8_t *dmabuf=CVI_RT_MemGetVAddr(dmabuf_mem);

  /* bmk1822 full-path timing: convert + LoadDmabuf + RunCmdbufEx */
  struct timespec t0,t1, tC0,tC1;
  clock_gettime(CLOCK_MONOTONIC,&tC0);
  bmk1822_dmabuf_convert(cmd,cmd_sz,dmabuf);
  bmk1822_arraybase_set(dmabuf,pa,0,0,0);
  CVI_RT_MemFlush(rt,dmabuf_mem);
  clock_gettime(CLOCK_MONOTONIC,&tC1);
  int64_t t_convert=(tC1.tv_sec-tC0.tv_sec)*1000000000LL+(tC1.tv_nsec-tC0.tv_nsec);

  clock_gettime(CLOCK_MONOTONIC,&t0);
  CVI_RT_MEM loaded;
  CVI_RT_LoadDmabuf(rt,dmabuf_mem,psize+pmu_size,pa,0,false,&loaded);
  CVI_RT_ARRAYBASE arr={.gaddr_base0=pa};
  CVI_RT_RunCmdbufEx(rt,loaded,&arr);
  clock_gettime(CLOCK_MONOTONIC,&t1);
  int64_t t_loadrun=(t1.tv_sec-t0.tv_sec)*1000000000LL+(t1.tv_nsec-t0.tv_nsec);
  int64_t t_full=t_convert+t_loadrun;

  CVI_RT_MemInvld(rt,neuron_mem);
  int ok=1;
  for(int i=0;i<o_sz;i++)if(va[0x2000+i]!=exp_out[i]){ok=0;break;}

  CVI_RT_MemFree(rt,dmabuf_mem);
  bmk1822_cleanup(bmk);
  return ok?t_full:-1;
}

/* ================================================================
 *  MATMUL: cvikernel
 * ================================================================ */
static int64_t matmul_cvikernel(tpu_ctx *ctx, cvk_context_t *cvk,
                                 int M, int K, int N,
                                 int8_t *left, int8_t *right, int8_t *exp_out)
{
  int l_sz=M*K, r_sz=K*N, o_sz=M*N;
  memcpy(ctx->neuron_vaddr,left,l_sz);
  memcpy(ctx->neuron_vaddr+0x1000,right,r_sz);
  CVI_RT_MemFlush(ctx->rt_handle,ctx->neuron_mem);

  cvk_ml_shape_t sl=cvk->ops->ml_default_shape(cvk,M,K,CVK_FMT_I8);
  cvk_ml_shape_t sr=cvk->ops->ml_default_shape(cvk,K,N,CVK_FMT_I8);
  cvk_ml_shape_t so=cvk->ops->ml_default_shape(cvk,M,N,CVK_FMT_I8);
  cvk_ml_t *ml_l=cvk->ops->lmem_alloc_matrix(cvk,sl,CVK_FMT_I8,1);
  cvk_ml_t *ml_r=cvk->ops->lmem_alloc_matrix(cvk,sr,CVK_FMT_I8,1);
  cvk_ml_t *ml_o=cvk->ops->lmem_alloc_matrix(cvk,so,CVK_FMT_I8,1);
  if(!ml_l||!ml_r||!ml_o)return -1;

  cvk->ops->tdma_g2l_matrix_copy(cvk,&(cvk_tdma_g2l_matrix_copy_param_t){
    &(cvk_mg_t){0,TPU_PA(ctx,0),CVK_FMT_I8,{M,K},{K}},ml_l});
  cvk->ops->tdma_g2l_matrix_copy(cvk,&(cvk_tdma_g2l_matrix_copy_param_t){
    &(cvk_mg_t){0,TPU_PA(ctx,0x1000),CVK_FMT_I8,{K,N},{N}},ml_r});

  cvk->ops->tiu_matrix_multiplication(cvk,&(cvk_tiu_matrix_multiplication_param_t){
    .res=ml_o,.left=ml_l,.right=ml_r,.bias=NULL,
    .lshift_bits=0,.rshift_bits=0,.res_is_int8=1,.relu_enable=0,
    .add_result=0,.ps32_mode=0,
  });

  cvk->ops->tdma_l2g_matrix_copy(cvk,&(cvk_tdma_l2g_matrix_copy_param_t){
    ml_o,&(cvk_mg_t){0,TPU_PA(ctx,0x2000),CVK_FMT_I8,{M,N},{N}}});

  struct timespec t0,t1;
  clock_gettime(CLOCK_MONOTONIC,&t0);
  CVI_RT_Submit(ctx->rt_khandle);
  clock_gettime(CLOCK_MONOTONIC,&t1);
  int64_t t_ns=(t1.tv_sec-t0.tv_sec)*1000000000LL+(t1.tv_nsec-t0.tv_nsec);

  CVI_RT_MemInvld(ctx->rt_handle,ctx->neuron_mem);
  int ok=1;
  for(int i=0;i<o_sz;i++)if(ctx->neuron_vaddr[0x2000+i]!=exp_out[i]){ok=0;break;}

  cvk->ops->lmem_free_matrix(cvk,ml_o);
  cvk->ops->lmem_free_matrix(cvk,ml_r);
  cvk->ops->lmem_free_matrix(cvk,ml_l);
  return ok?t_ns:-1;
}

/* ================================================================
 *  Compute expected output for conv
 * ================================================================ */
static void compute_conv_exp(int ic, int oc, int ih, int iw, int kh, int kw,
                              int8_t *ifmap, int8_t *weight, int8_t *exp_out)
{
  int oh=ih-kh+1, ow=iw-kw+1;
  for(int o=0;o<oc;o++){
    for(int y=0;y<oh;y++){
      for(int x=0;x<ow;x++){
        int sum=0;
        for(int c=0;c<ic;c++)
          for(int ky=0;ky<kh;ky++)
            for(int kx=0;kx<kw;kx++)
              sum+=ifmap[c*ih*iw+(y+ky)*iw+(x+kx)]*weight[o*ic*kh*kw+c*kh*kw+ky*kw+kx];
        exp_out[o*oh*ow+y*ow+x]=(int8_t)sum;
      }
    }
  }
}

static void compute_matmul_exp(int M, int K, int N,
                                int8_t *l, int8_t *r, int8_t *exp)
{
  for(int i=0;i<M;i++)
    for(int j=0;j<N;j++){
      int sum=0;
      for(int k=0;k<K;k++) sum+=l[i*K+k]*r[k*N+j];
      exp[i*N+j]=(int8_t)sum;
    }
}

/* ================================================================
 *  MAIN
 * ================================================================ */
int main(){
  fprintf(stderr,"\n========== bmk1822 vs cvikernel PATH COMPARISON ==========\n\n");

  /* --- Conv: small 3x3x1→1x1, corners weight --- */
  {
    int ic=1,oc=1,ih=3,iw=3,kh=3,kw=3;
    int8_t ifmap[9]={1,2,3,4,5,6,7,8,9};
    int8_t weight[9]=CORNERS_WGT;
    int8_t exp[1]; compute_conv_exp(ic,oc,ih,iw,kh,kw,ifmap,weight,exp);

    fprintf(stderr,"--- CONV %dx%dx%d -> %dx%dx%d (corners weight, exp=%d) ---\n",
            ic,ih,iw,oc,ih-kh+1,iw-kw+1,exp[0]);

    /* bmk1822 */
    {
      CVI_RT_HANDLE rt; CVI_RT_Init(&rt);
      CVI_RT_MEM mem=CVI_RT_MemAlloc(rt,65536);
      uint64_t pa=CVI_RT_MemGetPAddr(mem);
      uint8_t *va=CVI_RT_MemGetVAddr(mem);
      CVI_RT_SetBaseReg(rt,0,pa);

      double sum=0,sum2=0,tmin=1e9,tmax=0; int ok=1,n=0;
      for(int r=0;r<N_WARM+N_RUNS;r++){
        int64_t t=conv_bmk1822(rt,mem,pa,va,ic,oc,ih,iw,kh,kw,
                                ifmap,weight,exp);
        if(t<0){ok=0;break;}
        if(r<N_WARM) continue;
        double us=t/1000.0; n++; sum+=us; sum2+=us*us; if(us<tmin)tmin=us; if(us>tmax)tmax=us;
      }
      if(ok) { double a=sum/n, st=sqrt(sum2/n - a*a);
        fprintf(stderr,"  bmk1822   avg=%7.1f ± %5.1f  min=%7.1f  max=%7.1f us  [convert+load+run]\n",a,st,tmin,tmax); }
      else    fprintf(stderr,"  bmk1822   FAIL\n");

      CVI_RT_MemFree(rt,mem);
      CVI_RT_DeInit(rt);
    }

    /* cvikernel */
    {
      tpu_ctx ctx;
      if(tpu_init(&ctx,65536)==0){
        cvk_context_t *cvk=ctx.cvk_ctx;
        double sum=0,sum2=0,tmin=1e9,tmax=0; int ok=1,n=0;
        for(int r=0;r<N_WARM+N_RUNS;r++){
          int64_t t=conv_cvikernel(&ctx,cvk,ic,oc,ih,iw,kh,kw,
                                    ifmap,weight,exp);
          if(t<0){ok=0;break;}
          if(r<N_WARM) continue;
          double us=t/1000.0; n++; sum+=us; sum2+=us*us; if(us<tmin)tmin=us; if(us>tmax)tmax=us;
        }
        if(ok) { double a=sum/n, st=sqrt(sum2/n - a*a);
          fprintf(stderr,"  cvikernel avg=%7.1f ± %5.1f  min=%7.1f  max=%7.1f us\n",a,st,tmin,tmax); }
        else    fprintf(stderr,"  cvikernel FAIL\n");
        tpu_cleanup(&ctx);
      }
    }
  }

  /* --- Conv: larger 64x64x1 depthwise 3x3 -> 62x62x1 --- */
  {
    int ic=1,oc=1,ih=64,iw=64,kh=3,kw=3,oh=62,ow=62;
    int8_t ifmap[64*64], weight[9], exp[62*62];
    for(int i=0;i<64*64;i++) ifmap[i]=(int8_t)(i&3);
    for(int i=0;i<9;i++) weight[i]=(int8_t)((i==4)?1:0);  /* center=1 */
    compute_conv_exp(ic,oc,ih,iw,kh,kw,ifmap,weight,exp);

    fprintf(stderr,"\n--- CONV %dx%dx%d -> %dx%dx%d (dw center=1, exp[0]=%d) ---\n",
            ic,ih,iw,oc,oh,ow,exp[0]);

    /* bmk1822 */
    {
      CVI_RT_HANDLE rt; CVI_RT_Init(&rt);
      CVI_RT_MEM mem=CVI_RT_MemAlloc(rt,65536);
      uint64_t pa=CVI_RT_MemGetPAddr(mem);
      uint8_t *va=CVI_RT_MemGetVAddr(mem);
      CVI_RT_SetBaseReg(rt,0,pa);

      double sum=0,sum2=0,tmin=1e9,tmax=0; int ok=1,n=0;
      for(int r=0;r<N_WARM+N_RUNS;r++){
        int64_t t=conv_bmk1822(rt,mem,pa,va,ic,oc,ih,iw,kh,kw,
                                ifmap,weight,exp);
        if(t<0){ok=0;break;}
        if(r<N_WARM) continue;
        double us=t/1000.0; n++; sum+=us; sum2+=us*us; if(us<tmin)tmin=us; if(us>tmax)tmax=us;
      }
      if(ok) { double a=sum/n, st=sqrt(sum2/n - a*a);
        fprintf(stderr,"  bmk1822   avg=%7.1f ± %5.1f  min=%7.1f  max=%7.1f us  [convert+load+run]\n",a,st,tmin,tmax); }
      else    fprintf(stderr,"  bmk1822   FAIL\n");

      CVI_RT_MemFree(rt,mem);
      CVI_RT_DeInit(rt);
    }

    /* cvikernel */
    {
      tpu_ctx ctx;
      if(tpu_init(&ctx,65536)==0){
        cvk_context_t *cvk=ctx.cvk_ctx;
        double sum=0,sum2=0,tmin=1e9,tmax=0; int ok=1,n=0;
        for(int r=0;r<N_WARM+N_RUNS;r++){
          int64_t t=conv_cvikernel(&ctx,cvk,ic,oc,ih,iw,kh,kw,
                                    ifmap,weight,exp);
          if(t<0){ok=0;break;}
          if(r<N_WARM) continue;
          double us=t/1000.0; n++; sum+=us; sum2+=us*us; if(us<tmin)tmin=us; if(us>tmax)tmax=us;
        }
        if(ok) { double a=sum/n, st=sqrt(sum2/n - a*a);
          fprintf(stderr,"  cvikernel avg=%7.1f ± %5.1f  min=%7.1f  max=%7.1f us\n",a,st,tmin,tmax); }
        else    fprintf(stderr,"  cvikernel FAIL\n");
        tpu_cleanup(&ctx);
      }
    }
  }

  /* --- Matmul: 2x2 * 2x2 --- */
  {
    int M=2,K=2,N=2;
    int8_t ml[4]={1,2,3,4}, mr[4]={1,2,3,4}, mexp[4];
    compute_matmul_exp(M,K,N,ml,mr,mexp);

    fprintf(stderr,"\n--- MATMUL %dx%d * %dx%d = %dx%d (exp=[%d,%d,%d,%d]) ---\n",
            M,K,K,N,M,N,mexp[0],mexp[1],mexp[2],mexp[3]);

    /* bmk1822 */
    {
      CVI_RT_HANDLE rt; CVI_RT_Init(&rt);
      CVI_RT_MEM mem=CVI_RT_MemAlloc(rt,65536);
      uint64_t pa=CVI_RT_MemGetPAddr(mem);
      uint8_t *va=CVI_RT_MemGetVAddr(mem);
      CVI_RT_SetBaseReg(rt,0,pa);

      double sum=0,sum2=0,tmin=1e9,tmax=0; int ok=1,n=0;
      for(int r=0;r<N_WARM+N_RUNS;r++){
        int64_t t=matmul_bmk1822(rt,mem,pa,va,M,K,N,ml,mr,mexp);
        if(t<0){ok=0;break;}
        if(r<N_WARM) continue;
        double us=t/1000.0; n++; sum+=us; sum2+=us*us; if(us<tmin)tmin=us; if(us>tmax)tmax=us;
      }
      if(ok) { double a=sum/n, st=sqrt(sum2/n - a*a);
        fprintf(stderr,"  bmk1822   avg=%7.1f ± %5.1f  min=%7.1f  max=%7.1f us  [convert+load+run]\n",a,st,tmin,tmax); }
      else    fprintf(stderr,"  bmk1822   FAIL\n");

      CVI_RT_MemFree(rt,mem);
      CVI_RT_DeInit(rt);
    }

    /* cvikernel */
    {
      tpu_ctx ctx;
      if(tpu_init(&ctx,65536)==0){
        cvk_context_t *cvk=ctx.cvk_ctx;
        double sum=0,sum2=0,tmin=1e9,tmax=0; int ok=1,n=0;
        for(int r=0;r<N_WARM+N_RUNS;r++){
          int64_t t=matmul_cvikernel(&ctx,cvk,M,K,N,ml,mr,mexp);
          if(t<0){ok=0;break;}
          if(r<N_WARM) continue;
          double us=t/1000.0; n++; sum+=us; sum2+=us*us; if(us<tmin)tmin=us; if(us>tmax)tmax=us;
        }
        if(ok) { double a=sum/n, st=sqrt(sum2/n - a*a);
          fprintf(stderr,"  cvikernel avg=%7.1f ± %5.1f  min=%7.1f  max=%7.1f us\n",a,st,tmin,tmax); }
        else    fprintf(stderr,"  cvikernel FAIL\n");
        tpu_cleanup(&ctx);
      }
    }
  }

  /* --- Matmul: 16x16 * 16x16 --- */
  {
    int M=16,K=16,N=16;
    int8_t ml[256], mr[256], mexp[256];
    for(int i=0;i<256;i++){ml[i]=(int8_t)(i&3);mr[i]=(int8_t)((i+1)&3);}
    compute_matmul_exp(M,K,N,ml,mr,mexp);

    fprintf(stderr,"\n--- MATMUL %dx%d * %dx%d = %dx%d (exp[0]=%d) ---\n",
            M,K,K,N,M,N,mexp[0]);

    /* bmk1822 */
    {
      CVI_RT_HANDLE rt; CVI_RT_Init(&rt);
      CVI_RT_MEM mem=CVI_RT_MemAlloc(rt,65536);
      uint64_t pa=CVI_RT_MemGetPAddr(mem);
      uint8_t *va=CVI_RT_MemGetVAddr(mem);
      CVI_RT_SetBaseReg(rt,0,pa);

      double sum=0,sum2=0,tmin=1e9,tmax=0; int ok=1,n=0;
      for(int r=0;r<N_WARM+N_RUNS;r++){
        int64_t t=matmul_bmk1822(rt,mem,pa,va,M,K,N,ml,mr,mexp);
        if(t<0){ok=0;break;}
        if(r<N_WARM) continue;
        double us=t/1000.0; n++; sum+=us; sum2+=us*us; if(us<tmin)tmin=us; if(us>tmax)tmax=us;
      }
      if(ok) { double a=sum/n, st=sqrt(sum2/n - a*a);
        fprintf(stderr,"  bmk1822   avg=%7.1f ± %5.1f  min=%7.1f  max=%7.1f us  [convert+load+run]\n",a,st,tmin,tmax); }
      else    fprintf(stderr,"  bmk1822   FAIL\n");

      CVI_RT_MemFree(rt,mem);
      CVI_RT_DeInit(rt);
    }

    /* cvikernel */
    {
      tpu_ctx ctx;
      if(tpu_init(&ctx,65536)==0){
        cvk_context_t *cvk=ctx.cvk_ctx;
        double sum=0,sum2=0,tmin=1e9,tmax=0; int ok=1,n=0;
        for(int r=0;r<N_WARM+N_RUNS;r++){
          int64_t t=matmul_cvikernel(&ctx,cvk,M,K,N,ml,mr,mexp);
          if(t<0){ok=0;break;}
          if(r<N_WARM) continue;
          double us=t/1000.0; n++; sum+=us; sum2+=us*us; if(us<tmin)tmin=us; if(us>tmax)tmax=us;
        }
        if(ok) { double a=sum/n, st=sqrt(sum2/n - a*a);
          fprintf(stderr,"  cvikernel avg=%7.1f ± %5.1f  min=%7.1f  max=%7.1f us\n",a,st,tmin,tmax); }
        else    fprintf(stderr,"  cvikernel FAIL\n");
        tpu_cleanup(&ctx);
      }
    }
  }

  /* --- Matmul: 64x64 * 64x64 --- */
  {
    int M=64,K=64,N=64;
    int8_t ml[64*64], mr[64*64], mexp[64*64];
    for(int i=0;i<64*64;i++){ml[i]=(int8_t)(i&3);mr[i]=(int8_t)((i+1)&3);}
    compute_matmul_exp(M,K,N,ml,mr,mexp);

    fprintf(stderr,"\n--- MATMUL %dx%d * %dx%d = %dx%d (exp[0]=%d) ---\n",
            M,K,K,N,M,N,mexp[0]);

    /* bmk1822 */
    {
      CVI_RT_HANDLE rt; CVI_RT_Init(&rt);
      CVI_RT_MEM mem=CVI_RT_MemAlloc(rt,65536);
      uint64_t pa=CVI_RT_MemGetPAddr(mem);
      uint8_t *va=CVI_RT_MemGetVAddr(mem);
      CVI_RT_SetBaseReg(rt,0,pa);

      double sum=0,sum2=0,tmin=1e9,tmax=0; int ok=1,n=0;
      for(int r=0;r<N_WARM+N_RUNS;r++){
        int64_t t=matmul_bmk1822(rt,mem,pa,va,M,K,N,ml,mr,mexp);
        if(t<0){ok=0;break;}
        if(r<N_WARM) continue;
        double us=t/1000.0; n++; sum+=us; sum2+=us*us; if(us<tmin)tmin=us; if(us>tmax)tmax=us;
      }
      if(ok) { double a=sum/n, st=sqrt(sum2/n - a*a);
        fprintf(stderr,"  bmk1822   avg=%7.1f ± %5.1f  min=%7.1f  max=%7.1f us  [convert+load+run]\n",a,st,tmin,tmax); }
      else    fprintf(stderr,"  bmk1822   FAIL\n");

      CVI_RT_MemFree(rt,mem);
      CVI_RT_DeInit(rt);
    }

    /* cvikernel */
    {
      tpu_ctx ctx;
      if(tpu_init(&ctx,65536)==0){
        cvk_context_t *cvk=ctx.cvk_ctx;
        double sum=0,sum2=0,tmin=1e9,tmax=0; int ok=1,n=0;
        for(int r=0;r<N_WARM+N_RUNS;r++){
          int64_t t=matmul_cvikernel(&ctx,cvk,M,K,N,ml,mr,mexp);
          if(t<0){ok=0;break;}
          if(r<N_WARM) continue;
          double us=t/1000.0; n++; sum+=us; sum2+=us*us; if(us<tmin)tmin=us; if(us>tmax)tmax=us;
        }
        if(ok) { double a=sum/n, st=sqrt(sum2/n - a*a);
          fprintf(stderr,"  cvikernel avg=%7.1f ± %5.1f  min=%7.1f  max=%7.1f us\n",a,st,tmin,tmax); }
        else    fprintf(stderr,"  cvikernel FAIL\n");
        tpu_cleanup(&ctx);
      }
    }
  }

  fprintf(stderr,"\n========== DONE path_cmp ==========\n\n");
  return 0;
}
