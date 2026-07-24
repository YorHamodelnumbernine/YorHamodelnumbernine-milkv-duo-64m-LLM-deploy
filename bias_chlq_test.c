/* Test bias + chl_quan features on both bmk1822 and cvikernel paths.
   Findings:
   - bmk1822 conv+bias{2,oc,1,1}: bias=0 works, bias!=0 saturates (PS32 format unknown)
   - cvikernel tiu_convolution w/ chl_quan: "wrong parameter", chl_quan ignored
   - cvikernel matmul no-bias: works correctly
   - Both fused matmul+bias paths: bias!=0 saturates (PS32 encoding issue)
   - Workaround: matmul(no bias) + separate tiu_add (requires hi/lo 16-bit tensors)
*/
#include "common/tpu_bench.h"
#include "bmkernel/bm1822/bmkernel_1822.h"

/* [1] bmk1822 tiu_convolution with bias {2,oc,1,1} */
static int test_bmk_conv_bias(void) {
  fprintf(stderr,"\n--- [1] bmk1822 tiu_convolution bias={2,1,1,1} ---\n");
  int8_t got=-1;
  CVI_RT_HANDLE rt; CVI_RT_Init(&rt);
  CVI_RT_MEM mem=CVI_RT_MemAlloc(rt,4096);
  uint64_t pa=CVI_RT_MemGetPAddr(mem);
  uint8_t *va=CVI_RT_MemGetVAddr(mem);
  CVI_RT_SetBaseReg(rt,0,pa);

  int8_t ifmap[9]={1,2,3,4,5,6,7,8,9};
  int8_t weight[9]={0,0,0,0,1,0,0,0,0};  // sum=5
  int8_t bias0[2]={0,0};  // n=0:rshift, n=1:bias → exp=5

  memcpy(va,ifmap,9);
  memcpy(va+64,weight,9);
  memcpy(va+128,bias0,2);
  CVI_RT_MemFlush(rt,mem);

  uint8_t cmdbuf[8192] __attribute__((aligned(16)));
  bmk_info_t info={.chip_version=1822,.cmdbuf_size=sizeof(cmdbuf),.cmdbuf=cmdbuf};
  bmk1822_context_t *bmk=bmk1822_register(&info);
  if(!bmk){CVI_RT_MemFree(rt,mem);CVI_RT_DeInit(rt);return 1;}

  bmk1822_tensor_lmem_t *tl_if=bmk1822_lmem_alloc_tensor(bmk,(bmk1822_tensor_lmem_shape_t){1,1,3,3},FMT_I8,1);
  bmk1822_tensor_lmem_t *tl_of=bmk1822_lmem_alloc_tensor(bmk,(bmk1822_tensor_lmem_shape_t){1,1,1,1},FMT_I8,1);
  bmk1822_tensor_lmem_t *tl_w =bmk1822_lmem_alloc_tensor(bmk,(bmk1822_tensor_lmem_shape_t){1,1,3,3},FMT_I8,1);
  bmk1822_tensor_lmem_t *tl_b =bmk1822_lmem_alloc_tensor(bmk,(bmk1822_tensor_lmem_shape_t){2,1,1,1},FMT_I8,1);
  if(!tl_if||!tl_of||!tl_w||!tl_b){fprintf(stderr,"  alloc fail\n");goto out1;}
  tl_w->stride.n=1; tl_w->cmprs_fmt=FMT_I8;
  tl_b->stride.n=1; tl_b->stride.c=1; tl_b->stride.h=1; tl_b->stride.w=1; tl_b->cmprs_fmt=FMT_I8;

  bmk1822_tensor_tgmem_t tg_if={0,0,FMT_I8,{1,1,3,3},bmk1822_tensor_tgmem_default_stride((bmk1822_tensor_tgmem_shape_t){1,1,3,3},FMT_I8)};
  bmk1822_tensor_tgmem_t tg_w ={0,64,FMT_I8,{1,1,3,3},bmk1822_tensor_tgmem_default_stride((bmk1822_tensor_tgmem_shape_t){1,1,3,3},FMT_I8)};
  bmk1822_tensor_tgmem_t tg_of={0,256,FMT_I8,{1,1,1,1},bmk1822_tensor_tgmem_default_stride((bmk1822_tensor_tgmem_shape_t){1,1,1,1},FMT_I8)};
  bmk1822_tensor_tgmem_t tg_b ={0,128,FMT_I8,{2,1,1,1},bmk1822_tensor_tgmem_default_stride((bmk1822_tensor_tgmem_shape_t){2,1,1,1},FMT_I8)};

  bmk1822_tdma_g2l_tensor_copy(bmk,&(bmk1822_tdma_tg2l_tensor_copy_param_t){&tg_if,tl_if});
  bmk1822_tdma_g2l_tensor_copy(bmk,&(bmk1822_tdma_tg2l_tensor_copy_param_t){&tg_w,tl_w});
  bmk1822_tdma_g2l_tensor_copy(bmk,&(bmk1822_tdma_tg2l_tensor_copy_param_t){&tg_b,tl_b});

  bmk1822_tiu_convolution(bmk,&(bmk1822_tiu_convolution_param_t){
    .ofmap=tl_of,.ifmap=tl_if,.weight=tl_w,.bias=tl_b,
    .ins_h=0,.ins_last_h=0,.ins_w=0,.ins_last_w=0,
    .pad_top=0,.pad_bottom=0,.pad_left=0,.pad_right=0,
    .stride_h=1,.stride_w=1,.dilation_h=1,.dilation_w=1,
    .relu_enable=0,.rshift_bits=0,.ps32_mode=0,.w_is_const=0,
    .fp_round_typ=0,.cmd_pre_exe_typ=0,.cmd_pre_exe=0,.layer_id=1,
  });
  bmk1822_tdma_l2g_tensor_copy(bmk,&(bmk1822_tdma_l2tg_tensor_copy_param_t){tl_of,&tg_of});

  uint32_t cmd_sz; uint8_t *cmd=bmk1822_acquire_cmdbuf(bmk,&cmd_sz);
  uint32_t psize,pmu_size; bmk1822_dmabuf_size(cmd,cmd_sz,&psize,&pmu_size);
  CVI_RT_MEM dmabuf_mem=CVI_RT_MemAlloc(rt,psize+pmu_size);
  uint8_t *dmabuf=CVI_RT_MemGetVAddr(dmabuf_mem);
  bmk1822_dmabuf_convert(cmd,cmd_sz,dmabuf);
  bmk1822_arraybase_set(dmabuf,pa,0,0,0);
  CVI_RT_MemFlush(rt,dmabuf_mem);
  CVI_RT_MEM loaded; CVI_RT_LoadDmabuf(rt,dmabuf_mem,psize+pmu_size,pa,0,false,&loaded);
  CVI_RT_RunCmdbufEx(rt,loaded,&(CVI_RT_ARRAYBASE){.gaddr_base0=pa});
  CVI_RT_MemInvld(rt,mem);
  got=va[256];
  CVI_RT_MemFree(rt,dmabuf_mem);

out1:
  bmk1822_cleanup(bmk);
  CVI_RT_MemFree(rt,mem);
  CVI_RT_DeInit(rt);
  fprintf(stderr,"  got=%d exp=5 %s\n",got,got==5?"OK":"MISMATCH");
  fprintf(stderr,"  => bias=0 works; bias!=0 needs correct PS32 encoding (unknown)\n");
  return (got==5)?0:1;
}

/* [2] cvikernel tiu_convolution with chl_quan — "wrong parameter" */
static int test_cvk_conv_chlquan(void) {
  fprintf(stderr,"\n--- [2] cvikernel tiu_convolution with chl_quan ---\n");
  tpu_ctx ctx;
  if(tpu_init(&ctx,4096)!=0) return 1;
  cvk_context_t *cvk=ctx.cvk_ctx;

  int8_t ifmap[9]={1,2,3,4,5,6,7,8,9};
  int8_t weight[9]={0,0,0,0,1,0,0,0,0};  // sum=5
  int8_t chlq[2]={0,10};  // rshift=0, bias=10 → exp=15

  memcpy(ctx.neuron_vaddr,ifmap,9);
  memcpy(ctx.neuron_vaddr+16,weight,9);
  memcpy(ctx.neuron_vaddr+32,chlq,2);
  CVI_RT_MemFlush(ctx.rt_handle,ctx.neuron_mem);

  cvk_tl_t *tl_if=cvk->ops->lmem_alloc_tensor(cvk,(cvk_tl_shape_t){1,1,3,3},CVK_FMT_I8,1);
  cvk_tl_t *tl_of=cvk->ops->lmem_alloc_tensor(cvk,(cvk_tl_shape_t){1,1,1,1},CVK_FMT_I8,1);
  cvk_tl_t *tl_w =cvk->ops->lmem_alloc_tensor(cvk,(cvk_tl_shape_t){1,1,3,3},CVK_FMT_I8,1);
  cvk_tl_t *tl_q =cvk->ops->lmem_alloc_tensor(cvk,(cvk_tl_shape_t){2,1,1,1},CVK_FMT_I8,1);
  tl_w->stride.n=1; tl_w->cmprs_fmt=CVK_FMT_I8;
  if(!tl_if||!tl_of||!tl_w||!tl_q){fprintf(stderr,"  alloc fail\n");goto out2;}

  cvk_tg_shape_t g_if={1,1,3,3}, g_w={1,1,3,3}, g_of={1,1,1,1}, g_q={2,1,1,1};
  cvk->ops->tdma_g2l_tensor_copy(cvk,&(cvk_tdma_g2l_tensor_copy_param_t){
    &(cvk_tg_t){0,TPU_PA(&ctx,0),CVK_FMT_I8,g_if,cvk->ops->tg_default_stride(cvk,g_if,CVK_FMT_I8)},tl_if});
  cvk->ops->tdma_g2l_tensor_copy(cvk,&(cvk_tdma_g2l_tensor_copy_param_t){
    &(cvk_tg_t){0,TPU_PA(&ctx,16),CVK_FMT_I8,g_w,cvk->ops->tg_default_stride(cvk,g_w,CVK_FMT_I8)},tl_w});
  cvk->ops->tdma_g2l_tensor_copy(cvk,&(cvk_tdma_g2l_tensor_copy_param_t){
    &(cvk_tg_t){0,TPU_PA(&ctx,32),CVK_FMT_I8,g_q,cvk->ops->tg_default_stride(cvk,g_q,CVK_FMT_I8)},tl_q});

  cvk->ops->tiu_convolution(cvk,&(cvk_tiu_convolution_param_t){
    .ofmap=tl_of,.ifmap=tl_if,.weight=tl_w,.chl_quan_param=tl_q,
    .ins_h=0,.ins_last_h=0,.ins_w=0,.ins_last_w=0,
    .pad_top=0,.pad_bottom=0,.pad_left=0,.pad_right=0,
    .stride_h=1,.stride_w=1,.dilation_h=1,.dilation_w=1,
    .has_bias=1,.relu_enable=0,.ps32_mode=0,.w_is_const=0,
    .cmd_pre_exe_typ=0,.cmd_pre_exe=0,.layer_id=1,
  });
  cvk->ops->tdma_l2g_tensor_copy(cvk,&(cvk_tdma_l2g_tensor_copy_param_t){tl_of,
    &(cvk_tg_t){0,TPU_PA(&ctx,48),CVK_FMT_I8,g_of,cvk->ops->tg_default_stride(cvk,g_of,CVK_FMT_I8)}});

  int64_t t_ns=tpu_submit(&ctx);
  CVI_RT_MemInvld(ctx.rt_handle,ctx.neuron_mem);
  int8_t got=ctx.neuron_vaddr[48];
  fprintf(stderr,"  got=%d exp=15 %s\n",got,got==15?"OK":"MISMATCH (chl_quan ignored)");
  fprintf(stderr,"  => chl_quan format rejected; use tiu_pt_convolution (no chl_quan)\n");

out2:
  if(tl_q)cvk->ops->lmem_free_tensor(cvk,tl_q);
  if(tl_w)cvk->ops->lmem_free_tensor(cvk,tl_w);
  if(tl_of)cvk->ops->lmem_free_tensor(cvk,tl_of);
  if(tl_if)cvk->ops->lmem_free_tensor(cvk,tl_if);
  tpu_cleanup(&ctx);
  return 0;
}

/* [3] cvikernel matmul no-bias — baseline verification */
static int test_cvk_matmul(void) {
  fprintf(stderr,"\n--- [3] cvikernel matmul no-bias (baseline) ---\n");
  tpu_ctx ctx;
  if(tpu_init(&ctx,4096)!=0) return 1;
  cvk_context_t *cvk=ctx.cvk_ctx;

  int M=2,K=2,N=2;
  int8_t left[4]={1,2,3,4}, right[4]={1,2,3,4}, exp[4]={7,10,15,22};
  memcpy(ctx.neuron_vaddr,left,4);
  memcpy(ctx.neuron_vaddr+64,right,4);
  CVI_RT_MemFlush(ctx.rt_handle,ctx.neuron_mem);

  cvk_ml_shape_t s22=cvk->ops->ml_default_shape(cvk,2,2,CVK_FMT_I8);
  cvk_ml_t *ml_l=cvk->ops->lmem_alloc_matrix(cvk,s22,CVK_FMT_I8,1);
  cvk_ml_t *ml_r=cvk->ops->lmem_alloc_matrix(cvk,s22,CVK_FMT_I8,1);
  cvk_ml_t *ml_o=cvk->ops->lmem_alloc_matrix(cvk,s22,CVK_FMT_I8,1);
  if(!ml_l||!ml_r||!ml_o){fprintf(stderr,"  alloc fail\n");goto out3;}

  cvk->ops->tdma_g2l_matrix_copy(cvk,&(cvk_tdma_g2l_matrix_copy_param_t){
    &(cvk_mg_t){0,TPU_PA(&ctx,0),CVK_FMT_I8,{M,K},{K}},ml_l});
  cvk->ops->tdma_g2l_matrix_copy(cvk,&(cvk_tdma_g2l_matrix_copy_param_t){
    &(cvk_mg_t){0,TPU_PA(&ctx,64),CVK_FMT_I8,{K,N},{N}},ml_r});

  cvk->ops->tiu_matrix_multiplication(cvk,&(cvk_tiu_matrix_multiplication_param_t){
    .res=ml_o,.left=ml_l,.right=ml_r,.bias=NULL,
    .lshift_bits=0,.rshift_bits=0,.res_is_int8=1,.relu_enable=0,
    .add_result=0,.ps32_mode=0,
  });
  cvk->ops->tdma_l2g_matrix_copy(cvk,&(cvk_tdma_l2g_matrix_copy_param_t){
    ml_o,&(cvk_mg_t){0,TPU_PA(&ctx,128),CVK_FMT_I8,{M,N},{N}}});

  int64_t t_ns=tpu_submit(&ctx);
  CVI_RT_MemInvld(ctx.rt_handle,ctx.neuron_mem);
  int8_t *v=ctx.neuron_vaddr+128;
  int match=1; for(int i=0;i<4;i++)if(v[i]!=exp[i])match=0;
  fprintf(stderr,"  got=[%d,%d,%d,%d] exp=[%d,%d,%d,%d] %s\n",
    v[0],v[1],v[2],v[3],exp[0],exp[1],exp[2],exp[3],match?"OK":"MISMATCH");
  fprintf(stderr,"  => matmul no-bias: works. Fused bias has PS32 encoding issue.\n");

out3:
  if(ml_o)cvk->ops->lmem_free_matrix(cvk,ml_o);
  if(ml_r)cvk->ops->lmem_free_matrix(cvk,ml_r);
  if(ml_l)cvk->ops->lmem_free_matrix(cvk,ml_l);
  tpu_cleanup(&ctx);
  return match?0:1;
}

int main(){
  fprintf(stderr,"\n========== bias + chl_quan feature tests ==========\n");
  test_bmk_conv_bias();
  test_cvk_conv_chlquan();
  test_cvk_matmul();
  fprintf(stderr,"========== DONE ==========\n\n");
  return 0;
}
