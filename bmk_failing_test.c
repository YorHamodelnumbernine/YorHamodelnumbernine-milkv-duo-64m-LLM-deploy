/* Test the 6 cvikernel-failing operations on bmk1822 path.
   Report which work, which don't, with correctness and timing. */
#include "common/tpu_bench.h"
#include "bmkernel/bm1822/bmkernel_1822.h"
#include <math.h>

#define N_RUNS 100
#define N_WARM 3

/* ===== [1] bmk1822 tiu_convolution_qdm (chl_quan fused) ===== */
static int test_qdm_conv(void) {
  fprintf(stderr,"\n--- [1] bmk1822 tiu_convolution_qdm (chl_quan) ---\n");
  CVI_RT_HANDLE rt; CVI_RT_Init(&rt);
  CVI_RT_MEM mem=CVI_RT_MemAlloc(rt,4096);
  uint64_t pa=CVI_RT_MemGetPAddr(mem);
  uint8_t *va=CVI_RT_MemGetVAddr(mem);
  CVI_RT_SetBaseReg(rt,0,pa);

  int8_t ifmap[9]={1,2,3,4,5,6,7,8,9};
  int8_t weight[9]={0,0,0,0,1,0,0,0,0}; /* center=1, sum=5 */
  /* bmk1822 QDM chl_quan shape = {1,oc,1,1} — different from cvikernel's {2,oc,1,1} */
  int8_t chlq[1]={0}; /* rshift=0, no multiplier → raw sum */

  memcpy(va,ifmap,9);
  memcpy(va+64,weight,9);
  memcpy(va+128,chlq,1);
  CVI_RT_MemFlush(rt,mem);

  uint8_t cmdbuf[8192] __attribute__((aligned(16)));
  bmk_info_t info={.chip_version=1822,.cmdbuf_size=sizeof(cmdbuf),.cmdbuf=cmdbuf};
  bmk1822_context_t *bmk=bmk1822_register(&info);
  if(!bmk){CVI_RT_MemFree(rt,mem);CVI_RT_DeInit(rt);return 1;}

  bmk1822_tensor_lmem_t *tl_if=bmk1822_lmem_alloc_tensor(bmk,(bmk1822_tensor_lmem_shape_t){1,1,3,3},FMT_I8,1);
  bmk1822_tensor_lmem_t *tl_of=bmk1822_lmem_alloc_tensor(bmk,(bmk1822_tensor_lmem_shape_t){1,1,1,1},FMT_I8,1);
  bmk1822_tensor_lmem_t *tl_w=bmk1822_lmem_alloc_tensor(bmk,(bmk1822_tensor_lmem_shape_t){1,1,3,3},FMT_I8,1);
  bmk1822_tensor_lmem_t *tl_q=bmk1822_lmem_alloc_tensor(bmk,(bmk1822_tensor_lmem_shape_t){1,1,1,1},FMT_I8,1);
  if(!tl_if||!tl_of||!tl_w||!tl_q){fprintf(stderr,"  alloc fail\n");goto out1;}
  tl_w->stride.n=1; tl_w->cmprs_fmt=FMT_I8;
  tl_q->stride.n=1; tl_q->stride.c=1; tl_q->stride.h=1; tl_q->stride.w=1; tl_q->cmprs_fmt=FMT_I8;

  bmk1822_tensor_tgmem_t tg_if={0,0,FMT_I8,{1,1,3,3},bmk1822_tensor_tgmem_default_stride((bmk1822_tensor_tgmem_shape_t){1,1,3,3},FMT_I8)};
  bmk1822_tensor_tgmem_t tg_w={0,64,FMT_I8,{1,1,3,3},bmk1822_tensor_tgmem_default_stride((bmk1822_tensor_tgmem_shape_t){1,1,3,3},FMT_I8)};
  bmk1822_tensor_tgmem_t tg_of={0,256,FMT_I8,{1,1,1,1},bmk1822_tensor_tgmem_default_stride((bmk1822_tensor_tgmem_shape_t){1,1,1,1},FMT_I8)};
  bmk1822_tensor_tgmem_t tg_q={0,128,FMT_I8,{1,1,1,1},bmk1822_tensor_tgmem_default_stride((bmk1822_tensor_tgmem_shape_t){1,1,1,1},FMT_I8)};

  bmk1822_tdma_g2l_tensor_copy(bmk,&(bmk1822_tdma_tg2l_tensor_copy_param_t){&tg_if,tl_if});
  bmk1822_tdma_g2l_tensor_copy(bmk,&(bmk1822_tdma_tg2l_tensor_copy_param_t){&tg_w,tl_w});
  bmk1822_tdma_g2l_tensor_copy(bmk,&(bmk1822_tdma_tg2l_tensor_copy_param_t){&tg_q,tl_q});

  /* Use tiu_convolution_qdm (the QDM variant) */
  bmk1822_tiu_convolution_qdm(bmk,&(bmk1822_tiu_convolution_qdm_param_t){
    .ofmap=tl_of,.ifmap=tl_if,.weight=tl_w,.chl_quan_param=tl_q,
    .ins_h=0,.ins_last_h=0,.ins_w=0,.ins_last_w=0,
    .pad_top=0,.pad_bottom=0,.pad_left=0,.pad_right=0,
    .stride_h=1,.stride_w=1,.dilation_h=1,.dilation_w=1,
    .has_bias=0,.relu_enable=0,.ps32_mode=0,.w_is_const=0,
    .cmd_pre_exe_typ=0,.cmd_pre_exe=0,.ins_val=0,.ins_fp=0,.layer_id=1,
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
  int8_t got=va[256];
  CVI_RT_MemFree(rt,dmabuf_mem);

  /* exp=5 (1*0+2*0+3*0+4*0+5*1+6*0+7*0+8*0+9*0 = 5) */
  int pass=(got==5);
  fprintf(stderr,"  got=%d exp=5 %s\n",got,pass?"OK":"FAIL");
  printf("BMK_TEST|qdm_conv|%s|%d\n",pass?"OK":"FAIL",got);

out1:
  bmk1822_cleanup(bmk);
  CVI_RT_MemFree(rt,mem);
  CVI_RT_DeInit(rt);
  return pass?0:1;
}

/* ===== [2] bmk1822 matmul + bias (bias=0 test, then bias≠0) ===== */
static int test_matmul_bias(void) {
  fprintf(stderr,"\n--- [2] bmk1822 matmul + bias ---\n");
  CVI_RT_HANDLE rt; CVI_RT_Init(&rt);
  CVI_RT_MEM mem=CVI_RT_MemAlloc(rt,4096);
  uint64_t pa=CVI_RT_MemGetPAddr(mem);
  uint8_t *va=CVI_RT_MemGetVAddr(mem);
  CVI_RT_SetBaseReg(rt,0,pa);

  int M=2,K=2,N=2;
  int8_t left[4]={1,2,3,4}, right[4]={1,2,3,4};
  /* bmk1822 bias shape = {2, N}: row0=rshift, row1=bias_val */
  int8_t bias_data[4]={0,0,0,0}; /* row0=rshift=0, row1=bias=0 */

  memcpy(va,left,4);
  memcpy(va+64,right,4);
  memcpy(va+128,bias_data,4);
  CVI_RT_MemFlush(rt,mem);

  uint8_t cmdbuf[8192] __attribute__((aligned(16)));
  bmk_info_t info={.chip_version=1822,.cmdbuf_size=sizeof(cmdbuf),.cmdbuf=cmdbuf};
  bmk1822_context_t *bmk=bmk1822_register(&info);
  if(!bmk){CVI_RT_MemFree(rt,mem);CVI_RT_DeInit(rt);return 1;}

  /* Need 4 matrices: left, right, bias, result */
  bmk1822_matrix_lmem_shape_t sl=bmk1822_matrix_lmem_default_shape(bmk,M,K,FMT_I8);
  bmk1822_matrix_lmem_shape_t sr=bmk1822_matrix_lmem_default_shape(bmk,K,N,FMT_I8);
  bmk1822_matrix_lmem_shape_t sb=bmk1822_matrix_lmem_default_shape(bmk,2,N,FMT_I8);
  bmk1822_matrix_lmem_shape_t so=bmk1822_matrix_lmem_default_shape(bmk,M,N,FMT_I8);
  bmk1822_matrix_lmem_t *ml_l=bmk1822_lmem_alloc_matrix(bmk,sl,FMT_I8,1);
  bmk1822_matrix_lmem_t *ml_r=bmk1822_lmem_alloc_matrix(bmk,sr,FMT_I8,1);
  bmk1822_matrix_lmem_t *ml_b=bmk1822_lmem_alloc_matrix(bmk,sb,FMT_I8,1);
  bmk1822_matrix_lmem_t *ml_o=bmk1822_lmem_alloc_matrix(bmk,so,FMT_I8,1);
  if(!ml_l||!ml_r||!ml_b||!ml_o){fprintf(stderr,"  alloc fail (bmk1822 matmul bias requires 4-matrix alloc)\n"); goto out2;}

  bmk1822_matrix_tgmem_t mg_l={0,0,FMT_I8,{M,K},{K}};
  bmk1822_matrix_tgmem_t mg_r={0,64,FMT_I8,{K,N},{N}};
  bmk1822_matrix_tgmem_t mg_b={0,128,FMT_I8,{2,N},{N}};
  bmk1822_matrix_tgmem_t mg_o={0,256,FMT_I8,{M,N},{N}};

  bmk1822_tdma_g2l_matrix_copy(bmk,&(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_l,ml_l});
  bmk1822_tdma_g2l_matrix_copy(bmk,&(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_r,ml_r});
  bmk1822_tdma_g2l_matrix_copy(bmk,&(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_b,ml_b});

  bmk1822_tiu_matrix_multiplication_param_t p={
    .res=ml_o,.left=ml_l,.right=ml_r,.bias=ml_b,
    .lshift_bits=0,.rshift_bits=0,.res_is_int8=1,.relu_enable=0,
    .add_result=0,.ps32_mode=0,.layer_id=1,
  };
  bmk1822_op_t *op=bmk1822_tiu_matrix_multiplication(bmk,&p);
  if(!op){
    fprintf(stderr,"  API call FAILED (assertion?)\n");
    printf("BMK_TEST|matmul_bias|FAIL|api_assert\n");
    goto out2;
  }
  bmk1822_tdma_l2g_matrix_copy(bmk,&(bmk1822_tdma_l2tg_matrix_copy_param_t){ml_o,&mg_o});

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

  int8_t *v=va+256;
  int8_t exp[4]={7,10,15,22}; /* bias=0: same as no-bias */
  int match=(v[0]==exp[0]&&v[1]==exp[1]&&v[2]==exp[2]&&v[3]==exp[3]);
  fprintf(stderr,"  bias=0: got=[%d,%d,%d,%d] exp=[%d,%d,%d,%d] %s\n",
    v[0],v[1],v[2],v[3],exp[0],exp[1],exp[2],exp[3],match?"OK":"FAIL");
  printf("BMK_TEST|matmul_bias_bias0|%s\n",match?"OK":"FAIL");

  CVI_RT_MemFree(rt,dmabuf_mem);

out2:
  bmk1822_cleanup(bmk);
  CVI_RT_MemFree(rt,mem);
  CVI_RT_DeInit(rt);
  return 0;
}

/* ===== [3] bmk1822 matmul + add_result=1 ===== */
static int test_matmul_addres(void) {
  fprintf(stderr,"\n--- [3] bmk1822 matmul add_result=1 ---\n");
  CVI_RT_HANDLE rt; CVI_RT_Init(&rt);
  CVI_RT_MEM mem=CVI_RT_MemAlloc(rt,4096);
  uint64_t pa=CVI_RT_MemGetPAddr(mem);
  uint8_t *va=CVI_RT_MemGetVAddr(mem);
  CVI_RT_SetBaseReg(rt,0,pa);

  int M=2,K=2,N=2;
  int8_t left[4]={1,0,0,1}, right[4]={2,0,0,2}; /* identity * 2I = 2I */
  int8_t init[4]={5,5,5,5};

  memcpy(va,left,4);
  memcpy(va+64,right,4);
  memcpy(va+128,init,4);
  CVI_RT_MemFlush(rt,mem);

  uint8_t cmdbuf[8192] __attribute__((aligned(16)));
  bmk_info_t info={.chip_version=1822,.cmdbuf_size=sizeof(cmdbuf),.cmdbuf=cmdbuf};
  bmk1822_context_t *bmk=bmk1822_register(&info);
  if(!bmk){CVI_RT_MemFree(rt,mem);CVI_RT_DeInit(rt);return 1;}

  /* add_result=1 requires res_row = left_row * 2 */
  bmk1822_matrix_lmem_shape_t sl=bmk1822_matrix_lmem_default_shape(bmk,M,K,FMT_I8);
  bmk1822_matrix_lmem_shape_t sr=bmk1822_matrix_lmem_default_shape(bmk,K,N,FMT_I8);
  bmk1822_matrix_lmem_shape_t so=bmk1822_matrix_lmem_default_shape(bmk,M*2,N,FMT_I8);
  bmk1822_matrix_lmem_t *ml_l=bmk1822_lmem_alloc_matrix(bmk,sl,FMT_I8,1);
  bmk1822_matrix_lmem_t *ml_r=bmk1822_lmem_alloc_matrix(bmk,sr,FMT_I8,1);
  bmk1822_matrix_lmem_t *ml_o=bmk1822_lmem_alloc_matrix(bmk,so,FMT_I8,1);
  if(!ml_l||!ml_r||!ml_o){fprintf(stderr,"  alloc fail\n");goto out3;}

  bmk1822_matrix_tgmem_t mg_l={0,0,FMT_I8,{M,K},{K}};
  bmk1822_matrix_tgmem_t mg_r={0,64,FMT_I8,{K,N},{N}};
  bmk1822_matrix_tgmem_t mg_o={0,128,FMT_I8,{M*2,N},{N}};

  bmk1822_tdma_g2l_matrix_copy(bmk,&(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_l,ml_l});
  bmk1822_tdma_g2l_matrix_copy(bmk,&(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_r,ml_r});
  /* Pre-load result with init values to add to */
  bmk1822_tdma_g2l_matrix_copy(bmk,&(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_o,ml_o});

  bmk1822_tiu_matrix_multiplication_param_t p={
    .res=ml_o,.left=ml_l,.right=ml_r,.bias=NULL,
    .lshift_bits=0,.rshift_bits=0,.res_is_int8=1,.relu_enable=0,
    .add_result=1,.ps32_mode=0,.layer_id=1,
  };
  int api_ok=(bmk1822_tiu_matrix_multiplication(bmk,&p)!=NULL);
  if(!api_ok){
    fprintf(stderr,"  API call FAILED (assertion?)\n");
    printf("BMK_TEST|matmul_addres|FAIL|api_assert\n");
    goto out3;
  }
  bmk1822_tdma_l2g_matrix_copy(bmk,&(bmk1822_tdma_l2tg_matrix_copy_param_t){ml_o,&mg_o});

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

  int8_t *v=va+128;
  int8_t exp[4]={7,5,5,7}; /* init=5 + matmul result (2,0,0,2) = (7,5,5,7) */
  int match=(v[0]==exp[0]&&v[1]==exp[1]&&v[2]==exp[2]&&v[3]==exp[3]);
  fprintf(stderr,"  got=[%d,%d,%d,%d] exp=[%d,%d,%d,%d] %s\n",
    v[0],v[1],v[2],v[3],exp[0],exp[1],exp[2],exp[3],match?"OK":"FAIL");
  printf("BMK_TEST|matmul_addres|%s\n",match?"OK":"FAIL");

  CVI_RT_MemFree(rt,dmabuf_mem);

out3:
  bmk1822_cleanup(bmk);
  CVI_RT_MemFree(rt,mem);
  CVI_RT_DeInit(rt);
  return 0;
}

/* ===== [4] bmk1822 element_wise_mul_qdm ===== */
static int test_mul_qdm(void) {
  fprintf(stderr,"\n--- [4] bmk1822 element_wise_mul_qdm ---\n");
  CVI_RT_HANDLE rt; CVI_RT_Init(&rt);
  CVI_RT_MEM mem=CVI_RT_MemAlloc(rt,4096);
  uint64_t pa=CVI_RT_MemGetPAddr(mem);
  uint8_t *va=CVI_RT_MemGetVAddr(mem);
  CVI_RT_SetBaseReg(rt,0,pa);

  int8_t a[16]={1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
  int8_t b[16]={2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2};
  memcpy(va,a,16);
  memcpy(va+32,b,16);
  CVI_RT_MemFlush(rt,mem);

  uint8_t cmdbuf[8192] __attribute__((aligned(16)));
  bmk_info_t info={.chip_version=1822,.cmdbuf_size=sizeof(cmdbuf),.cmdbuf=cmdbuf};
  bmk1822_context_t *bmk=bmk1822_register(&info);
  if(!bmk){CVI_RT_MemFree(rt,mem);CVI_RT_DeInit(rt);return 1;}

  bmk1822_tensor_lmem_t *tl_a=bmk1822_lmem_alloc_tensor(bmk,(bmk1822_tensor_lmem_shape_t){1,1,4,4},FMT_I8,1);
  bmk1822_tensor_lmem_t *tl_b=bmk1822_lmem_alloc_tensor(bmk,(bmk1822_tensor_lmem_shape_t){1,1,4,4},FMT_I8,1);
  bmk1822_tensor_lmem_t *tl_rl=bmk1822_lmem_alloc_tensor(bmk,(bmk1822_tensor_lmem_shape_t){1,1,4,4},FMT_I8,1);
  bmk1822_tensor_lmem_t *tl_rh=bmk1822_lmem_alloc_tensor(bmk,(bmk1822_tensor_lmem_shape_t){1,1,4,4},FMT_I8,1);
  if(!tl_a||!tl_b||!tl_rl||!tl_rh){fprintf(stderr,"  alloc fail\n");goto out4;}

  bmk1822_tensor_tgmem_t tg_a={0,0,FMT_I8,{1,1,4,4},bmk1822_tensor_tgmem_default_stride((bmk1822_tensor_tgmem_shape_t){1,1,4,4},FMT_I8)};
  bmk1822_tensor_tgmem_t tg_b={0,32,FMT_I8,{1,1,4,4},bmk1822_tensor_tgmem_default_stride((bmk1822_tensor_tgmem_shape_t){1,1,4,4},FMT_I8)};
  bmk1822_tensor_tgmem_t tg_r={0,64,FMT_I8,{1,1,4,4},bmk1822_tensor_tgmem_default_stride((bmk1822_tensor_tgmem_shape_t){1,1,4,4},FMT_I8)};

  bmk1822_tdma_g2l_tensor_copy(bmk,&(bmk1822_tdma_tg2l_tensor_copy_param_t){&tg_a,tl_a});
  bmk1822_tdma_g2l_tensor_copy(bmk,&(bmk1822_tdma_tg2l_tensor_copy_param_t){&tg_b,tl_b});

  bmk1822_tiu_element_wise_mul_qdm(bmk,&(bmk1822_tiu_element_wise_mul_qdm_param_t){
    .res_high=tl_rh,.res_low=tl_rl,.a=tl_a,
    .b_is_const=1,.b_const={.val=1,.is_signed=1},
    .multiplier=2,.rshift_bits=0,.relu_enable=0,.layer_id=1,
  });
  bmk1822_tdma_l2g_tensor_copy(bmk,&(bmk1822_tdma_l2tg_tensor_copy_param_t){tl_rl,&tg_r});

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

  int8_t *got=va+64;
  int ok=1, nz=0;
  for(int i=0;i<16;i++){
    int exp_i=a[i]*b[i];
    if(got[i]!=exp_i){ok=0; if(got[i]!=0)nz=1;}
  }
  fprintf(stderr,"  got=[%d,%d,...,%d] exp=[%d,%d,...,%d] %s (nz=%d)\n",
    got[0],got[1],got[15],a[0]*b[0],a[1]*b[1],a[15]*b[15],ok?"OK":"FAIL",nz);
  printf("BMK_TEST|mul_qdm|%s|all_zero=%d\n",ok?"OK":"FAIL",!nz);
  CVI_RT_MemFree(rt,dmabuf_mem);

out4:
  bmk1822_cleanup(bmk);
  CVI_RT_MemFree(rt,mem);
  CVI_RT_DeInit(rt);
  return 0;
}

/* ===== [5] bmk1822 matrix_multiplication_qdm ===== */
static int test_matmul_qdm(void) {
  fprintf(stderr,"\n--- [5] bmk1822 matrix_multiplication_qdm ---\n");
  CVI_RT_HANDLE rt; CVI_RT_Init(&rt);
  CVI_RT_MEM mem=CVI_RT_MemAlloc(rt,4096);
  uint64_t pa=CVI_RT_MemGetPAddr(mem);
  uint8_t *va=CVI_RT_MemGetVAddr(mem);
  CVI_RT_SetBaseReg(rt,0,pa);

  int M=2,K=2,N=2;
  int8_t left[4]={1,2,3,4}, right[4]={1,2,3,4};
  memcpy(va,left,4);
  memcpy(va+64,right,4);
  CVI_RT_MemFlush(rt,mem);

  uint8_t cmdbuf[8192] __attribute__((aligned(16)));
  bmk_info_t info={.chip_version=1822,.cmdbuf_size=sizeof(cmdbuf),.cmdbuf=cmdbuf};
  bmk1822_context_t *bmk=bmk1822_register(&info);
  if(!bmk){CVI_RT_MemFree(rt,mem);CVI_RT_DeInit(rt);return 1;}

  bmk1822_matrix_lmem_shape_t sl=bmk1822_matrix_lmem_default_shape(bmk,M,K,FMT_I8);
  bmk1822_matrix_lmem_shape_t sr=bmk1822_matrix_lmem_default_shape(bmk,K,N,FMT_I8);
  bmk1822_matrix_lmem_shape_t so=bmk1822_matrix_lmem_default_shape(bmk,M,N,FMT_I8);
  bmk1822_matrix_lmem_t *ml_l=bmk1822_lmem_alloc_matrix(bmk,sl,FMT_I8,1);
  bmk1822_matrix_lmem_t *ml_r=bmk1822_lmem_alloc_matrix(bmk,sr,FMT_I8,1);
  bmk1822_matrix_lmem_t *ml_o=bmk1822_lmem_alloc_matrix(bmk,so,FMT_I8,1);
  if(!ml_l||!ml_r||!ml_o){fprintf(stderr,"  alloc fail\n");goto out5;}

  bmk1822_matrix_tgmem_t mg_l={0,0,FMT_I8,{M,K},{K}};
  bmk1822_matrix_tgmem_t mg_r={0,64,FMT_I8,{K,N},{N}};
  bmk1822_matrix_tgmem_t mg_o={0,128,FMT_I8,{M,N},{N}};

  bmk1822_tdma_g2l_matrix_copy(bmk,&(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_l,ml_l});
  bmk1822_tdma_g2l_matrix_copy(bmk,&(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_r,ml_r});

  int api_ok=(bmk1822_tiu_matrix_multiplication_qdm(bmk,&(bmk1822_tiu_matrix_multiplication_qdm_param_t){
    .res=ml_o,.left=ml_l,.right=ml_r,.bias=NULL,
    .lshift_bits=0,.rshift_bits=0,.res_is_int8=1,.relu_enable=0,
    .add_result=0,.ps32_mode=0,.layer_id=1,
  })!=NULL);
  if(!api_ok){
    fprintf(stderr,"  API call FAILED (assertion?)\n");
    printf("BMK_TEST|matmul_qdm|FAIL|api_assert\n");
    goto out5;
  }
  bmk1822_tdma_l2g_matrix_copy(bmk,&(bmk1822_tdma_l2tg_matrix_copy_param_t){ml_o,&mg_o});

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

  int8_t *v=va+128;
  int8_t exp[4]={7,10,15,22};
  int match=(v[0]==exp[0]&&v[1]==exp[1]&&v[2]==exp[2]&&v[3]==exp[3]);
  fprintf(stderr,"  got=[%d,%d,%d,%d] exp=[%d,%d,%d,%d] %s\n",
    v[0],v[1],v[2],v[3],exp[0],exp[1],exp[2],exp[3],match?"OK":"FAIL");
  printf("BMK_TEST|matmul_qdm|%s\n",match?"OK":"FAIL");
  CVI_RT_MemFree(rt,dmabuf_mem);

out5:
  bmk1822_cleanup(bmk);
  CVI_RT_MemFree(rt,mem);
  CVI_RT_DeInit(rt);
  return 0;
}

/* ===== [6] bmk1822 tiu_add (element-wise add, 16-bit hi/lo) ===== */
static int test_elem_add(void) {
  fprintf(stderr,"\n--- [6] bmk1822 element_wise_add (hi/lo 16-bit) ---\n");
  CVI_RT_HANDLE rt; CVI_RT_Init(&rt);
  CVI_RT_MEM mem=CVI_RT_MemAlloc(rt,4096);
  uint64_t pa=CVI_RT_MemGetPAddr(mem);
  uint8_t *va=CVI_RT_MemGetVAddr(mem);
  CVI_RT_SetBaseReg(rt,0,pa);

  int8_t a[16]={1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
  int8_t b[16]={5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5};
  memcpy(va,a,16);
  memcpy(va+32,b,16);
  CVI_RT_MemFlush(rt,mem);

  uint8_t cmdbuf[8192] __attribute__((aligned(16)));
  bmk_info_t info={.chip_version=1822,.cmdbuf_size=sizeof(cmdbuf),.cmdbuf=cmdbuf};
  bmk1822_context_t *bmk=bmk1822_register(&info);
  if(!bmk){CVI_RT_MemFree(rt,mem);CVI_RT_DeInit(rt);return 1;}

  bmk1822_tensor_lmem_t *tl_al=bmk1822_lmem_alloc_tensor(bmk,(bmk1822_tensor_lmem_shape_t){1,1,4,4},FMT_I8,1);
  bmk1822_tensor_lmem_t *tl_ah=bmk1822_lmem_alloc_tensor(bmk,(bmk1822_tensor_lmem_shape_t){1,1,4,4},FMT_I8,1);
  bmk1822_tensor_lmem_t *tl_bl=bmk1822_lmem_alloc_tensor(bmk,(bmk1822_tensor_lmem_shape_t){1,1,4,4},FMT_I8,1);
  bmk1822_tensor_lmem_t *tl_bh=bmk1822_lmem_alloc_tensor(bmk,(bmk1822_tensor_lmem_shape_t){1,1,4,4},FMT_I8,1);
  bmk1822_tensor_lmem_t *tl_rl=bmk1822_lmem_alloc_tensor(bmk,(bmk1822_tensor_lmem_shape_t){1,1,4,4},FMT_I8,1);
  bmk1822_tensor_lmem_t *tl_rh=bmk1822_lmem_alloc_tensor(bmk,(bmk1822_tensor_lmem_shape_t){1,1,4,4},FMT_I8,1);
  if(!tl_al||!tl_ah||!tl_bl||!tl_bh||!tl_rl||!tl_rh){fprintf(stderr,"  alloc fail\n");goto out6;}

  bmk1822_tensor_tgmem_t tg={0,0,FMT_I8,{1,1,4,4},bmk1822_tensor_tgmem_default_stride((bmk1822_tensor_tgmem_shape_t){1,1,4,4},FMT_I8)};
  tg.start_address=0; bmk1822_tdma_g2l_tensor_copy(bmk,&(bmk1822_tdma_tg2l_tensor_copy_param_t){&tg,tl_al});
  tg.start_address=32; bmk1822_tdma_g2l_tensor_copy(bmk,&(bmk1822_tdma_tg2l_tensor_copy_param_t){&tg,tl_bl});

  bmk1822_tiu_element_wise_add_param_t p={
    .res_high=tl_rh,.res_low=tl_rl,.a_high=tl_ah,.a_low=tl_al,
    .b_is_const=0,.b_high=tl_bh,.b_low=tl_bl,.relu_enable=0,.rshift_bits=0,.layer_id=1,
  };
  if(!bmk1822_tiu_element_wise_add(bmk,&p)){
    fprintf(stderr,"  API call FAILED (assertion?)\n");
    printf("BMK_TEST|elem_add_hi_lo|FAIL|api_assert\n");
    goto out6;
  }
  tg.start_address=64;
  bmk1822_tdma_l2g_tensor_copy(bmk,&(bmk1822_tdma_l2tg_tensor_copy_param_t){tl_rl,&tg});

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

  int8_t *got=va+64;
  int ok=1;
  for(int i=0;i<16;i++){int exp_i=a[i]+b[i];if(got[i]!=exp_i){ok=0;break;}}
  fprintf(stderr,"  got=[%d,%d,...,%d] exp=[%d,%d,...,%d] %s\n",
    got[0],got[1],got[15],a[0]+b[0],a[1]+b[1],a[15]+b[15],ok?"OK":"FAIL");
  printf("BMK_TEST|elem_add_hi_lo|%s\n",ok?"OK":"FAIL");
  CVI_RT_MemFree(rt,dmabuf_mem);

out6:
  bmk1822_cleanup(bmk);
  CVI_RT_MemFree(rt,mem);
  CVI_RT_DeInit(rt);
  return 0;
}

/* ===== MAIN ===== */
int main(void) {
  fprintf(stderr,"\n========== bmk1822: 6 failing cvikernel ops test ==========\n");
  test_qdm_conv();
  test_matmul_bias();
  test_matmul_addres();
  test_mul_qdm();
  test_matmul_qdm();
  test_elem_add();
  fprintf(stderr,"\n========== DONE ==========\n\n");
  return 0;
}
