/* cvikernel all-operations 100-run benchmark.
   Every working cvikernel op with 100 runs + 3 warmup.
   Outputs machine-parseable BENCH| lines for HTML generation. */
#include "common/tpu_bench.h"
#include <math.h>

#define N_WARM 3
#define N_RUNS 100
#define NEURON_SZ 65536

static tpu_ctx g_ctx;
static cvk_context_t *cvk;

static double bench_one(void (*build)(void)) {
  struct timespec t0, t1;
  build();
  clock_gettime(CLOCK_MONOTONIC, &t0);
  CVI_RT_Submit(g_ctx.rt_khandle);
  clock_gettime(CLOCK_MONOTONIC, &t1);
  CVI_RT_MemInvld(g_ctx.rt_handle, g_ctx.neuron_mem);
  return (double)((t1.tv_sec-t0.tv_sec)*1000000000LL + (t1.tv_nsec-t0.tv_nsec))/1000.0;
}

static void print_stats(const char *name, double *us, int n) {
  double sum=0, sum2=0, tmin=1e9, tmax=0;
  for(int i=0;i<n;i++){ sum+=us[i]; sum2+=us[i]*us[i]; if(us[i]<tmin)tmin=us[i]; if(us[i]>tmax)tmax=us[i]; }
  double avg=sum/n, st=sqrt(sum2/n - avg*avg);
  fprintf(stderr,"  %-28s avg=%8.1f ± %5.1f  min=%8.1f  max=%8.1f us\n", name, avg, st, tmin, tmax);
  printf("BENCH|%s|%.1f|%.1f|%.1f|%.1f|OK\n", name, avg, st, tmin, tmax);
}

/* ===== [1] PT CONVOLUTION 3x3->1x1 ===== */
static int8_t c3_ifmap[9]={1,2,3,4,5,6,7,8,9};
static int8_t c3_wgt[9]={1,0,1, 0,0,0, 1,0,1}; /* corners, exp=20 */

static void build_conv3x3(void) {
  memcpy(g_ctx.neuron_vaddr, c3_ifmap, 9);
  memcpy(g_ctx.neuron_vaddr+16, c3_wgt, 9);
  CVI_RT_MemFlush(g_ctx.rt_handle, g_ctx.neuron_mem);
  cvk_tl_t *tl_if=cvk->ops->lmem_alloc_tensor(cvk,(cvk_tl_shape_t){1,1,3,3},CVK_FMT_I8,1);
  cvk_tl_t *tl_of=cvk->ops->lmem_alloc_tensor(cvk,(cvk_tl_shape_t){1,1,1,1},CVK_FMT_I8,1);
  cvk_tl_t *tl_w =cvk->ops->lmem_alloc_tensor(cvk,(cvk_tl_shape_t){1,1,3,3},CVK_FMT_I8,1);
  tl_w->stride.n=1; tl_w->cmprs_fmt=CVK_FMT_I8;
  cvk_tg_shape_t g={1,1,3,3}, g1={1,1,1,1};
  cvk->ops->tdma_g2l_tensor_copy(cvk,&(cvk_tdma_g2l_tensor_copy_param_t){
    &(cvk_tg_t){0,TPU_PA(&g_ctx,0),CVK_FMT_I8,g,cvk->ops->tg_default_stride(cvk,g,CVK_FMT_I8)},tl_if});
  cvk->ops->tdma_g2l_tensor_copy(cvk,&(cvk_tdma_g2l_tensor_copy_param_t){
    &(cvk_tg_t){0,TPU_PA(&g_ctx,16),CVK_FMT_I8,g,cvk->ops->tg_default_stride(cvk,g,CVK_FMT_I8)},tl_w});
  cvk->ops->tiu_pt_convolution(cvk,&(cvk_tiu_pt_convolution_param_t){
    .ofmap=tl_of,.ifmap=tl_if,.weight=tl_w,.bias=NULL,
    .ins_h=0,.ins_last_h=0,.ins_w=0,.ins_last_w=0,
    .pad_top=0,.pad_bottom=0,.pad_left=0,.pad_right=0,
    .stride_h=1,.stride_w=1,.dilation_h=1,.dilation_w=1,
    .relu_enable=0,.rshift_bits=0,.ps32_mode=0,.w_is_const=0,.layer_id=1,
  });
  cvk->ops->tdma_l2g_tensor_copy(cvk,&(cvk_tdma_l2g_tensor_copy_param_t){tl_of,
    &(cvk_tg_t){0,TPU_PA(&g_ctx,32),CVK_FMT_I8,g1,cvk->ops->tg_default_stride(cvk,g1,CVK_FMT_I8)}});
  cvk->ops->lmem_free_tensor(cvk,tl_w);
  cvk->ops->lmem_free_tensor(cvk,tl_of);
  cvk->ops->lmem_free_tensor(cvk,tl_if);
}

/* ===== [2] DEPTHWISE PT CONV 64x64->62x62 ===== */
#define D64 64
static int8_t d64_map[D64*D64], d64_wgt[9];

static void build_dw64(void) {
  memcpy(g_ctx.neuron_vaddr, d64_map, D64*D64);
  memcpy(g_ctx.neuron_vaddr+0x2000, d64_wgt, 9);
  CVI_RT_MemFlush(g_ctx.rt_handle, g_ctx.neuron_mem);
  cvk_tl_t *tl_if=cvk->ops->lmem_alloc_tensor(cvk,(cvk_tl_shape_t){1,1,D64,D64},CVK_FMT_I8,1);
  cvk_tl_t *tl_of=cvk->ops->lmem_alloc_tensor(cvk,(cvk_tl_shape_t){1,1,62,62},CVK_FMT_I8,1);
  cvk_tl_t *tl_w =cvk->ops->lmem_alloc_tensor(cvk,(cvk_tl_shape_t){1,1,3,3},CVK_FMT_I8,1);
  tl_w->stride.n=1; tl_w->cmprs_fmt=CVK_FMT_I8;
  cvk_tg_shape_t gi={1,1,D64,D64}, gw={1,1,3,3}, go={1,1,62,62};
  cvk->ops->tdma_g2l_tensor_copy(cvk,&(cvk_tdma_g2l_tensor_copy_param_t){
    &(cvk_tg_t){0,TPU_PA(&g_ctx,0),CVK_FMT_I8,gi,cvk->ops->tg_default_stride(cvk,gi,CVK_FMT_I8)},tl_if});
  cvk->ops->tdma_g2l_tensor_copy(cvk,&(cvk_tdma_g2l_tensor_copy_param_t){
    &(cvk_tg_t){0,TPU_PA(&g_ctx,0x2000),CVK_FMT_I8,gw,cvk->ops->tg_default_stride(cvk,gw,CVK_FMT_I8)},tl_w});
  cvk->ops->tiu_pt_convolution(cvk,&(cvk_tiu_pt_convolution_param_t){
    .ofmap=tl_of,.ifmap=tl_if,.weight=tl_w,.bias=NULL,
    .ins_h=0,.ins_last_h=0,.ins_w=0,.ins_last_w=0,
    .pad_top=0,.pad_bottom=0,.pad_left=0,.pad_right=0,
    .stride_h=1,.stride_w=1,.dilation_h=1,.dilation_w=1,
    .relu_enable=0,.rshift_bits=0,.ps32_mode=0,.w_is_const=0,.layer_id=1,
  });
  cvk->ops->tdma_l2g_tensor_copy(cvk,&(cvk_tdma_l2g_tensor_copy_param_t){tl_of,
    &(cvk_tg_t){0,TPU_PA(&g_ctx,0x4000),CVK_FMT_I8,go,cvk->ops->tg_default_stride(cvk,go,CVK_FMT_I8)}});
  cvk->ops->lmem_free_tensor(cvk,tl_w);
  cvk->ops->lmem_free_tensor(cvk,tl_of);
  cvk->ops->lmem_free_tensor(cvk,tl_if);
}

/* ===== [3] MATMUL 2x2 x 2x2 ===== */
static int8_t m2l[4]={1,2,3,4}, m2r[4]={1,2,3,4};

static void build_mat2(void) {
  int M=2,K=2,N=2;
  memcpy(g_ctx.neuron_vaddr,m2l,4);
  memcpy(g_ctx.neuron_vaddr+64,m2r,4);
  CVI_RT_MemFlush(g_ctx.rt_handle,g_ctx.neuron_mem);
  cvk_ml_t *ml_l=cvk->ops->lmem_alloc_matrix(cvk,cvk->ops->ml_default_shape(cvk,M,K,CVK_FMT_I8),CVK_FMT_I8,1);
  cvk_ml_t *ml_r=cvk->ops->lmem_alloc_matrix(cvk,cvk->ops->ml_default_shape(cvk,K,N,CVK_FMT_I8),CVK_FMT_I8,1);
  cvk_ml_t *ml_o=cvk->ops->lmem_alloc_matrix(cvk,cvk->ops->ml_default_shape(cvk,M,N,CVK_FMT_I8),CVK_FMT_I8,1);
  cvk->ops->tdma_g2l_matrix_copy(cvk,&(cvk_tdma_g2l_matrix_copy_param_t){
    &(cvk_mg_t){0,TPU_PA(&g_ctx,0),CVK_FMT_I8,{M,K},{K}},ml_l});
  cvk->ops->tdma_g2l_matrix_copy(cvk,&(cvk_tdma_g2l_matrix_copy_param_t){
    &(cvk_mg_t){0,TPU_PA(&g_ctx,64),CVK_FMT_I8,{K,N},{N}},ml_r});
  cvk->ops->tiu_matrix_multiplication(cvk,&(cvk_tiu_matrix_multiplication_param_t){
    .res=ml_o,.left=ml_l,.right=ml_r,.bias=NULL,
    .lshift_bits=0,.rshift_bits=0,.res_is_int8=1,.relu_enable=0,.add_result=0,.ps32_mode=0,
  });
  cvk->ops->tdma_l2g_matrix_copy(cvk,&(cvk_tdma_l2g_matrix_copy_param_t){
    ml_o,&(cvk_mg_t){0,TPU_PA(&g_ctx,128),CVK_FMT_I8,{M,N},{N}}});
  cvk->ops->lmem_free_matrix(cvk,ml_o);
  cvk->ops->lmem_free_matrix(cvk,ml_r);
  cvk->ops->lmem_free_matrix(cvk,ml_l);
}

/* ===== [4] MATMUL 16x16 x 16x16 ===== */
static int8_t m16l[256], m16r[256];

static void build_mat16(void) {
  int M=16,K=16,N=16;
  memcpy(g_ctx.neuron_vaddr,m16l,256);
  memcpy(g_ctx.neuron_vaddr+512,m16r,256);
  CVI_RT_MemFlush(g_ctx.rt_handle,g_ctx.neuron_mem);
  cvk_ml_t *ml_l=cvk->ops->lmem_alloc_matrix(cvk,cvk->ops->ml_default_shape(cvk,M,K,CVK_FMT_I8),CVK_FMT_I8,1);
  cvk_ml_t *ml_r=cvk->ops->lmem_alloc_matrix(cvk,cvk->ops->ml_default_shape(cvk,K,N,CVK_FMT_I8),CVK_FMT_I8,1);
  cvk_ml_t *ml_o=cvk->ops->lmem_alloc_matrix(cvk,cvk->ops->ml_default_shape(cvk,M,N,CVK_FMT_I8),CVK_FMT_I8,1);
  cvk->ops->tdma_g2l_matrix_copy(cvk,&(cvk_tdma_g2l_matrix_copy_param_t){
    &(cvk_mg_t){0,TPU_PA(&g_ctx,0),CVK_FMT_I8,{M,K},{K}},ml_l});
  cvk->ops->tdma_g2l_matrix_copy(cvk,&(cvk_tdma_g2l_matrix_copy_param_t){
    &(cvk_mg_t){0,TPU_PA(&g_ctx,512),CVK_FMT_I8,{K,N},{N}},ml_r});
  cvk->ops->tiu_matrix_multiplication(cvk,&(cvk_tiu_matrix_multiplication_param_t){
    .res=ml_o,.left=ml_l,.right=ml_r,.bias=NULL,
    .lshift_bits=0,.rshift_bits=0,.res_is_int8=1,.relu_enable=0,.add_result=0,.ps32_mode=0,
  });
  cvk->ops->tdma_l2g_matrix_copy(cvk,&(cvk_tdma_l2g_matrix_copy_param_t){
    ml_o,&(cvk_mg_t){0,TPU_PA(&g_ctx,1024),CVK_FMT_I8,{M,N},{N}}});
  cvk->ops->lmem_free_matrix(cvk,ml_o);
  cvk->ops->lmem_free_matrix(cvk,ml_r);
  cvk->ops->lmem_free_matrix(cvk,ml_l);
}

/* ===== [5] ADD CONSTANT (16-bit a required) ===== */
static int8_t add_in[256], add_exp[256];
static void build_addc(void) {
  memcpy(g_ctx.neuron_vaddr, add_in, 256);
  memset(g_ctx.neuron_vaddr+256, 0, 256); /* a_high = 0 */
  CVI_RT_MemFlush(g_ctx.rt_handle, g_ctx.neuron_mem);
  cvk_tl_shape_t s={1,1,16,16};
  cvk_tl_t *tl_al=cvk->ops->lmem_alloc_tensor(cvk,s,CVK_FMT_I8,1);
  cvk_tl_t *tl_ah=cvk->ops->lmem_alloc_tensor(cvk,s,CVK_FMT_I8,1);
  cvk_tl_t *tl_rl=cvk->ops->lmem_alloc_tensor(cvk,s,CVK_FMT_I8,1);
  cvk_tl_t *tl_rh=cvk->ops->lmem_alloc_tensor(cvk,s,CVK_FMT_I8,1);
  cvk_tg_shape_t gs={1,1,16,16};
  cvk_tg_stride_t st=cvk->ops->tg_default_stride(cvk,gs,CVK_FMT_I8);
  cvk->ops->tdma_g2l_tensor_copy(cvk,&(cvk_tdma_g2l_tensor_copy_param_t){
    &(cvk_tg_t){0,TPU_PA(&g_ctx,0),CVK_FMT_I8,gs,st},tl_al});
  cvk->ops->tdma_g2l_tensor_copy(cvk,&(cvk_tdma_g2l_tensor_copy_param_t){
    &(cvk_tg_t){0,TPU_PA(&g_ctx,256),CVK_FMT_I8,gs,st},tl_ah});
  cvk->ops->tiu_add(cvk,&(cvk_tiu_add_param_t){
    .res_high=tl_rh,.res_low=tl_rl,.a_high=tl_ah,.a_low=tl_al,
    .b_is_const=1,.b_const.val=5,.b_const.is_signed=1,
    .rshift_bits=0,.relu_enable=0,.layer_id=1,
  });
  cvk->ops->tdma_l2g_tensor_copy(cvk,&(cvk_tdma_l2g_tensor_copy_param_t){tl_rl,
    &(cvk_tg_t){0,TPU_PA(&g_ctx,512),CVK_FMT_I8,gs,st}});
  cvk->ops->lmem_free_tensor(cvk,tl_rh);
  cvk->ops->lmem_free_tensor(cvk,tl_rl);
  cvk->ops->lmem_free_tensor(cvk,tl_ah);
  cvk->ops->lmem_free_tensor(cvk,tl_al);
}

/* ===== [6] MUL CONSTANT (8-bit a, no hi/lo needed) ===== */
static int8_t mul_in[256], mul_exp[256];
static void build_mulc(void) {
  memcpy(g_ctx.neuron_vaddr, mul_in, 256);
  CVI_RT_MemFlush(g_ctx.rt_handle, g_ctx.neuron_mem);
  cvk_tl_shape_t s={1,1,16,16};
  cvk_tl_t *tl_a=cvk->ops->lmem_alloc_tensor(cvk,s,CVK_FMT_I8,1);
  cvk_tl_t *tl_r=cvk->ops->lmem_alloc_tensor(cvk,s,CVK_FMT_I8,1);
  cvk_tg_shape_t gs={1,1,16,16};
  cvk_tg_stride_t st=cvk->ops->tg_default_stride(cvk,gs,CVK_FMT_I8);
  cvk->ops->tdma_g2l_tensor_copy(cvk,&(cvk_tdma_g2l_tensor_copy_param_t){
    &(cvk_tg_t){0,TPU_PA(&g_ctx,0),CVK_FMT_I8,gs,st},tl_a});
  cvk->ops->tiu_mul(cvk,&(cvk_tiu_mul_param_t){
    .res_high=NULL,.res_low=tl_r,.a=tl_a,
    .b_is_const=1,.b_const.val=2,.b_const.is_signed=1,
    .rshift_bits=0,.relu_enable=0,.layer_id=1,
  });
  cvk->ops->tdma_l2g_tensor_copy(cvk,&(cvk_tdma_l2g_tensor_copy_param_t){tl_r,
    &(cvk_tg_t){0,TPU_PA(&g_ctx,256),CVK_FMT_I8,gs,st}});
  cvk->ops->lmem_free_tensor(cvk,tl_r);
  cvk->ops->lmem_free_tensor(cvk,tl_a);
}

/* ===== [7] MUL TENSOR (element-wise) ===== */
static int8_t mult_b[256], mult_exp[256];
static void build_mult(void) {
  memcpy(g_ctx.neuron_vaddr, mul_in, 256);
  memcpy(g_ctx.neuron_vaddr+256, mult_b, 256);
  CVI_RT_MemFlush(g_ctx.rt_handle, g_ctx.neuron_mem);
  cvk_tl_shape_t s={1,1,16,16};
  cvk_tl_t *tl_a=cvk->ops->lmem_alloc_tensor(cvk,s,CVK_FMT_I8,1);
  cvk_tl_t *tl_b=cvk->ops->lmem_alloc_tensor(cvk,s,CVK_FMT_I8,1);
  cvk_tg_shape_t gs={1,1,16,16};
  cvk_tg_stride_t st=cvk->ops->tg_default_stride(cvk,gs,CVK_FMT_I8);
  cvk->ops->tdma_g2l_tensor_copy(cvk,&(cvk_tdma_g2l_tensor_copy_param_t){
    &(cvk_tg_t){0,TPU_PA(&g_ctx,0),CVK_FMT_I8,gs,st},tl_a});
  cvk->ops->tdma_g2l_tensor_copy(cvk,&(cvk_tdma_g2l_tensor_copy_param_t){
    &(cvk_tg_t){0,TPU_PA(&g_ctx,256),CVK_FMT_I8,gs,st},tl_b});
  cvk->ops->tiu_mul(cvk,&(cvk_tiu_mul_param_t){
    .res_high=NULL,.res_low=tl_a,.a=tl_a,
    .b_is_const=0,.b=tl_b,
    .rshift_bits=0,.relu_enable=0,.layer_id=1,
  });
  cvk->ops->tdma_l2g_tensor_copy(cvk,&(cvk_tdma_l2g_tensor_copy_param_t){tl_a,
    &(cvk_tg_t){0,TPU_PA(&g_ctx,512),CVK_FMT_I8,gs,st}});
  cvk->ops->lmem_free_tensor(cvk,tl_b);
  cvk->ops->lmem_free_tensor(cvk,tl_a);
}

/* ===== [8] MAX CONSTANT ===== */
static int8_t max_in[256], max_exp[256];
static void build_maxc(void) {
  memcpy(g_ctx.neuron_vaddr, max_in, 256);
  CVI_RT_MemFlush(g_ctx.rt_handle, g_ctx.neuron_mem);
  cvk_tl_shape_t s={1,1,16,16};
  cvk_tl_t *tl_a=cvk->ops->lmem_alloc_tensor(cvk,s,CVK_FMT_I8,1);
  cvk_tl_t *tl_r=cvk->ops->lmem_alloc_tensor(cvk,s,CVK_FMT_I8,1);
  cvk_tg_shape_t gs={1,1,16,16};
  cvk_tg_stride_t st=cvk->ops->tg_default_stride(cvk,gs,CVK_FMT_I8);
  cvk->ops->tdma_g2l_tensor_copy(cvk,&(cvk_tdma_g2l_tensor_copy_param_t){
    &(cvk_tg_t){0,TPU_PA(&g_ctx,0),CVK_FMT_I8,gs,st},tl_a});
  cvk->ops->tiu_max(cvk,&(cvk_tiu_max_param_t){
    .max=tl_r,.a=tl_a,
    .b_is_const=1,.b_const.val=0,.b_const.is_signed=1,.layer_id=1,
  });
  cvk->ops->tdma_l2g_tensor_copy(cvk,&(cvk_tdma_l2g_tensor_copy_param_t){tl_r,
    &(cvk_tg_t){0,TPU_PA(&g_ctx,256),CVK_FMT_I8,gs,st}});
  cvk->ops->lmem_free_tensor(cvk,tl_r);
  cvk->ops->lmem_free_tensor(cvk,tl_a);
}

/* ===== [9] MIN CONSTANT ===== */
static int8_t min_in[256], min_exp[256];
static void build_minc(void) {
  memcpy(g_ctx.neuron_vaddr, min_in, 256);
  CVI_RT_MemFlush(g_ctx.rt_handle, g_ctx.neuron_mem);
  cvk_tl_shape_t s={1,1,16,16};
  cvk_tl_t *tl_a=cvk->ops->lmem_alloc_tensor(cvk,s,CVK_FMT_I8,1);
  cvk_tl_t *tl_r=cvk->ops->lmem_alloc_tensor(cvk,s,CVK_FMT_I8,1);
  cvk_tg_shape_t gs={1,1,16,16};
  cvk_tg_stride_t st=cvk->ops->tg_default_stride(cvk,gs,CVK_FMT_I8);
  cvk->ops->tdma_g2l_tensor_copy(cvk,&(cvk_tdma_g2l_tensor_copy_param_t){
    &(cvk_tg_t){0,TPU_PA(&g_ctx,0),CVK_FMT_I8,gs,st},tl_a});
  cvk->ops->tiu_min(cvk,&(cvk_tiu_min_param_t){
    .min=tl_r,.a=tl_a,
    .b_is_const=1,.b_const.val=0,.b_const.is_signed=1,.layer_id=1,
  });
  cvk->ops->tdma_l2g_tensor_copy(cvk,&(cvk_tdma_l2g_tensor_copy_param_t){tl_r,
    &(cvk_tg_t){0,TPU_PA(&g_ctx,256),CVK_FMT_I8,gs,st}});
  cvk->ops->lmem_free_tensor(cvk,tl_r);
  cvk->ops->lmem_free_tensor(cvk,tl_a);
}

/* ===== [10] COPY ===== */
static int8_t cpy_in[256];
static void build_copy(void) {
  memcpy(g_ctx.neuron_vaddr, cpy_in, 256);
  CVI_RT_MemFlush(g_ctx.rt_handle, g_ctx.neuron_mem);
  cvk_tl_shape_t s={1,1,16,16};
  cvk_tl_t *tl_s=cvk->ops->lmem_alloc_tensor(cvk,s,CVK_FMT_I8,1);
  cvk_tl_t *tl_d=cvk->ops->lmem_alloc_tensor(cvk,s,CVK_FMT_I8,1);
  cvk_tg_shape_t gs={1,1,16,16};
  cvk_tg_stride_t st=cvk->ops->tg_default_stride(cvk,gs,CVK_FMT_I8);
  cvk->ops->tdma_g2l_tensor_copy(cvk,&(cvk_tdma_g2l_tensor_copy_param_t){
    &(cvk_tg_t){0,TPU_PA(&g_ctx,0),CVK_FMT_I8,gs,st},tl_s});
  cvk->ops->tiu_copy(cvk,&(cvk_tiu_copy_param_t){.dst=tl_d,.src=tl_s,.layer_id=1});
  cvk->ops->tdma_l2g_tensor_copy(cvk,&(cvk_tdma_l2g_tensor_copy_param_t){tl_d,
    &(cvk_tg_t){0,TPU_PA(&g_ctx,256),CVK_FMT_I8,gs,st}});
  cvk->ops->lmem_free_tensor(cvk,tl_d);
  cvk->ops->lmem_free_tensor(cvk,tl_s);
}

/* ===== [11] AND INT8 ===== */
static int8_t and_a[256], and_b[256], and_exp[256];
static void build_and(void) {
  memcpy(g_ctx.neuron_vaddr, and_a, 256);
  memcpy(g_ctx.neuron_vaddr+256, and_b, 256);
  CVI_RT_MemFlush(g_ctx.rt_handle, g_ctx.neuron_mem);
  cvk_tl_shape_t s={1,1,16,16};
  cvk_tl_t *tl_a=cvk->ops->lmem_alloc_tensor(cvk,s,CVK_FMT_I8,1);
  cvk_tl_t *tl_b=cvk->ops->lmem_alloc_tensor(cvk,s,CVK_FMT_I8,1);
  cvk_tl_t *tl_r=cvk->ops->lmem_alloc_tensor(cvk,s,CVK_FMT_I8,1);
  cvk_tg_shape_t gs={1,1,16,16};
  cvk_tg_stride_t st=cvk->ops->tg_default_stride(cvk,gs,CVK_FMT_I8);
  cvk->ops->tdma_g2l_tensor_copy(cvk,&(cvk_tdma_g2l_tensor_copy_param_t){
    &(cvk_tg_t){0,TPU_PA(&g_ctx,0),CVK_FMT_I8,gs,st},tl_a});
  cvk->ops->tdma_g2l_tensor_copy(cvk,&(cvk_tdma_g2l_tensor_copy_param_t){
    &(cvk_tg_t){0,TPU_PA(&g_ctx,256),CVK_FMT_I8,gs,st},tl_b});
  cvk->ops->tiu_and_int8(cvk,&(cvk_tiu_and_int8_param_t){tl_r,tl_a,tl_b});
  cvk->ops->tdma_l2g_tensor_copy(cvk,&(cvk_tdma_l2g_tensor_copy_param_t){tl_r,
    &(cvk_tg_t){0,TPU_PA(&g_ctx,512),CVK_FMT_I8,gs,st}});
  cvk->ops->lmem_free_tensor(cvk,tl_r);
  cvk->ops->lmem_free_tensor(cvk,tl_b);
  cvk->ops->lmem_free_tensor(cvk,tl_a);
}

/* ===== [12] OR INT8 ===== */
static int8_t or_a[256], or_b[256], or_exp[256];
static void build_or(void) {
  memcpy(g_ctx.neuron_vaddr, or_a, 256);
  memcpy(g_ctx.neuron_vaddr+256, or_b, 256);
  CVI_RT_MemFlush(g_ctx.rt_handle, g_ctx.neuron_mem);
  cvk_tl_shape_t s={1,1,16,16};
  cvk_tl_t *tl_a=cvk->ops->lmem_alloc_tensor(cvk,s,CVK_FMT_I8,1);
  cvk_tl_t *tl_b=cvk->ops->lmem_alloc_tensor(cvk,s,CVK_FMT_I8,1);
  cvk_tl_t *tl_r=cvk->ops->lmem_alloc_tensor(cvk,s,CVK_FMT_I8,1);
  cvk_tg_shape_t gs={1,1,16,16};
  cvk_tg_stride_t st=cvk->ops->tg_default_stride(cvk,gs,CVK_FMT_I8);
  cvk->ops->tdma_g2l_tensor_copy(cvk,&(cvk_tdma_g2l_tensor_copy_param_t){
    &(cvk_tg_t){0,TPU_PA(&g_ctx,0),CVK_FMT_I8,gs,st},tl_a});
  cvk->ops->tdma_g2l_tensor_copy(cvk,&(cvk_tdma_g2l_tensor_copy_param_t){
    &(cvk_tg_t){0,TPU_PA(&g_ctx,256),CVK_FMT_I8,gs,st},tl_b});
  cvk->ops->tiu_or_int8(cvk,&(cvk_tiu_or_int8_param_t){tl_r,tl_a,tl_b});
  cvk->ops->tdma_l2g_tensor_copy(cvk,&(cvk_tdma_l2g_tensor_copy_param_t){tl_r,
    &(cvk_tg_t){0,TPU_PA(&g_ctx,512),CVK_FMT_I8,gs,st}});
  cvk->ops->lmem_free_tensor(cvk,tl_r);
  cvk->ops->lmem_free_tensor(cvk,tl_b);
  cvk->ops->lmem_free_tensor(cvk,tl_a);
}

/* ===== [13] XOR INT8 ===== */
static int8_t xor_a[256], xor_b[256], xor_exp[256];
static void build_xor(void) {
  memcpy(g_ctx.neuron_vaddr, xor_a, 256);
  memcpy(g_ctx.neuron_vaddr+256, xor_b, 256);
  CVI_RT_MemFlush(g_ctx.rt_handle, g_ctx.neuron_mem);
  cvk_tl_shape_t s={1,1,16,16};
  cvk_tl_t *tl_a=cvk->ops->lmem_alloc_tensor(cvk,s,CVK_FMT_I8,1);
  cvk_tl_t *tl_b=cvk->ops->lmem_alloc_tensor(cvk,s,CVK_FMT_I8,1);
  cvk_tl_t *tl_r=cvk->ops->lmem_alloc_tensor(cvk,s,CVK_FMT_I8,1);
  cvk_tg_shape_t gs={1,1,16,16};
  cvk_tg_stride_t st=cvk->ops->tg_default_stride(cvk,gs,CVK_FMT_I8);
  cvk->ops->tdma_g2l_tensor_copy(cvk,&(cvk_tdma_g2l_tensor_copy_param_t){
    &(cvk_tg_t){0,TPU_PA(&g_ctx,0),CVK_FMT_I8,gs,st},tl_a});
  cvk->ops->tdma_g2l_tensor_copy(cvk,&(cvk_tdma_g2l_tensor_copy_param_t){
    &(cvk_tg_t){0,TPU_PA(&g_ctx,256),CVK_FMT_I8,gs,st},tl_b});
  cvk->ops->tiu_xor_int8(cvk,&(cvk_tiu_xor_int8_param_t){tl_r,tl_a,tl_b});
  cvk->ops->tdma_l2g_tensor_copy(cvk,&(cvk_tdma_l2g_tensor_copy_param_t){tl_r,
    &(cvk_tg_t){0,TPU_PA(&g_ctx,512),CVK_FMT_I8,gs,st}});
  cvk->ops->lmem_free_tensor(cvk,tl_r);
  cvk->ops->lmem_free_tensor(cvk,tl_b);
  cvk->ops->lmem_free_tensor(cvk,tl_a);
}

/* ===== [14] GE CONSTANT ===== */
static int8_t ge_in[256], ge_exp[256];
static void build_ge(void) {
  memcpy(g_ctx.neuron_vaddr, ge_in, 256);
  CVI_RT_MemFlush(g_ctx.rt_handle, g_ctx.neuron_mem);
  cvk_tl_shape_t s={1,1,16,16};
  cvk_tl_t *tl_a=cvk->ops->lmem_alloc_tensor(cvk,s,CVK_FMT_I8,1);
  cvk_tl_t *tl_r=cvk->ops->lmem_alloc_tensor(cvk,s,CVK_FMT_I8,1);
  cvk_tg_shape_t gs={1,1,16,16};
  cvk_tg_stride_t st=cvk->ops->tg_default_stride(cvk,gs,CVK_FMT_I8);
  cvk->ops->tdma_g2l_tensor_copy(cvk,&(cvk_tdma_g2l_tensor_copy_param_t){
    &(cvk_tg_t){0,TPU_PA(&g_ctx,0),CVK_FMT_I8,gs,st},tl_a});
  cvk->ops->tiu_ge(cvk,&(cvk_tiu_ge_param_t){
    .ge=tl_r,.a=tl_a,
    .b_is_const=1,.b_const.val=0,.b_const.is_signed=1,.layer_id=1,
  });
  cvk->ops->tdma_l2g_tensor_copy(cvk,&(cvk_tdma_l2g_tensor_copy_param_t){tl_r,
    &(cvk_tg_t){0,TPU_PA(&g_ctx,256),CVK_FMT_I8,gs,st}});
  cvk->ops->lmem_free_tensor(cvk,tl_r);
  cvk->ops->lmem_free_tensor(cvk,tl_a);
}

/* ===== [15] ARITHMETIC SHIFT (16-bit, all hi/lo required) ===== */
static int8_t ash_a[16], ash_bits[16], ash_exp[16];
static void build_ash(void) {
  memcpy(g_ctx.neuron_vaddr, ash_a, 16);
  memset(g_ctx.neuron_vaddr+16, 0, 16); /* a_hi = 0 */
  memcpy(g_ctx.neuron_vaddr+32, ash_bits, 16); /* bits lo = -1 */
  memset(g_ctx.neuron_vaddr+48, 0, 16); /* bits hi */
  CVI_RT_MemFlush(g_ctx.rt_handle, g_ctx.neuron_mem);
  cvk_tl_shape_t s={1,1,1,16};
  cvk_tl_t *tl_al=cvk->ops->lmem_alloc_tensor(cvk,s,CVK_FMT_I8,1);
  cvk_tl_t *tl_ah=cvk->ops->lmem_alloc_tensor(cvk,s,CVK_FMT_I8,1);
  cvk_tl_t *tl_bl=cvk->ops->lmem_alloc_tensor(cvk,s,CVK_FMT_I8,1);
  cvk_tl_t *tl_bh=cvk->ops->lmem_alloc_tensor(cvk,s,CVK_FMT_I8,1);
  cvk_tl_t *tl_rl=cvk->ops->lmem_alloc_tensor(cvk,s,CVK_FMT_I8,1);
  cvk_tl_t *tl_rh=cvk->ops->lmem_alloc_tensor(cvk,s,CVK_FMT_I8,1);
  cvk_tg_shape_t gs={1,1,1,16};
  cvk_tg_stride_t st=cvk->ops->tg_default_stride(cvk,gs,CVK_FMT_I8);
  cvk->ops->tdma_g2l_tensor_copy(cvk,&(cvk_tdma_g2l_tensor_copy_param_t){
    &(cvk_tg_t){0,TPU_PA(&g_ctx,0),CVK_FMT_I8,gs,st},tl_al});
  cvk->ops->tdma_g2l_tensor_copy(cvk,&(cvk_tdma_g2l_tensor_copy_param_t){
    &(cvk_tg_t){0,TPU_PA(&g_ctx,16),CVK_FMT_I8,gs,st},tl_ah});
  cvk->ops->tdma_g2l_tensor_copy(cvk,&(cvk_tdma_g2l_tensor_copy_param_t){
    &(cvk_tg_t){0,TPU_PA(&g_ctx,32),CVK_FMT_I8,gs,st},tl_bl});
  cvk->ops->tdma_g2l_tensor_copy(cvk,&(cvk_tdma_g2l_tensor_copy_param_t){
    &(cvk_tg_t){0,TPU_PA(&g_ctx,48),CVK_FMT_I8,gs,st},tl_bh});
  cvk->ops->tiu_arith_shift(cvk,&(cvk_tiu_arith_shift_param_t){
    .res_high=tl_rh,.res_low=tl_rl,.a_high=tl_ah,.a_low=tl_al,
    .bits=tl_bl,.layer_id=1,
  });
  cvk->ops->tdma_l2g_tensor_copy(cvk,&(cvk_tdma_l2g_tensor_copy_param_t){tl_rl,
    &(cvk_tg_t){0,TPU_PA(&g_ctx,64),CVK_FMT_I8,gs,st}});
  cvk->ops->lmem_free_tensor(cvk,tl_rh);
  cvk->ops->lmem_free_tensor(cvk,tl_rl);
  cvk->ops->lmem_free_tensor(cvk,tl_bh);
  cvk->ops->lmem_free_tensor(cvk,tl_bl);
  cvk->ops->lmem_free_tensor(cvk,tl_ah);
  cvk->ops->lmem_free_tensor(cvk,tl_al);
}

/* ===== [16] MAX POOLING 8x8->4x4 ===== */
static int8_t pool_in[64];
static void build_maxpool(void) {
  memcpy(g_ctx.neuron_vaddr, pool_in, 64);
  CVI_RT_MemFlush(g_ctx.rt_handle, g_ctx.neuron_mem);
  cvk_tl_t *tl_i=cvk->ops->lmem_alloc_tensor(cvk,(cvk_tl_shape_t){1,1,8,8},CVK_FMT_I8,1);
  cvk_tl_t *tl_o=cvk->ops->lmem_alloc_tensor(cvk,(cvk_tl_shape_t){1,1,4,4},CVK_FMT_I8,1);
  cvk_tg_shape_t gi={1,1,8,8}, go={1,1,4,4};
  cvk->ops->tdma_g2l_tensor_copy(cvk,&(cvk_tdma_g2l_tensor_copy_param_t){
    &(cvk_tg_t){0,TPU_PA(&g_ctx,0),CVK_FMT_I8,gi,cvk->ops->tg_default_stride(cvk,gi,CVK_FMT_I8)},tl_i});
  cvk->ops->tiu_max_pooling(cvk,&(cvk_tiu_max_pooling_param_t){
    .ofmap=tl_o,.ifmap=tl_i,.pad_top=0,.pad_bottom=0,.pad_left=0,.pad_right=0,
    .stride_h=2,.stride_w=2,.kh=2,.kw=2,.layer_id=1,
  });
  cvk->ops->tdma_l2g_tensor_copy(cvk,&(cvk_tdma_l2g_tensor_copy_param_t){tl_o,
    &(cvk_tg_t){0,TPU_PA(&g_ctx,64),CVK_FMT_I8,go,cvk->ops->tg_default_stride(cvk,go,CVK_FMT_I8)}});
  cvk->ops->lmem_free_tensor(cvk,tl_o);
  cvk->ops->lmem_free_tensor(cvk,tl_i);
}

/* ===== [17] AVG POOLING 8x8->4x4 ===== */
static void build_avgpool(void) {
  memcpy(g_ctx.neuron_vaddr, pool_in, 64);
  CVI_RT_MemFlush(g_ctx.rt_handle, g_ctx.neuron_mem);
  cvk_tl_t *tl_i=cvk->ops->lmem_alloc_tensor(cvk,(cvk_tl_shape_t){1,1,8,8},CVK_FMT_I8,1);
  cvk_tl_t *tl_o=cvk->ops->lmem_alloc_tensor(cvk,(cvk_tl_shape_t){1,1,4,4},CVK_FMT_I8,1);
  cvk_tg_shape_t gi={1,1,8,8}, go={1,1,4,4};
  cvk->ops->tdma_g2l_tensor_copy(cvk,&(cvk_tdma_g2l_tensor_copy_param_t){
    &(cvk_tg_t){0,TPU_PA(&g_ctx,0),CVK_FMT_I8,gi,cvk->ops->tg_default_stride(cvk,gi,CVK_FMT_I8)},tl_i});
  cvk->ops->tiu_average_pooling(cvk,&(cvk_tiu_average_pooling_param_t){
    .ofmap=tl_o,.ifmap=tl_i,.pad_top=0,.pad_bottom=0,.pad_left=0,.pad_right=0,
    .stride_h=2,.stride_w=2,.kh=2,.kw=2,.layer_id=1,
  });
  cvk->ops->tdma_l2g_tensor_copy(cvk,&(cvk_tdma_l2g_tensor_copy_param_t){tl_o,
    &(cvk_tg_t){0,TPU_PA(&g_ctx,64),CVK_FMT_I8,go,cvk->ops->tg_default_stride(cvk,go,CVK_FMT_I8)}});
  cvk->ops->lmem_free_tensor(cvk,tl_o);
  cvk->ops->lmem_free_tensor(cvk,tl_i);
}

/* ===== [18] MIN POOLING 8x8->4x4 ===== */
static void build_minpool(void) {
  memcpy(g_ctx.neuron_vaddr, pool_in, 64);
  CVI_RT_MemFlush(g_ctx.rt_handle, g_ctx.neuron_mem);
  cvk_tl_t *tl_i=cvk->ops->lmem_alloc_tensor(cvk,(cvk_tl_shape_t){1,1,8,8},CVK_FMT_I8,1);
  cvk_tl_t *tl_o=cvk->ops->lmem_alloc_tensor(cvk,(cvk_tl_shape_t){1,1,4,4},CVK_FMT_I8,1);
  cvk_tg_shape_t gi={1,1,8,8}, go={1,1,4,4};
  cvk->ops->tdma_g2l_tensor_copy(cvk,&(cvk_tdma_g2l_tensor_copy_param_t){
    &(cvk_tg_t){0,TPU_PA(&g_ctx,0),CVK_FMT_I8,gi,cvk->ops->tg_default_stride(cvk,gi,CVK_FMT_I8)},tl_i});
  cvk->ops->tiu_min_pooling(cvk,&(cvk_tiu_min_pooling_param_t){
    .ofmap=tl_o,.ifmap=tl_i,.pad_top=0,.pad_bottom=0,.pad_left=0,.pad_right=0,
    .stride_h=2,.stride_w=2,.kh=2,.kw=2,.layer_id=1,
  });
  cvk->ops->tdma_l2g_tensor_copy(cvk,&(cvk_tdma_l2g_tensor_copy_param_t){tl_o,
    &(cvk_tg_t){0,TPU_PA(&g_ctx,64),CVK_FMT_I8,go,cvk->ops->tg_default_stride(cvk,go,CVK_FMT_I8)}});
  cvk->ops->lmem_free_tensor(cvk,tl_o);
  cvk->ops->lmem_free_tensor(cvk,tl_i);
}

/* ===== MAIN ===== */
int main(void) {
  fprintf(stderr,"\n========== cvikernel 100-run all-operations benchmark ==========\n\n");
  if(tpu_init(&g_ctx, NEURON_SZ)!=0) return 1;
  cvk=g_ctx.cvk_ctx;

  /* init data */
  for(int i=0;i<D64*D64;i++) d64_map[i]=(int8_t)(i&3);
  for(int i=0;i<9;i++) d64_wgt[i]=(int8_t)((i==4)?1:0);
  for(int i=0;i<256;i++){m16l[i]=(int8_t)(i&3);m16r[i]=(int8_t)((i+1)&3);}
  for(int i=0;i<256;i++){add_in[i]=(int8_t)(i-128);add_exp[i]=(int8_t)(add_in[i]+5);}
  for(int i=0;i<256;i++){mul_in[i]=(int8_t)(i-128);mul_exp[i]=(int8_t)(mul_in[i]*2);}
  for(int i=0;i<256;i++){mult_b[i]=(int8_t)((i&7)-3);mult_exp[i]=(int8_t)(mul_in[i]*mult_b[i]);}
  for(int i=0;i<256;i++){max_in[i]=(int8_t)(i-128);max_exp[i]=max_in[i]>0?max_in[i]:0;}
  for(int i=0;i<256;i++){min_in[i]=(int8_t)(i-128);min_exp[i]=min_in[i]<0?min_in[i]:0;}
  for(int i=0;i<256;i++)cpy_in[i]=(int8_t)(i&0x7f);
  for(int i=0;i<256;i++){and_a[i]=(int8_t)(i|0x55);and_b[i]=(int8_t)(i|0xAA);and_exp[i]=and_a[i]&and_b[i];}
  for(int i=0;i<256;i++){or_a[i]=(int8_t)(i|0x55);or_b[i]=(int8_t)(i|0xAA);or_exp[i]=or_a[i]|or_b[i];}
  for(int i=0;i<256;i++){xor_a[i]=(int8_t)(i|0x55);xor_b[i]=(int8_t)(i|0xAA);xor_exp[i]=xor_a[i]^xor_b[i];}
  for(int i=0;i<256;i++){ge_in[i]=(int8_t)(i-128);ge_exp[i]=ge_in[i]>=0?(int8_t)1:(int8_t)0;}
  for(int i=0;i<16;i++){ash_a[i]=(int8_t)((i&0x3f)-32);ash_exp[i]=(int8_t)(ash_a[i]>>1);}
  for(int i=0;i<16;i++)ash_bits[i]=(int8_t)(-1); /* Right shift by 1 */
  for(int i=0;i<64;i++)pool_in[i]=(int8_t)(i+1);

  double us[N_RUNS];
#define DO_BENCH(name, fn) do { \
  for(int r=0;r<N_WARM+N_RUNS;r++){ \
    double t=bench_one(fn); \
    if(r>=N_WARM) us[r-N_WARM]=t; \
  } \
  print_stats(name, us, N_RUNS); \
} while(0)

  DO_BENCH("01_pt_conv_3x3",        build_conv3x3);
  DO_BENCH("02_dw_pt_conv_64x64",   build_dw64);
  DO_BENCH("03_matmul_2x2",         build_mat2);
  DO_BENCH("04_matmul_16x16",       build_mat16);
  DO_BENCH("05_add_const",          build_addc);
  DO_BENCH("06_mul_const",          build_mulc);
  DO_BENCH("07_mul_tensor",         build_mult);
  DO_BENCH("08_max_const",          build_maxc);
  DO_BENCH("09_min_const",          build_minc);
  DO_BENCH("10_copy",               build_copy);
  DO_BENCH("11_and_int8",           build_and);
  DO_BENCH("12_or_int8",            build_or);
  DO_BENCH("13_xor_int8",           build_xor);
  DO_BENCH("14_ge_const",           build_ge);
  DO_BENCH("15_arith_shift",        build_ash);
  DO_BENCH("16_max_pooling",        build_maxpool);
  DO_BENCH("17_avg_pooling",        build_avgpool);
  DO_BENCH("18_min_pooling",        build_minpool);

#undef DO_BENCH

  tpu_cleanup(&g_ctx);
  fprintf(stderr,"\n========== DONE cvk_100bench ==========\n\n");
  return 0;
}
