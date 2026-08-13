#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "cviruntime_context.h"
#include "bmkernel/bm1822/bmkernel_1822.h"
#define NEURON_SZ 65536
static void run(int use_default, int K, int N, int rshift) {
  CVI_RT_HANDLE rt; CVI_RT_Init(&rt);
  CVI_RT_MEM mem = CVI_RT_MemAlloc(rt, NEURON_SZ);
  uint64_t pa = CVI_RT_MemGetPAddr(mem);
  uint8_t *va = CVI_RT_MemGetVAddr(mem);
  CVI_RT_SetBaseReg(rt, 0, pa);
  memset(va, 0, NEURON_SZ);
  /* left[K] = k+1 ; right[k*N+n] = (k+1)*(n+1) -> acc = sum_k (k+1)^2 * (n+1) */
  int8_t *left=(int8_t*)(va+0), *right=(int8_t*)(va+2048);
  for(int k=0;k<K;k++){ left[k]=k+1; for(int n=0;n<N;n++) right[k*N+n]=(k+1)*(n+1); }
  CVI_RT_MemFlush(rt, mem);
  uint8_t cmdbuf[16384] __attribute__((aligned(16)));
  bmk_info_t info={.chip_version=1822,.cmdbuf_size=sizeof(cmdbuf),.cmdbuf=cmdbuf};
  bmk1822_context_t *bmk=bmk1822_register(&info);
  bmk1822_matrix_lmem_shape_t sl, sr, so;
  if (use_default) {
    sl=bmk1822_matrix_lmem_default_shape(bmk,1,K,FMT_I8);
    sr=bmk1822_matrix_lmem_default_shape(bmk,K,N,FMT_I8);
    so=bmk1822_matrix_lmem_default_shape(bmk,1,N,FMT_I8);
  } else {
    sl=(bmk1822_matrix_lmem_shape_t){.n=1,.c=1,.w=(uint32_t)K,.col=(uint32_t)K};
    sr=(bmk1822_matrix_lmem_shape_t){.n=(uint32_t)K,.c=1,.w=(uint32_t)N,.col=(uint32_t)N};
    so=(bmk1822_matrix_lmem_shape_t){.n=1,.c=1,.w=(uint32_t)N,.col=(uint32_t)N};
  }
  bmk1822_matrix_lmem_t *ml_l=bmk1822_lmem_alloc_matrix(bmk,sl,FMT_I8,1);
  bmk1822_matrix_lmem_t *ml_r=bmk1822_lmem_alloc_matrix(bmk,sr,FMT_I8,1);
  bmk1822_matrix_lmem_t *ml_o=bmk1822_lmem_alloc_matrix(bmk,so,FMT_I8,1);
  if(!ml_l||!ml_r||!ml_o){printf("alloc fail\n");return;}
  printf("K=%d N=%d %s: l{n=%u,c=%u,w=%u,col=%u} r{n=%u,c=%u,w=%u,col=%u} o{n=%u,c=%u,w=%u,col=%u}\n",
    K,N,use_default?"DEFAULT":"FORCED", sl.n,sl.c,sl.w,sl.col, sr.n,sr.c,sr.w,sr.col, so.n,so.c,so.w,so.col);
  bmk1822_matrix_tgmem_t mg_l={0,0,FMT_I8,{1,(uint32_t)K},{(uint32_t)K}};
  bmk1822_matrix_tgmem_t mg_r={0,2048,FMT_I8,{(uint32_t)K,(uint32_t)N},{(uint32_t)N}};
  bmk1822_tdma_g2l_matrix_copy(bmk,&(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_l,ml_l});
  bmk1822_tdma_g2l_matrix_copy(bmk,&(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_r,ml_r});
  bmk1822_tiu_matrix_multiplication_param_t p={.res=ml_o,.left=ml_l,.right=ml_r,.bias=NULL,
    .lshift_bits=0,.rshift_bits=(uint8_t)rshift,.res_is_int8=1,.relu_enable=0,.add_result=0,.ps32_mode=0,.layer_id=1};
  if(!bmk1822_tiu_matrix_multiplication(bmk,&p)){printf("  REJECTED\n");return;}
  bmk1822_matrix_tgmem_t mg_o={0,4096,FMT_I8,{1,(uint32_t)N},{(uint32_t)N}};
  bmk1822_tdma_l2g_matrix_copy(bmk,&(bmk1822_tdma_l2tg_matrix_copy_param_t){ml_o,&mg_o});
  uint32_t cmd_sz; uint8_t *cmd=bmk1822_acquire_cmdbuf(bmk,&cmd_sz);
  uint32_t psize,pmu_size; bmk1822_dmabuf_size(cmd,cmd_sz,&psize,&pmu_size);
  CVI_RT_MEM dm=CVI_RT_MemAlloc(rt,psize+pmu_size);
  uint8_t *db=CVI_RT_MemGetVAddr(dm);
  bmk1822_dmabuf_convert(cmd,cmd_sz,db); bmk1822_arraybase_set(db,pa,0,0,0);
  CVI_RT_MemFlush(rt,dm);
  CVI_RT_MEM ld; CVI_RT_LoadDmabuf(rt,dm,psize+pmu_size,pa,0,false,&ld);
  CVI_RT_RunCmdbufEx(rt,ld,&(CVI_RT_ARRAYBASE){.gaddr_base0=pa});
  CVI_RT_MemInvld(rt,mem);
  int8_t *r=(int8_t*)(va+4096);
  printf("  out[0..%d]:", N-1); for(int i=0;i<N&&i<8;i++) printf(" %d", r[i]);
  printf("  (host expect:");
  for(int n=0;n<N&&n<8;n++){ int32_t acc=0; for(int k=0;k<K;k++) acc+=(k+1)*(k+1)*(n+1); int v=acc>>rshift; if(v>127)v=127; if(v<-128)v=-128; printf(" %d",v);} printf(")\n");
  CVI_RT_MemFree(rt,dm); bmk1822_cleanup(bmk); CVI_RT_MemFree(rt,mem); CVI_RT_DeInit(rt);
}
int main(){
  printf("== dbg_mm ==\n");
  run(0, 2, 2, 0);
  run(1, 2, 2, 0);
  run(0, 4, 4, 0);
  run(1, 4, 4, 0);
  run(0, 32, 32, 8);
  run(1, 32, 32, 8);
  return 0;
}
