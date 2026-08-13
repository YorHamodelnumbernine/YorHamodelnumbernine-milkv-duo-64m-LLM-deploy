/* rs_noreload_check.c — Gate ④a lightweight on-board verification.
 *
 * Confirms bmk1822 supports two-pass INT8 matmul sharing the SAME LMEM
 * weight block with only a per-call rshift_bits change and NO g2l reload
 * between passes.  Also re-confirms N=512 tiling (④b) and full-N-column
 * correctness (④c).
 *
 * Flow (matches DESIGN_PATH_A_TWOPASS per-chunk two-pass):
 *   Test A: single cmdbuf — allocate ml_l/ml_r/ml_o ONCE, g2l once,
 *           matmul(11)->l2g P1, matmul(5)->l2g P2 (same ml_r, no reload).
 *   Test B: real flow — cmdbuf1 (g2l + matmul(rsafe) + l2g P1), run, read P1,
 *           compute r_opt, then cmdbuf2 (fresh ctx, same alloc layout,
 *           matmul(r_opt) + l2g P2, NO g2l). Verify P2 == ref_div(acc, r_opt).
 *   N=512, K=32.  |left|,|right| <= 100 -> r=11 safe, r=5 saturates some.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "cviruntime_context.h"
#include "bmkernel/bm1822/bmkernel_1822.h"

#define NEURON_SZ 0x800000UL
#define L_OFF     0
#define R_OFF     0x100000UL
#define P1_OFF    0x600000UL
#define P2_OFF    0x640000UL
#define CMDBUF_SZ 131072

static inline int sat8(int v){ return v>127?127:(v<-128?-128:v); }
static inline int8_t ref_div(int32_t acc,int rs){
  if(rs<=0) return (int8_t)sat8(acc);
  return (int8_t)sat8((acc+(1<<(rs-1)))>>rs);
}
static int ceil_log2_div(int x,int d){ int r=0; while((long long)x>((1LL<<r)*(long long)d)) r++; return r; }

typedef struct { CVI_RT_HANDLE rt; CVI_RT_MEM mem; uint64_t pa; uint8_t *va;
  uint8_t cmdbuf[CMDBUF_SZ] __attribute__((aligned(16))); } probe_t;

static int probe_init(probe_t *p){
  memset(p,0,sizeof(*p));
  if(CVI_RT_Init(&p->rt)) return -1;
  p->mem=CVI_RT_MemAlloc(p->rt,NEURON_SZ);
  p->pa=CVI_RT_MemGetPAddr(p->mem);
  p->va=CVI_RT_MemGetVAddr(p->mem);
  CVI_RT_SetBaseReg(p->rt,0,p->pa);
  return 0;
}

/* Allocate the 3 LMEM tensors once (deterministic layout). */
static int alloc3(bmk1822_context_t *bmk,int KC,int N,
                  bmk1822_matrix_lmem_t **ml_l,bmk1822_matrix_lmem_t **ml_r,
                  bmk1822_matrix_lmem_t **ml_o){
  bmk1822_matrix_lmem_shape_t sl={.n=1,.c=1,.w=(uint32_t)KC,.col=(uint32_t)KC};
  bmk1822_matrix_lmem_shape_t sr={.n=(uint32_t)KC,.c=1,.w=(uint32_t)N,.col=(uint32_t)N};
  bmk1822_matrix_lmem_shape_t so={.n=1,.c=1,.w=(uint32_t)N,.col=(uint32_t)N};
  *ml_l=bmk1822_lmem_alloc_matrix(bmk,sl,FMT_I8,1);
  *ml_r=bmk1822_lmem_alloc_matrix(bmk,sr,FMT_I8,1);
  *ml_o=bmk1822_lmem_alloc_matrix(bmk,so,FMT_I8,1);
  return (*ml_l&&*ml_r&&*ml_o)?1:0;
}

/* g2l left; if g2l_right, g2l right; then matmul(rs) -> l2g out_off.
 * ml_* pointers given so callers can reuse the same LMEM block. */
static void emit_phase(probe_t *p, bmk1822_context_t *bmk, int KC, int N,
                       bmk1822_matrix_lmem_t *ml_l, bmk1822_matrix_lmem_t *ml_r,
                       bmk1822_matrix_lmem_t *ml_o, int rs, int g2l_right,
                       uint64_t out_off){
  bmk1822_matrix_tgmem_t mg_l={0,L_OFF,FMT_I8,{1,(uint32_t)KC},{(uint32_t)KC}};
  bmk1822_tdma_g2l_matrix_copy(bmk,&(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_l,ml_l});
  if(g2l_right){
    bmk1822_matrix_tgmem_t mg_r={0,R_OFF,FMT_I8,{(uint32_t)KC,(uint32_t)N},{(uint32_t)N}};
    bmk1822_tdma_g2l_matrix_copy(bmk,&(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_r,ml_r});
  }
  bmk1822_tiu_matrix_multiplication_param_t pm={
    .res=ml_o,.left=ml_l,.right=ml_r,.bias=NULL,.lshift_bits=0,
    .rshift_bits=(uint8_t)rs,.res_is_int8=1,.relu_enable=0,.add_result=0,
    .ps32_mode=0,.layer_id=1};
  bmk1822_tiu_matrix_multiplication(bmk,&pm);
  bmk1822_matrix_tgmem_t mg_o={0,out_off,FMT_I8,{1,(uint32_t)N},{(uint32_t)N}};
  bmk1822_tdma_l2g_matrix_copy(bmk,&(bmk1822_tdma_l2tg_matrix_copy_param_t){ml_o,&mg_o});
}

static void run_cmdbuf(probe_t *p, bmk1822_context_t *bmk){
  uint32_t cmd_sz; uint8_t *cmd=bmk1822_acquire_cmdbuf(bmk,&cmd_sz);
  uint32_t psz,pmu; bmk1822_dmabuf_size(cmd,cmd_sz,&psz,&pmu);
  CVI_RT_MEM dm=CVI_RT_MemAlloc(p->rt,psz+pmu);
  uint8_t *dv=CVI_RT_MemGetVAddr(dm);
  bmk1822_dmabuf_convert(cmd,cmd_sz,dv);
  bmk1822_arraybase_set(dv,p->pa,0,0,0);
  CVI_RT_MemFlush(p->rt,dm);
  CVI_RT_MEM loaded;
  CVI_RT_LoadDmabuf(p->rt,dm,psz+pmu,p->pa,0,false,&loaded);
  CVI_RT_RunCmdbufEx(p->rt,loaded,&(CVI_RT_ARRAYBASE){.gaddr_base0=p->pa});
  CVI_RT_MemFree(p->rt,dm);
}

static int verify(const char *tag, probe_t *p, uint64_t off, int N,
                  const int8_t *ref, int rshift){
  const int8_t *r=(const int8_t*)(p->va+off);
  int bad=0, sat=0, first=-1;
  for(int n=0;n<N;n++){ if(r[n]!=ref[n]){ bad++; if(first<0)first=n; } if(r[n]==127||r[n]==-128)sat++; }
  printf("  %s rshift=%2d: ok=%d/%d bad=%d first@%d sat8=%d %s\n",
         tag,rshift,N-bad,N,bad,first,sat,bad==0?"PASS":"FAIL");
  return bad;
}

int main(void){
  printf("== rs_noreload_check: same-LMEM weight, per-call rshift, NO reload ==\n");
  int KC=32, N=512;
  int8_t *left=malloc(KC), *right=malloc(KC*N), *p1ref=malloc(N), *p2ref=malloc(N);
  int32_t *acc=malloc(N*sizeof(int32_t));
  srand(20260813);
  for(int k=0;k<KC;k++) left[k]=(int8_t)(rand()%200-100);
  for(int n=0;n<N;n++){ acc[n]=0;
    for(int k=0;k<KC;k++){ right[k*N+n]=(int8_t)(rand()%200-100); acc[n]+=(int32_t)left[k]*right[k*N+n]; }
    p1ref[n]=ref_div(acc[n],11);
    p2ref[n]=ref_div(acc[n],5);
  }

  probe_t p;
  if(probe_init(&p)){ printf("init fail\n"); return 1; }
  memset(p.va,0,NEURON_SZ);
  memcpy(p.va+L_OFF,left,KC);
  for(int k=0;k<KC;k++) memcpy(p.va+R_OFF+(size_t)k*N,right+(size_t)k*N,N);
  CVI_RT_MemFlush(p.rt,p.mem);

  /* ---- Test A: one cmdbuf, BOTH matmuls on the SAME ml_r, no reload ---- */
  printf("  [Test A] single cmdbuf: alloc once, g2l once, matmul(11)+matmul(5)\n");
  int badA=0;
  {
    bmk_info_t info={.chip_version=1822,.cmdbuf_size=sizeof(p.cmdbuf),.cmdbuf=p.cmdbuf};
    bmk1822_context_t *bmk=bmk1822_register(&info);
    bmk1822_matrix_lmem_t *ml_l,*ml_r,*ml_o;
    if(alloc3(bmk,KC,N,&ml_l,&ml_r,&ml_o)){
      emit_phase(&p,bmk,KC,N,ml_l,ml_r,ml_o,11,1,P1_OFF);  /* g2l both */
      emit_phase(&p,bmk,KC,N,ml_l,ml_r,ml_o,5,0,P2_OFF);   /* NO g2l */
      run_cmdbuf(&p,bmk);
      CVI_RT_MemInvld(p.rt,p.mem);
      badA += verify("A-pass1",&p,P1_OFF,N,p1ref,11);
      badA += verify("A-pass2",&p,P2_OFF,N,p2ref,5);
    } else printf("  [Test A] alloc FAIL\n");
    bmk1822_cleanup(bmk);
  }

  /* ---- Test B: real two-cmdbuf flow, fresh ctx reuses same LMEM layout ---- */
  printf("  [Test B] two cmdbufs: cmdbuf1 g2l+mm(rsafe), cmdbuf2 mm(r_opt) NO g2l\n");
  int badB=0;
  {
    bmk_info_t info1={.chip_version=1822,.cmdbuf_size=sizeof(p.cmdbuf),.cmdbuf=p.cmdbuf};
    bmk1822_context_t *bmk1=bmk1822_register(&info1);
    bmk1822_matrix_lmem_t *l1,*r1,*o1;
    int rsafe=11;
    if(alloc3(bmk1,KC,N,&l1,&r1,&o1)){
      emit_phase(&p,bmk1,KC,N,l1,r1,o1,rsafe,1,P1_OFF);
      run_cmdbuf(&p,bmk1);
      CVI_RT_MemInvld(p.rt,p.mem);
      int maxabs=0; const int8_t *r=(const int8_t*)(p.va+P1_OFF);
      for(int n=0;n<N;n++){ int a=r[n]; if(a<0)a=-a; if(a>maxabs)maxabs=a; }
      int r_opt=(maxabs>0)?(rsafe+ceil_log2_div(maxabs,127)):0;
      int8_t *p2opt=malloc(N);
      for(int n=0;n<N;n++) p2opt[n]=ref_div(acc[n],r_opt);
      printf("    rsafe=%d maxabs=%d -> r_opt=%d (design flow)\n",rsafe,maxabs,r_opt);
      uint8_t cmdbuf2[CMDBUF_SZ] __attribute__((aligned(16)));
      bmk_info_t info2={.chip_version=1822,.cmdbuf_size=sizeof(cmdbuf2),.cmdbuf=cmdbuf2};
      bmk1822_context_t *bmk2=bmk1822_register(&info2);
      bmk1822_matrix_lmem_t *l2,*r2,*o2;
      if(alloc3(bmk2,KC,N,&l2,&r2,&o2)){   /* same shapes -> same LMEM addrs */
        emit_phase(&p,bmk2,KC,N,l2,r2,o2,r_opt,0,P2_OFF);   /* NO g2l right */
        run_cmdbuf(&p,bmk2);
        CVI_RT_MemInvld(p.rt,p.mem);
        badB += verify("B-pass2",&p,P2_OFF,N,p2opt,r_opt);
      } else printf("    [Test B] pass2 alloc FAIL\n");
      bmk1822_cleanup(bmk2);
      free(p2opt);
    } else printf("  [Test B] pass1 alloc FAIL\n");
    bmk1822_cleanup(bmk1);
  }

  printf("== TOTAL badA=%d badB=%d ==\n",badA,badB);
  CVI_RT_MemFree(p.rt,p.mem); CVI_RT_DeInit(p.rt);
  free(left); free(right); free(p1ref); free(p2ref); free(acc);
  return badA+badB;
}
