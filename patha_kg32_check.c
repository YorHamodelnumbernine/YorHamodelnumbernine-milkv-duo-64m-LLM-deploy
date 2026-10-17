/* patha_kg32_check.c — Path A (INT4 K-aligned, KG=32=G) two-pass INT8 matmul
 * on-board gate verification.  CEO task 2026-08-13, updated target: KG=32.
 *
 * Qwen2.5-0.5B shapes (decode M=1):
 *   q/wo: K=896 N=896 ; k/v: K=896 N=128 ; up/gate: K=896 N=4864 ;
 *   down: K=4864 N=896
 * KG=32  =>  K=896 -> 28 chunks, K=4864 -> 152 chunks, x2 passes.
 *
 * Gates:
 *  ① pass1 int8 readback = standard matrix_multiplication semantics, directly
 *    readable via TDMA l2g matrix copy + MemInvld (P0)
 *  ② per-chunk 2-pass submit latency @ KG=32, extrapolate per-token vs SD
 *    9.2s/token budget (P3)
 *  ④ rshift 1..15 precision/saturation at KG=32, host 3/3 impact (P1)
 *  (correctness two-pass BIT-EXACT vs host in P2; ③ ION layout analytic)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include "cviruntime_context.h"
#include "bmkernel/bm1822/bmkernel_1822.h"

#define NEURON_SZ  0x800000UL
#define L_OFF      0
#define R_OFF      0x100000UL
#define P1_OFF     0x600000UL
#define P2_OFF     0x640000UL
#define CMDBUF_SZ  262144

static inline int sat8(int v){ return v>127?127:(v<-128?-128:v); }
static inline int8_t ref_div(int32_t acc, int rshift){
  if (rshift<=0) return (int8_t)sat8(acc);
  return (int8_t)sat8((acc + (1<<(rshift-1))) >> rshift);
}
/* smallest r>=0 with x <= (1<<r)*d   (ceil(log2(x/d))) */
static int ceil_log2_div(int x, int d){
  int r=0;
  while ((long long)x > ((1LL<<r)*(long long)d)) r++;
  return r;
}
static double now_ms(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
  return t.tv_sec*1000.0 + t.tv_nsec/1e6; }

static uint32_t rng_state=0x9E3779B9;
static int rng_int(int lo, int hi){
  uint32_t x=rng_state; x^=x<<13; x^=x>>17; x^=x<<5; rng_state=x;
  return lo + (int)(x % (uint32_t)(hi-lo+1));
}

typedef struct {
  CVI_RT_HANDLE rt;
  CVI_RT_MEM mem;
  uint64_t pa;
  uint8_t *va;
  uint8_t cmdbuf1[CMDBUF_SZ] __attribute__((aligned(16)));
  uint8_t cmdbuf2[CMDBUF_SZ] __attribute__((aligned(16)));
} probe_t;

static int probe_init(probe_t *p, size_t bytes){
  memset(p,0,sizeof(*p));
  if (CVI_RT_Init(&p->rt)) { printf("CVI_RT_Init fail\n"); return -1; }
  p->mem = CVI_RT_MemAlloc(p->rt, bytes);
  if (!p->mem) { printf("MemAlloc fail\n"); return -1; }
  p->pa = CVI_RT_MemGetPAddr(p->mem);
  p->va = CVI_RT_MemGetVAddr(p->mem);
  CVI_RT_SetBaseReg(p->rt, 0, p->pa);
  return 0;
}
static void probe_cleanup(probe_t *p){
  if (p->mem) CVI_RT_MemFree(p->rt, p->mem);
  if (p->rt)  CVI_RT_DeInit(p->rt);
}

static double run_bmk(probe_t *p, bmk1822_context_t *bmk, double *build_ms){
  double t0=now_ms();
  uint32_t cmd_sz; uint8_t *cmd = bmk1822_acquire_cmdbuf(bmk, &cmd_sz);
  uint32_t psize, pmu_size; bmk1822_dmabuf_size(cmd, cmd_sz, &psize, &pmu_size);
  CVI_RT_MEM dm = CVI_RT_MemAlloc(p->rt, psize+pmu_size);
  uint8_t *dv = CVI_RT_MemGetVAddr(dm);
  bmk1822_dmabuf_convert(cmd, cmd_sz, dv);
  bmk1822_arraybase_set(dv, p->pa, 0, 0, 0);
  CVI_RT_MemFlush(p->rt, dm);
  CVI_RT_MEM loaded;
  CVI_RT_LoadDmabuf(p->rt, dm, psize+pmu_size, p->pa, 0, false, &loaded);
  double t1=now_ms();
  CVI_RT_RunCmdbufEx(p->rt, loaded, &(CVI_RT_ARRAYBASE){.gaddr_base0=p->pa});
  double t2=now_ms();
  CVI_RT_MemFree(p->rt, dm);
  if (build_ms) *build_ms = t1-t0;
  return t2-t1;
}

/* one phase of one K-chunk: g2l left chunk + ntiles*[g2l right, matmul, l2g out] */
static void build_chunk_phase(probe_t *p, bmk1822_context_t *bmk,
                              int g, int KC, int N_pad, int Ntile, int ntiles,
                              int rshift, uint64_t out_off){
  bmk1822_matrix_lmem_shape_t sl={.n=1,.c=1,.w=(uint32_t)KC,.col=(uint32_t)KC};
  bmk1822_matrix_lmem_shape_t sr={.n=(uint32_t)KC,.c=1,.w=(uint32_t)Ntile,.col=(uint32_t)Ntile};
  bmk1822_matrix_lmem_shape_t so={.n=1,.c=1,.w=(uint32_t)Ntile,.col=(uint32_t)Ntile};
  bmk1822_matrix_lmem_t *ml_l=bmk1822_lmem_alloc_matrix(bmk,sl,FMT_I8,1);
  bmk1822_matrix_lmem_t *ml_r=bmk1822_lmem_alloc_matrix(bmk,sr,FMT_I8,1);
  bmk1822_matrix_lmem_t *ml_res=bmk1822_lmem_alloc_matrix(bmk,so,FMT_I8,1);
  if(!ml_l||!ml_r||!ml_res){ printf("    [build] lmem alloc FAIL g=%d KC=%d Ntile=%d\n",g,KC,Ntile); return; }
  bmk1822_matrix_tgmem_t mg_l={0,L_OFF+(uint64_t)g,FMT_I8,{1,(uint32_t)KC},{(uint32_t)KC}};
  bmk1822_tdma_g2l_matrix_copy(bmk,&(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_l,ml_l});
  for(int t=0;t<ntiles;t++){
    uint64_t ns=(uint64_t)t*Ntile;
    bmk1822_matrix_tgmem_t mg_r={0,R_OFF+(uint64_t)g*N_pad+ns,FMT_I8,
                                 {(uint32_t)KC,(uint32_t)Ntile},{(uint32_t)N_pad}};
    bmk1822_tdma_g2l_matrix_copy(bmk,&(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_r,ml_r});
    bmk1822_tiu_matrix_multiplication_param_t pm={
      .res=ml_res,.left=ml_l,.right=ml_r,.bias=NULL,.lshift_bits=0,
      .rshift_bits=(uint8_t)rshift,.res_is_int8=1,.relu_enable=0,.add_result=0,
      .ps32_mode=0,.layer_id=1};
    if(!bmk1822_tiu_matrix_multiplication(bmk,&pm)) printf("    [build] P REJECTED t=%d\n",t);
    bmk1822_matrix_tgmem_t mg_o={0,out_off+ns,FMT_I8,{1,(uint32_t)Ntile},{(uint32_t)Ntile}};
    bmk1822_tdma_l2g_matrix_copy(bmk,&(bmk1822_tdma_l2tg_matrix_copy_param_t){ml_res,&mg_o});
  }
}

/* per-chunk two-pass data flow (M=1). right pre-placed padded [Kdim,N_pad] at R_OFF. */
static int two_pass_matmul(probe_t *p,
                           const int8_t *left, int K,
                           const int8_t *right, int Kdim, int N,
                           int KG, int Ntile, int rsafe,
                           float *acc,
                           double *t_p1b, double *t_p1r,
                           double *t_p2b, double *t_p2r,
                           double *t_read_cpu,
                           int *r_opt_min, int *r_opt_max){
  int N_pad=((N+Ntile-1)/Ntile)*Ntile;
  int ntiles=N_pad/Ntile;
  int nchunks=(K+KG-1)/KG;
  memset(p->va+R_OFF,0,(size_t)Kdim*N_pad);
  for(int r=0;r<Kdim;r++) memcpy(p->va+R_OFF+(size_t)r*N_pad, right+(size_t)r*N, (size_t)N);
  memcpy(p->va+L_OFF, left, (size_t)K);
  CVI_RT_MemFlush(p->rt, p->mem);
  int8_t *p1=(int8_t*)malloc(N_pad);
  int8_t *p2=(int8_t*)malloc(N_pad);
  int rmin=99,rmax=-1;
  double t_read=0;
  for(int g=0;g<K;g+=KG){
    int KC=(g+KG<=K)?KG:(K-g);
    /* ---- pass1 ---- */
    bmk_info_t info1={.chip_version=1822,.cmdbuf_size=sizeof(p->cmdbuf1),.cmdbuf=p->cmdbuf1};
    double tb0=now_ms();
    bmk1822_context_t *bmk1=bmk1822_register(&info1);
    build_chunk_phase(p,bmk1,g,KC,N_pad,Ntile,ntiles,rsafe,P1_OFF);
    double bms; double rms=run_bmk(p,bmk1,&bms);
    bmk1822_cleanup(bmk1);
    *t_p1b+=bms; *t_p1r+=rms;
    /* read P1 + max -> r_opt */
    double tr0=now_ms();
    CVI_RT_MemInvld(p->rt,p->mem);
    memcpy(p1,p->va+P1_OFF,N_pad);
    int maxabs=0;
    for(int n=0;n<N;n++){ int a=p1[n]; if(a<0)a=-a; if(a>maxabs)maxabs=a; }
    int r_opt=(maxabs>0)?(rsafe+ceil_log2_div(maxabs,127)):0;
    if(r_opt<0)r_opt=0;
    if(r_opt<rmin)rmin=r_opt; if(r_opt>rmax)rmax=r_opt;
    /* ---- pass2 ---- */
    bmk_info_t info2={.chip_version=1822,.cmdbuf_size=sizeof(p->cmdbuf2),.cmdbuf=p->cmdbuf2};
    double tb1=now_ms();
    bmk1822_context_t *bmk2=bmk1822_register(&info2);
    build_chunk_phase(p,bmk2,g,KC,N_pad,Ntile,ntiles,r_opt,P2_OFF);
    double bms2; double rms2=run_bmk(p,bmk2,&bms2);
    bmk1822_cleanup(bmk2);
    *t_p2b+=bms2; *t_p2r+=rms2;
    /* read P2 + CPU fp32 accumulate */
    double tc0=now_ms();
    CVI_RT_MemInvld(p->rt,p->mem);
    memcpy(p2,p->va+P2_OFF,N_pad);
    int sh=1<<r_opt;
    for(int n=0;n<N;n++) acc[n]+=(float)((int)p2[n]*sh);
    double tc1=now_ms();
    t_read+=(tc1-tc0);
  }
  *t_read_cpu=t_read;
  *r_opt_min=rmin; *r_opt_max=rmax;
  free(p1); free(p2);
  return nchunks;
}

/* ============ P0: Gate ① pass1 int8 readback, production shapes ============ */
/* standard INT8 matmul (M=1) one K-chunk, rshift, readback via l2g+MemInvld,
 * verify == ref_div.  Multi-tile for wide N. */
static void gate1_readback(probe_t *p, const char *name, int K, int N,
                           int Ntile, int rshift){
  int N_pad=((N+Ntile-1)/Ntile)*Ntile, ntiles=N_pad/Ntile;
  int KC=(K<32)?K:32;                 /* one KG=32 chunk (or tail) */
  int8_t *host=malloc(N_pad);
  memset(p->va+L_OFF,0,KC+64);
  memset(p->va+R_OFF,0,(size_t)KC*N_pad);
  srand(7+K*131+N);
  for(int k=0;k<KC;k++) p->va[L_OFF+k]=(int8_t)(rand()%200-100);
  for(int n=0;n<N;n++){ int32_t acc=0;
    for(int k=0;k<KC;k++){ int8_t rv=(int8_t)(rand()%200-100);
      p->va[R_OFF+(size_t)k*N_pad+n]=rv; acc+=(int32_t)(int8_t)p->va[L_OFF+k]*rv; }
    host[n]=ref_div(acc,rshift); }
  for(int n=N;n<N_pad;n++) host[n]=0;
  CVI_RT_MemFlush(p->rt,p->mem);
  bmk_info_t info={.chip_version=1822,.cmdbuf_size=sizeof(p->cmdbuf1),.cmdbuf=p->cmdbuf1};
  bmk1822_context_t *bmk=bmk1822_register(&info);
  bmk1822_matrix_lmem_shape_t sl={.n=1,.c=1,.w=(uint32_t)KC,.col=(uint32_t)KC};
  bmk1822_matrix_lmem_shape_t sr={.n=(uint32_t)KC,.c=1,.w=(uint32_t)Ntile,.col=(uint32_t)Ntile};
  bmk1822_matrix_lmem_shape_t so={.n=1,.c=1,.w=(uint32_t)Ntile,.col=(uint32_t)Ntile};
  bmk1822_matrix_lmem_t *ml_l=bmk1822_lmem_alloc_matrix(bmk,sl,FMT_I8,1);
  bmk1822_matrix_lmem_t *ml_r=bmk1822_lmem_alloc_matrix(bmk,sr,FMT_I8,1);
  bmk1822_matrix_lmem_t *ml_res=bmk1822_lmem_alloc_matrix(bmk,so,FMT_I8,1);
  if(!ml_l||!ml_r||!ml_res){ printf("  [G1] %-7s K=%d N=%d: alloc FAIL\n",name,K,N);
    free(host); bmk1822_cleanup(bmk); return; }
  bmk1822_matrix_tgmem_t mg_l={0,L_OFF,FMT_I8,{1,(uint32_t)KC},{(uint32_t)KC}};
  bmk1822_tdma_g2l_matrix_copy(bmk,&(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_l,ml_l});
  for(int t=0;t<ntiles;t++){
    uint64_t ns=(uint64_t)t*Ntile;
    bmk1822_matrix_tgmem_t mg_r={0,R_OFF+ns,FMT_I8,{(uint32_t)KC,(uint32_t)Ntile},{(uint32_t)N_pad}};
    bmk1822_tdma_g2l_matrix_copy(bmk,&(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_r,ml_r});
    bmk1822_tiu_matrix_multiplication_param_t pm={
      .res=ml_res,.left=ml_l,.right=ml_r,.bias=NULL,.lshift_bits=0,
      .rshift_bits=(uint8_t)rshift,.res_is_int8=1,.relu_enable=0,.add_result=0,
      .ps32_mode=0,.layer_id=1};
    if(!bmk1822_tiu_matrix_multiplication(bmk,&pm)) printf("  [G1] REJECTED t=%d\n",t);
    bmk1822_matrix_tgmem_t mg_o={0,P1_OFF+ns,FMT_I8,{1,(uint32_t)Ntile},{(uint32_t)Ntile}};
    bmk1822_tdma_l2g_matrix_copy(bmk,&(bmk1822_tdma_l2tg_matrix_copy_param_t){ml_res,&mg_o});
  }
  double bms; run_bmk(p,bmk,&bms); bmk1822_cleanup(bmk);
  CVI_RT_MemInvld(p->rt,p->mem);
  int8_t *r=(int8_t*)(p->va+P1_OFF);
  int bad=0, sat=0;
  for(int n=0;n<N_pad;n++){ if(r[n]!=host[n])bad++; if(r[n]==127||r[n]==-128)sat++; }
  printf("  [G1] %-7s K=%2d N=%-5d ntile=%-3d tiles=%d rshift=%2d: ok=%d/%d bad=%d sat8=%d%s\n",
         name,KC,N,Ntile,ntiles,rshift,N_pad-bad,N_pad,bad,sat,
         bad==0?"  <- pass1 READBACK OK":"  <- MISMATCH");
  free(host);
}

/* ============ P1: Gate ④ rshift sweep at KG=32 ============ */
static void part1_rshift(probe_t *p){
  printf("\n===== P1: TIU rshift precision/saturation at KG=32 =====\n");
  int rs[]={1,2,4,8,12,15};
  int KG=32, Nt=896;
  for(int i=0;i<6;i++){
    int rsh=rs[i];
    memset(p->va+L_OFF,0,KG+64);
    memset(p->va+R_OFF,0,(size_t)KG*Nt);
    rng_state=3000+i;
    for(int k=0;k<KG;k++) p->va[L_OFF+k]=(int8_t)rng_int(-80,80);
    int8_t *host=malloc(Nt);
    for(int n=0;n<Nt;n++){ int32_t acc=0;
      for(int k=0;k<KG;k++){ int8_t rv=(int8_t)rng_int(-30,30);
        p->va[R_OFF+(size_t)k*Nt+n]=rv; acc+=(int32_t)(int8_t)p->va[L_OFF+k]*rv; }
      host[n]=ref_div(acc,rsh); }
    CVI_RT_MemFlush(p->rt,p->mem);
    bmk_info_t info={.chip_version=1822,.cmdbuf_size=sizeof(p->cmdbuf1),.cmdbuf=p->cmdbuf1};
    bmk1822_context_t *bmk=bmk1822_register(&info);
    bmk1822_matrix_lmem_shape_t sl={.n=1,.c=1,.w=KG,.col=(uint32_t)KG};
    bmk1822_matrix_lmem_shape_t sr={.n=KG,.c=1,.w=Nt,.col=(uint32_t)Nt};
    bmk1822_matrix_lmem_shape_t so={.n=1,.c=1,.w=Nt,.col=(uint32_t)Nt};
    bmk1822_matrix_lmem_t *ml_l=bmk1822_lmem_alloc_matrix(bmk,sl,FMT_I8,1);
    bmk1822_matrix_lmem_t *ml_r=bmk1822_lmem_alloc_matrix(bmk,sr,FMT_I8,1);
    bmk1822_matrix_lmem_t *ml_res=bmk1822_lmem_alloc_matrix(bmk,so,FMT_I8,1);
    bmk1822_matrix_tgmem_t mg_l={0,L_OFF,FMT_I8,{1,KG},{KG}};
    bmk1822_tdma_g2l_matrix_copy(bmk,&(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_l,ml_l});
    bmk1822_matrix_tgmem_t mg_r={0,R_OFF,FMT_I8,{KG,Nt},{Nt}};
    bmk1822_tdma_g2l_matrix_copy(bmk,&(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_r,ml_r});
    bmk1822_tiu_matrix_multiplication_param_t pm={
      .res=ml_res,.left=ml_l,.right=ml_r,.bias=NULL,.lshift_bits=0,
      .rshift_bits=(uint8_t)rsh,.res_is_int8=1,.relu_enable=0,.add_result=0,.ps32_mode=0,.layer_id=1};
    bmk1822_tiu_matrix_multiplication(bmk,&pm);
    bmk1822_matrix_tgmem_t mg_o={0,P1_OFF,FMT_I8,{1,Nt},{Nt}};
    bmk1822_tdma_l2g_matrix_copy(bmk,&(bmk1822_tdma_l2tg_matrix_copy_param_t){ml_res,&mg_o});
    double bms; run_bmk(p,bmk,&bms); bmk1822_cleanup(bmk);
    CVI_RT_MemInvld(p->rt,p->mem);
    int8_t *r=(int8_t*)(p->va+P1_OFF);
    int bad=0, sat=0;
    for(int n=0;n<Nt;n++){ if(r[n]!=host[n])bad++; if(r[n]==127||r[n]==-128)sat++; }
    printf("  rshift=%2d K=%d N=%d: ok=%d/%d bad=%d sat8=%d | r0..7=%d %d %d %d %d %d %d %d\n",
           rsh,KG,Nt,Nt-bad,Nt,bad,sat,r[0],r[1],r[2],r[3],r[4],r[5],r[6],r[7]);
    free(host);
  }
}

/* ============ P2: two-pass correctness at KG=32 ============ */
static void part2_twopass(probe_t *p){
  printf("\n===== P2: two-pass BIT-EXACT vs host at KG=32 =====\n");
  struct { const char *name; int K,N; } shapes[]={
    {"q/wo",896,896},{"k/v",896,128},{"up/gate",896,4864},{"down",4864,896}};
  int KG=32;
  for(int si=0;si<4;si++){
    int K=shapes[si].K, N=shapes[si].N;
    int Ntile=(N<896)?N:896;
    while((int)(KG*Ntile + KG + Ntile) > 30000) Ntile-=16;
    int N_pad=((N+Ntile-1)/Ntile)*Ntile;
    int rsafe=ceil_log2_div(KG*80*30,127)+1;   /* |left|<=80, |right|<=30 */
    int8_t *left=malloc(K), *right=malloc((size_t)K*N);
    float *lsc=malloc(N*sizeof(float));
    rng_state=1000+si;
    for(int k=0;k<K;k++) left[k]=(int8_t)rng_int(-80,80);
    for(int n=0;n<N;n++){ lsc[n]=1.0f/(float)rng_int(5,60);
      for(int k=0;k<K;k++) right[(size_t)k*N+n]=(int8_t)rng_int(-30,30); }
    /* host two-pass reference */
    float *ref_tp=calloc(N,sizeof(float));
    int32_t *acc=malloc(N*sizeof(int32_t)); int8_t *p1=malloc(N);
    for(int g=0;g<K;g+=KG){
      int KC=(g+KG<=K)?KG:(K-g);
      for(int n=0;n<N;n++){ int32_t s=0;
        for(int k=0;k<KC;k++) s+=(int32_t)left[g+k]*right[(size_t)(g+k)*N+n];
        acc[n]=s; }
      int mx1=0; for(int n=0;n<N;n++){ p1[n]=ref_div(acc[n],rsafe); int a=p1[n]; if(a<0)a=-a; if(a>mx1)mx1=a; }
      int r_tp=(mx1>0)?(rsafe+ceil_log2_div(mx1,127)):0;
      for(int n=0;n<N;n++) ref_tp[n]+=(float)((int)ref_div(acc[n],r_tp)*(1<<r_tp));
    }
    for(int n=0;n<N;n++) ref_tp[n]*=lsc[n];
    /* on-board */
    float *acc_on=calloc(N,sizeof(float));
    double t1b=0,t1r=0,t2b=0,t2r=0,trc=0; int rmin,rmax;
    int nch=two_pass_matmul(p,left,K,right,K,N,KG,Ntile,rsafe,acc_on,
                            &t1b,&t1r,&t2b,&t2r,&trc,&rmin,&rmax);
    for(int n=0;n<N;n++) acc_on[n]*=lsc[n];
    int bad=0; double maxrel=0;
    for(int n=0;n<N;n++){
      if(acc_on[n]!=ref_tp[n]) bad++;
      double d=acc_on[n]-ref_tp[n]; if(d<0)d=-d;
      double rel=(fabsf(ref_tp[n])>1e-6)?d/fabsf(ref_tp[n]):0;
      if(rel>maxrel)maxrel=rel;
    }
    printf("  %-7s KG=%-3d Ntile=%-3d Npad=%-4d rsafe=%-2d chunks=%d r_opt[%d..%d] submits=%d\n",
           shapes[si].name,KG,Ntile,N_pad,rsafe,nch,rmin,rmax,nch*2);
    printf("           vs host-two-pass: %s (%d/%d mism, maxrel=%.2e) | t_p1=%.2f t_p2=%.2f read+cpu=%.2f ms\n",
           bad==0?"BIT-EXACT":"MISMATCH",N-bad,N,maxrel,t1b+t1r,t2b+t2r,trc);
    free(left); free(right); free(lsc); free(ref_tp); free(acc); free(p1); free(acc_on);
  }
}

/* ============ P3: Gate ② submit latency at KG=32 ============ */
static void part3_latency(probe_t *p){
  printf("\n===== P3: per-chunk 2-pass submit latency at KG=32 =====\n");
  struct { const char *name; int K,N; } shapes[]={
    {"q/wo",896,896},{"k/v",896,128},{"up/gate",896,4864},{"down",4864,896}};
  int KG=32;
  for(int si=0;si<4;si++){
    int K=shapes[si].K, N=shapes[si].N;
    int Ntile=(N<896)?N:896;
    while((int)(KG*Ntile+KG+Ntile)>30000) Ntile-=16;
    int N_pad=((N+Ntile-1)/Ntile)*Ntile;
    int rsafe=ceil_log2_div(KG*80*30,127)+1;
    int8_t *left=malloc(K), *right=malloc((size_t)K*N);
    rng_state=2000+si;
    for(int k=0;k<K;k++) left[k]=(int8_t)rng_int(-80,80);
    for(int n=0;n<N;n++) for(int k=0;k<K;k++) right[(size_t)k*N+n]=(int8_t)rng_int(-30,30);
    float *acc=calloc(N,sizeof(float));
    double t1b=0,t1r=0,t2b=0,t2r=0,trc=0; int rmin,rmax;
    two_pass_matmul(p,left,K,right,K,N,KG,Ntile,rsafe,acc,&t1b,&t1r,&t2b,&t2r,&trc,&rmin,&rmax);
    memset(acc,0,N*sizeof(float));
    t1b=t1r=t2b=t2r=trc=0;
    int nch=two_pass_matmul(p,left,K,right,K,N,KG,Ntile,rsafe,acc,&t1b,&t1r,&t2b,&t2r,&trc,&rmin,&rmax);
    double tot=t1b+t1r+t2b+t2r+trc;
    int tiles=(N_pad/Ntile);
    printf("  %-7s [%4dx%-4d] chunks=%d tiles/chunk=%d | 2-pass/chunk: build=%.3f run=%.3f read+cpu=%.3f total=%.3fms\n",
           shapes[si].name,K,N,nch,tiles,t1b+t2b,t1r+t2r,trc,tot);
    printf("           per-chunk=%.1f us | total per matrix=%.3f ms | submits=%d\n",
           tot*1000/nch, tot, nch*2);
    free(left); free(right); free(acc);
  }
}

int main(void){
  printf("===== patha_kg32_check: Path A INT4 K-aligned, KG=32=G, two-pass =====\n");
  probe_t p;
  if(probe_init(&p,NEURON_SZ)) return 1;
  printf("\n===== Gate ①: pass1 int8 readback (standard matmul, l2g+MemInvld) =====\n");
  gate1_readback(&p,"q",32,896,896,11);      /* safe rshift, KG=32 */
  gate1_readback(&p,"k/v",32,128,128,11);
  gate1_readback(&p,"up",32,4864,896,11);    /* wide N, 6 tiles */
  gate1_readback(&p,"q",32,896,896,5);       /* tighter, some saturate */
  gate1_readback(&p,"down",32,896,896,11);   /* one chunk of down K=4864 */
  part1_rshift(&p);
  part2_twopass(&p);
  part3_latency(&p);
  probe_cleanup(&p);
  printf("===== done =====\n");
  return 0;
}
