/* pathb_chunk_probe.c — Path B per-chunk INT8 matmul on-board verification
 * (ps32-free exit, CEO task 2026-08-13).
 *
 * Qwen2.5-0.5B shapes (decode M=1):
 *   q/wo: K=896 N=896 ; k/v: K=896 N=128 ; up/gate: K=896 N=4864 ;
 *   down: K=4864 N=896
 *
 * Algorithm under test (matches emu_chunk8.chunk8_rnd, host sim 3/3 @ KG 32..256):
 *   split K into KG-chunks; per chunk (M=1 => "per-row rshift" = one scalar):
 *     pass1: TIU INT8 matmul rshift=rsafe -> int8 [1,N] readback
 *            -> per-row max -> r_opt = rsafe + ceil(log2(max/127))
 *     pass2: TIU INT8 matmul rshift=r_opt -> int8 [1,N] readback
 *     CPU:   acc_f32[n] += p2[n] * 2^r_opt
 *   after all chunks: out[n] = acc[n] * sc_row * lsc_col[n]
 *
 * Parts:
 *   P1  KG / N-tile lmem sweep (bmkernel raw layout)  -> max N-tile per KG
 *   P2  full two-pass per-chunk data flow (KG=256, and KG=128 cross-check)
 *       on q/up/down shapes, verify == host two-pass simulation (bit-exact)
 *   P3  submit latency: build+run per phase, per chunk, per matrix;
 *       full-K baseline (single cmdbuf, fixed rshift) for the same shapes
 *   P4  rshift precision/saturation at the adaptive r values (KG=256)
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

/* xorshift PRNG, deterministic */
static uint32_t rng_state=0x9E3779B9;
static int rng_int(int lo, int hi){          /* [lo,hi] */
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
  printf("  [probe_init] pa=0x%llx va=%p sz=%zu\n",
         (unsigned long long)p->pa, p->va, bytes);
  return 0;
}
static void probe_cleanup(probe_t *p){
  if (p->mem) CVI_RT_MemFree(p->rt, p->mem);
  if (p->rt)  CVI_RT_DeInit(p->rt);
}

/* register+acquire+convert+load+run.  *build_ms = register..acquire (excl run). */
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

/* build one phase (pass1 or pass2) for a single K-chunk:
 * g2l left chunk once + ntiles * [g2l right tile, matmul rshift, l2g out tile]
 * out_off selects P1_OFF (pass1) or P2_OFF (pass2). */
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

/* per-chunk two-pass data flow (M=1).  right must already be placed as padded
 * [Kdim, N_pad] at R_OFF (zero pad cols); left placed as [1,K] at L_OFF.
 * Writes acc (pre-scale, res*2^r summed) into acc[N]. */
static int two_pass_matmul(probe_t *p,
                           const int8_t *left, int K,
                           const int8_t *right, int Kdim, int N,
                           int KG, int Ntile, int rsafe,
                           float *acc,                       /* [N] out */
                           double *t_p1b, double *t_p1r,
                           double *t_p2b, double *t_p2r,
                           double *t_read_cpu,
                           int *r_opt_min, int *r_opt_max){
  int N_pad=((N+Ntile-1)/Ntile)*Ntile;
  int ntiles=N_pad/Ntile;
  int nchunks=(K+KG-1)/KG;
  /* place padded right in neuron mem */
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

/* full-K baseline: one cmdbuf, all N-tiles, fixed rshift (current engine style).
 * right padded [Kdim,N_pad] at R_OFF, left [1,K] at L_OFF. */
static int fullk_matmul(probe_t *p,
                        const int8_t *left, int K,
                        const int8_t *right, int Kdim, int N,
                        int rshift, double *t_build, double *t_run){
  int Ntile=32;
  /* max Ntile s.t. right[K,Ntile]+left[1,K]+res[1,Ntile] <= ~30KB */
  while((int)(K*Ntile + K + Ntile) > 30000 && Ntile>1) Ntile--;
  int N_pad=((N+Ntile-1)/Ntile)*Ntile;
  int ntiles=N_pad/Ntile;
  memset(p->va+R_OFF,0,(size_t)Kdim*N_pad);
  for(int r=0;r<Kdim;r++) memcpy(p->va+R_OFF+(size_t)r*N_pad, right+(size_t)r*N, (size_t)N);
  memcpy(p->va+L_OFF,left,(size_t)K);
  CVI_RT_MemFlush(p->rt,p->mem);

  bmk_info_t info={.chip_version=1822,.cmdbuf_size=sizeof(p->cmdbuf1),.cmdbuf=p->cmdbuf1};
  double tb0=now_ms();
  bmk1822_context_t *bmk=bmk1822_register(&info);
  bmk1822_matrix_lmem_shape_t sl={.n=1,.c=1,.w=(uint32_t)K,.col=(uint32_t)K};
  bmk1822_matrix_lmem_shape_t sr={.n=(uint32_t)K,.c=1,.w=(uint32_t)Ntile,.col=(uint32_t)Ntile};
  bmk1822_matrix_lmem_shape_t so={.n=1,.c=1,.w=(uint32_t)Ntile,.col=(uint32_t)Ntile};
  bmk1822_matrix_lmem_t *ml_l=bmk1822_lmem_alloc_matrix(bmk,sl,FMT_I8,1);
  bmk1822_matrix_lmem_t *ml_r=bmk1822_lmem_alloc_matrix(bmk,sr,FMT_I8,1);
  bmk1822_matrix_lmem_t *ml_res=bmk1822_lmem_alloc_matrix(bmk,so,FMT_I8,1);
  if(!ml_l||!ml_r||!ml_res){ printf("  [fullk] lmem alloc FAIL K=%d Ntile=%d\n",K,Ntile);
    bmk1822_cleanup(bmk); return -1; }
  bmk1822_matrix_tgmem_t mg_l={0,L_OFF,FMT_I8,{1,(uint32_t)K},{(uint32_t)K}};
  bmk1822_tdma_g2l_matrix_copy(bmk,&(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_l,ml_l});
  for(int t=0;t<ntiles;t++){
    uint64_t ns=(uint64_t)t*Ntile;
    bmk1822_matrix_tgmem_t mg_r={0,R_OFF+ns,FMT_I8,{(uint32_t)K,(uint32_t)Ntile},{(uint32_t)N_pad}};
    bmk1822_tdma_g2l_matrix_copy(bmk,&(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_r,ml_r});
    bmk1822_tiu_matrix_multiplication_param_t pm={
      .res=ml_res,.left=ml_l,.right=ml_r,.bias=NULL,.lshift_bits=0,
      .rshift_bits=(uint8_t)rshift,.res_is_int8=1,.relu_enable=0,.add_result=0,
      .ps32_mode=0,.layer_id=1};
    if(!bmk1822_tiu_matrix_multiplication(bmk,&pm)) printf("  [fullk] REJECTED t=%d\n",t);
    bmk1822_matrix_tgmem_t mg_o={0,P1_OFF+ns,FMT_I8,{1,(uint32_t)Ntile},{(uint32_t)Ntile}};
    bmk1822_tdma_l2g_matrix_copy(bmk,&(bmk1822_tdma_l2tg_matrix_copy_param_t){ml_res,&mg_o});
  }
  double bms; double rms=run_bmk(p,bmk,&bms);
  bmk1822_cleanup(bmk);
  *t_build=bms; *t_run=rms;
  return ntiles;
}

/* ---------------- Part 1: KG / N-tile sweep ---------------- */
static void part1_sweep(probe_t *p){
  printf("\n===== P1: KG / N-tile lmem sweep (bmkernel raw, M=1) =====\n");
  int kvs[][2]={{32,1},{64,1},{128,1},{256,1},{256,2},{512,1}};
  for(int i=0;i<6;i++){
    int KG=kvs[i][0], seed=kvs[i][1];
    printf("  KG=%d:",KG);
    int Ntiles[]={32,48,64,80,96,112,128,192,224,256,384,448,512,896};
    for(int j=0;j<14;j++){
      int Nt=Ntiles[j];
      int left_bytes=(int)(1*KG + 8)&~7, right_bytes=KG*Nt, res_bytes=(int)(Nt+8)&~7;
      if(left_bytes+right_bytes+res_bytes>32000) continue; /* skip over-size */
      /* build+verify one matmul */
      memset(p->va+L_OFF,0,(size_t)KG+64);
      memset(p->va+R_OFF,0,(size_t)KG*Nt);
      srand(KG*1000+Nt+seed);
      int8_t *host=(int8_t*)malloc(Nt);
      for(int k=0;k<KG;k++) p->va[L_OFF+k]=(int8_t)(rand()%200-100);
      for(int n=0;n<Nt;n++){ int32_t acc=0;
        for(int k=0;k<KG;k++){ int8_t rv=(int8_t)(rand()%200-100); p->va[R_OFF+(size_t)k*Nt+n]=rv; acc+=(int32_t)(int8_t)p->va[L_OFF+k]*rv; }
        host[n]=ref_div(acc,8); }
      CVI_RT_MemFlush(p->rt,p->mem);
      bmk_info_t info={.chip_version=1822,.cmdbuf_size=sizeof(p->cmdbuf1),.cmdbuf=p->cmdbuf1};
      bmk1822_context_t *bmk=bmk1822_register(&info);
      bmk1822_matrix_lmem_shape_t sl={.n=1,.c=1,.w=(uint32_t)KG,.col=(uint32_t)KG};
      bmk1822_matrix_lmem_shape_t sr={.n=(uint32_t)KG,.c=1,.w=(uint32_t)Nt,.col=(uint32_t)Nt};
      bmk1822_matrix_lmem_shape_t so={.n=1,.c=1,.w=(uint32_t)Nt,.col=(uint32_t)Nt};
      bmk1822_matrix_lmem_t *ml_l=bmk1822_lmem_alloc_matrix(bmk,sl,FMT_I8,1);
      bmk1822_matrix_lmem_t *ml_r=bmk1822_lmem_alloc_matrix(bmk,sr,FMT_I8,1);
      bmk1822_matrix_lmem_t *ml_res=bmk1822_lmem_alloc_matrix(bmk,so,FMT_I8,1);
      if(!ml_l||!ml_r||!ml_res){ free(host); bmk1822_cleanup(bmk); continue; }
      uint32_t lr=bmk1822_lmem_matrix_to_size(bmk,sr,FMT_I8,1);
      bmk1822_matrix_tgmem_t mg_l={0,L_OFF,FMT_I8,{1,(uint32_t)KG},{(uint32_t)KG}};
      bmk1822_tdma_g2l_matrix_copy(bmk,&(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_l,ml_l});
      bmk1822_matrix_tgmem_t mg_r={0,R_OFF,FMT_I8,{(uint32_t)KG,(uint32_t)Nt},{(uint32_t)Nt}};
      bmk1822_tdma_g2l_matrix_copy(bmk,&(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_r,ml_r});
      bmk1822_tiu_matrix_multiplication_param_t pm={
        .res=ml_res,.left=ml_l,.right=ml_r,.bias=NULL,.lshift_bits=0,
        .rshift_bits=8,.res_is_int8=1,.relu_enable=0,.add_result=0,.ps32_mode=0,.layer_id=1};
      bmk1822_tiu_matrix_multiplication(bmk,&pm);
      bmk1822_matrix_tgmem_t mg_o={0,P1_OFF,FMT_I8,{1,(uint32_t)Nt},{(uint32_t)Nt}};
      bmk1822_tdma_l2g_matrix_copy(bmk,&(bmk1822_tdma_l2tg_matrix_copy_param_t){ml_res,&mg_o});
      double bms; double rms=run_bmk(p,bmk,&bms);
      bmk1822_cleanup(bmk);
      CVI_RT_MemInvld(p->rt,p->mem);
      int8_t *r=(int8_t*)(p->va+P1_OFF);
      int bad=0; for(int n=0;n<Nt;n++) if(r[n]!=host[n]) bad++;
      printf(" N=%d(%s)%s",Nt,bad==0?"ok":"BAD","");
      free(host);
    }
    printf("\n");
  }
}

/* ---------------- Part 2: full two-pass data flow ---------------- */
static void part2_twopass(probe_t *p){
  printf("\n===== P2: full two-pass per-chunk data flow (KG=256, KG=128) =====\n");
  struct { const char *name; int K,N; } shapes[]={
    {"q/wo",896,896},{"k/v",896,128},{"up/gate",896,4864},{"down",4864,896}};
  int KGs[]={256,128};
  for(int si=0;si<4;si++){
    for(int ki=0;ki<2;ki++){
      int K=shapes[si].K, N=shapes[si].N, KG=KGs[ki];
      int Ntile=(KG==256)?112:(KG==128?192:224);
      /* check lmem fits; shrink if needed */
      while((int)(KG*Ntile + KG + Ntile) > 30000) Ntile-=16;
      int N_pad=((N+Ntile-1)/Ntile)*Ntile;
      int rsafe=ceil_log2_div(KG*80*30,127)+1;   /* bound: |left|<=80,|right|<=30 */
      /* data */
      int8_t *left=malloc(K), *right=malloc((size_t)K*N);
      float *lsc=malloc(N*sizeof(float));
      rng_state=1000+si*10+ki;
      for(int k=0;k<K;k++) left[k]=(int8_t)rng_int(-80,80);
      for(int n=0;n<N;n++){ lsc[n]=1.0f/(float)rng_int(5,60);   /* per-channel wt scale */
        for(int k=0;k<K;k++) right[(size_t)k*N+n]=(int8_t)rng_int(-30,30); }
      /* host references */
      float *ref_exact=calloc(N,sizeof(float)), *ref_tp=calloc(N,sizeof(float));
      int32_t *acc=malloc(N*sizeof(int32_t)); int8_t *p1=malloc(N);
      for(int g=0;g<K;g+=KG){
        int KC=(g+KG<=K)?KG:(K-g);
        for(int n=0;n<N;n++){ int32_t s=0;
          for(int k=0;k<KC;k++) s+=(int32_t)left[g+k]*right[(size_t)(g+k)*N+n];
          acc[n]=s; }
        int mx=0; for(int n=0;n<N;n++){int a=acc[n]; if(a<0)a=-a; if(a>mx)mx=a;}
        int r_exact=(mx>0)?ceil_log2_div(mx,127):0;
        int mx1=0; for(int n=0;n<N;n++){ p1[n]=ref_div(acc[n],rsafe); int a=p1[n]; if(a<0)a=-a; if(a>mx1)mx1=a; }
        int r_tp=(mx1>0)?(rsafe+ceil_log2_div(mx1,127)):0;
        for(int n=0;n<N;n++){
          ref_exact[n]+=(float)((int)ref_div(acc[n],r_exact)*(1<<r_exact));
          ref_tp[n]+=(float)((int)ref_div(acc[n],r_tp)*(1<<r_tp));
        }
      }
      for(int n=0;n<N;n++){ ref_exact[n]*=lsc[n]; ref_tp[n]*=lsc[n]; }
      /* on-board */
      float *acc_on=calloc(N,sizeof(float));
      double t1b=0,t1r=0,t2b=0,t2r=0,trc=0; int rmin,rmax;
      int nch=two_pass_matmul(p,left,K,right,K,N,KG,Ntile,rsafe,acc_on,
                              &t1b,&t1r,&t2b,&t2r,&trc,&rmin,&rmax);
      for(int n=0;n<N;n++) acc_on[n]*=lsc[n];   /* sc_row=1.0 in probe */
      /* verify: on-board vs ref_tp (should be bit-exact), and vs ref_exact */
      int bad_tp=0,bad_ex=0; double maxrel_tp=0,maxrel_ex=0;
      for(int n=0;n<N;n++){
        if(acc_on[n]!=ref_tp[n]){ bad_tp++; }
        double d=ref_tp[n]-ref_exact[n]; if(d<0)d=-d;
        double rel=(fabsf(ref_exact[n])>1e-6)?d/fabsf(ref_exact[n]):0;
        if(rel>maxrel_ex)maxrel_ex=rel;
        if(acc_on[n]!=ref_tp[n]){ double d2=acc_on[n]-ref_tp[n]; if(d2<0)d2=-d2;
          double rel2=(fabsf(ref_tp[n])>1e-6)?d2/fabsf(ref_tp[n]):0; if(rel2>maxrel_tp)maxrel_tp=rel2; }
      }
      printf("  %-7s KG=%-3d Ntile=%-3d Npad=%-4d rsafe=%-2d chunks=%d r_opt[%d..%d] t_p1=%.3fms(t_b %.3f/t_r %.3f) t_p2=%.3fms(t_b %.3f/t_r %.3f) t_rdcpu=%.3fms\n",
             shapes[si].name,KG,Ntile,N_pad,rsafe,nch,rmin,rmax,
             t1b+t1r,t1b,t1r,t2b+t2r,t2b,t2r,trc);
      printf("           vs host-two-pass: %s (%d/%d mism, maxrel=%.2e)  |  exact-vs-twopass gap maxrel=%.2e\n",
             bad_tp==0?"BIT-EXACT":"MISMATCH",N-bad_tp,N,maxrel_tp,maxrel_ex);
      free(left); free(right); free(lsc); free(ref_exact); free(ref_tp); free(acc); free(p1); free(acc_on);
    }
  }
}

/* ---------------- Part 3: latency + full-K baseline ---------------- */
static void part3_latency(probe_t *p){
  printf("\n===== P3: per-chunk submit latency + full-K baseline =====\n");
  struct { const char *name; int K,N; } shapes[]={
    {"q",896,896},{"k",896,128},{"up",896,4864},{"down",4864,896}};
  int KG=256, Ntile=112;
  for(int si=0;si<4;si++){
    int K=shapes[si].K, N=shapes[si].N;
    while((int)(KG*Ntile+KG+Ntile)>30000) Ntile-=16;
    int rsafe=ceil_log2_div(KG*80*30,127)+1;
    int8_t *left=malloc(K), *right=malloc((size_t)K*N);
    rng_state=2000+si;
    for(int k=0;k<K;k++) left[k]=(int8_t)rng_int(-80,80);
    for(int n=0;n<N;n++) for(int k=0;k<K;k++) right[(size_t)k*N+n]=(int8_t)rng_int(-30,30);
    float *acc=calloc(N,sizeof(float));
    double t1b=0,t1r=0,t2b=0,t2r=0,trc=0; int rmin,rmax;
    /* warm-up once (also allocs/init) */
    two_pass_matmul(p,left,K,right,K,N,KG,Ntile,rsafe,acc,&t1b,&t1r,&t2b,&t2r,&trc,&rmin,&rmax);
    /* measured run */
    memset(acc,0,N*sizeof(float));
    t1b=t1r=t2b=t2r=trc=0;
    int nch=two_pass_matmul(p,left,K,right,K,N,KG,Ntile,rsafe,acc,&t1b,&t1r,&t2b,&t2r,&trc,&rmin,&rmax);
    double tot=t1b+t1r+t2b+t2r+trc;
    double per_chunk=tot/nch;
    /* full-K baseline */
    int rshift_fk=(K==4864)?15:12;   /* matmul_rshift(K)-5 */
    double fkb=0,fkr=0;
    int fk_nt=fullk_matmul(p,left,K,right,K,N,rshift_fk,&fkb,&fkr);
    double fk_tot=fkb+fkr;
    printf("  %-5s [%4dx%-4d] chunks=%d tiles/chunk=%d\n",shapes[si].name,K,N,nch,(N+Ntile-1)/Ntile);
    printf("     per-chunk 2-pass: build=%.3fms run=%.3fms read+cpu=%.3fms  total=%.3fms  (%.1f us/chunk)\n",
           t1b+t2b, t1r+t2r, trc, tot, per_chunk*1000);
    if(fk_nt<0){
      printf("     full-K baseline: INFEASIBLE (K=%d right tile cannot fit lmem; per-chunk K-split required)\n",K);
      printf("     per-chunk is the ONLY viable layout for this K (no baseline ratio)\n");
    } else {
      printf("     full-K baseline (rshift=%d, ntiles=%d): build=%.3fms run=%.3fms total=%.3fms  (%.1f us/matmul)\n",
             rshift_fk, fk_nt, fkb, fkr, fk_tot, fk_tot*1000);
      printf("     per-chunk overhead ratio: %.1fx full-K time  (per-chunk submit count %d vs full-K %d)\n",
             (fk_tot>0?tot/fk_tot:0), nch*2, fk_nt);
    }
    free(left); free(right); free(acc);
  }
}

/* ---------------- Part 4: rshift precision/saturation at KG=256 ---------------- */
static void part4_rshift(probe_t *p){
  printf("\n===== P4: TIU rshift precision/saturation at KG=256 =====\n");
  int rs[]={1,2,4,8,12,15};
  for(int i=0;i<6;i++){
    int rsh=rs[i];
    int KG=256, Nt=112;
    memset(p->va+L_OFF,0,KG+64);
    memset(p->va+R_OFF,0,(size_t)KG*Nt);
    rng_state=3000+i;
    for(int k=0;k<KG;k++) p->va[L_OFF+k]=(int8_t)rng_int(-80,80);
    int8_t *host=malloc(Nt);
    for(int n=0;n<Nt;n++){ int32_t acc=0;
      for(int k=0;k<KG;k++){ int8_t rv=(int8_t)rng_int(-30,30); p->va[R_OFF+(size_t)k*Nt+n]=rv; acc+=(int32_t)(int8_t)p->va[L_OFF+k]*rv; }
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
    int bad=0, sat=0; for(int n=0;n<Nt;n++){ if(r[n]!=host[n])bad++; if(r[n]==127||r[n]==-128)sat++; }
    printf("  rshift=%2d K=256 N=%d: ok=%d/%d bad=%d sat8=%d | r0..7=%d %d %d %d %d %d %d %d\n",
           rsh,Nt,Nt-bad,Nt,bad,sat,r[0],r[1],r[2],r[3],r[4],r[5],r[6],r[7]);
    free(host);
  }
}

int main(void){
  printf("===== pathb_chunk_probe: Path B per-chunk INT8 matmul (ps32-free) =====\n");
  probe_t p;
  if(probe_init(&p,NEURON_SZ)) return 1;
  part1_sweep(&p);
  part2_twopass(&p);
  part3_latency(&p);
  part4_rshift(&p);
  probe_cleanup(&p);
  printf("===== done =====\n");
  return 0;
}
