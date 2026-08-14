/* submit_trigger2.c — bisect the slow-state (~22ms/submit) trigger.
 *   submit_trigger confirmed a SINGLE g2l+p5+inv+read stays fast (0.11ms).
 *   submit_budget C (28-blk loop of g2l/pass1/inv/read/pass2/inv) goes slow
 *   (~22ms).  This probe replicates the loop with toggles to find the trigger.
 *
 * Phases (each followed by p5 x10 state check):
 *   F1: 28 blk C-exact (g2l+p5+inv+read+p_ld[r]+inv)   -> confirm slow
 *   F2: 1  blk C-exact                                  -> threshold
 *   F3: 8  blk C-exact                                  -> threshold
 *   F4: 28 blk g2l+p5 only (no inv/read/pass2)          -> need inv/read?
 *   F5: 28 blk p5 only (no g2l)                         -> need g2l at all?
 *   F6: 28 blk g2l only (no matmul)                     -> g2l accumulate alone?
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include "cviruntime_context.h"
#include "bmkernel/bm1822/bmkernel_1822.h"

#define NEURON_SZ 262144
#define RIGHT_OFF 4096
#define OUT1_OFF  65536
#define NREP 10

static inline double now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static bmk1822_matrix_lmem_shape_t SL = {.n = 1, .c = 1, .w = 32, .col = 32};
static bmk1822_matrix_lmem_shape_t SR = {.n = 32, .c = 1, .w = 896, .col = 896};
static bmk1822_matrix_lmem_shape_t SO = {.n = 1, .c = 1, .w = 896, .col = 896};

static CVI_RT_MEM var_mem[16], p_ld[16], g_ld;

static void build_variants(CVI_RT_HANDLE rt, uint64_t pa) {
    for (int i = 0; i < 16; i++) {
        uint8_t cmdbuf[65536] __attribute__((aligned(16)));
        bmk_info_t info = { .chip_version = 1822, .cmdbuf_size = sizeof(cmdbuf), .cmdbuf = cmdbuf };
        bmk1822_context_t *bmk = bmk1822_register(&info);
        bmk1822_matrix_lmem_t *ml_l = bmk1822_lmem_alloc_matrix(bmk, SL, FMT_I8, 1);
        bmk1822_matrix_lmem_t *ml_r = bmk1822_lmem_alloc_matrix(bmk, SR, FMT_I8, 1);
        bmk1822_matrix_lmem_t *ml_o = bmk1822_lmem_alloc_matrix(bmk, SO, FMT_I8, 1);
        bmk1822_matrix_tgmem_t mg_o = {0, OUT1_OFF, FMT_I8, {1, 896}, {896}};
        bmk1822_tiu_matrix_multiplication_param_t mm = {
            .res = ml_o, .left = ml_l, .right = ml_r, .bias = NULL,
            .lshift_bits = 0, .rshift_bits = (uint8_t)i, .res_is_int8 = 1, .relu_enable = 0,
            .add_result = 0, .ps32_mode = 0, .layer_id = 1 };
        bmk1822_tiu_matrix_multiplication(bmk, &mm);
        bmk1822_tdma_l2g_matrix_copy(bmk, &(bmk1822_tdma_l2tg_matrix_copy_param_t){ml_o, &mg_o});
        uint32_t cmd_sz; uint8_t *cmd = bmk1822_acquire_cmdbuf(bmk, &cmd_sz);
        uint32_t psize, pmu; bmk1822_dmabuf_size(cmd, cmd_sz, &psize, &pmu);
        var_mem[i] = CVI_RT_MemAlloc(rt, psize + pmu);
        uint8_t *db = CVI_RT_MemGetVAddr(var_mem[i]);
        bmk1822_dmabuf_convert(cmd, cmd_sz, db);
        bmk1822_arraybase_set(db, pa, 0, 0, 0);
        CVI_RT_MemFlush(rt, var_mem[i]);
        bmk1822_cleanup(bmk);
    }
    for (int i = 0; i < 16; i++) CVI_RT_LoadDmabuf(rt, var_mem[i], 65536, pa, 0, false, &p_ld[i]);
}

static void build_g2l(CVI_RT_HANDLE rt, uint64_t pa) {
    uint8_t cmdbuf[65536] __attribute__((aligned(16)));
    bmk_info_t info = { .chip_version = 1822, .cmdbuf_size = sizeof(cmdbuf), .cmdbuf = cmdbuf };
    bmk1822_context_t *bmk = bmk1822_register(&info);
    bmk1822_matrix_lmem_t *ml_l = bmk1822_lmem_alloc_matrix(bmk, SL, FMT_I8, 1);
    bmk1822_matrix_lmem_t *ml_r = bmk1822_lmem_alloc_matrix(bmk, SR, FMT_I8, 1);
    bmk1822_matrix_tgmem_t mg_l = {0, 0, FMT_I8, {1, 32}, {32}};
    bmk1822_matrix_tgmem_t mg_r = {0, RIGHT_OFF, FMT_I8, {32, 896}, {896}};
    bmk1822_tdma_g2l_matrix_copy(bmk, &(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_l, ml_l});
    bmk1822_tdma_g2l_matrix_copy(bmk, &(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_r, ml_r});
    uint32_t cmd_sz; uint8_t *cmd = bmk1822_acquire_cmdbuf(bmk, &cmd_sz);
    uint32_t psize, pmu; bmk1822_dmabuf_size(cmd, cmd_sz, &psize, &pmu);
    CVI_RT_MEM gmem = CVI_RT_MemAlloc(rt, psize + pmu);
    uint8_t *db = CVI_RT_MemGetVAddr(gmem);
    bmk1822_dmabuf_convert(cmd, cmd_sz, db);
    bmk1822_arraybase_set(db, pa, 0, 0, 0);
    CVI_RT_MemFlush(rt, gmem);
    CVI_RT_LoadDmabuf(rt, gmem, psize + pmu, pa, 0, false, &g_ld);
    bmk1822_cleanup(bmk);
}

/* run a pattern of n blocks; mode: 0=C-exact, 1=g2l+p5 only, 2=p5 only, 3=g2l only */
static void blk_loop(CVI_RT_HANDLE rt, uint64_t pa, int n, int mode, double *per_blk) {
    double t0 = now();
    for (int blk = 0; blk < n; blk++) {
        if (mode != 2) CVI_RT_RunCmdbufEx(rt, g_ld, &(CVI_RT_ARRAYBASE){.gaddr_base0 = pa});
        if (mode != 3) CVI_RT_RunCmdbufEx(rt, p_ld[5], &(CVI_RT_ARRAYBASE){.gaddr_base0 = pa});
        if (mode == 0) {
            CVI_RT_MemInvld(rt, (CVI_RT_MEM)0); /* placeholder, replaced below */
        }
    }
    *per_blk = (now() - t0) / n;
}

static void state_check(const char *tag, CVI_RT_HANDLE rt, uint64_t pa) {
    double t0 = now();
    for (int i = 0; i < NREP; i++)
        CVI_RT_RunCmdbufEx(rt, p_ld[5], &(CVI_RT_ARRAYBASE){.gaddr_base0 = pa});
    double dt = (now() - t0) / NREP;
    printf("  state after %-28s: %.3f ms/run  %s\n", tag, dt * 1e3,
           dt < 1e-3 ? "(FAST)" : "(SLOW !)");
}

int main(void) {
    printf("===== submit_trigger2: bisect slow-state trigger =====\n");
    CVI_RT_HANDLE rt; CVI_RT_Init(&rt);
    CVI_RT_MEM mem = CVI_RT_MemAlloc(rt, NEURON_SZ);
    uint64_t pa = CVI_RT_MemGetPAddr(mem);
    uint8_t *va = CVI_RT_MemGetVAddr(mem);
    CVI_RT_SetBaseReg(rt, 0, pa);
    memset(va, 0, NEURON_SZ);
    srand(7);
    int8_t *right = (int8_t*)(va + RIGHT_OFF);
    int8_t *left = (int8_t*)(va + 0);
    for (int i = 0; i < 32 * 896; i++) right[i] = (int8_t)(rand() % 15 - 7);
    for (int i = 0; i < 32; i++) left[i] = (int8_t)(rand() % 200 - 100);
    CVI_RT_MemFlush(rt, mem);

    build_variants(rt, pa);
    build_g2l(rt, pa);

    /* NOTE: mode==0 exact-C needs MemInvld + CPU read; emulate with the real
       mem handle.  We pass mem via a global to keep blk_loop simple. */
    extern void *g_mem; (void)g_mem;

    state_check("baseline (pre-loop)", rt, pa);

    double per_blk;
    /* F1 exact-C 28 blocks */
    {
        double t0 = now();
        for (int blk = 0; blk < 28; blk++) {
            CVI_RT_RunCmdbufEx(rt, g_ld, &(CVI_RT_ARRAYBASE){.gaddr_base0 = pa});
            CVI_RT_RunCmdbufEx(rt, p_ld[5], &(CVI_RT_ARRAYBASE){.gaddr_base0 = pa});
            CVI_RT_MemInvld(rt, mem);
            int8_t *p1 = (int8_t*)(va + OUT1_OFF);
            int maxabs = 0; for (int n = 0; n < 896; n++) { int a = p1[n]; if (a < 0) a = -a; if (a > maxabs) maxabs = a; }
            long long est = (long long)maxabs << 5; int r = 0; while (est > (127LL << r)) r++;
            CVI_RT_RunCmdbufEx(rt, p_ld[r], &(CVI_RT_ARRAYBASE){.gaddr_base0 = pa});
            CVI_RT_MemInvld(rt, mem);
        }
        printf("F1 28blk C-exact        : %.2f ms/blk\n", (now() - t0) / 28 * 1e3);
    }
    state_check("F1 (28blk C-exact)", rt, pa);

    /* F2 1 blk C-exact */
    {
        double t0 = now();
        CVI_RT_RunCmdbufEx(rt, g_ld, &(CVI_RT_ARRAYBASE){.gaddr_base0 = pa});
        CVI_RT_RunCmdbufEx(rt, p_ld[5], &(CVI_RT_ARRAYBASE){.gaddr_base0 = pa});
        CVI_RT_MemInvld(rt, mem);
        int8_t *p1 = (int8_t*)(va + OUT1_OFF);
        int maxabs = 0; for (int n = 0; n < 896; n++) { int a = p1[n]; if (a < 0) a = -a; if (a > maxabs) maxabs = a; }
        long long est = (long long)maxabs << 5; int r = 0; while (est > (127LL << r)) r++;
        CVI_RT_RunCmdbufEx(rt, p_ld[r], &(CVI_RT_ARRAYBASE){.gaddr_base0 = pa});
        CVI_RT_MemInvld(rt, mem);
        printf("F2 1blk C-exact         : %.2f ms/blk\n", (now() - t0) * 1e3);
    }
    state_check("F2 (1blk C-exact)", rt, pa);

    /* F3 8 blk C-exact */
    {
        double t0 = now();
        for (int blk = 0; blk < 8; blk++) {
            CVI_RT_RunCmdbufEx(rt, g_ld, &(CVI_RT_ARRAYBASE){.gaddr_base0 = pa});
            CVI_RT_RunCmdbufEx(rt, p_ld[5], &(CVI_RT_ARRAYBASE){.gaddr_base0 = pa});
            CVI_RT_MemInvld(rt, mem);
            int8_t *p1 = (int8_t*)(va + OUT1_OFF);
            int maxabs = 0; for (int n = 0; n < 896; n++) { int a = p1[n]; if (a < 0) a = -a; if (a > maxabs) maxabs = a; }
            long long est = (long long)maxabs << 5; int r = 0; while (est > (127LL << r)) r++;
            CVI_RT_RunCmdbufEx(rt, p_ld[r], &(CVI_RT_ARRAYBASE){.gaddr_base0 = pa});
            CVI_RT_MemInvld(rt, mem);
        }
        printf("F3 8blk C-exact         : %.2f ms/blk\n", (now() - t0) / 8 * 1e3);
    }
    state_check("F3 (8blk C-exact)", rt, pa);

    /* F4 28 blk g2l+p5 only */
    {
        double t0 = now();
        for (int blk = 0; blk < 28; blk++) {
            CVI_RT_RunCmdbufEx(rt, g_ld, &(CVI_RT_ARRAYBASE){.gaddr_base0 = pa});
            CVI_RT_RunCmdbufEx(rt, p_ld[5], &(CVI_RT_ARRAYBASE){.gaddr_base0 = pa});
        }
        printf("F4 28blk g2l+p5        : %.2f ms/blk\n", (now() - t0) / 28 * 1e3);
    }
    state_check("F4 (28blk g2l+p5)", rt, pa);

    /* F5 28 blk p5 only */
    {
        double t0 = now();
        for (int blk = 0; blk < 28; blk++)
            CVI_RT_RunCmdbufEx(rt, p_ld[5], &(CVI_RT_ARRAYBASE){.gaddr_base0 = pa});
        printf("F5 28blk p5-only       : %.2f ms/blk\n", (now() - t0) / 28 * 1e3);
    }
    state_check("F5 (28blk p5-only)", rt, pa);

    /* F6 28 blk g2l only */
    {
        double t0 = now();
        for (int blk = 0; blk < 28; blk++)
            CVI_RT_RunCmdbufEx(rt, g_ld, &(CVI_RT_ARRAYBASE){.gaddr_base0 = pa});
        printf("F6 28blk g2l-only      : %.2f ms/blk\n", (now() - t0) / 28 * 1e3);
    }
    state_check("F6 (28blk g2l-only)", rt, pa);

    printf("===== trigger2 done =====\n");
    for (int i = 0; i < 16; i++) if (var_mem[i]) CVI_RT_MemFree(rt, var_mem[i]);
    CVI_RT_MemFree(rt, mem); CVI_RT_DeInit(rt);
    return 0;
}
