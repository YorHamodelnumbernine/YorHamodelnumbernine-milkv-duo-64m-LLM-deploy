/* submit_trigger3.c — reproduce submit_budget's B->C sequence to find which
 *   step flips the TPU into the ~22ms slow path.
 *
 *   submit_trigger2 showed the clean engine pattern (prebuilt cmdbufs loaded
 *   once, then the two-pass loop) runs at 0.42 ms/block.  submit_budget's C
 *   goes 44 ms/block.  Differences: (X1) var_mem loaded TWICE (B loads p_ld,
 *   C loads p_ld2 again), (X2) a NEW bmk1822_register + g_ld built AFTER
 *   cmdbufs already ran.
 *
 *   Sequence with state checks:
 *     build_variants -> B1 p5 x5 (Run+Invld)  -> check
 *     B2 p5 x5 (Run only)                     -> check
 *     X1 load p_ld2[16] again (double-load)   -> check
 *     X2 register+build g_ld fresh, load      -> check
 *     C 28-block two-pass loop                -> measure + check
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
#define NREP 5

static inline double now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static bmk1822_matrix_lmem_shape_t SL = {.n = 1, .c = 1, .w = 32, .col = 32};
static bmk1822_matrix_lmem_shape_t SR = {.n = 32, .c = 1, .w = 896, .col = 896};
static bmk1822_matrix_lmem_shape_t SO = {.n = 1, .c = 1, .w = 896, .col = 896};

static CVI_RT_MEM var_mem[16], p_ld[16], p_ld2[16], g_ld;
static CVI_RT_HANDLE rt;
static CVI_RT_MEM mem;
static uint64_t pa;
static uint8_t *va;

static void build_variants(void) {
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
}

static void build_g2l(CVI_RT_MEM *out) {
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
    CVI_RT_LoadDmabuf(rt, gmem, psize + pmu, pa, 0, false, out);
    bmk1822_cleanup(bmk);
}

static void state_check(const char *tag) {
    double t0 = now();
    for (int i = 0; i < NREP; i++)
        CVI_RT_RunCmdbufEx(rt, p_ld[5], &(CVI_RT_ARRAYBASE){.gaddr_base0 = pa});
    double dt = (now() - t0) / NREP;
    printf("  state after %-26s: %.3f ms/run  %s\n", tag, dt * 1e3,
           dt < 1e-3 ? "(FAST)" : "(SLOW !)");
}

int main(void) {
    printf("===== submit_trigger3: B->C sequence bisect =====\n");
    CVI_RT_Init(&rt);
    mem = CVI_RT_MemAlloc(rt, NEURON_SZ);
    pa = CVI_RT_MemGetPAddr(mem);
    va = CVI_RT_MemGetVAddr(mem);
    CVI_RT_SetBaseReg(rt, 0, pa);
    memset(va, 0, NEURON_SZ);
    srand(7);
    int8_t *right = (int8_t*)(va + RIGHT_OFF);
    int8_t *left = (int8_t*)(va + 0);
    for (int i = 0; i < 32 * 896; i++) right[i] = (int8_t)(rand() % 15 - 7);
    for (int i = 0; i < 32; i++) left[i] = (int8_t)(rand() % 200 - 100);
    CVI_RT_MemFlush(rt, mem);

    build_variants();
    for (int i = 0; i < 16; i++) CVI_RT_LoadDmabuf(rt, var_mem[i], 65536, pa, 0, false, &p_ld[i]);
    state_check("build_variants+load");

    /* B1: Run+Invld x5 */
    for (int i = 0; i < NREP; i++) {
        CVI_RT_RunCmdbufEx(rt, p_ld[5], &(CVI_RT_ARRAYBASE){.gaddr_base0 = pa});
        CVI_RT_MemInvld(rt, mem);
    }
    state_check("B1 (Run+Invld x5)");

    /* X1: double-load var_mem into p_ld2 (this is what submit_budget C does) */
    for (int i = 0; i < 16; i++) CVI_RT_LoadDmabuf(rt, var_mem[i], 65536, pa, 0, false, &p_ld2[i]);
    state_check("X1 (double-load)");

    /* X2: build g_ld fresh (mid-program register) */
    build_g2l(&g_ld);
    state_check("X2 (fresh g_ld build)");

    /* C: 28-block two-pass */
    {
        double t0 = now();
        for (int blk = 0; blk < 28; blk++) {
            CVI_RT_RunCmdbufEx(rt, g_ld, &(CVI_RT_ARRAYBASE){.gaddr_base0 = pa});
            CVI_RT_RunCmdbufEx(rt, p_ld2[5], &(CVI_RT_ARRAYBASE){.gaddr_base0 = pa});
            CVI_RT_MemInvld(rt, mem);
            int8_t *p1 = (int8_t*)(va + OUT1_OFF);
            int maxabs = 0; for (int n = 0; n < 896; n++) { int a = p1[n]; if (a < 0) a = -a; if (a > maxabs) maxabs = a; }
            long long est = (long long)maxabs << 5; int r = 0; while (est > (127LL << r)) r++;
            CVI_RT_RunCmdbufEx(rt, p_ld2[r], &(CVI_RT_ARRAYBASE){.gaddr_base0 = pa});
            CVI_RT_MemInvld(rt, mem);
        }
        printf("C 28blk two-pass        : %.2f ms/blk\n", (now() - t0) / 28 * 1e3);
    }
    state_check("C (28blk two-pass)");

    printf("===== trigger3 done =====\n");
    for (int i = 0; i < 16; i++) if (var_mem[i]) CVI_RT_MemFree(rt, var_mem[i]);
    CVI_RT_MemFree(rt, mem); CVI_RT_DeInit(rt);
    return 0;
}
