/* submit_trigger.c — isolate the "slow state" (~22ms/submit) trigger seen in
 *   submit_budget.c: B2 (prebuilt Run-only) measured 0.115ms, but after the C
 *   two-pass loop (g2l + matmul + invld + CPU read) every submit stays ~21.7ms.
 *
 * Questions:
 *   A. baseline p5 x20                          -> fast (0.1ms)?
 *   B. g2l x1 then p5 x20                       -> does one g2l flip to slow?
 *   C. g2l x1 + p5 + invld + CPU-read once then p5 x20  -> does readback flip it?
 *   D. DeInit/ReInit + rebuild then p5 x20      -> does fresh context reset?
 *   E. l2g-readback-only cmdbuf xN then p5 x20  -> is l2g the trigger?
 *
 * Build: same riscv64 cross as submit_budget.c
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
#define NREP 20

static inline double now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static bmk1822_matrix_lmem_shape_t SL = {.n = 1, .c = 1, .w = 32, .col = 32};
static bmk1822_matrix_lmem_shape_t SR = {.n = 32, .c = 1, .w = 896, .col = 896};
static bmk1822_matrix_lmem_shape_t SO = {.n = 1, .c = 1, .w = 896, .col = 896};

static void build_variants(CVI_RT_HANDLE rt, uint64_t pa, CVI_RT_MEM *var_mem, CVI_RT_MEM *p_ld) {
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

static void build_g2l(CVI_RT_HANDLE rt, uint64_t pa, CVI_RT_MEM *g_ld) {
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
    CVI_RT_LoadDmabuf(rt, gmem, psize + pmu, pa, 0, false, g_ld);
    bmk1822_cleanup(bmk);
}

static void run_avg(const char *tag, CVI_RT_HANDLE rt, CVI_RT_MEM p5, uint64_t pa, int n) {
    double t0 = now();
    for (int i = 0; i < n; i++)
        CVI_RT_RunCmdbufEx(rt, p5, &(CVI_RT_ARRAYBASE){.gaddr_base0 = pa});
    double dt = (now() - t0) / n;
    printf("%-32s : %.3f ms/run  (x15360 -> %.2f s/token)\n", tag, dt * 1e3, dt * 15360);
}

int main(void) {
    printf("===== submit_trigger: slow-state trigger isolation =====\n");
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

    CVI_RT_MEM var_mem[16], p_ld[16], g_ld;
    build_variants(rt, pa, var_mem, p_ld);
    build_g2l(rt, pa, &g_ld);

    /* A. baseline */
    run_avg("A baseline p5 x20", rt, p_ld[5], pa, NREP);

    /* B. g2l x1 then p5 */
    CVI_RT_RunCmdbufEx(rt, g_ld, &(CVI_RT_ARRAYBASE){.gaddr_base0 = pa});
    run_avg("B after 1x g2l", rt, p_ld[5], pa, NREP);

    /* C. g2l + p5 + invld + CPU read once */
    CVI_RT_RunCmdbufEx(rt, g_ld, &(CVI_RT_ARRAYBASE){.gaddr_base0 = pa});
    CVI_RT_RunCmdbufEx(rt, p_ld[5], &(CVI_RT_ARRAYBASE){.gaddr_base0 = pa});
    CVI_RT_MemInvld(rt, mem);
    volatile int8_t sink = ((int8_t*)va)[OUT1_OFF];
    run_avg("C after g2l+p5+inv+read", rt, p_ld[5], pa, NREP);
    (void)sink;

    /* D. DeInit/ReInit + rebuild = fresh context */
    for (int i = 0; i < 16; i++) CVI_RT_MemFree(rt, var_mem[i]);
    CVI_RT_MemFree(rt, mem); CVI_RT_DeInit(rt);
    CVI_RT_Init(&rt);
    mem = CVI_RT_MemAlloc(rt, NEURON_SZ);
    pa = CVI_RT_MemGetPAddr(mem);
    va = CVI_RT_MemGetVAddr(mem);
    CVI_RT_SetBaseReg(rt, 0, pa);
    memset(va, 0, NEURON_SZ);
    for (int i = 0; i < 32 * 896; i++) right[i] = (int8_t)(rand() % 15 - 7);
    for (int i = 0; i < 32; i++) left[i] = (int8_t)(rand() % 200 - 100);
    CVI_RT_MemFlush(rt, mem);
    build_variants(rt, pa, var_mem, p_ld);
    run_avg("D fresh ctx (reinit+rebuild)", rt, p_ld[5], pa, NREP);
    /* E. after fresh ctx, one g2l then p5 again (confirm g2l trigger in fresh ctx) */
    build_g2l(rt, pa, &g_ld);
    CVI_RT_RunCmdbufEx(rt, g_ld, &(CVI_RT_ARRAYBASE){.gaddr_base0 = pa});
    run_avg("E fresh ctx, after 1x g2l", rt, p_ld[5], pa, NREP);

    printf("===== trigger done =====\n");
    for (int i = 0; i < 16; i++) if (var_mem[i]) CVI_RT_MemFree(rt, var_mem[i]);
    CVI_RT_MemFree(rt, mem); CVI_RT_DeInit(rt);
    return 0;
}
