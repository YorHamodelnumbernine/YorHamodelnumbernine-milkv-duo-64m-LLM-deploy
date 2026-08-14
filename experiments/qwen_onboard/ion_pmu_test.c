/* ion_pmu_test.c — does LoadDmabuf(enable_pmu=false) work when the source
 * dmabuf is allocated psize-only (NO pmu region)?  If yes, per-cmdbuf ION cost
 * drops from ~1MB (pmu) to ~640B, unlocking large prebuilt pools.
 *
 * Builds ONE pass cmdbuf (matmul+l2g, rshift=5, N=896), allocates psize-only,
 * LoadDmabuf + Run, then reads back p1 and checks a known element.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "cviruntime_context.h"
#include "bmkernel/bm1822/bmkernel_1822.h"

#define NEURON_SZ 262144
#define ACTQ_OFF  0
#define DQ_OFF    4096
#define P1_OFF    65536

static bmk1822_matrix_lmem_shape_t SL = {.n = 1, .c = 1, .w = 32, .col = 32};
static bmk1822_matrix_lmem_shape_t SR = {.n = 32, .c = 1, .w = 896, .col = 896};
static bmk1822_matrix_lmem_shape_t SO = {.n = 1, .c = 1, .w = 896, .col = 896};

int main(void) {
    printf("== ion_pmu_test ==\n"); fflush(stdout);
    CVI_RT_HANDLE rt; CVI_RT_Init(&rt);
    CVI_RT_MEM mem = CVI_RT_MemAlloc(rt, NEURON_SZ);
    uint64_t pa = CVI_RT_MemGetPAddr(mem);
    uint8_t *va = CVI_RT_MemGetVAddr(mem);
    CVI_RT_SetBaseReg(rt, 0, pa);
    memset(va, 0, NEURON_SZ);

    /* synthetic left/right */
    int8_t *left = (int8_t*)(va + ACTQ_OFF);
    int8_t *right = (int8_t*)(va + DQ_OFF);
    for (int i = 0; i < 32; i++) left[i] = (int8_t)(i - 16);
    for (int i = 0; i < 32 * 896; i++) right[i] = (int8_t)(i % 15 - 7);
    CVI_RT_MemFlush(rt, mem);

    /* build g2l + pass1 in ONE cmdbuf */
    uint8_t cmdbuf[65536] __attribute__((aligned(16)));
    bmk_info_t info = { .chip_version = 1822, .cmdbuf_size = sizeof(cmdbuf), .cmdbuf = cmdbuf };
    bmk1822_context_t *bmk = bmk1822_register(&info);
    bmk1822_matrix_lmem_t *ml_l = bmk1822_lmem_alloc_matrix(bmk, SL, FMT_I8, 1);
    bmk1822_matrix_lmem_t *ml_r = bmk1822_lmem_alloc_matrix(bmk, SR, FMT_I8, 1);
    bmk1822_matrix_lmem_t *ml_o = bmk1822_lmem_alloc_matrix(bmk, SO, FMT_I8, 1);
    bmk1822_matrix_tgmem_t mg_l = {0, ACTQ_OFF, FMT_I8, {1, 32}, {32}};
    bmk1822_matrix_tgmem_t mg_r = {0, DQ_OFF, FMT_I8, {32, 896}, {896}};
    bmk1822_tdma_g2l_matrix_copy(bmk, &(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_l, ml_l});
    bmk1822_tdma_g2l_matrix_copy(bmk, &(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_r, ml_r});
    bmk1822_matrix_tgmem_t mg_o = {0, P1_OFF, FMT_I8, {1, 896}, {896}};
    bmk1822_tiu_matrix_multiplication_param_t mm = {
        .res = ml_o, .left = ml_l, .right = ml_r, .bias = NULL,
        .lshift_bits = 0, .rshift_bits = 5, .res_is_int8 = 1, .relu_enable = 0,
        .add_result = 0, .ps32_mode = 0, .layer_id = 1 };
    bmk1822_tiu_matrix_multiplication(bmk, &mm);
    bmk1822_tdma_l2g_matrix_copy(bmk, &(bmk1822_tdma_l2tg_matrix_copy_param_t){ml_o, &mg_o});
    uint32_t cmd_sz; uint8_t *cmd = bmk1822_acquire_cmdbuf(bmk, &cmd_sz);
    uint32_t psize, pmu; bmk1822_dmabuf_size(cmd, cmd_sz, &psize, &pmu);
    printf("cmd_sz=%u psize=%u pmu=%u\n", cmd_sz, psize, pmu); fflush(stdout);

    /* allocate psize ONLY (no pmu region) */
    CVI_RT_MEM dm = CVI_RT_MemAlloc(rt, psize);
    uint8_t *db = CVI_RT_MemGetVAddr(dm);
    bmk1822_dmabuf_convert(cmd, cmd_sz, db);
    bmk1822_arraybase_set(db, pa, 0, 0, 0);
    CVI_RT_MemFlush(rt, dm);

    CVI_RT_MEM ld = NULL;
    CVI_RC rc = CVI_RT_LoadDmabuf(rt, dm, psize, pa, 0, false, &ld);
    printf("LoadDmabuf(psize-only, pmu=false) rc=%d ld=%p\n", (int)rc, (void*)ld); fflush(stdout);
    if (ld) {
        rc = CVI_RT_RunCmdbufEx(rt, ld, &(CVI_RT_ARRAYBASE){.gaddr_base0 = pa});
        CVI_RT_MemInvld(rt, mem);
        int8_t *p1 = (int8_t*)(va + P1_OFF);
        /* host ref for n=0 */
        int32_t s = 0; for (int k = 0; k < 32; k++) s += (int32_t)left[k] * (int32_t)right[(size_t)k * 896];
        int32_t ref = (s + 16) >> 5; if (ref > 127) ref = 127; if (ref < -128) ref = -128;
        printf("p1[0]=%d ref=%d %s\n", p1[0], (int8_t)ref, p1[0] == (int8_t)ref ? "MATCH" : "MISMATCH");
    }
    CVI_RT_MemFree(rt, dm);
    CVI_RT_MemFree(rt, mem); CVI_RT_DeInit(rt);
    printf("== done ==\n");
    return 0;
}
