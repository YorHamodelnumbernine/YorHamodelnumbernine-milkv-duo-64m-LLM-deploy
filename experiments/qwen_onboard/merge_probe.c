/* merge_probe.c — isolate acquire-time assert_ml_mg_same_size on merged cmdbuf.
 *
 * Usage: merge_probe <nt> <mode>
 *   mode=reuse : single ml_r/ml_o reused across tiles with WAR deps (my design)
 *   mode=fresh : fresh ml_r/ml_o allocated per tile, same op order (no reuse)
 *
 * Builds ONE merged cmdbuf (r=5, dest=P1).  If acquire succeeds, prints
 * "ACQUIRE_OK psize=..." then runs it and checks P1 vs host ref.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "cviruntime_context.h"
#include "bmkernel/bm1822/bmkernel_1822.h"

#define ACTQ_OFF 0
#define DQ_OFF   4096
#define P1_OFF   163840

static inline int32_t round_sat(int32_t acc, int r) {
    int32_t scale = 1 << r, half = scale >> 1;
    int32_t v = (acc + half) >> r;
    if (v > 127) v = 127; if (v < -128) v = -128;
    return v;
}

int main(int argc, char **argv) {
    int nt = argc > 1 ? atoi(argv[1]) : 2;
    int mode = argc > 2 && !strcmp(argv[2], "fresh") ? 1 : 0;
    int tilew = argc > 3 ? atoi(argv[3]) : 256;
    printf("===== merge_probe nt=%d mode=%s tilew=%d =====\n", nt, mode ? "fresh" : "reuse", tilew);
    fflush(stdout);

    CVI_RT_HANDLE rt; CVI_RT_Init(&rt);
    CVI_RT_MEM mem = CVI_RT_MemAlloc(rt, P1_OFF + nt * tilew + 4096);
    uint64_t pa = CVI_RT_MemGetPAddr(mem);
    uint8_t *va = CVI_RT_MemGetVAddr(mem);
    CVI_RT_SetBaseReg(rt, 0, pa);
    memset(va, 0, P1_OFF + nt * tilew + 4096);
    srand(7);
    int8_t *left = (int8_t*)(va + ACTQ_OFF);
    for (int i = 0; i < 32; i++) left[i] = (int8_t)(rand() % 200 - 100);
    int8_t *dq = (int8_t*)(va + DQ_OFF);
    for (int t = 0; t < nt; t++) {
        int8_t *base = dq + (size_t)t * 32 * tilew;
        for (int k = 0; k < 32; k++)
            for (int n = 0; n < tilew; n++)
                base[(size_t)k * tilew + n] = (int8_t)(rand() % 15 - 7);
    }
    CVI_RT_MemFlush(rt, mem);

    uint8_t *cmdbuf = malloc(131072);
    if (!cmdbuf) { fprintf(stderr, "oom\n"); return 2; }
    memset(cmdbuf, 0, 131072);
    bmk_info_t info = { .chip_version = 1822, .cmdbuf_size = 131072, .cmdbuf = cmdbuf };
    bmk1822_context_t *bmk = bmk1822_register(&info);

    bmk1822_matrix_lmem_shape_t SL = {.n=1,.c=1,.w=32,.col=32};
    bmk1822_matrix_lmem_shape_t SR = {.n=32,.c=1,.w=tilew,.col=tilew};
    bmk1822_matrix_lmem_shape_t SO = {.n=1,.c=1,.w=tilew,.col=tilew};

    bmk1822_matrix_lmem_t *ml_l = bmk1822_lmem_alloc_matrix(bmk, SL, FMT_I8, 1);
    bmk1822_matrix_tgmem_t mg_l = {0, ACTQ_OFF, FMT_I8, {1, 32}, {32}};
    bmk1822_op_t *op_g2lL = bmk1822_tdma_g2l_matrix_copy(
        bmk, &(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_l, ml_l});

    /* lmem budget: reuse mode uses 1x R(8192)+1x O(256); fresh mode uses nt x each */
    bmk1822_matrix_lmem_t *ml_r[16], *ml_o[16];
    bmk1822_matrix_lmem_t *r_shared = bmk1822_lmem_alloc_matrix(bmk, SR, FMT_I8, 1);
    bmk1822_matrix_lmem_t *o_shared = bmk1822_lmem_alloc_matrix(bmk, SO, FMT_I8, 1);
    for (int t = 0; t < nt && t < 16; t++) {
        ml_r[t] = mode ? bmk1822_lmem_alloc_matrix(bmk, SR, FMT_I8, 1) : r_shared;
        ml_o[t] = mode ? bmk1822_lmem_alloc_matrix(bmk, SO, FMT_I8, 1) : o_shared;
    }

    bmk1822_op_t *prev_mm = NULL, *prev_l2g = NULL;
    for (int t = 0; t < nt && t < 16; t++) {
        bmk1822_matrix_tgmem_t mg_r = {0, DQ_OFF + (size_t)t*32*tilew, FMT_I8, {32, tilew}, {tilew}};
        bmk1822_matrix_tgmem_t mg_o = {0, P1_OFF + (size_t)t*tilew, FMT_I8, {1, tilew}, {tilew}};
        bmk1822_op_t *op_g2lR = bmk1822_tdma_g2l_matrix_copy(
            bmk, &(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_r, ml_r[t]});
        bmk1822_tiu_matrix_multiplication_param_t mm = {
            .res = ml_o[t], .left = ml_l, .right = ml_r[t], .bias = NULL,
            .lshift_bits = 0, .rshift_bits = 5, .res_is_int8 = 1,
            .relu_enable = 0, .add_result = 0, .ps32_mode = 0, .layer_id = 1 };
        bmk1822_op_t *op_mm = bmk1822_tiu_matrix_multiplication(bmk, &mm);
        bmk1822_op_t *op_l2g = bmk1822_tdma_l2g_matrix_copy(
            bmk, &(bmk1822_tdma_l2tg_matrix_copy_param_t){ml_o[t], &mg_o});
        bmk1822_add_dependency(bmk, op_g2lR, op_mm);
        bmk1822_add_dependency(bmk, op_mm, op_l2g);
        bmk1822_add_dependency(bmk, op_g2lL, op_mm);
        if (prev_mm)  bmk1822_add_dependency(bmk, prev_mm, op_g2lR);
        if (prev_l2g) bmk1822_add_dependency(bmk, prev_l2g, op_mm);
        prev_mm = op_mm; prev_l2g = op_l2g;
    }
    printf("[probe] ops built nt=%d mode=%s\n", nt, mode ? "fresh" : "reuse");
    fflush(stdout);

    uint32_t cmd_sz; uint8_t *cmd = bmk1822_acquire_cmdbuf(bmk, &cmd_sz);
    printf("[probe] acquire OK cmd_sz=%u\n", cmd_sz); fflush(stdout);
    uint32_t psize, pmu; bmk1822_dmabuf_size(cmd, cmd_sz, &psize, &pmu);
    printf("[probe] dmabuf_size OK psize=%u pmu=%u\n", psize, pmu); fflush(stdout);
    CVI_RT_MEM dm = CVI_RT_MemAlloc(rt, psize);
    uint8_t *db = CVI_RT_MemGetVAddr(dm);
    bmk1822_dmabuf_convert(cmd, cmd_sz, db);
    printf("[probe] convert OK\n"); fflush(stdout);
    bmk1822_arraybase_set(db, pa, 0, 0, 0);
    CVI_RT_MemFlush(rt, dm);
    CVI_RT_MEM ld;
    CVI_RT_LoadDmabuf(rt, dm, psize, pa, 0, false, &ld);
    printf("[probe] load OK\n"); fflush(stdout);

    CVI_RT_RunCmdbufEx(rt, ld, &(CVI_RT_ARRAYBASE){.gaddr_base0 = pa});
    CVI_RT_MemInvldEx(rt, mem, P1_OFF + nt * tilew);
    int8_t *p1 = (int8_t*)(va + P1_OFF);
    int bad = 0;
    for (int t = 0; t < nt; t++) {
        const int8_t *R = dq + (size_t)t * 32 * tilew;
        for (int n = 0; n < tilew; n++) {
            int32_t s = 0;
            for (int k = 0; k < 32; k++) s += (int32_t)left[k] * R[(size_t)k * tilew + n];
            int g = (int)round_sat(s, 5);
            if (p1[(size_t)t * tilew + n] != g) { if (bad < 5) printf("  MIS t=%d n=%d got=%d exp=%d\n", t, n, p1[(size_t)t*tilew+n], g); bad++; }
        }
    }
    printf("RESULT nt=%d mode=%s psize=%u pmu=%u run=%s (%d mismatches)\n",
           nt, mode ? "fresh" : "reuse", psize, pmu, bad == 0 ? "BIT-EXACT" : "MISMATCH", bad);

    bmk1822_cleanup(bmk);
    free(cmdbuf);
    CVI_RT_MemFree(rt, dm); CVI_RT_MemFree(rt, mem); CVI_RT_DeInit(rt);
    return 0;
}
