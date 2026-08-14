/* m1_merge_test.c — Phase 7d: validate merged N-tile cmdbuf for up/gate M=1.
 *
 * Key idea: tilew=256 divides N=4864 exactly (19 tiles, all uniform [32,256]),
 * so ONE cmdbuf does g2l(L) + 19 x [g2l(R_t)+TIU(r)+l2g(O_t)] with R/O lmem
 * REUSED across tiles (explicit WAR deps).  This replaces 6 separate
 * RunCmdbufEx calls (tilew=896: 5x[32,896]+[32,384]) with 1 call -> saves 5x
 * driver floor per K-block per pass.
 *
 * Validates bit-exactness vs host int32 + measures submit-time ratio.
 *
 * Layout (planned Phase 7d engine):
 *   ACTQ_OFF 0        [1,32] left
 *   DQ_OFF   4096     19 tiles [32,256] at DQ_OFF + t*8192  (155648 B)
 *   P1_OFF   163840   pass1 out O_t at P1_OFF + t*256
 *   P2_OFF   172032   pass2 out
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
#define ACTQ_OFF  0
#define DQ_OFF    4096
#define P1_OFF    163840
#define P2_OFF    172032

#define TILEW 256
#define N     4864
#define NT    (N / TILEW)       /* 19 */

static inline double now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static inline int32_t round_sat(int32_t acc, int r) {
    int32_t scale = 1 << r, half = scale >> 1;
    int32_t v = (acc + half) >> r;
    if (v > 127) v = 127; if (v < -128) v = -128;
    return v;
}

/* Build merged cmdbuf: g2l(L) + NT x [g2l(R_t)+TIU(r)+l2g(O_t)].
 * R and O lmem reused across tiles -> explicit RAW + WAR dependencies. */
static CVI_RT_MEM build_merged_nt(CVI_RT_HANDLE rt, uint64_t pa, int r, int dest,
                               int nt, uint32_t *psize_out) {
    uint32_t ooff = dest ? P2_OFF : P1_OFF;
    uint8_t *cmdbuf = malloc(131072);
    if (!cmdbuf) { fprintf(stderr, "oom cmdbuf\n"); exit(2); }
    memset(cmdbuf, 0, 131072);
    bmk_info_t info = { .chip_version = 1822, .cmdbuf_size = 131072, .cmdbuf = cmdbuf };
    bmk1822_context_t *bmk = bmk1822_register(&info);

    bmk1822_matrix_lmem_shape_t SL  = {.n=1, .c=1, .w=32, .col=32};
    bmk1822_matrix_lmem_shape_t SR  = {.n=32,.c=1, .w=TILEW, .col=TILEW};
    bmk1822_matrix_lmem_shape_t SO  = {.n=1, .c=1, .w=TILEW, .col=TILEW};

    bmk1822_matrix_lmem_t *ml_l = bmk1822_lmem_alloc_matrix(bmk, SL, FMT_I8, 1);
    bmk1822_matrix_lmem_t *ml_r = bmk1822_lmem_alloc_matrix(bmk, SR, FMT_I8, 1);
    bmk1822_matrix_lmem_t *ml_o = bmk1822_lmem_alloc_matrix(bmk, SO, FMT_I8, 1);

    bmk1822_matrix_tgmem_t mg_l = {0, ACTQ_OFF, FMT_I8, {1, 32}, {32}};
    fprintf(stderr, "  [tr] g2lL ...\n"); fflush(stderr);
    bmk1822_op_t *op_g2lL = bmk1822_tdma_g2l_matrix_copy(
        bmk, &(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_l, ml_l});
    fprintf(stderr, "  [tr] g2lL done\n"); fflush(stderr);

    bmk1822_op_t *prev_mm = NULL, *prev_l2g = NULL;
    bmk1822_matrix_tgmem_t mg_r = {0, 0, FMT_I8, {32, TILEW}, {TILEW}};
    bmk1822_matrix_tgmem_t mg_o = {0, 0, FMT_I8, {1, TILEW}, {TILEW}};
    for (int t = 0; t < nt; t++) {
        uint32_t dqoff = DQ_OFF + (size_t)t * 32 * TILEW;
        uint32_t ooff_t = ooff + (size_t)t * TILEW;
        mg_r.start_address = dqoff;
        mg_o.start_address = ooff_t;

        fprintf(stderr, "  [tr] t=%d g2lR ...\n", t); fflush(stderr);
        bmk1822_op_t *op_g2lR = bmk1822_tdma_g2l_matrix_copy(
            bmk, &(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_r, ml_r});
        fprintf(stderr, "  [tr] t=%d TIU ...\n", t); fflush(stderr);
        bmk1822_tiu_matrix_multiplication_param_t mm = {
            .res = ml_o, .left = ml_l, .right = ml_r, .bias = NULL,
            .lshift_bits = 0, .rshift_bits = (uint8_t)r, .res_is_int8 = 1,
            .relu_enable = 0, .add_result = 0, .ps32_mode = 0, .layer_id = 1 };
        bmk1822_op_t *op_mm = bmk1822_tiu_matrix_multiplication(bmk, &mm);
        fprintf(stderr, "  [tr] t=%d l2g ...\n", t); fflush(stderr);
        bmk1822_op_t *op_l2g = bmk1822_tdma_l2g_matrix_copy(
            bmk, &(bmk1822_tdma_l2tg_matrix_copy_param_t){ml_o, &mg_o});
        fprintf(stderr, "  [tr] t=%d deps ...\n", t); fflush(stderr);
        bmk1822_add_dependency(bmk, op_g2lR, op_mm);      /* RAW R */
        bmk1822_add_dependency(bmk, op_mm, op_l2g);       /* RAW O */
        bmk1822_add_dependency(bmk, op_g2lL, op_mm);      /* RAW L */
        if (prev_mm)  bmk1822_add_dependency(bmk, prev_mm, op_g2lR);   /* WAR R */
        if (prev_l2g) bmk1822_add_dependency(bmk, prev_l2g, op_mm);    /* WAR O */
        prev_mm = op_mm; prev_l2g = op_l2g;
        fprintf(stderr, "  [tr] t=%d done\n", t); fflush(stderr);
    }

    uint32_t cmd_sz; uint8_t *cmd = bmk1822_acquire_cmdbuf(bmk, &cmd_sz);
    uint32_t psize, pmu; bmk1822_dmabuf_size(cmd, cmd_sz, &psize, &pmu);
    /* engine pattern: allocate ONLY psize (pmu ignored) */
    CVI_RT_MEM dm = CVI_RT_MemAlloc(rt, psize);
    uint8_t *db = CVI_RT_MemGetVAddr(dm);
    bmk1822_dmabuf_convert(cmd, cmd_sz, db);
    bmk1822_arraybase_set(db, pa, 0, 0, 0);
    CVI_RT_MemFlush(rt, dm);
    bmk1822_cleanup(bmk);
    free(cmdbuf);
    if (psize_out) *psize_out = psize;
    printf("  [build_merged] r=%d dest=%d nt=%d cmd_sz=%u psize=%u (pmu=%u ignored)\n", r, dest, nt, cmd_sz, psize, pmu);
    return dm;
}

int main(void) {
    printf("===== Phase 7d m1_merge_test (merged %d-tile cmdbuf, tilew=%d) =====\n", NT, TILEW);
    CVI_RT_HANDLE rt; CVI_RT_Init(&rt);
    CVI_RT_MEM mem = CVI_RT_MemAlloc(rt, NEURON_SZ);
    uint64_t pa = CVI_RT_MemGetPAddr(mem);
    uint8_t *va = CVI_RT_MemGetVAddr(mem);
    CVI_RT_SetBaseReg(rt, 0, pa);
    memset(va, 0, NEURON_SZ);

    srand(7);
    int8_t *left = (int8_t*)(va + ACTQ_OFF);
    for (int i = 0; i < 32; i++) left[i] = (int8_t)(rand() % 200 - 100);
    int8_t *dq = (int8_t*)(va + DQ_OFF);
    for (int t = 0; t < NT; t++) {
        int8_t *base = dq + (size_t)t * 32 * TILEW;
        for (int k = 0; k < 32; k++)
            for (int n = 0; n < TILEW; n++)
                base[(size_t)k * TILEW + n] = (int8_t)(rand() % 15 - 7);
    }
    CVI_RT_MemFlush(rt, mem);

    uint32_t s1, s2;
    CVI_RT_MEM dm1 = build_merged_nt(rt, pa, 5, 0, NT, &s1);
    CVI_RT_MEM dm2 = build_merged_nt(rt, pa, 3, 1, NT, &s2);
    CVI_RT_MEM ld1, ld2;
    CVI_RT_LoadDmabuf(rt, dm1, s1, pa, 0, false, &ld1);
    CVI_RT_LoadDmabuf(rt, dm2, s2, pa, 0, false, &ld2);

    /* pass1 r=5 */
    CVI_RT_RunCmdbufEx(rt, ld1, &(CVI_RT_ARRAYBASE){.gaddr_base0 = pa});
    CVI_RT_MemInvldEx(rt, mem, P1_OFF + N);
    int8_t *p1 = (int8_t*)(va + P1_OFF);
    int bad = 0;
    for (int t = 0; t < NT; t++) {
        const int8_t *R = dq + (size_t)t * 32 * TILEW;
        for (int n = 0; n < TILEW; n++) {
            int32_t s = 0;
            for (int k = 0; k < 32; k++) s += (int32_t)left[k] * R[(size_t)k * TILEW + n];
            int g = (int)round_sat(s, 5);
            if (p1[(size_t)t * TILEW + n] != g) { if (bad < 10) printf("  P1 t=%d n=%d got=%d exp=%d\n", t, n, p1[(size_t)t*TILEW+n], g); bad++; }
        }
    }
    printf("pass1 (r=5): %s (%d mismatches)\n", bad == 0 ? "BIT-EXACT" : "MISMATCH", bad);

    /* pass2 r=3 */
    CVI_RT_RunCmdbufEx(rt, ld2, &(CVI_RT_ARRAYBASE){.gaddr_base0 = pa});
    CVI_RT_MemInvldEx(rt, mem, P2_OFF + N);
    int8_t *p2 = (int8_t*)(va + P2_OFF);
    bad = 0;
    for (int t = 0; t < NT; t++) {
        const int8_t *R = dq + (size_t)t * 32 * TILEW;
        for (int n = 0; n < TILEW; n++) {
            int32_t s = 0;
            for (int k = 0; k < 32; k++) s += (int32_t)left[k] * R[(size_t)k * TILEW + n];
            int g = (int)round_sat(s, 3);
            if (p2[(size_t)t * TILEW + n] != g) { if (bad < 10) printf("  P2 t=%d n=%d got=%d exp=%d\n", t, n, p2[(size_t)t*TILEW+n], g); bad++; }
        }
    }
    printf("pass2 (r=3): %s (%d mismatches)\n", bad == 0 ? "BIT-EXACT" : "MISMATCH", bad);

    /* timing: merged 19-tile vs current 6x single-tile (tilew=896) submissions */
    {
        bmk1822_matrix_lmem_shape_t SL  = {.n=1,.c=1,.w=32,.col=32};
        bmk1822_matrix_lmem_shape_t SR  = {.n=32,.c=1,.w=896,.col=896};
        bmk1822_matrix_lmem_shape_t SO  = {.n=1,.c=1,.w=896,.col=896};
        static uint8_t cb[65536] __attribute__((aligned(16)));
        CVI_RT_MEM sld[6];
        int tn[6] = {896,896,896,896,896,384};
        for (int t = 0; t < 6; t++) {
            memset(cb, 0, sizeof cb);
            bmk_info_t info = { .chip_version = 1822, .cmdbuf_size = sizeof(cb), .cmdbuf = cb };
            bmk1822_context_t *bmk = bmk1822_register(&info);
            bmk1822_matrix_lmem_t *ml_l = bmk1822_lmem_alloc_matrix(bmk, SL, FMT_I8, 1);
            bmk1822_matrix_lmem_t *ml_r = bmk1822_lmem_alloc_matrix(bmk, SR, FMT_I8, 1);
            bmk1822_matrix_lmem_t *ml_o = bmk1822_lmem_alloc_matrix(bmk, SO, FMT_I8, 1);
            bmk1822_matrix_tgmem_t mg_l = {0, ACTQ_OFF, FMT_I8, {1, 32}, {32}};
            bmk1822_matrix_tgmem_t mg_r = {0, DQ_OFF + (size_t)t*32*896, FMT_I8, {32, tn[t]}, {tn[t]}};
            bmk1822_matrix_tgmem_t mg_o = {0, P1_OFF + (size_t)t*896, FMT_I8, {1, tn[t]}, {tn[t]}};
            bmk1822_tdma_g2l_matrix_copy(bmk, &(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_l, ml_l});
            bmk1822_tdma_g2l_matrix_copy(bmk, &(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_r, ml_r});
            bmk1822_tiu_matrix_multiplication_param_t mm = {
                .res = ml_o, .left = ml_l, .right = ml_r, .bias = NULL,
                .lshift_bits = 0, .rshift_bits = 5, .res_is_int8 = 1,
                .relu_enable = 0, .add_result = 0, .ps32_mode = 0, .layer_id = 1 };
            bmk1822_tiu_matrix_multiplication(bmk, &mm);
            bmk1822_tdma_l2g_matrix_copy(bmk, &(bmk1822_tdma_l2tg_matrix_copy_param_t){ml_o, &mg_o});
            uint32_t cmd_sz; uint8_t *cmd = bmk1822_acquire_cmdbuf(bmk, &cmd_sz);
            uint32_t psize, pmu; bmk1822_dmabuf_size(cmd, cmd_sz, &psize, &pmu);
            CVI_RT_MEM dm = CVI_RT_MemAlloc(rt, psize);
            uint8_t *db = CVI_RT_MemGetVAddr(dm);
            bmk1822_dmabuf_convert(cmd, cmd_sz, db);
            bmk1822_arraybase_set(db, pa, 0, 0, 0);
            CVI_RT_MemFlush(rt, dm);
            CVI_RT_LoadDmabuf(rt, dm, psize, pa, 0, false, &sld[t]);
            bmk1822_cleanup(bmk);
        }
        int REP = 300;
        double t0 = now();
        for (int i = 0; i < REP; i++)
            CVI_RT_RunCmdbufEx(rt, ld1, &(CVI_RT_ARRAYBASE){.gaddr_base0 = pa});
        double t_merged = (now() - t0) / REP;

        t0 = now();
        for (int i = 0; i < REP; i++)
            for (int t = 0; t < 6; t++)
                CVI_RT_RunCmdbufEx(rt, sld[t], &(CVI_RT_ARRAYBASE){.gaddr_base0 = pa});
        double t_single = (now() - t0) / REP;
        printf("timing: merged 1-cmdbuf %.3f ms | 6x single %.3f ms | save %.3f ms/call-set | ratio %.2fx\n",
               t_merged*1e3, t_single*1e3, (t_single - t_merged)*1e3, t_single/t_merged);
    }

    CVI_RT_MemFree(rt, dm1); CVI_RT_MemFree(rt, dm2);
    CVI_RT_MemFree(rt, mem); CVI_RT_DeInit(rt);
    printf("===== m1_merge_test done =====\n");
    return bad == 0 ? 0 : 1;
}
