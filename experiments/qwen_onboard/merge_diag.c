/* merge_diag.c — diagnose merged up/gate mismatch with REAL layer0 weights.
 *
 * Flow (mirrors eng_matmul_merged for one K-block g=0 of "up"):
 *   1. parse layer0_kal.bin up nib+gsc
 *   2. dequant_kal_rvv full [32,4864] -> DQ_OFF
 *   3. set ACTQ = deterministic xi (first 32)
 *   4. MemFlushEx(0, DQ_OFF + 32*4864)
 *   5. run merged pass1 cmdbuf (r=5, dest=P1)
 *   6. MemInvldEx(P1), compare P1 vs host int32 ref
 * Also runs the SAME block through the PER-TILE cmdbuf (tilew=608, one tile at a
 * time, like old path) so we can compare merged vs per-tile on identical data.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "cviruntime_context.h"
#include "bmkernel/bm1822/bmkernel_1822.h"
#include "dequant_kal.c"          /* dequant_kal_rvv + dequant_kal_scalar (engine) */

#define ACTQ_OFF 0
#define DQ_OFF   4096
#define P1_OFF   163840
#define P2_OFF   172032
#define D 896
#define F 4864
#define G 32
#define MTILEW 608
#define MNT (F / MTILEW)
#define DKV 128

static inline int32_t round_div(int32_t acc, int r) {
    int32_t scale = 1 << r, half = scale >> 1;
    int32_t v = (acc + half) >> r;
    if (v > 127) v = 127; if (v < -128) v = -128;
    return v;
}

/* build merged pass1 cmdbuf; returns loaded dmabuf */
static CVI_RT_MEM build_merged(CVI_RT_HANDLE rt, uint64_t pa, int r, int dest, CVI_RT_MEM *ld) {
    uint32_t ooff = dest ? P2_OFF : P1_OFF;
    uint8_t cmdbuf[131072] __attribute__((aligned(16)));
    memset(cmdbuf, 0, sizeof cmdbuf);
    bmk_info_t info = { .chip_version = 1822, .cmdbuf_size = sizeof cmdbuf, .cmdbuf = cmdbuf };
    bmk1822_context_t *bmk = bmk1822_register(&info);
    bmk1822_matrix_lmem_shape_t SL = {.n=1,.c=1,.w=32,.col=32};
    bmk1822_matrix_lmem_shape_t SR = {.n=32,.c=1,.w=MTILEW,.col=MTILEW};
    bmk1822_matrix_lmem_shape_t SO = {.n=1,.c=1,.w=MTILEW,.col=MTILEW};
    bmk1822_matrix_lmem_t *ml_l = bmk1822_lmem_alloc_matrix(bmk, SL, FMT_I8, 1);
    bmk1822_matrix_lmem_t *ml_r = bmk1822_lmem_alloc_matrix(bmk, SR, FMT_I8, 1);
    bmk1822_matrix_lmem_t *ml_o = bmk1822_lmem_alloc_matrix(bmk, SO, FMT_I8, 1);
    bmk1822_matrix_tgmem_t mg_l = {0, ACTQ_OFF, FMT_I8, {1, 32}, {32}};
    bmk1822_op_t *op_g2lL = bmk1822_tdma_g2l_matrix_copy(bmk, &(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_l, ml_l});
    bmk1822_op_t *prev_mm = NULL, *prev_l2g = NULL;
    for (int t = 0; t < MNT; t++) {
        uint32_t dqoff = DQ_OFF + (size_t)t * 32 * MTILEW;
        uint32_t ooff_t = ooff + (size_t)t * MTILEW;
        bmk1822_matrix_tgmem_t mg_r = {0, dqoff, FMT_I8, {32, MTILEW}, {MTILEW}};
        bmk1822_matrix_tgmem_t mg_o = {0, ooff_t, FMT_I8, {1, MTILEW}, {MTILEW}};
        bmk1822_op_t *op_g2lR = bmk1822_tdma_g2l_matrix_copy(bmk, &(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_r, ml_r});
        bmk1822_tiu_matrix_multiplication_param_t mm = {
            .res = ml_o, .left = ml_l, .right = ml_r, .bias = NULL,
            .lshift_bits = 0, .rshift_bits = (uint8_t)r, .res_is_int8 = 1,
            .relu_enable = 0, .add_result = 0, .ps32_mode = 0, .layer_id = 1 };
        bmk1822_op_t *op_mm = bmk1822_tiu_matrix_multiplication(bmk, &mm);
        bmk1822_op_t *op_l2g = bmk1822_tdma_l2g_matrix_copy(bmk, &(bmk1822_tdma_l2tg_matrix_copy_param_t){ml_o, &mg_o});
        bmk1822_add_dependency(bmk, op_g2lR, op_mm);
        bmk1822_add_dependency(bmk, op_mm, op_l2g);
        bmk1822_add_dependency(bmk, op_g2lL, op_mm);
        if (prev_mm)  bmk1822_add_dependency(bmk, prev_mm, op_g2lR);
        if (prev_l2g) bmk1822_add_dependency(bmk, prev_l2g, op_mm);
        prev_mm = op_mm; prev_l2g = op_l2g;
    }
    uint32_t cmd_sz; uint8_t *cmd = bmk1822_acquire_cmdbuf(bmk, &cmd_sz);
    uint32_t psize, pmu; bmk1822_dmabuf_size(cmd, cmd_sz, &psize, &pmu);
    CVI_RT_MEM dm = CVI_RT_MemAlloc(rt, psize);
    uint8_t *db = CVI_RT_MemGetVAddr(dm);
    bmk1822_dmabuf_convert(cmd, cmd_sz, db);
    bmk1822_arraybase_set(db, pa, 0, 0, 0);
    CVI_RT_MemFlush(rt, dm);
    CVI_RT_LoadDmabuf(rt, dm, psize, pa, 0, false, ld);
    bmk1822_cleanup(bmk);
    printf("  [diag] merged r=%d dest=%d psize=%u\n", r, dest, psize);
    return dm;
}

/* per-tile cmdbuf [1,32]x[32,608]->[1,608], reads DQ_OFF+tile slice, writes P1_OFF+tilew*idx */
static CVI_RT_MEM build_single(CVI_RT_HANDLE rt, uint64_t pa, int t, CVI_RT_MEM *ld) {
    uint32_t dqoff = DQ_OFF + (size_t)t * 32 * MTILEW;
    uint32_t ooff = P1_OFF + (size_t)t * MTILEW;
    uint8_t cmdbuf[65536] __attribute__((aligned(16)));
    memset(cmdbuf, 0, sizeof cmdbuf);
    bmk_info_t info = { .chip_version = 1822, .cmdbuf_size = sizeof cmdbuf, .cmdbuf = cmdbuf };
    bmk1822_context_t *bmk = bmk1822_register(&info);
    bmk1822_matrix_lmem_shape_t SL = {.n=1,.c=1,.w=32,.col=32};
    bmk1822_matrix_lmem_shape_t SR = {.n=32,.c=1,.w=MTILEW,.col=MTILEW};
    bmk1822_matrix_lmem_shape_t SO = {.n=1,.c=1,.w=MTILEW,.col=MTILEW};
    bmk1822_matrix_lmem_t *ml_l = bmk1822_lmem_alloc_matrix(bmk, SL, FMT_I8, 1);
    bmk1822_matrix_lmem_t *ml_r = bmk1822_lmem_alloc_matrix(bmk, SR, FMT_I8, 1);
    bmk1822_matrix_lmem_t *ml_o = bmk1822_lmem_alloc_matrix(bmk, SO, FMT_I8, 1);
    bmk1822_matrix_tgmem_t mg_l = {0, ACTQ_OFF, FMT_I8, {1, 32}, {32}};
    bmk1822_matrix_tgmem_t mg_r = {0, dqoff, FMT_I8, {32, MTILEW}, {MTILEW}};
    bmk1822_matrix_tgmem_t mg_o = {0, ooff, FMT_I8, {1, MTILEW}, {MTILEW}};
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
    CVI_RT_LoadDmabuf(rt, dm, psize, pa, 0, false, ld);
    bmk1822_cleanup(bmk);
    return dm;
}

int main(void) {
    printf("===== merge_diag (real layer0 up, K-block 0) =====\n");
    CVI_RT_HANDLE rt; CVI_RT_Init(&rt);
    CVI_RT_MEM mem = CVI_RT_MemAlloc(rt, 262144);
    uint64_t pa = CVI_RT_MemGetPAddr(mem);
    uint8_t *va = CVI_RT_MemGetVAddr(mem);
    CVI_RT_SetBaseReg(rt, 0, pa);
    memset(va, 0, 262144);

    FILE *f = fopen("/data/qwen/layer0_kal.bin", "rb");
    if (!f) { fprintf(stderr, "no layer0_kal.bin\n"); return 2; }
    long sz; fseek(f, 0, SEEK_END); sz = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *lb = malloc(sz); if (fread(lb, 1, sz, f) != (size_t)sz) { fprintf(stderr, "read err\n"); return 2; }
    fclose(f);
    size_t off = D * 4;
    off += (size_t)(D / G) * D * (16 + 2);
    off += (size_t)(D / G) * DKV * (16 + 2);
    off += (size_t)(D / G) * DKV * (16 + 2);
    off += (size_t)(D / G) * D * (16 + 2);
    const uint8_t *up_nib = lb + off;
    printf("  up nib offset=%zu nib[0..3]=%02x%02x%02x%02x\n", off, up_nib[0], up_nib[1], up_nib[2], up_nib[3]);

    int8_t xi[D]; for (int i = 0; i < D; i++) xi[i] = (int8_t)((i * 37) % 200 - 100);
    const uint8_t *nib0 = up_nib;

    /* host ref via engine dequant_kal_scalar */
    int32_t ref[F];
    {
        int8_t wbuf[32 * F];
        dequant_kal_scalar(nib0, F, wbuf);
        for (int n = 0; n < F; n++) {
            int32_t s = 0; for (int k = 0; k < 32; k++) s += (int32_t)xi[k] * wbuf[(size_t)k * F + n];
            ref[n] = s;
        }
    }

    /* ---- coherence: dequant (engine RVV) -> flush -> invld -> readback compare ---- */
    {
        static int8_t sav[32 * F];   /* static: avoid 155KB stack */
        dequant_kal_rvv(nib0, F, sav);
        memcpy(va + DQ_OFF, sav, sizeof sav);
        CVI_RT_MemFlushEx(rt, mem, DQ_OFF + (size_t)32 * F);
        CVI_RT_MemInvldEx(rt, mem, DQ_OFF + (size_t)32 * F);
        int dq_bad = 0, dq_first = -1;
        for (int i = 0; i < 32 * F; i++)
            if (((int8_t *)(va + DQ_OFF))[i] != sav[i]) { if (dq_bad < 5) printf("  DQ-RDBK MIS i=%d got=%d exp=%d\n", i, ((int8_t *)(va + DQ_OFF))[i], sav[i]); if (dq_first < 0) dq_first = i; dq_bad++; }
        printf("  DQ-COHERENCE: %s (%d mismatches, first@%d)\n", dq_bad == 0 ? "OK" : "BROKEN", dq_bad, dq_first);
    }

    /* ---- merged pass1 ---- */
    CVI_RT_MEM ld_m, dm_m;
    dm_m = build_merged(rt, pa, 5, 0, &ld_m);
    dequant_kal_rvv(nib0, F, (int8_t *)(va + DQ_OFF));
    memcpy(va + ACTQ_OFF, xi, 32);
    CVI_RT_MemFlushEx(rt, mem, DQ_OFF + (size_t)32 * F);
    CVI_RT_RunCmdbufEx(rt, ld_m, &(CVI_RT_ARRAYBASE){.gaddr_base0 = pa});
    CVI_RT_MemInvldEx(rt, mem, P1_OFF + F);
    int8_t *p1m = (int8_t *)(va + P1_OFF);
    int mb = 0, mb_first = -1;
    for (int n = 0; n < F; n++) {
        int g = (int)round_div(ref[n], 5);
        if (p1m[n] != g) { if (mb < 8) printf("  MERGED MIS n=%d got=%d exp=%d (ref=%d)\n", n, p1m[n], g, ref[n]); if (mb_first < 0) mb_first = n; mb++; }
    }
    printf("  MERGED: %s (%d mismatches, first@%d)\n", mb == 0 ? "BIT-EXACT" : "MISMATCH", mb, mb_first);

    /* ---- per-tile pass1 (old path, same DQ data) ---- */
    CVI_RT_MEM ld_s[8], dm_s[8];
    for (int t = 0; t < MNT; t++) dm_s[t] = build_single(rt, pa, t, &ld_s[t]);
    /* re-dequant full block (mimic merged-style full DQ) then run single tiles */
    dequant_kal_rvv(nib0, F, (int8_t *)(va + DQ_OFF));
    CVI_RT_MemFlushEx(rt, mem, DQ_OFF + (size_t)32 * F);
    memset((int8_t *)(va + P1_OFF), 0, F);
    CVI_RT_MemFlushEx(rt, mem, P1_OFF + F);
    for (int t = 0; t < MNT; t++) {
        /* DQ already holds full block; flush the tile slice like old path would */
        CVI_RT_MemFlushEx(rt, mem, DQ_OFF + (size_t)32 * MTILEW * (t + 1));
        CVI_RT_RunCmdbufEx(rt, ld_s[t], &(CVI_RT_ARRAYBASE){.gaddr_base0 = pa});
    }
    CVI_RT_MemInvldEx(rt, mem, P1_OFF + F);
    int8_t *p1s = (int8_t *)(va + P1_OFF);
    int sb = 0;
    for (int t = 0; t < MNT; t++)
        for (int n = 0; n < MTILEW; n++) {
            int nn = t * MTILEW + n;
            int g = (int)round_div(ref[nn], 5);
            if (p1s[nn] != g) { if (sb < 8) printf("  SINGLE MIS t=%d n=%d got=%d exp=%d\n", t, n, p1s[nn], g); sb++; }
        }
    printf("  SINGLE: %s (%d mismatches)\n", sb == 0 ? "BIT-EXACT" : "MISMATCH", sb);

    CVI_RT_MemFree(rt, dm_m); CVI_RT_MemFree(rt, mem); CVI_RT_DeInit(rt);
    printf("===== merge_diag done =====\n");
    return (mb + sb) == 0 ? 0 : 1;
}
