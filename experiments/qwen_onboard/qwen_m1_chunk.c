/* qwen_m1_chunk.c — M1 milestone: on-board pass1 readback alignment with REAL
 *   Qwen Path A data.
 *
 * Real data path exercised (per case):
 *   weights_kal INT4 K-aligned block -> CPU per-group dequant (raw int8 [-8..7])
 *   + per_row activation quant (Step ①) -> TIU two-pass (KG=32, N=896):
 *       pass1 rshift=rsafe -> l2g readback -> CPU max -> r_opt
 *       pass2 rshift=r_opt  -> l2g readback
 *   verify p1/p2 element-exact vs host int8_round_div(acc,.) and report the
 *   fp32-accumulate (p2 * 2^r * gsc[n] * sc_row) vs fp64 gold max diff.
 *
 * TIU plumbing copied from gate1_mrow_check (FORCED linear lmem shape:
 *   {n,c=1,w=full,col=full}; default_shape c-split is NOT l2g-deinterleavable).
 * Embedding: qwen_m1_data.h (gen_m1_data.py).
 *
 * Build: make qwen_m1_chunk ; deploy: python3 ~/Documents/MilkV_duo_project/duo_run.py qwen_m1_chunk
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "cviruntime_context.h"
#include "bmkernel/bm1822/bmkernel_1822.h"
#include "qwen_m1_data.h"

#define NEURON_SZ 262144
#define RIGHT_OFF 4096
#define OUT1_OFF  65536
#define OUT2_OFF  131072

/* ---------------- host semantic helpers (qwen_kal_ref exact) ---------------- */
static inline int8_t int8_round_div(int32_t acc, int rshift) {
    int32_t half = 1 << (rshift - 1);
    int32_t r = (acc + half) >> rshift;
    if (r > 127) r = 127;
    if (r < -128) r = -128;
    return (int8_t)r;
}
static int32_t max_i8(const int8_t *a, size_t n) {
    int32_t m = 0; for (size_t i = 0; i < n; i++) { int32_t v = a[i]; if (v < 0) v = -v; if (v > m) m = v; } return m;
}
static int matmul_rshift_w(int K, int wmax) {
    int r = 0; long long md = (long long)K * 127 * wmax;
    while ((md >> r) > 127) r++;
    return r;
}
static float fp16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16, exp = (h >> 10) & 0x1f, man = h & 0x3ff, f;
    if (exp == 0) { if (man == 0) f = sign; else { exp = 127 - 15 + 1; while (!(man & 0x400)) { man <<= 1; exp--; } man &= 0x3ff; f = sign | (exp << 23) | (man << 13); } }
    else if (exp == 31) f = sign | 0x7f800000u | (man << 13);
    else f = sign | ((exp + (127 - 15)) << 23) | (man << 13);
    float out; memcpy(&out, &f, 4); return out;
}
/* round-half-to-even (numpy np.round parity) — per_row_quant */
static inline int32_t round_bankers(float v) {
    float f = floorf(v); float d = v - f;
    if (d > 0.5f) return (int32_t)(f + 1.0f);
    if (d < 0.5f) return (int32_t)f;
    return ((int32_t)f % 2 == 0) ? (int32_t)f : (int32_t)(f + 1.0f);
}
static void per_row_quant(const float *x, int M, int K, int8_t *q, float *sc) {
    for (int m = 0; m < M; m++) {
        const float *xr = x + (size_t)m * K; float mx = 0;
        for (int k = 0; k < K; k++) { float a = fabsf(xr[k]); if (a > mx) mx = a; }
        float s = mx / 127.0f; if (s < 1e-12f) s = 1e-12f; sc[m] = s;
        int8_t *qr = q + (size_t)m * K;
        for (int k = 0; k < K; k++) {
            int32_t ri = round_bankers(xr[k] / s); if (ri > 127) ri = 127; if (ri < -128) ri = -128;
            qr[k] = (int8_t)ri;
        }
    }
}
/* dequant one K-block: nib [N][16] packed along K -> w_i8 [32,N] (raw int4 -8..7) */
static void dequant_block(const unsigned char *nib, int N, int8_t *w) {
    for (int n = 0; n < N; n++) {
        const unsigned char *b = nib + (size_t)n * 16;
        for (int j = 0; j < 16; j++) {
            int lo = b[j] & 0xF, hi = b[j] >> 4;
            lo = lo > 7 ? lo - 16 : lo; hi = hi > 7 ? hi - 16 : hi;
            w[(size_t)(2 * j) * N + n] = (int8_t)lo;
            w[(size_t)(2 * j + 1) * N + n] = (int8_t)hi;
        }
    }
}

static int run_case(const M1Case *cs, int idx) {
    const int M = 1, K = M1_G, N = M1_N;
    CVI_RT_HANDLE rt; CVI_RT_Init(&rt);
    CVI_RT_MEM mem = CVI_RT_MemAlloc(rt, NEURON_SZ);
    uint64_t pa = CVI_RT_MemGetPAddr(mem);
    uint8_t *va = CVI_RT_MemGetVAddr(mem);
    CVI_RT_SetBaseReg(rt, 0, pa);
    memset(va, 0, NEURON_SZ);

    /* activation -> x_i8 [1,896] + sc_row */
    static int8_t xi[896];
    static float scr[1];
    per_row_quant(cs->act, 1, M1_N, xi, scr);
    int8_t *left = (int8_t*)(va + 0);
    memcpy(left, xi + cs->koff, K);            /* this chunk's 32 act cols */

    /* dequant INT4 block -> right [32,896] in neuron buffer */
    int8_t *right = (int8_t*)(va + RIGHT_OFF);
    dequant_block(cs->nib, N, right);
    int wmax = max_i8(right, (size_t)K * N);
    int rsafe = matmul_rshift_w(K, wmax) - 3; if (rsafe < 4) rsafe = 4;

    /* host acc + ref */
    int32_t *acc = malloc((size_t)N * sizeof(int32_t));
    for (int n = 0; n < N; n++) {
        int32_t s = 0;
        for (int k = 0; k < K; k++) s += (int32_t)left[k] * (int32_t)right[(size_t)k * N + n];
        acc[n] = s;
    }
    CVI_RT_MemFlush(rt, mem);

    /* ---- pass1 cmdbuf ---- */
    uint8_t cmdbuf[65536] __attribute__((aligned(16)));
    bmk_info_t info = { .chip_version = 1822, .cmdbuf_size = sizeof(cmdbuf), .cmdbuf = cmdbuf };
    bmk1822_context_t *bmk = bmk1822_register(&info);
    bmk1822_matrix_lmem_shape_t sl = {.n = M, .c = 1, .w = K, .col = K};
    bmk1822_matrix_lmem_shape_t sr = {.n = K, .c = 1, .w = N, .col = N};
    bmk1822_matrix_lmem_shape_t so = {.n = M, .c = 1, .w = N, .col = N};
    bmk1822_matrix_lmem_t *ml_l = bmk1822_lmem_alloc_matrix(bmk, sl, FMT_I8, 1);
    bmk1822_matrix_lmem_t *ml_r = bmk1822_lmem_alloc_matrix(bmk, sr, FMT_I8, 1);
    bmk1822_matrix_lmem_t *ml_o = bmk1822_lmem_alloc_matrix(bmk, so, FMT_I8, 1);
    uint32_t ls = bmk1822_lmem_matrix_to_size(bmk, sr, FMT_I8, 1)
                + bmk1822_lmem_matrix_to_size(bmk, so, FMT_I8, 1)
                + bmk1822_lmem_matrix_to_size(bmk, sl, FMT_I8, 1);
    if (!ml_l || !ml_r || !ml_o) { printf("  [%s] lmem_alloc FAIL (%u bytes)\n", cs->name, ls); goto fail; }
    bmk1822_matrix_tgmem_t mg_l = {0, 0, FMT_I8, {(uint32_t)M, (uint32_t)K}, {(uint32_t)K}};
    bmk1822_matrix_tgmem_t mg_r = {0, RIGHT_OFF, FMT_I8, {(uint32_t)K, (uint32_t)N}, {(uint32_t)N}};
    bmk1822_tdma_g2l_matrix_copy(bmk, &(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_l, ml_l});
    bmk1822_tdma_g2l_matrix_copy(bmk, &(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_r, ml_r});
    bmk1822_matrix_tgmem_t mg_o1 = {0, OUT1_OFF, FMT_I8, {(uint32_t)M, (uint32_t)N}, {(uint32_t)N}};
    bmk1822_tiu_matrix_multiplication_param_t mm1 = {
        .res = ml_o, .left = ml_l, .right = ml_r, .bias = NULL,
        .lshift_bits = 0, .rshift_bits = (uint8_t)rsafe, .res_is_int8 = 1, .relu_enable = 0,
        .add_result = 0, .ps32_mode = 0, .layer_id = 1 };
    if (!bmk1822_tiu_matrix_multiplication(bmk, &mm1)) { printf("  [%s] P1 REJECTED\n", cs->name); goto fail; }
    bmk1822_tdma_l2g_matrix_copy(bmk, &(bmk1822_tdma_l2tg_matrix_copy_param_t){ml_o, &mg_o1});
    uint32_t cmd_sz; uint8_t *cmd = bmk1822_acquire_cmdbuf(bmk, &cmd_sz);
    uint32_t psize, pmu_size; bmk1822_dmabuf_size(cmd, cmd_sz, &psize, &pmu_size);
    CVI_RT_MEM dmabuf_mem = CVI_RT_MemAlloc(rt, psize + pmu_size);
    uint8_t *dmabuf = CVI_RT_MemGetVAddr(dmabuf_mem);
    bmk1822_dmabuf_convert(cmd, cmd_sz, dmabuf);
    bmk1822_arraybase_set(dmabuf, pa, 0, 0, 0);
    CVI_RT_MemFlush(rt, dmabuf_mem);
    CVI_RT_MEM loaded; CVI_RT_LoadDmabuf(rt, dmabuf_mem, psize + pmu_size, pa, 0, false, &loaded);
    CVI_RT_RunCmdbufEx(rt, loaded, &(CVI_RT_ARRAYBASE){.gaddr_base0 = pa});
    CVI_RT_MemInvld(rt, mem);

    int8_t *p1 = (int8_t*)(va + OUT1_OFF);
    int bad1 = 0, maxabs = 0;
    for (int n = 0; n < N; n++) {
        if (p1[n] != int8_round_div(acc[n], rsafe)) bad1++;
        int a = p1[n]; if (a < 0) a = -a; if (a > maxabs) maxabs = a;
    }
    long long est = (long long)maxabs << rsafe;
    int r_opt = 0; while (est > (127LL << r_opt)) r_opt++;

    /* ---- pass2 cmdbuf (same lmem, no reload, only rshift changes) ---- */
    uint8_t cmdbuf2[65536] __attribute__((aligned(16)));
    bmk_info_t info2 = { .chip_version = 1822, .cmdbuf_size = sizeof(cmdbuf2), .cmdbuf = cmdbuf2 };
    bmk1822_context_t *bmk2 = bmk1822_register(&info2);
    bmk1822_matrix_lmem_t *ml_l2 = bmk1822_lmem_alloc_matrix(bmk2, sl, FMT_I8, 1);
    bmk1822_matrix_lmem_t *ml_r2 = bmk1822_lmem_alloc_matrix(bmk2, sr, FMT_I8, 1);
    bmk1822_matrix_lmem_t *ml_o2 = bmk1822_lmem_alloc_matrix(bmk2, so, FMT_I8, 1);
    if (!ml_l2 || !ml_r2 || !ml_o2) { printf("  [%s] P2 alloc FAIL\n", cs->name); goto fail2; }
    bmk1822_matrix_tgmem_t mg_o2 = {0, OUT2_OFF, FMT_I8, {(uint32_t)M, (uint32_t)N}, {(uint32_t)N}};
    bmk1822_tiu_matrix_multiplication_param_t mm2 = {
        .res = ml_o2, .left = ml_l2, .right = ml_r2, .bias = NULL,
        .lshift_bits = 0, .rshift_bits = (uint8_t)r_opt, .res_is_int8 = 1, .relu_enable = 0,
        .add_result = 0, .ps32_mode = 0, .layer_id = 2 };
    if (!bmk1822_tiu_matrix_multiplication(bmk2, &mm2)) { printf("  [%s] P2 REJECTED\n", cs->name); goto fail2; }
    bmk1822_tdma_l2g_matrix_copy(bmk2, &(bmk1822_tdma_l2tg_matrix_copy_param_t){ml_o2, &mg_o2});
    cmd = bmk1822_acquire_cmdbuf(bmk2, &cmd_sz);
    bmk1822_dmabuf_size(cmd, cmd_sz, &psize, &pmu_size);
    CVI_RT_MEM dmabuf2 = CVI_RT_MemAlloc(rt, psize + pmu_size);
    dmabuf = CVI_RT_MemGetVAddr(dmabuf2);
    bmk1822_dmabuf_convert(cmd, cmd_sz, dmabuf);
    bmk1822_arraybase_set(dmabuf, pa, 0, 0, 0);
    CVI_RT_MemFlush(rt, dmabuf2);
    CVI_RT_MEM loaded2; CVI_RT_LoadDmabuf(rt, dmabuf2, psize + pmu_size, pa, 0, false, &loaded2);
    CVI_RT_RunCmdbufEx(rt, loaded2, &(CVI_RT_ARRAYBASE){.gaddr_base0 = pa});
    CVI_RT_MemInvld(rt, mem);

    int8_t *p2 = (int8_t*)(va + OUT2_OFF);
    int bad2 = 0, sat2 = 0;
    for (int n = 0; n < N; n++) {
        if (p2[n] != int8_round_div(acc[n], r_opt)) bad2++;
        if (p2[n] == 127 || p2[n] == -128) sat2++;
    }
    /* fp32 accumulate (engine) vs fp64 gold: out = p2 * 2^r * gsc * sc_row */
    double maxdiff = 0.0; float outmax = 0;
    for (int n = 0; n < N; n++) {
        float gsc = fp16_to_f32(cs->gsc[n]);
        double ref = (double)p2[n] * (double)(1 << r_opt) * (double)gsc * (double)scr[0];
        float  tpu = (float)((float)(p2[n] * (1 << r_opt)) * gsc * scr[0]);
        double d = fabs((double)tpu - ref); if (d > maxdiff) maxdiff = d;
        if (fabsf(tpu) > outmax) outmax = fabsf(tpu);
    }
    printf("  [%s] lmem=%u wmax=%d rsafe=%d | P1 %d/%d bad=%d | maxabs=%d r_opt=%d | "
           "P2 %d/%d bad=%d sat8=%d | fp32maxdiff=%.3e | outmax=%.3f\n",
           cs->name, ls, wmax, rsafe, N, N - bad1, bad1, maxabs, r_opt, N, N - bad2, bad2, sat2, maxdiff, outmax);

    free(acc);
    CVI_RT_MemFree(rt, dmabuf2); bmk1822_cleanup(bmk2);
    CVI_RT_MemFree(rt, dmabuf_mem); bmk1822_cleanup(bmk);
    CVI_RT_MemFree(rt, mem); CVI_RT_DeInit(rt);
    return bad1 + bad2;
fail:
    free(acc); CVI_RT_MemFree(rt, dmabuf_mem); bmk1822_cleanup(bmk); CVI_RT_MemFree(rt, mem); CVI_RT_DeInit(rt); return 1;
fail2:
    free(acc); CVI_RT_MemFree(rt, dmabuf_mem); bmk1822_cleanup(bmk); bmk1822_cleanup(bmk2); CVI_RT_MemFree(rt, mem); CVI_RT_DeInit(rt); return 1;
}

int main(void) {
    printf("===== M1: pass1 readback alignment, REAL Qwen Path A blocks (KG=32, N=896) =====\n");
    int rc = 0;
    for (int i = 0; i < M1_NCASE; i++) rc += run_case(&m1_cases[i], i);
    printf("===== rc=%d (%s) =====\n", rc, rc == 0 ? "ALL PASS" : "FAIL");
    return rc ? 1 : 0;
}
