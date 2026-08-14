/* submit_budget.c — on-board submit budget for Path A two-pass (KG=32, N=896).
 *
 * Answers CEO/TPU Gate② hard-constraint: prebuilt cmdbuf amortization must turn
 * the naive 0.55ms/submit into ~15us select+Run, else the serial 16.99s/token
 * breaks SD-bound (9.18s/token).
 *
 * Measures on CV1800B:
 *   A. naive per-submit  : register+alloc+g2l+matmul+l2g+acquire+convert+load+Run
 *   B. prebuilt per-submit: cmdbuf built once per rshift variant, submit=Run only
 *   C. full q_proj 28-block two-pass (prebuilt cmdbufs + g2l per block) total,
 *      scaled to per-token CPU+TIU budget vs SD 9.18s
 *   D. scalar Path A nibble-extract dequant throughput (one Qwen layer),
 *      scaled to per-token
 *
 * Shape: M=1, K=32, N=896 (q_proj chunk).  Prebuild 16 rshift variants of the
 * matmul+l2g cmdbuf (assumes left/right already in LMEM); g2l is a separate
 * fixed cmdbuf per block.
 *
 * Build: riscv64 cross (see qwen_m1_chunk.c header) ; run: duo_run.py submit_budget
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
#define OUT2_OFF  131072
#define NREP 50

static inline double now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static bmk1822_matrix_lmem_shape_t SL = {.n = 1, .c = 1, .w = 32, .col = 32};
static bmk1822_matrix_lmem_shape_t SR = {.n = 32, .c = 1, .w = 896, .col = 896};
static bmk1822_matrix_lmem_shape_t SO = {.n = 1, .c = 1, .w = 896, .col = 896};

int main(void) {
    printf("===== Path A submit budget (KG=32, N=896, M=1) =====\n");
    CVI_RT_HANDLE rt; CVI_RT_Init(&rt);
    CVI_RT_MEM mem = CVI_RT_MemAlloc(rt, NEURON_SZ);
    uint64_t pa = CVI_RT_MemGetPAddr(mem);
    uint8_t *va = CVI_RT_MemGetVAddr(mem);
    CVI_RT_SetBaseReg(rt, 0, pa);
    memset(va, 0, NEURON_SZ);

    /* synthetic right [32,896] + left [1,32] */
    srand(7);
    int8_t *right = (int8_t*)(va + RIGHT_OFF);
    int8_t *left = (int8_t*)(va + 0);
    for (int i = 0; i < 32 * 896; i++) right[i] = (int8_t)(rand() % 15 - 7);
    for (int i = 0; i < 32; i++) left[i] = (int8_t)(rand() % 200 - 100);
    CVI_RT_MemFlush(rt, mem);

    /* ---- B. prebuilt: 16 rshift variants of matmul+l2g (no g2l), select+Run ---- */
    CVI_RT_MEM *var_mem = calloc(16, sizeof(CVI_RT_MEM));
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
    /* prebuilt: LoadDmabuf ONCE per variant (the engine startup cost).  CRITICAL:
       never LoadDmabuf the same dmabuf twice — trigger3 proved a second load
       flips the runtime into the ~22ms/submit slow path permanently. */
    CVI_RT_MEM p_ld[16];
    for (int i = 0; i < 16; i++)
        CVI_RT_LoadDmabuf(rt, var_mem[i], 65536, pa, 0, false, &p_ld[i]);
    {
        /* per-submit = select cmdbuf + Run + Invld.  Break out Run-only and
           Invld-only to find the floor. */
        double t0 = now();
        for (int r = 0; r < NREP; r++) {
            CVI_RT_RunCmdbufEx(rt, p_ld[5], &(CVI_RT_ARRAYBASE){.gaddr_base0 = pa});
            CVI_RT_MemInvld(rt, mem);
        }
        double dt = (now() - t0) / NREP;
        printf("B1 prebuilt Run+Invld    : %.3f ms  (%.1f us x15360 -> %.2f s/token)\n",
               dt * 1e3, dt * 1e6, dt * 15360);

        t0 = now();
        for (int r = 0; r < NREP; r++)
            CVI_RT_RunCmdbufEx(rt, p_ld[5], &(CVI_RT_ARRAYBASE){.gaddr_base0 = pa});
        dt = (now() - t0) / NREP;
        printf("B2 prebuilt Run-only     : %.3f ms  (%.1f us x15360 -> %.2f s/token)\n",
               dt * 1e3, dt * 1e6, dt * 15360);

        t0 = now();
        for (int r = 0; r < NREP; r++) CVI_RT_MemInvld(rt, mem);
        dt = (now() - t0) / NREP;
        printf("B3 Invld-only (256KB)    : %.3f ms  (%.1f us x15360 -> %.2f s/token)\n",
               dt * 1e3, dt * 1e6, dt * 15360);
    }

    /* ---- C. full q_proj 28-block two-pass with prebuilt cmdbufs + g2l ---- */
    {
        /* g2l cmdbuf (left+right), built once */
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
        CVI_RT_MEM g_ld; CVI_RT_LoadDmabuf(rt, gmem, psize + pmu, pa, 0, false, &g_ld);
        bmk1822_cleanup(bmk);

        /* reuse the single-loaded p_ld from section B — do NOT re-load */

        const int KB = 28;
        double t_g2l = 0, t_p1 = 0, t_inv1 = 0, t_cpu = 0, t_p2 = 0, t_inv2 = 0;
        for (int blk = 0; blk < KB; blk++) {
            double a0 = now();
            CVI_RT_RunCmdbufEx(rt, g_ld, &(CVI_RT_ARRAYBASE){.gaddr_base0 = pa});   /* g2l */
            double a1 = now(); t_g2l += a1 - a0;
            CVI_RT_RunCmdbufEx(rt, p_ld[5], &(CVI_RT_ARRAYBASE){.gaddr_base0 = pa});/* pass1 rsafe */
            double a2 = now(); t_p1 += a2 - a1;
            CVI_RT_MemInvld(rt, mem);
            double a3 = now(); t_inv1 += a3 - a2;
            int8_t *p1 = (int8_t*)(va + OUT1_OFF);
            int maxabs = 0; for (int n = 0; n < 896; n++) { int a = p1[n]; if (a < 0) a = -a; if (a > maxabs) maxabs = a; }
            long long est = (long long)maxabs << 5; int r = 0; while (est > (127LL << r)) r++;
            double a4 = now(); t_cpu += a4 - a3;
            CVI_RT_RunCmdbufEx(rt, p_ld[r], &(CVI_RT_ARRAYBASE){.gaddr_base0 = pa});/* pass2 r_opt */
            double a5 = now(); t_p2 += a5 - a4;
            CVI_RT_MemInvld(rt, mem);
            double a6 = now(); t_inv2 += a6 - a5;
        }
        double per_block = (t_g2l + t_p1 + t_inv1 + t_cpu + t_p2 + t_inv2) / KB * 1e3;
        double per_token_cpu = (t_g2l + t_p1 + t_inv1 + t_cpu + t_p2 + t_inv2) * 6.0;
        printf("C q_proj 28blk 2-pass    : %.2f ms/block  (x6+lmmisc -> ~%.2f s/token CPU+TIU)\n",
               per_block, per_token_cpu);
        printf("  breakdown/block: g2l %.3f ms | p1 %.3f ms | inv1 %.3f ms | cpu %.3f ms | p2 %.3f ms | inv2 %.3f ms\n",
               t_g2l / KB * 1e3, t_p1 / KB * 1e3, t_inv1 / KB * 1e3, t_cpu / KB * 1e3,
               t_p2 / KB * 1e3, t_inv2 / KB * 1e3);

        /* ---- E. isolate cmdbuf-switch cost: is the 22ms in p1/p2 a
                 pipeline-drain on switching DISTINCT cmdbufs? ---- */
        double t0 = now();
        for (int i = 0; i < 28; i++) {
            CVI_RT_RunCmdbufEx(rt, g_ld, &(CVI_RT_ARRAYBASE){.gaddr_base0 = pa});
            CVI_RT_RunCmdbufEx(rt, p_ld[5], &(CVI_RT_ARRAYBASE){.gaddr_base0 = pa});
        }
        double dt = (now() - t0) / 28;
        printf("E1 g2l+pass1 alternate  : %.3f ms/iter (expect ~0.35 if switch cheap)\n", dt * 1e3);

        t0 = now();
        for (int i = 0; i < 28; i++) {
            CVI_RT_RunCmdbufEx(rt, p_ld[5], &(CVI_RT_ARRAYBASE){.gaddr_base0 = pa});
            CVI_RT_RunCmdbufEx(rt, p_ld[6], &(CVI_RT_ARRAYBASE){.gaddr_base0 = pa});
        }
        dt = (now() - t0) / 28;
        printf("E2 pass1+pass2 rswitch  : %.3f ms/iter (r5/r6 variant switch)\n", dt * 1e3);

        t0 = now();
        for (int i = 0; i < 28; i++) {
            CVI_RT_RunCmdbufEx(rt, p_ld[5], &(CVI_RT_ARRAYBASE){.gaddr_base0 = pa});
            CVI_RT_RunCmdbufEx(rt, p_ld[5], &(CVI_RT_ARRAYBASE){.gaddr_base0 = pa});
        }
        dt = (now() - t0) / 28;
        printf("E3 same cmdbuf 2x       : %.3f ms/iter (baseline)\n", dt * 1e3);

        /* ---- F. per-run timing to find the 22ms trigger ---- */
        printf("F per-run timing:\n");
        for (int pass = 0; pass < 3; pass++) {
            printf("  seq%d (p5 x12): ", pass);
            for (int i = 0; i < 12; i++) {
                double s = now();
                CVI_RT_RunCmdbufEx(rt, p_ld[5], &(CVI_RT_ARRAYBASE){.gaddr_base0 = pa});
                printf("%.2f ", (now() - s) * 1e3);
            }
            printf("ms\n");
        }
        for (int pass = 0; pass < 3; pass++) {
            printf("  inv-p5 x8: ");
            for (int i = 0; i < 8; i++) {
                CVI_RT_MemInvld(rt, mem);
                double s = now();
                CVI_RT_RunCmdbufEx(rt, p_ld[5], &(CVI_RT_ARRAYBASE){.gaddr_base0 = pa});
                printf("%.2f ", (now() - s) * 1e3);
            }
            printf("ms\n");
        }
        for (int pass = 0; pass < 3; pass++) {
            printf("  g2l-p5 x8: ");
            for (int i = 0; i < 8; i++) {
                CVI_RT_RunCmdbufEx(rt, g_ld, &(CVI_RT_ARRAYBASE){.gaddr_base0 = pa});
                double s = now();
                CVI_RT_RunCmdbufEx(rt, p_ld[5], &(CVI_RT_ARRAYBASE){.gaddr_base0 = pa});
                printf("%.2f ", (now() - s) * 1e3);
            }
            printf("ms\n");
        }
    }

    /* ---- D. scalar Path A nibble-extract dequant, one Qwen layer ---- */
    {
        /* one layer = 14,906,368 values = 7,453,184 nibble bytes (q,k,v,o,up,gate,down) */
        size_t nvals = 14906368ULL;
        size_t nnib = nvals / 2;
        uint8_t *nib = malloc(nnib);
        int8_t *w = malloc(nvals);
        for (size_t i = 0; i < nnib; i++) nib[i] = (uint8_t)rand();
        double t0 = now();
        int reps = 3;
        for (int rr = 0; rr < reps; rr++) {
            /* process per K-block of 32: nib[N][16], output [32,N] k-major.
               Here flat: each 16 bytes -> 32 output values at stride N. */
            const int N = 896;
            size_t off = 0;
            for (int kblk = 0; kblk < (int)(nnib / (N * 16)); kblk++) {
                for (int n = 0; n < N; n++) {
                    const uint8_t *b = nib + off; off += 16;
                    for (int j = 0; j < 16; j++) {
                        int lo = b[j] & 0xF, hi = b[j] >> 4;
                        lo = lo > 7 ? lo - 16 : lo; hi = hi > 7 ? hi - 16 : hi;
                        w[(size_t)(2 * j) * N + n] = (int8_t)lo;
                        w[(size_t)(2 * j + 1) * N + n] = (int8_t)hi;
                    }
                }
            }
        }
        double dt = (now() - t0) / reps;
        double MBs = nnib / dt / 1e6;
        double per_token = dt * 24;
        printf("D scalar PathA dequant  : %.0f MB/s nibble  (%.2f ms/layer -> %.2f s/token)\n",
               MBs, dt * 1e3, per_token);
        free(nib); free(w);
    }

    /* ---- A. naive per-submit (full path).  Runs LAST: its leaked LoadDmabuf
             handles (SDK pattern frees only the source dmabuf) would otherwise
             poison the TPU command queue and flip later submits to the slow
             path (measured ~22ms).  Placing it last keeps B/C/D/E/F clean. ---- */
    {
        double t0 = now();
        for (int r = 0; r < NREP; r++) {
            uint8_t cmdbuf[65536] __attribute__((aligned(16)));
            bmk_info_t info = { .chip_version = 1822, .cmdbuf_size = sizeof(cmdbuf), .cmdbuf = cmdbuf };
            bmk1822_context_t *bmk = bmk1822_register(&info);
            bmk1822_matrix_lmem_t *ml_l = bmk1822_lmem_alloc_matrix(bmk, SL, FMT_I8, 1);
            bmk1822_matrix_lmem_t *ml_r = bmk1822_lmem_alloc_matrix(bmk, SR, FMT_I8, 1);
            bmk1822_matrix_lmem_t *ml_o = bmk1822_lmem_alloc_matrix(bmk, SO, FMT_I8, 1);
            bmk1822_matrix_tgmem_t mg_l = {0, 0, FMT_I8, {1, 32}, {32}};
            bmk1822_matrix_tgmem_t mg_r = {0, RIGHT_OFF, FMT_I8, {32, 896}, {896}};
            bmk1822_tdma_g2l_matrix_copy(bmk, &(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_l, ml_l});
            bmk1822_tdma_g2l_matrix_copy(bmk, &(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_r, ml_r});
            bmk1822_matrix_tgmem_t mg_o = {0, OUT1_OFF, FMT_I8, {1, 896}, {896}};
            bmk1822_tiu_matrix_multiplication_param_t mm = {
                .res = ml_o, .left = ml_l, .right = ml_r, .bias = NULL,
                .lshift_bits = 0, .rshift_bits = 5, .res_is_int8 = 1, .relu_enable = 0,
                .add_result = 0, .ps32_mode = 0, .layer_id = 1 };
            bmk1822_tiu_matrix_multiplication(bmk, &mm);
            bmk1822_tdma_l2g_matrix_copy(bmk, &(bmk1822_tdma_l2tg_matrix_copy_param_t){ml_o, &mg_o});
            uint32_t cmd_sz; uint8_t *cmd = bmk1822_acquire_cmdbuf(bmk, &cmd_sz);
            uint32_t psize, pmu; bmk1822_dmabuf_size(cmd, cmd_sz, &psize, &pmu);
            CVI_RT_MEM dm = CVI_RT_MemAlloc(rt, psize + pmu);
            uint8_t *db = CVI_RT_MemGetVAddr(dm);
            bmk1822_dmabuf_convert(cmd, cmd_sz, db);
            bmk1822_arraybase_set(db, pa, 0, 0, 0);
            CVI_RT_MemFlush(rt, dm);
            CVI_RT_MEM ld; CVI_RT_LoadDmabuf(rt, dm, psize + pmu, pa, 0, false, &ld);
            CVI_RT_RunCmdbufEx(rt, ld, &(CVI_RT_ARRAYBASE){.gaddr_base0 = pa});
            CVI_RT_MemInvld(rt, mem);
            CVI_RT_MemFree(rt, dm);
            bmk1822_cleanup(bmk);
        }
        double dt = (now() - t0) / NREP;
        printf("A naive per-submit       : %.3f ms  (%.1f us x15360 -> %.2f s/token)\n",
               dt * 1e3, dt * 1e6, dt * 15360);
    }

    printf("===== submit budget done =====\n");
    for (int i = 0; i < 16; i++) if (var_mem[i]) CVI_RT_MemFree(rt, var_mem[i]);
    CVI_RT_MemFree(rt, mem); CVI_RT_DeInit(rt);
    return 0;
}
