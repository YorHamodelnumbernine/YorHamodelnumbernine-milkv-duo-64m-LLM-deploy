/* tiu_impact_probe.c — 复现: TIU 初始化/使用是否导致 C906B 点积变慢 60x?
 * 流程: 1) f32 dot 计时 (基线)  2) CVI_RT_Init + 一次 TIU matmul  3) f32 dot 再计时
 * 用法: tiu_impact_probe
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include "cviruntime_context.h"
#include "bmkernel/bm1822/bmkernel_1822.h"

#define C 1024
#define D 896
#define NEURON_SZ (16 * 1024 * 1024)

static double now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static float *cent; static float *h; static float sc[C];

static double dot_once(void) {
    double t0 = now();
    for (int c = 0; c < C; c++) {
        const float *cd = cent + (size_t)c * D;
        float s = 0;
        for (int j = 0; j < D; j++) s += h[j] * cd[j];
        sc[c] = s;
    }
    return now() - t0;
}

/* 纯寄存器 FMADD (不读内存): 估 CPU FPU 时钟 */
static double fmadd_loop(int n) {
    double t0 = now();
    volatile float acc = 0;
    float a = 1.0000001f, b = 1.0000001f;
    for (int i = 0; i < n; i++) { acc = acc * a + b; }
    (void)acc;
    return now() - t0;
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    cent = malloc((size_t)C * D * sizeof(float));
    h = malloc(D * sizeof(float));
    if (!cent || !h) { fprintf(stderr, "oom\n"); return 2; }
    for (int i = 0; i < C * D; i++) cent[i] = (float)((i * 2654435761u) % 1000) / 1000.0f - 0.5f;
    for (int j = 0; j < D; j++) h[j] = (float)((j * 40503u) % 1000) / 1000.0f - 0.5f;

    double dt;
    dt = dot_once(); printf("dot BEFORE tiu : %.3fs  (fmadd100M=%.3fs)\n", dt, fmadd_loop(100000000));

    /* ---- TIU init + 一次 matmul (模拟 pool_build / 引擎第一层) ---- */
    CVI_RT_HANDLE rt; CVI_RT_Init(&rt);
    CVI_RT_MEM mem = CVI_RT_MemAlloc(rt, NEURON_SZ);
    uint64_t pa = CVI_RT_MemGetPAddr(mem);
    uint8_t *va = CVI_RT_MemGetVAddr(mem);
    CVI_RT_SetBaseReg(rt, 0, pa);
    memset(va, 0, NEURON_SZ);

    bmk_info_t info = { .chip_version = 1822, .cmdbuf_size = 65536, .cmdbuf = malloc(65536) };
    bmk1822_context_t *bmk = bmk1822_register(&info);
    bmk1822_matrix_lmem_shape_t SL = {.n = 8, .c = 1, .w = 32, .col = 32};
    bmk1822_matrix_lmem_shape_t SR = {.n = 32, .c = 1, .w = 32, .col = 32};
    bmk1822_matrix_lmem_shape_t SO = {.n = 8, .c = 1, .w = 32, .col = 32};
    bmk1822_matrix_lmem_t *ml_l = bmk1822_lmem_alloc_matrix(bmk, SL, FMT_I8, 1);
    bmk1822_matrix_lmem_t *ml_r = bmk1822_lmem_alloc_matrix(bmk, SR, FMT_I8, 1);
    bmk1822_matrix_lmem_t *ml_o = bmk1822_lmem_alloc_matrix(bmk, SO, FMT_I8, 1);
    bmk1822_matrix_tgmem_t mg_l = {0, 0, FMT_I8, {8, 32}, {32}};
    bmk1822_matrix_tgmem_t mg_r = {0, 4096, FMT_I8, {32, 32}, {32}};
    bmk1822_matrix_tgmem_t mg_o = {0, 8192, FMT_I8, {8, 32}, {32}};
    bmk1822_tdma_g2l_matrix_copy(bmk, &(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_l, ml_l});
    bmk1822_tdma_g2l_matrix_copy(bmk, &(bmk1822_tdma_tg2l_matrix_copy_param_t){&mg_r, ml_r});
    bmk1822_tiu_matrix_multiplication_param_t mm = {
        .res = ml_o, .left = ml_l, .right = ml_r, .bias = NULL,
        .lshift_bits = 0, .rshift_bits = 0, .res_is_int8 = 1, .relu_enable = 0,
        .add_result = 0, .ps32_mode = 0, .layer_id = 1 };
    bmk1822_tiu_matrix_multiplication(bmk, &mm);
    bmk1822_tdma_l2g_matrix_copy(bmk, &(bmk1822_tdma_l2tg_matrix_copy_param_t){ml_o, &mg_o});
    uint32_t cmd_sz; uint8_t *cmd = bmk1822_acquire_cmdbuf(bmk, &cmd_sz);
    uint32_t psize, pmu; bmk1822_dmabuf_size(cmd, cmd_sz, &psize, &pmu);
    CVI_RT_MEM cm = CVI_RT_MemAlloc(rt, psize);
    uint8_t *db = CVI_RT_MemGetVAddr(cm);
    bmk1822_dmabuf_convert(cmd, cmd_sz, db);
    bmk1822_arraybase_set(db, pa, 0, 0, 0);
    CVI_RT_MemFlush(rt, cm);
    CVI_RT_MEM ld;
    CVI_RT_LoadDmabuf(rt, cm, psize, pa, 0, false, &ld);

    dt = dot_once(); printf("dot AFTER tiu  : %.3fs\n", dt);

    /* ---- 多次 TIU matmul + MemFlushEx (模拟引擎 24 层, ~37k runs/prompt) ---- */
    double t0 = now();
    for (int i = 0; i < 40000; i++) {
        CVI_RT_MemFlushEx(rt, mem, 0);
        CVI_RT_RunCmdbufEx(rt, ld, &(CVI_RT_ARRAYBASE){.gaddr_base0 = pa});
    }
    printf("40000 TIU+flush runs: %.3fs\n", now() - t0);
    dt = dot_once(); printf("dot AFTER 40k TIU: %.3fs  (fmadd100M=%.3fs)\n", dt, fmadd_loop(100000000));
    dt = dot_once(); printf("dot AFTER 40k x2 : %.3fs\n", dt);

    /* ---- 大 SD 顺序读 (模拟权重流, ~200MB) ---- */
    {
        FILE *f = fopen("/data/qwen/layer0_kal.bin", "rb");
        static int8_t sdbuf[1048576];
        long got = 0;
        if (f) {
            t0 = now();
            while (got < 100000000) { size_t n = fread(sdbuf, 1, sizeof sdbuf, f); if (n == 0) break; got += (long)n; }
            printf("SD read 100MB: %.3fs (%.1f MB/s)\n", now() - t0, got / 1e6 / (now() - t0));
            fclose(f);
        } else perror("open layer0");
        dt = dot_once(); printf("dot AFTER SD read: %.3fs  (fmadd100M=%.3fs)\n", dt, fmadd_loop(100000000));
    }

    bmk1822_cleanup(bmk);

    volatile float sink = sc[0] + sc[C - 1];
    printf("sink=%f\n", sink);
    return 0;
}
