/* stage1_bench.c — Duo 上板: 度量 stage1 的 f32 vs f64 点积速度 + 时钟分辨率.
 * 用法: stage1_bench   (无参; C=1024, D=896 与引擎一致)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#define C 1024
#define D 896

static double now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

int main(void) {
    float *cent = malloc((size_t)C * D * sizeof(float));
    float *h = malloc(D * sizeof(float));
    if (!cent || !h) { fprintf(stderr, "oom\n"); return 2; }
    for (int i = 0; i < C * D; i++) cent[i] = (float)((i * 2654435761u) % 1000) / 1000.0f - 0.5f;
    for (int j = 0; j < D; j++) h[j] = (float)((j * 40503u) % 1000) / 1000.0f - 0.5f;

    /* 时钟分辨率 */
    double t0 = now(); volatile double t1 = now();
    printf("clock_gettime overhead ~%.1fns\n", (t1 - t0) * 1e9);

    /* ---- f32 标量 (与编译生成的 stage1 内循环一致) ---- */
    static float sc[C];
    t0 = now();
    for (int rep = 0; rep < 5; rep++)
        for (int c = 0; c < C; c++) {
            const float *cd = cent + (size_t)c * D;
            float s = 0;
            for (int j = 0; j < D; j++) s += h[j] * cd[j];
            sc[c] = s;
        }
    double dt32 = now() - t0;
    double macs = 5.0 * C * D;
    printf("f32 scalar: %d iter %.3fs = %.2f MMAC/s  (%.3f ms/iter)\n",
           (int)(5 * C * D), dt32, macs / 1e6 / dt32, dt32 / 5 * 1000);

    /* ---- f64 标量 (double 累加对照) ---- */
    static double scd[C];
    t0 = now();
    for (int rep = 0; rep < 5; rep++)
        for (int c = 0; c < C; c++) {
            const float *cd = cent + (size_t)c * D;
            double s = 0;
            for (int j = 0; j < D; j++) s += (double)h[j] * (double)cd[j];
            scd[c] = s;
        }
    double dt64 = now() - t0;
    printf("f64 scalar: %d iter %.3fs = %.2f MMAC/s  (%.3f ms/iter)\n",
           (int)(5 * C * D), dt64, macs / 1e6 / dt64, dt64 / 5 * 1000);

    /* ---- f32 4路展开 (打破依赖链, 4 个累加器) ---- */
    float s0, s1, s2, s3;
    t0 = now();
    for (int rep = 0; rep < 5; rep++)
        for (int c = 0; c < C; c++) {
            const float *cd = cent + (size_t)c * D;
            s0 = s1 = s2 = s3 = 0;
            for (int j = 0; j < D; j += 4) {
                s0 += h[j + 0] * cd[j + 0];
                s1 += h[j + 1] * cd[j + 1];
                s2 += h[j + 2] * cd[j + 2];
                s3 += h[j + 3] * cd[j + 3];
            }
            sc[c] = (s0 + s1) + (s2 + s3);
        }
    double dt32u = now() - t0;
    printf("f32 unroll4: %d iter %.3fs = %.2f MMAC/s  (%.3f ms/iter)\n",
           (int)(5 * C * D), dt32u, macs / 1e6 / dt32u, dt32u / 5 * 1000);

    volatile float sink = sc[0] + scd[0] + sc[C - 1];
    printf("sink=%f\n", sink);
    free(cent); free(h);
    return 0;
}
