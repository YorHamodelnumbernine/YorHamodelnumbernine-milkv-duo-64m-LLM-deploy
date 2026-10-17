/* ion_frag_probe.c — 验证 carveout 碎片化假设:
 *   prefill/decode run 释放 pools(大量小 cmdbuf) 后, 大块连续 free 区域是否仍能容纳
 *   下一次 run 的 gsc 单块 21.33MB 分配.
 * 流程: init -> alloc gsc(22,364,160) OK? -> free gsc -> alloc N 个小块模拟 pools
 *        -> free pools -> 再次 alloc gsc OK?  (若第二次 FAIL => 碎片化根因)
 * 用法: ion_frag_probe [iterations]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "cviruntime_context.h"

#define GSC_BYTES 22364160u
#define NEURON    262144u

int main(int argc, char **argv) {
    int iters = argc > 1 ? atoi(argv[1]) : 5;
    for (int it = 0; it < iters; it++) {
        CVI_RT_HANDLE rt; CVI_RT_Init(&rt);
        CVI_RT_MEM neu = CVI_RT_MemAlloc(rt, NEURON);
        CVI_RT_MEM g1 = CVI_RT_MemAlloc(rt, GSC_BYTES);
        printf("[%d] gsc#1 = %s\n", it, g1 ? "OK" : "FAIL"); fflush(stdout);
        if (g1) CVI_RT_MemFree(rt, g1);

        /* 模拟 pools: 随机大小小块, 总量 ~1.7MB, 再 + merged ~0.26MB */
        CVI_RT_MEM arr[512]; int n = 0;
        size_t total = 0;
        unsigned seed = (unsigned)(it * 7919 + 13);
        while (total < 2000000u && n < 500) {
            size_t sz = 2048 + (size_t)(rand_r(&seed) % 30000);  /* 2KB..32KB */
            CVI_RT_MEM m = CVI_RT_MemAlloc(rt, sz);
            if (!m) { printf("[%d] pool#%d alloc %zu FAIL\n", it, n, sz); break; }
            arr[n++] = m; total += sz;
        }
        printf("[%d] pools allocated %d chunks / %zu B\n", it, n, total); fflush(stdout);
        for (int i = 0; i < n; i++) CVI_RT_MemFree(rt, arr[i]);

        CVI_RT_MEM g2 = CVI_RT_MemAlloc(rt, GSC_BYTES);
        printf("[%d] gsc#2 (after pools free) = %s  <= 碎片化根因判定\n",
               it, g2 ? "OK" : "FAIL"); fflush(stdout);
        if (g2) CVI_RT_MemFree(rt, g2);
        CVI_RT_MemFree(rt, neu);
        CVI_RT_DeInit(rt);
        printf("[%d] iter done\n", it); fflush(stdout);
    }
    return 0;
}
