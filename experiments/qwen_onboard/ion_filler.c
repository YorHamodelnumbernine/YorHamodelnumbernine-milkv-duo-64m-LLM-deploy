/* ion_filler.c — 测试工具: 从 ION carveout 分配并持有指定字节数 (模拟 carveout 占用).
 * 用法: ion_filler <bytes> <hold_seconds>
 * 用于验证 GSC_ION=1 在 carveout 余量不足时干净回退冷 mmap (不 SIGABRT/不泄漏). */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "cviruntime_context.h"

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: ion_filler <bytes> <hold_sec>\n"); return 2; }
    size_t n = (size_t)atol(argv[1]);
    int hold = atoi(argv[2]);
    CVI_RT_HANDLE rt; CVI_RT_Init(&rt);
    CVI_RT_MEM m = CVI_RT_MemAlloc(rt, n);
    if (!m) { fprintf(stderr, "filler alloc %zu B FAILED\n", n); return 1; }
    uint8_t *p = CVI_RT_MemGetVAddr(m);
    for (size_t i = 0; i < n; i += 4096) p[i] = 1;   /* touch */
    printf("filler held %zu B for %ds\n", n, hold);
    fflush(stdout);
    sleep(hold);
    CVI_RT_MemFree(rt, m);
    CVI_RT_DeInit(rt);
    return 0;
}
