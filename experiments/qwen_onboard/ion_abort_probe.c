/* ion_abort_probe.c — 诊断 SIGABRT / SDK reopen-retry 是否泄漏 ION carveout.
 * mode 0: 分配 1MB -> abort()            (测 SIGABRT 本身是否泄漏)
 * mode 1: 分配 1MB -> 再分配 30MB 触发 reopen-retry -> __assert_fail
 *         (测 SDK mem_alloc_raw 重试/assert 路径是否泄漏已分配的 1MB)
 * 用法: ion_abort_probe <mode>
 * 调用方在运行前/后读 cvi_carveout_heap_dump/summary 对比 used.
 */
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include "cviruntime_context.h"

int main(int argc, char **argv) {
    int mode = argc > 1 ? atoi(argv[1]) : 0;
    CVI_RT_HANDLE rt; CVI_RT_Init(&rt);
    CVI_RT_MEM m1 = CVI_RT_MemAlloc(rt, 1048576);
    if (!m1) { fprintf(stderr, "1MB alloc FAILED\n"); return 2; }
    printf("alloc 1MB OK  va=%p\n", (void*)CVI_RT_MemGetVAddr(m1)); fflush(stdout);

    if (mode == 1) {
        fprintf(stderr, "about to alloc 30MB (expect SDK reopen-retry + assert)\n"); fflush(stderr);
        CVI_RT_MEM big = CVI_RT_MemAlloc(rt, 30u << 20);
        fprintf(stderr, "unexpected: 30MB alloc returned %p (should have asserted)\n", (void*)big);
        if (big) CVI_RT_MemFree(rt, big);
    } else {
        fprintf(stderr, "about to abort()\n"); fflush(stderr);
        abort();
    }
    CVI_RT_MemFree(rt, m1);
    CVI_RT_DeInit(rt);
    fprintf(stderr, "clean exit (no abort path)\n");
    return 0;
}
