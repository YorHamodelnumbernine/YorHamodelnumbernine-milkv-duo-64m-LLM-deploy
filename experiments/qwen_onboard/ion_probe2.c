/* ion_probe2.c — measure ION carveout headroom: after one CVI_RT_Init, how many
 * 1MB allocations fit?  Prints cumulative alloc before each step. */
#include <stdio.h>
#include <stdlib.h>
#include "cviruntime_context.h"

#define MAXN 40
int main(void) {
    printf("== ion_probe2 ==\n"); fflush(stdout);
    CVI_RT_HANDLE rt; CVI_RT_Init(&rt);
    CVI_RT_MEM arr[MAXN];
    int n = 0;
    for (; n < MAXN; n++) {
        CVI_RT_MEM m = CVI_RT_MemAlloc(rt, 1048576);
        if (!m) break;
        arr[n] = m;
        printf("  #%02d 1MB OK\n", n); fflush(stdout);
    }
    printf("fit %d x 1MB after Init\n", n); fflush(stdout);
    for (int i = 0; i < n; i++) CVI_RT_MemFree(rt, arr[i]);
    /* now 256KB (like engine main buffer) + 1MB cmdbuf chunks */
    CVI_RT_MEM m256 = CVI_RT_MemAlloc(rt, 262144);
    CVI_RT_MEM m1a = CVI_RT_MemAlloc(rt, 1048576);
    CVI_RT_MEM m1b = CVI_RT_MemAlloc(rt, 1048576);
    CVI_RT_MEM m1c = CVI_RT_MemAlloc(rt, 1048576);
    printf("after free: 256KB=%s 1MB=%s 1MB=%s 1MB=%s\n",
           m256 ? "OK" : "FAIL", m1a ? "OK" : "FAIL", m1b ? "OK" : "FAIL", m1c ? "OK" : "FAIL");
    fflush(stdout);
    if (m1c) CVI_RT_MemFree(rt, m1c);
    if (m1b) CVI_RT_MemFree(rt, m1b);
    if (m1a) CVI_RT_MemFree(rt, m1a);
    if (m256) CVI_RT_MemFree(rt, m256);
    CVI_RT_DeInit(rt);
    printf("== done ==\n");
    return 0;
}
