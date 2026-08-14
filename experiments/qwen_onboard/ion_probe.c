/* ion_probe.c — minimal ION alloc probe. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cviruntime_context.h"

static void try_alloc(const char *tag) {
    CVI_RT_HANDLE rt; CVI_RT_Init(&rt);
    CVI_RT_MEM mem = CVI_RT_MemAlloc(rt, 262144);
    if (mem) {
        printf("%s: MemAlloc(256KB) OK pa=0x%llx\n", tag, (unsigned long long)CVI_RT_MemGetPAddr(mem));
        CVI_RT_MemFree(rt, mem);
    } else {
        printf("%s: MemAlloc(256KB) FAILED\n", tag);
    }
    CVI_RT_DeInit(rt);
}

int main(void) {
    printf("== ion_probe ==\n"); fflush(stdout);
    try_alloc("A immediate            "); fflush(stdout);
    void *p = malloc(8393728); memset(p, 0x55, 8393728);
    try_alloc("B after 8.39MB malloc  "); fflush(stdout);
    free(p);
    printf("== done ==\n");
    return 0;
}
