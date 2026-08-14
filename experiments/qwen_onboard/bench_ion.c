/* bench_ion.c — test ION: max alloc size, CPU write/read bandwidth (cached?).
 * Allocates a large ION buffer, fills it, measures sequential read/write.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include "cviruntime_context.h"

static inline double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * ts.tv_nsec;
}

int main(int argc, char **argv) {
    size_t MB = argc > 1 ? (size_t)atoi(argv[1]) : 21;
    size_t bytes = MB << 20;
    CVI_RT_HANDLE rt; if (CVI_RT_Init(&rt) != 0) { fprintf(stderr, "rt init fail\n"); return 2; }
    CVI_RT_MEM mem = CVI_RT_MemAlloc(rt, bytes);
    if (!mem) { fprintf(stderr, "ION alloc %zu MB FAILED\n", MB); CVI_RT_DeInit(rt); return 1; }
    uint8_t *va = CVI_RT_MemGetVAddr(mem);
    uint64_t pa = CVI_RT_MemGetPAddr(mem);
    printf("ION alloc %zu MB OK  va=%p pa=%#llx\n", MB, (void*)va, (unsigned long long)pa);

    /* fill */
    double t0 = now_s();
    for (size_t i = 0; i < bytes; i += 64) { uint64_t *p = (uint64_t *)(va + i); *p = (uint64_t)i; }
    double tfill = now_s() - t0;
    CVI_RT_MemFlush(rt, mem);
    printf("fill  %8.2f MB -> %8.1f MB/s\n", (double)bytes / 1e6, (double)bytes / 1e6 / tfill);

    /* read: sum bytes */
    volatile uint64_t acc = 0;
    t0 = now_s();
    for (size_t i = 0; i < bytes; i += 64) acc += *(uint64_t *)(va + i);
    double tread = now_s() - t0;
    printf("read  %8.2f MB -> %8.1f MB/s  (acc=%llu)\n", (double)bytes / 1e6, (double)bytes / 1e6 / tread, (unsigned long long)acc);

    /* read again (hot) */
    t0 = now_s();
    for (size_t i = 0; i < bytes; i += 64) acc += *(uint64_t *)(va + i);
    double tread2 = now_s() - t0;
    printf("read2 %8.2f MB -> %8.1f MB/s\n", (double)bytes / 1e6, (double)bytes / 1e6 / tread2);

    CVI_RT_MemFree(rt, mem);
    CVI_RT_DeInit(rt);
    printf("done\n");
    return 0;
}
