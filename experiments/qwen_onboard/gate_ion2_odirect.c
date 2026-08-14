/* gate_ion2_odirect.c — G-ION-2: Linux O_DIRECT pread -> ION carveout 验证.
 *
 * 目的 (CEO 立项前置 Gate):
 *   (a) O_DIRECT 对齐: buffer(ION VA) / offset / len 满足 fs 块对齐, pread 成功;
 *   (b) DMA 后一致性: O_DIRECT 直写 ION 后, ION VA 内容 == buffered read 参考 (memcmp);
 *   (c) 带宽复现: layer_read_bench 曾测 pread_odirect=20.57MB/s (读到 malloc 对齐缓冲),
 *       本 probe 验证直接读到 ION carveout 是否可达同量级.
 *
 * 用法: gate_ion2_odirect [layer_path] [reps]
 *   默认 layer_path=/data/qwen/layer0_kal.bin, reps=5.
 *   若 ION VA 不被 direct-IO 接受 (EFAULT), 打印失败原因 -> 选项 A 需降级.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>
#include <sys/stat.h>
#include "cviruntime_context.h"

static inline double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * ts.tv_nsec;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/data/qwen/layer0_kal.bin";
    int reps = argc > 2 ? atoi(argv[2]) : 5;
    if (reps < 1) reps = 1;

    struct stat st;
    if (stat(path, &st) != 0) { perror("stat"); return 2; }
    size_t fsz = (size_t)st.st_size;
    long fsblk = st.st_blksize > 0 ? st.st_blksize : 512;
    printf("G-ION-2 O_DIRECT->ION  file=%s size=%zu  st_blksize=%ld\n", path, fsz, fsblk);

    /* --- 参考数据: buffered read 全量 --- */
    uint8_t *ref = malloc(fsz);
    if (!ref) { fprintf(stderr, "oom ref\n"); return 2; }
    {
        int fd = open(path, O_RDONLY);
        if (fd < 0) { perror("open ref"); return 2; }
        size_t off = 0;
        while (off < fsz) {
            ssize_t n = read(fd, ref + off, fsz - off);
            if (n <= 0) { perror("read ref"); return 2; }
            off += (size_t)n;
        }
        close(fd);
    }

    /* --- ION alloc --- */
    CVI_RT_HANDLE rt;
    if (CVI_RT_Init(&rt) != 0) { fprintf(stderr, "CVI_RT_Init fail\n"); return 2; }
    CVI_RT_MEM ion = CVI_RT_MemAlloc(rt, fsz + 4096);
    if (!ion) { fprintf(stderr, "ION alloc %zu fail\n", fsz); return 2; }
    uint8_t *va = CVI_RT_MemGetVAddr(ion);
    uint64_t pa = CVI_RT_MemGetPAddr(ion);
    printf("ION alloc OK va=%p pa=%#llx  (page-align va=%lu off=%lu)\n",
           (void*)va, (unsigned long long)pa,
           (unsigned long)((uintptr_t)va & 4095), (unsigned long)((uintptr_t)va & (fsblk - 1)));

    /* --- O_DIRECT: 对齐验证 --- */
    int ofd = open(path, O_RDONLY | O_DIRECT);
    if (ofd < 0) { fprintf(stderr, "open(O_DIRECT) FAILED errno=%d (%s)\n", errno, strerror(errno)); return 3; }
    printf("open(O_DIRECT) OK\n");

    /* 故意错位 offset 验证对齐约束 (期望 EINVAL) */
    {
        ssize_t r = pread(ofd, va, fsblk, 1);   /* offset 1 未对齐 */
        printf("  [align-probe] misaligned offset=1: r=%zd errno=%d (%s)  %s\n",
               r, errno, strerror(errno), (r < 0 && errno == EINVAL) ? "OK: O_DIRECT 强制对齐" : "UNEXPECTED");
    }

    /* --- 正式读: 1MiB chunk 顺序 pread 到 ION VA, 计时 --- */
    const size_t CHUNK = 1u << 20;
    double best = 0, sum = 0;
    long bad = -1;
    for (int it = 0; it < reps; it++) {
        double t0 = now_s();
        size_t off = 0;
        while (off < fsz) {
            size_t n = (fsz - off > CHUNK) ? CHUNK : (fsz - off);
            ssize_t r = pread(ofd, va + off, n, (off_t)off);
            if (r != (ssize_t)n) {
                fprintf(stderr, "  pread(O_DIRECT) off=%zu n=%zu r=%zd errno=%d (%s)\n",
                        off, n, r, errno, strerror(errno));
                bad = (long)off; break;
            }
            off += n;
        }
        double dt = now_s() - t0;
        if (bad < 0) {
            double bw = (double)fsz / 1e6 / dt;
            printf("  iter %d: %zu bytes in %.3f s -> %.2f MB/s\n", it, fsz, dt, bw);
            if (bw > best) best = bw;
            sum += bw;
        } else break;
    }
    if (bad >= 0) {
        fprintf(stderr, "O_DIRECT->ION pread FAILED at off=%ld\n", bad);
        fprintf(stderr, "  => 结论: ION VA 不被 direct-IO 接受, 选项 A 需降级(malloc对齐+memcpy) 或走选项 B.\n");
        return 4;
    }

    /* --- 一致性: ION VA vs 参考 --- */
    CVI_RT_MemInvld(rt, ion);
    long nbad = 0, first = -1;
    for (size_t i = 0; i < fsz; i++) {
        if (va[i] != ref[i]) { if (nbad == 0) first = (long)i; nbad++; if (nbad > 8) break; }
    }
    if (nbad == 0) printf("CONSISTENCY: OK (ION VA == buffered ref, all %zu bytes)\n", fsz);
    else printf("CONSISTENCY: BAD nbad=%ld first=%ld (0x%lx)\n", nbad, first, (unsigned long)first);

    printf("BANDWIDTH: best=%.2f MB/s  avg=%.2f MB/s  (reps=%d, chunk=1MiB, direct-to-ION)\n",
           best, reps ? sum / reps : 0, reps);

    CVI_RT_MemFree(rt, ion);
    CVI_RT_DeInit(rt);
    free(ref);
    return 0;
}
