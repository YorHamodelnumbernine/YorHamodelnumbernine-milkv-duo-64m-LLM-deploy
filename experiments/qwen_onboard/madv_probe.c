/* madv_probe.c — 诊断 madvise(MADV_DONTNEED) 释放文件页是否可用 (B-P0 释放机制替代).
 * 流程: mmap 层文件 MAP_PRIVATE -> 触碰 nibble 段页 -> mincore 驻留快照
 *       -> madvise(MADV_DONTNEED) 整文件 -> 再 mincore 看是否被丢弃
 *       -> 再次触碰 测 majflt 是否重现 (真释放) vs 无释放 (仍驻留).
 * 另对照 MADV_WILLNEED 预取有效性与 madvise 的 errno. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/resource.h>

static long majflt_delta(void) {
    struct rusage ru0, ru1; getrusage(RUSAGE_SELF, &ru0);
    return 0; /* caller does delta */
}
int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/data/qwen/layer0_kal.bin";
    size_t sz = (size_t)(argc > 2 ? atoll(argv[2]) : 8393728);
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return 2; }
    unsigned char *m = mmap(NULL, sz, PROT_READ, MAP_PRIVATE, fd, 0);
    if (m == MAP_FAILED) { perror("mmap"); return 2; }
    size_t npg = (sz + 4095) / 4096;
    unsigned char *mc = calloc(npg, 1);
    volatile unsigned char sink = 0;

    /* 1. mincore 前 (应全 0, 未驻留) */
    mincore(m, sz, mc); size_t res0 = 0; for (size_t i = 0; i < npg; i++) if (mc[i] & 1) res0++;
    /* 2. 触碰 nibble 段页 (7 段表) */
    static const size_t offs[7] = {3584,455168,519680,584192,1035776,3487232,5938688};
    static const size_t lens[7] = {401408,57344,57344,401408,2179072,2179072,2179072};
    for (int i = 0; i < 7; i++)
        for (size_t o = offs[i]; o < offs[i] + lens[i]; o += 4096) sink += m[o];
    mincore(m, sz, mc); size_t res1 = 0; for (size_t i = 0; i < npg; i++) if (mc[i] & 1) res1++;
    /* 3. madvise advice (argv[3] 可选覆盖: dontneed|willneed|sequential|normal) */
    int adv = MADV_DONTNEED; const char *advn = "DONTNEED";
    if (argc > 3) {
        if (!strcmp(argv[3], "willneed")) { adv = MADV_WILLNEED; advn = "WILLNEED"; }
        else if (!strcmp(argv[3], "sequential")) { adv = MADV_SEQUENTIAL; advn = "SEQUENTIAL"; }
        else if (!strcmp(argv[3], "normal")) { adv = MADV_NORMAL; advn = "NORMAL"; }
    }
    int r = madvise(m, sz, adv);
    int en = errno;
    mincore(m, sz, mc); size_t res2 = 0; for (size_t i = 0; i < npg; i++) if (mc[i] & 1) res2++;
    /* 4. 再触碰 -> 若真释放则 majflt 重现 */
    struct rusage a0, a1; getrusage(RUSAGE_SELF, &a0);
    for (int i = 0; i < 7; i++)
        for (size_t o = offs[i]; o < offs[i] + lens[i]; o += 4096) sink += m[o];
    getrusage(RUSAGE_SELF, &a1);
    long mf_retouch = a1.ru_majflt - a0.ru_majflt;

    printf("file=%s sz=%zu npg=%zu\n", path, sz, npg);
    printf("madvise(MADV_%s)=%d errno=%d (%s)\n", advn, r, en, strerror(en));
    printf("resident before-touch=%zu touched=%zu after-DONTNEED=%zu (pages)\n", res0, res1, res2);
    printf("majflt on retouch after DONTNEED=%ld (0 => pages NOT released)\n", mf_retouch);
    munmap(m, sz); close(fd); free(mc);
    (void)majflt_delta;
    return 0;
}
