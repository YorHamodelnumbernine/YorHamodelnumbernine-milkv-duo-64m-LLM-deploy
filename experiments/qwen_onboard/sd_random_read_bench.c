/* sd_random_read_bench.c — Duo 上板: 度量 embed_i8_cl.bin 的散读吞吐.
 *
 * 模拟 LM head Stage2 的访问模式:
 *   用真实 clust_idx.bin (1024 簇 span) + 真实候选簇选择, 对 11 prompt 的 top-Kc
 *   簇 span 做 pread, 度量: 随机序 vs offset 升序 的有效吞吐.
 * 用法: sd_random_read_bench <embed_cl> <clust_idx> <ncand>
 *   ncand: 用多少个 top 簇模拟 (实际是取随机 128/64 个簇 span)
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>

#define V 151936
#define D 896
#define C 1024

static double now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s <embed_cl> <clust_idx> <nspan>\n", argv[0]); return 2; }
    int nspan = atoi(argv[3]);
    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) { perror("open embed_cl"); return 2; }
    FILE *cf = fopen(argv[2], "rb");
    if (!cf) { perror("open clust_idx"); return 2; }
    int32_t idx[C * 2];
    if (fread(idx, 4, C * 2, cf) != C * 2) { perror("read idx"); return 2; }
    fclose(cf);

    /* 候选 span: 随机取 nspan 个非空簇 (span 长度分布与真实一致) */
    int chosen[C]; int nch = 0;
    srand(42);
    for (int c = 0; c < C && nch < nspan; c++) {
        int r = rand() % C;
        if (idx[r * 2 + 1] > 0) chosen[nch++] = r;
    }
    /* 按 (byte_off, byte_len) 组织 */
    typedef struct { long off; long len; } Span;
    Span spans[4096];
    for (int i = 0; i < nch; i++) {
        int c = chosen[i];
        spans[i].off = (long)idx[c * 2 + 0] * D;
        spans[i].len = (long)idx[c * 2 + 1] * D;
    }
    long total_bytes = 0; for (int i = 0; i < nch; i++) total_bytes += spans[i].len;

    static int8_t buf[1024 * 1024];

    /* 模式 1: 原始顺序 (模拟 mmap 触页, 无排序) */
    double t0 = now();
    long got = 0;
    for (int i = 0; i < nch; i++) {
        pread(fd, buf, spans[i].len, spans[i].off); got += spans[i].len;
    }
    double dt1 = now() - t0;
    printf("random-order: %ld B in %.3fs = %.2f MB/s\n", got, dt1, got / 1e6 / dt1);

    /* 模式 2: offset 升序 */
    /* 简单插入排序 */
    Span s2[4096]; memcpy(s2, spans, sizeof(spans[0]) * nch);
    for (int i = 1; i < nch; i++) {
        Span key = s2[i]; int j = i - 1;
        while (j >= 0 && s2[j].off > key.off) { s2[j + 1] = s2[j]; j--; }
        s2[j + 1] = key;
    }
    t0 = now();
    got = 0;
    for (int i = 0; i < nch; i++) {
        pread(fd, buf, s2[i].len, s2[i].off); got += s2[i].len;
    }
    double dt2 = now() - t0;
    printf("sorted-order: %ld B in %.3fs = %.2f MB/s\n", got, dt2, got / 1e6 / dt2);

    /* 模式 3: 顺序读同量字节 (对照地板) */
    t0 = now();
    got = 0;
    long off = 0;
    while (got < total_bytes) {
        long n = total_bytes - got; if (n > 1048576) n = 1048576;
        pread(fd, buf, n, off); got += n; off += n;
    }
    double dt3 = now() - t0;
    printf("sequential  : %ld B in %.3fs = %.2f MB/s\n", got, dt3, got / 1e6 / dt3);

    close(fd);
    return 0;
}
