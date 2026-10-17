/* fadv_probe.c — 诊断 posix_fadvise(DONTNEED) 在 Duo 上的 errno (B-P0 dn_err 全失败).
 * 用法: ./fadv_probe <layer_file> [off len]...
 * 无段参数时用 B-P0 的 7 nibble 段表. 打印每次 fadvise 的返回值与 errno. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <stdint.h>

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <file> [off len]...\n", argv[0]); return 2; }
    const char *path = argv[1];
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return 2; }
    struct { off_t off; off_t len; } seg[7];
    int nseg = 0;
    if (argc >= 4) {
        for (int i = 2; i + 1 < argc && nseg < 7; i += 2) {
            seg[nseg].off = atoll(argv[i]); seg[nseg].len = atoll(argv[i+1]); nseg++;
        }
    } else {
        static const size_t offs[7] = {3584,455168,519680,584192,1035776,3487232,5938688};
        static const size_t lens[7] = {401408,57344,57344,401408,2179072,2179072,2179072};
        for (int i = 0; i < 7; i++) { seg[i].off = (off_t)offs[i]; seg[i].len = (off_t)lens[i]; }
        nseg = 7;
    }
    printf("fd=%d nseg=%d file=%s\n", fd, nseg, path);
    for (int i = 0; i < nseg; i++) {
        errno = 0;
        int r = posix_fadvise(fd, seg[i].off, seg[i].len, POSIX_FADV_DONTNEED);
        printf("  seg[%d] off=%lld len=%lld -> posix_fadvise=%d errno=%d (%s)\n",
               i, (long long)seg[i].off, (long long)seg[i].len, r, errno, strerror(errno));
    }
    /* 对照: readahead 也打一次 errno */
    errno = 0;
    long rr = readahead(fd, (off_t)3584, (size_t)401408);
    printf("  readahead seg0 -> %ld errno=%d (%s)\n", rr, errno, strerror(errno));
    close(fd);
    return 0;
}
