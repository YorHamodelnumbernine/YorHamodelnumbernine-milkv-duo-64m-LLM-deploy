/* layer_read_bench.c — Duo 上板: 层权重读路径带宽归因.
 *
 * Phase 7 任务: decode 层权重 201MB/token 实测 ~8MB/s, SD 顺序读天花板 ~21.5MB/s.
 * 本工具对 /data/qwen/layer0..23_kal.bin 实测每种读机制的有效吞吐, 复现 decode
 * "每 token 全量重读 24 层" 的模式 (201MB/pass), 输出归因表.
 *
 * 模式:
 *   1  mmap_bare       裸 mmap + 顺序触页 (Phase 6 当前路径, 无 hint)
 *   2  mmap_madv_seq   mmap + madvise(MADV_SEQUENTIAL)
 *   3  mmap_madv_will  mmap + madvise(MADV_WILLNEED)
 *   4  mmap_readahead  mmap + readahead(fd,0,lsz)  (全文件预取到 page cache)
 *   5  mmap_fadv_will  mmap + posix_fadvise(POSIX_FADV_WILLNEED)
 *   6  pread_1m        pread 1MB 块顺序读入复用 buffer (offset 升序)
 *   7  read_256k       read() 256KB 块 (smollm2_pool_demo 路径)
 *   8  mmap_engine     mmap + 按引擎 eng_matmul 的 tensor/K-block 顺序触页 (无 TIU)
 *   9  pread_odirect   O_DIRECT + 1MB 块 (绕过 page cache, 对齐 buffer)
 *
 * 用法: layer_read_bench [layer_dir] [modes] [--reps N] [--cold]
 *   layer_dir 默认 /data/qwen; modes 如 "1,4,6" 默认全跑; --reps 默认 1.
 *   --cold 每次模式前 drop_caches(3)+sync (需要 root; 板上为 root).
 *
 * 构建: riscv64 交叉, 与引擎同 flags.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>

#define MAXL 32
#define NLAYER 24
#define D 896
#define F 4864
#define DKV 128
#define G 32
#define L 24
#define CHUNK 1048576          /* pread 1MB 块 */
#define RIOBUF 262144          /* read 256KB 块 */

static double now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static volatile uint64_t g_sink = 0;   /* 防优化吞读 */

static long g_lsz[MAXL];
static char g_dir[512];

static void touch_all(const uint8_t *p, long n) {
    uint64_t s = 0;
    for (long i = 0; i + 8 <= n; i += 8) {
        uint64_t v; memcpy(&v, p + i, 8);
        s ^= v;
    }
    g_sink ^= s;
}

/* 引擎 eng_matmul 的读序 (无 TIU): 每个 matmul 按 K-block 读 nib 块 + gsc 块. */
static void touch_engine(const uint8_t *layer, long lsz) {
    long off = 0;
    const float *rms_attn = (const float *)(layer + off); off += D * 4;
    touch_all((const uint8_t *)rms_attn, D * 4);
    /* q,k,v,wo,up,gate,down: nib N*16 字节 + gsc N*2 字节, K-block 序 */
    int Kq = D, Nq = D;
    int Kk = D, Nk = DKV;
    int Kw = D, Nw = D;
    int Ku = D, Nu = F;
    int Kd = F, Nd = D;
    int mats[7][2] = {
        {Kq, Nq}, {Kk, Nk}, {Kk, Nk}, {Kw, Nw},
        {Ku, Nu}, {Ku, Nu}, {Kd, Nd}
    };
    for (int m = 0; m < 7; m++) {
        int K = mats[m][0], N = mats[m][1];
        int KG = K / G;
        const uint8_t *nib = layer + off; off += (size_t)KG * N * 16;
        const uint8_t *gsc = layer + off; off += (size_t)KG * N * 2;
        for (int g = 0; g < KG; g++) {
            touch_all(nib + (size_t)g * N * 16, N * 16);
            touch_all(gsc + (size_t)g * N * 2, N * 2);
        }
    }
    const float *rms_ffn = (const float *)(layer + off); off += D * 4;
    touch_all((const uint8_t *)rms_ffn, D * 4);
    (void)lsz;
}

static void drop_caches(void) {
    int fd = open("/proc/sys/vm/drop_caches", O_WRONLY);
    if (fd < 0) return;
    (void)!write(fd, "3", 1);
    close(fd);
    /* 等回写完成 */
    sync();
    /* 给内核一点时间回收 */
    for (volatile int i = 0; i < 1000000; i++) ;
}

/* ---- 各模式: 返回 24 层总耗时 ---- */
static double mode_mmap_bare(int cold, int reps) {
    double worst = 0;
    for (int r = 0; r < reps; r++) {
        if (cold) drop_caches();
        double t0 = now();
        for (int l = 0; l < L; l++) {
            char path[512]; snprintf(path, sizeof path, "%s/layer%d_kal.bin", g_dir, l);
            int fd = open(path, O_RDONLY);
            if (fd < 0) { fprintf(stderr, "open %s\n", path); exit(2); }
            uint8_t *lm = mmap(NULL, (size_t)g_lsz[l], PROT_READ, MAP_PRIVATE, fd, 0);
            if (lm == MAP_FAILED) { fprintf(stderr, "mmap %s\n", path); exit(2); }
            close(fd);
            touch_all(lm, g_lsz[l]);
            munmap(lm, (size_t)g_lsz[l]);
        }
        double dt = now() - t0;
        if (dt > worst) worst = dt;
    }
    return worst;
}

static double mode_mmap_madv(int cold, int reps, int which) {
    double worst = 0;
    for (int r = 0; r < reps; r++) {
        if (cold) drop_caches();
        double t0 = now();
        for (int l = 0; l < L; l++) {
            char path[512]; snprintf(path, sizeof path, "%s/layer%d_kal.bin", g_dir, l);
            int fd = open(path, O_RDONLY);
            if (fd < 0) { fprintf(stderr, "open %s\n", path); exit(2); }
            uint8_t *lm = mmap(NULL, (size_t)g_lsz[l], PROT_READ, MAP_PRIVATE, fd, 0);
            if (lm == MAP_FAILED) { fprintf(stderr, "mmap %s\n", path); exit(2); }
            madvise(lm, (size_t)g_lsz[l], which);
            close(fd);
            touch_all(lm, g_lsz[l]);
            munmap(lm, (size_t)g_lsz[l]);
        }
        double dt = now() - t0;
        if (dt > worst) worst = dt;
    }
    return worst;
}

static double mode_mmap_readahead(int cold, int reps) {
    double worst = 0;
    for (int r = 0; r < reps; r++) {
        if (cold) drop_caches();
        double t0 = now();
        for (int l = 0; l < L; l++) {
            char path[512]; snprintf(path, sizeof path, "%s/layer%d_kal.bin", g_dir, l);
            int fd = open(path, O_RDONLY);
            if (fd < 0) { fprintf(stderr, "open %s\n", path); exit(2); }
            uint8_t *lm = mmap(NULL, (size_t)g_lsz[l], PROT_READ, MAP_PRIVATE, fd, 0);
            if (lm == MAP_FAILED) { fprintf(stderr, "mmap %s\n", path); exit(2); }
            /* 全文件预取到 page cache (512KB 分片循环, 规避单次 readahead 上限) */
            {
                long off = 0;
                while (off < g_lsz[l]) {
                    size_t n = (size_t)g_lsz[l] - (size_t)off < 524288
                             ? (size_t)g_lsz[l] - (size_t)off : 524288;
                    readahead(fd, off, n);
                    off += n;
                }
            }
            madvise(lm, (size_t)g_lsz[l], MADV_SEQUENTIAL);
            close(fd);
            touch_all(lm, g_lsz[l]);
            munmap(lm, (size_t)g_lsz[l]);
        }
        double dt = now() - t0;
        if (dt > worst) worst = dt;
    }
    return worst;
}

static double mode_mmap_fadv(int cold, int reps) {
    double worst = 0;
    for (int r = 0; r < reps; r++) {
        if (cold) drop_caches();
        double t0 = now();
        for (int l = 0; l < L; l++) {
            char path[512]; snprintf(path, sizeof path, "%s/layer%d_kal.bin", g_dir, l);
            int fd = open(path, O_RDONLY);
            if (fd < 0) { fprintf(stderr, "open %s\n", path); exit(2); }
            uint8_t *lm = mmap(NULL, (size_t)g_lsz[l], PROT_READ, MAP_PRIVATE, fd, 0);
            if (lm == MAP_FAILED) { fprintf(stderr, "mmap %s\n", path); exit(2); }
            posix_fadvise(fd, 0, 0, POSIX_FADV_WILLNEED);
            madvise(lm, (size_t)g_lsz[l], MADV_SEQUENTIAL);
            close(fd);
            touch_all(lm, g_lsz[l]);
            munmap(lm, (size_t)g_lsz[l]);
        }
        double dt = now() - t0;
        if (dt > worst) worst = dt;
    }
    return worst;
}

static double mode_pread(int cold, int reps) {
    uint8_t *buf = malloc((size_t)g_lsz[0]);
    if (!buf) { fprintf(stderr, "oom pread buf\n"); exit(2); }
    double worst = 0;
    for (int r = 0; r < reps; r++) {
        if (cold) drop_caches();
        double t0 = now();
        for (int l = 0; l < L; l++) {
            char path[512]; snprintf(path, sizeof path, "%s/layer%d_kal.bin", g_dir, l);
            int fd = open(path, O_RDONLY);
            if (fd < 0) { fprintf(stderr, "open %s\n", path); exit(2); }
            long remain = g_lsz[l], off = 0;
            while (remain > 0) {
                long n = remain < CHUNK ? remain : CHUNK;
                if (pread(fd, buf + off, n, off) != n) { fprintf(stderr, "pread %s\n", path); exit(2); }
                off += n; remain -= n;
            }
            close(fd);
            touch_all(buf, g_lsz[l]);
        }
        double dt = now() - t0;
        if (dt > worst) worst = dt;
    }
    free(buf);
    return worst;
}

static double mode_read(int cold, int reps) {
    static uint8_t iobuf[RIOBUF];
    uint8_t *dst = malloc((size_t)g_lsz[0]);
    if (!dst) { fprintf(stderr, "oom read buf\n"); exit(2); }
    double worst = 0;
    for (int r = 0; r < reps; r++) {
        if (cold) drop_caches();
        double t0 = now();
        for (int l = 0; l < L; l++) {
            char path[512]; snprintf(path, sizeof path, "%s/layer%d_kal.bin", g_dir, l);
            int fd = open(path, O_RDONLY);
            if (fd < 0) { fprintf(stderr, "open %s\n", path); exit(2); }
            long remain = g_lsz[l]; uint8_t *p = dst;
            while (remain > 0) {
                long n = remain < RIOBUF ? remain : RIOBUF;
                if (read(fd, iobuf, n) != n) { fprintf(stderr, "read %s\n", path); exit(2); }
                memcpy(p, iobuf, n); p += n; remain -= n;
            }
            close(fd);
            touch_all(dst, g_lsz[l]);
        }
        double dt = now() - t0;
        if (dt > worst) worst = dt;
    }
    free(dst);
    return worst;
}

/* O_DIRECT 对照: 绕过 page cache, 直接读到对齐 buffer.
 * 层文件 8393728B = 16394*512, 1MB 块 + 4096 对齐 buffer 满足块对齐约束. */
static double mode_pread_odirect(int cold, int reps) {
    void *raw = NULL; uint8_t *buf;
    if (posix_memalign(&raw, 4096, (size_t)g_lsz[0]) != 0) { fprintf(stderr, "align oom\n"); exit(2); }
    buf = raw;
    double worst = 0;
    for (int r = 0; r < reps; r++) {
        if (cold) drop_caches();
        double t0 = now();
        for (int l = 0; l < L; l++) {
            char path[512]; snprintf(path, sizeof path, "%s/layer%d_kal.bin", g_dir, l);
            int fd = open(path, O_RDONLY | O_DIRECT);
            if (fd < 0) { fprintf(stderr, "open(O_DIRECT) %s: O_DIRECT 可能不支持\n", path); exit(2); }
            long remain = g_lsz[l], off = 0;
            while (remain > 0) {
                long n = remain < CHUNK ? remain : CHUNK;
                if (pread(fd, buf + off, n, off) != n) { fprintf(stderr, "pread(O_DIRECT) %s\n", path); exit(2); }
                off += n; remain -= n;
            }
            close(fd);
            touch_all(buf, g_lsz[l]);
        }
        double dt = now() - t0;
        if (dt > worst) worst = dt;
    }
    free(raw);
    return worst;
}

static double mode_mmap_engine(int cold, int reps) {
    double worst = 0;
    for (int r = 0; r < reps; r++) {
        if (cold) drop_caches();
        double t0 = now();
        for (int l = 0; l < L; l++) {
            char path[512]; snprintf(path, sizeof path, "%s/layer%d_kal.bin", g_dir, l);
            int fd = open(path, O_RDONLY);
            if (fd < 0) { fprintf(stderr, "open %s\n", path); exit(2); }
            uint8_t *lm = mmap(NULL, (size_t)g_lsz[l], PROT_READ, MAP_PRIVATE, fd, 0);
            if (lm == MAP_FAILED) { fprintf(stderr, "mmap %s\n", path); exit(2); }
            close(fd);
            touch_engine(lm, g_lsz[l]);
            munmap(lm, (size_t)g_lsz[l]);
        }
        double dt = now() - t0;
        if (dt > worst) worst = dt;
    }
    return worst;
}

int main(int argc, char **argv) {
    strcpy(g_dir, "/data/qwen");
    int mode_sel[16]; int nmode = 0;
    int reps = 1, cold = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cold")) cold = 1;
        else if (!strcmp(argv[i], "--dir") && i + 1 < argc) { strncpy(g_dir, argv[++i], sizeof g_dir - 1); }
        else {
            /* 逗号分隔的模式号 */
            char *p = strdup(argv[i]);
            char *tok = strtok(p, ",");
            while (tok) { mode_sel[nmode++] = atoi(tok); tok = strtok(NULL, ","); }
            free(p);
        }
    }
    if (nmode == 0) for (int m = 1; m <= 9; m++) mode_sel[nmode++] = m;

    /* 读取层大小 */
    double total = 0;
    for (int l = 0; l < L; l++) {
        char path[512]; snprintf(path, sizeof path, "%s/layer%d_kal.bin", g_dir, l);
        struct stat st;
        if (stat(path, &st)) { fprintf(stderr, "stat %s\n", path); return 2; }
        g_lsz[l] = (long)st.st_size; total += g_lsz[l];
    }
    printf("layer_read_bench: dir=%s layers=%d total=%.1fMB reps=%d cold=%d\n",
           g_dir, L, total / 1048576.0, reps, cold);

    const char *names[10] = {0,
        "mmap_bare", "mmap_madv_seq", "mmap_madv_will",
        "mmap_readahead", "mmap_fadv_will", "pread_1m",
        "read_256k", "mmap_engine", "pread_odirect"};
    printf("%-18s %10s %12s %14s\n", "mode", "sec/pass", "MB/s", "vs_mmap_bare");
    double base = 0;
    for (int i = 0; i < nmode; i++) {
        int m = mode_sel[i];
        if (m < 1 || m > 9) continue;
        double dt;
        switch (m) {
            case 1: dt = mode_mmap_bare(cold, reps); break;
            case 2: dt = mode_mmap_madv(cold, reps, MADV_SEQUENTIAL); break;
            case 3: dt = mode_mmap_madv(cold, reps, MADV_WILLNEED); break;
            case 4: dt = mode_mmap_readahead(cold, reps); break;
            case 5: dt = mode_mmap_fadv(cold, reps); break;
            case 6: dt = mode_pread(cold, reps); break;
            case 7: dt = mode_read(cold, reps); break;
            case 8: dt = mode_mmap_engine(cold, reps); break;
            case 9: dt = mode_pread_odirect(cold, reps); break;
            default: continue;
        }
        double mbs = total / 1048576.0 / dt;
        if (m == 1) base = mbs;
        printf("%-18s %10.3f %12.2f %14.2fx\n", names[m], dt, mbs, mbs / base);
    }
    printf("(SD sequential ceiling ~21.5 MB/s; decode layer read budget 201MB @ 21.5 = ~9.4s)\n");
    return 0;
}
