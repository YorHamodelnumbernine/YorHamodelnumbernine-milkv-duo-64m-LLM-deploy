/* gate_ion1_sdhci.c — G-ION-1: 副核 C906L SDHCI 裸读带宽 (SD -> ION DMA) 实测.
 *
 * 目的 (CEO 立项前置 Gate):
 *   副核 FreeRTOS 已内置 CMD_MHA_SD_TAKE_OWNER/SD_READ_LAYER/SD_RELEASE
 *   (sdhci_cv180x.c CMD18+ADMA2), 但从未在 Linux 侧调用/实测.
 *   本 probe: 文件->LBA 映射 (FIEMAP/FIBMAP) -> 副核 SDHCI 直读 layer 到 ION
 *   carveout -> 一致性 memcmp (vs O_DIRECT 参考) -> 裸读带宽.
 *   判定: >=18MB/s 走选项 B; <15MB/s 二期砍掉.
 *
 * 用法: gate_ion1_sdhci [layer_path] [reps] [part_start_sector_override]
 *   默认 layer_path=/data/qwen/layer0_kal.bin, reps=3.
 *
 * 注意: 副核 TAKE_OWNER 会重初始化 SDHCI 控制器, RELEASE 后 SDHCI_RESET_ALL;
 *       Linux 侧 mmc 驱动状态可能受影响 (只读不写, 风险可控; 若后续 IO 异常请重启).
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>
#include <dirent.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include "cviruntime_context.h"
#include "rtos_cmdqu.h"
#include "mha_descriptor.h"

/* ---- 手动定义 FIEMAP (musl sysroot 无 linux/fiemap.h) ---- */
struct fiemap_extent {
    uint64_t fe_logical;
    uint64_t fe_physical;
    uint64_t fe_length;
    uint64_t fe_reserved64[2];
    uint32_t fe_flags;
    uint32_t fe_reserved[3];
};
struct fiemap {
    uint64_t fm_start;
    uint64_t fm_length;
    uint32_t fm_flags;
    uint32_t fm_mapped_extents;
    uint32_t fm_extent_count;
    uint32_t fm_reserved;
    struct fiemap_extent fm_extents[0];
};
#define FS_IOC_FIEMAP _IOWR('f', 11, struct fiemap)
#define FIEMAP_FLAG_SYNC 0x00000001
#define FIEMAP_EXTENT_LAST 0x00000001
#define FIBMAP _IO(0x00, 1)

/* ---- mbox helpers (mirror smollm2_pool_demo.c) ---- */
static int mbox_fd = -1;
static int mbox_open(void) {
    if (mbox_fd >= 0) return 0;
    mbox_fd = open("/dev/cvi-rtos-cmdqu", O_RDWR);
    return (mbox_fd < 0) ? -1 : 0;
}
static int mbox_send_async(cmdqu_t *cmdq) {
    if (mbox_fd < 0) return -1;
    return ioctl(mbox_fd, RTOS_CMDQU_SEND, cmdq);
}
static int mbox_send_cmd(CVI_RT_HANDLE rt, CVI_RT_MEM ion, uint8_t *ion_va,
                         uint64_t ion_pa, int cmd_id, uint32_t param_pa) {
    uint8_t *nm = ion_va;
    uint64_t nm_pa = ion_pa;
    cmdqu_t cmdq;
    memset(&cmdq, 0, sizeof(cmdq));
    cmdq.ip_id     = IP_SYSTEM;
    cmdq.cmd_id    = (uint8_t)cmd_id;
    cmdq.block     = 0;
    cmdq.param_ptr = param_pa;
    (void)nm; (void)nm_pa;
    return mbox_send_async(&cmdq);
}
/* poll desc.result; 0=success <0=error; resets result=-1 */
static int mbox_poll_desc(CVI_RT_HANDLE rt, CVI_RT_MEM ion, mha_dma_desc_t *d, int timeout_us) {
    int waited = 0;
    while (1) {
        CVI_RT_MemInvld(rt, ion);
        if (d->result != -1) break;
        usleep(10);
        waited += 10;
        if (timeout_us > 0 && waited >= timeout_us) {
            fprintf(stderr, "  MBOX poll timeout %d us result=%d\n", waited, d->result);
            return -1;
        }
    }
    int rc = d->result;
    d->result = -1;
    return rc;
}

static inline double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * ts.tv_nsec;
}

/* ---- 设备/分区检测: 由 st_dev(major:minor) 找 /sys/block 下对应分区 start ---- */
static long find_part_start(dev_t dev) {
    unsigned dmaj = (unsigned)major(dev);
    unsigned dmin = (unsigned)minor(dev);
    char want[64]; snprintf(want, sizeof(want), "%u:%u\n", dmaj, dmin);
    DIR *d = opendir("/sys/block");
    if (!d) return -1;
    struct dirent *e;
    long start = -1;
    char path[512];
    while ((e = readdir(d)) != NULL) {
        if (e->d_name[0] == '.') continue;
        snprintf(path, sizeof(path), "/sys/block/%s", e->d_name);
        DIR *sub = opendir(path);
        if (!sub) continue;
        struct dirent *p;
        while ((p = readdir(sub)) != NULL) {
            if (p->d_name[0] == '.') continue;
            snprintf(path, sizeof(path), "/sys/block/%s/%s/dev", e->d_name, p->d_name);
            FILE *f = fopen(path, "r");
            if (!f) continue;
            char buf[64]; if (fgets(buf, sizeof(buf), f)) {
                if (strcmp(buf, want) == 0) {
                    snprintf(path, sizeof(path), "/sys/block/%s/%s/start", e->d_name, p->d_name);
                    FILE *fs = fopen(path, "r");
                    if (fs) { if (fgets(buf, sizeof(buf), fs)) start = atol(buf); fclose(fs); }
                    fclose(f);
                    closedir(sub); closedir(d);
                    return start;
                }
            }
            fclose(f);
        }
        closedir(sub);
    }
    closedir(d);
    return start;
}

/* ---- 文件->物理扇区(相对分区) 映射, 返回 extent 列表 ---- */
typedef struct { uint64_t logical; uint64_t phys_sector; uint64_t bytes; } ext_t;
static int map_file_extents(int fd, size_t fsz, long fsblk, ext_t *exts, int maxext) {
    /* 优先 FIEMAP */
    struct fiemap *fm = calloc(1, sizeof(struct fiemap) + 64 * sizeof(struct fiemap_extent));
    if (!fm) return -1;
    uint64_t start = 0;
    int n = 0;
    while (start < fsz && n < maxext) {
        memset(fm, 0, sizeof(struct fiemap) + 64 * sizeof(struct fiemap_extent));
        fm->fm_start = start;
        fm->fm_length = fsz - start;
        fm->fm_flags = FIEMAP_FLAG_SYNC;
        fm->fm_extent_count = 64;
        if (ioctl(fd, FS_IOC_FIEMAP, fm) < 0) {
            free(fm);
            fprintf(stderr, "  FIEMAP not supported (errno=%d), fallback FIBMAP\n", errno);
            goto fibmap;
        }
        if (fm->fm_mapped_extents == 0) break;
        for (uint32_t i = 0; i < fm->fm_mapped_extents && n < maxext; i++) {
            struct fiemap_extent *fe = &fm->fm_extents[i];
            exts[n].logical = fe->fe_logical;
            exts[n].phys_sector = fe->fe_physical / 512;
            exts[n].bytes = fe->fe_length;
            n++;
        }
        start = fm->fm_extents[fm->fm_mapped_extents - 1].fe_logical
              + fm->fm_extents[fm->fm_mapped_extents - 1].fe_length;
    }
    free(fm);
    return n;

fibmap: {
        /* FIBMAP: input fs-block idx (fsblk bytes), output sector(512B) rel partition */
        int nf = 0;
        for (uint64_t off = 0; off < fsz && nf < maxext; off += (uint64_t)fsblk) {
            int blk = (int)(off / (uint64_t)fsblk);
            if (ioctl(fd, FIBMAP, &blk) < 0) { perror("FIBMAP"); break; }
            if (nf > 0 && exts[nf-1].phys_sector + exts[nf-1].bytes/512 == (uint64_t)blk) {
                exts[nf-1].bytes += fsblk;
            } else {
                exts[nf].logical = off;
                exts[nf].phys_sector = (uint64_t)blk;
                exts[nf].bytes = fsblk;
                nf++;
            }
            off += (uint64_t)fsblk;
        }
        return nf;
    }
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/data/qwen/layer0_kal.bin";
    int reps = argc > 2 ? atoi(argv[2]) : 3;
    long part_override = argc > 3 ? atol(argv[3]) : -1;
    if (reps < 1) reps = 1;

    struct stat st;
    if (stat(path, &st) != 0) { perror("stat"); return 2; }
    size_t fsz = (size_t)st.st_size;
    long fsblk = st.st_blksize > 0 ? st.st_blksize : 512;

    long part_start = part_override >= 0 ? part_override : find_part_start(st.st_dev);
    if (part_start < 0) {
        fprintf(stderr, "cannot find partition start for dev=%#lx; pass override\n", (unsigned long)st.st_dev);
        return 2;
    }
    printf("G-ION-1 SDHCI->ION  file=%s size=%zu dev=%#lx part_start_sector=%ld fsblk=%ld\n",
           path, fsz, (unsigned long)st.st_dev, part_start, fsblk);

    /* ---- 文件->LBA extent ---- */
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return 2; }
    ext_t exts[256];
    int ne = map_file_extents(fd, fsz, fsblk, exts, 256);
    if (ne <= 0) { fprintf(stderr, "no extents\n"); return 2; }
    uint64_t mapped = 0;
    printf("  extents: %d\n", ne);
    for (int i = 0; i < ne; i++) {
        mapped += exts[i].bytes;
        printf("    [%d] logical=%llu phys_sector=%llu bytes=%llu  -> LBA=%llu\n",
               i, (unsigned long long)exts[i].logical, (unsigned long long)exts[i].phys_sector,
               (unsigned long long)exts[i].bytes,
               (unsigned long long)(part_start + exts[i].phys_sector));
    }
    if (mapped < fsz) { fprintf(stderr, "mapped %llu < fsz %zu\n", (unsigned long long)mapped, fsz); return 2; }

    /* ---- 参考数据: O_DIRECT 读盘 (与 SDHCI 同读盘上字节, 避免 page cache 脏页) ---- */
    uint8_t *ref = NULL;
    if (posix_memalign((void**)&ref, 4096, fsz) != 0) return 2;
    close(fd);
    {
        int ofd = open(path, O_RDONLY | O_DIRECT);
        if (ofd < 0) { perror("open ref O_DIRECT"); free(ref); return 2; }
        size_t off = 0;
        while (off < fsz) {
            size_t n = (fsz - off > (1u<<20)) ? (1u<<20) : (fsz - off);
            ssize_t r = pread(ofd, ref + off, n, (off_t)off);
            if (r != (ssize_t)n) { perror("pread ref"); return 2; }
            off += n;
        }
        close(ofd);
    }

    /* ---- ION alloc (file + desc 槽) ---- */
    CVI_RT_HANDLE rt;
    if (CVI_RT_Init(&rt) != 0) { fprintf(stderr, "CVI_RT_Init fail\n"); return 2; }
    uint64_t ion_sz = (fsz + 4095) & ~4095ull;   /* 对齐 4KB */
    CVI_RT_MEM ion = CVI_RT_MemAlloc(rt, ion_sz + 4096);
    if (!ion) { fprintf(stderr, "ION alloc %llu fail\n", (unsigned long long)(ion_sz + 4096)); return 2; }
    uint8_t *ion_va = CVI_RT_MemGetVAddr(ion);
    uint64_t ion_pa = CVI_RT_MemGetPAddr(ion);
    printf("ION alloc OK va=%p pa=%#llx\n", (void*)ion_va, (unsigned long long)ion_pa);

    /* desc 槽在 ion 末尾 */
    uint64_t desc_off = (ion_sz + 63) & ~63ull;
    mha_dma_desc_t *d = (mha_dma_desc_t *)(ion_va + desc_off);
    uint64_t desc_pa = ion_pa + desc_off;

    if (mbox_open() != 0) { fprintf(stderr, "mbox_open fail\n"); return 2; }
    printf("mbox OK\n");

    /* ---- TAKE_OWNER ---- */
    memset(d, 0, sizeof(*d)); d->result = -1;
    CVI_RT_MemFlush(rt, ion);
    if (mbox_send_cmd(rt, ion, ion_va, ion_pa, CMD_MHA_SD_TAKE_OWNER, (uint32_t)desc_pa) < 0) { fprintf(stderr, "TAKE_OWNER send fail\n"); return 2; }
    int rc = mbox_poll_desc(rt, ion, d, 3000000);
    if (rc != 0) { fprintf(stderr, "TAKE_OWNER rc=%d -> SDHCI init FAIL, 选项 B 不可用\n", rc); return 5; }
    printf("TAKE_OWNER OK\n");

    /* ---- 分块读: 每 extent 按 <=4MB 切, SD_READ_LAYER ---- */
    const uint32_t CHUNK = 4u << 20;   /* ADMA 上限 4MB/次 */
    double t_read_sum = 0;             /* 纯读墙钟 (跨 reps) */
    int chunks_total = 0;
    long nbad = 0, first_bad = -1;

    for (int it = 0; it < reps; it++) {
        for (int i = 0; i < ne; i++) {
            uint64_t off = 0;
            while (off < exts[i].bytes) {
                uint32_t n = (exts[i].bytes - off > CHUNK) ? CHUNK : (uint32_t)(exts[i].bytes - off);
                uint64_t lba = (uint64_t)(part_start + exts[i].phys_sector) + (off / 512);
                uint64_t dst = ion_pa + exts[i].logical + off;

                memset(d, 0, sizeof(*d));
                d->src_paddr = (uint32_t)lba;
                d->dst_paddr = (uint32_t)dst;
                d->size      = n;
                d->result    = -1;
                CVI_RT_MemFlush(rt, ion);

                double t0 = now_s();
                if (mbox_send_cmd(rt, ion, ion_va, ion_pa, CMD_MHA_SD_READ_LAYER, (uint32_t)desc_pa) < 0) {
                    fprintf(stderr, "READ_LAYER send fail\n"); goto fail;
                }
                rc = mbox_poll_desc(rt, ion, d, 15000000);
                double dt = now_s() - t0;
                if (rc != 0) {
                    fprintf(stderr, "READ_LAYER LBA=%llu n=%u rc=%d (SDHCI 读失败)\n",
                            (unsigned long long)lba, n, rc);
                    goto fail;
                }
                t_read_sum += dt;
                chunks_total++;
                off += n;
            }
        }
        /* 一致性检查 (仅第一次) */
        if (it == 0) {
            CVI_RT_MemInvld(rt, ion);
            for (uint64_t i = 0; i < fsz; i++) {
                if (ion_va[i] != ref[i]) {
                    if (nbad == 0) first_bad = (long)i;
                    nbad++;
                    if (nbad > 8) break;
                }
            }
        }
        printf("  rep %d: %llu chunks, read_wall=%.3f s  (SDHCI raw)\n",
               it, (unsigned long long)ne, t_read_sum / (it + 1));
    }

    /* ---- RELEASE ---- */
    memset(d, 0, sizeof(*d)); d->result = -1;
    CVI_RT_MemFlush(rt, ion);
    if (mbox_send_cmd(rt, ion, ion_va, ion_pa, CMD_MHA_SD_RELEASE, (uint32_t)desc_pa) < 0) fprintf(stderr, "RELEASE send fail\n");
    rc = mbox_poll_desc(rt, ion, d, 1000000);
    printf("RELEASE rc=%d\n", rc);

    /* ---- 汇总 ---- */
    double total_bytes = (double)fsz * reps;
    double bw = total_bytes / 1e6 / t_read_sum;
    printf("CONSISTENCY: %s (nbad=%ld first=%ld)\n", nbad == 0 ? "OK" : "BAD", nbad, first_bad);
    printf("BANDWIDTH (SDHCI raw -> ION): total=%.3f MB in %.3f s -> %.2f MB/s  (chunks=%d, reps=%d, chunk<=4MB)\n",
           total_bytes / 1e6, t_read_sum, bw, chunks_total, reps);
    if (bw >= 18.0) printf("VERDICT: >=18MB/s -> 选项 B (副核 SDHCI) 可行, 可立项\n");
    else if (bw >= 15.0) printf("VERDICT: 15-18MB/s -> 边缘, 与选项 A 对比再定\n");
    else printf("VERDICT: <15MB/s -> 选项 B 无收益, 二期砍掉, 只走选项 A (O_DIRECT->ION)\n");

    CVI_RT_MemFree(rt, ion);
    CVI_RT_DeInit(rt);
    free(ref);
    return 0;
fail:
    memset(d, 0, sizeof(*d)); d->result = -1;
    CVI_RT_MemFlush(rt, ion);
    mbox_send_cmd(rt, ion, ion_va, ion_pa, CMD_MHA_SD_RELEASE, (uint32_t)desc_pa);
    mbox_poll_desc(rt, ion, d, 1000000);
    return 5;
}
