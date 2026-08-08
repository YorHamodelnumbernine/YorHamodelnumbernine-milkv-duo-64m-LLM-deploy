/* Micro-benchmark for LM_Head decode cost analysis (Phase 2).
 * Measures:
 *   1) CVI_RT_MemFlush(1MB neuron) cost
 *   2) tile memcpy + flush pattern (r_is_nm=false path in tpu_matmul_build)
 *   3) mbox EMBED_XPOSE time for CHUNK=1024 / 2048 (current firmware limit)
 *   4) CPU transpose (big-core) time for CHUNK=1024 / 2048 / 4096
 *
 * Build: riscv64-unknown-linux-musl-gcc ... (see Makefile pattern)
 */
#include "common/tpu_bench.h"
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include "common/rtos_cmdqu.h"
#include "common/mha_descriptor.h"

static int mbox_fd = -1;
static int mbox_open(void) {
    mbox_fd = open("/dev/cvi-rtos-cmdqu", O_RDWR);
    return (mbox_fd < 0) ? -1 : 0;
}
static int mbox_send_async(cmdqu_t *cmdq) {
    if (mbox_fd < 0) return -1;
    return ioctl(mbox_fd, RTOS_CMDQU_SEND, cmdq);
}
static int mbox_send_wait(cmdqu_t *cmdq) {
    if (mbox_fd < 0) return -1;
    return ioctl(mbox_fd, RTOS_CMDQU_SEND_WAIT, cmdq);
}
static mha_dma_desc_t *mbox_desc_ptr(uint8_t *nm, uint64_t nm_pa, int slot) {
    return (mha_dma_desc_t *)(nm + MHA_OFF_DMA_DESC + slot * sizeof(mha_dma_desc_t));
}
static uint64_t mbox_desc_pa(uint64_t nm_pa, int slot) {
    return nm_pa + MHA_OFF_DMA_DESC + slot * sizeof(mha_dma_desc_t);
}
static int mbox_poll_desc(tpu_ctx *ctx, mha_dma_desc_t *d, int timeout_us) {
    int waited = 0;
    while (1) {
        CVI_RT_MemInvld(ctx->rt_handle, ctx->neuron_mem);
        if (d->result != -1) break;
        usleep(10);
        waited += 10;
        if (timeout_us > 0 && waited >= timeout_us) return -1;
    }
    int rc = d->result;
    d->result = -1;
    return rc;
}

static inline uint64_t now_ns(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

int main(void) {
    const int D = 576;
    const int V = 49152;
    tpu_ctx ctx;
    if (tpu_init(&ctx, 0x100000) != 0) return 1;
    uint8_t *nm = ctx.neuron_vaddr;
    uint64_t nm_pa = CVI_RT_MemGetPAddr(ctx.neuron_mem);
    uint64_t t0, t1;

    /* ---- 1. MemFlush(1MB) cost ---- */
    int n_rep = 200;
    memset(nm, 0, ctx.neuron_size);
    t0 = now_ns();
    for (int i = 0; i < n_rep; i++) CVI_RT_MemFlush(ctx.rt_handle, ctx.neuron_mem);
    t1 = now_ns();
    fprintf(stderr, "[1] MemFlush(1MB): avg %.1f us  (%.2f MB/ms)\n",
            (double)(t1 - t0) / n_rep / 1000.0, 1000.0 / ((double)(t1 - t0) / n_rep / 1000.0));

    /* ---- 2. tile memcpy + flush (r_is_nm=false pattern) ----
       K=576, tile_n=96 → 55KB per tile */
    const int K = 576, tile_n = 96;
    int8_t *r_i8 = (int8_t *)malloc(K * 2048); /* simulate ION right matrix */
    for (int i = 0; i < K * 2048; i++) r_i8[i] = (int8_t)(i & 0x7f);
    uint32_t off_r = 576;
    n_rep = 100;
    t0 = now_ns();
    for (int rep = 0; rep < n_rep; rep++) {
        for (int ns = 0; ns < 2048; ns += tile_n) {
            int cn = (ns + tile_n <= 2048) ? tile_n : 2048 - ns;
            uint8_t *td = nm + off_r;
            for (int r = 0; r < K; r++) memcpy(td + r * cn, r_i8 + r * 2048 + ns, cn);
            CVI_RT_MemFlush(ctx.rt_handle, ctx.neuron_mem);
        }
    }
    t1 = now_ns();
    fprintf(stderr, "[2] tile copy+flush (CHUNK=2048, %d tiles): %.2f ms/chunk  (%.1f us/tile)\n",
            (2048 + tile_n - 1) / tile_n, (double)(t1 - t0) / n_rep / 1e6,
            (double)(t1 - t0) / n_rep / ((2048 + tile_n - 1) / tile_n) / 1000.0);

    /* ---- 3. mbox EMBED_XPOSE for CHUNK=1024 / 2048 ---- */
    if (mbox_open() == 0) {
        fprintf(stderr, "[3] mbox open OK\n");
        int bufsz = D * 2048;
        CVI_RT_MEM ion_src = CVI_RT_MemAlloc(ctx.rt_handle, bufsz);
        CVI_RT_MEM ion_dst = CVI_RT_MemAlloc(ctx.rt_handle, bufsz);
        if (!ion_src || !ion_dst) {
            fprintf(stderr, "  ION alloc failed\n");
        } else {
            uint8_t *src = (uint8_t *)CVI_RT_MemGetVAddr(ion_src);
            uint8_t *dst = (uint8_t *)CVI_RT_MemGetVAddr(ion_dst);
            uint64_t src_pa = CVI_RT_MemGetPAddr(ion_src);
            uint64_t dst_pa = CVI_RT_MemGetPAddr(ion_dst);
            for (int i = 0; i < bufsz; i++) src[i] = (uint8_t)(i & 0xff);
            for (int cv = 1024; cv <= 2048; cv += 1024) {
                int sz = D * cv;
                memset(dst, 0, sz);
                mha_dma_desc_t *d = mbox_desc_ptr(nm, nm_pa, 0);
                d->src_paddr = (uint32_t)src_pa;
                d->dst_paddr = (uint32_t)dst_pa;
                d->size = sz; d->rows = D; d->cols = cv;
                d->scale = 0; d->zero_point = 0; d->result = -1;
                CVI_RT_MemFlush(ctx.rt_handle, ctx.neuron_mem);
                CVI_RT_MemFlush(ctx.rt_handle, ion_src);
                cmdqu_t cmdq; memset(&cmdq, 0, sizeof(cmdq));
                cmdq.ip_id = IP_SYSTEM; cmdq.cmd_id = CMD_MHA_EMBED_XPOSE;
                cmdq.block = 0; cmdq.param_ptr = (uint32_t)mbox_desc_pa(nm_pa, 0);
                int rc = mbox_send_async(&cmdq);
                if (rc < 0) { fprintf(stderr, "  mbox send failed rc=%d\n", rc); break; }
                t0 = now_ns();
                rc = mbox_poll_desc(&ctx, mbox_desc_ptr(nm, nm_pa, 0), 2000000);
                t1 = now_ns();
                fprintf(stderr, "  mbox EMBED_XPOSE cur_v=%d (%d KB): rc=%d  %.2f ms  (%.2f MB/ms)\n",
                        cv, sz / 1024, rc, (double)(t1 - t0) / 1e6,
                        (double)sz / 1e6 / ((double)(t1 - t0) / 1e6) * 1000.0);
            }
            CVI_RT_MemFree(ctx.rt_handle, ion_src);
            CVI_RT_MemFree(ctx.rt_handle, ion_dst);
        }
    } else {
        fprintf(stderr, "[3] mbox open FAILED — skipping\n");
    }

    /* ---- 3b. mbox EMBED_XPOSE with SEND_WAIT (blocking) vs poll ---- */
    if (mbox_fd >= 0) {
        int bufsz = D * 2048;
        CVI_RT_MEM ion_src = CVI_RT_MemAlloc(ctx.rt_handle, bufsz);
        CVI_RT_MEM ion_dst = CVI_RT_MemAlloc(ctx.rt_handle, bufsz);
        if (ion_src && ion_dst) {
            uint8_t *src = (uint8_t *)CVI_RT_MemGetVAddr(ion_src);
            uint8_t *dst = (uint8_t *)CVI_RT_MemGetVAddr(ion_dst);
            uint64_t src_pa = CVI_RT_MemGetPAddr(ion_src);
            uint64_t dst_pa = CVI_RT_MemGetPAddr(ion_dst);
            for (int i = 0; i < bufsz; i++) src[i] = (uint8_t)(i & 0xff);
            for (int cv = 1024; cv <= 2048; cv += 1024) {
                int sz = D * cv;
                memset(dst, 0, sz);
                mha_dma_desc_t *d = mbox_desc_ptr(nm, nm_pa, 0);
                d->src_paddr = (uint32_t)src_pa;
                d->dst_paddr = (uint32_t)dst_pa;
                d->size = sz; d->rows = D; d->cols = cv;
                d->scale = 0; d->zero_point = 0; d->result = -1;
                CVI_RT_MemFlush(ctx.rt_handle, ctx.neuron_mem);
                CVI_RT_MemFlush(ctx.rt_handle, ion_src);
                cmdqu_t cmdq; memset(&cmdq, 0, sizeof(cmdq));
                cmdq.ip_id = IP_SYSTEM; cmdq.cmd_id = CMD_MHA_EMBED_XPOSE;
                cmdq.block = 0;
                cmdq.resv.mstime = 0xFFFF; /* block infinite */
                cmdq.param_ptr = (uint32_t)mbox_desc_pa(nm_pa, 0);
                t0 = now_ns();
                int rc = mbox_send_wait(&cmdq);
                t1 = now_ns();
                /* verify: dst[j*cv+v] == (int8)src[v*D+j] */
                CVI_RT_MemInvld(ctx.rt_handle, ion_dst);
                int bad = 0;
                for (int j = 0; j < D && bad < 5; j++)
                    for (int v = 0; v < cv && bad < 5; v++)
                        if ((int8_t)dst[j * cv + v] != (int8_t)src[v * D + j]) bad++;
                fprintf(stderr, "  mbox EMBED_XPOSE cur_v=%d SEND_WAIT: rc=%d  %.2f ms  bad=%d\n",
                        cv, rc, (double)(t1 - t0) / 1e6, bad);
            }
            CVI_RT_MemFree(ctx.rt_handle, ion_src);
            CVI_RT_MemFree(ctx.rt_handle, ion_dst);
        }
    }

    /* ---- 4. CPU transpose (big core) for CHUNK=1024 / 2048 / 4096 ---- */
    for (int cv = 1024; cv <= 4096; cv *= 2) {
        int sz = D * cv;
        int8_t *src = (int8_t *)malloc(sz);
        int8_t *dst = (int8_t *)malloc(sz);
        for (int i = 0; i < sz; i++) src[i] = (int8_t)(i & 0x7f);
        n_rep = 5;
        t0 = now_ns();
        for (int rep = 0; rep < n_rep; rep++) {
            for (int j = 0; j < D; j++)
                for (int v = 0; v < cv; v++)
                    dst[j * cv + v] = src[v * D + j];
        }
        t1 = now_ns();
        fprintf(stderr, "[4] CPU transpose naive cur_v=%d (%d KB): %.2f ms  (%.2f MB/ms)\n",
                cv, sz / 1024, (double)(t1 - t0) / n_rep / 1e6, (double)sz / 1e6 / ((double)(t1 - t0) / n_rep / 1e6) * 1000.0);
        free(src); free(dst);
    }

    /* ---- 4b. CPU transpose BLOCKED (BS=32) for CHUNK=2048 / 4096 ---- */
    for (int cv = 2048; cv <= 4096; cv *= 2) {
        int sz = D * cv;
        int8_t *src = (int8_t *)malloc(sz);
        int8_t *dst = (int8_t *)malloc(sz);
        for (int i = 0; i < sz; i++) src[i] = (int8_t)(i & 0x7f);
        n_rep = 5;
        t0 = now_ns();
        for (int rep = 0; rep < n_rep; rep++) {
            for (int j0 = 0; j0 < D; j0 += 32)
                for (int v0 = 0; v0 < cv; v0 += 32)
                    for (int j = j0; j < j0 + 32 && j < D; j++)
                        for (int v = v0; v < v0 + 32 && v < cv; v++)
                            dst[j * cv + v] = src[v * D + j];
        }
        t1 = now_ns();
        fprintf(stderr, "[4b] CPU transpose blocked cur_v=%d (%d KB): %.2f ms  (%.2f MB/ms)\n",
                cv, sz / 1024, (double)(t1 - t0) / n_rep / 1e6, (double)sz / 1e6 / ((double)(t1 - t0) / n_rep / 1e6) * 1000.0);
        free(src); free(dst);
    }

    free(r_i8);
    tpu_cleanup(&ctx);
    return 0;
}
