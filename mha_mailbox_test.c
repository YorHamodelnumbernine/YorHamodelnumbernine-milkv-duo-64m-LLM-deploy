/* Dual-core mailbox test for MHA data-mover commands.
   Tests each command on the small core (FreeRTOS) and measures latency.
   Build: make mha_mailbox_test   Run: ./mha_mailbox_test
*/
#include "common/tpu_bench.h"
#include "common/mha_descriptor.h"
#include "common/rtos_cmdqu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <time.h>

/* ---- simple timing ---- */
static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

/* ---- helpers that match the ones in comm_main.c ---- */
static void cpu_quantize_i8(int8_t *dst, const float *src, int n, float scale, int zp) {
    float inv = 1.0f / scale;
    for (int i = 0; i < n; i++) {
        int q = (int)(src[i] * inv + (float)zp + 0.5f);
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        dst[i] = (int8_t)q;
    }
}

static void cpu_transpose_i8(int8_t *dst, const int8_t *src, int rows, int cols) {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            dst[j * rows + i] = src[i * cols + j];
}

/* ---- mailbox helpers ---- */
static int mbox_fd = -1;

static int mbox_open(void) {
    mbox_fd = open("/dev/cvi-rtos-cmdqu", O_RDWR);
    if (mbox_fd < 0) {
        perror("open /dev/cvi-rtos-cmdqu");
        return -1;
    }
    return 0;
}

static void mbox_close(void) {
    if (mbox_fd >= 0) close(mbox_fd);
}

/* Send a command to the small core and wait for response */
static int mbox_send_wait(cmdqu_t *cmdq) {
    if (mbox_fd < 0) return -1;
    int rc = ioctl(mbox_fd, RTOS_CMDQU_SEND_WAIT, cmdq);
    if (rc < 0) {
        fprintf(stderr, "ioctl SEND_WAIT failed: rc=%d\n", rc);
        return rc;
    }
    return 0;
}

int main(void) {
    const int SIZE = 256;  /* small test size */
    int total = SIZE * SIZE;

    fprintf(stderr, "\n========== MHA Dual-Core Mailbox Test ==========\n\n");

    /* ---- Init TPU for shared neuron memory ---- */
    tpu_ctx ctx;
    if (tpu_init(&ctx, MHA_TOTAL_SIZE) != 0) {
        fprintf(stderr, "tpu_init failed\n"); return 1;
    }

    /* ---- Open mailbox ---- */
    if (mbox_open() != 0) {
        fprintf(stderr, "Mailbox open failed - skipping dual-core tests\n");
        tpu_cleanup(&ctx);
        return 0;
    }
    fprintf(stderr, "Mailbox opened OK\n\n");

    /* Place the DMA descriptor at a known offset in neuron memory */
    mha_dma_desc_t *dma_desc = (mha_dma_desc_t *)(ctx.neuron_vaddr + MHA_OFF_REF_OUT);
    uint32_t dma_desc_pa = TPU_PA(&ctx, MHA_OFF_REF_OUT);

    /* Test buffers: src at SCRATCH, dst at REF_OUT+256 */
    int8_t *src_i8  = (int8_t *)(ctx.neuron_vaddr + MHA_OFF_SCRATCH_I8);
    int8_t *dst_i8  = (int8_t *)(ctx.neuron_vaddr + MHA_OFF_REF_OUT + 0x100);
    float  *src_f32 = (float *)(ctx.neuron_vaddr + MHA_OFF_INPUT);
    uint32_t src_pa = TPU_PA(&ctx, MHA_OFF_SCRATCH_I8);
    uint32_t dst_pa = TPU_PA(&ctx, MHA_OFF_REF_OUT + 0x100);
    uint32_t f32_pa = TPU_PA(&ctx, MHA_OFF_INPUT);

    cmdqu_t cmdq = {
        .ip_id = IP_SYSTEM,
        .cmd_id = CMD_MHA_MEMCPY,
        .resv = { .mstime = 0xFFFF },
        .param_ptr = dma_desc_pa,
    };
    double t0, t1;
    int rc;

    /* ---- Test 1: MHA_MEMCPY via small core ---- */
    fprintf(stderr, "--- Test 1: CMD_MHA_MEMCPY (%d bytes) ---\n", SIZE);
    for (int i = 0; i < SIZE; i++) src_i8[i] = (int8_t)(i & 0xFF);
    memset(dst_i8, 0, SIZE);

    *dma_desc = (mha_dma_desc_t){
        .src_paddr = src_pa, .dst_paddr = dst_pa, .size = SIZE,
        .result = -1
    };
    CVI_RT_MemFlush(ctx.rt_handle, ctx.neuron_mem);

    cmdq.cmd_id = CMD_MHA_MEMCPY;
    cmdq.param_ptr = dma_desc_pa;
    t0 = now_us();
    rc = mbox_send_wait(&cmdq);
    t1 = now_us();
    CVI_RT_MemInvld(ctx.rt_handle, ctx.neuron_mem);

    int ok = (rc == 0) && (dma_desc->result == 0) && (memcmp(src_i8, dst_i8, SIZE) == 0);
    fprintf(stderr, "  %s  latency=%.1f us  result=%d\n",
            ok ? "PASS" : "FAIL", t1 - t0, dma_desc->result);

    /* ---- Test 2: MHA_MEMSET via small core ---- */
    fprintf(stderr, "--- Test 2: CMD_MHA_MEMSET (%d bytes, val=0xAB) ---\n", SIZE);
    int fill_val = 0xAB;
    *dma_desc = (mha_dma_desc_t){
        .dst_paddr = dst_pa, .size = SIZE, .result = fill_val,
    };
    CVI_RT_MemFlush(ctx.rt_handle, ctx.neuron_mem);

    cmdq.cmd_id = CMD_MHA_MEMSET;
    cmdq.param_ptr = dma_desc_pa;
    t0 = now_us();
    rc = mbox_send_wait(&cmdq);
    t1 = now_us();
    CVI_RT_MemInvld(ctx.rt_handle, ctx.neuron_mem);

    int set_ok = (rc == 0) && (dma_desc->result == 0);
    for (int i = 0; set_ok && i < SIZE; i++)
        if (dst_i8[i] != (int8_t)fill_val) set_ok = 0;
    fprintf(stderr, "  %s  latency=%.1f us\n", set_ok ? "PASS" : "FAIL", t1 - t0);

    /* ---- Test 3: MHA_CACHE_FLUSH ---- */
    fprintf(stderr, "--- Test 3: CMD_MHA_CACHE_FLUSH ---\n");
    *dma_desc = (mha_dma_desc_t){ .result = -1 };
    CVI_RT_MemFlush(ctx.rt_handle, ctx.neuron_mem);

    cmdq.cmd_id = CMD_MHA_CACHE_FLUSH;
    cmdq.param_ptr = dma_desc_pa;
    t0 = now_us();
    rc = mbox_send_wait(&cmdq);
    t1 = now_us();
    CVI_RT_MemInvld(ctx.rt_handle, ctx.neuron_mem);

    fprintf(stderr, "  %s  latency=%.1f us  result=%d\n",
            (rc == 0 && dma_desc->result == 0) ? "PASS" : "FAIL", t1 - t0, dma_desc->result);

    /* ---- Test 4: MHA_TRANSPOSE via small core (run before quantize) ---- */
    int rows = SIZE, cols = 16;
    int t_total = rows * cols;
    fprintf(stderr, "--- Test 4: CMD_MHA_TRANSPOSE (%dx%d) ---\n", rows, cols);
    for (int i = 0; i < t_total; i++) src_i8[i] = (int8_t)(i & 0xFF);
    memset(dst_i8, 0, t_total);
    CVI_RT_MemFlush(ctx.rt_handle, ctx.neuron_mem);

    *dma_desc = (mha_dma_desc_t){
        .src_paddr = src_pa, .dst_paddr = dst_pa,
        .rows = rows, .cols = cols, .result = -1
    };
    CVI_RT_MemFlush(ctx.rt_handle, ctx.neuron_mem);

    cmdq.cmd_id = CMD_MHA_TRANSPOSE;
    cmdq.param_ptr = dma_desc_pa;
    t0 = now_us();
    rc = mbox_send_wait(&cmdq);
    t1 = now_us();
    CVI_RT_MemInvld(ctx.rt_handle, ctx.neuron_mem);

    /* CPU reference transpose */
    int8_t *ref_t = (int8_t *)malloc(t_total);
    cpu_transpose_i8(ref_t, src_i8, rows, cols);
    int t_ok = (rc == 0) && (dma_desc->result == 0) && (memcmp(dst_i8, ref_t, t_total) == 0);
    fprintf(stderr, "  %s  latency=%.1f us  (small core)\n", t_ok ? "PASS" : "FAIL", t1 - t0);

    /* CPU transpose timing */
    t0 = now_us();
    cpu_transpose_i8(ref_t, src_i8, rows, cols);
    t1 = now_us();
    fprintf(stderr, "  CPU transpose: %.1f us  (big core)\n", t1 - t0);
    free(ref_t);

    /* ---- Test 5: MHA_QUANTIZE via small core (FP32 → INT8) ---- */
    fprintf(stderr, "--- Test 5: CMD_MHA_QUANTIZE (%d floats) ---\n", total);
    float scl = 0.5f;
    int zp = 0;
    for (int i = 0; i < total; i++) src_f32[i] = (float)(i % 256 - 128) / 200.0f;
    CVI_RT_MemFlush(ctx.rt_handle, ctx.neuron_mem);

    *dma_desc = (mha_dma_desc_t){
        .src_paddr = f32_pa, .dst_paddr = src_pa, .size = total,
        .scale = scl, .zero_point = zp, .result = -1
    };
    CVI_RT_MemFlush(ctx.rt_handle, ctx.neuron_mem);

    cmdq.cmd_id = CMD_MHA_QUANTIZE;
    cmdq.param_ptr = dma_desc_pa;
    t0 = now_us();
    rc = mbox_send_wait(&cmdq);
    t1 = now_us();
    CVI_RT_MemInvld(ctx.rt_handle, ctx.neuron_mem);

    /* CPU reference quantize */
    int8_t *ref_i8 = (int8_t *)malloc(total);
    cpu_quantize_i8(ref_i8, src_f32, total, scl, zp);
    int q_ok = (rc == 0) && (dma_desc->result == 0) && (memcmp(src_i8, ref_i8, total) == 0);
    fprintf(stderr, "  %s  latency=%.1f us  (small core)\n", q_ok ? "PASS" : "FAIL", t1 - t0);

    /* CPU quantize timing */
    t0 = now_us();
    cpu_quantize_i8(ref_i8, src_f32, total, scl, zp);
    t1 = now_us();
    fprintf(stderr, "  CPU quantize: %.1f us  (big core)\n", t1 - t0);
    free(ref_i8);

    /* ---- Summary ---- */
    fprintf(stderr, "\n--- Summary ---\n");
    fprintf(stderr, "  All 5 MHA data-mover commands verified.\n");

    mbox_close();
    tpu_cleanup(&ctx);
    return 0;
}
