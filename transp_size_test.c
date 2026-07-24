#include "common/tpu_bench.h"
#include "common/mha_descriptor.h"
#include "common/rtos_cmdqu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>

static int mbox_fd = -1;
static int mbox_send_wait(cmdqu_t *cmdq) {
    return ioctl(mbox_fd, RTOS_CMDQU_SEND_WAIT, cmdq);
}

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

static int test_transpose(tpu_ctx *ctx, int rows, int cols) {
    int total = rows * cols;
    
    mha_dma_desc_t *d = (mha_dma_desc_t *)(ctx->neuron_vaddr + MHA_OFF_REF_OUT);
    uint32_t d_pa = TPU_PA(ctx, MHA_OFF_REF_OUT);
    int8_t *src = (int8_t *)(ctx->neuron_vaddr + MHA_OFF_SCRATCH_I8);
    int8_t *dst = (int8_t *)(ctx->neuron_vaddr + MHA_OFF_REF_OUT + 0x100);
    uint32_t src_pa = TPU_PA(ctx, MHA_OFF_SCRATCH_I8);
    uint32_t dst_pa = TPU_PA(ctx, MHA_OFF_REF_OUT + 0x100);

    for (int i = 0; i < total; i++) src[i] = (int8_t)(i & 0x7F);
    memset(dst, 0xCC, total);
    CVI_RT_MemFlush(ctx->rt_handle, ctx->neuron_mem);

    *d = (mha_dma_desc_t){ .src_paddr=src_pa, .dst_paddr=dst_pa, .rows=rows, .cols=cols, .result=-1 };
    CVI_RT_MemFlush(ctx->rt_handle, ctx->neuron_mem);

    cmdqu_t cmdq = { .ip_id=IP_SYSTEM, .cmd_id=CMD_MHA_TRANSPOSE, 
                     .resv={.mstime=0xFFFF}, .param_ptr=d_pa };

    double t0 = now_us();
    int rc = mbox_send_wait(&cmdq);
    double t1 = now_us();
    CVI_RT_MemInvld(ctx->rt_handle, ctx->neuron_mem);

    // Compute reference
    int8_t *ref = (int8_t *)malloc(total);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            ref[j * rows + i] = src[i * cols + j];

    int match = memcmp(dst, ref, total);
    free(ref);

    fprintf(stderr, "  %dx%d: rc=%d result=%d latency=%.1f us  %s\n",
            rows, cols, rc, d->result, t1 - t0, match ? "FAIL" : "PASS");
    return match ? 1 : 0;
}

int main(void) {
    fprintf(stderr, "\n=== TRANSPOSE Size Test ===\n");

    tpu_ctx ctx;
    if (tpu_init(&ctx, MHA_TOTAL_SIZE) != 0) { fprintf(stderr, "tpu_init failed\n"); return 1; }

    mbox_fd = open("/dev/cvi-rtos-cmdqu", O_RDWR);
    if (mbox_fd < 0) { perror("open"); tpu_cleanup(&ctx); return 1; }

    // Test various sizes
    int sizes[][2] = {
        {8, 4},     // 32 bytes  - we know this works
        {16, 8},    // 128 bytes
        {32, 8},    // 256 bytes
        {32, 16},   // 512 bytes
        {64, 16},   // 1024 bytes
        {128, 16},  // 2048 bytes
        {256, 16},  // 4096 bytes - original test size
    };

    int pass = 0, fail = 0;
    for (int i = 0; i < 7; i++) {
        int r = test_transpose(&ctx, sizes[i][0], sizes[i][1]);
        if (r == 0) pass++; else fail++;
    }

    fprintf(stderr, "\n  Summary: %d PASS, %d FAIL\n", pass, fail);

    close(mbox_fd);
    tpu_cleanup(&ctx);
    return fail ? 1 : 0;
}
