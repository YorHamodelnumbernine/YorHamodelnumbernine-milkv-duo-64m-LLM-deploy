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

int main(void) {
    int rows = 8, cols = 4;  // tiny test
    int total = rows * cols;

    fprintf(stderr, "\n=== TRANSPOSE Diagnostic (rows=%d cols=%d total=%d) ===\n", rows, cols, total);

    tpu_ctx ctx;
    if (tpu_init(&ctx, MHA_TOTAL_SIZE) != 0) { fprintf(stderr, "tpu_init failed\n"); return 1; }

    mbox_fd = open("/dev/cvi-rtos-cmdqu", O_RDWR);
    if (mbox_fd < 0) { perror("open"); tpu_cleanup(&ctx); return 1; }

    mha_dma_desc_t *d = (mha_dma_desc_t *)(ctx.neuron_vaddr + MHA_OFF_REF_OUT);
    uint32_t d_pa = TPU_PA(&ctx, MHA_OFF_REF_OUT);
    int8_t *src = (int8_t *)(ctx.neuron_vaddr + MHA_OFF_SCRATCH_I8);
    int8_t *dst = (int8_t *)(ctx.neuron_vaddr + MHA_OFF_REF_OUT + 0x100);
    uint32_t src_pa = TPU_PA(&ctx, MHA_OFF_SCRATCH_I8);
    uint32_t dst_pa = TPU_PA(&ctx, MHA_OFF_REF_OUT + 0x100);

    // Fill src with pattern: 0,1,2,3,4,...
    for (int i = 0; i < total; i++) src[i] = (int8_t)(i & 0x7F);
    memset(dst, 0xCC, total);
    CVI_RT_MemFlush(ctx.rt_handle, ctx.neuron_mem);

    fprintf(stderr, "src pattern: ");
    for(int i=0; i<total; i++) fprintf(stderr, "%d ", src[i]);
    fprintf(stderr, "\n");

    *d = (mha_dma_desc_t){ .src_paddr=src_pa, .dst_paddr=dst_pa, .rows=rows, .cols=cols, .result=-1 };
    CVI_RT_MemFlush(ctx.rt_handle, ctx.neuron_mem);

    fprintf(stderr, "DMA desc: src=0x%x dst=0x%x rows=%u cols=%u result=%d\n",
            d->src_paddr, d->dst_paddr, d->rows, d->cols, d->result);

    cmdqu_t cmdq = { .ip_id=IP_SYSTEM, .cmd_id=CMD_MHA_TRANSPOSE, 
                     .resv={.mstime=0xFFFF}, .param_ptr=d_pa };
    
    int rc = mbox_send_wait(&cmdq);
    CVI_RT_MemInvld(ctx.rt_handle, ctx.neuron_mem);

    fprintf(stderr, "After call: rc=%d result=%d rows=%u cols=%u\n", rc, d->result, d->rows, d->cols);
    fprintf(stderr, "dst data: ");
    for(int i=0; i<total; i++) fprintf(stderr, "%d ", dst[i]);
    fprintf(stderr, "\n");

    // Compute expected
    int8_t ref[256];
    for(int i=0;i<rows;i++) for(int j=0;j<cols;j++) ref[j*rows+i]=src[i*cols+j];
    fprintf(stderr, "expected: ");
    for(int i=0;i<total;i++) fprintf(stderr, "%d ", ref[i]);
    fprintf(stderr, "\n");

    int match = memcmp(dst, ref, total);
    fprintf(stderr, "RESULT: %s (memcmp=%d)\n", match ? "FAIL" : "PASS", match);

    close(mbox_fd);
    tpu_cleanup(&ctx);
    return match ? 1 : 0;
}
