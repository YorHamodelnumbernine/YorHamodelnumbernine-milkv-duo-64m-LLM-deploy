/* Shared header for Multihead Attention dual-core implementation.
   Used by both Linux big-core (mha_attention.c) and FreeRTOS small-core (comm_main.c). */
#ifndef MHA_DESCRIPTOR_H
#define MHA_DESCRIPTOR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- MHA configuration ---- */
typedef struct {
    int d_model;
    int n_heads;
    int head_dim;
    int seq_len;
    float softmax_scale;   /* 1.0f / sqrtf(head_dim) */
} mha_config_t;

/* ---- Shared DRAM memory layout (within TPU neuron memory) ---- */
#define MHA_OFF_WEIGHTS    0x000000   /* Wq, Wk, Wv, Wo pre-quantized INT8 */
#define MHA_OFF_INPUT      0x040000   /* input x [seq_len, d_model] FP32 */
#define MHA_OFF_Q_F32      0x050000   /* Q after dequant + RoPE [seq_len, d_model] FP32 */
#define MHA_OFF_K_F32      0x060000   /* K after dequant + RoPE [seq_len, d_model] FP32 */
#define MHA_OFF_V_F32      0x070000   /* V [seq_len, d_model] FP32 */
#define MHA_OFF_SCORES_F32 0x080000   /* Scores [n_heads, seq_len, seq_len] FP32 */
#define MHA_OFF_ATTN_F32   0x090000   /* Attn output [seq_len, d_model] FP32 */
#define MHA_OFF_OUT_F32    0x0A0000   /* Final output [seq_len, d_model] FP32 */
#define MHA_OFF_SCRATCH_I8 0x0C0000   /* INT8 scratch buffers */
#define MHA_OFF_REF_OUT    0x0E0000   /* FP32 reference output */
#define MHA_OFF_DMA_DESC   0x1F800   /* DMA descriptors (safe near end of 1MB region) */
#define MHA_TOTAL_SIZE     0x100000   /* 1MB total */

/* INT8 scratch offsets */
#define MHA_OFF_Q_I8       (MHA_OFF_SCRATCH_I8 + 0x00000)
#define MHA_OFF_K_I8       (MHA_OFF_SCRATCH_I8 + 0x04000)
#define MHA_OFF_V_I8       (MHA_OFF_SCRATCH_I8 + 0x08000)
#define MHA_OFF_S_I8       (MHA_OFF_SCRATCH_I8 + 0x0C000)
#define MHA_OFF_A_I8       (MHA_OFF_SCRATCH_I8 + 0x10000)
#define MHA_OFF_OUT_I8     (MHA_OFF_SCRATCH_I8 + 0x14000)

/* Weight offsets within MHA_OFF_WEIGHTS */
#define MHA_WQ_OFF 0x00000   /* Wq [d_model, d_model] INT8 */
#define MHA_WK_OFF 0x04000   /* Wk [d_model, d_model] INT8 */
#define MHA_WV_OFF 0x08000   /* Wv [d_model, d_model] INT8 */
#define MHA_WO_OFF 0x0C000   /* Wo [d_model, d_model] INT8 */

/* ---- FreeRTOS data-mover command IDs ---- */
enum MHA_CMD_ID {
    CMD_MHA_MEMCPY      = 0x20,
    CMD_MHA_MEMSET      = 0x21,
    CMD_MHA_CACHE_FLUSH  = 0x22,
    CMD_MHA_CACHE_INVLD  = 0x23,
    CMD_MHA_QUANTIZE    = 0x24,
    CMD_MHA_TRANSPOSE   = 0x25,
    CMD_MHA_DEQUANTIZE  = 0x26,
    CMD_MHA_DDR_TO_ION   = 0x27,
    CMD_MHA_EMBED_XPOSE  = 0x28,
    CMD_MHA_DONE        = 0x7F,
};

/* ---- DMA descriptor (shared in DRAM, passed via mailbox param_ptr) ---- */
typedef struct __attribute__((packed)) {
    uint32_t src_paddr;
    uint32_t dst_paddr;
    uint32_t size;
    uint32_t rows;
    uint32_t cols;
    float    scale;        /* quantization scale */
    int32_t  zero_point;   /* quantization zero-point */
    int32_t  result;       /* return code: 0=OK, <0=error */
} mha_dma_desc_t;

/* ---- Utility functions ---- */
static inline int mha_round_up(int x, int align) {
    return (x + align - 1) & ~(align - 1);
}

static inline int mha_lmem_matrix_bytes(int rows, int cols) {
    int c = (rows + 1) / 2;
    int w = (cols + 31) / 32;
    return c * w * 32;
}

#ifdef __cplusplus
}
#endif
#endif /* MHA_DESCRIPTOR_H */
