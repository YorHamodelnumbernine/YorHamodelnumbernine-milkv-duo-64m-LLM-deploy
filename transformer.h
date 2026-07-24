/* Shared definitions for Transformer inference on CV1800B.
   Mini Llama-style decoder: RMSNorm, RoPE, SwiGLU FFN, KV-cache.
   d_model=128, n_heads=4, head_dim=32, n_layers=2, ffn=256, vocab=128 */
#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <stdint.h>

/* ---- Model Config ---- */
#define TR_D_MODEL    128
#define TR_N_HEADS    4
#define TR_HEAD_DIM   32
#define TR_N_LAYERS   2
#define TR_FFN_HIDDEN 256
#define TR_VOCAB_SIZE 128
#define TR_MAX_SEQ    64

/* Derived sizes */
#define TR_D_SQ       (TR_D_MODEL * TR_D_MODEL)       /* 16384 */
#define TR_FFN_SZ     (TR_D_MODEL * TR_FFN_HIDDEN)    /* 32768 */
#define TR_VOCAB_SZ   (TR_D_MODEL * TR_VOCAB_SIZE)    /* 16384 */
#define TR_RMS_SZ     (TR_D_MODEL)                     /* 128 */

/* Per-layer weight sizes (INT8 bytes) */
#define TR_LAYER_W_SZ (4 * TR_D_SQ + 3 * TR_FFN_SZ + 2 * TR_RMS_SZ)  /* ~200 KB */

/* ---- Weight file names ---- */
#define TR_WGT_EMBED    "embed.f32"
#define TR_WGT_LM_HEAD  "lm_head.f32"

/* Per-layer weight file name patterns (use snprintf with layer index) */
#define TR_FMT_RMS_ATTN  "layer%d_rms_attn.f32"
#define TR_FMT_WQ        "layer%d_Wq.i8"
#define TR_FMT_WK        "layer%d_Wk.i8"
#define TR_FMT_WV        "layer%d_Wv.i8"
#define TR_FMT_WO        "layer%d_Wo.i8"
#define TR_FMT_RMS_FFN   "layer%d_rms_ffn.f32"
#define TR_FMT_FFN_UP    "layer%d_ffn_up.i8"
#define TR_FMT_FFN_GATE  "layer%d_ffn_gate.i8"
#define TR_FMT_FFN_DOWN  "layer%d_ffn_down.i8"
#define TR_FMT_FINAL_RMS "final_rms.f32"

/* ---- Neuron Memory Layout (1MB total) ---- */
#define TR_NEURON_SIZE  0x100000

/* FP32 buffers (each sized for max_seq × d_model or equivalent) */
#define TR_OFF_X_F32      0x000000   /* current input/hidden [seq, D] */
#define TR_OFF_Q_F32      0x004000   /* Q after dequant [seq, D] */
#define TR_OFF_K_F32      0x008000   /* K after dequant [seq, D] */
#define TR_OFF_V_F32      0x00C000   /* V after dequant [seq, D] */
#define TR_OFF_SCORES_F32 0x010000   /* attention scores [H, S, S] */
#define TR_OFF_ATTN_F32   0x014000   /* attention output [seq, D] */
#define TR_OFF_FFN_F32    0x018000   /* FFN intermediate [seq, ffn] */
#define TR_OFF_OUT_F32    0x01C000   /* layer output [seq, D] */
#define TR_OFF_HIDDEN2_F32 0x01C000  /* alias: reuse OUT space */

/* KV Cache: grows with seq_len. 2 layers × 2 (K,V) × max_seq × D × 4B */
/* K0: [0x020000, +32KB), V0: [+32KB, +32KB), K1: [+32KB, +32KB), V1: [+32KB, +32KB) */
#define TR_OFF_KV_CACHE   0x020000
#define TR_KV_LAYER_STRIDE  (2 * TR_MAX_SEQ * TR_D_MODEL * 4)   /* 64 KB */
#define TR_KV_SLOT_STRIDE   (TR_MAX_SEQ * TR_D_MODEL * 4)        /* 32 KB */
#define TR_OFF_K_CACHE(l)   (TR_OFF_KV_CACHE + (l) * TR_KV_LAYER_STRIDE)
#define TR_OFF_V_CACHE(l)   (TR_OFF_KV_CACHE + (l) * TR_KV_LAYER_STRIDE + TR_KV_SLOT_STRIDE)

/* INT8 scratch (for TPU matmul intermediates) */
#define TR_OFF_SCRATCH_I8 0x040000
#define TR_OFF_Q_I8       (TR_OFF_SCRATCH_I8 + 0x00000)
#define TR_OFF_K_I8       (TR_OFF_SCRATCH_I8 + 0x04000)
#define TR_OFF_V_I8       (TR_OFF_SCRATCH_I8 + 0x08000)
#define TR_OFF_S_I8       (TR_OFF_SCRATCH_I8 + 0x0C000)
#define TR_OFF_A_I8       (TR_OFF_SCRATCH_I8 + 0x10000)
#define TR_OFF_O_I8       (TR_OFF_SCRATCH_I8 + 0x14000)
#define TR_OFF_FFN_UP_I8  (TR_OFF_SCRATCH_I8 + 0x18000)
#define TR_OFF_FFN_GT_I8  (TR_OFF_SCRATCH_I8 + 0x20000)
#define TR_OFF_FFN_DN_I8  (TR_OFF_SCRATCH_I8 + 0x24000)
#define TR_OFF_LOGITS_I8  (TR_OFF_SCRATCH_I8 + 0x28000)

/* TPU matmul scratch space */
#define TR_MATMUL_SCR      0x00000

/* ---- KV Cache (runtime, in neuron memory) ---- */
typedef struct {
    float *K;   /* points to K cache slot in neuron memory */
    float *V;   /* points to V cache slot in neuron memory */
    int    len; /* current sequence length cached */
} tr_kv_cache_t;

/* ---- Utility ---- */
static inline int tr_round_up(int x, int a) { return (x + a - 1) & ~(a - 1); }
static inline int tr_lmem_matrix_bytes(int rows, int cols) {
    int c = (rows + 1) / 2, w = (cols + 31) / 32;
    return c * w * 32;
}

#endif /* TRANSFORMER_H */
