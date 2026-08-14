#!/usr/bin/env python3
"""convert_qwen.py — Qwen2.5-0.5B-Instruct bf16 safetensors → on-board weight files.

Quantization (Phase 4 CEO-approved):
  - linear layers : RTN symmetric INT4, G=32, per-group fp16 scale
                    (dequantized per-channel to INT8 for on-board compute,
                     per-output-channel fp32 scale)
  - embedding     : INT8 per-row (fp16-derived fp32 scale)

On-board file formats:
  config.bin          8×int32: (D, n_heads, n_kv_heads, head_dim,
                                n_layers, FFN, vocab, max_seq)
  embed_i8.bin        [vocab, D] INT8 (weight-tied with lm_head)
  embed_scales.f32    [vocab] fp32 per-row scales
  layerN_i4.bin       per layer:
                        [rms_attn f32 : D*4]
                        [Wq nib D*D/2][Wq grpsc (D*D/G)*2 fp16]
                        [Wk nib][Wk grpsc]
                        [Wv nib][Wv grpsc]
                        [Wo nib][Wo grpsc]
                        [rms_ffn f32 : D*4]
                        [up   nib][up   grpsc]
                        [gate nib][gate grpsc]
                        [down nib][down grpsc]
  layer_scales.bin    per-channel fp32 scales, all layers:
                        [Wq(D) Wk(dkv) Wv(dkv) Wo(D) up(F) gate(F) down(D)]
  layerN_bias.f32     per layer: [q_bias(D) k_bias(dkv) v_bias(dkv)] fp32
                        (Qwen2.5 q/k/v have bias=True; o/MLP bias=False)
  final_rms.f32       [D] fp32
  scales.bin          legacy per-tensor (1 + L*7 floats), ignored when
                      layer_scales.bin present.

Uses only numpy (+ safetensors via sf_io).  No torch required.
"""
import os, sys, struct, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PRE = os.path.join(HERE, "..", "qwen_int4")
sys.path.insert(0, PRE)
from sf_io import SF                      # noqa: E402
from q4 import group_quantize, SYM_QMAX   # noqa: E402

D = 896
H = 14
KVH = 2
HEAD_DIM = 64     # Qwen2.5-0.5B: hidden 896 / 14 heads = 64
L = 24
F = 4864
V = 151936
G = 32
DKV = KVH * HEAD_DIM
ROPE_THETA = 1000000.0
NORM_EPS = 1e-6
MAX_SEQ = 64  # config.bin default; engine can override

# matrix geometry (per layer, engine order): name, [K, N]
MATS = [
    ("q_proj", D, D),
    ("k_proj", D, KVH * HEAD_DIM),
    ("v_proj", D, KVH * HEAD_DIM),
    ("o_proj", D, D),
    ("gate_proj", D, F),
    ("up_proj", D, F),
    ("down_proj", F, D),
]
# engine layer_scales.bin order: Wq Wk Wv Wo up gate down
SCALE_ORDER = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"]


def fp16_bytes(arr_fp32):
    """Convert fp32 numpy array to fp16 raw little-endian bytes (exact fp16)."""
    return arr_fp32.astype(np.float16).tobytes()


def pack_matrix(W_hf, G):
    """W_hf: HF [out, in] bf16->f32.  Returns (nib_bytes, grpsc_bytes, dq) where
    dq is the per-channel-int8-dequantized float [in,out] for scale computation.
    Flow:
      1. Wt = W_hf.T                 -> [in, out] engine layout
      2. INT4 group quantize G=32 fp16 -> nibbles + group scales
      3. dequant to fp32             -> W_dq
      4. per-output-channel INT8 scale = max|W_dq[:,j]|/127
    Returns nib bytes, group-scale bytes, and per-channel scale array [N].
    """
    Wt = np.ascontiguousarray(W_hf.T.astype(np.float32))          # [in, out]
    nib, s_fp16, _, q, W_dq = group_quantize(Wt, G, asym=False, scale_dtype="fp16")
    # per-output-channel scale of the dequantized weight
    K, N = Wt.shape
    mx = np.max(np.abs(W_dq), axis=0)
    sc = np.maximum(mx / 127.0, 1e-12)
    return nib, s_fp16.tobytes(), sc


def quant_embed(E, G=None):
    """E: [vocab, D] bf16->f32.  Per-row INT8.  Returns (i8 [vocab,D], scales [vocab])."""
    mx = np.max(np.abs(E), axis=1, keepdims=True)
    s = np.maximum(mx / 127.0, 1e-12)
    q = np.clip(np.round(E / s), -128, 127).astype(np.int8)
    return q, s.ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(PRE, "model"))
    ap.add_argument("--out", default=os.path.join(HERE, "weights"))
    ap.add_argument("--max-seq", type=int, default=MAX_SEQ)
    ap.add_argument("--no-embed", action="store_true",
                    help="skip embed/lm-head files (fast iteration)")
    args = ap.parse_args()
    out = args.out
    os.makedirs(out, exist_ok=True)

    sf = SF(os.path.join(args.model, "model.safetensors"))
    names = sf.keys()
    print(f"[conv] safetensors keys={len(names)}")

    # ---- config.bin ----
    cfg = struct.pack("iiiiiiii", D, H, KVH, HEAD_DIM, L, F, V, args.max_seq)
    open(os.path.join(out, "config.bin"), "wb").write(cfg)
    print(f"[conv] config.bin: D={D} H={H} KVH={KVH} hd={HEAD_DIM} L={L} F={F} V={V} max_seq={args.max_seq}")

    # ---- embedding (weight-tied) ----
    if not args.no_embed:
        E = sf.get("model.embed_tokens.weight")          # [V, D]
        print(f"[conv] embed {E.shape} dtype->f32")
        Ei8, esc = quant_embed(E)
        Ei8.tofile(os.path.join(out, "embed_i8.bin"))
        esc.astype(np.float32).tofile(os.path.join(out, "embed_scales.f32"))
        print(f"[conv] embed_i8.bin {Ei8.nbytes/1e6:.1f} MB, embed_scales.f32 {esc.nbytes} B")

    # ---- layers ----
    all_lsc = []
    tot_i4 = 0
    for l in range(L):
        p = f"model.layers.{l}"
        rms_attn = sf.get(f"{p}.input_layernorm.weight").astype(np.float32)
        rms_ffn = sf.get(f"{p}.post_attention_layernorm.weight").astype(np.float32)

        # first pass: quantize each matrix, collect per-channel scales + packed bytes
        packed = {}          # name -> (nib_bytes, grpsc_bytes)
        lsc = {}             # name -> per-channel scale array
        for name, K, N in MATS:
            if name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                w = sf.get(f"{p}.self_attn.{name}.weight")
            else:
                w = sf.get(f"{p}.mlp.{name}.weight")
            if w.shape[0] != N or w.shape[1] != K:
                raise SystemExit(f"shape mismatch {name}: got {w.shape}, want [{N},{K}]")
            nib_b, gs_b, sc = pack_matrix(w, G)
            packed[name] = (nib_b, gs_b)
            lsc[name] = sc
        # layer_scales order
        lsc_flat = np.concatenate([lsc[n] for n in SCALE_ORDER])
        all_lsc.append(lsc_flat)

        # Qwen2.5 q/k/v have bias=True -> sidecar bias file
        bq = sf.get(f"{p}.self_attn.q_proj.bias").astype(np.float32)
        bk = sf.get(f"{p}.self_attn.k_proj.bias").astype(np.float32)
        bv = sf.get(f"{p}.self_attn.v_proj.bias").astype(np.float32)
        np.concatenate([bq, bk, bv]).tofile(os.path.join(out, f"layer{l}_bias.f32"))
        print(f"[conv] layer{l}_bias.f32 {bq.nbytes+bk.nbytes+bv.nbytes} B "
              f"(q{D} k{DKV} v{DKV})")

        # second pass: write layerN_i4.bin
        fname = os.path.join(out, f"layer{l}_i4.bin")
        with open(fname, "wb") as f:
            f.write(rms_attn.astype(np.float32).tobytes())
            for name, K, N in MATS:
                nib_b, gs_b = packed[name]
                f.write(nib_b)
                f.write(gs_b)
            f.write(rms_ffn.astype(np.float32).tobytes())
        sz = os.path.getsize(fname)
        tot_i4 += sz
        print(f"[conv] layer{l}_i4.bin {sz/1e6:.2f} MB  "
              f"(q={len(packed['q_proj'][0])+len(packed['q_proj'][1])} "
              f"up={len(packed['up_proj'][0])+len(packed['up_proj'][1])})")

    lsc_all = np.concatenate(all_lsc)
    lsc_all.tofile(os.path.join(out, "layer_scales.bin"))
    print(f"[conv] layer_scales.bin {lsc_all.nbytes/1e6:.2f} MB ({len(lsc_all)} scales)")

    # ---- final RMS ----
    fr = sf.get("model.norm.weight").astype(np.float32)
    fr.tofile(os.path.join(out, "final_rms.f32"))
    print(f"[conv] final_rms.f32 {fr.nbytes} B")

    # ---- legacy scales.bin ----
    n_sc = 1 + L * len(MATS)
    arr = np.zeros(212, dtype=np.float32)
    arr[:n_sc] = 0.01  # placeholder; ignored when layer_scales.bin present
    arr.tofile(os.path.join(out, "scales.bin"))

    # ---- verification ----
    dkv = DKV
    exp_layer = (D * 4
                 + (D * D // 2 + (D * D // G) * 2)
                 + 2 * (D * dkv // 2 + (D * dkv // G) * 2)
                 + (D * D // 2 + (D * D // G) * 2)
                 + D * 4
                 + 2 * (D * F // 2 + (D * F // G) * 2)
                 + (F * D // 2 + (F * D // G) * 2))
    l0 = os.path.getsize(os.path.join(out, "layer0_i4.bin"))
    print(f"[conv] layer0 size {l0} (expected {exp_layer}) {'OK' if l0 == exp_layer else 'MISMATCH'}")
    print(f"[conv] total i4 layers: {tot_i4/1e6:.1f} MB")
    if not args.no_embed:
        print(f"[conv] embed_i8 expected {V*D} B = {V*D/1e6:.1f} MB")
    print(f"[conv] done -> {out}/")


if __name__ == "__main__":
    main()
