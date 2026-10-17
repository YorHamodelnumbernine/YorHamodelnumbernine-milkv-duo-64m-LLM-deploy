#!/usr/bin/env python3
"""INT4 group quantization (bf16 -> int4, per-group fp16 scale) for Qwen2.5-0.5B.

Symmetric RTN baseline + optional asymmetric zero-point + optional AWQ-style
per-input-channel activation scaling.  Pure host tooling, numpy + torch only.

Quantized layout (matches repo DESIGN_INT4_WEIGHTS.md spirit, but native bf16):
  nibbles   : n/2 bytes  (low nibble = elem 2i, high nibble = elem 2i+1)
  scales    : per G elems, fp16 (2B)
  [asym] zp : per G elems, fp16 (2B), offset applied before round
"""
import numpy as np

SYM_QMAX = 7  # symmetric range [-8,7], scale=max/7 (matches repo int4_pack)

def group_quantize(W, G, asym=False, scale_dtype="fp16"):
    """W: [out,in] row-major. Returns (nib, scale_arr, zp_arr_or_None, q, dequant)."""
    Wf = W.reshape(-1).astype(np.float32)
    n = Wf.shape[0]
    assert n % G == 0, f"{n} not divisible by G={G}"
    grp = Wf.reshape(-1, G)
    if asym:
        mn = grp.min(axis=1)
        mx = grp.max(axis=1)
        scale = (mx - mn) / 15.0
        zp = np.round(-mn / np.maximum(scale, 1e-12)).astype(np.float32)
        scale = np.maximum(scale, 1e-12)
        q = np.clip(np.round(grp / scale[:, None]) + zp[:, None], 0, 15).astype(np.int8)
        dequant = (q.astype(np.float32) - zp[:, None]) * scale[:, None]
    else:
        mx = np.max(np.abs(grp), axis=1)
        scale = np.maximum(mx / SYM_QMAX, 1e-12)
        q = np.clip(np.round(grp / scale[:, None]), -8, 7).astype(np.int8)
        zp = None
        dequant = q.astype(np.float32) * scale[:, None]
    dequant = dequant.reshape(W.shape)
    # pack nibbles: low nibble = even idx, high nibble = odd idx
    qi = q.reshape(-1).astype(np.uint8)
    if qi.shape[0] % 2:
        qi = np.concatenate([qi, np.zeros(1, np.uint8)])
    nib = (qi[0::2] & 0x0F) | ((qi[1::2] & 0x0F) << 4)
    if scale_dtype == "fp16":
        s = scale.astype(np.float16)
        zp_a = zp.astype(np.float16) if zp is not None else None
    else:  # fp32
        s = scale.astype(np.float32)
        zp_a = zp.astype(np.float32) if zp is not None else None
    return nib.tobytes(), s, zp_a, q, dequant


def quant_error(W, G, asym=False):
    _, _, _, _, dq = group_quantize(W, G, asym=asym)
    d = W.astype(np.float32) - dq
    return dq


def volume_table(weights, vocab):
    """weights: dict name->tensor; returns (rows for md, total_mb per config)."""
    rows = []
    total_elems = 0
    for name, w in weights.items():
        n = w.reshape(-1).shape[0]
        total_elems += n
        rows.append((name, w.shape, n))
    res = {}
    for G in (32, 64, 128):
        nib_mb = total_elems / 2 / 1e6
        sc_mb = (total_elems / G) * 2 / 1e6
        res[G] = dict(nib_mb=nib_mb, sc_mb=sc_mb,
                      total_mb=nib_mb + sc_mb,
                      ratio=(nib_mb + sc_mb) / (total_elems * 2 / 1e6))
    return rows, res


def linear_names(cfg):
    """All nn.Linear weight matrices to quantize (attention + MLP)."""
    names = []
    for l in range(cfg["num_hidden_layers"]):
        for p in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                  "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
            names.append(f"model.layers.{l}.{p}.weight")
    return names
