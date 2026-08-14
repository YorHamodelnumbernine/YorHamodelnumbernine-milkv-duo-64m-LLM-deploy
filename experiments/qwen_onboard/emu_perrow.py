#!/usr/bin/env python3
"""emu_perrow.py — Path B / A' 质量验证 with per-row (per-token) INT8 activation.

Closes two open questions after the ps32 on-board spike (per-group dequant matmul
likely DEAD on CV1800B):

  A' = INT4 G32 stored -> CPU weight-side dequant to per-channel INT8
       + per-row INT8 activation          (insurance route; TPU zero new ops)
  B  = direct per-channel INT8 from bf16  (no INT4 intermediate)
       + per-row INT8 activation          (Path B engine-realistic)

!!! IMPORTANT (2026-08-13) !!!
This script emulates a NAIVE SINGLE-PASS matmul with a fixed conservative
rshift = matmul_rshift(K)-5 (>=8).  That single-pass model is TOO LOSSY:
A' = 0/3, B = 1/3 (see perrow_ref.json).  The on-board engine uses the
TWO-PASS data-adaptive rshift (pass1 safe -> read max -> pass2 refined),
which is what emu_wsdq.py (A') and emu_b_perrow_twopass.py (B) implement.
Both A' and B are 3/3 with per-row + two-pass.  Do NOT use this script's
verdict as the engine-quality signal; use emu_wsdq.py / emu_b_perrow_twopass.py.

Reference next_token (bf16): 2130 / 12095 / 99366.

Usage:
  python3 emu_perrow.py                 # run A' (reuse weights/layerN_i4.bin)
  python3 emu_perrow.py --pathB         # run B (build direct per-ch INT8 from bf16)
"""
import os, sys, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from qwen_emu import (D, H, KVH, HD, L, F, V, G, DKV, GROUPS, EPS,
                      ROPE_THETA, matmul_rshift, int8_round_div, precompute_rope,
                      rope, rms_norm, silu, fp16_to_f32, LAYOUT, LS, LS_PL)
import qwen_emu as QE

PRE = os.path.join(HERE, "..", "qwen_int4")
sys.path.insert(0, PRE)
from sf_io import SF

REF = {  # bf16 next_token per prompt (from p0_refs.json)
    "中国的首都是": 2130,
    "The capital of France is": 12095,
    "今天天气很好，我们去公园": 99366,
}
PROMPTS = list(REF.keys())


# ----------------------------------------------------------------------
#  per-row INT8 activation (engine semantics)
# ----------------------------------------------------------------------
def quant_per_row(x):
    """x [M,K] fp32 -> (q_i8 [M,K], scale [M]) per-row symmetric INT8."""
    mx = np.max(np.abs(x), axis=-1, keepdims=True)
    sc = np.maximum(mx / 127.0, 1e-12)
    q = np.clip(np.round(x / sc), -128, 127).astype(np.int8)
    return q, sc.ravel()


def matmul_i8_perrow(x_i8, w_i8, rshift, sc_row, lsc_col):
    """x_i8 [M,K], w_i8 [K,N] -> fp32 [M,N].
    out[m,n] = res_i8[m,n] * sc_row[m] * 2^rshift * lsc[n]."""
    acc = x_i8.astype(np.int32) @ w_i8.astype(np.int32)
    res = int8_round_div(acc, rshift)
    b = sc_row[:, None] * (1 << rshift)
    return res.astype(np.float32) * b * lsc_col.reshape(1, -1)


def forward_perrow(tokens, embed, esc, frms, layers, max_pos=None):
    """Full forward with per-row INT8 activation + per-channel INT8 weights."""
    seq = len(tokens)
    if max_pos is None:
        max_pos = max(64, seq + 8)
    cos, sin = precompute_rope(max_pos, HD, ROPE_THETA)
    x = np.zeros((seq, D), dtype=np.float32)
    for i, t in enumerate(tokens):
        t = 0 if (t < 0 or t >= V) else t
        x[i] = embed[t].astype(np.float32) * esc[t]

    for l, (rms_attn, mats, rms_ffn, lsc, bias) in enumerate(layers):
        bq, bk, bv = bias[:D], bias[D:D + DKV], bias[D + DKV:]
        # ---- attention QKV ----
        h = rms_norm(x, rms_attn)
        x_i8, sc_row = quant_per_row(h)
        rsh = max(matmul_rshift(D) - 5, 8)
        q = matmul_i8_perrow(x_i8, mats["Wq"], rsh, sc_row, lsc[:D]) + bq
        k = matmul_i8_perrow(x_i8, mats["Wk"], rsh, sc_row, lsc[D:D + DKV]) + bk
        v = matmul_i8_perrow(x_i8, mats["Wv"], rsh, sc_row, lsc[D + DKV:D + 2 * DKV]) + bv
        # ---- GQA attention (fp32) ----
        q = q.reshape(seq, H, HD); k = k.reshape(seq, KVH, HD); v = v.reshape(seq, KVH, HD)
        for s in range(seq):
            for hh in range(H):
                q[s, hh] = rope(q[s, hh], s, cos, sin, HD)
            for hh in range(KVH):
                k[s, hh] = rope(k[s, hh], s, cos, sin, HD)
        attn = np.zeros((seq, H, HD), dtype=np.float32)
        for hh in range(H):
            kvh = hh // GROUPS
            qh = q[:, hh, :]; kh = k[:, kvh, :]; vh = v[:, kvh, :]
            logits = (qh @ kh.T) * (1.0 / math_sqrt(HD))
            mask = np.triu(np.ones((seq, seq)), 1).astype(bool)
            logits = np.where(mask, -1e30, logits)
            probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
            probs /= probs.sum(axis=-1, keepdims=True)
            attn[:, hh, :] = probs @ vh
        attn = attn.reshape(seq, D)
        # ---- WO ----
        a_i8, sc_a = quant_per_row(attn)
        rsh = max(matmul_rshift(D) - 5, 8)
        wo = matmul_i8_perrow(a_i8, mats["Wo"], rsh, sc_a,
                              lsc[2 * DKV + D:2 * DKV + 2 * D])
        x = x + wo
        # ---- FFN ----
        h = rms_norm(x, rms_ffn)
        x_i8, sc_row = quant_per_row(h)
        rsh = max(matmul_rshift(D) - 5, 8)
        up = matmul_i8_perrow(x_i8, mats["up"], rsh, sc_row, lsc[LS["up"]:LS["up"] + F])
        gate = matmul_i8_perrow(x_i8, mats["gate"], rsh, sc_row, lsc[LS["gate"]:LS["gate"] + F])
        mid = up * silu(gate)
        # down: K-chunked with per-chunk per-row scale
        KC = 1024
        rsh_d = max(matmul_rshift(KC) - 5, 8)
        out = np.zeros((seq, D), dtype=np.float32)
        for kc in range(0, F, KC):
            kcn = min(KC, F - kc)
            chunk = mid[:, kc:kc + kcn]
            m_i8, sc_m = quant_per_row(chunk)
            wchunk = mats["down"][kc:kc + kcn, :]
            res = matmul_i8_perrow(m_i8, wchunk, rsh_d, sc_m,
                                   lsc[LS["down"]:LS["down"] + D])
            out += res
        x = x + out

    h = rms_norm(x, frms)
    logits = h[-1] @ (embed.astype(np.float32) * esc.reshape(-1, 1)).T
    return logits


import math
def math_sqrt(v):
    return math.sqrt(v)


# ----------------------------------------------------------------------
#  Weight loaders
# ----------------------------------------------------------------------
def load_weights_Aprime():
    """A': reuse existing INT4 layerN_i4.bin -> per-channel INT8 (weight-side dequant)."""
    return QE.load_weights()


def load_weights_B():
    """B: direct per-channel INT8 from bf16 (no INT4 intermediate)."""
    sf = SF(os.path.join(PRE, "model", "model.safetensors"))
    embed, esc, frms, layers = QE.load_weights()  # embed/scales/frms reuse

    def pc8(W_hf):
        Wt = np.ascontiguousarray(W_hf.T.astype(np.float32))   # [K,N]
        K, N = Wt.shape
        mx = np.max(np.abs(Wt), axis=0)
        sc = np.maximum(mx / 127.0, 1e-12)
        q = np.clip(np.round(Wt / sc), -128, 127).astype(np.int8)
        return q, sc

    MATS = [("q_proj", "self_attn", D, D), ("k_proj", "self_attn", D, DKV),
            ("v_proj", "self_attn", D, DKV), ("o_proj", "self_attn", D, D),
            ("up_proj", "mlp", D, F), ("gate_proj", "mlp", D, F),
            ("down_proj", "mlp", F, D)]
    SC_ORDER = ["Wq", "Wk", "Wv", "Wo", "up", "gate", "down"]

    new_layers = []
    for l in range(L):
        p = f"model.layers.{l}"
        rms_attn = sf.get(f"{p}.input_layernorm.weight").astype(np.float32)
        rms_ffn = sf.get(f"{p}.post_attention_layernorm.weight").astype(np.float32)
        mats, lsc = {}, []
        for name, grp, K, N in MATS:
            w = sf.get(f"{p}.{grp}.{name}.weight")
            q, sc = pc8(w)
            key = {"q_proj": "Wq", "k_proj": "Wk", "v_proj": "Wv", "o_proj": "Wo",
                   "up_proj": "up", "gate_proj": "gate", "down_proj": "down"}[name]
            mats[key] = q
            lsc.append(sc)
        lsc = np.concatenate(lsc)  # Wq Wk Wv Wo up gate down
        bias = np.concatenate([sf.get(f"{p}.self_attn.{n}.bias").astype(np.float32)
                               for n in ("q_proj", "k_proj", "v_proj")])
        new_layers.append((rms_attn, mats, rms_ffn, lsc, bias))
    return embed, esc, frms, new_layers


def run(layers, embed, esc, frms, tag):
    tok = None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(os.path.join(PRE, "model"),
                                            trust_remote_code=True)
    except Exception as e:
        print(f"[{tag}] tokenizer load failed: {e}")
    res = {"tag": tag, "prompts": {}}
    passed = 0
    for p in PROMPTS:
        ids = tok.encode(p, add_special_tokens=False) if tok else [0]
        logits = forward_perrow(ids, embed, esc, frms, layers)
        t5 = np.argsort(-logits)[:5].tolist()
        v5 = logits[t5].tolist()
        gap = (v5[0] - v5[1]) if len(v5) > 1 else 0.0
        ok = (t5[0] == REF[p])
        passed += ok
        res["prompts"][p] = {"n_tokens": len(ids), "next": t5[0], "ref": REF[p],
                             "ok": bool(ok), "top5": t5, "gap": float(gap)}
        print(f"[{tag}] '{p}' next={t5[0]} ref={REF[p]} {'OK' if ok else 'MISMATCH'} "
              f"gap={gap:.3f} top5={t5}", flush=True)
    res["verdict"] = f"{passed}/3"
    print(f"[{tag}] ============ VERDICT {passed}/3 ============", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pathB", action="store_true", help="run Path B (direct per-ch INT8 from bf16)")
    ap.add_argument("--out", default=os.path.join(HERE, "perrow_ref.json"))
    args = ap.parse_args()
    tag = "B_perrow" if args.pathB else "Aprime_perrow"
    print(f"[{tag}] loading weights...", flush=True)
    if args.pathB:
        embed, esc, frms, layers = load_weights_B()
    else:
        embed, esc, frms, layers = load_weights_Aprime()
    print(f"[{tag}] loaded.", flush=True)
    res = run(layers, embed, esc, frms, tag)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"[{tag}] wrote {args.out}")


if __name__ == "__main__":
    main()
