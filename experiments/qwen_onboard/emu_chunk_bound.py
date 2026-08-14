#!/usr/bin/env python3
"""emu_chunk_bound.py — ps32-free hardware-realization test.

emu_chunk8.rnd's per-row rshift uses the EXACT int32 acc max (needs ps32 readback
to know the output magnitude before rounding).  On CV1800B ps32 is unreadable, so
the hardware can only use a DETERMINISTIC bound computed pre-matmul.

Bound per output row m:
    B_m = sum_k |x_i8[m,k]| * wcolmax[k],   wcolmax[k] = max_n |w_i8[k,n]|
    r_m  = ceil(log2(B_m / 127))            (rshift so int8 output fills range)
This is computable in the CPU/TPU pipeline before the matmul (x known, wcolmax
precomputed per weight matrix).  Question: does bound-based rshift keep 3/3, or
does it degrade toward chunk8_fixed (2/3) because B_m overestimates the exact max
(loses ~1-2 bits to cancellation/column variation)?
"""
import os, sys, json, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import emu_chunk8 as C8
import emu_perrow as EP
from emu_perrow import (D, H, KVH, HD, L, F, V, DKV, GROUPS, PROMPTS, REF,
                        matmul_rshift, int8_round_div, precompute_rope,
                        rope, rms_norm, silu, LS, LS_PL)
from qwen_emu import ROPE_THETA, EPS


def matmul_chunk_bound(x_i8, w_i8, sc_row, lsc_col, KG):
    """Per-chunk K matmul; per-row rshift from deterministic bound B_m."""
    M, K = x_i8.shape
    N = w_i8.shape[1]
    out = np.zeros((M, N), dtype=np.float64)
    x = x_i8.astype(np.int32)
    w = w_i8.astype(np.int32)
    wcolmax = np.max(np.abs(w), axis=1)          # [K] max |w| per K-row
    for g in range(0, K, KG):
        kc = min(KG, K - g)
        acc = x[:, g:g + kc] @ w[g:g + kc, :]     # [M,N] int32
        part = np.zeros_like(acc, dtype=np.float64)
        xs = np.abs(x[:, g:g + kc])               # [M,kc]
        for m in range(M):
            B = float(xs[m] @ wcolmax[g:g + kc])  # bound for row m
            if B > 1e-6:
                r = int(math.ceil(math.log2(B / 127.0)))
                r = max(r, 0)
            else:
                r = 0
            res = int8_round_div(acc[m], r)
            part[m] = res.astype(np.float64) * (1 << r)
        out += part
    out = out.astype(np.float32)
    return out * sc_row[:, None] * lsc_col.reshape(1, -1)


def forward(tokens, embed, esc, frms, layers, KG, max_pos=None):
    seq = len(tokens)
    max_pos = max_pos or max(64, seq + 8)
    cos, sin = precompute_rope(max_pos, HD, ROPE_THETA)
    x = np.zeros((seq, D), dtype=np.float32)
    for i, t in enumerate(tokens):
        t = 0 if (t < 0 or t >= V) else t
        x[i] = embed[t].astype(np.float32) * esc[t]

    for l, (rms_attn, mats, rms_ffn, lsc, bias) in enumerate(layers):
        bq, bk, bv = bias[:D], bias[D:D + DKV], bias[D + DKV:]
        h = rms_norm(x, rms_attn)
        x_i8, sc_row = C8.quant_per_row(h)
        q = matmul_chunk_bound(x_i8, mats["Wq"], sc_row, lsc[:D], KG) + bq
        k = matmul_chunk_bound(x_i8, mats["Wk"], sc_row, lsc[D:D + DKV], KG) + bk
        v = matmul_chunk_bound(x_i8, mats["Wv"], sc_row, lsc[D + DKV:D + 2 * DKV], KG) + bv
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
            logits = (qh @ kh.T) * (1.0 / math.sqrt(HD))
            mask = np.triu(np.ones((seq, seq)), 1).astype(bool)
            logits = np.where(mask, -1e30, logits)
            probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
            probs /= probs.sum(axis=-1, keepdims=True)
            attn[:, hh, :] = probs @ vh
        attn = attn.reshape(seq, D)
        a_i8, sc_a = C8.quant_per_row(attn)
        wo = matmul_chunk_bound(a_i8, mats["Wo"], sc_a, lsc[2 * DKV + D:2 * DKV + 2 * D], KG)
        x = x + wo
        h = rms_norm(x, rms_ffn)
        x_i8, sc_row = C8.quant_per_row(h)
        up = matmul_chunk_bound(x_i8, mats["up"], sc_row, lsc[LS["up"]:LS["up"] + F], KG)
        gate = matmul_chunk_bound(x_i8, mats["gate"], sc_row, lsc[LS["gate"]:LS["gate"] + F], KG)
        mid = up * silu(gate)
        out = np.zeros((seq, D), dtype=np.float32)
        for kc in range(0, F, 1024):
            kcn = min(1024, F - kc)
            m_i8, sc_m = C8.quant_per_row(mid[:, kc:kc + kcn])
            wc = mats["down"][kc:kc + kcn, :]
            out += matmul_chunk_bound(m_i8, wc, sc_m, lsc[LS["down"]:LS["down"] + D], KG)
        x = x + out

    h = rms_norm(x, frms)
    logits = h[-1] @ (embed.astype(np.float32) * esc.reshape(-1, 1)).T
    return logits


def main():
    embed, esc, frms, layers = EP.load_weights_B()
    tok = None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(os.path.join(C8.PRE, "model"), trust_remote_code=True)
    except Exception:
        pass
    results = {}
    for KG in (32, 128, 256):
        passed = 0
        row = {}
        for p in PROMPTS:
            ids = tok.encode(p, add_special_tokens=False) if tok else [0]
            logits = forward(ids, embed, esc, frms, layers, KG)
            t5 = np.argsort(-logits)[:5].tolist()
            v5 = logits[t5].tolist()
            gap = v5[0] - v5[1] if len(v5) > 1 else 0.0
            ok = t5[0] == REF[p]
            passed += ok
            row[p] = {"next": t5[0], "ref": REF[p], "ok": bool(ok), "gap": float(gap)}
            print(f"[bound KG={KG}] '{p}' next={t5[0]} ref={REF[p]} {'OK' if ok else 'MISMATCH'} gap={gap:.3f}", flush=True)
        results[KG] = row
        print(f"[bound KG={KG}] ============ VERDICT {passed}/3 ============", flush=True)
    with open(os.path.join(HERE, "chunk_bound.json"), "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
