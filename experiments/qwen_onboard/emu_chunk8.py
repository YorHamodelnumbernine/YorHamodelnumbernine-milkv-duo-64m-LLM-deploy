#!/usr/bin/env python3
"""emu_chunk8.py — ps32-free fallback test: per-chunk K=32 matmul with
per-chunk adaptive int8 rshift + CPU fp32 accumulate.

Motivation: ps32 int32 output may be dead on CV1800B.  hilo_exact shows exact
int32 accumulation is required for Qwen 3/3.  The only ps32-free way to shrink
the int8 output rounding error is to split the K contraction into small chunks
(K=32) and pick the rshift PER CHUNK so each int8 output fills its range
(relative error ~1/254 instead of ~4% for full-K fixed rshift).  CPU then
dequantizes per chunk (per-row act scale * per-chunk rshift * per-ch w scale)
and accumulates in fp32.

Variants:
  chunk8_exact : per-chunk, NO int8 rounding (exact int32) -> sanity upper bound
  chunk8_rnd   : per-chunk, adaptive int8 rounding (realistic ps32-free path)
  chunk8_fixed : per-chunk, FIXED rshift (matmul_rshift(32)-5) -> shows the
                 per-chunk rshift adaptation is what matters

Reuses emu_perrow.load_weights_B() (direct per-channel INT8 weights).
"""
import os, sys, json, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import emu_perrow as EP
from emu_perrow import (D, H, KVH, HD, L, F, V, DKV, GROUPS, PROMPTS, REF,
                        matmul_rshift, int8_round_div, precompute_rope,
                        rope, rms_norm, silu, LS, LS_PL)
from qwen_emu import ROPE_THETA, EPS

PRE = os.path.join(HERE, "..", "qwen_int4")
G = 32


def quant_per_row(x):
    mx = np.max(np.abs(x), axis=-1, keepdims=True)
    sc = np.maximum(mx / 127.0, 1e-12)
    q = np.clip(np.round(x / sc), -128, 127).astype(np.int8)
    return q, sc.ravel()


def matmul_chunk(x_i8, w_i8, sc_row, lsc_col, variant, KG=32):
    """x_i8 [M,K], w_i8 [K,N] -> fp32 [M,N].  Split K into KG-chunks.
    Per chunk: acc_g [M,N] int32; round to int8 with (variant-dependent)
    rshift; dequant per-row and per-ch; fp32 accumulate over chunks."""
    M, K = x_i8.shape
    N = w_i8.shape[1]
    out = np.zeros((M, N), dtype=np.float64)
    x = x_i8.astype(np.int32)
    w = w_i8.astype(np.int32)
    for g in range(0, K, KG):
        kc = min(KG, K - g)
        acc = x[:, g:g + kc] @ w[g:g + kc, :]           # [M,N] int32
        if variant == "chunk8_exact":
            part = acc.astype(np.float64)
        elif variant == "chunk8_fixed":
            rsh = max(matmul_rshift(kc) - 5, 8)
            res = int8_round_div(acc, rsh)
            part = res.astype(np.float64) * (1 << rsh)
        else:  # chunk8_rnd: adaptive per-chunk rshift (per row)
            part = np.zeros_like(acc, dtype=np.float64)
            for m in range(M):
                am = acc[m]
                # rshift so max|am|/2^r ~ 127
                mx = np.max(np.abs(am))
                if mx > 1e-6:
                    r = int(math.ceil(math.log2(mx / 127.0)))
                    r = max(r, 0)
                else:
                    r = 0
                res = int8_round_div(am, r)
                part[m] = res.astype(np.float64) * (1 << r)
        out += part
    out = out.astype(np.float32)
    return out * sc_row[:, None] * lsc_col.reshape(1, -1)


def forward(tokens, embed, esc, frms, layers, variant, max_pos=None):
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
        x_i8, sc_row = quant_per_row(h)
        q = matmul_chunk(x_i8, mats["Wq"], sc_row, lsc[:D], variant) + bq
        k = matmul_chunk(x_i8, mats["Wk"], sc_row, lsc[D:D + DKV], variant) + bk
        v = matmul_chunk(x_i8, mats["Wv"], sc_row, lsc[D + DKV:D + 2 * DKV], variant) + bv
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
        a_i8, sc_a = quant_per_row(attn)
        wo = matmul_chunk(a_i8, mats["Wo"], sc_a, lsc[2 * DKV + D:2 * DKV + 2 * D], variant)
        x = x + wo
        # FFN
        h = rms_norm(x, rms_ffn)
        x_i8, sc_row = quant_per_row(h)
        up = matmul_chunk(x_i8, mats["up"], sc_row, lsc[LS["up"]:LS["up"] + F], variant)
        gate = matmul_chunk(x_i8, mats["gate"], sc_row, lsc[LS["gate"]:LS["gate"] + F], variant)
        mid = up * silu(gate)
        out = np.zeros((seq, D), dtype=np.float32)
        for kc in range(0, F, 1024):
            kcn = min(1024, F - kc)
            m_i8, sc_m = quant_per_row(mid[:, kc:kc + kcn])
            wc = mats["down"][kc:kc + kcn, :]
            out += matmul_chunk(m_i8, wc, sc_m, lsc[LS["down"]:LS["down"] + D], variant)
        x = x + out

    h = rms_norm(x, frms)
    logits = h[-1] @ (embed.astype(np.float32) * esc.reshape(-1, 1)).T
    return logits


def main():
    embed, esc, frms, layers = EP.load_weights_B()
    tok = None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(os.path.join(PRE, "model"), trust_remote_code=True)
    except Exception:
        pass
    out = {}
    for variant in ("chunk8_rnd", "chunk8_fixed", "chunk8_exact"):
        passed = 0
        row = {}
        for p in PROMPTS:
            ids = tok.encode(p, add_special_tokens=False) if tok else [0]
            logits = forward(ids, embed, esc, frms, layers, variant)
            t5 = np.argsort(-logits)[:5].tolist()
            v5 = logits[t5].tolist()
            gap = v5[0] - v5[1] if len(v5) > 1 else 0.0
            ok = t5[0] == REF[p]
            passed += ok
            row[p] = {"next": t5[0], "ref": REF[p], "ok": bool(ok), "gap": float(gap), "top5": t5}
            print(f"[{variant}] '{p}' next={t5[0]} ref={REF[p]} {'OK' if ok else 'MISMATCH'} gap={gap:.3f}", flush=True)
        out[variant] = row
        print(f"[{variant}] ============ VERDICT {passed}/3 ============", flush=True)
    with open(os.path.join(HERE, "chunk8_ref.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
