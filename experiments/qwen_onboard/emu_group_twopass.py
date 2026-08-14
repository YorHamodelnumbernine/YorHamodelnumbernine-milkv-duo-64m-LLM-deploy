#!/usr/bin/env python3
"""emu_group_twopass.py — Path A (K-aligned per-group INT8) + two-pass rshift.

Path A's original blocker was per-group dequant matmul needing exact int32 (ps32).
This test: K-aligned per-group INT8 weights (K-groups of G=32), per-chunk KG=G
matmul with two-pass rshift (pass1 safe -> read max -> pass2 refined), CPU fp32
accumulate with per-(chunk,col) group scale.  ps32-FREE.  Keeps INT4-size SD
footprint if INT4 stored + dequant to per-group INT8 on CPU; here we build
per-group INT8 directly from bf16 to test quality only.

Dequant: out[m,n] = sum_g res_i8_g[m,n] * 2^r_g * gscale[g,n], then * sc_row[m].
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

PRE = os.path.join(HERE, "..", "qwen_int4")
sys.path.insert(0, PRE)
from sf_io import SF
import qwen_emu as QE

G = 32


def quant_groups(W_hf):
    """[N,K] hf weight -> (q [K,N] int8, gscale [K/G,N] fp32), K-aligned G=32."""
    Wt = np.ascontiguousarray(W_hf.T.astype(np.float32))    # [K,N]
    K, N = Wt.shape
    q = np.zeros((K, N), dtype=np.int8)
    gs = np.zeros((K // G, N), dtype=np.float32)
    for g in range(0, K, G):
        seg = Wt[g:g + G, :]                                # [G,N]
        mx = np.max(np.abs(seg), axis=0)
        sc = np.maximum(mx / 127.0, 1e-12)
        q[g:g + G, :] = np.clip(np.round(seg / sc), -128, 127).astype(np.int8)
        gs[g // G, :] = sc
    return q, gs


def matmul_group_twopass(x_i8, w_i8, gscale, sc_row, KG=G):
    """w_i8 [K,N] per-group INT8, gscale [K/G,N].  Two-pass rshift per chunk.
    Per-group scale is folded in dequant; no separate lsc_col (would double-scale)."""
    M, K = x_i8.shape
    N = w_i8.shape[1]
    out = np.zeros((M, N), dtype=np.float64)
    x = x_i8.astype(np.int32)
    w = w_i8.astype(np.int32)
    for g in range(0, K, KG):
        kc = min(KG, K - g)
        acc = x[:, g:g + kc] @ w[g:g + kc, :]               # [M,N] int32
        rsafe = max(matmul_rshift(kc) - 5, 8)
        p1 = int8_round_div(acc, rsafe)
        est = int(np.max(np.abs(p1))) * (1 << rsafe)
        r = 0
        if est > 1e-6:
            r = int(math.ceil(math.log2(est / 127.0)))
            r = max(r, 0)
        p2 = int8_round_div(acc, r).astype(np.float64) * (1 << r)
        out += p2 * gscale[g // KG, :].reshape(1, -1)       # per-(chunk,col) group scale
    out = out.astype(np.float32)
    return out * sc_row[:, None]


def load_weights_group():
    """Build per-group INT8 K-aligned weights from bf16 safetensors."""
    sf = SF(os.path.join(PRE, "model", "model.safetensors"))
    embed, esc, frms, layers0 = QE.load_weights()           # reuse embed/esc/frms
    MATS = [("q_proj", "self_attn", D, D), ("k_proj", "self_attn", D, DKV),
            ("v_proj", "self_attn", D, DKV), ("o_proj", "self_attn", D, D),
            ("up_proj", "mlp", D, F), ("gate_proj", "mlp", D, F),
            ("down_proj", "mlp", F, D)]
    new_layers = []
    for l in range(L):
        p = f"model.layers.{l}"
        rms_attn = sf.get(f"{p}.input_layernorm.weight").astype(np.float32)
        rms_ffn = sf.get(f"{p}.post_attention_layernorm.weight").astype(np.float32)
        mats, gsc = {}, {}
        for name, grp, K, N in MATS:
            w = sf.get(f"{p}.{grp}.{name}.weight")
            q, gs = quant_groups(w)
            key = {"q_proj": "Wq", "k_proj": "Wk", "v_proj": "Wv", "o_proj": "Wo",
                   "up_proj": "up", "gate_proj": "gate", "down_proj": "down"}[name]
            mats[key] = q
            gsc[key] = gs
        lsc = np.concatenate([np.max(np.abs(sf.get(f"{p}.{grp}.{n}.weight").T.astype(np.float32)), axis=0) / 127.0
                              for (n, grp) in (("q_proj", "self_attn"), ("k_proj", "self_attn"),
                                               ("v_proj", "self_attn"), ("o_proj", "self_attn"),
                                               ("up_proj", "mlp"), ("gate_proj", "mlp"),
                                               ("down_proj", "mlp"))])
        bias = np.concatenate([sf.get(f"{p}.self_attn.{n}.bias").astype(np.float32)
                               for n in ("q_proj", "k_proj", "v_proj")])
        new_layers.append((rms_attn, mats, rms_ffn, lsc, bias, gsc))
    return embed, esc, frms, new_layers


def forward(tokens, embed, esc, frms, layers, max_pos=None):
    seq = len(tokens)
    max_pos = max_pos or max(64, seq + 8)
    cos, sin = precompute_rope(max_pos, HD, ROPE_THETA)
    x = np.zeros((seq, D), dtype=np.float32)
    for i, t in enumerate(tokens):
        t = 0 if (t < 0 or t >= V) else t
        x[i] = embed[t].astype(np.float32) * esc[t]

    for l, (rms_attn, mats, rms_ffn, lsc, bias, gsc) in enumerate(layers):
        bq, bk, bv = bias[:D], bias[D:D + DKV], bias[D + DKV:]
        h = rms_norm(x, rms_attn)
        x_i8, sc_row = C8.quant_per_row(h)
        q = matmul_group_twopass(x_i8, mats["Wq"], gsc["Wq"], sc_row) + bq
        k = matmul_group_twopass(x_i8, mats["Wk"], gsc["Wk"], sc_row) + bk
        v = matmul_group_twopass(x_i8, mats["Wv"], gsc["Wv"], sc_row) + bv
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
        wo = matmul_group_twopass(a_i8, mats["Wo"], gsc["Wo"], sc_a)
        x = x + wo
        h = rms_norm(x, rms_ffn)
        x_i8, sc_row = C8.quant_per_row(h)
        up = matmul_group_twopass(x_i8, mats["up"], gsc["up"], sc_row)
        gate = matmul_group_twopass(x_i8, mats["gate"], gsc["gate"], sc_row)
        mid = up * silu(gate)
        out = np.zeros((seq, D), dtype=np.float32)
        for kc in range(0, F, 1024):
            kcn = min(1024, F - kc)
            m_i8, sc_m = C8.quant_per_row(mid[:, kc:kc + kcn])
            wc = mats["down"][kc:kc + kcn, :]
            gsw = gsc["down"][kc // G:(kc + kcn) // G, :]
            out += matmul_group_twopass(m_i8, wc, gsw, sc_m)
        x = x + out

    h = rms_norm(x, frms)
    logits = h[-1] @ (embed.astype(np.float32) * esc.reshape(-1, 1)).T
    return logits


def main():
    print("[A+twopass] loading per-group INT8 weights...", flush=True)
    embed, esc, frms, layers = load_weights_group()
    print("[A+twopass] loaded.", flush=True)
    tok = None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(os.path.join(PRE, "model"), trust_remote_code=True)
    except Exception as e:
        print("tokenizer fail", e)
    passed = 0
    row = {}
    for p in PROMPTS:
        ids = tok.encode(p, add_special_tokens=False) if tok else [0]
        logits = forward(ids, embed, esc, frms, layers)
        t5 = np.argsort(-logits)[:5].tolist()
        v5 = logits[t5].tolist()
        gap = v5[0] - v5[1] if len(v5) > 1 else 0.0
        ok = t5[0] == REF[p]
        passed += ok
        row[p] = {"next": t5[0], "ref": REF[p], "ok": bool(ok), "gap": float(gap), "top5": t5}
        print(f"[A+twopass] '{p}' next={t5[0]} ref={REF[p]} {'OK' if ok else 'MISMATCH'} gap={gap:.3f}", flush=True)
    print(f"[A+twopass] ============ VERDICT {passed}/3 ============", flush=True)
    with open(os.path.join(HERE, "group_twopass.json"), "w") as f:
        json.dump({"verdict": f"{passed}/3", "prompts": row}, f, indent=1)


if __name__ == "__main__":
    main()
