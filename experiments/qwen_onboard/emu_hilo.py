#!/usr/bin/env python3
"""emu_hilo.py — Path B activation-precision experiment.

Question: Path B (direct per-channel INT8 weights) failed 1/3 with per-row INT8
activation.  Is the failure activation-precision-limited (fixable with a
hi/lo split = ~15-bit activation through 2x int8 matmuls, hardware-feasible)
or weight-error-limited (Path B dead)?

Variants:
  hilo_exact : hi/lo split activation, int32 exact accumulate (no rshift
               rounding) -> activation precision ceiling through int8 matmuls.
  hilo_rs    : hi/lo split + per-row rshift rounding (realistic TIU path).
  perrow_r0  : per-row single-scale, rshift = matmul_rshift(K) (no -5 headroom).
  perrow_r2  : per-row single-scale, rshift = matmul_rshift(K)+2 (more headroom).

Reuses emu_perrow.load_weights_B() (direct per-channel INT8 from bf16).
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


# ---- per-row quant + hi/lo split ----------------------------------------
def quant_per_row(x):
    mx = np.max(np.abs(x), axis=-1, keepdims=True)
    sc = np.maximum(mx / 127.0, 1e-12)
    q = np.clip(np.round(x / sc), -128, 127).astype(np.int8)
    return q, sc.ravel()


def quant_per_row_hilo(x):
    """x [M,K] -> (x1 int8 [M,K], x2 int8 [M,K], sc_fine [M]).
    x ~ (x1*128 + x2) * sc_fine, sc_fine = max|x|/127/128."""
    mx = np.max(np.abs(x), axis=-1, keepdims=True)
    sc = np.maximum(mx / 127.0, 1e-12)
    sc_fine = sc / 128.0
    x1 = np.clip(np.round(x / sc), -128, 127).astype(np.int32)
    r = x - x1 * sc
    x2 = np.clip(np.round(r / sc_fine), -128, 127).astype(np.int32)
    return x1.astype(np.int8), x2.astype(np.int8), sc_fine.ravel()


def matmul_hilo(x1, x2, w_i8, sc_fine, rshift, lsc_col):
    """hilo split through int8 matmul: acc = 128*acc1 + acc2, then per-row
    rshift rounding on each, dequant per-row.  Exact int32 if rshift<=0."""
    acc1 = x1.astype(np.int32) @ w_i8.astype(np.int32)
    acc2 = x2.astype(np.int32) @ w_i8.astype(np.int32)
    acc = 128 * acc1 + acc2
    if rshift is None or rshift <= 0:
        res = acc
    else:
        res = int8_round_div(acc, rshift).astype(np.int32) * (1 << rshift)
    return res.astype(np.float32) * sc_fine[:, None] * lsc_col.reshape(1, -1)


# ---- forward -------------------------------------------------------------
def forward_variant(tokens, embed, esc, frms, layers, variant, max_pos=None):
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

        if variant == "hilo_exact":
            x1, x2, sf = quant_per_row_hilo(h)
            q = matmul_hilo(x1, x2, mats["Wq"], sf, None, lsc[:D]) + bq
            k = matmul_hilo(x1, x2, mats["Wk"], sf, None, lsc[D:D + DKV]) + bk
            v = matmul_hilo(x1, x2, mats["Wv"], sf, None, lsc[D + DKV:D + 2 * DKV]) + bv
        elif variant == "hilo_rs":
            x1, x2, sf = quant_per_row_hilo(h)
            # hilo acc is ~128x larger (x1*128+x2), so rshift needs +7 over the
            # per-row -5 config -> matmul_rshift(K) + 2
            rsh = max(matmul_rshift(D) + 2, 8)
            q = matmul_hilo(x1, x2, mats["Wq"], sf, rsh, lsc[:D]) + bq
            k = matmul_hilo(x1, x2, mats["Wk"], sf, rsh, lsc[D:D + DKV]) + bk
            v = matmul_hilo(x1, x2, mats["Wv"], sf, rsh, lsc[D + DKV:D + 2 * DKV]) + bv
        else:
            x_i8, sc_row = quant_per_row(h)
            if variant == "perrow_r0":
                rsh = max(matmul_rshift(D), 8)
            elif variant == "perrow_r2":
                rsh = max(matmul_rshift(D) + 2, 8)
            else:
                rsh = max(matmul_rshift(D) - 5, 8)
            q = EP.matmul_i8_perrow(x_i8, mats["Wq"], rsh, sc_row, lsc[:D]) + bq
            k = EP.matmul_i8_perrow(x_i8, mats["Wk"], rsh, sc_row, lsc[D:D + DKV]) + bk
            v = EP.matmul_i8_perrow(x_i8, mats["Wv"], rsh, sc_row, lsc[D + DKV:D + 2 * DKV]) + bv

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

        # WO (use same variant for the attention output)
        if variant in ("hilo_exact", "hilo_rs"):
            a1, a2, sf_a = quant_per_row_hilo(attn)
            rsh_a = None if variant == "hilo_exact" else max(matmul_rshift(D) + 2, 8)
            wo = matmul_hilo(a1, a2, mats["Wo"], sf_a, rsh_a, lsc[2 * DKV + D:2 * DKV + 2 * D])
        else:
            a_i8, sc_a = quant_per_row(attn)
            wo = EP.matmul_i8_perrow(a_i8, mats["Wo"], rsh, sc_a, lsc[2 * DKV + D:2 * DKV + 2 * D])
        x = x + wo

        # FFN
        h = rms_norm(x, rms_ffn)
        if variant in ("hilo_exact", "hilo_rs"):
            x1, x2, sf = quant_per_row_hilo(h)
            rsh_ffn = max(matmul_rshift(D) + 2, 8)
            up = matmul_hilo(x1, x2, mats["up"], sf, None if variant == "hilo_exact" else rsh_ffn, lsc[LS["up"]:LS["up"] + F])
            gate = matmul_hilo(x1, x2, mats["gate"], sf, None if variant == "hilo_exact" else rsh_ffn, lsc[LS["gate"]:LS["gate"] + F])
            mid = up * silu(gate)
            out = np.zeros((seq, D), dtype=np.float32)
            KC = 1024
            rsh_d = None if variant == "hilo_exact" else max(matmul_rshift(KC) + 2, 8)
            for kc in range(0, F, KC):
                kcn = min(KC, F - kc)
                chunk = mid[:, kc:kc + kcn]
                m1, m2, sf_m = quant_per_row_hilo(chunk)
                wc = mats["down"][kc:kc + kcn, :]
                out += matmul_hilo(m1, m2, wc, sf_m, rsh_d, lsc[LS["down"]:LS["down"] + D])
        else:
            x_i8, sc_row = quant_per_row(h)
            up = EP.matmul_i8_perrow(x_i8, mats["up"], rsh, sc_row, lsc[LS["up"]:LS["up"] + F])
            gate = EP.matmul_i8_perrow(x_i8, mats["gate"], rsh, sc_row, lsc[LS["gate"]:LS["gate"] + F])
            mid = up * silu(gate)
            KC = 1024
            rsh_d = max(matmul_rshift(KC), 8) if variant == "perrow_r0" else (max(matmul_rshift(KC) + 2, 8) if variant == "perrow_r2" else max(matmul_rshift(KC) - 5, 8))
            out = np.zeros((seq, D), dtype=np.float32)
            for kc in range(0, F, KC):
                kcn = min(KC, F - kc)
                m_i8, sc_m = quant_per_row(mid[:, kc:kc + kcn])
                wc = mats["down"][kc:kc + kcn, :]
                out += EP.matmul_i8_perrow(m_i8, wc, rsh_d, sc_m, lsc[LS["down"]:LS["down"] + D])
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
    for variant in ("perrow_r2", "perrow_r0", "hilo_rs", "hilo_exact"):
        passed = 0
        row = {}
        for p in PROMPTS:
            ids = tok.encode(p, add_special_tokens=False) if tok else [0]
            logits = forward_variant(ids, embed, esc, frms, layers, variant)
            t5 = np.argsort(-logits)[:5].tolist()
            v5 = logits[t5].tolist()
            gap = v5[0] - v5[1] if len(v5) > 1 else 0.0
            ok = t5[0] == REF[p]
            passed += ok
            row[p] = {"next": t5[0], "ref": REF[p], "ok": bool(ok), "gap": float(gap), "top5": t5}
            print(f"[{variant}] '{p}' next={t5[0]} ref={REF[p]} {'OK' if ok else 'MISMATCH'} gap={gap:.3f}", flush=True)
        out[variant] = row
        print(f"[{variant}] ============ VERDICT {passed}/3 ============", flush=True)
    with open(os.path.join(HERE, "hilo_ref.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
