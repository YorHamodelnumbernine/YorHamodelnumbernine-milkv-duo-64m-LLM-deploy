#!/usr/bin/env python3
"""qwen_emu.py — numpy emulator of the Qwen2.5-0.5B on-board engine.

Loads the CONVERTED weight files (experiments/qwen_onboard/weights) and runs a
full forward pass replicating the on-board math, producing logits / next_token
for fixed prompts.  Two activation modes:

  default : fp32 activations, per-channel INT8 compute weights
            (validates converter weight precision only)
  --int8  : full INT8 activation + INT8 matmul emulation with the same
            rshift / per-channel dequant semantics as sm_layer_forward
            (closest prediction of the on-board engine)

Weights:
  layerN_i4.bin   INT4 G=32 -> unpacked to per-channel INT8 on the fly
                  (same math as the on-board int4_unpack_to_i8)
  layer_scales.bin per-channel fp32 scales
  embed_i8.bin    [V,D] INT8 per-row, embed_scales.f32 per-row scales
"""
import os, sys, struct, json, math, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.join(HERE, "weights")

D = 896; H = 14; KVH = 2; HD = 64; L = 24; F = 4864; V = 151936
G = 32; ROPE_THETA = 1e6; EPS = 1e-6
DKV = KVH * HD
GROUPS = H // KVH

# engine layer_scales.bin order (per layer): Wq(D) Wk(dkv) Wv(dkv) Wo(D) up(F) gate(F) down(D)
LS_PL = D + DKV + DKV + D + F + F + D
LS = {"Wq": 0, "Wk": D, "Wv": D + DKV, "Wo": D + 2 * DKV,
      "up": 2 * D + 2 * DKV, "gate": 2 * D + 2 * DKV + F, "down": 2 * D + 2 * DKV + 2 * F}
# engine layer file matrix order: Wq Wk Wv Wo up gate down
LAYOUT = [("Wq", D, D), ("Wk", D, DKV), ("Wv", D, DKV), ("Wo", D, D),
          ("up", D, F), ("gate", D, F), ("down", F, D)]


def fp16_to_f32(u16):
    u = u16.astype(np.uint32)
    s = (u >> 15) & 1
    e = (u >> 10) & 0x1F
    m = u & 0x3FF
    out = np.zeros_like(u)
    nz = (e != 0) & (e != 31)
    out[nz] = (s[nz] << 31) | ((e[nz] + 112) << 23) | (m[nz] << 13)
    sn = (e == 0) & (m != 0)
    if sn.any():
        m2 = m[sn].astype(np.uint64); e2 = np.full(m2.shape, 113, np.uint64)
        while (m2 & 0x400) == 0:
            m2 = m2 << 1; e2 -= 1
        m2 = m2 & 0x3FF
        out[sn] = (s[sn] << 31) | (e2 << 23) | (m2 << 13)
    z = (e == 0) & (m == 0)
    out[z] = s[z] << 31
    inf = (e == 31) & (m == 0); out[inf] = (s[inf] << 31) | 0x7F800000
    nan = (e == 31) & (m != 0); out[nan] = (s[nan] << 31) | 0x7FC00000
    return out.view("<f4")


def unpack_layer_i8(path, lsc):
    """Unpack layerN_i4.bin -> dict name->int8 [K,N] (per-channel), + rms arrays."""
    with open(path, "rb") as f:
        data = f.read()
    off = 0
    rms_attn = np.frombuffer(data[off:off + D * 4], dtype="<f4").copy(); off += D * 4
    mats = {}
    for name, K, N in LAYOUT:
        n = K * N
        nib = np.frombuffer(data[off:off + n // 2], dtype=np.uint8); off += n // 2
        gs = fp16_to_f32(np.frombuffer(data[off:off + (n // G) * 2], dtype="<u2")); off += (n // G) * 2
        q = np.empty(n, dtype=np.int8)
        lo = nib & 0x0F
        hi = (nib >> 4) & 0x0F
        lo = np.where(lo >= 8, lo - 16, lo)
        hi = np.where(hi >= 8, hi - 16, hi)
        q[0::2] = lo
        q[1::2] = hi
        qf = q.astype(np.float32).reshape(n // G, G)          # groups
        s_grp = gs.reshape(-1, 1)
        W_dq = qf * s_grp                                      # dequant INT4
        W_dq = W_dq.reshape(K, N)
        # per-channel int8 of the dequantized weight (scale slice from layer_scales.bin)
        off_sc = LS[name]
        sc = lsc[off_sc:off_sc + N]
        W_i8 = np.clip(np.round(W_dq / sc), -128, 127).astype(np.int8)
        mats[name] = W_i8
        # note: we drop W_dq; on-board stores only int8 compute weights
    rms_ffn = np.frombuffer(data[off:off + D * 4], dtype="<f4").copy(); off += D * 4
    assert off == len(data), f"trailing {len(data)-off} bytes"
    return rms_attn, mats, rms_ffn


def load_weights():
    cfg = np.fromfile(os.path.join(W, "config.bin"), dtype="<i4")
    lsc_all = np.fromfile(os.path.join(W, "layer_scales.bin"), dtype="<f4")
    embed = np.fromfile(os.path.join(W, "embed_i8.bin"), dtype=np.int8).reshape(V, D)
    esc = np.fromfile(os.path.join(W, "embed_scales.f32"), dtype="<f4")
    frms = np.fromfile(os.path.join(W, "final_rms.f32"), dtype="<f4")
    layers = []
    for l in range(L):
        lsc = lsc_all[l * LS_PL:(l + 1) * LS_PL]
        rms_attn, mats, rms_ffn = unpack_layer_i8(os.path.join(W, f"layer{l}_i4.bin"), lsc)
        # bias: [q(D) k(dkv) v(dkv)] fp32
        bias = np.fromfile(os.path.join(W, f"layer{l}_bias.f32"), dtype="<f4")
        layers.append((rms_attn, mats, rms_ffn, lsc, bias))
    return embed, esc, frms, layers


def rms_norm(x, g, eps=EPS):
    x = x.astype(np.float64)
    ss = np.mean(x * x, axis=-1, keepdims=True)
    return (x / np.sqrt(ss + eps) * g).astype(np.float32)


def silu(x):
    return x / (1.0 + np.exp(-x))


def rope(x, pos, cos, sin, hd):
    """x [.., hd], apply rope.  x is [., hd] with pairs (x[...,0:hd/2], x[...,hd/2:])."""
    half = hd // 2
    out = x.copy()
    x0 = x[..., :half]; x1 = x[..., half:]
    c = cos[pos]; s = sin[pos]
    out[..., :half] = x0 * c - x1 * s
    out[..., half:] = x0 * s + x1 * c
    return out


def precompute_rope(max_len, hd, theta):
    half = hd // 2
    freqs = 1.0 / (theta ** (np.arange(half) / half))
    pos = np.arange(max_len)
    ang = np.outer(pos, freqs)
    return np.cos(ang), np.sin(ang)


def matmul_rshift(K):
    r = 0; md = K * 127 * 127
    while (md >> r) > 127:
        r += 1
    return r


def int8_round_div(acc, rshift):
    """acc int32, result = round(acc / 2^rshift) clamped int8.
    TIU-confirmed (commit 23f300e / GATE_A_SIGNOFF): round-half-up toward +inf,
    sat8((acc + 2^(r-1)) >> r) — for negative exact half-way ties this rounds
    toward zero (e.g. -16>>5 -> 0), NOT round-half-away-from-zero."""
    scale = 1 << rshift
    r = (acc + (scale >> 1)) >> rshift   # numpy int32 >> is arithmetic shift (floor)
    return np.clip(r, -128, 127).astype(np.int8)


def matmul_i8(x_i8, w_i8, rshift, sc_x, lsc_col):
    """x_i8 [M,K], w_i8 [K,N] -> fp32 [M,N] using on-board semantics.
    out = round(Σ x*w / 2^rshift) * sc_x * 2^rshift * lsc[col]."""
    acc = x_i8.astype(np.int32) @ w_i8.astype(np.int32)   # [M,N] int32
    res = int8_round_div(acc, rshift)
    b = sc_x * (1 << rshift)
    return res.astype(np.float32) * b * lsc_col.reshape(1, -1)


def quant_i8_sym(x):
    mx = np.max(np.abs(x)) if x.size else 0.0
    sc = mx / 127.0 if mx > 1e-10 else 1.0
    q = np.clip(np.round(x / sc), -128, 127).astype(np.int8)
    return q, sc


def forward(tokens, embed, esc, frms, layers, int8_act=False, max_pos=None):
    seq = len(tokens)
    if max_pos is None:
        max_pos = max(64, seq + 8)
    cos, sin = precompute_rope(max_pos, HD, ROPE_THETA)
    # embed lookup (per-row int8 dequant)
    x = np.zeros((seq, D), dtype=np.float32)
    for i, t in enumerate(tokens):
        if t < 0 or t >= V:
            t = 0
        x[i] = embed[t].astype(np.float32) * esc[t]

    for l, (rms_attn, mats, rms_ffn, lsc, bias) in enumerate(layers):
        # --- attention ---
        bq, bk, bv = bias[:D], bias[D:D + DKV], bias[D + DKV:]
        h = rms_norm(x, rms_attn)
        if int8_act:
            x_i8, sc_x = quant_i8_sym(h)
            rsh = matmul_rshift(D) - 5; rsh = max(rsh, 8)
            q = matmul_i8(x_i8, mats["Wq"], rsh, sc_x, lsc[:D]) + bq
            k = matmul_i8(x_i8, mats["Wk"], rsh, sc_x, lsc[D:D + DKV]) + bk
            v = matmul_i8(x_i8, mats["Wv"], rsh, sc_x, lsc[D + DKV:D + 2 * DKV]) + bv
        else:
            q = h @ (mats["Wq"].astype(np.float32) * lsc[:D].reshape(1, D)) + bq
            k = h @ (mats["Wk"].astype(np.float32) * lsc[D:D + DKV].reshape(1, DKV)) + bk
            v = h @ (mats["Wv"].astype(np.float32) * lsc[D + DKV:D + 2 * DKV].reshape(1, DKV)) + bv
        # reshape to heads, rope
        q = q.reshape(seq, H, HD)
        k = k.reshape(seq, KVH, HD)
        v = v.reshape(seq, KVH, HD)
        for s in range(seq):
            for hh in range(H):
                q[s, hh] = rope(q[s, hh], s, cos, sin, HD)
            for hh in range(KVH):
                k[s, hh] = rope(k[s, hh], s, cos, sin, HD)
        # GQA attention
        attn = np.zeros((seq, H, HD), dtype=np.float32)
        for hh in range(H):
            kvh = hh // GROUPS
            qh = q[:, hh, :]                                # [seq, HD]
            kh = k[:, kvh, :]                               # [seq, HD]
            vh = v[:, kvh, :]
            sc = 1.0 / math.sqrt(HD)
            logits = (qh @ kh.T) * sc                       # [seq, seq]
            # causal mask
            mask = np.triu(np.ones((seq, seq)), 1).astype(bool)
            logits = np.where(mask, -1e30, logits)
            probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
            probs /= probs.sum(axis=-1, keepdims=True)
            attn[:, hh, :] = probs @ vh
        attn = attn.reshape(seq, D)
        if int8_act:
            a_i8, sc_a = quant_i8_sym(attn)
            rsh = matmul_rshift(D) - 5; rsh = max(rsh, 8)
            wo = matmul_i8(a_i8, mats["Wo"], rsh, sc_a, lsc[2 * DKV + D:2 * DKV + 2 * D])
        else:
            wo = attn @ (mats["Wo"].astype(np.float32) * lsc[2 * DKV + D:2 * DKV + 2 * D].reshape(1, D))
        x = x + wo
        # --- FFN ---
        h = rms_norm(x, rms_ffn)
        if int8_act:
            x_i8, sc_x = quant_i8_sym(h)
            rsh = matmul_rshift(D) - 5; rsh = max(rsh, 8)
            up = matmul_i8(x_i8, mats["up"], rsh, sc_x, lsc[LS["up"]:LS["up"] + F])
            gate = matmul_i8(x_i8, mats["gate"], rsh, sc_x, lsc[LS["gate"]:LS["gate"] + F])
            mid = up * silu(gate)
            # down: K-chunked
            KC = 1024
            m_i8, sc_m = quant_i8_sym(mid)
            rsh_d = matmul_rshift(KC) - 5; rsh_d = max(rsh_d, 8)
            out = np.zeros((seq, D), dtype=np.float32)
            for kc in range(0, F, KC):
                kcn = min(KC, F - kc)
                chunk_x = m_i8[:, kc:kc + kcn]
                wchunk = mats["down"][kc:kc + kcn, :]
                # need per-chunk input scale; approximate: use sc_m (engine uses per-chunk)
                res = matmul_i8(chunk_x, wchunk, rsh_d, sc_m, lsc[LS["down"]:LS["down"] + D])
                out += res
        else:
            up = h @ (mats["up"].astype(np.float32) * lsc[LS["up"]:LS["up"] + F].reshape(1, F))
            gate = h @ (mats["gate"].astype(np.float32) * lsc[LS["gate"]:LS["gate"] + F].reshape(1, F))
            mid = up * silu(gate)
            out = mid @ (mats["down"].astype(np.float32) * lsc[LS["down"]:LS["down"] + D].reshape(1, D))
        x = x + out

    # final rms + lm head (per-row embed)
    h = rms_norm(x, frms)
    # logits = h @ embed.T (per-row: embed row = int8 * esc[row])
    # For top-k, only need dot products; compute full for simplicity (V*D matmul)
    logits = h[-1] @ (embed.astype(np.float32) * esc.reshape(-1, 1)).T   # [V]
    return logits


def topk(logits, k=5):
    idx = np.argsort(-logits)[:k]
    return idx.tolist(), logits[idx].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--int8", action="store_true", help="full INT8 activation emulation")
    ap.add_argument("--prompts", nargs="*", default=[
        "中国的首都是",
        "The capital of France is",
        "今天天气很好，我们去公园",
    ])
    ap.add_argument("--out", default=os.path.join(HERE, "host_ref.json"))
    args = ap.parse_args()
    print("[emu] loading weights...", flush=True)
    embed, esc, frms, layers = load_weights()
    print("[emu] loaded.", flush=True)

    tok = None
    sys.path.insert(0, os.path.join(HERE, "..", "qwen_int4"))
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(os.path.join(HERE, "..", "qwen_int4", "model"),
                                            trust_remote_code=True)
    except Exception as e:
        print(f"[emu] tokenizer load failed: {e}")

    res = {"int8_act": args.int8, "prompts": {}}
    for p in args.prompts:
        ids = tok.encode(p, add_special_tokens=False) if tok else [0]
        logits = forward(ids, embed, esc, frms, layers, int8_act=args.int8)
        t5, v5 = topk(logits)
        gap = (v5[0] - v5[1]) if len(v5) > 1 else 0.0
        res["prompts"][p] = {
            "n_tokens": len(ids), "next": t5[0], "top5": t5,
            "gap": float(gap), "logits_top5": [float(x) for x in v5],
            "tokens": ids,
        }
        print(f"[emu] '{p}' ntok={len(ids)} next={t5[0]} gap={gap:.3f} top5={t5}", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"[emu] wrote {args.out}")


if __name__ == "__main__":
    main()
