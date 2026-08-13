#!/usr/bin/env python3
"""emu_wsdq.py — CEO Phase-4 一票否决质量测试：weight-side dequant（A'）.

Route under test (A'): K-aligned INT4 G32 (weights_kal/) stored on SD
  -> CPU unpack * per-group scale  -> per-channel INT8 compute weights
  -> ordinary INT8 matmul (exact / two-pass rshift) + per-row INT8 act
  + per-channel fp32 post-processing.  TPU zero new ops.

This folds INT4 dequant into per-channel INT8 — the exact risk zone of the
P0 FAIL ("5 per-channel scale heuristics all 0/3").  We isolate two axes:

  * mode "exact"      : exact int32 accumulation (NO int8 output rounding)
                        -> isolates weight-folding + per-row-act error only.
  * mode "twopass128" : faithful two-pass rshift KG=128 (the proven
                        engine-realistic ps32-free path) -> full pipeline.

Weight sources (layout axis) + lsc axis:
  kal/nat   : K-aligned INT4 G32, per-ch scale = max|W_dq[:,n]|/127 (natural fold)
  kal/bf16  : K-aligned INT4 G32, per-ch scale = stored layer_scales.bin (bf16)
  flat/bf16 : flat INT4 G32 (existing weights/), scale = stored layer_scales.bin
  flat/nat  : flat INT4 G32, per-ch scale = max|W_dq[:,n]|/127

Pass criterion: next_token == host-INT4 reference (2130 / 12095 / 99366)
for all 3 fixed prompts, same as int4_g32_bf16 (p0_refs).
"""
import os, sys, json, math, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import emu_chunk8 as C8
import emu_perrow as EP
from emu_perrow import (D, H, KVH, HD, L, F, V, DKV, GROUPS, PROMPTS, REF,
                        matmul_rshift, int8_round_div, precompute_rope,
                        rope, rms_norm, silu, LS, LS_PL)
from qwen_emu import ROPE_THETA, EPS
import qwen_emu as QE

PRE = os.path.join(HERE, "..", "qwen_int4")
G = 32
KAL = os.path.join(HERE, "weights_kal")
WFLAT = os.path.join(HERE, "weights")

MATS = [("q_proj", D, D), ("k_proj", D, DKV), ("v_proj", D, DKV),
        ("o_proj", D, D), ("up_proj", D, F), ("gate_proj", D, F), ("down_proj", F, D)]
KEYMAP = {"q_proj": "Wq", "k_proj": "Wk", "v_proj": "Wv", "o_proj": "Wo",
          "up_proj": "up", "gate_proj": "gate", "down_proj": "down"}
LS_ORDER = ["Wq", "Wk", "Wv", "Wo", "up", "gate", "down"]


# ----------------------------------------------------------------------
#  INT4 -> fp32 dequant  (weight-side dequant, host-exact)
# ----------------------------------------------------------------------
def unpack_kal(nib, gs, K, N):
    """K-aligned INT4: nib (K/G)*N*16 B, gs (K/G)*N*2 B (fp16)
    -> (q [K,N] int8 -8..7, gscale [K/G,N] fp32)."""
    KG = K // G
    nb = np.frombuffer(nib, dtype=np.uint8).reshape(KG, N, G // 2)
    gsc = np.frombuffer(gs, dtype=np.float16).reshape(KG, N).astype(np.float32)
    lo = (nb & 0x0F).astype(np.int8).transpose(0, 2, 1)
    hi = ((nb >> 4) & 0x0F).astype(np.int8).transpose(0, 2, 1)
    q = np.zeros((KG, G, N), dtype=np.int8)
    q[:, 0::2, :] = lo
    q[:, 1::2, :] = hi
    q = np.where(q > 7, q - 16, q).astype(np.int8)
    return np.ascontiguousarray(q.reshape(K, N)), gsc


def unpack_flat(nib, gs, K, N):
    """Flat INT4 (q4.py group_quantize): nib n/2 B, gs (n/G)*2 B (fp16)
    -> (q [K,N] int8 -8..7, gscale [K,N] fp32 broadcast to elements)."""
    n = K * N
    lo = (nib & 0x0F).astype(np.int8)
    hi = ((nib >> 4) & 0x0F).astype(np.int8)
    lo = np.where(lo >= 8, lo - 16, lo)
    hi = np.where(hi >= 8, hi - 16, hi)
    q = np.empty(n, dtype=np.int8)
    q[0::2] = lo
    q[1::2] = hi
    gsc = np.frombuffer(gs, dtype=np.float16).astype(np.float32)  # (n/G,)
    qf = q.astype(np.float32).reshape(n // G, G)                  # (n/G, G)
    gs_e = gsc.reshape(-1, 1)                                     # per flat group
    W_dq = (qf * gs_e).reshape(K, N)
    return W_dq


def load_Wdq(layout, wdir):
    """Return list per layer of dict name->W_dq [K,N] fp32 (INT4 dequant),
    plus rms_attn, rms_ffn, bias."""
    layers = []
    for l in range(L):
        if layout == "kal":
            path = os.path.join(wdir, f"layer{l}_kal.bin")
            data = open(path, "rb").read()
            off = 0
            def take(n):
                nonlocal off
                b = data[off:off + n]; off += n
                return b
            rms_attn = np.frombuffer(take(D * 4), dtype=np.float32).copy()
            mats = {}
            for name, K, N in MATS:
                nib = take((K // G) * N * 16)
                gs = take((K // G) * N * 2)
                q, gscale = unpack_kal(nib, gs, K, N)
                W_dq = q.astype(np.float32) * gscale.repeat(G, axis=0)
                mats[KEYMAP[name]] = W_dq
            rms_ffn = np.frombuffer(take(D * 4), dtype=np.float32).copy()
            assert off == len(data), f"layer{l} trailing {len(data)-off}"
            bias = np.fromfile(os.path.join(wdir, f"layer{l}_bias.f32"), dtype=np.float32)
        else:  # flat
            path = os.path.join(wdir, f"layer{l}_i4.bin")
            data = open(path, "rb").read()
            off = 0
            def take(n):
                nonlocal off
                b = data[off:off + n]; off += n
                return b
            rms_attn = np.frombuffer(take(D * 4), dtype=np.float32).copy()
            mats = {}
            for name, K, N in MATS:
                n = K * N
                nib = np.frombuffer(take(n // 2), dtype=np.uint8)
                gs = np.frombuffer(take((n // G) * 2), dtype=np.uint8)
                W_dq = unpack_flat(nib, gs, K, N)
                mats[KEYMAP[name]] = W_dq
            rms_ffn = np.frombuffer(take(D * 4), dtype=np.float32).copy()
            assert off == len(data), f"layer{l} trailing {len(data)-off}"
            bias = np.fromfile(os.path.join(wdir, f"layer{l}_bias.f32"), dtype=np.float32)
        layers.append((rms_attn, mats, rms_ffn, bias))
    return layers


def build_perch(layers, lsc_mode):
    """layers with W_dq -> (rms_attn, mats_int8, rms_ffn, lsc, bias) in emu_perrow order."""
    if lsc_mode == "bf16":
        # stored per-channel scales from layer_scales.bin (bf16-derived), engine format
        _, _, _, qe_layers = QE.load_weights()
    out = []
    for l, (rms_attn, mats, rms_ffn, bias) in enumerate(layers):
        mats8, lsc = {}, []
        for key in LS_ORDER:
            W_dq = mats[key]
            if lsc_mode == "nat":
                sc = np.maximum(np.max(np.abs(W_dq), axis=0) / 127.0, 1e-12)
            else:
                sc = qe_layers[l][3][LS[key]:LS[key] + W_dq.shape[1]].astype(np.float32)
            mats8[key] = np.clip(np.round(W_dq / sc), -128, 127).astype(np.int8)
            lsc.append(sc)
        out.append((rms_attn, mats8, rms_ffn, np.concatenate(lsc), bias))
    return out


# ----------------------------------------------------------------------
#  matmul: exact int32 accumulation OR faithful two-pass rshift
# ----------------------------------------------------------------------
def matmul_wsdq(x_i8, w_i8, sc_row, lsc_col, mode, KG=128):
    M, K = x_i8.shape
    N = w_i8.shape[1]
    x = x_i8.astype(np.int32)
    w = w_i8.astype(np.int32)
    if mode == "exact":
        out = (x @ w).astype(np.float64)
    else:  # twopass
        out = np.zeros((M, N), dtype=np.float64)
        for g in range(0, K, KG):
            kc = min(KG, K - g)
            acc = x[:, g:g + kc] @ w[g:g + kc, :]
            rsafe = max(matmul_rshift(kc) - 5, 8)
            p1 = int8_round_div(acc, rsafe)
            est = int(np.max(np.abs(p1))) * (1 << rsafe)
            r = 0
            if est > 1e-6:
                r = int(math.ceil(math.log2(est / 127.0)))
                r = max(r, 0)
            p2 = int8_round_div(acc, r).astype(np.float64) * (1 << r)
            out += p2
    out = out.astype(np.float32)
    return out * sc_row[:, None] * lsc_col.reshape(1, -1)


def forward_wsdq(tokens, embed, esc, frms, layers, mode, max_pos=None):
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
        q = matmul_wsdq(x_i8, mats["Wq"], sc_row, lsc[:D], mode) + bq
        k = matmul_wsdq(x_i8, mats["Wk"], sc_row, lsc[D:D + DKV], mode) + bk
        v = matmul_wsdq(x_i8, mats["Wv"], sc_row, lsc[D + DKV:D + 2 * DKV], mode) + bv
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
        wo = matmul_wsdq(a_i8, mats["Wo"], sc_a, lsc[2 * DKV + D:2 * DKV + 2 * D], mode)
        x = x + wo
        h = rms_norm(x, rms_ffn)
        x_i8, sc_row = C8.quant_per_row(h)
        up = matmul_wsdq(x_i8, mats["up"], sc_row, lsc[LS["up"]:LS["up"] + F], mode)
        gate = matmul_wsdq(x_i8, mats["gate"], sc_row, lsc[LS["gate"]:LS["gate"] + F], mode)
        mid = up * silu(gate)
        out = np.zeros((seq, D), dtype=np.float32)
        for kc in range(0, F, 1024):
            kcn = min(1024, F - kc)
            m_i8, sc_m = C8.quant_per_row(mid[:, kc:kc + kcn])
            wc = mats["down"][kc:kc + kcn, :]
            out += matmul_wsdq(m_i8, wc, sc_m, lsc[LS["down"]:LS["down"] + D], mode)
        x = x + out

    h = rms_norm(x, frms)
    logits = h[-1] @ (embed.astype(np.float32) * esc.reshape(-1, 1)).T
    return logits


# ----------------------------------------------------------------------
def run_config(layers, embed, esc, frms, tag, mode, tok, out):
    passed = 0
    row = {}
    for p in PROMPTS:
        ids = tok.encode(p, add_special_tokens=False) if tok else [0]
        logits = forward_wsdq(ids, embed, esc, frms, layers, mode)
        t5 = np.argsort(-logits)[:5].tolist()
        v5 = logits[t5].tolist()
        gap = v5[0] - v5[1] if len(v5) > 1 else 0.0
        ok = t5[0] == REF[p]
        passed += ok
        row[p] = {"next": t5[0], "ref": REF[p], "ok": bool(ok), "gap": float(gap), "top5": t5}
        print(f"[{tag}/{mode}] '{p}' next={t5[0]} ref={REF[p]} "
              f"{'OK' if ok else 'MISMATCH'} gap={gap:.3f}", flush=True)
    print(f"[{tag}/{mode}] ============ VERDICT {passed}/3 ============", flush=True)
    out[tag][mode] = {"verdict": f"{passed}/3", "prompts": row}
    return passed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "wsdq_ref.json"))
    args = ap.parse_args()

    tok = None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(os.path.join(PRE, "model"), trust_remote_code=True)
    except Exception as e:
        print("tokenizer fail:", e)

    # reference embed / final_rms (on-board INT8 embed is part of the pipeline)
    embed = np.fromfile(os.path.join(KAL, "embed_i8.bin"), dtype=np.int8).reshape(V, D)
    esc = np.fromfile(os.path.join(KAL, "embed_scales.f32"), dtype=np.float32)
    frms = np.fromfile(os.path.join(KAL, "final_rms.f32"), dtype=np.float32)

    out = {}

    # (tag, layout, wdir, lsc_mode) — build per variant once, run both modes
    variants = [
        ("kal_nat", "kal", KAL, "nat"),
        ("kal_bf16lsc", "kal", KAL, "bf16"),
        ("flat_nat", "flat", WFLAT, "nat"),
        ("flat_bf16lsc", "flat", WFLAT, "bf16"),
    ]
    for tag, layout, wdir, lsc_mode in variants:
        print(f"[wsdq] building {tag} (layout={layout}, lsc={lsc_mode})...", flush=True)
        layers = build_perch(load_Wdq(layout, wdir), lsc_mode)
        out[tag] = {}
        run_config(layers, embed, esc, frms, tag, "exact", tok, out)
        run_config(layers, embed, esc, frms, tag, "twopass128", tok, out)

    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[wsdq] wrote {args.out}")


if __name__ == "__main__":
    main()
