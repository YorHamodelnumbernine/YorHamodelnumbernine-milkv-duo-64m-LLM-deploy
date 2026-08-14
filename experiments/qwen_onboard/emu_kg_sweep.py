#!/usr/bin/env python3
"""emu_kg_sweep.py — CEO KG 扫参: Path A per-group 缩放权重在 KG=64/128 是否仍 3/3.

Path A 组结构锁 KG=G=32（group_int4.json 3/3）。CEO 裁定: 若 per-group 缩放权重在
KG=64/128 仍 3/3, 则 K-block 数降 ~2x/~4x -> TIU 提交降 ~2x/~4x; 不成立保持 32.

语义（诚实部署口径）: 对每个目标组大小 G, 直接从 fp32 权重按 group=G 重新量化
(q4 网格 + per-[K/G,N] scale), 两遍法 matmul 块 KG=G。即「以 KG=G 部署 Path A
的完整质量」, 与 group_int4.json (G=32) 同一条验证链。

用法:
  python3 emu_kg_sweep.py                 # KG in {32,64,128}
  python3 emu_kg_sweep.py --kgs 64 128    # 只跑 64/128
  python3 emu_kg_sweep.py --out kg_sweep.json
"""
import os, sys, json, math, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import emu_chunk8 as C8
from emu_perrow import (D, H, KVH, HD, L, F, V, DKV, GROUPS, PROMPTS, REF,
                        matmul_rshift, int8_round_div, precompute_rope,
                        rope, rms_norm, silu)
from qwen_emu import ROPE_THETA

PRE = os.path.join(HERE, "..", "qwen_int4")
sys.path.insert(0, PRE)
from sf_io import SF
import qwen_emu as QE

MATS = [("q_proj", "self_attn", D, D), ("k_proj", "self_attn", D, DKV),
        ("v_proj", "self_attn", D, DKV), ("o_proj", "self_attn", D, D),
        ("up_proj", "mlp", D, F), ("gate_proj", "mlp", D, F),
        ("down_proj", "mlp", F, D)]
KEYMAP = {"q_proj": "Wq", "k_proj": "Wk", "v_proj": "Wv", "o_proj": "Wo",
          "up_proj": "up", "gate_proj": "gate", "down_proj": "down"}


def quant_groups_i4(W_hf, G):
    """[N,K] -> (q4 int8 -8..7, gscale [K/G,N]) K-aligned INT4, group=G."""
    Wt = np.ascontiguousarray(W_hf.T.astype(np.float32))   # [K,N]
    K, N = Wt.shape
    q = np.zeros((K, N), dtype=np.int8)
    gs = np.zeros((K // G, N), dtype=np.float32)
    for g in range(0, K, G):
        seg = Wt[g:g + G, :]
        mx = np.max(np.abs(seg), axis=0)
        sc = np.maximum(mx / 7.0, 1e-12)
        q[g:g + G, :] = np.clip(np.round(seg / sc), -8, 7).astype(np.int8)
        gs[g // G, :] = sc
    return q, gs


def matmul_rshift_w(K, wmax):
    r = 0
    md = K * 127 * wmax
    while (md >> r) > 127:
        r += 1
    return r


def matmul_group_i4(x_i8, w_i8, gscale, sc_row, KG, mode="twopass"):
    """Per-group INT4 matmul. KG == quant group G (Path A lock).
    mode="twopass" = two-pass rshift (engine-realistic); mode="exact" = int64
    exact accumulation (isolates weight-quantization error from int8 rounding)."""
    M, K = x_i8.shape
    N = w_i8.shape[1]
    wmax = int(np.max(np.abs(w_i8)))
    out = np.zeros((M, N), dtype=np.float64)
    x = x_i8.astype(np.int32)
    w = w_i8.astype(np.int32)
    for g in range(0, K, KG):
        kc = min(KG, K - g)
        acc = x[:, g:g + kc] @ w[g:g + kc, :]
        if mode == "exact":
            out += acc.astype(np.float64) * gscale[g // KG, :].reshape(1, -1)
            continue
        rsafe = max(matmul_rshift_w(kc, wmax) - 3, 4)
        p1 = int8_round_div(acc, rsafe)
        est = int(np.max(np.abs(p1))) * (1 << rsafe)
        r = 0
        if est > 1e-6:
            r = int(math.ceil(math.log2(est / 127.0)))
            r = max(r, 0)
        p2 = int8_round_div(acc, r).astype(np.float64) * (1 << r)
        out += p2 * gscale[g // KG, :].reshape(1, -1)
    return out.astype(np.float32) * sc_row[:, None]


def load_variants(Gs):
    """Load safetensors once; return (embed, esc, frms, layers) where each layer
    is (rms_attn, {G: {key: (q4, gscale)}}, rms_ffn, bias)."""
    sf = SF(os.path.join(PRE, "model", "model.safetensors"))
    embed, esc, frms, _ = QE.load_weights()
    layers = []
    for l in range(L):
        p = f"model.layers.{l}"
        mats = {G: {} for G in Gs}
        for name, grp, K, N in MATS:
            w = sf.get(f"{p}.{grp}.{name}.weight")
            key = KEYMAP[name]
            for G in Gs:
                q, gs = quant_groups_i4(w, G)
                mats[G][key] = (q, gs)
        bias = np.concatenate([sf.get(f"{p}.self_attn.{n}.bias").astype(np.float32)
                               for n in ("q_proj", "k_proj", "v_proj")])
        layers.append((sf.get(f"{p}.input_layernorm.weight").astype(np.float32), mats,
                       sf.get(f"{p}.post_attention_layernorm.weight").astype(np.float32), bias))
    return embed, esc, frms, layers


def forward(ids, embed, esc, frms, layers, KG, mode="twopass"):
    seq = len(ids)
    cos, sin = precompute_rope(max(64, seq + 8), HD, ROPE_THETA)
    x = np.zeros((seq, D), np.float32)
    for i, t in enumerate(ids):
        t = 0 if (t < 0 or t >= V) else t
        x[i] = embed[t].astype(np.float32) * esc[t]
    for rms_attn, mats, rms_ffn, bias in layers:
        m = mats[KG]
        bq, bk, bv = bias[:D], bias[D:D + DKV], bias[D + DKV:]
        h = rms_norm(x, rms_attn)
        x_i8, sc_row = C8.quant_per_row(h)
        q = matmul_group_i4(x_i8, m["Wq"][0], m["Wq"][1], sc_row, KG, mode) + bq
        k = matmul_group_i4(x_i8, m["Wk"][0], m["Wk"][1], sc_row, KG, mode) + bk
        v = matmul_group_i4(x_i8, m["Wv"][0], m["Wv"][1], sc_row, KG, mode) + bv
        q = q.reshape(seq, H, HD); k = k.reshape(seq, KVH, HD); v = v.reshape(seq, KVH, HD)
        for s in range(seq):
            for hh in range(H):
                q[s, hh] = rope(q[s, hh], s, cos, sin, HD)
            for hh in range(KVH):
                k[s, hh] = rope(k[s, hh], s, cos, sin, HD)
        attn = np.zeros((seq, H, HD), np.float32)
        for hh in range(H):
            kvh = hh // GROUPS
            qh = q[:, hh, :]; kh = k[:, kvh, :]; vh = v[:, kvh, :]
            lg = (qh @ kh.T) * (1.0 / math.sqrt(HD))
            mask = np.triu(np.ones((seq, seq)), 1).astype(bool)
            lg = np.where(mask, -1e30, lg)
            pr = np.exp(lg - lg.max(-1, keepdims=True))
            pr /= pr.sum(-1, keepdims=True)
            attn[:, hh, :] = pr @ vh
        attn = attn.reshape(seq, D)
        a_i8, sc_a = C8.quant_per_row(attn)
        x = x + matmul_group_i4(a_i8, m["Wo"][0], m["Wo"][1], sc_a, KG, mode)
        h = rms_norm(x, rms_ffn)
        x_i8, sc_row = C8.quant_per_row(h)
        up = matmul_group_i4(x_i8, m["up"][0], m["up"][1], sc_row, KG, mode)
        gate = matmul_group_i4(x_i8, m["gate"][0], m["gate"][1], sc_row, KG, mode)
        mid = up * silu(gate)
        out = np.zeros((seq, D), np.float32)
        for kc in range(0, F, 1024):
            kcn = min(1024, F - kc)
            m_i8, sc_m = C8.quant_per_row(mid[:, kc:kc + kcn])
            wc = m["down"][0][kc:kc + kcn, :]
            gsw = m["down"][1][kc // KG:(kc + kcn) // KG, :]
            out += matmul_group_i4(m_i8, wc, gsw, sc_m, KG, mode)
        x = x + out
    h = rms_norm(x, frms)
    return h[-1] @ (embed.astype(np.float32) * esc.reshape(-1, 1)).T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kgs", nargs="+", type=int, default=[32, 64, 128])
    ap.add_argument("--out", default=os.path.join(HERE, "kg_sweep.json"))
    ap.add_argument("--mode", default="twopass", choices=["twopass", "exact"],
                    help="twopass=engine-realistic two-pass rshift; exact=int64 accumulation")
    args = ap.parse_args()

    tok = None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(os.path.join(PRE, "model"), trust_remote_code=True)
    except Exception as e:
        print("tok fail", e)

    print(f"[kg_sweep] loading safetensors + quantizing at G={args.kgs} ...", flush=True)
    embed, esc, frms, layers = load_variants(args.kgs)
    print("[kg_sweep] loaded.", flush=True)

    result = {"note": "Path A per-group scale sweep, KG=G (group-structure lock); "
                      "requant from fp32 at each group size",
              "kgs": args.kgs, "mode": args.mode}
    for G in args.kgs:
        passed = 0
        row = {}
        for p in PROMPTS:
            ids = tok.encode(p, add_special_tokens=False) if tok else [0]
            lg = forward(ids, embed, esc, frms, layers, G, args.mode)
            t5 = np.argsort(-lg)[:5].tolist()
            v5 = lg[t5].tolist()
            gap = v5[0] - v5[1] if len(v5) > 1 else 0.0
            ok = t5[0] == REF[p]
            passed += ok
            row[p] = {"next": t5[0], "ref": REF[p], "ok": bool(ok),
                      "gap": float(gap), "top5": t5}
            print(f"[kg_sweep/KG={G}] '{p}' next={t5[0]} ref={REF[p]} "
                  f"{'OK' if ok else 'MISMATCH'} gap={gap:.3f}", flush=True)
        print(f"[kg_sweep/KG={G}] ============ VERDICT {passed}/3 ============", flush=True)
        result[G] = {"verdict": f"{passed}/3", "prompts": row}

    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print(f"[kg_sweep] wrote {args.out}")


if __name__ == "__main__":
    main()
