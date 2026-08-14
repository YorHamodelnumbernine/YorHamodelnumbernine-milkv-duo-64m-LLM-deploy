#!/usr/bin/env python3
"""convert_qwen_kal.py — Qwen2.5-0.5B → K-aligned INT4 G32 on-board weights.

Phase 5 / CEO-adopted Path A: per-chunk (KG=32=G) two-pass matmul needs the INT4
group scale to be a K-slice (same output column), NOT a flat N-segment.  This
converter writes the K-aligned layout so the engine can, per K-chunk g:
  - read 16 packed bytes per output column (G=32 int4 values, K-aligned)
  - CPU-dequant to per-group INT8 (value * gscale[g,n])
  - two-pass matmul, fp32 accumulate with gscale[g,n] * 2^r * sc_row[m]

On-board file formats (K-aligned):
  config.bin            same as before
  embed_i8.bin/scales   same as before
  layerN_kal.bin        per layer, engine matmul order:
                          [rms_attn f32: D*4]
                          Wq: [nib (D/G)*D*16][gsc fp16 (D/G)*D]
                          Wk: [nib (D/G)*DKV*16][gsc fp16 (D/G)*DKV]
                          Wv: [nib (D/G)*DKV*16][gsc fp16 (D/G)*DKV]
                          Wo: [nib (D/G)*D*16][gsc fp16 (D/G)*D]
                          [rms_ffn f32: D*4]
                          up:   [nib (D/G)*F*16][gsc fp16 (D/G)*F]
                          gate: [nib (D/G)*F*16][gsc fp16 (D/G)*F]
                          down: [nib (F/G)*D*16][gsc fp16 (F/G)*D]
                        nib layout: per K-block g, [N][16] packed int4 along K.
                        gsc layout: per K-block g, [N] fp16.
  layerN_bias.f32       same as before
  layer_scales.bin      per-channel fp32 of INT4-dequant (reference only;
                        the two-pass accumulate uses gscale, not lsc)
  final_rms.f32         same
"""
import os, sys, struct, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PRE = os.path.join(HERE, "..", "qwen_int4")
sys.path.insert(0, PRE)
from sf_io import SF                      # noqa: E402

D = 896
H = 14
KVH = 2
HEAD_DIM = 64
L = 24
F = 4864
V = 151936
G = 32
DKV = KVH * HEAD_DIM
MAX_SEQ = 64
SYM_QMAX = 7

MATS = [  # name, [K, N] (engine layout)
    ("q_proj", D, D),
    ("k_proj", D, DKV),
    ("v_proj", D, DKV),
    ("o_proj", D, D),
    ("up_proj", D, F),
    ("gate_proj", D, F),
    ("down_proj", F, D),
]
SCALE_ORDER = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"]


def pack_kaligned(W_hf):
    """W_hf [N,K] bf16->f32 -> (nib_bytes, gsc_bytes, dq_per_ch_scale).
    K-aligned INT4 G=32 symmetric RTN.  Returns:
      nib_bytes: (K/G)*N*16 bytes, nib[g,n,j] = (q[g,2j,n] | q[g,2j+1,n]<<4)
      gsc_bytes: (K/G)*N*2 bytes fp16, gscale[g,n]
      sc: per-output-channel fp32 of dequant [N]
    """
    Wt = np.ascontiguousarray(W_hf.T.astype(np.float32))     # [K, N]
    K, N = Wt.shape
    assert K % G == 0 and N % 2 == 0
    grp = Wt.reshape(K // G, G, N)                            # [KG, G, N]
    mx = np.max(np.abs(grp), axis=1)                          # [KG, N]
    gsc = np.maximum(mx / SYM_QMAX, 1e-12)                    # [KG, N]
    q = np.clip(np.round(grp / gsc[:, None, :]), -8, 7).astype(np.int8)  # [KG, G, N]
    dq = q.astype(np.float32) * gsc[:, None, :]
    dq2 = dq.reshape(K, N)
    lsc = np.maximum(np.max(np.abs(dq2), axis=0) / 127.0, 1e-12)  # [N] per-ch ref

    # pack: [KG, G, N] -> [KG, N, G/2] bytes (pair along K)
    qu = (q & 0x0F).astype(np.uint8)                          # [KG, G, N]
    even = qu[:, 0::2, :]                                     # [KG, G/2, N]
    odd = qu[:, 1::2, :]
    nib = np.ascontiguousarray((even | (odd << 4)).transpose(0, 2, 1))  # [KG, N, G/2]
    gscf = np.ascontiguousarray(gsc.astype(np.float16))       # [KG, N]
    return nib.tobytes(), gscf.tobytes(), lsc, q              # q: [KG,G,N] int8


def matmul_rshift_w(wmax):
    md = 32 * 127 * wmax          # K=G=32
    r = 0
    while (md >> r) > 127:
        r += 1
    return r


def rsafe_from_q(q):
    """引擎语义: wmax = max|signed INT4|, rsafe = rshift_w(G,wmax)-3 (floor 4)."""
    wmax = int(np.abs(q).max())
    r = matmul_rshift_w(wmax) - 3
    return r if r >= 4 else 4


def quant_embed(E):
    mx = np.max(np.abs(E), axis=1, keepdims=True)
    s = np.maximum(mx / 127.0, 1e-12)
    q = np.clip(np.round(E / s), -128, 127).astype(np.int8)
    return q, s.ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(PRE, "model"))
    ap.add_argument("--out", default=os.path.join(HERE, "weights_kal"))
    ap.add_argument("--max-seq", type=int, default=MAX_SEQ)
    ap.add_argument("--no-embed", action="store_true")
    args = ap.parse_args()
    out = args.out
    os.makedirs(out, exist_ok=True)

    sf = SF(os.path.join(args.model, "model.safetensors"))
    print(f"[conv-kal] keys={len(sf.keys())}")

    cfg = struct.pack("iiiiiiii", D, H, KVH, HEAD_DIM, L, F, V, args.max_seq)
    open(os.path.join(out, "config.bin"), "wb").write(cfg)

    if not args.no_embed:
        E = sf.get("model.embed_tokens.weight")
        Ei8, esc = quant_embed(E)
        Ei8.tofile(os.path.join(out, "embed_i8.bin"))
        esc.astype(np.float32).tofile(os.path.join(out, "embed_scales.f32"))
        print(f"[conv-kal] embed {Ei8.nbytes/1e6:.1f} MB")

    all_lsc = []
    tot = 0
    rsh_tbl = []          # Phase 7c: 离线 rsafe 预标定 (查表跳过运行时 wmax 预扫)
    DCHUNK = 1024         # 引擎 down K-chunk (kc += 1024)
    for l in range(L):
        p = f"model.layers.{l}"
        rms_attn = sf.get(f"{p}.input_layernorm.weight").astype(np.float32)
        rms_ffn = sf.get(f"{p}.post_attention_layernorm.weight").astype(np.float32)
        segs, lsc, rsh_row = {}, {}, []
        for name, K, N in MATS:
            w = sf.get(f"{p}.self_attn.{name}.weight" if name in
                       ("q_proj", "k_proj", "v_proj", "o_proj") else f"{p}.mlp.{name}.weight")
            nib_b, gs_b, sc, q = pack_kaligned(w)
            segs[name] = (nib_b, gs_b)
            lsc[name] = sc
            if name == "down_proj":
                # down K-chunk 与引擎一致: 每 1024 的 K 块 (KG 块宽 32)
                kpg = DCHUNK // G
                for c in range((K + DCHUNK - 1) // DCHUNK):
                    rsh_row.append(rsafe_from_q(q[c * kpg:(c + 1) * kpg]))
            else:
                rsh_row.append(rsafe_from_q(q))
        rsh_tbl.append(bytes(rsh_row))
        all_lsc.append(np.concatenate([lsc[n] for n in SCALE_ORDER]))

        bq = sf.get(f"{p}.self_attn.q_proj.bias").astype(np.float32)
        bk = sf.get(f"{p}.self_attn.k_proj.bias").astype(np.float32)
        bv = sf.get(f"{p}.self_attn.v_proj.bias").astype(np.float32)
        np.concatenate([bq, bk, bv]).tofile(os.path.join(out, f"layer{l}_bias.f32"))

        fname = os.path.join(out, f"layer{l}_kal.bin")
        with open(fname, "wb") as f:
            f.write(rms_attn.tobytes())
            for name, _, _ in MATS:
                f.write(segs[name][0])
                f.write(segs[name][1])
            f.write(rms_ffn.tobytes())
        sz = os.path.getsize(fname)
        tot += sz
        print(f"[conv-kal] layer{l}_kal.bin {sz/1e6:.2f} MB")

    np.concatenate(all_lsc).tofile(os.path.join(out, "layer_scales.bin"))
    # Phase 7c: 离线 rsafe 预标定表 (每层 11 字节: q,k,v,wo,up,gate + down 5 chunk)
    rsafe_bin = b"".join(rsh_tbl)
    open(os.path.join(out, "rsafe.bin"), "wb").write(rsafe_bin)
    print(f"[conv-kal] rsafe.bin {len(rsafe_bin)} B (24 layers x 11) distinct={sorted(set(rsafe_bin))}")
    sf.get("model.norm.weight").astype(np.float32).tofile(os.path.join(out, "final_rms.f32"))
    # legacy placeholder
    arr = np.zeros(212, dtype=np.float32)
    arr[:1 + L * len(MATS)] = 0.01
    arr.tofile(os.path.join(out, "scales.bin"))

    # verify layer0 size
    K = D; N = D
    exp0 = (D * 4
            + (K // G) * D * (16 + 2)
            + 2 * (D // G) * DKV * (16 + 2)
            + (D // G) * D * (16 + 2)
            + D * 4
            + 2 * (D // G) * F * (16 + 2)
            + (F // G) * D * (16 + 2))
    l0 = os.path.getsize(os.path.join(out, "layer0_kal.bin"))
    print(f"[conv-kal] layer0 {l0} (expected {exp0}) {'OK' if l0 == exp0 else 'MISMATCH'}")
    print(f"[conv-kal] total {tot/1e6:.1f} MB -> {out}/")


if __name__ == "__main__":
    main()
