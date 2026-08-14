#!/usr/bin/env python3
"""verify_kal_roundtrip.py — load K-aligned INT4 files, run two-pass, expect 3/3.

Byte-level round-trip check of convert_qwen_kal.py output: read layerN_kal.bin,
unpack nibbles to per-group INT4 (values -8..7) + fp16 gscale, feed the Path A
two-pass forward (emu_group_int4).  If the converter's layout is correct this
reproduces the 3/3 of emu_group_int4 (direct bf16->INT4).
"""
import os, sys, json, struct
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import emu_group_int4 as E4
from emu_perrow import D, KVH, DKV, L, F, G, PROMPTS, REF

D = 896
KVH = 2
DKV = KVH * 64
L = 24
F = 4864
G = 32

MATS = [("q_proj", D, D), ("k_proj", D, DKV), ("v_proj", D, DKV),
        ("o_proj", D, D), ("up_proj", D, F), ("gate_proj", D, F), ("down_proj", F, D)]


def unpack(nib, gsc, K, N):
    """nib (K/G)*N*16 bytes, gsc (K/G)*N*2 bytes -> (q [K,N] int8 -8..7, gscale [K/G,N])."""
    KG = K // G
    nb = np.frombuffer(nib, dtype=np.uint8).reshape(KG, N, G // 2)
    gs = np.frombuffer(gsc, dtype=np.float16).reshape(KG, N).astype(np.float32)
    lo = (nb & 0x0F).astype(np.int8).transpose(0, 2, 1)      # [KG, G/2, N]
    hi = ((nb >> 4) & 0x0F).astype(np.int8).transpose(0, 2, 1)
    q = np.zeros((KG, G, N), dtype=np.int8)
    q[:, 0::2, :] = lo
    q[:, 1::2, :] = hi
    # int4 [-8,7]; values 8..15 stored in low nibble map to -8..-1 (two's complement)
    q = np.where(q > 7, q - 16, q).astype(np.int8)
    return np.ascontiguousarray(q.reshape(K, N)), gs


def load_layers(wdir):
    layers = []
    for l in range(L):
        p = os.path.join(wdir, f"layer{l}_kal.bin")
        data = open(p, "rb").read()
        off = 0
        def take(n):
            nonlocal off
            b = data[off:off + n]; off += n
            return b
        rms_attn = np.frombuffer(take(D * 4), dtype=np.float32).copy()
        mats, gsc = {}, {}
        KEYMAP = {"q_proj": "Wq", "k_proj": "Wk", "v_proj": "Wv", "o_proj": "Wo",
                  "up_proj": "up", "gate_proj": "gate", "down_proj": "down"}
        for name, K, N in MATS:
            nib = take((K // G) * N * 16)
            gs = take((K // G) * N * 2)
            q, g = unpack(nib, gs, K, N)
            mats[KEYMAP[name]] = q
            gsc[KEYMAP[name]] = g
        rms_ffn = np.frombuffer(take(D * 4), dtype=np.float32).copy()
        assert off == len(data), f"layer{l}: {off} vs {len(data)}"
        bias = np.fromfile(os.path.join(wdir, f"layer{l}_bias.f32"), dtype=np.float32)
        layers.append((rms_attn, mats, rms_ffn, bias, gsc))
    return layers


def main():
    wdir = os.path.join(HERE, "weights_kal")
    print(f"[rt] loading from {wdir}")
    layers = load_layers(wdir)
    embed = np.fromfile(os.path.join(wdir, "embed_i8.bin"), dtype=np.int8).reshape(151936, D)
    esc = np.fromfile(os.path.join(wdir, "embed_scales.f32"), dtype=np.float32)
    frms = np.fromfile(os.path.join(wdir, "final_rms.f32"), dtype=np.float32)
    tok = None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(os.path.join(E4.PRE, "model"), trust_remote_code=True)
    except Exception as e:
        print("tok fail", e)
    passed = 0
    for p in PROMPTS:
        ids = tok.encode(p, add_special_tokens=False) if tok else [0]
        lg = E4.forward(ids, embed, esc, frms, layers)
        t5 = np.argsort(-lg)[:5].tolist()
        v5 = lg[t5].tolist()
        gap = v5[0] - v5[1] if len(v5) > 1 else 0.0
        ok = t5[0] == REF[p]
        passed += ok
        print(f"[rt] '{p}' next={t5[0]} ref={REF[p]} {'OK' if ok else 'MISMATCH'} gap={gap:.3f}", flush=True)
    print(f"[rt] ============ VERDICT {passed}/3 ============", flush=True)


if __name__ == "__main__":
    main()
