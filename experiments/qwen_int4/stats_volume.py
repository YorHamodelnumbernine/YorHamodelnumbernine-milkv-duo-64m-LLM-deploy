#!/usr/bin/env python3
"""Weight statistics + INT4 volume table for Qwen2.5-0.5B (host, numpy only)."""
import os, sys, json
import numpy as np
from sf_io import SF
from q4 import group_quantize

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "model")
SF_PATH = os.path.join(MODEL, "model.safetensors")
cfg = json.load(open(os.path.join(MODEL, "config.json")))

f = SF(SF_PATH)
names = f.keys()
lin = [n for n in names if n.endswith(".weight") and
       ("self_attn" in n or "mlp" in n or "embed_tokens" in n)]
print(f"total tensors={len(names)}  linear+embed={len(lin)}")

tot = 0
shape_map = {}
for n in lin:
    shape = f.header[n]["shape"]
    shape_map[n] = shape
    tot += int(np.prod(shape))
embed_n = int(np.prod(shape_map["model.embed_tokens.weight"]))
lin_elems = tot - embed_n
bf16_mb = tot * 2 / 1e6
lin_bf16_mb = lin_elems * 2 / 1e6
print(f"total elems={tot/1e6:.1f}M (embed {embed_n/1e6:.1f}M, linear {lin_elems/1e6:.1f}M)")
print(f"bf16 total = {bf16_mb:.1f} MB (embed {embed_n*2/1e6:.1f} MB, linear {lin_bf16_mb:.1f} MB)")

print("\n=== INT4 volume (attention+MLP linear only; embed kept bf16) ===")
for G in (32, 64, 128):
    nib_b = lin_elems / 2
    sc_b = lin_elems / G * 2
    lin_b = nib_b + sc_b
    full_b = lin_b + embed_n * 2
    print(f"G={G:4d}: nib={nib_b/1e6:7.2f}MB + sc(fp16)={sc_b/1e6:6.2f}MB = "
          f"linear {lin_b/1e6:7.2f}MB | +embed-bf16 -> {full_b/1e6:7.2f}MB "
          f"({full_b/(tot*2):.1%} of bf16) | linear-only {lin_b/(lin_elems*2):.1%}")

print("\n=== INT4 volume (everything quantized incl. embed, per-row int4) ===")
for G in (32, 64, 128):
    nib_b = tot / 2
    sc_b = tot / G * 2
    full_b = nib_b + sc_b
    print(f"G={G:4d}: {full_b/1e6:7.2f}MB  ({full_b/(tot*2):.1%} of bf16)")

print("\n=== per-matrix weight stats + INT4 RTN recon error (layers 0/12/23) ===")
print(f"{'matrix':<28}{'elems':>9}{'max|w|':>9}{'std':>9}{'G':>5}{'rms_err':>9}{'rel_rms%':>9}")
for li in (0, 12, 23):
    mats = sorted([n for n in lin if f"layers.{li}." in n])
    for n in mats:
        W = f.get(n)
        mx = float(np.abs(W).max())
        sd = float(W.std())
        for G in (32, 128):
            _, _, _, q, dq = group_quantize(W, G, asym=False)
            err = W - dq
            rms_e = float(np.sqrt((err ** 2).mean()))
            rel = rms_e / float(np.sqrt((W ** 2).mean())) * 100
            short = n.replace(f"model.layers.{li}.", f"L{li} ")
            print(f"{short:<28}{W.size:>9}{mx:>9.5f}{sd:>9.5f}{G:>5}{rms_e:>9.5f}{rel:>9.3f}")

print("\n=== group scale / weight magnitude distributions (L0 Wq, G=64) ===")
Wq = f.get("model.layers.0.self_attn.q_proj.weight")
grp = Wq.reshape(-1, 64)
scl = np.max(np.abs(grp), axis=1) / 7.0
aw = np.abs(Wq)
print(f"n_groups={scl.shape[0]}  scale mean={scl.mean():.5f} p50={np.percentile(scl,50):.5f} "
      f"p90={np.percentile(scl,90):.5f} max={scl.max():.5f}")
print(f"|w| mean={aw.mean():.5f} p50={np.percentile(aw,50):.5f} "
      f"p99={np.percentile(aw,99):.5f} max={aw.max():.5f}")
print("embed_tokens weight: mean=%.5f p99=%.5f max=%.5f" % (
    np.abs(f.get("model.embed_tokens.weight")).mean(),
    np.percentile(np.abs(f.get("model.embed_tokens.weight")), 99),
    np.abs(f.get("model.embed_tokens.weight")).max()))
