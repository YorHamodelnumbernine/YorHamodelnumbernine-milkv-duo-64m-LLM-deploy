#!/usr/bin/env python3
"""dbg_ffn_layer0.py — dump layer-0 FFN intermediates (weights_kal path) to
match against qwen_kal_ref.c debug prints."""
import os, sys, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import emu_group_int4 as E4
from emu_perrow import D, KVH, DKV, L, F, G, PROMPTS, REF
import verify_kal_roundtrip as R

wdir = os.path.join(HERE, "weights_kal")
print(f"[dbg] loading {wdir}")
layers = R.load_layers(wdir)
wo0 = layers[0][1]["Wo"]
print(f"[dbg] Wo.q[0..3]={wo0[0][0]} {wo0[0][1]} {wo0[0][2]} {wo0[0][3]} q[100..103]={wo0[0][100]} {wo0[0][101]} {wo0[0][102]} {wo0[0][103]} gsc[0..3]={layers[0][4]['Wo'][0][0]:.9f} {layers[0][4]['Wo'][0][1]:.9f} {layers[0][4]['Wo'][0][2]:.9f} {layers[0][4]['Wo'][0][3]:.9f}")
print("[dbg] Wo chunk0 rows0..7 col0..7:")
for kk in range(8):
    print(f"  r{kk}: {' '.join(str(int(wo0[kk, nn])) for nn in range(8))}")
embed = np.fromfile(os.path.join(wdir, "embed_i8.bin"), dtype=np.int8).reshape(151936, D)
esc = np.fromfile(os.path.join(wdir, "embed_scales.f32"), dtype=np.float32)
frms = np.fromfile(os.path.join(wdir, "final_rms.f32"), dtype=np.float32)

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(os.path.join(E4.PRE, "model"), trust_remote_code=True)
ids = tok.encode(PROMPTS[0], add_special_tokens=False)

# --- replicate E4.forward up to layer-0 FFN (weights_kal path) ---
seq = len(ids)
cos, sin = E4.precompute_rope(max(64, seq + 8), E4.HD, E4.ROPE_THETA)
x = np.zeros((seq, D), np.float32)
for i, t in enumerate(ids):
    t = 0 if (t < 0 or t >= 151936) else t
    x[i] = embed[t].astype(np.float32) * esc[t]

l = 0
rms_attn, mats, rms_ffn, bias, gsc = layers[l]
bq, bk, bv = bias[:D], bias[D:D + DKV], bias[D + DKV:]
h = E4.rms_norm(x, rms_attn)
x_i8, sc_row = E4.C8.quant_per_row(h)
q = E4.matmul_group_i4(x_i8, mats["Wq"], gsc["Wq"], sc_row) + bq
k = E4.matmul_group_i4(x_i8, mats["Wk"], gsc["Wk"], sc_row) + bk
v = E4.matmul_group_i4(x_i8, mats["Wv"], gsc["Wv"], sc_row) + bv
q = q.reshape(seq, E4.H, E4.HD); k = k.reshape(seq, KVH, E4.HD); v = v.reshape(seq, KVH, E4.HD)
for s in range(seq):
    for hh in range(E4.H): q[s, hh] = E4.rope(q[s, hh], s, cos, sin, E4.HD)
    for hh in range(KVH): k[s, hh] = E4.rope(k[s, hh], s, cos, sin, E4.HD)
attn = np.zeros((seq, E4.H, E4.HD), np.float32)
for hh in range(E4.H):
    kvh = hh // E4.GROUPS
    qh = q[:, hh, :]; kh = k[:, kvh, :]; vh = v[:, kvh, :]
    lg = (qh @ kh.T) * (1.0 / np.sqrt(E4.HD))
    mask = np.triu(np.ones((seq, seq)), 1).astype(bool)
    lg = np.where(mask, -1e30, lg)
    pr = np.exp(lg - lg.max(-1, keepdims=True)); pr /= pr.sum(-1, keepdims=True)
    attn[:, hh, :] = pr @ vh
attn = attn.reshape(seq, D)
a_i8, sc_a = E4.C8.quant_per_row(attn)
print(f"[dbg] l0 attn[0..3]={attn[0][0]:.9f} {attn[0][1]:.9f} {attn[0][2]:.9f} {attn[0][3]:.9f} a_i8[0..3]={a_i8[0][0]} {a_i8[0][1]} {a_i8[0][2]} {a_i8[0][3]} sc_a={sc_a[0]:.9f}")
print(f"[dbg] l0 a_i8[0..31]={' '.join(str(int(v)) for v in a_i8[0][0:32])}")
# manual Wo matmul with per-chunk r dump
wo = mats["Wo"]; gw = gsc["Wo"]; KG = wo.shape[0] // G
rs = []
wmax = int(np.max(np.abs(wo)))
rsafe = max(E4.matmul_rshift_w(G, wmax) - 3, 4)
for g in range(0, wo.shape[0], G):
    acc = a_i8[:, g:g + G].astype(np.int32) @ wo[g:g + G, :].astype(np.int32)
    if g == 0:
        print(f"[dbg] py chunk0 rsafe={rsafe} wmax={wmax} maxacc={int(np.max(np.abs(acc)))} maxp1={int(np.max(np.abs(E4.int8_round_div(acc, rsafe))))}")
        print(f"[dbg] py acc m0 n0..7={' '.join(str(int(v)) for v in acc[0, 0:8])}")
        am = int(np.unravel_index(np.argmax(np.abs(acc)), acc.shape)[0]); an = int(np.unravel_index(np.argmax(np.abs(acc)), acc.shape)[1])
        print(f"[dbg] py chunk0 argmax m={am} n={an} val={int(acc[am, an])}")
        print(f"[dbg] py xi row1[0..31]={' '.join(str(int(v)) for v in a_i8[1, 0:32])}")
    p1 = E4.int8_round_div(acc, rsafe)
    est = int(np.max(np.abs(p1))) * (1 << rsafe)
    r = 0
    if est > 1e-6:
        r = int(math.ceil(math.log2(est / 127.0))); r = max(r, 0)
    rs.append(r)
print(f"[dbg] l0 wo r[0..27]={' '.join(map(str, rs[:28]))}")
x = x + E4.matmul_group_i4(a_i8, mats["Wo"], gsc["Wo"], sc_a)
print(f"[dbg] l0 after-wo x[0..3]={x[0][0]:.9f} {x[0][1]:.9f} {x[0][2]:.9f} {x[0][3]:.9f} spread@100,300,500,700={x[0][100]:.9f} {x[0][300]:.9f} {x[0][500]:.9f} {x[0][700]:.9f}")

# FFN
h = E4.rms_norm(x, rms_ffn)
x_i8, sc_row = E4.C8.quant_per_row(h)
print(f"[dbg] l0 ffn h[0..2]={h[0][0]:.9f} {h[0][1]:.9f} {h[0][2]:.9f} xi[0..2]={x_i8[0][0]} {x_i8[0][1]} {x_i8[0][2]} scr[0]={sc_row[0]:.9f}")
up = E4.matmul_group_i4(x_i8, mats["up"], gsc["up"], sc_row)
gate = E4.matmul_group_i4(x_i8, mats["gate"], gsc["gate"], sc_row)
mid = up * E4.silu(gate)
print(f"[dbg] l0 ffn up[0..3]={up[0][0]:.6f} {up[0][1]:.6f} {up[0][2]:.6f} {up[0][3]:.6f} gate[0..3]={gate[0][0]:.6f} {gate[0][1]:.6f} {gate[0][2]:.6f} {gate[0][3]:.6f} mid[0..3]={mid[0][0]:.6f} {mid[0][1]:.6f} {mid[0][2]:.6f} {mid[0][3]:.6f}")
out = np.zeros((seq, D), np.float32)
for kc in range(0, F, 1024):
    kcn = min(1024, F - kc)
    m_i8, sc_m = E4.C8.quant_per_row(mid[:, kc:kc + kcn])
    wc = mats["down"][kc:kc + kcn, :]
    gsw = gsc["down"][kc // G:(kc + kcn) // G, :]
    out += E4.matmul_group_i4(m_i8, wc, gsw, sc_m)
x = x + out
print(f"[dbg] l0 after-ffn x[0..3]={x[0][0]:.9f} {x[0][1]:.9f} {x[0][2]:.9f} {x[0][3]:.9f}")
