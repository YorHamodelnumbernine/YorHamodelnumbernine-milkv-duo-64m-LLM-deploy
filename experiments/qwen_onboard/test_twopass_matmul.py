#!/usr/bin/env python3
"""test_twopass_matmul.py — verify engine-oriented two-pass matmul (Step ②).

Verifies the restructured per-chunk KG=32 two-pass microkernel (twopass_matmul.c):
  1. C --selftest: r_opt BIT-IDENTICAL vs reference on all production shapes
     (integer two-pass semantics preserved exactly).
  2. REAL Qwen2.5-0.5B layer-0 weights (weights_kal/): for all 7 matmul shapes,
     C binary output (fp32 accumulate) vs numpy fp64-gold reference — report
     maxrel and confirm logit-scale error is far below the ~0.5 gap threshold.
"""
import os, sys, struct, subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
C_BIN = os.path.join(HERE, "twopass_matmul")
C_SRC = os.path.join(HERE, "twopass_matmul.c")
WDIR = os.path.join(HERE, "weights_kal")

G = 32
D, H, KVH, HD, L, F, V = 896, 14, 2, 64, 24, 4864, 151936
DKV = KVH * HD

# ---------------- TIU semantics (locked) ----------------
def i8_round_div(acc, r):
    half = (1 << r) >> 1 if r > 0 else 0
    # (acc + half) >> r with arithmetic shift; clamp
    x = (acc + half) >> r
    return np.clip(x, -128, 127).astype(np.int8)

def mm_rshift_w(K, wmax):
    md = K * 127 * wmax
    r = 0
    while (md >> r) > 127:
        r += 1
    return r

def ref_chunk_twopass(x, w, gsc, sc_row):
    """fp64-gold reference matching qwen_kal_ref.c::chunk_matmul_twopass."""
    x = np.asarray(x, dtype=np.int32)   # [M,K]
    w = np.asarray(w, dtype=np.int32)   # [K,N]
    gsc = np.asarray(gsc, dtype=np.float64)
    M, K = x.shape
    K, N = w.shape
    KG = K // G
    wmax = int(np.abs(w).max())
    rsafe = mm_rshift_w(G, wmax) - 3
    rsafe = max(rsafe, 4)
    accd = np.zeros((M, N), dtype=np.float64)
    rlist = []
    for g in range(KG):
        xb = x[:, g * G:(g + 1) * G]                       # [M,G]
        wb = w[g * G:(g + 1) * G, :]                       # [G,N]
        acc = xb.astype(np.int64) @ wb.astype(np.int64)    # exact int64
        p1 = i8_round_div(acc, rsafe)
        est = int(np.abs(p1).max()) * (1 << rsafe)
        r = 0
        while est > (127 << r):
            r += 1
        rlist.append(r)
        p2 = i8_round_div(acc, r).astype(np.float64)
        accd += p2 * (1 << r) * gsc[g, :][None, :]
    out = accd * np.asarray(sc_row, dtype=np.float64)[:, None]
    return out, rlist, rsafe

# ---------------- weight unpack (weights_kal/) ----------------
def unpack_mat(nib, gsc_b):
    """nib: [KG,N,16] uint8 packed; gsc_b: [KG,N] fp16 -> raw int8 [K,N] + fp32 gsc."""
    KG, N = gsc_b.shape
    q = np.zeros((KG, G, N), dtype=np.int8)
    nib = nib.reshape(KG, N, G // 2)
    lo = (nib & 0x0F).astype(np.int8)
    hi = (nib >> 4).astype(np.int8)
    lo = np.where(lo > 7, lo - 16, lo)     # [KG,N,G/2]
    hi = np.where(hi > 7, hi - 16, hi)
    q[:, 0::2, :] = lo.transpose(0, 2, 1)  # q[g,2j,n] = lo[g,n,j]
    q[:, 1::2, :] = hi.transpose(0, 2, 1)  # q[g,2j+1,n] = hi[g,n,j]
    q = q.reshape(KG * G, N)              # [K,N]
    gsc = gsc_b.astype(np.float32)        # [KG,N]
    return q, gsc

def load_layer0():
    path = os.path.join(WDIR, "layer0_kal.bin")
    with open(path, "rb") as f:
        rms_attn = np.frombuffer(f.read(D * 4), dtype=np.float32).copy()
        mats = {}
        for name, K, N in [("q_proj", D, D), ("k_proj", D, DKV), ("v_proj", D, DKV),
                           ("o_proj", D, D), ("up_proj", D, F), ("gate_proj", D, F),
                           ("down_proj", F, D)]:
            nib = np.frombuffer(f.read((K // G) * N * 16), dtype=np.uint8)
            gsc = np.frombuffer(f.read((K // G) * N * 2), dtype=np.float16)
            nib = nib.reshape((K // G), N, 16)
            gsc = gsc.reshape((K // G), N)
            mats[name] = unpack_mat(nib, gsc)
        rms_ffn = np.frombuffer(f.read(D * 4), dtype=np.float32).copy()
    return rms_attn, mats, rms_ffn

def run_c(M, K, N, x, w, gsc, srow):
    inp = bytes(x.tobytes()) + bytes(w.tobytes()) + bytes(gsc.tobytes()) + bytes(srow.tobytes())
    r = subprocess.run([C_BIN, str(M), str(K), str(N)], input=inp,
                       capture_output=True, check=True)
    return np.frombuffer(r.stdout, dtype=np.float32).reshape(M, N)

def per_row_np(x):
    x = np.asarray(x, dtype=np.float32)
    M, K = x.shape
    mx = np.max(np.abs(x), axis=1, keepdims=True)
    s = np.maximum(mx / 127.0, 1e-12).astype(np.float32)
    q = np.clip(np.round(x / s), -128, 127).astype(np.int8)
    return q, s.ravel()

def main():
    print("== step ② two-pass matmul restructure ==")
    rc = os.system(f"gcc -O2 -o {C_BIN} {C_SRC} -lm")
    assert rc == 0, "C build failed"
    st = subprocess.run([C_BIN, "--selftest"], capture_output=True)
    print(st.stdout.decode())
    assert st.returncode == 0, "C selftest failed"
    print("   [1/2] C selftest PASS: r_opt IDENTICAL on all production shapes")

    # 2. real layer-0 weights
    rms_attn, mats, _ = load_layer0()
    emb = np.fromfile(os.path.join(WDIR, "embed_i8.bin"), dtype=np.int8).reshape(-1, D)
    esc = np.fromfile(os.path.join(WDIR, "embed_scales.f32"), dtype=np.float32)
    toks = [105538, 59975, 100132]
    x_emb = emb[toks].astype(np.float32) * esc[toks][:, None]
    rms = np.sqrt((x_emb.astype(np.float64) ** 2).mean(axis=1, keepdims=True) + 1e-6)
    h = (x_emb / rms).astype(np.float32)
    xi, scr = per_row_np(h)          # [3,896] + sc_row
    M = 3

    worst = 0.0
    def check(name, w, gsc, xx, srow, tag):
        nonlocal worst
        K, N = w.shape
        out_c = run_c(M, K, N, xx, w, gsc, srow)
        out_ref, rlist, rsafe = ref_chunk_twopass(xx, w, gsc, srow)
        diff = np.abs(out_c.astype(np.float64) - out_ref)
        denom = np.maximum(np.abs(out_ref), 1e-12)
        maxrel = float((diff / denom).max())
        absmax = float(diff.max())
        ok = maxrel < 1e-4 or absmax < 1e-3
        worst = max(worst, maxrel)
        print(f"   {name:9s} [{K}x{N}]{tag} rsafe={rsafe} r_opt[min={min(rlist)},max={max(rlist)}] "
              f"maxrel={maxrel:.3e} absmax={absmax:.3e} {'OK' if ok else 'CHECK'}")

    for name in ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj"]:
        w, gsc = mats[name]          # [K,N], [KG,N]
        check(name, w, gsc, xi, scr, "")

    # down_proj real input: mid = up(x) * silu(gate(x)) via fp64 reference, then per-row quant
    wup, gup = mats["up_proj"]; wgt, ggt = mats["gate_proj"]
    up_ref, _, _ = ref_chunk_twopass(xi, wup, gup, scr)
    gt_ref, _, _ = ref_chunk_twopass(xi, wgt, ggt, scr)
    def silu(z): return z / (1.0 + np.exp(-z))
    mid = (up_ref * silu(gt_ref)).astype(np.float32)         # [3,4864]
    mid_i8, mid_sc = per_row_np(mid)
    wdn, gdn = mats["down_proj"]
    check("down_proj", wdn, gdn, mid_i8, mid_sc, " (real mid)")
    print(f"   [2/2] real layer-0 7 shapes: worst maxrel={worst:.3e}")
    if worst > 1e-3:
        print("   WARNING: fp32-acc maxrel > 1e-3 on real data — check accumulate path")
        sys.exit(1)
    print("== step ② twopass_matmul: PASS (r_opt IDENTICAL + real-data fp32-acc OK) ==")

if __name__ == "__main__":
    main()
