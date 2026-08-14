#!/usr/bin/env python3
"""test_per_row_quant.py — verify engine-ready per_row_quant.c against numpy.

Step ① integration gate: the per-row (per-token) INT8 activation quant is the
common item for both Path A and B.  This test locks the exact semantics
(sc = max|row|/127, round-half-even) that the C906B TIU microkernel must use.

Checks:
  1. C --selftest RC=0 (edge cases, banker's rounding, saturation, decode fastpath)
  2. Random [M,K] matrices: C output == numpy reference (sc + q) bit-exact
  3. Real layer-0 activation (RMS-norm of embed) from the Qwen path: C == numpy
"""
import os, sys, struct, subprocess, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
C_BIN = os.path.join(HERE, "per_row_quant")
C_SRC = os.path.join(HERE, "per_row_quant.c")

def np_round_bankers(v):
    """numpy np.round = round-half-to-even."""
    return np.round(v)

def per_row_np(x):
    """Reference per-row quant matching per_row_quant.c semantics."""
    x = np.asarray(x, dtype=np.float32)
    M, K = x.shape
    mx = np.max(np.abs(x), axis=1, keepdims=True)          # [M,1]
    s = np.maximum(mx / 127.0, 1e-12).astype(np.float32)   # [M,1]
    q = np.clip(np_round_bankers(x / s), -128, 127).astype(np.int8)
    return q, s.ravel().astype(np.float32)

def run_c(M, K, x):
    inp = np.asarray(x, dtype=np.float32).tobytes()
    r = subprocess.run([C_BIN, str(M), str(K)], input=inp,
                       capture_output=True, check=True)
    out = r.stdout
    q = np.frombuffer(out[:M * K], dtype=np.int8)
    sc = np.frombuffer(out[M * K:], dtype=np.float32)
    return q.reshape(M, K), sc

def main():
    print("== step ① per_row_quant integration ==")
    # 1. compile + selftest
    rc = os.system(f"gcc -O2 -o {C_BIN} {C_SRC} -lm")
    assert rc == 0, "C build failed"
    st = subprocess.run([C_BIN, "--selftest"], capture_output=True)
    print(st.stdout.decode())
    print(st.stderr.decode(), end="")
    assert st.returncode == 0, "C selftest failed"
    print("   [1/3] C selftest PASS (RC=0)")

    # 2. random matrices, bit-exact vs numpy
    rng = np.random.default_rng(42)
    cases = [(1, 896), (3, 896), (10, 896), (1, 1024), (7, 4864)]
    for M, K in cases:
        x = rng.standard_normal((M, K)).astype(np.float32) * rng.uniform(0.01, 3.0)
        # add occasional large outliers to force saturation
        x += (rng.random((M, K)) < 0.01) * rng.uniform(3, 20)
        q_c, sc_c = run_c(M, K, x)
        q_n, sc_n = per_row_np(x)
        ok = np.array_equal(q_c, q_n) and np.array_equal(sc_c, sc_n)
        status = "BIT-EXACT" if ok else "MISMATCH"
        if not ok:
            ndiff = np.count_nonzero(q_c != q_n)
            print(f"   [{M}x{K}] {status} ndiff={ndiff} sc_diff={np.abs(sc_c-sc_n).max():.3e}")
        else:
            print(f"   [{M}x{K}] {status}")
        assert ok, f"random case {M}x{K} mismatch"
    print("   [2/3] random MxK BIT-EXACT vs numpy")

    # 3. real layer-0 activation: RMS-norm(embed[row]) from Qwen path
    wdir = os.path.join(HERE, "weights_kal")
    emb = np.fromfile(os.path.join(wdir, "embed_i8.bin"), dtype=np.int8).reshape(-1, 896)
    esc = np.fromfile(os.path.join(wdir, "embed_scales.f32"), dtype=np.float32)
    toks = [105538, 59975, 100132]   # P0 prompt 1
    x = emb[toks].astype(np.float32) * esc[toks][:, None]
    # rms_norm (Qwen eps 1e-6), matching qwen_kal_ref.c
    rms = np.sqrt((x.astype(np.float64) ** 2).mean(axis=1, keepdims=True) + 1e-6)
    h = (x / rms).astype(np.float32)
    M, K = h.shape
    q_c, sc_c = run_c(M, K, h)
    q_n, sc_n = per_row_np(h)
    ok = np.array_equal(q_c, q_n) and np.array_equal(sc_c, sc_n)
    print(f"   [real rms_attn layer0 {M}x{K}] {'BIT-EXACT' if ok else 'MISMATCH'}")
    assert ok
    print("   [3/3] real layer-0 activation BIT-EXACT vs numpy")
    print("== step ① per_row_quant integration: PASS 3/3 ==")

if __name__ == "__main__":
    main()
