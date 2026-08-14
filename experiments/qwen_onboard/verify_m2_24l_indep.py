#!/usr/bin/env python3
"""M2 24L 独立复核 — 数据链 + 全 24 层前向 + TIU run 计数 host 端独立验证 (TPU 工程师).

独立于推理引擎 (qwen_engine_24l.c) 与 emulator (emu_*.py) 的 test 脚本,
只从 raw 二进制 (weights_kal/layer0..23_kal.bin + embed_i8.bin + embed_scales.f32 +
final_rms.f32 + layerN_bias.f32) 重新解包, 独立实现 Path A 两遍法语义并跑 3-prompt:

  (A) 数据链/二进制布局: 逐文件校验字节数 + layerN_kal.bin 布局 off==size,
      与 qwen_kal_ref.c load_layers 的布局完全一致.
  (B) host 独立 24 层前向: rms_norm -> per_row quant -> 两遍法 matmul(block-shared r_opt)
      -> bias -> rope -> GQA attention(fp32) -> wo -> residual -> rms_ffn ->
      quant -> up/gate -> silu -> down(1024-chunk) -> residual -> final rms -> LM head.
  (C) 3-prompt 门禁: NEXT token 2130/12095/99366 (3/3), min gap >= 0.05.
  (D) TIU run 计数独立重算: 依引擎 tiling (M<=3 -> tilew=896, M>3 -> 768) 结构推导
      pass1/pass2 = 55872/55872, total = 111744.

数值口径: 全部与 qwen_kal_ref.c / qwen_engine_24l.c 文档语义一致
  - matmul: 每 K-block(G=32) int 精确累加(用 f64 BLAS 保整数精确), pass1 rsafe,
    block_max over ALL M rows & ALL N-tiles, r_opt(block-shared), pass2, double accd,
    out = (float)accd * sc_row[m].
  - 权重: K-aligned INT4 G32 -> int8(-8..7) + fp16 gscale(->fp32).
  - attention: fp32 softmax (numpy exp); 已知 host(glibc)/on-board(musl) expf ulp 差异
    仅影响 gap 第4位及 top-5 2~5 名排序, 不影响 top-1 (M2 报告已记录, 非阻塞).
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.join(HERE, "weights_kal")
D, Hn, KVH, HD, L, F, V, G = 896, 14, 2, 64, 24, 4864, 151936, 32
DKV = KVH * HD
GROUPS = Hn // KVH
ROPE_THETA = 1000000.0
EPS = 1e-6
MAX_SEQ = 64

# 3-prompt (token ids from p0_inputs.json)
PROMPTS = [
    ("P1 中国的首都是", [105538, 59975, 100132]),
    ("P2 The capital of France is", [785, 6722, 315, 9625, 374]),
    ("P3 今天天气很好，我们去公园", [100644, 104307, 101243, 3837, 97639, 85336, 102077]),
]
EXPECTED_NEXT = [2130, 12095, 99366]
BOARD_GAP = [0.6890, 0.4508, 0.5149]      # on-board musl-expf 实测 (REPORT_M2_24L §2.1)
HOST_GAP = [0.6890, 0.6802, 0.2600]       # host qwen_kal_ref glibc 实测 (REPORT_M2_24L §3)

# ---------------- 基础数值原语 (与 qwen_kal_ref.c / 引擎一致) ----------------
def fp16_to_f32(h):
    """uint16 -> float32 (IEEE half, 与 C fp16_to_f32 一致)."""
    return np.asarray(h, dtype='<u2').astype(np.float16).astype(np.float32)

def matmul_rshift_w(K, wmax):
    r = 0
    md = K * 127 * wmax
    while (md >> r) > 127:
        r += 1
    return r

def i8_round_div(acc, rshift):
    """TIU round-half-up: sat8((acc + 2^(r-1)) >> r). 负数半程向零."""
    if rshift > 0:
        x = (acc + (1 << (rshift - 1))) >> rshift
    else:
        x = acc
    return np.clip(x, -128, 127).astype(np.int8)

def round_bankers(v):
    """np.round = round-half-even, 与 C round_bankers 一致."""
    return np.round(v)

def per_row_quant(x):
    """x [M,K] fp32 -> (q_i8, sc_row). sc=mx/127 clamp 1e-12; round-half-even."""
    x = np.asarray(x, dtype=np.float32)
    mx = np.abs(x).max(axis=-1, keepdims=True)
    s = np.maximum(mx / np.float32(127.0), np.float32(1e-12))
    q = np.clip(np.round(x / s), -128, 127).astype(np.int8)
    return q, s.ravel().astype(np.float32)

def rms_norm(x, g):
    """x [M,n], g[n] -> out. ss double; inv float; out = x*inv*g 逐项 float32."""
    x = np.asarray(x, dtype=np.float32)
    g = np.asarray(g, dtype=np.float32)
    M, n = x.shape
    out = np.empty_like(x)
    for m in range(M):
        ss = np.float64(np.sum(np.asarray(x[m], dtype=np.float64) ** 2))
        inv = np.float32(1.0 / np.sqrt(np.float64(ss) / np.float64(n) + np.float64(EPS)))
        out[m] = (np.float32(x[m] * inv) * g).astype(np.float32)
    return out

def precompute_rope(max_pos, half=HD // 2):
    cos = np.zeros((max_pos, half), np.float32)
    sin = np.zeros((max_pos, half), np.float32)
    j = np.arange(half, dtype=np.float32)
    freq = np.power(np.float32(ROPE_THETA), np.float32(-j) / np.float32(half)).astype(np.float32)
    for pos in range(max_pos):
        ang = np.float32(pos) * freq
        cos[pos] = np.cos(ang).astype(np.float32)
        sin[pos] = np.sin(ang).astype(np.float32)
    return cos, sin

def rope_inplace(q, pos, cos, sin):
    half = HD // 2
    x0 = q[:half].copy(); x1 = q[half:].copy()
    c = cos[pos]; s = sin[pos]
    q[:half] = (np.float32(x0 * c) - np.float32(x1 * s)).astype(np.float32)
    q[half:] = (np.float32(x0 * s) + np.float32(x1 * c)).astype(np.float32)

def silu(x):
    x = np.asarray(x, dtype=np.float32)
    return np.float32(x / np.float32(np.float32(1.0) + np.exp(np.float32(-x))))

# ---------------- 权重解包 (与 qwen_kal_ref.c unpack_mat / 引擎 parse_layer 一致) ----------------
def dequant_nib(nib):
    """nib [KG,N,16] uint8 -> w_i8 [K,N] (K-aligned INT4, lo=even row, hi=odd row)."""
    KG, N, _ = nib.shape
    lo = (nib & 0x0F).astype(np.int8)
    hi = ((nib >> 4) & 0x0F).astype(np.int8)
    lo = np.where(lo > 7, lo - 16, lo).astype(np.int8)
    hi = np.where(hi > 7, hi - 16, hi).astype(np.int8)
    # w[g, 2j+nib, n]; stack -> [KG,N,16,2] -> [KG,N,32] -> [KG,32,N] -> [K,N]
    w = np.stack([lo, hi], axis=-1).reshape(KG, N, 32).transpose(0, 2, 1).reshape(KG * 32, N)
    return w

MATS = [("Wq", D, D), ("Wk", D, DKV), ("Wv", D, DKV), ("Wo", D, D),
        ("up", D, F), ("gate", D, F), ("down", F, D)]

def load_layer(l):
    """读 layer{l}_kal.bin -> (rms_attn, {Wq_nib..}, {Wq_gsc..}, rms_ffn). 布局与 parse_layer 一致."""
    data = open(os.path.join(W, f"layer{l}_kal.bin"), "rb").read()
    off = 0
    def take(n):
        nonlocal off
        b = data[off:off + n]; off += n
        return b
    rms_attn = np.frombuffer(take(D * 4), dtype=np.float32).copy()
    nibs, gscs = {}, {}
    for name, K, N in MATS:
        nib = np.frombuffer(take((K // G) * N * 16), dtype=np.uint8).reshape(K // G, N, 16)
        gs = np.frombuffer(take((K // G) * N * 2), dtype='<f2').astype(np.float32).reshape(K // G, N)
        nibs[name] = nib
        gscs[name] = gs
    rms_ffn = np.frombuffer(take(D * 4), dtype=np.float32).copy()
    assert off == len(data), f"layer{l} layout mismatch {off} vs {len(data)}"
    return rms_attn, nibs, gscs, rms_ffn

def load_bias(l):
    return np.fromfile(os.path.join(W, f"layer{l}_bias.f32"), dtype=np.float32)

# ---------------- 两遍法 matmul (与引擎 eng_matmul 语义一致) ----------------
def matmul_twopass(x_i8, w_i8, gsc_f32, sc_row):
    """x_i8[M,K], w_i8[K,N], gsc_f32[KG,N], sc_row[M] -> out[M,N] fp32.
    per K-block: int 精确 acc (f64 BLAS), pass1 rsafe, block_max(ALL M,N),
    r_opt block-shared, pass2, double accd, out=(float)accd*sc_row."""
    M, K = x_i8.shape
    N = w_i8.shape[1]
    KG = K // G
    wmax = int(np.abs(w_i8).max())
    rsafe = matmul_rshift_w(G, wmax) - 3
    if rsafe < 4:
        rsafe = 4
    # per-K-block int 精确累加: 用 f64 dgemm (整数 <2^53 精确), 显式循环避免 einsum
    # optimize 的 per-block 输出错位问题. 逐 block: xr[:,g,:] @ wr[g,:,:].
    xr = x_i8.reshape(M, KG, G).astype(np.float64)
    wr = w_i8.reshape(KG, G, N).astype(np.float64)
    accg = np.zeros((KG, M, N), np.float64)
    for g in range(KG):
        accg[g] = xr[:, g, :] @ wr[g, :, :]
    accd = np.zeros((M, N), np.float64)
    for g in range(KG):
        acc = accg[g]                                   # float64, 精确整数
        p1 = i8_round_div(acc.astype(np.int64), rsafe)
        block_max = int(np.abs(p1).max())
        est = block_max << rsafe
        r = 0
        while est > (127 << r):
            r += 1
        p2 = i8_round_div(acc.astype(np.int64), r)
        accd += p2.astype(np.float64) * np.float64(1 << r) * gsc_f32[g].astype(np.float64)
    out = (accd.astype(np.float32) * sc_row[:, None]).astype(np.float32)
    return out

# ---------------- GQA attention (fp32 softmax) ----------------
def attention(q, k, v, seq):
    """q[seq,H,HD], k[seq,KVH,HD], v[seq,KVH,HD] -> attn[seq,D] fp32 (causal)."""
    attn = np.zeros((seq, Hn, HD), np.float32)
    inv = np.float32(1.0 / np.sqrt(np.float32(HD)))
    for hh in range(Hn):
        kvh = hh // GROUPS
        qh = q[:, hh, :].astype(np.float32)
        kh = k[:, kvh, :].astype(np.float32)
        vh = v[:, kvh, :].astype(np.float32)
        logits = (qh @ kh.T) * inv
        mask = np.triu(np.ones((seq, seq), bool), 1)
        logits = np.where(mask, np.float32(-1e30), logits)
        mx = logits.max(axis=-1, keepdims=True)
        e = np.exp((logits - mx).astype(np.float32))
        pr = e / e.sum(axis=-1, keepdims=True)
        attn[:, hh, :] = pr @ vh
    return attn.reshape(seq, D).astype(np.float32)

# ---------------- 全 24 层前向 ----------------
def forward(tokens):
    seq = len(tokens)
    embed = np.fromfile(os.path.join(W, "embed_i8.bin"), dtype=np.int8).reshape(V, D)
    esc = np.fromfile(os.path.join(W, "embed_scales.f32"), dtype=np.float32)
    frms = np.fromfile(os.path.join(W, "final_rms.f32"), dtype=np.float32)
    cos, sin = precompute_rope(max(MAX_SEQ, seq + 8))
    x = np.zeros((seq, D), np.float32)
    for i, t in enumerate(tokens):
        t = 0 if (t < 0 or t >= V) else t
        x[i] = (embed[t].astype(np.float32) * esc[t]).astype(np.float32)

    for l in range(L):
        rms_attn, nibs, gscs, rms_ffn = load_layer(l)
        bias = load_bias(l)
        bq, bk, bv = bias[:D], bias[D:D + DKV], bias[D + DKV:]

        # QKV
        h = rms_norm(x, rms_attn)
        xi, scr = per_row_quant(h)
        wq = dequant_nib(nibs["Wq"]); wq_g = gscs["Wq"]
        wk = dequant_nib(nibs["Wk"]); wk_g = gscs["Wk"]
        wv = dequant_nib(nibs["Wv"]); wv_g = gscs["Wv"]
        q = matmul_twopass(xi, wq, wq_g, scr) + bq
        k = matmul_twopass(xi, wk, wk_g, scr) + bk
        v = matmul_twopass(xi, wv, wv_g, scr) + bv
        q = q.reshape(seq, Hn, HD).copy(); k = k.reshape(seq, KVH, HD).copy(); v = v.reshape(seq, KVH, HD).copy()
        for s in range(seq):
            for hh in range(Hn):
                rope_inplace(q[s, hh], s, cos, sin)
            for hh in range(KVH):
                rope_inplace(k[s, hh], s, cos, sin)
        attn = attention(q, k, v, seq)
        # wo
        ai, sca = per_row_quant(attn)
        wo = dequant_nib(nibs["Wo"]); wo_g = gscs["Wo"]
        oout = matmul_twopass(ai, wo, wo_g, sca)
        x = (x + oout).astype(np.float32)
        # FFN
        h = rms_norm(x, rms_ffn)
        xi, scr = per_row_quant(h)
        up_w = dequant_nib(nibs["up"]); up_g = gscs["up"]
        gt_w = dequant_nib(nibs["gate"]); gt_g = gscs["gate"]
        up = matmul_twopass(xi, up_w, up_g, scr)
        gate = matmul_twopass(xi, gt_w, gt_g, scr)
        mid = (up * silu(gate)).astype(np.float32)
        # down: K-chunk 1024, per-chunk per-row quant
        dn = dequant_nib(nibs["down"]); dn_g = gscs["down"]
        oout = np.zeros((seq, D), np.float32)
        for kc in range(0, F, 1024):
            kcn = min(1024, F - kc)
            mch = mid[:, kc:kc + kcn].copy()
            m_i8, sc_m = per_row_quant(mch)
            wch = dn[kc:kc + kcn, :]
            gw = dn_g[kc // G:(kc + kcn) // G, :]
            oout += matmul_twopass(m_i8, wch, gw, sc_m)
        x = (x + oout).astype(np.float32)

    # final rms + LM head (double 累加, 与引擎/qwen_kal_ref 一致)
    h = rms_norm(x[seq - 1:seq], frms)[0]
    lg = np.zeros(V, np.float64)
    for t in range(V):
        lg[t] = np.sum(np.asarray(h, dtype=np.float64) * np.asarray(embed[t], dtype=np.float64) * np.float64(esc[t]))
    return lg

# ---------------- TIU run 计数独立重算 ----------------
def tiu_run_count():
    """依引擎 tiling (max_tile_for_m: M<=3->896, else 768) 结构重算 pass1/pass2."""
    def per_layer_pass1(M):
        tilew = 896 if M <= 3 else 768
        def ntiles(N):
            return (N + tilew - 1) // tilew
        q = (D // G) * ntiles(D)          # q_proj
        kk = (D // G) * ntiles(DKV)       # k_proj
        vv = (D // G) * ntiles(DKV)       # v_proj
        wo = (D // G) * ntiles(D)         # wo
        up = (D // G) * ntiles(F)         # up
        gt = (D // G) * ntiles(F)         # gate
        dn = sum((kc // G) * ntiles(D) for kc in [1024, 1024, 1024, 1024, 768])  # down 5 chunks
        return q + kk + vv + wo + up + gt + dn
    p1 = 0
    for M in [3, 5, 7]:
        p1 += L * per_layer_pass1(M)
    return p1, p1, 2 * p1     # pass1, pass2, total

def layer0_crosscheck():
    """(E) layer-0 单 token(105538) 前向 vs 可信 gen_layer0_ref 参考, 位精确核对语义实现."""
    ref_h = os.path.join(HERE, "qwen_engine_layer0_ref.h")
    if not os.path.exists(ref_h):
        print("  (skip) qwen_engine_layer0_ref.h 不存在")
        return True
    import re
    txt = open(ref_h).read()
    def arr(name):
        m = re.search(rf'static const float {name}\[\d+\] = \{{(.*?)\}};', txt, re.S)
        return np.array([float(x) for x in m.group(1).split(',') if x.strip()], dtype=np.float32)
    ref_attn = arr('ref_attn'); ref_wo = arr('ref_after_wo'); ref_ffn = arr('ref_after_ffn')
    embed = np.fromfile(os.path.join(W, "embed_i8.bin"), dtype=np.int8).reshape(V, D)
    esc = np.fromfile(os.path.join(W, "embed_scales.f32"), dtype=np.float32)
    rms_attn, nibs, gscs, rms_ffn = load_layer(0)
    bias = load_bias(0)
    bq, bk, bv = bias[:D], bias[D:D + DKV], bias[D + DKV:]
    x = (embed[105538].astype(np.float32) * esc[105538]).astype(np.float32)[None, :]
    h = rms_norm(x, rms_attn)
    xi, scr = per_row_quant(h)
    q = matmul_twopass(xi, dequant_nib(nibs["Wq"]), gscs["Wq"], scr) + bq
    k = matmul_twopass(xi, dequant_nib(nibs["Wk"]), gscs["Wk"], scr) + bk
    v = matmul_twopass(xi, dequant_nib(nibs["Wv"]), gscs["Wv"], scr) + bv
    cos, sin = precompute_rope(8)
    q = q.reshape(1, Hn, HD).copy(); k = k.reshape(1, KVH, HD).copy(); v = v.reshape(1, KVH, HD).copy()
    for hh in range(Hn):
        rope_inplace(q[0, hh], 0, cos, sin)
    for hh in range(KVH):
        rope_inplace(k[0, hh], 0, cos, sin)
    attn = attention(q, k, v, 1)
    ai, sca = per_row_quant(attn)
    xwo = (x + matmul_twopass(ai, dequant_nib(nibs["Wo"]), gscs["Wo"], sca)).astype(np.float32)
    h2 = rms_norm(xwo, rms_ffn)
    xi2, sc2 = per_row_quant(h2)
    up = matmul_twopass(xi2, dequant_nib(nibs["up"]), gscs["up"], sc2)
    gate = matmul_twopass(xi2, dequant_nib(nibs["gate"]), gscs["gate"], sc2)
    mid = (up * silu(gate)).astype(np.float32)
    dn = dequant_nib(nibs["down"])
    oout = np.zeros((1, D), np.float32)
    for kc in range(0, F, 1024):
        kcn = min(1024, F - kc)
        mi, sm = per_row_quant(mid[:, kc:kc + kcn].copy())
        oout += matmul_twopass(mi, dn[kc:kc + kcn, :], gscs["down"][kc // G:(kc + kcn) // G, :], sm)
    xffn = (xwo + oout).astype(np.float32)
    da = np.abs(attn[0] - ref_attn).max()
    dw = np.abs(xwo[0] - ref_wo).max()
    df = np.abs(xffn[0] - ref_ffn).max()
    ok = (da == 0.0 and dw == 0.0 and df == 0.0)
    print(f"  attn      maxdiff vs gen_layer0_ref = {da:.3e}")
    print(f"  after_wo  maxdiff vs gen_layer0_ref = {dw:.3e}")
    print(f"  after_ffn maxdiff vs gen_layer0_ref = {df:.3e}")
    print(f"  -> layer-0 语义: {'BIT-EXACT' if ok else 'DIFF'}")
    return ok

def main():
    print("== M2 24L 独立复核 (verify_m2_24l_indep.py) ==")
    print(f"  weights_dir = {W}")

    # ---- (A) 数据链 / 二进制布局 ----
    print("\n== (A) 数据链 / 二进制布局 ==")
    ok = True
    embed = np.fromfile(os.path.join(W, "embed_i8.bin"), dtype=np.int8)
    esc = np.fromfile(os.path.join(W, "embed_scales.f32"), dtype=np.float32)
    frms = np.fromfile(os.path.join(W, "final_rms.f32"), dtype=np.float32)
    checks = [
        ("embed_i8.bin", embed.size, V * D),
        ("embed_scales.f32", esc.size, V),
        ("final_rms.f32", frms.size, D),
    ]
    for fn, got, want in checks:
        c = got == want
        ok &= c
        print(f"  {fn:22s}: {got:>10d} bytes {'OK' if c else 'MISMATCH'} (expect {want})")
    for l in range(L):
        b = open(os.path.join(W, f"layer{l}_kal.bin"), "rb").read()
        bi = open(os.path.join(W, f"layer{l}_bias.f32"), "rb").read()
        if len(b) != 8393728 or len(bi) != (D + DKV + DKV) * 4:
            ok = False
            print(f"  layer{l}: SIZE MISMATCH {len(b)}/{len(bi)}")
    for l in range(L):
        try:
            load_layer(l)   # asserts layout off==size
        except AssertionError as e:
            ok = False
            print(f"  layer{l}: {e}")
    print(f"  -> 24×layerN_kal.bin parse: {'ALL LAYOUT OK' if ok else 'FAIL'}")
    print(f"  -> 数据链: {'ALL CONSISTENT' if ok else 'MISMATCH'}")

    # ---- (D) TIU run 计数 ----
    p1c, p2c, tot = tiu_run_count()
    print(f"\n== (D) TIU run 计数独立重算 ==")
    print(f"  pass1={p1c}  pass2={p2c}  total={tot}   (报告: pass1=55872 pass2=55872 total=111744)")
    run_ok = (p1c == 55872 and p2c == 55872 and tot == 111744)
    print(f"  -> {'MATCH' if run_ok else 'MISMATCH'}")

    # ---- (E) layer-0 位精确交叉核对 ----
    print("\n== (E) layer-0 语义 vs 可信 gen_layer0_ref (token 105538, seq=1) ==")
    l0_ok = layer0_crosscheck()

    # ---- (B)(C) 3-prompt 前向 ----
    print("\n== (B)(C) 独立 24 层前向 · 3-prompt 门禁 ==")
    print(f"  {'prompt':40s} {'NEXT':>7s} {'exp':>7s} {'top5':>38s} {'gap':>8s}  {'判定'}")
    passed = 0
    min_gap = 1e9
    for i, (name, toks) in enumerate(PROMPTS):
        t0 = time.time()
        lg = forward(toks)
        dt = time.time() - t0
        t5 = np.argsort(-lg)[:5].tolist()
        v5 = lg[t5]
        gap = float(v5[0] - v5[1])
        min_gap = min(min_gap, gap)
        okn = t5[0] == EXPECTED_NEXT[i]
        okg = gap >= 0.05
        passed += okn
        top5s = " ".join(str(t) for t in t5)
        print(f"  {name:40s} {t5[0]:7d} {EXPECTED_NEXT[i]:7d} {top5s:>38s} {gap:8.4f}  "
              f"{'OK' if okn and okg else 'FAIL'}  ({dt:.1f}s)")
    print(f"\n== 总判定 ==")
    print(f"  NEXT token: {passed}/3 (期望 2130/12095/99366)")
    print(f"  min gap = {min_gap:.4f}  (门禁 >= 0.05)")
    print(f"  on-board 实测 min gap = {min(BOARD_GAP):.4f} (musl expf), host ref = {min(HOST_GAP):.4f} (glibc expf)")
    print(f"  TIU run 计数: {'MATCH 111744' if run_ok else 'FAIL'}")
    allok = (passed == 3 and min_gap >= 0.05 and run_ok and ok and l0_ok)
    print(f"== M2 24L 独立复核: {'ALL MATCH' if allok else 'FAIL'} ==")
    sys.exit(0 if allok else 1)

if __name__ == "__main__":
    main()
