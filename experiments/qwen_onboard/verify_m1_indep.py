#!/usr/bin/env python3
"""M1 独立复核 — 数据链 + 两遍法语义 host 端独立验证 (TPU 工程师).

独立于推理引擎的 test 脚本, 从 layer0_kal.bin 原始二进制重新解包 5 个 M1 case,
验证:
  (A) qwen_m1_data.h 中嵌入的 nib/gsc/act 与 layer0_kal.bin 原始数据逐字节一致;
  (B) host 独立实现两遍法整数语义 (rsafe / maxabs / r_opt / p1 / p2 / sat8),
      与 REPORT_M1_PASS1 上板实测表逐项对照;
  (C) fp32 累加 maxdiff 复算 (与 qwen_m1_chunk.c 同顺序), 确认 ~1e-8 量级.
"""
import os, sys, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.join(HERE, "weights_kal")
D, KVH, HD, F, G = 896, 2, 64, 4864, 32
DKV = KVH * HD
MATS = [("q_proj", D, D), ("k_proj", D, DKV), ("v_proj", D, DKV),
        ("o_proj", D, D), ("up_proj", D, F), ("gate_proj", D, F), ("down_proj", F, D)]

# ---- TIU 语义 (与 twopass_matmul.c / gate1_mrow_check.c 一致) ----
def i8_round_div(acc, r):
    half = (1 << (r - 1)) if r > 0 else 0
    x = (acc + half) >> r
    return int(max(-128, min(127, x)))

def mm_rshift_w(K, wmax):
    md = K * 127 * wmax
    r = 0
    while (md >> r) > 127:
        r += 1
    return r

def read_layer0():
    data = open(os.path.join(W, "layer0_kal.bin"), "rb").read()
    off = 0
    def take(n):
        nonlocal off
        b = data[off:off + n]; off += n
        return b
    rms_attn = np.frombuffer(take(D * 4), dtype=np.float32).copy()
    mats = {}
    for name, K, N in MATS:
        nib = np.frombuffer(take((K // G) * N * 16), dtype=np.uint8).reshape(K // G, N, 16)
        gsc = np.frombuffer(take((K // G) * N * 2), dtype=np.float16).reshape(K // G, N)
        mats[name] = (nib, gsc.astype(np.float16))
    rms_ffn = np.frombuffer(take(D * 4), dtype=np.float32).copy()
    assert off == len(data)
    return rms_attn, mats, rms_ffn

def dequant_block(nib):
    """nib [N,16] -> w_i8 [32,N]  (raw int4, K 交错 nib 解包, 与 qwen_m1_chunk.c 一致)"""
    N = nib.shape[0]
    w = np.zeros((G, N), dtype=np.int8)
    for n in range(N):
        b = nib[n].astype(np.int32)
        for j in range(16):
            lo = int(b[j] & 0xF)
            hi = int(b[j] >> 4)
            lo = lo - 16 if lo > 7 else lo
            hi = hi - 16 if hi > 7 else hi
            w[2 * j, n] = np.int8(lo)
            w[2 * j + 1, n] = np.int8(hi)
    return w

def per_row_quant(x):
    """round-half-even (bankers), 与 qwen_m1_chunk.c per_row_quant 一致"""
    x = np.asarray(x, dtype=np.float64)
    mx = np.abs(x).max()
    s = mx / 127.0
    if s < 1e-12:
        s = 1e-12
    q = np.round(x / s).astype(np.int32)   # np.round = round-half-even
    q = np.clip(q, -128, 127).astype(np.int8)
    return q, np.float32(s)

def parse_data_h():
    """解析 qwen_m1_data.h 中的 m1_act / m1_<tag>_nib / m1_<tag>_gsc"""
    txt = open(os.path.join(HERE, "qwen_m1_data.h")).read()
    def arr(name):
        m = re.search(rf"static const (?:unsigned \w+|\w+) {name}\[.*?\] = \{{(.*?)\}};", txt, re.S)
        assert m, f"missing {name}"
        body = m.group(1)
        if "float" in m.group(0):
            return np.array([float(x) for x in body.split(",") if x.strip()], dtype=np.float32)
        if "unsigned short" in m.group(0):
            return np.array([int(x) for x in body.split(",") if x.strip()], dtype=np.uint16)
        return np.array([int(x) for x in body.split(",") if x.strip()], dtype=np.uint8)
    act = arr("m1_act")
    cases = []
    for tag in ["q0", "q13", "q27", "dn0", "up0"]:
        nib = arr(f"m1_{tag}_nib").reshape(896, 16)
        gs = arr(f"m1_{tag}_gsc").reshape(896)
        cases.append((tag, nib, gs))
    return act, cases

def fp32_maxdiff(p2, r, gsc, scr):
    """与 qwen_m1_chunk.c 完全一致的 fp32 累加顺序"""
    p2 = np.asarray(p2, dtype=np.int32)
    gsc = np.asarray(gsc, dtype=np.float32)   # fp16->fp32 已存为 uint16, 需按 bit 转换
    maxd = 0.0
    for n in range(len(p2)):
        ref = float(p2[n]) * float(1 << r) * float(gsc[n]) * float(scr)
        tpu = np.float32(np.float32(np.float32(p2[n] * (1 << r)) * gsc[n]) * scr)
        d = abs(float(tpu) - ref)
        if d > maxd:
            maxd = d
    return maxd

def main():
    rms_attn, mats, _ = read_layer0()
    act_h, cases_h = parse_data_h()

    # ---- (A) 数据链一致性 ----
    print("== (A) qwen_m1_data.h vs layer0_kal.bin 数据链核对 ==")
    # activation
    emb = np.fromfile(os.path.join(W, "embed_i8.bin"), dtype=np.int8).reshape(-1, D)
    esc = np.fromfile(os.path.join(W, "embed_scales.f32"), dtype=np.float32)
    x = emb[105538].astype(np.float64) * esc[105538]
    ss = np.sqrt((x ** 2).mean() + 1e-6)
    act_ref = (x / ss * rms_attn.astype(np.float64)).astype(np.float32)
    # gen_m1_data.py 写入时 round(...,8); 这里重算再 round 对比
    act_ref_r = np.array([round(float(v), 8) for v in act_ref], dtype=np.float32)
    adiff = np.abs(act_ref_r.astype(np.float32) - act_h).max()
    print(f"  activation: max|recomputed - embedded| = {adiff:.3e}  ({'OK' if adiff < 1e-6 else 'MISMATCH'})")

    case_cfg = [("q0", "q_proj", 0, 896), ("q13", "q_proj", 13, 896), ("q27", "q_proj", 27, 896),
                ("dn0", "down_proj", 0, 896), ("up0", "up_proj", 0, 896)]
    ok_all = adiff < 1e-6
    data_tbl = {}
    for tag, name, g, nslice in case_cfg:
        nib_ref, gsc_ref = mats[name]
        nib_ref_b = nib_ref[g][:nslice].ravel()      # [896,16]
        gsc_ref_b = gsc_ref[g][:nslice].astype(np.float16).ravel()
        _, nib_h, gsc_h = [c for c in cases_h if c[0] == tag][0]
        nib_eq = np.array_equal(nib_h.ravel(), nib_ref_b)
        # gsc: data_h 存 fp16 的 uint16 bit, 与原始 fp16 比对
        gsc_h_u16 = gsc_h.astype(np.uint16)
        gsc_ref_u16 = gsc_ref_b.view(np.uint16)
        gsc_eq = np.array_equal(gsc_h_u16, gsc_ref_u16)
        nib_ok = int(nib_eq); gsc_ok = int(gsc_eq)
        ok_all &= nib_eq and gsc_eq
        print(f"  {tag:4s} ({name} g={g}): nib 逐字节 {nib_ok}/{nib_h.size} | gsc 逐字节 {gsc_ok}/{gsc_h.size}")
        data_tbl[tag] = (nib_h, gsc_h)
    print(f"  -> 数据链: {'ALL BYTE-IDENTICAL' if ok_all else 'MISMATCH'}\n")

    # ---- (B)(C) host 独立两遍法 + fp32 maxdiff ----
    print("== (B)(C) host 独立两遍法语义 vs 上板实测表 ==")
    print(f"  {'case':5s} {'wmax':>4s} {'rsafe':>5s} {'maxabs':>6s} {'r_opt':>5s} {'sat8':>4s} "
          f"{'fp32maxdiff':>12s} {'outmax':>8s}  {'判定'}")
    report = {  # REPORT_M1_PASS1 上板实测
        "q0":  (7, 5, 7, 1, 0), "q13": (7, 5, 9, 2, 0), "q27": (7, 5, 7, 1, 0),
        "dn0": (7, 5, 6, 1, 0), "up0": (7, 5, 7, 1, 0),
    }
    for tag, name, g, nslice in case_cfg:
        nib_h, gsc_h = data_tbl[tag]
        w = dequant_block(nib_h)
        wmax = int(np.abs(w).max())
        rsafe = mm_rshift_w(G, wmax) - 3
        rsafe = max(rsafe, 4)
        # activation per_row (同 gen_m1_data act)
        x = np.asarray(act_ref, dtype=np.float64)
        xi, scr = per_row_quant(x)
        koff = g * G
        left = xi[koff:koff + G].astype(np.int32)
        acc = left @ w.astype(np.int32)     # [32] x [32,896] -> [896], 精确 int
        p1 = np.array([i8_round_div(int(a), rsafe) for a in acc], dtype=np.int8)
        maxabs = int(np.abs(p1).max())
        est = maxabs * (1 << rsafe)
        r = 0
        while est > (127 << r):
            r += 1
        p2 = np.array([i8_round_div(int(a), r) for a in acc], dtype=np.int8)
        sat = int(np.sum((p2 == 127) | (p2 == -128)))
        # gsc: data_h 是 fp16 bit -> fp32
        gsc_f32 = np.asarray([np.frombuffer(np.uint16(v).tobytes(), dtype=np.float16)[0].astype(np.float32) for v in gsc_h])
        md = fp32_maxdiff(p2, r, gsc_f32, scr)
        out = p2.astype(np.float64) * (1 << r) * gsc_f32.astype(np.float64) * float(scr)
        outmax = float(np.abs(out).max())
        r_wmax, r_rsafe, r_maxabs, r_r, r_sat = report[tag]
        ok = (rsafe == r_rsafe and maxabs == r_maxabs and r == r_r and sat == r_sat)
        print(f"  {tag:5s} {wmax:4d} {rsafe:5d} {maxabs:6d} {r:5d} {sat:4d} "
              f"{md:12.3e} {outmax:8.3f}  {'OK' if ok else 'MISMATCH'}")
        if rsafe != r_rsafe or maxabs != r_maxabs or r != r_r or sat != r_sat:
            ok_all = False
    print(f"\n== 总判定: {'ALL MATCH' if ok_all else 'FAIL'} ==")
    sys.exit(0 if ok_all else 1)

if __name__ == "__main__":
    main()
