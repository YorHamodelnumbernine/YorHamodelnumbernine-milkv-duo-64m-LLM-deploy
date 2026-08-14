#!/usr/bin/env python3
"""verify_phase6_lmhead_indep.py — Phase 6 两段式 LM head 独立复核 (TPU 底层工程师).

独立于引擎 (qwen_engine_lmhead2.c) 与所有 emulator/构建脚本 (lmhead_*.py),
仅从 raw 二进制 + h/logit 缓存重新核验两段式 LM head 的数据链与语义:

  (A) 数据链 / 排列合法性:
      - row_to_tok_cl.bin 是 0..V-1 的排列 (cluster-major 是排列, 候选行 1:1 映射 token)
      - embed_i8_cl[perm] == embed_i8  (重排行与原行逐位一致)
      - embed_scales_cl[perm] == embed_scales
      - clust_idx 各簇 span 互不重叠且覆盖全部 V 行
  (B) 两段式语义 (独立 numpy 实现, f64 累加 = 与 host C lmhead_twostage_c 同口径):
      - Stage1: score[c] = h·centroid[c]; top-Kc=128 簇
      - Stage2: 读候选簇 span 精确 logits, 映射回原 token 空间, top-1
      - 断言 top-1 == g1 (full dense lg 的 argmax); P0 gap >= 0.05
  (C) 稳定性口径: f64 vs f32 累加对 top-1 的影响 (板上引擎用 f32)

用法: python3 verify_phase6_lmhead_indep.py
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.join(HERE, "weights_kal")
D, V, C = 896, 151936, 1024
Kc = 128
P0_NAMES = ["P1 中国的首都是", "P2 The capital of France is", "P3 今天天气很好，我们去公园"]

ok_all = True
def check(name, cond, detail=""):
    global ok_all
    ok_all &= bool(cond)
    print(f"  [{'PASS' if cond else '** FAIL **':>9}] {name} {detail}")

def main():
    print("==== Phase 6 LM head 独立复核 (data-chain + two-stage semantics) ====")
    t0 = __import__("time").time()

    # ---- load raw files ----
    emb_orig = np.fromfile(os.path.join(W, "embed_i8.bin"), dtype=np.int8).reshape(V, D)
    esc_orig = np.fromfile(os.path.join(W, "embed_scales.f32"), dtype=np.float32)
    emb_cl = np.fromfile(os.path.join(W, "embed_i8_cl.bin"), dtype=np.int8).reshape(V, D)
    esc_cl = np.fromfile(os.path.join(W, "embed_scales_cl.f32"), dtype=np.float32)
    tok_cl = np.fromfile(os.path.join(W, "row_to_tok_cl.bin"), dtype=np.int32)
    cent = np.fromfile(os.path.join(W, "centroid_f16.bin"), dtype=np.float16).reshape(C, D)
    clidx = np.fromfile(os.path.join(W, "clust_idx.bin"), dtype=np.int32).reshape(C, 2)
    print(f"  loaded raw files ({__import__('time').time()-t0:.1f}s)")
    print(f"  sizes: emb_orig={emb_orig.shape} esc={esc_orig.shape} emb_cl={emb_cl.shape} "
          f"tok_cl={tok_cl.shape} cent={cent.shape} clidx={clidx.shape}")

    # ---- (A) 数据链 / 排列合法性 ----
    print("\n== (A) cluster-major 排列合法性 ==")
    perm = tok_cl.astype(np.int64)
    check("row_to_tok 是 0..V-1 排列", np.array_equal(np.sort(perm), np.arange(V)))
    # 重排行校验: 用 perm 反查原表
    inv = np.empty(V, np.int64); inv[perm] = np.arange(V)
    check("embed_i8_cl 与原表逐位一致 (perm 反查)", np.array_equal(emb_cl[inv], emb_orig))
    check("embed_scales_cl 与原表逐位一致 (perm 反查)", np.array_equal(esc_cl[inv], esc_orig))
    # clust_idx span 覆盖
    offs, cnts = clidx[:, 0].astype(np.int64), clidx[:, 1].astype(np.int64)
    check("clust_idx span 覆盖 V 行 (sum cnt==V)", int(cnts.sum()) == V)
    # span 互不重叠且连续覆盖 [0,V)
    boundaries = []
    for c in range(C):
        if cnts[c] > 0:
            boundaries.append((offs[c], offs[c] + cnts[c]))
    boundaries.sort()
    contig = all(boundaries[i][1] == boundaries[i+1][0] for i in range(len(boundaries)-1))
    check("clust_idx span 互不重叠且连续", contig)
    # 簇内行: embed_i8_cl 的簇 span 内部应同属一个原始簇 (由 clidx 决定, 已由排列覆盖)

    # ---- (B) 两段式语义 (独立 numpy, f64) ----
    print("\n== (B) 两段式语义 vs full dense (f64 口径) ==")
    z = np.load(os.path.join(HERE, "lmhead_h_cache.npz"))
    # 用 meta 顺序保证与 lmhead_twostage_c 一致
    meta = json.load(open(os.path.join(HERE, "lmhead_h_meta.json")))
    names = meta["names"]; g1s = meta["g1"]
    Cf = cent.astype(np.float64)
    # 预解出 cluster-major 的 row = emb_cl*esc_cl
    rows_cl = (emb_cl.astype(np.float32) * esc_cl[:, None]).astype(np.float32)
    del emb_orig, esc_orig, emb_cl, esc_cl

    all_hit = 0; p0_hit = 0; p0_gap_min = 1e300
    for i, nm in enumerate(names):
        h = z[nm + "_h"].astype(np.float64)
        lg = z[nm + "_lg"]
        g1 = int(np.argmax(lg))
        assert g1 == g1s[i], f"meta g1 mismatch {nm}"
        # Stage1 (f64)
        sc = h @ Cf.T
        sel = np.argsort(-sc)[:Kc]
        # Stage2: 精确 logits over candidate spans
        logits = np.full(V, -np.inf, np.float64)
        for c in sel:
            o, cnt = int(clidx[c, 0]), int(clidx[c, 1])
            if cnt <= 0: continue
            logits[o:o+cnt] = rows_cl[o:o+cnt] @ h
        finite = np.isfinite(logits)
        logit_by_tok = np.full(V, -np.inf, np.float64)
        logit_by_tok[tok_cl[finite]] = logits[finite]
        top2 = np.argsort(-logit_by_tok)[:2]
        hit = (g1 == int(top2[0]))
        all_hit += hit
        cand = int(finite.sum())
        gap = float(logit_by_tok[top2[0]] - logit_by_tok[top2[1]])
        tag = " P0" if nm in P0_NAMES else ""
        if nm in P0_NAMES:
            p0_hit += hit
            p0_gap_min = min(p0_gap_min, gap)
        print(f"  [{i:2d}] {nm[:32]:32s} g1={g1:7d} top1={int(top2[0]):7d} hit={hit} "
              f"cand={cand:6d} gap={gap:.4f}{tag}")
    print(f"  -> recall_all={all_hit}/{len(names)}  P0={p0_hit}/3  P0_min_cand_gap={p0_gap_min:.4f}")
    check("recall_all == 11/11", all_hit == len(names))
    check("P0 == 3/3", p0_hit == 3)
    check("P0_min_cand_gap >= 0.05", p0_gap_min >= 0.05)

    # ---- (C) f32 vs f64 累加稳定性 (板上引擎用 f32, host C 用 f64) ----
    print("\n== (C) f32 vs f64 累加对 top-1 稳定性 (板上引擎口径) ==")
    Cf32 = cent.astype(np.float32)
    for i, nm in enumerate(names[:3]):
        h32 = z[nm + "_h"]
        lg = z[nm + "_lg"]; g1 = int(np.argmax(lg))
        # stage1 f32 (板上 float dot)
        sc32 = h32.astype(np.float32) @ Cf32.T
        sel32 = np.argsort(-sc32)[:Kc]
        # stage2 f32 (板上 float logits)
        logits32 = np.full(V, -np.inf, np.float32)
        for c in sel32:
            o, cnt = int(clidx[c, 0]), int(clidx[c, 1])
            if cnt <= 0: continue
            logits32[o:o+cnt] = rows_cl[o:o+cnt] @ h32.astype(np.float32)
        finite = np.isfinite(logits32)
        lt32 = np.full(V, -np.inf, np.float32)
        lt32[tok_cl[finite]] = logits32[finite]
        t32 = np.argsort(-lt32)[:2]
        gap32 = float(lt32[t32[0]] - lt32[t32[1]])
        print(f"  [{i}] {nm[:24]:24s} f64top1={g1} f32top1={int(t32[0])} f32gap={gap32:.4f} "
              f"stable={'Y' if g1==int(t32[0]) else 'N'}")
        check(f"f32 top-1 稳定 ({nm[:16]})", g1 == int(t32[0]))

    print(f"\n==== 判定: {'PASS' if ok_all else '** FAIL **'}  ({(__import__('time').time()-t0):.1f}s) ====")
    sys.exit(0 if ok_all else 1)

if __name__ == "__main__":
    main()
