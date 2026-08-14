#!/usr/bin/env python3
"""Phase 6 · LM head 两段式离线构建 (host).

输入: weights_kal/embed_i8.bin + embed_scales.f32 (V=151936, D=896)
产出 (均入 weights_kal/, 供 qwen_engine_24l.c Stage1/Stage2 消费):
  embed_i8_cl.bin     [V, D] int8  cluster-major (按 MDP 簇 id 排序)
  embed_scales_cl.f32 [V]    fp32  cluster-major (与 embed_i8_cl 行对齐)
  row_to_tok_cl.bin   [V]    int32 reordered 行下标 -> 原 token id
  centroid_f16.bin    [C, D] fp16  MDP 单位 centroid (Stage1 用)
  clust_idx.bin       [C, 2] int32 (offset, count) 每簇在 cluster-major 中的 span
  clust_meta.json             C / D / V / Kc 标定信息

验证: 用缓存 11 prompt 的 h/gold, 模拟 Stage1(top-Kc 簇) + Stage2(span 精确 logits),
      断言 P0 3/3 + 候选 gap >= 0.05 + 全 11 recall@1 (默认 C=1024, Kc=128)。
"""
import os, sys, time, json, argparse
import numpy as np
import collections

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.join(HERE, "weights_kal")
CACHE = os.path.join(HERE, "lmhead_h_cache.npz")
sys.path.insert(0, HERE)
import lmhead_mips_cmp as cmp   # kmeans_mdp
import lmhead_topk_spike as base

D, V = base.D, base.V
P0_NAMES = ["P1 中国的首都是", "P2 The capital of France is", "P3 今天天气很好，我们去公园"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--C", type=int, default=1024)
    ap.add_argument("--kc", type=int, default=128)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=W)
    args = ap.parse_args()
    C, Kc = args.C, args.kc

    t0 = time.time()
    embed = np.fromfile(os.path.join(W, "embed_i8.bin"), dtype=np.int8).reshape(V, D)
    esc = np.fromfile(os.path.join(W, "embed_scales.f32"), dtype=np.float32)
    print(f"[build] embed loaded {V}x{D} ({time.time()-t0:.1f}s)", flush=True)

    # ---- 1. MDP 聚类 (确定性 seed) ----
    print(f"[build] MDP k-means C={C} (seed={args.seed}) ...", flush=True)
    rows = (embed.astype(np.float32) * esc[:, None]).astype(np.float32)
    Cm = cmp.kmeans_mdp(rows, C, iters=6, seed=args.seed, subsample=12288)
    Xf = rows.astype(np.float32)
    cl = np.empty(V, np.int64)
    for i in range(0, V, 4096):
        cl[i:i + 4096] = (Xf[i:i + 4096] @ Cm.T).argmax(1)
    del rows
    print(f"[build]   assign done ({time.time()-t0:.1f}s)", flush=True)

    # ---- 2. cluster-major 排序 ----
    order = np.argsort(cl, kind="stable")
    cl_sorted = cl[order]
    offs = [0]; cur = cl_sorted[0]
    for i in range(1, V):
        if cl_sorted[i] != cur:
            offs.append(i); cur = cl_sorted[i]
    offs.append(V)
    # 每簇 span (order 中该簇成员的起止)
    clust_idx = np.zeros((C, 2), np.int32)
    pos_of = np.empty(V, np.int64)  # order 逆映射: 原行 -> reordered 位置
    pos_of[order] = np.arange(V)
    for c in range(C):
        m = np.where(cl == c)[0]
        if m.size == 0:
            clust_idx[c] = (0, 0); continue
        p = pos_of[m]
        clust_idx[c] = (int(p.min()), int(p.max() - p.min() + 1))
    assert clust_idx[:, 1].sum() == V, "span counts must cover all rows"

    # 重排写文件
    emb_cl = embed[order].copy()
    esc_cl = esc[order].copy()
    tok_cl = order.astype(np.int32)
    emb_cl.tofile(os.path.join(args.out, "embed_i8_cl.bin"))
    esc_cl.tofile(os.path.join(args.out, "embed_scales_cl.f32"))
    tok_cl.tofile(os.path.join(args.out, "row_to_tok_cl.bin"))
    Cm.astype(np.float16).tofile(os.path.join(args.out, "centroid_f16.bin"))
    clust_idx.tofile(os.path.join(args.out, "clust_idx.bin"))
    print(f"[build] wrote cluster-major files ({time.time()-t0:.1f}s)", flush=True)

    # ---- 3. 验证 (Stage1+Stage2 模拟, 用缓存 h/gold) ----
    names, Hs, LGs = cmp.load_prompts()
    n = len(names)
    print(f"[build] verify: {n} prompts, C={C} Kc={Kc} ...", flush=True)
    Ct = Cm.astype(np.float64)
    all_hit = 0; p0_hit = 0; p0_gap_min = 1e9
    rows_cl = (emb_cl.astype(np.float32) * esc_cl[:, None]).astype(np.float32)
    for i, (h, lg) in enumerate(zip(Hs, LGs)):
        g1 = int(np.argmax(lg))
        sc = h.astype(np.float64) @ Ct.T                    # [C]
        sel = np.argsort(-sc)[:Kc]
        # Stage2: 读各候选簇 span, 精确 logits
        logits = np.full(V, -np.inf, np.float64)
        for c in sel:
            o, cnt = int(clust_idx[c, 0]), int(clust_idx[c, 1])
            if cnt == 0: continue
            logits[o:o + cnt] = rows_cl[o:o + cnt] @ h.astype(np.float64)  # 原始行序, 未映射 token
        finite = np.isfinite(logits)
        # 映射回原 token 空间: reordered 行下标 -> 原 token id (每 tok 恰一行)
        logit_by_tok = np.full(V, -np.inf, np.float64)
        logit_by_tok[tok_cl[finite]] = logits[finite]
        top = np.argsort(-logit_by_tok)[:2]
        hit = g1 == int(top[0])
        all_hit += hit
        if names[i] in P0_NAMES:
            p0_hit += hit
            p0_gap_min = min(p0_gap_min, float(logit_by_tok[top[0]] - logit_by_tok[top[1]]))
        cand_rows = int(finite.sum())
        print(f"  [{i}] {names[i][:32]:32s} g1={g1:7d} top1={int(top[0]):7d} hit={hit} "
              f"cand_rows={cand_rows:6d} gap={float(logit_by_tok[top[0]]-logit_by_tok[top[1]]):.4f}", flush=True)
    print(f"[build] recall_all={all_hit}/{n}  P0={p0_hit}/3  P0_min_cand_gap={p0_gap_min:.4f}", flush=True)
    ok = (all_hit == n and p0_hit == 3 and p0_gap_min >= 0.05)
    print(f"[build] 判定: {'PASS' if ok else '** FAIL **'}", flush=True)

    meta = {"C": C, "D": D, "V": V, "Kc": Kc, "seed": args.seed,
            "recall_all": int(all_hit), "recall_P0": int(p0_hit),
            "P0_min_cand_gap": float(p0_gap_min), "pass": bool(ok),
            "centroid_bytes": int(Cm.nbytes // 2 * 2), "clust_idx_bytes": int(clust_idx.nbytes)}
    with open(os.path.join(HERE, "clust_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[build] wrote clust_meta.json {meta}", flush=True)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
