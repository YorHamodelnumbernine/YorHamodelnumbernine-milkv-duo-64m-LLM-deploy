#!/usr/bin/env python3
"""Phase 6 · LM head top-k 两段式 spike (host, Path A weights_kal/).

量化判定: 两段式 LM head (k-means 聚类 → top-Kc 簇 → 簇成员精算 exact logits)
在 3 P0 prompt 上能否保住 NEXT 3/3 + min gap>=0.05, 并定标 Kc (候选集大小) 与召回余量.

结构:
  Stage 1: score[c] = h · centroid[c]  (C x 896, trivial) -> top-Kc 簇
  Stage 2: 精确 logits = h · row[t], 对 t in union(members of top-Kc clusters)
           SD 读 = candidate_size x 896 B @ 21.5MB/s (现全量 136MB/6.12s)

关键性质: 若 gold top-1 ∈ 候选集, 候选 gap = logit(top1) - logit(次佳候选)
>= true gap (top-2 被剪时 gap 只会更大) -> 门禁 gap 项自动满足, 唯一约束是 recall@1.

方法 (host, 纯 numpy, 无 sklearn/faiss):
  1. 24 层前向 (复用 verify_m2_24l_indep 语义) 对 P0 + eval prompts 产 final h + gold logits.
     结果缓存到 lmhead_h_cache.npz, 重跑直接复用.
  2. embed 行 R[t] = embed_i8[t]*esc[t] 做 k-means (增量 kmeans++ init + Lloyd, 全量分配).
  3. 扫 Kc -> recall@1 / top-5 recall / 候选集大小 / Stage2 SD 延迟曲线.
"""
import os, sys, time, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.join(HERE, "weights_kal")
CACHE = os.path.join(HERE, "lmhead_h_cache.npz")
sys.path.insert(0, HERE)

import verify_m2_24l_indep as vfy   # 复用其数值原语 (Path A 两遍法语义, 3/3 已验证)

D, V = vfy.D, vfy.V


# ---------------- 1. 前向 (返回 final h 与 gold logits) ----------------
def forward_h(tokens):
    """同 vfy.forward, 但返回 (h, logits). h = final rms_norm(x[seq-1]) [D]."""
    seq = len(tokens)
    embed = np.fromfile(os.path.join(W, "embed_i8.bin"), dtype=np.int8).reshape(V, D)
    esc = np.fromfile(os.path.join(W, "embed_scales.f32"), dtype=np.float32)
    frms = np.fromfile(os.path.join(W, "final_rms.f32"), dtype=np.float32)
    cos, sin = vfy.precompute_rope(max(vfy.MAX_SEQ, seq + 8))
    x = np.zeros((seq, D), np.float32)
    for i, t in enumerate(tokens):
        t = 0 if (t < 0 or t >= V) else t
        x[i] = (embed[t].astype(np.float32) * esc[t]).astype(np.float32)

    for l in range(vfy.L):
        rms_attn, nibs, gscs, rms_ffn = vfy.load_layer(l)
        bias = vfy.load_bias(l)
        bq, bk, bv = bias[:D], bias[D:D + vfy.DKV], bias[D + vfy.DKV:]
        h = vfy.rms_norm(x, rms_attn)
        xi, scr = vfy.per_row_quant(h)
        q = vfy.matmul_twopass(xi, vfy.dequant_nib(nibs["Wq"]), gscs["Wq"], scr) + bq
        k = vfy.matmul_twopass(xi, vfy.dequant_nib(nibs["Wk"]), gscs["Wk"], scr) + bk
        v = vfy.matmul_twopass(xi, vfy.dequant_nib(nibs["Wv"]), gscs["Wv"], scr) + bv
        q = q.reshape(seq, vfy.Hn, vfy.HD).copy()
        k = k.reshape(seq, vfy.KVH, vfy.HD).copy()
        v = v.reshape(seq, vfy.KVH, vfy.HD).copy()
        for s in range(seq):
            for hh in range(vfy.Hn):
                vfy.rope_inplace(q[s, hh], s, cos, sin)
            for hh in range(vfy.KVH):
                vfy.rope_inplace(k[s, hh], s, cos, sin)
        attn = vfy.attention(q, k, v, seq)
        ai, sca = vfy.per_row_quant(attn)
        oout = vfy.matmul_twopass(ai, vfy.dequant_nib(nibs["Wo"]), gscs["Wo"], sca)
        x = (x + oout).astype(np.float32)
        h = vfy.rms_norm(x, rms_ffn)
        xi, scr = vfy.per_row_quant(h)
        up = vfy.matmul_twopass(xi, vfy.dequant_nib(nibs["up"]), gscs["up"], scr)
        gate = vfy.matmul_twopass(xi, vfy.dequant_nib(nibs["gate"]), gscs["gate"], scr)
        mid = (up * vfy.silu(gate)).astype(np.float32)
        dn = vfy.dequant_nib(nibs["down"]); dn_g = gscs["down"]
        oout = np.zeros((seq, D), np.float32)
        for kc in range(0, vfy.F, 1024):
            kcn = min(1024, vfy.F - kc)
            mch = mid[:, kc:kc + kcn].copy()
            m_i8, sc_m = vfy.per_row_quant(mch)
            wch = dn[kc:kc + kcn, :]
            gw = dn_g[kc // vfy.G:(kc + kcn) // vfy.G, :]
            oout += vfy.matmul_twopass(m_i8, wch, gw, sc_m)
        x = (x + oout).astype(np.float32)

    h = vfy.rms_norm(x[seq - 1:seq], frms)[0]
    row = (embed.astype(np.float32) * esc[:, None])
    lg = h.astype(np.float64) @ row.astype(np.float64).T
    return h, lg


# ---------------- 2. k-means (纯 numpy) ----------------
def _dist2(X, C):
    """[n,D] x [C,D] -> [n,C] squared euclidean (float32, clip >=0)."""
    Xf = X.astype(np.float32); Cf = C.astype(np.float32)
    Xn = (Xf * Xf).sum(1)
    Cn = (Cf * Cf).sum(1)
    return np.maximum(Xn[:, None] + Cn[None, :] - 2.0 * (Xf @ Cf.T), 0.0)


def kmeans(rows, C, iters=8, seed=0, subsample=16384):
    rng = np.random.default_rng(seed)
    n = rows.shape[0]
    sub_idx = rng.choice(n, size=min(subsample, n), replace=False)
    X = rows[sub_idx].astype(np.float32)
    S = X.shape[0]
    # ---- 增量 kmeans++ init (O(C·S·D), 非 O(C²·S·D)) ----
    first = int(rng.integers(0, S))
    Cmlist = [X[first].copy()]
    mind = _dist2(X, X[first:first + 1])[:, 0]
    for _ in range(1, C):
        s = mind.sum()
        p = (mind / s) if (s > 0 and np.isfinite(s)) else np.full(S, 1.0 / S)
        pick = int(rng.choice(S, p=p))
        Cmlist.append(X[pick].copy())
        nd = _dist2(X, X[pick:pick + 1])[:, 0]
        mind = np.minimum(mind, nd)
    Cm = np.array(Cmlist)
    # ---- Lloyd ----
    for it in range(iters):
        lab = _dist2(X, Cm).argmin(1)
        newC = np.zeros_like(Cm); cnt = np.zeros(C, np.int64)
        for c in range(C):
            m = np.where(lab == c)[0]
            if m.size:
                newC[c] = X[m].mean(0); cnt[c] = m.size
        empty = np.where(cnt == 0)[0]
        for c in empty:
            j = int(rng.integers(0, S)); newC[c] = X[j]
        Cm = newC
    return Cm.astype(np.float32)


def assign(rows, Cm, chunk=4096):
    """rows [V,D] -> cl [V] nearest centroid id (分块, 控内存)."""
    n = rows.shape[0]; C = Cm.shape[0]
    cl = np.empty(n, np.int64)
    Xf = rows.astype(np.float32)
    Cn = (Cm.astype(np.float32) ** 2).sum(1)
    for i in range(0, n, chunk):
        Xb = Xf[i:i + chunk]
        Xn = (Xb ** 2).sum(1)[:, None]
        d2 = np.maximum(Xn + Cn[None, :] - 2.0 * (Xb @ Cm.T), 0.0)
        cl[i:i + chunk] = d2.argmin(1)
    return cl


# ---------------- 3. 评估 ----------------
def evaluate(h, lg, Cm, cl, Kc_list):
    """单 prompt: (per-Kc {recall1, top5, cand_size}, gold_top1, gold_top5, g1_cluster, cluster_rank)."""
    gold_top1 = int(np.argmax(lg))
    gold_top5 = np.argsort(-lg)[:5].tolist()
    sc = h.astype(np.float64) @ Cm.astype(np.float64).T          # [C]
    top = np.argsort(-sc)[:max(Kc_list)].tolist()
    g1c = int(cl[gold_top1])
    rank = int((sc > sc[g1c]).sum())
    sizes = np.bincount(cl, minlength=Cm.shape[0])
    res = {}
    for Kc in Kc_list:
        sel = top[:Kc]
        cand = np.concatenate([np.where(cl == c)[0] for c in sel])
        cand_set = set(cand.tolist())
        res[Kc] = {"recall1": gold_top1 in cand_set,
                   "top5": sum(t in cand_set for t in gold_top5),
                   "cand_size": len(cand_set)}
    return res, gold_top1, gold_top5, g1c, rank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cs", default="512,1024")
    ap.add_argument("--kcs", default="4,8,16,32,64,128,256")
    ap.add_argument("--subsample", type=int, default=12288)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "lmhead_topk.json"))
    args = ap.parse_args()

    Cs = [int(x) for x in args.cs.split(",")]
    Kc_list = [int(x) for x in args.kcs.split(",")]

    P0 = [
        ("P1 中国的首都是", [105538, 59975, 100132]),
        ("P2 The capital of France is", [785, 6722, 315, 9625, 374]),
        ("P3 今天天气很好，我们去公园", [100644, 104307, 101243, 3837, 97639, 85336, 102077]),
    ]
    EVAL_TEXTS = [
        "The capital of Japan is",
        "Python is a programming language used for",
        "The largest ocean on Earth is the",
        "Who invented the telephone?",
        "人工智能的未来发展方向是",
        "上海是中国最大的城市",
        "量子计算的基本原理是",
        "光合作用的过程是",
    ]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.path.join(HERE, "..", "qwen_int4", "model"),
                                        trust_remote_code=True)
    Eval = [(t, tok.encode(t, add_special_tokens=False)) for t in EVAL_TEXTS]
    prompts = P0 + Eval
    names = [n for n, _ in prompts]
    print(f"[spike] prompts: {len(P0)} P0 + {len(Eval)} eval", flush=True)

    # ---- 1. h + gold (带缓存) ----
    Hs = []; LGs = []
    cache = {}
    if not args.no_cache and os.path.exists(CACHE):
        z = np.load(CACHE)
        cache = {k: (z[k + "_h"], z[k + "_lg"], z[k + "_toks"]) for k in z.files if k.endswith("_h")}
        print(f"[spike] cache loaded ({len(cache)} prompts)", flush=True)
    print("[spike] forward (24 layers, 每层读 weights_kal) ...", flush=True)
    t0 = time.time()
    for i, (name, toks) in enumerate(prompts):
        if name in cache:
            h, lg = cache[name][0], cache[name][1]
        else:
            h, lg = forward_h(toks)
            cache[name] = (h, lg, np.asarray(toks, np.int32))
        Hs.append(h); LGs.append(lg)
        t1 = int(np.argmax(lg)); gap = float(lg[t1] - np.sort(lg)[-2])
        print(f"  [{i}] {name[:38]:38s} ntok={len(toks):2d} gold_top1={t1:7d} gap={gap:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    if not args.no_cache:
        arr = {}
        for k, (h, lg, toks) in cache.items():
            arr[k + "_h"] = h.astype(np.float32); arr[k + "_lg"] = lg.astype(np.float64)
            arr[k + "_toks"] = toks
        np.savez(CACHE, **arr)
        print(f"[spike] cache saved {CACHE}", flush=True)
    print(f"[spike] forward done in {time.time()-t0:.1f}s", flush=True)

    # ---- 2. embed rows ----
    embed = np.fromfile(os.path.join(W, "embed_i8.bin"), dtype=np.int8).reshape(V, D)
    esc = np.fromfile(os.path.join(W, "embed_scales.f32"), dtype=np.float32)
    rows = (embed.astype(np.float32) * esc[:, None]).astype(np.float32)
    del embed, esc

    # ---- 3. cluster + evaluate ----
    out = {"subsample": args.subsample, "iters": args.iters, "seed": args.seed,
           "prompts": [{"name": n, "tokens": toks, "gold_top1": int(np.argmax(lg))}
                       for (n, toks), lg in zip(prompts, LGs)],
           "results": {}}
    for C in Cs:
        print(f"\n[spike] k-means C={C} (sub={args.subsample}, iters={args.iters}) ...", flush=True)
        t0 = time.time()
        Cm = kmeans(rows, C, iters=args.iters, seed=args.seed, subsample=args.subsample)
        print(f"[spike]   train {time.time()-t0:.1f}s; assign {V} rows ...", flush=True)
        t1 = time.time()
        cl = assign(rows, Cm)
        print(f"[spike]   assign {time.time()-t1:.1f}s; evaluate {len(prompts)} prompts ...", flush=True)
        C_res = {}
        for i, (h, lg, nm) in enumerate(zip(Hs, LGs, names)):
            res, g1, g5, g1c, rank = evaluate(h, lg, Cm, cl, Kc_list)
            C_res[nm] = {"gold_top1": g1, "g1_cluster": g1c, "g1_cluster_rank": rank,
                         "kcs": {str(Kc): r for Kc, r in res.items()}}
            print(f"  [{i}] {nm[:36]:36s} g1_cluster={g1c:6d} rank={rank:4d}/{C}", flush=True)
        out["results"][str(C)] = {"per_prompt": C_res}
        print(f"[spike] C={C} done ({time.time()-t0:.1f}s total)", flush=True)

    # ---- 4. 汇总 ----
    print(f"\n[spike] == 汇总: recall@1 (P0+eval {len(prompts)} prompts) ==", flush=True)
    print("Kc    " + "".join(f"{('C='+str(C)):>16s}" for C in Cs), flush=True)
    recap = {}
    for Kc in Kc_list:
        row = f"{Kc:<6d}"
        for C in Cs:
            r1 = [out["results"][str(C)]["per_prompt"][n]["kcs"][str(Kc)]["recall1"] for n in names]
            sz = [out["results"][str(C)]["per_prompt"][n]["kcs"][str(Kc)]["cand_size"] for n in names]
            avg_sz = int(np.mean(sz)); recall = sum(r1)
            row += f"  {recall}/{len(names)} sz={avg_sz:6d}"
            recap[(C, Kc)] = (recall, avg_sz)
        print(row, flush=True)
    print(f"\n[spike] == Stage-2 SD 延迟 (cand_size x 896B @ 21.5MB/s; 现全量 136.1MB=6.12s) ==", flush=True)
    for C in Cs:
        for Kc in Kc_list:
            _, avg_sz = recap[(C, Kc)]
            mb = avg_sz * D / 1e6; dt = mb / 21.5
            if dt <= 2.0:
                print(f"  C={C:5d} Kc={Kc:4d}: cand~{avg_sz:7d} rows = {mb:6.2f}MB = {dt:5.2f}s", flush=True)

    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n[spike] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
