#!/usr/bin/env python3
"""Phase 6 · LM head MIPS 检索方法对比 (host, 复用 lmhead_h_cache.npz).

目标: 对比候选检索方法在「候选集大小 (->SD 读) → recall@1」上的效率, 为两段式
LM head 定标. 方法:
  A 欧氏 k-means (lmhead_topk_spike 现实现, 基线)
  B MDP/球面 k-means (按方向聚类, max-dot 分配, 单位 centroid)
  C 随机投影精确 top-B (rows 投影到 dim 维, Stage1 全量得分, top-B 精算)
  D 范数排序上界 (||row[t]|| 降序取 top-B, 最廉价)

指标: recall@1 (gold top-1 ∈ 候选) + 候选集大小 -> SD 读 (MB) / @21.5MB/s.
全 11 prompt (3 P0 + 8 eval); 重点看 P0 (门禁) 与 11-prompt 总召回.
"""
import os, sys, time, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.join(HERE, "weights_kal")
CACHE = os.path.join(HERE, "lmhead_h_cache.npz")
sys.path.insert(0, HERE)
import lmhead_topk_spike as base   # kmeans/assign 复用

D, V = base.D, base.V
P0_NAMES = ["P1 中国的首都是", "P2 The capital of France is", "P3 今天天气很好，我们去公园"]


# ---------------- 加载缓存 h / lg ----------------
def load_prompts():
    z = np.load(CACHE)
    names, Hs, LGs = [], [], []
    for k in z.files:
        if k.endswith("_h"):
            nm = k[:-2]
            names.append(nm)
    # 按 P0 优先 + 其余
    def key(nm):
        return 0 if nm in P0_NAMES else 1
    names.sort(key=key)
    for nm in names:
        Hs.append(z[nm + "_h"].astype(np.float64))
        LGs.append(z[nm + "_lg"])
    return names, Hs, LGs


def recall_stats(Hs, LGs, gold_top1s, cand_for_prompt):
    """cand_for_prompt(p, K) -> set of candidate tokens (or (top-K ids) for top-B)."""
    pass


# ---------------- A: 欧氏 k-means ----------------
def run_euclid(rows, C, Hs, LGs):
    t0 = time.time()
    Cm = base.kmeans(rows, C, iters=6, seed=7, subsample=12288)
    cl = base.assign(rows, Cm)
    print(f"  A 欧氏 C={C}: train+assign {time.time()-t0:.1f}s")
    out = {}
    for i, (h, lg) in enumerate(zip(Hs, LGs)):
        g1 = int(np.argmax(lg))
        sc = h @ Cm.astype(np.float64).T
        top = np.argsort(-sc)
        mem = cl  # 成员按簇
        # precompute per-cluster member sets
        out[i] = {"g1": g1, "top": top}
    # 预计算簇成员
    import collections
    members = collections.defaultdict(list)
    for t in range(V):
        members[int(cl[t])].append(t)
    mem_sets = {c: set(m) for c, m in members.items()}
    sizes = np.bincount(cl, minlength=C)
    return {"kind": "euclid", "mem_sets": mem_sets, "sizes": sizes, "top": [out[i]["top"] for i in out],
            "g1": [out[i]["g1"] for i in out]}


# ---------------- B: MDP / 球面 k-means ----------------
def kmeans_mdp(rows, C, iters=6, seed=0, subsample=12288):
    rng = np.random.default_rng(seed)
    X = rows[:subsample].astype(np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    Xu = X / np.maximum(norms, 1e-8)
    S = Xu.shape[0]
    first = int(rng.integers(0, S))
    Cm = Xu[first:first + 1].copy()
    mindot = Xu @ Cm.T[:, 0]
    for _ in range(1, C):
        d = np.maximum(1.0 - mindot, 1e-6)
        p = d / d.sum()
        pick = int(rng.choice(S, p=p))
        Cm = np.vstack([Cm, Xu[pick:pick + 1]])
        nd = Xu @ Xu[pick:pick + 1].T
        mindot = np.maximum(mindot, nd[:, 0])
    for it in range(iters):
        sim = Xu @ Cm.T
        lab = sim.argmax(1)
        newC = np.zeros_like(Cm); cnt = np.zeros(C, np.int64)
        for c in range(C):
            m = np.where(lab == c)[0]
            if m.size:
                newC[c] = Xu[m].mean(0); cnt[c] = m.size
        nrm = np.linalg.norm(newC, axis=1, keepdims=True)
        newC = newC / np.maximum(nrm, 1e-8)
        empty = np.where(cnt == 0)[0]
        for c in empty:
            j = int(rng.integers(0, S)); newC[c] = Xu[j]
        Cm = newC
    return Cm.astype(np.float32)


def run_mdp(rows, C, Hs, LGs):
    t0 = time.time()
    Cm = kmeans_mdp(rows, C, iters=6, seed=7, subsample=12288)
    # MDP assign: argmax_c row·centroid (分块)
    Xf = rows.astype(np.float32)
    n = rows.shape[0]
    cl = np.empty(n, np.int64)
    for i in range(0, n, 4096):
        cl[i:i + 4096] = (Xf[i:i + 4096] @ Cm.T).argmax(1)
    print(f"  B MDP C={C}: train+assign {time.time()-t0:.1f}s")
    import collections
    members = collections.defaultdict(list)
    for t in range(V):
        members[int(cl[t])].append(t)
    mem_sets = {c: set(m) for c, m in members.items()}
    sizes = np.bincount(cl, minlength=C)
    top = []
    g1s = []
    for h, lg in zip(Hs, LGs):
        sc = h @ Cm.astype(np.float64).T
        top.append(np.argsort(-sc))
        g1s.append(int(np.argmax(lg)))
    return {"kind": "mdp", "mem_sets": mem_sets, "sizes": sizes, "top": top, "g1": g1s}


# ---------------- C: 随机投影 top-B ----------------
def run_proj(rows, Hs, LGs, dim):
    t0 = time.time()
    rng = np.random.default_rng(0)
    Wp = (rng.normal(0.0, 1.0 / np.sqrt(dim), (D, dim))).astype(np.float32)
    Rp = (rows.astype(np.float32) @ Wp).astype(np.float32)   # [V, dim]
    print(f"  C 随机投影 dim={dim}: proj {time.time()-t0:.1f}s")
    out = []
    for h, lg in zip(Hs, LGs):
        hproj = (h.astype(np.float32) @ Wp).astype(np.float32)
        score = Rp @ hproj      # [V]
        top = np.argsort(-score)
        out.append(top)
    return {"kind": "proj", "top": out, "g1": [int(np.argmax(lg)) for lg in LGs]}


# ---------------- D: 范数排序 top-B ----------------
def run_norm(rows, Hs, LGs):
    norms = np.linalg.norm(rows, axis=1)
    top = np.argsort(-norms)
    return {"kind": "norm", "top": top, "g1": [int(np.argmax(lg)) for lg in LGs]}


# ---------------- 评估 ----------
def eval_cluster(res, Kc_list):
    """候选集 = top-Kc 簇成员并集."""
    rows_out = []
    for pi in range(len(res["g1"])):
        g1 = res["g1"][pi]
        topK = res["top"][pi]
        # 从 Kc=1..max 累加
        cur = set()
        acc = []
        for Kc in range(1, max(Kc_list) + 1):
            c = int(topK[Kc - 1])
            cur |= res["mem_sets"][c]
            if Kc in Kc_list:
                acc.append((Kc, g1 in cur, len(cur)))
        rows_out.append(acc)
    return rows_out


def eval_topB(res, B_list):
    rows_out = []
    for pi in range(len(res["g1"])):
        g1 = res["g1"][pi]
        top = res["top"] if res["kind"] == "norm" else res["top"][pi]
        acc = []
        for B in B_list:
            cand = set(top[:B].tolist())
            acc.append((B, g1 in cand, len(cand)))
        rows_out.append(acc)
    return rows_out


def main():
    names, Hs, LGs = load_prompts()
    n = len(names)
    print(f"[cmp] {n} prompts (P0 first): {names}")
    embed = np.fromfile(os.path.join(W, "embed_i8.bin"), dtype=np.int8).reshape(V, D)
    esc = np.fromfile(os.path.join(W, "embed_scales.f32"), dtype=np.float32)
    rows = (embed.astype(np.float32) * esc[:, None]).astype(np.float32)
    del embed, esc

    Kc_list = [8, 16, 32, 64, 128, 256, 512]
    B_list = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    results = {}

    print("\n== A: 欧氏 k-means ==", flush=True)
    ra = run_euclid(rows, 1024, Hs, LGs)
    aa = eval_cluster(ra, Kc_list)
    results["A_euclid"] = (ra, aa, "cand_size", "Kc")

    print("\n== B: MDP 球面 k-means ==", flush=True)
    rb = run_mdp(rows, 1024, Hs, LGs)
    ab = eval_cluster(rb, Kc_list)
    results["B_mdp"] = (rb, ab, "cand_size", "Kc")

    print("\n== C: 随机投影 top-B ==", flush=True)
    rc = run_proj(rows, Hs, LGs, 64)
    ac = eval_topB(rc, B_list)
    results["C_proj64"] = (rc, ac, "B", "B")

    print("\n== D: 范数排序 top-B ==", flush=True)
    rd = run_norm(rows, Hs, LGs)
    ad = eval_topB(rd, B_list)
    results["D_norm"] = (rd, ad, "B", "B")

    # ---- 汇总表: recall@1(全部) + recall@1(P0) + avg cand_size + SD 延迟 ----
    print("\n== 汇总: recall@1 全 11 / P0 3 / avg cand_size -> SD 延迟 ==", flush=True)
    for meth, (res, acc, sz_label, x_label) in results.items():
        print(f"\n[{meth}]", flush=True)
        ks = [a[0] for a in acc[0]]   # Kc or B
        hdr = f"{x_label:>6s} " + "".join(f"{k:>6d}" for k in ks)
        print(hdr, flush=True)
        for pi in range(n):
            row = f"{acc[pi][0][0]:>6d} " + "".join(
                f"{'1' if a[1] else '.'}{a[2]:>5d}" for a in acc[pi])
            tag = " P0" if names[pi] in P0_NAMES else ""
            print(row + tag, flush=True)
        # 逐 K 汇总
        for idx in range(len(ks)):
            k = ks[idx]
            r_all = sum(1 for pi in range(n) if acc[pi][idx][1])
            r_p0 = sum(1 for pi in range(n) if names[pi] in P0_NAMES and acc[pi][idx][1])
            avg_sz = int(np.mean([acc[pi][idx][2] for pi in range(n)]))
            mb = avg_sz * D / 1e6; dt = mb / 21.5
            print(f"  {k:>6d}: recall_all={r_all:2d}/{n}  P0={r_p0:1d}/3  cand~{avg_sz:7d} = {mb:7.2f}MB = {dt:5.2f}s", flush=True)

    # 存 json
    ser = {}
    for meth, (res, acc, sz_label, x_label) in results.items():
        ks = [a[0] for a in acc[0]]
        ser[meth] = {"kind": res["kind"], "xs": ks,
                     "per_prompt": {names[pi]: [
                         {"x": acc[pi][idx][0], "recall1": bool(acc[pi][idx][1]),
                          "cand_size": acc[pi][idx][2]} for idx in range(len(ks))]
                         for pi in range(n)}}
    with open(os.path.join(HERE, "lmhead_mips_cmp.json"), "w") as f:
        json.dump(ser, f, indent=1)
    print(f"\n[cmp] wrote lmhead_mips_cmp.json", flush=True)


if __name__ == "__main__":
    main()
