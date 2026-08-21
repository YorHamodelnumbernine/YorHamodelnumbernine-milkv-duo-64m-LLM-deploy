#!/usr/bin/env python3
"""Phase 6 · MDP 召回率-延迟曲线 (细扫 Kc, 聚焦 C=1024 主工作点).

CEO 要求: "K 从 64 起扫, 产出召回率-延迟曲线 + K 定标建议; P3 显式覆盖."
产出: C=1024 下 Kc∈{64,80,96,112,128,160,192,256} 的
      recall@1 (全11 / P0 3) / avg cand_size / MB / SD 理论延迟 / P3 显式命中与候选 gap.
"""
import os, sys, time, json, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.join(HERE, "weights_kal")
CACHE = os.path.join(HERE, "lmhead_h_cache.npz")
sys.path.insert(0, HERE)
import lmhead_mips_cmp as cmp
import lmhead_topk_spike as base

D, V = base.D, base.V
P0_NAMES = ["P1 中国的首都是", "P2 The capital of France is", "P3 今天天气很好，我们去公园"]

names, Hs, LGs = cmp.load_prompts()
n = len(names)
print(f"[kcurve] {n} prompts", flush=True)

embed = np.fromfile(os.path.join(W, "embed_i8.bin"), dtype=np.int8).reshape(V, D)
esc = np.fromfile(os.path.join(W, "embed_scales.f32"), dtype=np.float32)
rows = (embed.astype(np.float32) * esc[:, None]).astype(np.float32)
del embed, esc

C = 1024
t0 = time.time()
Cm = cmp.kmeans_mdp(rows, C, iters=6, seed=7, subsample=12288)
Xf = rows.astype(np.float32)
cl = np.empty(V, np.int64)
for i in range(0, V, 4096):
    cl[i:i + 4096] = (Xf[i:i + 4096] @ Cm.T).argmax(1)
sizes = np.bincount(cl, minlength=C)
members = collections.defaultdict(list)
for t in range(V):
    members[int(cl[t])].append(t)
print(f"[kcurve] C={C} train+assign {time.time()-t0:.1f}s", flush=True)

# 预计算每个 prompt 的簇得分排序 (一次, 供所有 Kc 复用)
prep = []
for i, (h, lg) in enumerate(zip(Hs, LGs)):
    sc = h.astype(np.float64) @ Cm.astype(np.float64).T
    top = np.argsort(-sc)
    g1 = int(np.argmax(lg))
    g1c = int(cl[g1])
    rank = int((sc > sc[g1c]).sum())
    prep.append({"top": top, "g1": g1, "g1c": g1c, "rank": rank, "lg": lg})
    print(f"  [{i}] {names[i][:36]:36s} g1={g1:7d} g1_cluster={g1c:6d} rank={rank:4d}/{C}", flush=True)

Kc_list = [64, 80, 96, 112, 128, 160, 192, 256]
print(f"\n[kcurve] == C={C} MDP 召回率-延迟曲线 (Kc from 64) ==", flush=True)
print(f"{'Kc':>5s} {'recall':>7s} {'P0':>3s} {'cand~':>7s} {'max':>7s} {'MB':>7s} {'SD(s)':>7s}  P3_in  P3_cand_gap", flush=True)
out = {}
for Kc in Kc_list:
    all_hit = 0; p0_hit = 0; szs = []; p3_info = None
    for i, p in enumerate(prep):
        sel = p["top"][:Kc]
        cand = set(np.concatenate([np.where(cl == c)[0] for c in sel]).tolist())
        hit = p["g1"] in cand
        all_hit += hit
        if names[i] in P0_NAMES:
            p0_hit += hit
            if names[i] == "P3 今天天气很好，我们去公园":
                lg = p["lg"]
                mask = np.full(lg.shape, -np.inf)
                for t in cand:
                    mask[t] = lg[t]
                top2 = np.argsort(-mask)[:2]
                cg = float(lg[top2[0]] - lg[top2[1]])
                p3_info = (hit, cg, int(top2[0]), int(top2[1]))
        szs.append(len(cand))
    avg_sz = int(np.mean(szs)); mx = int(np.max(szs))
    mb = avg_sz * D / 1e6; dt = mb / 21.5
    p3s = f"{str(p3_info[0]):>5s}  {p3_info[1]:.4f}" if p3_info else "?"
    print(f"{Kc:5d} {all_hit:3d}/{n} {p0_hit:3d} {avg_sz:7d} {mx:7d} {mb:7.2f} {dt:7.2f}  {p3s}", flush=True)
    out[str(Kc)] = {"recall_all": all_hit, "recall_P0": p0_hit,
                    "cand_avg": avg_sz, "cand_max": mx, "MB": round(mb, 2),
                    "SD_theory_s": round(dt, 2), "P3": {"in_cand": bool(p3_info[0]),
                    "cand_gap": round(p3_info[1], 4)}}

with open(os.path.join(HERE, "lmhead_mdp_kcurve.json"), "w") as f:
    json.dump({"C": C, "Kc_list": Kc_list, "curve": out}, f, indent=1)
print(f"\n[kcurve] wrote lmhead_mdp_kcurve.json", flush=True)
