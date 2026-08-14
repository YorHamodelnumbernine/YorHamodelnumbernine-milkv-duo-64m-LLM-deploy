#!/usr/bin/env python3
"""Phase 6 · MDP (球面 k-means) 聚焦标定 + 门禁验证 (host).

复用 lmhead_mips_cmp.kmeans_mdp, 扫 C x Kc -> recall/候选集/SD 延迟,
并在推荐工作点验证 P0 门禁: 候选 gap = logit(best cand) - logit(2nd cand)
必须 >= true gap (若 gold top-1 在候选集内, 自动成立; 这里数值确认)。
"""
import os, sys, time, json
import numpy as np
import collections

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
print(f"[gate] {n} prompts", flush=True)

embed = np.fromfile(os.path.join(W, "embed_i8.bin"), dtype=np.int8).reshape(V, D)
esc = np.fromfile(os.path.join(W, "embed_scales.f32"), dtype=np.float32)
rows = (embed.astype(np.float32) * esc[:, None]).astype(np.float32)
del embed, esc

def cand_gap(lg, cand_set):
    """候选集内 top1 - top2 logits."""
    lg = np.asarray(lg)
    mask = np.full(lg.shape, -np.inf)
    for t in cand_set:
        mask[t] = lg[t]
    top = np.argsort(-mask)[:2]
    return float(lg[top[0]] - lg[top[1]]), int(top[0]), int(top[1])

def sweep(C_list, Kc_list):
    recap = {}
    for C in C_list:
        t0 = time.time()
        Cm = cmp.kmeans_mdp(rows, C, iters=6, seed=7, subsample=12288)
        Xf = rows.astype(np.float32)
        cl = np.empty(V, np.int64)
        for i in range(0, V, 4096):
            cl[i:i+4096] = (Xf[i:i+4096] @ Cm.T).argmax(1)
        members = collections.defaultdict(list)
        for t in range(V):
            members[int(cl[t])].append(t)
        sizes = np.bincount(cl, minlength=C)
        print(f"[gate] C={C}: train+assign {time.time()-t0:.1f}s", flush=True)
        for Kc in Kc_list:
            all_hit = 0; p0_hit = 0; szs = []
            for i, (h, lg) in enumerate(zip(Hs, LGs)):
                sc = h.astype(np.float64) @ Cm.astype(np.float64).T
                sel = np.argsort(-sc)[:Kc]
                cand = set(np.concatenate([np.where(cl == c)[0] for c in sel]).tolist())
                g1 = int(np.argmax(lg))
                hit = g1 in cand
                all_hit += hit
                if names[i] in P0_NAMES: p0_hit += hit
                szs.append(len(cand))
            recap[(C, Kc)] = (all_hit, p0_hit, int(np.mean(szs)), int(np.max(szs)))
            mb = np.mean(szs) * D / 1e6; dt = mb / 21.5
            print(f"  C={C:5d} Kc={Kc:4d}: recall={all_hit:2d}/{n} P0={p0_hit}/3 "
                  f"cand~{recap[(C,Kc)][2]:6d} (max {recap[(C,Kc)][3]:6d}) = {mb:6.2f}MB = {dt:5.2f}s", flush=True)
    return recap

C_list = [256, 512, 1024]
Kc_list = [32, 64, 128, 256]
recap = sweep(C_list, Kc_list)

# ---- P0 门禁: 推荐工作点 C=1024 Kc=128 候选 gap ----
print("\n== P0 门禁验证 @ C=1024, Kc=128 ==", flush=True)
Cm = cmp.kmeans_mdp(rows, 1024, iters=6, seed=7, subsample=12288)
Xf = rows.astype(np.float32)
cl = np.empty(V, np.int64)
for i in range(0, V, 4096):
    cl[i:i+4096] = (Xf[i:i+4096] @ Cm.T).argmax(1)
members = collections.defaultdict(list)
for t in range(V):
    members[int(cl[t])].append(t)
for Kc in (64, 128):
    print(f"--- Kc={Kc} ---", flush=True)
    for i, (h, lg) in enumerate(zip(Hs, LGs)):
        if names[i] not in P0_NAMES: continue
        sc = h.astype(np.float64) @ Cm.astype(np.float64).T
        sel = np.argsort(-sc)[:Kc]
        cand = set(np.concatenate([np.where(cl == c)[0] for c in sel]).tolist())
        tg = float(lg[int(np.argmax(lg))] - np.sort(lg)[-2])
        cg, c1, c2 = cand_gap(lg, cand)
        g1 = int(np.argmax(lg))
        tag = "OK" if (g1 in cand and cg >= 0.05) else "**FAIL**"
        print(f"  {names[i][:24]:24s} g1={g1:7d} in_cand={g1 in cand} "
              f"true_gap={tg:.4f} cand_gap={cg:.4f} (cand top1={c1}, top2={c2})  [{tag}]", flush=True)

with open(os.path.join(HERE, "lmhead_mdp_gate.json"), "w") as f:
    json.dump({"C_list": C_list, "Kc_list": Kc_list,
               "recap": {f"{C}/{Kc}": recap[(C, Kc)] for C in C_list for Kc in Kc_list}},
              f, indent=1)
print("\n[gate] wrote lmhead_mdp_gate.json", flush=True)
