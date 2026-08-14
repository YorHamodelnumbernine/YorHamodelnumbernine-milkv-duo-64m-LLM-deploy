#!/usr/bin/env python3
"""Phase 6 · LM head cluster 布局分析 (host).

问题: 上板 t_lmhead ~2.1s (SD 散读主导, 7x 每字节惩罚 vs 顺序读). 候选集 = top-Kc
簇成员并集, 若簇在文件中按"方向相似相邻"排布, top-Kc (对 h 方向邻近的簇) 会聚成
近连续窗口 -> 单次连续读 (21MB/s) 替代 128 次散读.

分析: 对 3 种布局 (当前 id / PCA-1D / PCA-2D 角序 / 随机投影序), 对每 prompt 的
top-Kc 簇, 测: 连续段数 runs、覆盖窗口 (max-min+1 簇)、读窗口的 SD 时间估计.
输入: weights_kal/centroid_f16.bin + lmhead_h_cache.bin (h 向量).
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.join(HERE, "weights_kal")
D, C, V = 896, 1024, 151936

cent = np.fromfile(os.path.join(W, "centroid_f16.bin"), dtype=np.float16).reshape(C, D).astype(np.float32)
Hs = np.fromfile("lmhead_h_cache.bin", dtype=np.float32).reshape(-1, D).astype(np.float32)
Kc = 128
FILE_MB = V * D / 1e6          # 136.13MB
BYTES_PER_CLUSTER = FILE_MB * 1e6 / C  # ~133KB/cluster (avg)

# 每 prompt 的 top-Kc 簇 (fp32 centroid 得分, 与引擎一致)
def top_clusters(h):
    sc = h @ cent.T
    return np.argsort(-sc)[:Kc]

# 布局函数返回 cluster 新序 (perm: 新位置 -> 原簇id)
def layout_current():
    return np.arange(C)
def layout_pca1():
    X = cent - cent.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    proj = X @ Vt[0]
    return np.argsort(proj)
def layout_pca2():
    X = cent - cent.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    p = X @ Vt[:2]
    ang = np.arctan2(p[:, 1], p[:, 0])
    return np.argsort(ang)
def layout_rproj(seed=0):
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 1, (D,)).astype(np.float32); w /= np.linalg.norm(w)
    proj = cent @ w
    return np.argsort(proj)

def eval_layout(perm, name):
    pos_of_cluster = np.empty(C, np.int64); pos_of_cluster[perm] = np.arange(C)
    print(f"\n[{name}]")
    print(f"  {'prompt':<28s} {'runs':>5s} {'win_cl':>7s} {'winMB':>7s} {'win_s':>6s}")
    for i, h in enumerate(Hs):
        top = top_clusters(h)
        poss = np.sort(pos_of_cluster[top])
        # 连续段数
        runs = 1 + int((np.diff(poss) > 1).sum())
        win_cl = int(poss.max() - poss.min() + 1)
        win_mb = win_cl * BYTES_PER_CLUSTER / 1e6
        win_s = win_mb / 21.5
        print(f"  {'prompt%d' % i:<28s} {runs:5d} {win_cl:7d} {win_mb:7.1f} {win_s:6.2f}")
    # 汇总
    runs_all = []; win_all = []
    for h in Hs:
        poss = np.sort(pos_of_cluster[top_clusters(h)])
        runs_all.append(1 + int((np.diff(poss) > 1).sum()))
        win_all.append(int(poss.max() - poss.min() + 1))
    print(f"  AVG: runs={np.mean(runs_all):.1f}  win_cl={np.mean(win_all):.0f}  "
          f"win_s(seq)={np.mean(win_all)*BYTES_PER_CLUSTER/1e6/21.5:.2f}s  "
          f"vs 散读实测 ~2.1s")

for fn, nm in [(layout_current, "current_id"), (layout_pca1, "pca1"), (layout_pca2, "pca2_angle"), (layout_rproj, "rproj")]:
    t0 = time.time()
    eval_layout(fn(), nm)
    print(f"  ({time.time()-t0:.1f}s)")
