# Phase 6 · LM head top-k 两段式 spike 报告 — 推理引擎工程师

日期：2026-08-13 | 作者：推理引擎工程师 | 状态：**spike 完成，MDP 方向聚类胜出** | 关联：`lmhead_topk_spike.py` / `lmhead_mips_cmp.py` / `lmhead_mdp_gate.py` / `lmhead_h_cache.npz` / `REPORT_TPU_PHASE6_SD_PRESC_20260813.md`
护栏：本报告与脚本均留在 untracked `experiments/qwen_onboard/`，不碰 master。

---

## 0. 结论摘要

1. **MDP（球面 k-means，方向聚类）两段式 LM head 胜出**：C=1024、Kc=128 下 **11/11 recall@1（含 3 P0 + 8 eval）**，候选集平均 **7,257 行 = 6.50MB/token**，SD 理论 0.30s/token（实测含散读开销估 ~0.5-0.8s），较全量 LM head（136.1MB / 6.12s）**降约 20×**。
2. **门禁红线显式覆盖**：P0 3 prompt NEXT 3/3，候选 gap 在 Kc=64 即全部 ≥ 0.05（P1 1.11 / P2 0.81 / P3 2.24）；Kc=128 下 P3 候选 gap 0.4386 ≥ 0.05。**「若 gold top-1 ∈ 候选集，则候选 gap ≥ true gap」性质数值确认成立**，门禁 gap 项自动满足，唯一约束是 recall@1。
3. **方法对比（4 候选）**：欧氏 k-means（现基线，Kc=256 才 11/11，cand~17,145=15.4MB）、**MDP 球面（Kc=128 即 11/11，6.5MB）**、随机投影 top-B（B=32768 仅 5/11）、范数排序（0/11 全败）。**聚类目标必须与 MIPS 查询一致（max-dot）**，欧氏/范数/随机投影均不匹配。
4. **落地需一次离线改动**：embed 行按簇 id 重排为 cluster-major 布局，使 Stage-2 每 token 只读 Kc 个连续 span（避免散行随机读）；索引表 1024×8B=8KB 常驻。
5. **decode 地板更新**：权重 9.54s + LM head ~0.5-0.8s ≈ **10.0-10.4 s/token**（现 15.7s，**降 ~34-36%**），为 Phase 6 唯一大杠杆。

## 1. 背景与门禁红线

- **问题**：LM head 每 token 全量流式读 `embed_i8.bin`（V=151,936 × 896B = 136.1MB @ 22.2MB/s = **6.12s/token** 固定成本），decode 地板 15.7s/token 第二大项。
- **方案**：两段式 MIPS。Stage 1 `score[c]=h·centroid[c]`（C 簇，trivial）→ top-Kc 簇；Stage 2 对候选簇成员精算 exact logits。关键性质：候选 gap ≥ true gap，门禁 gap 项自动满足。
- **CEO 红线**：3-prompt NEXT 3/3 + min gap ≥ 0.05 不得破；P3 gap 0.45 razor-thin 必须显式覆盖。

## 2. 方法对比（lmhead_mips_cmp.py，11 prompt）

| 方法 | 关键行为 | recall@1（全 11 / P0 3） | 候选集（avg） | SD 理论 |
|---|---|---|---|---|
| A 欧氏 k-means C=1024 | Kc=128 仅 6/11；Kc=256 才 11/11 | 11/11 @Kc=256 | 17,145 行 = 15.36MB | 0.71s |
| **B MDP 球面 k-means C=1024** | **Kc=128 即 11/11** | **11/11 @Kc=128** | **7,257 行 = 6.50MB** | **0.30s** |
| C 随机投影 top-B dim=64 | B=32768 仅 5/11 | 5/11 | 32,768 行 = 29.4MB | 1.37s |
| D 范数排序 top-B | 全败 | 0/11 | — | — |

- **结论**：欧氏 k-means 的聚类目标（平方距离）与 LM head 的 MIPS 查询（max-dot）目标不匹配，是首轮 spike（C=1024 需 Kc=256，且 P3 簇 rank 246/1024）召回差的根因。**MDP 按方向聚类（单位化行向量 + max-dot 分配），直接对齐 MIPS 查询**，recall-per-byte 提升 3×+。
- P3 在 MDP 下 Kc=8 即命中（欧氏下 Kc=256 才命中）—— razor-thin case 已被 MDP 显式覆盖。

## 3. MDP C×Kc 扫描（lmhead_mdp_gate.py）

| C | Kc | recall 全11 | P0 3 | cand~avg（max） | MB | SD 理论 |
|---|---|---|---|---|---|---|
| 256 | 128 | 10/11 | 2/3 | 22,572（35,536） | 20.2MB | 0.94s |
| 512 | 64 | 10/11 | 3/3 | 5,492（7,673） | 4.9MB | 0.23s |
| 512 | 128 | 11/11 | 3/3 | 11,923（14,730） | 10.7MB | 0.50s |
| **1024** | **64** | 10/11 | **3/3** | 3,405（5,758） | 3.1MB | 0.14s |
| **1024** | **128** | **11/11** | **3/3** | **7,257（10,189）** | **6.5MB** | **0.30s** |
| 1024 | 256 | 11/11 | 3/3 | 15,231（19,274） | 13.7MB | 0.63s |

- Kc=64 边界：唯一 miss 是 eval「量子计算」（非 P0）；P0 全中。
- Kc=128：11/11，比失效边界多 2× 余量。

## 4. 门禁验证（P0，C=1024）

| Prompt | NEXT | in_cand@64 | in_cand@128 | true_gap | cand_gap@64 | cand_gap@128 | 判定 |
|---|---|---|---|---|---|---|---|
| P1 中国的首都是 | 2130 | ✓ | ✓ | 1.1126 | 1.1126 | 1.1126 | ✓ |
| P2 The capital of France is | 12095 | ✓ | ✓ | 0.8055 | 0.8055 | 0.8055 | ✓ |
| P3 今天天气很好，我们去公园 | 99366 | ✓ | ✓ | 0.3413 | **2.2370** | **0.4386** | ✓ |

- 候选 gap 均 ≥ true gap（P3@Kc=64 top-2 被剪 → gap 增至 2.24；@128 恢复 true top-3 109280，gap 0.44 仍 ≥ 0.05）。
- **门禁红线未破，且余量 >8×（P0 min 0.4386 vs 0.05）。**
- 注：eval「Python…」（true gap 0.0293）与「量子计算…」（0.0127）**true gap 本已 < 0.05**，全量 LM head 亦不过 gap 门禁，故仅作召回余量样本、不作门禁判据。

## 5. 推荐工作点与延迟影响

- **主工作点：C=1024, Kc=128** → 11/11 recall，候选 6.50MB/token（max 9.13MB），SD 理论 0.30s。
- **保守档：C=1024, Kc=256** → 13.65MB，0.63s，召回余量 2×，供上板回归异常时降级。
- 建议 Kc 设为运行时参数（默认 128），上板以真实对话回归决定是否升档。
- 落地延迟核算（实测速率）：
  - Stage 1：centroid 表 C×D fp16 1.84MB（或 fp32 3.67MB）一次性 SD 加载 ≈ 0.17s；per-token 896×1024 matmul 微秒级（TIU）。
  - Stage 2：6.5MB/token 簇连续 span 读 @ ~21MB/s = 0.30s 理论；含散读/对齐开销实测估 0.5-0.8s。
  - **新 decode 地板 ≈ 9.54（权重）+ 0.5-0.8（LM head）≈ 10.0-10.4 s/token**（现 15.7s，**-34~36%**）。

## 6. 落地改动（一次离线 + 引擎小改）

1. **离线重排**：按 MDP 簇 id 将 `embed_i8.bin` 重排为 cluster-major（每簇成员连续），`embed_scales.f32` 同步重排；生成簇索引表 `clust_off[1024]+clust_cnt[1024]`（8KB）。
2. **Stage 1（每 token）**：`h [896] · centroid [1024×896]` → 1024 score → top-Kc 簇（直接取索引表 span）。
3. **Stage 2（每 token）**：读 Kc 个连续 span（int8 行 + fp32 scale）→ 精确 logits（两遍法或直接 fp32 累加，行数 7k×896 计算量 ~6.5M MAC，CPU/TIU 均微秒级）→ argmax → token。
4. ION：centroid 1.84-3.67MB + 索引 8KB 常驻（FREE 区 5.16MB 内，余 ~1.5MB）。

## 7. 风险与边界

- **recall 泛化**：仅在 11 prompt 标定；真实对话若 gold top-1 落出 top-Kc 簇则无兜底 → 默认 Kc=128 已含 2× 余量，Kc=256 双保险。
- **散读开销**：cluster-major 布局后为 Kc 个连续 span，块对齐到 4KB；最坏 128×~51KB 读估 <0.8s，仍 ≪ 6.12s。
- **MDP 对重复/罕见 token 方向性弱**：方向聚类偏好高频方向，罕见 token 行范数小、方向噪声大 → 召回偏低；已在 8 eval 中覆盖中英文多样输入，未现系统性偏置。

## 8. 复现

```
cd /home/vasilybabyboy/Documents/MilkV_duo_project/tpu_bench/experiments/qwen_onboard
python3 lmhead_topk_spike.py --cs 1024 --kcs 4,8,16,32,64,128,256   # 首轮欧氏基线
python3 -u lmhead_mips_cmp.py                                        # 4 方法对比
python3 -u lmhead_mdp_gate.py                                        # MDP 扫描 + 门禁
```
产物：`lmhead_topk.json` / `lmhead_mips_cmp.json` / `lmhead_mdp_gate.json` / `lmhead_h_cache.npz`。

## 9. 二期候选（本 spike 后）

- **gsc fp16→int8 并行试验**（CEO 备注）：需 3-prompt 回归，独立于本项。
- **SD 控制器 21.5MB/s 已确认协议上限**（`REPORT_TPU_PHASE6_SD_PRESC`）—— 21→29MB/s 不可达，关闭该项。
- 两段式 LM head 落地后，decode 地板由 SD 权重读（9.54s）主导，下一杠杆为权重压缩/缓存（受限）或双核 offload（需物理地址解析路径）。
