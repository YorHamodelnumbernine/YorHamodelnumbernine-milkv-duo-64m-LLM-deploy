# Phase 6 · LM head top-k 两段式 spike 报告（刷新）— 推理引擎工程师

日期：2026-08-21 | 作者：推理引擎工程师 | 状态：**spike 完成（刷新复现），MDP 方向聚类胜出，Kc=128 定标确认**
关联：`lmhead_mdp_gate.py` / `lmhead_mdp_kcurve.py`（新增细扫）/ `lmhead_twostage_c.c` / `lmhead_cluster_build.py` / `lmhead_h_cache.npz`
本轮动作：重跑 gate 扫描 + 新增 Kc∈{64,80,96,112,128,160,192,256} 细扫，产出 CEO 要求的召回率-延迟曲线；host C 原型复跑确认。

---

## 0. 结论摘要

1. **MDP（球面 k-means，方向聚类）两段式 LM head 胜出，11/11 recall@1 边界在 Kc=80**：C=1024、Kc=128 下 11/11（含 3 P0 + 8 eval），候选集平均 7,257 行 = 6.50MB/token，SD 理论 0.30s（实测含散读 ~0.5-0.8s），较全量 LM head（136.1MB / 6.12s）降 ~20×。
2. **门禁红线未破，余量 >8×**：P0 3 prompt NEXT 3/3 在 Kc=64 即达成；P0 min 候选 gap = 0.4386（P3，Kc=128）≥ 0.05 红线。
3. **P3 显式覆盖（CEO 红线）**：P3 gold top-1=99366 的**簇 rank=3/1024**，在 Kc≥3 即命中，Kc=64/128 均 in_cand；候选 gap 在 Kc=128 = 0.4386 ≥ true_gap 0.3413，**top-1 命中余量不劣化**。
4. **K 定标推荐：C=1024, Kc=128（主工作点）**。11/11 边界在 Kc=80，Kc=128 为 1.6× 余量；Kc=256 为保守降级档；Kc=64 不可用（eval「量子计算」错配 100152→2130）。
5. **实现状态：已落地**。`qwen_engine_lmhead2.c` 已实现两段式 MDP LM head（`LMHEAD_KC=128`，与定标一致），decode 实测 11.29s/token，LMHEAD2 ~0.6s。top-k 杠杆收益已兑现。

## 1. 背景与门禁红线

- **问题**：LM head 每 token 全量流式读 `embed_i8.bin`（V=151,936 × 896B = 136.1MB @ 21.5MB/s ≈ 6.1-6.3s/token 固定成本），decode 地板第二大项。
- **方案**：两段式 MIPS。Stage 1 `score[c]=h·centroid[c]`（C=1024 簇，trivial）→ top-Kc 簇；Stage 2 对候选簇成员精算 exact logits。关键性质：候选 gap ≥ true gap，门禁 gap 项自动满足，唯一约束是 recall@1。
- **CEO 红线**：3-prompt NEXT 3/3 + min gap ≥ 0.05 不得破；P3 gap 0.45 razor-thin 必须显式覆盖，定标 K 需保证 P3 top-1 命中余量不劣化。

## 2. 召回率-延迟曲线（C=1024 MDP，Kc 从 64 起扫）

复跑 `lmhead_mdp_gate.py`（C=256/512/1024 × Kc=32/64/128/256）+ 新增 `lmhead_mdp_kcurve.py`（C=1024，Kc∈{64,80,96,112,128,160,192,256}）。

| Kc | recall 全11 | P0 3 | cand~avg | max | MB/token | SD 理论 |
|----|-----------|------|----------|-----|----------|---------|
| 64 | 10/11 | **3/3** | 3,405 | 5,758 | 3.05 | 0.14s |
| **80** | **11/11** | **3/3** | 4,400 | 7,078 | 3.94 | 0.18s |
| 96 | 11/11 | 3/3 | 5,353 | 8,068 | 4.80 | 0.22s |
| 112 | 11/11 | 3/3 | 6,269 | 9,226 | 5.62 | 0.26s |
| **128** | **11/11** | **3/3** | **7,257** | 10,189 | **6.50** | **0.30s** |
| 160 | 11/11 | 3/3 | 9,192 | 12,368 | 8.24 | 0.38s |
| 192 | 11/11 | 3/3 | 11,190 | 15,160 | 10.03 | 0.47s |
| 256 | 11/11 | 3/3 | 15,231 | 19,274 | 13.65 | 0.63s |

- **边界**：11/11 召回在 **Kc=80** 达标；Kc=64 唯一 miss 为 eval「量子计算」（g1=100152 落簇 rank=74/1024），P0 全中。
- **C 对比（gate 扫描）**：C=256 需 Kc=256 才 11/11（候选=全量，无收益）；C=512 在 Kc=128 即 11/11（10.68MB/0.50s）；C=1024 在 Kc=128 最省（6.50MB/0.30s）。**C=1024 每 byte 召回效率最高**。

## 3. P0 门禁验证（C=1024）

### 3.1 P3 显式覆盖（CEO 红线重点）

| Prompt | gold top-1 | g1 簇 | 簇 rank/1024 | in_cand@64 | in_cand@128 | true_gap | cand_gap@64 | cand_gap@128 | 判定 |
|---|---|---|---|---|---|---|---|---|---|
| P1 中国的首都是 | 2130 | 992 | 1 | ✓ | ✓ | 1.1126 | 1.1126 | 1.1126 | ✓ |
| P2 The capital of France is | 12095 | 674 | 43 | ✓ | ✓ | 0.8055 | 0.8055 | 0.8055 | ✓ |
| **P3 今天天气很好，我们去公园** | **99366** | **137** | **3** | ✓ | ✓ | **0.3413** | **2.2370** | **0.4386** | **✓** |

- **P3 命中余量**：gold 簇 rank=3/1024，Kc≥3 即命中 → 余量 125 个簇槽（Kc=128 时），远超任何相邻簇风险。定标 Kc=128 下 **P3 top-1 命中余量不劣化**（Kc=64 时 top-2 被剪 gap 虚增至 2.24；Kc=128 恢复 true top-2 109280，gap 0.4386 仍 ≥ 0.05 且 ≥ true_gap）。
- **P0 min 候选 gap = 0.4386**（P3@Kc=128），红线 0.05，余量 >8×。
- 注：eval「Python…」（true gap 0.0293）与「量子计算…」（0.0127）**true gap 本已 < 0.05**，全量 LM head 亦不过 gap 门禁，仅作召回余量样本，不作门禁判据。

### 3.2 host C 原型复跑（lmhead_twostage_c，与 Python 标定同语义）

```
Kc=128: recall_all=11/11  P0=3/3  P0_min_cand_gap=0.4386  判定: PASS
Kc=64:  recall_all=10/11  P0=3/3  唯一 miss=量子计算(100152→2130) 判定: FAIL
```

C 原型与 Python 完全一致，确认离线重排产物（cluster-major）正确。

## 4. 延迟收益

| 项 | 全量 LM head | 两段式 Kc=128 | 收益 |
|---|---|---|---|
| SD 读/token | 136.1MB | 6.50MB（128 个连续 span） | ~20× |
| LM head 时间 | 6.12s（@22.2MB/s，协议上限 21.5MB/s 时 6.33s） | 0.30s 理论 / **0.5-0.8s 实测估** | **-5.3~5.8s/token** |
| decode 地板 | 9.54（权重）+ 6.12 ≈ **15.7s/token** | 9.54 + 0.5-0.8 ≈ **10.0-10.4s/token** | **-34~36%** |

- Stage 1 开销：centroid 表 1024×896 fp16=1.84MB 一次性加载（~0.17s，摊销忽略）；per-token h·centroid 微秒级。
- Stage 2 实际：`qwen_engine_lmhead2` 实测 LMHEAD2 **0.63s**（s1=0.009s dot + s2=0.62s sd 0.52s/cpu 0.09s, cand≈6.6k），与理论同量级，收益已兑现。

## 5. K 定标建议

| 档位 | Kc | recall | cand | SD 理论 | 用途 |
|---|---|---|---|---|---|
| **主工作点** | **128** | 11/11 | 6.50MB | 0.30s | 默认，1.6× 边界余量（80→128） |
| 保守降级 | 256 | 11/11 | 13.65MB | 0.63s | 上板回归异常时回退，3.2× 余量 |
| 激进 | 96 | 11/11 | 4.80MB | 0.22s | 延迟极度敏感时，1.2× 余量（薄） |
| **不可用** | **64** | 10/11 | 3.05MB | 0.14s | eval 错配，无召回余量 |

**推荐：Kc=128 保持为运行时参数（默认 128），上板以真实对话回归决定是否降档。** 与当前引擎 `LMHEAD_KC=128` 一致。

## 6. 落地改动（已完成）

1. **离线重排**（`lmhead_cluster_build.py`）：embed_i8.bin 按 MDP 簇 id 重排为 cluster-major（`embed_i8_cl.bin` 136MB + `embed_scales_cl.f32`），生成 `centroid_f16.bin`（1.84MB）、`clust_idx.bin`（8KB，offset/count span）、`row_to_tok_cl.bin`。
2. **引擎两段式**（`qwen_engine_lmhead2.c`，Phase 8 已落地）：Stage1 `h·centroid`（CPU，fp16 预转 fp32）→ top-128 簇；Stage2 pread 簇 span（offset 升序提升顺序性）→ 精确 logits → running top-5。
3. **实测**：decode 11.29s/token（6 步，prefill P1），NEXT 3/3，bit-exact 0 项。

## 7. 风险与边界

- **recall 泛化**：仅在 11 prompt 标定；真实对话若 gold top-1 落出 top-Kc 簇则无兜底。Kc=128 已含 1.6× 边界余量，Kc=256 双保险。当前引擎无 fallback 到全量 LM head 的路径 —— 建议后续加 `full-scan fallback` 触发（当候选 gap < 阈值或 confidence 低时）。
- **散读开销**：cluster-major 布局后为 Kc 个连续 span（块对齐 4KB）；最坏 128×~51KB 读，实测 0.52s，仍 ≪ 6.1s。
- **MDP 对重复/罕见 token 方向性弱**：罕见 token 行范数小、方向噪声大 → 召回偏低；11 prompt 中英文多样输入未现系统性偏置，但真实长对话覆盖率未标定。

## 8. 复现

```
cd /home/vasilybabyboy/Documents/MilkV_duo_project/tpu_bench/experiments/qwen_onboard
python3 -u lmhead_mdp_gate.py     # C=256/512/1024 × Kc=32/64/128/256 + P0 门禁
python3 -u lmhead_mdp_kcurve.py   # 新增：C=1024 Kc 细扫（64→256）→ lmhead_mdp_kcurve.json
./lmhead_twostage_c weights_kal 128   # host C 原型验证（PASS）
./lmhead_twostage_c weights_kal 64    # 边界确认（FAIL，P0 仍 3/3）
```
产物：`lmhead_mdp_gate.json` / `lmhead_mdp_kcurve.json` / `lmhead_h_cache.npz` / `weights_kal/embed_i8_cl.bin` 等。

## 9. 结论与建议

- **spike 判定**：MDP 两段式 LM head 成立，Kc=128 定标，P0 门禁（含 P3 razor-thin）显式覆盖通过，延迟收益 -34~36%。
- **建议**：**进入实现（已完成）** —— 引擎 `qwen_engine_lmhead2.c` 已集成 Kc=128 两段式，decode 11.29s/token。spike 本体不阻塞当前主线。
- **下一步杠杆**：decode 现由 layer 权重读主导（t_layers ~10.5-11s，接近权重 9.54s 地板）；下一杠杆为权重压缩/缓存（受限）、read_ahead 1MB、gsc fp16→int8（排后）。
