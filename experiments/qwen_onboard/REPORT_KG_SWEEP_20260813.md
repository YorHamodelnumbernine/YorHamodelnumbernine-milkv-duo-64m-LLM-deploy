# KG 扫参结论 — Path A per-group 缩放权重在 KG=64/128（host）

日期：2026-08-13 | 作者：推理引擎工程师 | 关联：`emu_kg_sweep.py` / `kg_sweep.json` / `group_int4.json`
状态：**不成立 — KG=64 与 KG=128 均 2/3，Path A 保持 KG=32=G（组结构锁）**
护栏：untracked（experiments/qwen_onboard/），不碰 master。

---

## 0. TL;DR（给 CEO）

| KG | 语义 | 判定 | prompt3 (今天天气…) |
|---|---|---|---|
| **32（基线，group_int4.json）** | per-(32,col) scale + 两遍法块 32 | **3/3** | 99366 ✓ (gap 0.485) |
| **64** | per-(64,col) scale + 两遍法块 64 | **2/3** | **翻转 111261 ✗**（ref 99366 跌至 top2，gap 0.470） |
| **128** | per-(128,col) scale + 两遍法块 128 | **2/3** | **翻转 111261 ✗**（ref 99366 跌至 top2，gap 1.039） |

**结论：Path A per-group 缩放权重在 KG=64/128 不成立 → submit 不能靠放大 KG 降 ~4x，
必须保持 KG=32=G。** 与 TPU 裁定表「KG=128 顺延假设」最终证伪（commit 0c56490 的悬置项
本次补扫关闭）。

---

## 1. 扫参方法（诚实部署口径）

`emu_kg_sweep.py` 对每个目标 KG，**直接从 fp32 权重按 group=KG 重新量化**（q4 网格 +
per-[K/KG,N] fp16 级 gscale），两遍法 matmul 块 = KG，per-(block,col) scale 后乘。
与 `emu_group_int4.py`（group_int4.json 3/3）同一条验证链，仅 KG/G 参数化：

- 3 固定 prompt（REF：2130 / 12095 / 99366），tokenizer 同源。
- 判据：top-1 == REF 且 min gap ≥ 0.05（沿用 M2 门禁）。

**组结构锁的数学本质**：Path A 的 gscale 是 **post-scale**（TIU 两遍法输出 p2 后乘
`p2×2^r×gscale[g,n]`），每 (32 行, 列) 一个 scale。若把两遍法块并到 KG=128，TIU 只给出
128 行累加和，无法拆回 4 个 32 行子累加分别乘各自 scale → **块 KG 必须等于 scale 组 G**。
因此「放大 KG 降提交」唯一的办法是把 per-group scale 也变粗（G=64/128），即本次扫参对象。

## 2. 数据（kg_sweep.json）

### KG=64 → 2/3
| prompt | next | ref | ok | gap | top5 |
|---|---|---|---|---|---|
| 中国的首都是 | 2130 | 2130 | ✓ | 0.852 | 2130 68990 9909 198 104673 |
| The capital of France is | 12095 | 12095 | ✓ | 3.448 | 12095 279 7407 264 32671 |
| 今天天气很好，我们去公园 | **111261** | 99366 | ✗ | 0.470 | 111261 **99366** 109280 69249 99632 |

### KG=128 → 2/3
| prompt | next | ref | ok | gap | top5 |
|---|---|---|---|---|---|
| 中国的首都是 | 2130 | 2130 | ✓ | 1.167 | 2130 198 68990 7 9909 |
| The capital of France is | 12095 | 12095 | ✓ | 0.812 | 12095 510 1447 7407 279 |
| 今天天气很好，我们去公园 | **111261** | 99366 | ✗ | 1.039 | 111261 **99366** 99924 109280 99632 |

- 失败模式一致：prompt3 的 bf16 gap 本就 razor-thin（KG=32 时仅 0.116–0.485），
  组 scale 变粗后权重重建误差增大 → top-2 翻转。KG=64/128 的 ref 均仍在 top5 第 2 位，
  说明是「余量耗尽」而非「崩溃」——但门禁 top-1 命中不满足。
- 基线复现：`emu_kg_sweep.py --kgs 32` 复跑 3/3（gap 1.910/1.189/0.116），
  与既有 group_int4.json（1.923/1.105/0.485）同判，仅浮点累加顺序导致 gap 微差。

### 2b. 根因隔离（exact int64 累加，无 int8 舍入）：仍是 2/3

`emu_kg_sweep.py --mode exact`（每 chunk 用 int64 精确累加，绕过两遍法 int8 舍入）对
KG=64/128 复跑，**结果不变——prompt3 同样翻转 99366→111261**：

| KG | exact 判定 | prompt3 next | 说明 |
|---|---|---|---|
| 64 | 2/3 | 111261 ✗（gap 0.358） | 与 twopass 同翻转 |
| 128 | 2/3 | 111261 ✗（gap 1.085） | 与 twopass 同翻转 |

→ **降质量根因 = 粗组 scale 的权重量化误差（post-scale per-group），与两遍法 int8 舍入无关。**
对照：Path B（per-channel scale，输出侧单 scale）KG=128/256 两遍法 3/3（wsdq_ref_tiu.json），
说明 KG 放大本身不伤两遍法——伤的是 Path A「组内共享一个 scale」在 64/128 行下的重建精度。

## 3. 对「submit 可降 ~4x」的判定

| 目标 KG | K-block/层（q/k/v/o/up/gate） | down K-block | 理论提交降幅 | 质量 | 结论 |
|---|---|---|---|---|---|
| 32 | 28 | 152 | —（基线） | 3/3 | **保持** |
| 64 | 14 | 76 | ~2x | 2/3 | 否决 |
| 128 | 7 | 38 | ~4x | 2/3 | 否决 |

- **submit 降幅无法兑现**：KG=64/128 均 2/3，门禁不过。Path A 维持 KG=32=G →
  引擎口径 M2 修正 1200 pass runs/layer（+600 g2l）不变。
- **若后续仍想降提交**：唯一不牺牲质量的路径是改 per-channel 语义（Path B，bf16 直存
  per-ch INT8，两遍法 KG=128 已 3/3，见 b_perrow_twopass_ref.json / wsdq_ref_tiu.json），
  但 SD 带宽 1.78x、ION 需拆子矩阵——CEO 已在 PS32 裁定中否决主路径。

## 4. 产物

- `emu_kg_sweep.py`：可复现扫参脚本（`--kgs 32 64 128`，`--mode twopass|exact`）。
- `kg_sweep.json`：两遍法数据（64/128 → 2/3）。`kg_sweep_smoke.json`：KG=32 基线复现。
- `kg_sweep_exact.json`：exact int64 累加根因隔离（64/128 → 2/3）。
- 未改动 `group_int4.json`（KG=32 权威基线保持）。

## 5. 复现

```
cd experiments/qwen_onboard
python3 emu_kg_sweep.py --kgs 32 64 128 --out kg_sweep.json        # 两遍法全扫（~2-3 min）
python3 emu_kg_sweep.py --kgs 64 128 --mode exact --out kg_sweep_exact.json  # 根因隔离
```
