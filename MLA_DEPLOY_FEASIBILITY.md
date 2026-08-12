# MLA 部署可行性调研 — Milk-V Duo (CV1800B) 推理引擎

日期：2026-08-12
作者：推理引擎工程师
范围：仅研究报告，不涉及生产代码改动。
目的：评估用 MLA（Multi-head Latent Attention，DeepSeek-V2 系）替换当前 MHA/GQA
架构，以"大幅提升上下文长度"的可行性。

---

## 0. 结论先行（TL;DR）

**不建议在 Duo 上为当前 SmolLM2-135M 部署 MLA。**

- 对 9 头 / GQA 的小模型，MLA 的 KV 压缩优势基本不存在（GQA 已经把 KV 压到 384
  值/头/token；MLA 的 per-head RoPE cache 反而把小模型拖大）。
- 100M–1B 区间**没有任何现成的 MLA 开源模型**（最小是 DeepSeek-V2-Lite 16B），
  要用必须自训练，工程量远超收益。
- 真正能同时放大上下文 + 降 Wt load 的杠杆是：**KV 改 INT8（上下文 ×4）** 和
  **权重改 INT4（Wt load ×2.8）**，与 MLA 无关。
- 若 CEO 的目标只是"更长上下文"，先做 INT8 KV + INT4 权重；MLA 应搁置。

---

## 1. KV 数学：当前 MHA/GQA vs MLA

### 1.1 当前实现（SmolLM2-135M，GQA）

- D=576, n_heads=9, n_kv_heads=3, head_dim=64 → `dkv = 3×64 = 192`
- 每 token 每层 KV 字节数（**当前代码存 FP32，非 INT8**）：

| 存储格式 | 每 token/层 | 30 层/ token | 1.7MB KV 预算下最大上下文 |
|---|---|---|---|
| FP32（现状） | 2×192×4 = **1536 B** | **46,080 B** | **≈39 token**（实测 max_seq=39）|
| INT8 | 2×192×1 = 384 B | 11,520 B | ≈156 token（**×4.0**）|

> 实测佐证：b2 运行日志 `[kv] ION: 1755 KB at off=23368704 (max_seq=39)`。
> 1,797,120 B / 39 token = 46,080 B/token，与 30×1536 完全一致。
> KV 预算受 ION 限制：24MB ION − 6 权重槽×3.38MB(20.3MB) − embed ≈ 1.7MB。

### 1.2 MLA 的 KV 存储结构（DeepSeek-V2 系）

MLA 每 token 缓存两样东西：
1. **压缩隐状态 `c_kv`**，维度 `d_c`（共享所有头，低秩）。
2. **解耦 RoPE cache `k_R`**，维度 `n_heads × d_r`（每头独立，因旋转频率不同）。

即：每 token/层 KV 值数 = `d_c + n_heads × d_r`。

对 SmolLM2（n_heads=9）代入常见潜维度（latent INT8 + RoPE 保持 FP32，理由见 §4）：

| 配置 | 每 token/层（latent INT8 + RoPE FP32）| 30 层/token | 1.7MB 预算上下文 | 相对现状 |
|---|---|---|---|---|
| MLA d_c=128, d_r=32 | 128 + 9×32×4 = **1280 B** | 38,400 B | ≈46 | **×1.2** |
| MLA d_c=64,  d_r=32 | 64 + 9×32×4 = **1216 B** | 36,480 B | ≈49 | ×1.26 |
| MLA d_c=64,  d_r=16 | 64 + 9×16×4 = **640 B** | 19,200 B | ≈93 | ×2.4 |
| MLA d_c=64,  d_r=16（RoPE 也 INT8，质量风险）| 208 B | 6,240 B | ≈288 | ×7.4（但见 §4 风险）|

**关键结论：**
- MLA 在小头数模型上，per-head RoPE cache（FP32）直接吃掉压缩收益；
  只有在 `d_r` 很小 + latent 很小 + RoPE 也量化时才接近 INT8-GQA。
- **上下文放大主要来自 KV 改 INT8（×4.0），MLA 只在其上叠 ×1.2~2.4，且要牺牲 RoPE 精度。**
- 若把权重也改 INT4，权重槽从 20.3MB 降到 10.6MB，ION 多出 ~10MB 给 KV：
  - GQA INT8 KV + INT4 权重：上下文 ≈ (1.7+10)MB / 11,520 B ≈ **1000+ token**。
  - 这才是"大幅提升上下文"的现实路径。

### 1.3 MLA 对 Wt load（权重字节）的影响

MLA 把 W_k/W_v（GQA）换成低秩 W_DKV/W_UK/W_UV。对 9 头模型：

- GQA KV 权重：2×D×dkv = 2×576×192 = **221,184 B/层**
- MLA（d_c=64）：≈3×D×d_c = 3×576×64 = **110,592 B/层**（省 ~110KB）
- 单层权重共 3,543,552 B，KV 权重只占 6%；省一半也仅让 Wt load **−3%**。
- 结论：**MLA 对 Wt load 几乎没有帮助**；Wt load 的杠杆在 INT4 权重（见任务一报告）。

---

## 2. 候选 MLA 模型（100M–1B，适配 CV1800B）

| 模型 | 参数量 | 是否 MLA | 上下文 | INT8 权重体积 | Duo 可行性 |
|---|---|---|---|---|---|
| DeepSeek-V2-Lite | 16B | ✅ MLA | 32K | ~16 GB | ❌ 远超 28MB RAM + SD 带宽 |
| DeepSeek-V3 | 671B | ✅ MLA+MoE | 128K | ~670 GB | ❌ |
| MiniMax-Text-01 | 456B | 混合线性+MLA | 1M | 数百 GB | ❌ |
| Qwen3-0.6B | 0.6B | ❌ GQA（非 MLA）| 32K | ~600 MB | ⚠️ 仅权重 600MB，SD 读 28s/token，不可交互 |
| **100M–1B 纯 MLA** | — | — | — | — | **❌ 不存在公开模型** |

**结论：CV1800B 的约束（28MB DDR + 24MB ION + 1MB neuron + SD 权重流式）只适合
~100–200M 级 INT8 模型（当前 SmolLM2-135M=101MB 已是上限附近）。该区间无任何 MLA 模型。
用 MLA 必须自训练一个 ~135M 的 MLA 模型，属"新模型从头做起"，工程量大。**
（Qwen3-0.6B 即便不是 MLA，也因权重过大不可用，可排除。）

---

## 3. bmk1822 计算适配与 TPU 障碍

MLA 相对当前 GQA attention 的额外计算：

1. **down-projection**：`c_kv = W_DKV · x`（D→d_c）→ 现有 INT8 `tpu_matmul_build`。
2. **up-projection（每 token 重建 K/V）**：`K = W_UK · c_kv`，`V = W_UV · c_kv`
   （d_c→n_heads×(d_h+d_r)）→ 现有 INT8 matmul，decode 单 token 时是 1×d_c×576，开销小。
3. **RoPE**：对 `k_R` 旋转 → 现有 elemwise（当前引擎已有 RoPE 步骤，~6ms）。
4. **注意力 scores/mix**：重建后的 K/V 走现有 scores/mix matmul 路径。
5. **KV cache 布局**：改存 `c_kv`（INT8+scale）+ `k_R`（FP32），替换现有 K/V float 数组
   （`sm_kv_alloc_ion` / `kv->K[l]` / `kv->V[l]`）。

**新增算子需求**：无全新算子，都是现有 matmul/elemwise 的重新组合。但数据流变化较大：
- 每次 attention 前需"从 latent 重建 K"，多一步 matmul（每 token 每层 1 次）。
- cache 读写由 2 块连续数组（K、V）变成 2 块异构数组（latent + RoPE），
  `sm_kv_alloc_ion` 的 `per_layer` 对齐逻辑要改。

**TPU 障碍：**
- **1MB neuron memory**：重建 K 的中间量 n_heads×(d_h+d_r)。9 头×(64+32)=864 值 < 1MB，可行；
  但若要支持更大头数/更大 latent，neuron 会吃紧。
- **无 INT4/FP8 原生 matmul**：bmk1822 只有 INT8。MLA 的 FP8 精度收益（DeepSeek 用 FP8）
  无法直接获得，只能 INT8。
- **RoPE cache 精度**：旋转在低精度下误差累积明显，`k_R` 需 FP32/FP16，
  这在多头小模型上直接主导 KV 体积（见 §1.2）。
- **mbox EMBED_XPOSE cur_v≤2048** 与 MLA 无直接关系，但限制了 embedding/LM-Head 通路，
  与 MLA 无关，属既有约束。

---

## 4. INT8 量化 latent / RoPE（参考 DeepSeek）

- **DeepSeek 参考做法**：DeepSeek-V2/V3 对**权重**用 FP8（cast 到 FP8 做 matmul），
  但对 **KV cache（含 latent `c_kv` 与 RoPE cache）保持 BF16/FP32**，不量化 cache。
- 因此对 Duo 的 INT8-only TPU，最接近的做法是：
  - latent `c_kv`：INT8 + per-channel/per-token scale（写入 cache 前量化）。可行，误差可控。
  - RoPE cache `k_R`：**必须保持 FP32**（旋转矩阵低秩 + 逐位累加，INT8 误差会随长度漂移）。
- 风险：
  - latent 也是 KV，若 INT8 误差随 token 累积，长上下文质量下降（需 per-token rescale）。
  - RoPE INT8（若要省字节）会造成旋转角度漂移，质量损失明显，**不建议**。
  - 即便 latent INT8 + RoPE FP32，9 头模型的 KV 也主要由 RoPE FP32 主导（§1.2），
    压缩效果远不如直接 GQA-INT8。

---

## 5. 实施路径、工作量与风险

| 路径 | 内容 | 工作量 | 风险 | 备注 |
|---|---|---|---|---|
| 改现有引擎 | attention 块改写：down/up 投影、新 KV 布局（`sm_kv_alloc_ion`、`sm_forward_pool` 注意力段）| 中—大（约 30% 引擎逻辑）| 中 | 但**没有 MLA 模型可跑**，改了也白改 |
| 新模型从头做起 | 自训练 ~135M MLA 模型（预训练数据/算力/蒸馏）+ 引擎适配 | 大（数周~数月）| 高 | 需要训练基础设施，超出本项目范围 |
| **推荐替代**：INT8 KV | `kv->K/V` 改 int8 + scale，写/读时量化 | **小（~1 天）** | 低 | 上下文 ×4（39→~150 token），无需换模型 |
| **推荐替代**：INT4 权重 | 权重打包 2:1，读时解包/反量化到 INT8 | 中（~1 周） | 中 | Wt load ×2.8，同时把 ION 权重槽减半→更多 KV |

**明确建议：**
1. **短期**：KV 改 INT8（上下文 ×4）+ 采纳 DDR embed=2MB 配置（decode 5166ms/tok，见任务一）。
2. **中期**：INT4 权重（Wt load ~1.2–1.5s，并释放 ION 给 KV → 上下文可达 ~1000 token）。
3. **MLA 搁置**：等未来有 100M–1B 级 MLA 开源模型，或需要极长上下文且愿意自训练时再评估。

---

## 附：关键数值速查

- 当前 KV：1536 B/token/层（FP32）→ 46,080 B/token → 1.7MB 预算 ≈ 39 token。
- GQA-INT8 KV：384 B/token/层 → 11,520 B/token → ≈156 token（×4）。
- MLA(9头,d_c=64,d_r=16, latent INT8+RoPE FP32)：640 B/token/层 → ≈93 token（×2.4）。
- MLA 对 Wt load：最多 −3%（KV 权重仅占单层 6%）。
- 模型可用性：100M–1B MLA 模型不存在；最小 MLA 为 DeepSeek-V2-Lite 16B（不可行）。
