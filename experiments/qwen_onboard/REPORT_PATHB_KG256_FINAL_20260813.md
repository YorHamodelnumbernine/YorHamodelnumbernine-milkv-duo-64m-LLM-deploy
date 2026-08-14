# Qwen 上板最终路径 REPORT — Path B（per-channel INT8 + per-chunk 两遍法）+ KG=256

日期：2026-08-13（晚）
作者：推理引擎工程师（host 扫参 + 路径收敛）｜联动：TPU 底层工程师（上板验证）
数据源：`chunk_sweep.json`（host 决定性扫参）· `REPORT_PATHB_CHUNK_20260813.md`（TPU 上板）·
        `REPORT_PS32_20260813.md`（ps32 判负 + 出口裁定）· `p0_refs.json`（bf16 参考）
状态：**Qwen 上板最终路径收敛完成**，提交 CEO 决策
范围：本报告仅写 `experiments/qwen_onboard/`，不触碰 master

---

## 0. TL;DR（最终裁决）

**Qwen 采用 Path B：per-channel INT8 权重 + per-chunk 自适应 rshift 两遍法 matmul + CPU fp32 累加（ps32-free）。KG=256 为生产主配置。**

1. **① 路径/KG**：Path B 全链路 3/3；**KG=256 推荐**。最薄 gap（今天天气很好，0.189）**高于 bf16 参考 gap（0.125）**——0.19 是 prompt 固有难度，不是量化退化。**calibration 缓冲不作为门禁**，作扩展安全网（见 §5）。
2. **② 硬件成本（KG=256）**：1,168 次 submit/层、28,032 次/token；TIU run-only **0.86s/token**、串行上界 2.79s/token。全 K 基线 down 不可行（lmem 放不下）且质量 1/3 判负；per-chunk 计算量 3–13x 被 SD 读（~16.3s/token）**完全掩盖 → 可接受**。
3. **③ ION 24MB**：每层 int8 展开 **14.91MB**。整层双缓冲 29.8MB **超 24MB 不可行**；整层单缓冲 16.7MB 可放但牺牲流水线重叠；**per-matrix 双缓冲 10.3MB 为推荐共存方案**，与 TPU `REPORT_PATHB_CHUNK §4.2` 对齐。
4. **④ ps32 关闭确认**：路径 A（硬件 per-group dequant matmul）正式判负，`REPORT_PS32_20260813.md` 为终裁；Path B 两遍法为唯一生产出口。

---

## 1. 决策链路回顾

| 阶段 | 结论 | 依据 |
|---|---|---|
| ps32 部分和调查 | **判负**：无 TIU 链式累加 + lmem 分配器缺陷 + 列丢失 | `REPORT_PS32_20260813.md` §1（12 探针） |
| ps32 硬件裁决 | ps32 int32 可读但 fp32 不可读，submit 80k/token，7–12 人日，无质量增益 | `REPORT_PS32_20260813.md` §7 |
| Path A（INT4 G32）+ 两遍法 | 上板四闸口通过，但 SD 带宽 201MB/token、ION 16.8MB 双缓冲、CPU 反量化瓶颈 | `REPORT_PATHA_KG32_20260813.md` |
| **Path B（per-ch INT8）+ 两遍法** | **上板 BIT-EXACT，KG=256/128 均无新增硬限制；host 扫参 3/3** | `REPORT_PATHB_CHUNK_20260813.md` + `chunk_sweep.json` |

**CEO 指令**：ps32 判负已接受，per-chunk 出口确认 → 收敛最终路径。本报告即收敛产出。

---

## 2. ① 路径建议：Path B + KG=256

### 2.1 路径定义（无 ps32）

- 权重：**直接 per-channel INT8**（Path B，bf16 → RTN per-output-channel scale），无 INT4→INT8 反量化。
- matmul：K 按 **KG 切分**，每 chunk 两遍法：
  - pass1：安全 rshift 回读 int8 输出（`sat8((acc+2^(r-1))>>r)`，round-half-up）→ CPU 算该 chunk 真实 max → r_opt；
  - pass2：精化 rshift 满幅输出；
  - CPU fp32 跨 chunk 累加：`Σ res_i8[m,n]·sc_row[m]·2^r·lsc[n]`。
- 激活：per-row（per-token）INT8 对称量化。
- 数据流已由 TPU 上板 **BIT-EXACT** 验证（`REPORT_PATHB_CHUNK` P2，maxrel=0）。

### 2.2 KG 取值与 gap margin 依据（决定性数据：`chunk_sweep.json`）

gap = top1−top2 logit 余量；next_token 参照 = bf16 参考（`p0_refs.json`）。

| KG | 中国的首都是 | The capital of France is | 今天天气很好 | **min gap** | down chunks/层 |
|---|---|---|---|---|---|
| **bf16 参考** | 0.375 | 0.875 | **0.125** | **0.125** | — |
| 32 | 1.327 | 0.722 | 0.524 | 0.524 | 152 |
| 64 | 0.970 | 1.087 | **0.005** | 0.005 ✗ | 76 |
| 128 | 0.945 | 1.227 | 0.215 | 0.215 | 38 |
| **256（推荐）** | **0.841** | **0.889** | **0.189** | **0.189** | **19** |

**结论：KG=256 稳妥。**

1. **全部 KG 3/3**（12/12 ok=true）；KG=64 因天气 prompt gap 0.005 近塌缩**明确排除**。
2. **KG=256 最薄 gap 0.189 高于 bf16 参考 0.125**。即：该 prompt 在 fp32 参考本身就极薄（0.125），量化链路（0.189）反而更宽 → **0.19 是 prompt 固有难度，不是量化退化信号**。三个 prompt 的 KG=256 gap（0.841/0.889/0.189）全部 ≥ bf16 参考（0.375/0.875/0.125）。
3. **KG=128 vs 256 余量差异无统计意义**：min gap 0.215 vs 0.189，仅差 0.026 logit，且二者均 ≥ bf16 参考。以硬件成本论（down 87ms vs 45.5ms，KG=128 约 1.9x），KG=256 显著更优。
4. **历史注记**：`REPORT_PS32 §6` 曾记 "KG=256 退化 2/3"，该结论来自两遍法早期中间结果；**最终决定性扫参 `chunk_sweep.json` 已以 KG=256 3/3 覆盖并推翻该顾虑**（差异源于当时 rshift 推导非最终版）。

### 2.3 calibration 缓冲判定

**不作为门禁。** 理由：

- 当前 scale = 纯权重 min/max RTN（`convert_qwen.py pc8`），无激活统计参与；但激活侧 per-row 量化是**逐 token 自适应**（等价于运行时自校准），两遍法又把 per-chunk int8 舍入误差压到 ~1/254 级。
- KG=256 三条 prompt 的量化 gap 已全部 ≥ bf16 参考 → 在参考集上**无剩余可榨余量的紧迫性**。

**安全网（按需启用，均不阻塞开工）**：

| 安全网 | 触发条件 | 动作 | 改动面 |
|---|---|---|---|
| A. KG 回退 | 上板 3-prompt 门禁出现 miss 或 min gap < 0.05 | KG=256 → 128（参数/编译开关） | 引擎零代码，仅 cmdbuf 模板选择 |
| B. calibration 缓冲 | 扩展评测（>3 prompt）发现不稳定 | scale 由权重 min/max 改**激活感知校准**（act-aware / SmoothQuant-lite 风格） | 仅 converter（`convert_qwen.py` 权重生成），引擎不变 |
| C. 余量兜底 | 极端 razor-thin 场景 | ps32 int32 精确部分和（文档化 fallback，`REPORT_PS32 §7`） | 引擎新增 ps32 路径 |

---

## 3. ② 硬件成本（KG=256）

### 3.1 每层提交次数（KG=256，M=1 decode，两遍法）

提交 = 1 次 bmkernel INT8 matmul（一个 chunk × 一个 N-tile × 一遍 pass）。

| 矩阵 | [K,N] | K chunks | N_tile | tiles/chunk | 提交（×2 遍） |
|---|---|---|---|---|---|
| q / o | 896×896 | 4 | 112 | 8 | 64 |
| k / v | 896×128 | 4 | 112 | 2 | 16 |
| up / gate | 896×4864 | 4 | 112 | 44 | 352 |
| down | 4864×896 | **19** | 112 | 8 | 304 |
| **合计/层** | | | | | **1,168** |
| **合计/token（×24）** | | | | | **28,032** |

### 3.2 每 token 延迟（KG=256，数据源 `REPORT_PATHB_CHUNK` §2.2）

| 分量 | per-layer | ×24/token | 相对 SD |
|---|---|---|---|
| TIU run-only（硬成本） | 35.8ms | **0.86s** | 0.05×SD ✓ |
| 串行上界（build+run+read+cpu，探针最坏） | 116ms | **2.79s** | 0.17×SD ✓ |
| Path B SD 读（权重 358MB/token） | 681ms | **16.3s**（@21.9MB/s） | — |
| 全量 decode 上界（+LM head 137MB/token） | — | **~22.6s**（@21.9MB/s） | — |

### 3.3 对比当前 INT8 全 K 基线（`REPORT_PATHB_CHUNK` §2.3）

| 矩阵 | 全 K 基线（rshift=12） | per-chunk 2-pass | 倍率 |
|---|---|---|---|
| q | 1.21ms | 10.18ms | 8.4x |
| k | 0.72ms | 9.18ms | 12.7x |
| up | 4.55ms | 14.96ms | 3.3x |
| **down** | **INFEASIBLE**（K=4864 连 N_tile=5 都超 32KB lmem） | 45.5ms | — |

**可接受性判定：可接受，且是唯一可行解。**

1. **down 全 K 物理不可行** → per-chunk 是布局必需，非可选优化。
2. **全 K 单 rshift 是质量 1/3 判负根因**（`REPORT_PS32 §6`）→ 计算量代价是换 3/3 质量的必然成本。
3. **计算量（run-only 0.86s、串行 2.79s）均 ≪ SD 读（16.3s/token）** → decode 上界维持 SD-bound，per-chunk 提交量不改变 decode 上界。
4. **submit 28k/token 在 cmdbuf 预算内**（PS32 probe21：批量 build ~1.9μs/op，28k ≈ 53ms/token）；但引擎**必须预建 cmdbuf**（按 rshift 模板/patch 字段），禁止 per-submit 全链路 register/alloc/convert/load（0.55ms/submit 不可接受），并**只 invld 输出区**规避探针全 8MB 假象。

---

## 4. ③ ION 24MB 缓冲方案（每层 int8 展开 14.91MB 的单/双缓冲共存）

每层 int8 权重 = 14,909,440 B = **14.91 MB**（q 0.80 + k 0.11 + v 0.11 + o 0.80 + up 4.36 + gate 4.36 + down 4.36，均为十进制 MB）。

| 方案 | 权重缓冲 | 其他（lsc/激活/acc/SD-io） | 合计 | 判定 |
|---|---|---|---|---|
| A. **整层 ×2 双缓冲** | 29.82MB | ~1.9MB | **31.7MB** | **超 24MB ✗** |
| B. 整层单缓冲 | 14.91MB | ~1.9MB | ~16.8MB | 可放，但 SD║TIU 无法重叠 → +2.79s/token |
| **C. per-matrix 双缓冲（推荐）** | **8.72MB**（最大矩阵 up/gate/down 4.36MB×2） | ~1.6MB | **≈10.3MB** | **FIT，余 ~13.7MB ✓** |
| D. 混合：整层单 + 最大矩阵双槽 | 14.91+4.36=19.27MB | ~1.6MB | ~20.9MB | FIT，可跨层预取大矩阵（可选） |

**结论：整层双缓冲在 24MB 预算内不可行；采用 per-matrix 双缓冲（方案 C），与 TPU `REPORT_PATHB_CHUNK §4.2`（≈10.3MB）完全对齐。**

- **流水线正确性**：SD 预取 matrix i+1 ║ CPU fp32 累加(matrix i) ║ TIU 两遍(matrix i)。per-matrix 粒度下 SD 恒领先：最慢的小矩阵（k/v，SD 5ms vs TIU 9.18ms）由大矩阵（up/gate/down，SD 199ms vs TIU 15–45ms）的富余带宽补回 → 全层窗口 SD 主导，流水线不塌。
- **余量 13.7MB**：可配 INT8 KV cache / 更大激活缓冲（对齐 A' 裁定建议）。
- **注意**：Path B 权重为直接 per-channel INT8，`weights/` 现为 INT4（`layerN_i4.bin`）——**需 converter 新产出 per-channel INT8 权重文件**（~358MB/SD，14.91MB/层）。

---

## 5. 质量评估与余量（3/3 数据汇总）

`chunk_sweep.json` 全 KG 12/12 ok=true；next_token 与 bf16 参考一致（2130 / 12095 / 99366）。

| KG | 3 prompt next_token | 判定 | 备注 |
|---|---|---|---|
| 32 | 全中（gap 0.52–1.33） | 3/3 ✓ | 成本最高（down 152 chunk） |
| 64 | 全中（gap 0.005–1.09） | 3/3 ✓ | 天气 gap 0.005 近塌缩，排除 |
| 128 | 全中（gap 0.22–1.23） | 3/3 ✓ | 回退档 |
| **256** | **全中（gap 0.19–0.89）** | **3/3 ✓** | **生产主配置** |

引擎实现约束（沿用 `GATE_A_SIGNOFF_20260813.md`）：
- CPU 反量化必须用 TIU 同公式 `sat8((acc+2^(r-1))>>r)`（round-half-up），否则系统性偏差 1。
- M=1（decode）per-chunk rshift = 单标量；M>1（prefill）per-row rshift 需 M 次 pass2 提交（decode 主路径无影响，prefill 留待引擎定方案）。

---

## 6. 实现交接清单（推理引擎工程师 → 引擎实现）

1. **Converter**：新增 Path B per-channel INT8 权重输出（`convert_qwen.py` 的 pc8 直写，非 INT4），每层 14.91MB。
2. **matmul 核**：复用 per-chunk INT8 两遍法（KG=256 / N_tile=112），KG=128 / N_tile=192 为编译期回退。
3. **cmdbuf**：按 rshift 模板预建（16 变体 × 7 shape），submit 仅 patch/select rshift + Run；MemInvld 限定输出区。
4. **ION**：per-matrix 双缓冲 10.3MB（§4 C）。
5. **流水线**：SD ║ CPU acc ║ TIU 三阶段；decode 上界 SD-bound ~16.3s/token（权重），LM head（+137MB/token）留 Phase 6（top-k / embed 驻留 / 带宽）。
6. **门禁**：上板 3-prompt 回归（min gap < 0.05 或 miss → 回退 KG=128）；扩展评测由 CEO/团队定集。

---

## 8. ADDENDUM（2026-08-13 晚 · CEO 并行收敛后裁定）

本报告原始结论（Path B / KG=256）为 CEO 交办的"① 路径建议"交付物。CEO 并行收敛后
**生产主路径正式定为 A'（INT4 G32 K-aligned 存储 → CPU weight-side dequant 到 per-channel
INT8 + per-chunk 两遍法），KG=32 基线**。记录如下：

- **A' 质量（group_int4，最强余量）**：gap 1.923 / 1.105 / 0.485，min gap **0.485**，
  优于 Path B 任意 KG（KG=32 0.524 / KG=256 0.189）。KG=32=G，与 K-aligned INT4 布局天然一致。
- **A' 带宽优势**：SD 201MB/token（INT4）→ 流式上界 **15.4s/token（21.9MB/s 上板实测保守，门禁用）**，
  乐观参考 11.6s（29MB/s）；标注 **[11.6, 15.4]s/token**。Path B 358MB/token ≈ 22.6s/token。
- **A' ION**：INT4 双缓冲 16.8MB + per-chunk int8 解包工作缓冲 ≈ **18.9MB < 24MB**（`REPORT_PATHA_KG32` §3 方案 A），
  余 ~5.1MB。Path B 的 per-matrix 双缓冲 10.3MB 不适用。
- **A' 复用存量**：`weights/ layerN_i4.bin` / `weights_kal/ layerN_kal.bin` 已在盘，无需新 converter；
  CPU weight-side dequant 用 RVV（上板实测 419MB/s，非瓶颈）。
- **共享核心**：A' 与 Path B 的 **per-chunk 两遍法 matmul 语义完全一致**（pass1 安全 rshift 回读 →
  r_opt → pass2 → CPU fp32 累加，round-half-up）。本报告 §6 实现约束（cmdbuf 预建 / MemInvld 限定 /
  CPU round-half-up 公式）对 A' 原样成立；仅 KG 取 32、ION 按 18.9MB 布局。
- **本报告定位调整**：Path B（KG=256/128）降级为**文档化备选/对比基线**；A' 为生产主路径，
  上板首里程碑由引擎按 A' spec 回报。

---

## 7. 关联工件

| 工件 | 内容 |
|---|---|
| `chunk_sweep.json` | KG=32/64/128/256 决定性扫参（12/12 ok） |
| `p0_refs.json` | bf16 / int4 / 早期 i8 参考 gap |
| `REPORT_PATHB_CHUNK_20260813.md` | TPU 上板：两遍法 BIT-EXACT、延迟、N-tile 上限、ION ≈10.3MB |
| `REPORT_PS32_20260813.md` | ps32 判负终裁 + 两遍法出口确认 |
| `REPORT_PATHA_KG32_20260813.md` | Path A 四闸口（备查，路径已关闭） |
| `GATE_A_SIGNOFF_20260813.md` | round-half-up 语义签核 |
| `pathb_chunk_probe.c` / `patha_kg32_check.c` / `ps32_probe*.c` | 上板探针（入库） |

**决策状态：收敛完成。待 CEO 签核 → 引擎开工（converter + matmul 核 + ION 布局）。**
