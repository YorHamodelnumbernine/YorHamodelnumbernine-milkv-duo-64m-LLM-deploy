# A' Weight-side Dequant 一票否决测试 + ION 架构对齐（Phase 4 决策点）

日期：2026-08-13
作者：推理引擎工程师
状态：**纯 host 验证，未上板、未接引擎、未提交 git**（experiments/ 全 untracked）
范围：CEO 阶段 4 决策点任务 1（一票否决）+ 任务 2（ION 缓冲架构对齐）

---

## 1. 结论摘要

**① 一票否决：通过（3/3）。** 固定配置「K-aligned INT4 G32 → CPU 解包×group scale →
per-channel INT8 权重 + per-row 激活 INT8 + per-channel fp32 后处理 → 普通 INT8 matmul」
3 个固定 prompt 与 host INT4 参考（2130 / 12095 / 99366）**全部一致**。

**关键发现：布局轴（K-aligned vs FLAT）决定成败，而非 per-channel scale 启发式。**
- K-aligned 布局下 4/4 配置全 3/3（含 engine-realistic 的 stored layer_scales.bin + 两遍法）。
- FLAT 布局下 4/4 配置全 0/3（logit 塌缩 gap 0.03–0.42）→ 这正是 P0 FAIL 的同一风险区。

**② 路径裁定依据成立：A'（weight-side INT4）保留可行。** SD/token 201MB（vs B 358MB，-44%），
工程增量集中在权重加载路径，matmul 核与 B 完全共用。

**③ ION 架构：FIT。** int4 层文件（8.39MB）+ int8 双缓冲工作槽（2×4.36MB）+ 激活/累加
（~1.5MB）+ 小矩阵槽（~1.8MB）≈ **20.6MB < 24MB**；流水线 SD 主导（289ms/layer），
CPU 解包（~40ms/layer）与 TIU（~5-25ms/layer）全隐藏，不空转。

---

## 2. Task 1 验证数据（emu_wsdq.py，host）

| 配置（INT4 布局 → per-ch INT8 方式） | exact int32 累加 | 两遍法 KG=128（engine-realistic） |
|---|---|---|
| **kal_nat**（K-aligned, scale=max\|W_dq\|/127） | **3/3** gap 1.420/0.933/0.563 | **3/3** gap 1.534/1.623/0.799 |
| **kal_bf16lsc**（K-aligned, bf16-derived layer_scales.bin） | **3/3** gap 0.784/1.245/0.574 | **3/3** gap 1.546/1.264/0.877 |
| flat_nat（FLAT, scale=max\|W_dq\|/127） | 0/3 gap 0.041/0.028/1.162 | 0/3 gap 0.415/0.136/0.264 |
| flat_bf16lsc（FLAT, bf16-derived scales） | 0/3 | 0/3 |

- **参考**：host INT4（int4_g32_bf16）gap 1.25/0.75/0.75；Path A per-group 两遍法
  （group_int4）gap 1.923/1.105/0.485。
- **engine-realistic 配置（kal + stored layer_scales.bin + 两遍法）gap 1.546/1.264/0.877，
  优于 host INT4 参考**，与 Path A per-group 相当。余量健康。

## 3. 根因：为什么 K-aligned 折叠安全、FLAT 塌缩

per-channel INT8 折叠要求**每输出列的组内动态结构可被单个列 scale 表达**：
- **K-aligned**：每个 G=32 组 = 单一输出列的 K 切片 → group scale 是列局部量 → 反量化后
  每列仍保留 bf16 的形状，per-ch INT8 只需重新满幅化（相对误差 ~0.4%）。
- **FLAT**：每个 G=32 组跨 32 个不同输出列（单行内 N 段）→ group scale 混合多列量纲 →
  反量化引入跨列畸变，per-ch INT8 无法修复 → logit 塌缩。

这与 P0 FAIL「5 种 per-channel scale 启发式全 0/3」不矛盾：那是在 **FLAT** 布局下的结论，
任何 scale 启发式都无法挽回；**修法在布局轴**。K-aligned 转换器（convert_qwen_kal.py）
已交付且字节级验证过，因此 A' 无需新转换工作。

## 4. 生产路径一致性校验

- convert_qwen_kal.py 产物 weights_kal/layer_scales.bin = **INT4-dequant 的 per-ch scale**
  （= 本测试 kal_nat），与 host 实测逐字节一致（max diff 2.9e-6，fp32 舍入级）。
- 即：**设备端只需读 layerN_kal.bin + layer_scales.bin，CPU 解包 int4→int8 时用 stored
  per-ch scale 即可**，无需设备端计算列 max。kal_bf16lsc 亦 3/3，两条 scale 源均可。

## 5. A'（weight-side INT4） vs B（per-ch INT8 直存）对比

| 维度 | A'：weight-side INT4 | B：per-ch INT8 直存 |
|---|---|---|
| 权重文件 | 201.4MB（8.39MB/layer） | ~495MB（14.91MB/layer） |
| SD / token | 201MB（**-44%**，主导瓶颈） | 358MB |
| matmul 核 | 复用 per-ch INT8 两遍法 + per-row act（公共，已 3/3） | 同左 |
| 权重加载 | pf_worker 读 int4 → CPU 解包×per-ch scale → ION 槽（~40ms/layer，隐藏于 SD 289ms） | 读 int8 直载 ION 槽（更简单，无解包） |
| ION | int4 层 8.39MB + int8 槽 2×4.36MB + 激活 ≈20.6MB | int8 槽 2×4.36MB + 小矩阵 + 激活 ≈12.5MB |
| 转换器 | convert_qwen_kal.py 已交付（+layer_scales.bin 已含） | 需改 convert_qwen.py 直写 per-ch INT8（~0.5d） |
| 新增工作量 | ~1–1.5d（解包入加载路径 + ION 重排 + 集成） | ~0.5–1d（直载 + ION 重排 + 集成） |
| 风险 | CPU 解包需被 SD 掩盖（估算可，需上板测 C906B 吞吐）；int4 文件与 scale 一致性 | ION 更宽裕；风险低；但 SD 大 1.78x |
| **合计** | **~2–2.5 人日** | **~1.5–2 人日** |

> 注：B 的 matmul/质量侧已在此前 Phase 5 验证（per-ch INT8 + per-chunk 两遍法 3/3）。
> 两者 matmul 核同源，差额仅权重加载与文件格式，工期差 ~0.5–1d。
> 注（decode 口径校准，2026-08-13）：表中 SD/token 为**权重-only**口径（A' 201MB / B 358MB）。
> A' 含 LM head embed 后 = **337MB/token** → SD 流式上界 [11.6, 15.4]s/token，门禁取 15.4s
> （实测 21.9MB/s，vs 设计 29MB/s）；详见 §6.4。

## 6. ION 24MB 缓冲架构结论（A' 路线）

### 6.1 精确尺寸（Qwen2.5-0.5B）

| 项 | 大小 |
|---|---|
| int4 层文件（layerN_kal.bin） | 8.39MB |
| int8 工作槽 up/gate/down 双缓冲（2×4.36MB） | 8.72MB |
| 小矩阵 int8（Wq/Wk/Wv/Wo，可复用 FFN 槽时序或单独） | 1.76MB |
| 激活/累加（x/h/attn/qkv/mid/out fp32） | ~1.5MB |
| pass1/pass2 int8 输出 + max 缓冲 | ~0.2MB |
| **合计** | **≈20.6MB < 24MB（余量 ~3.4MB）** |

### 6.2 共存与流水线（不空转判定）

- **int4 层文件单缓冲在 ION**：SD 读下一层 → DDR 暂存（8.39MB，64MB DDR 内），当前层
  解包完成后 memcpy DDR→ION（~10-20ms）→ 不与 TIU/CPU 争用 ION。
- **流水线重叠**：SD 读层 i+1（8.39MB @29MB/s ≈ **289ms**）‖ CPU 解包层 i 到 int8 槽
  （~40ms）‖ TIU 两遍 matmul（~5-25ms）。**SD 主导，CPU/TIU 全隐藏，无空转。**
- 备选更优：int4 层文件直接驻留 DDR（SD 目标），CPU 解包 DDR→ION int8 槽，ION 仅
  ~12.5MB，余量更大（利于 KV 扩容）。CEO 指定的「int4 在 ION」亦可满足（20.6MB FIT）。
- 余量 ~3.4MB 建议预留 KV（配合 INT8 KV，约 500+ token 上下文）。

### 6.3 TIU 提交量（两遍法 KG=128，per-ch INT8）

| matmul | K-blocks | N-tiles | ×2pass |
|---|---|---|---|
| q 896×896 | 7 | 2 | 28 |
| k/v 896×128 | 7 | 1 | 14 各 |
| wo 896×896 | 7 | 2 | 28 |
| up/gate 896×4864 | 7 | 10 | 140 各 |
| down 4864×896 | 38 | 2 | 152 |
| **合计/layer** | | | **516**（×24 ≈ 12,384/token） |

- @2μs/submit ≈ 25ms/token TIU，远小于 SD 流式上界（见 §6.4），可被完全掩盖。
- 若需更低 CPU/更高稳健可降 KG=32（≈49.5K submit/token，~100ms，仍隐藏）。

### 6.4 decode 口径校准（2026-08-13 CEO 定稿）

**带宽依据**：上板实测 **21.9MB/s**（pathb_chunk_probe / patha_kg32 P3）；29MB/s 为设计
估值，仅作乐观参考。

**A' 每 token 读量**：权重 201.4MB + LM head embed 136.1MB = **337MB/token**（§5 表中
201MB 为权重-only 口径，未含 embed）。

**SD 流式上界**：
- 实测保守：337 / 21.9 ≈ **15.4s/token**（**里程碑门禁用此值，避免乐观化**）
- 设计乐观：337 / 29 ≈ **11.6s/token**（仅上限参考）
- 部分权重驻留 / 带宽优化（Phase 6）可下移 → **标注「SD 流式上界 [11.6, 15.4]s/token，
  门禁取 15.4s（实测）」**

**片上计算管线（非 decode 上界）**：TIU + CPU（解包/累加/提交）摊销后约 **350–500ms/token**；
batch per-op 3.7μs 口径（PS32 probe21）下保守 80,544 op/token ≈ 0.30s，全被 SD 掩盖。
（对照：REPORT_PATHA_KG32 §2.2 的 3.37s/token 为 naive per-submit 最坏值，摊销后不适用。）

**Path B 对照**：358MB/token @21.9 ≈ **16.3s/token**（REPORT_PATHB_CHUNK §2.2 口径）。

---

## 7. 产物清单（均 untracked）

- `emu_wsdq.py`：A' 一票否决实验（kal/flat × nat/bf16lsc × exact/twopass128，8 配置）
- `wsdq_ref.json`：完整结果（8 配置 × 3 prompt next_token/gap/top5）
- `wsdq_run.log`：运行日志

## 8. 结论

**A'（weight-side INT4 → per-ch INT8）host 质量 3/3 通过，CEO 可据实推进 INT4 338MB 保留。**
唯一硬前提：**必须用 K-aligned 布局**（convert_qwen_kal.py 已交付），FLAT 布局判负（0/3）。
IONS 架构 FIT 且不空转。

> 闸口状态更新（CEO 裁定 5085dec 纠正）：**Gate①（pass1 int8 回读）与 Gate④b（N-tile=512）
> 已在 commit 5d00f05 由 TPU 工程师上板签核通过**，引擎接入无阻塞。A' 已由 CEO 裁定为
> Qwen 生产路径（REPORT_A_PRIME_RULING_20260813.md）。

## 9. TIU round-half-up 舍入对齐（2026-08-13 补验）

CEO GATE 派发要求 CPU 侧舍入与上板 TIU 一致：`sat8((acc + (1<<(rshift-1))) >> rshift)`
（round-half-up 向 +∞，负半值 tie 归 0；commit 23f300e 全范围确认）。

- **原 emulator / C 参考用 round-half-away-from-zero**（`-((-acc+half)/scale)`），在**负半值
  tie 点**与 TIU 不同（如 -16>>5：TIU=0 vs away=-1）。
- **已修正**：`qwen_emu.py::int8_round_div` 与 `qwen_kal_ref.c::int8_round_div` 均改为
  TIU 公式（算术右移）。
- **重验结果（wsdq_ref_tiu.json）**：A' 判定**不变**——
  - kal_nat 3/3（exact 1.420/0.933/0.563；twopass128 1.057/1.598/0.291）
  - **kal_bf16lsc（engine-realistic）3/3**（exact 0.784/1.245/0.574；twopass128 **1.794/0.961/1.010**）
  - flat_nat / flat_bf16lsc 全 0/3（塌缩不变）
- **C 参考重编译重验 3/3**：NEXT 2130/12095/99366，gap 0.689/0.680/0.260。
- **结论**：A' 质量判定对 TIU 舍入语义稳健，裁定不受影响。

## 10. 产物清单（均 untracked，CEO 已 commit 5085dec 入库 A' 证据）

- `emu_wsdq.py`：A' 一票否决实验（kal/flat × nat/bf16lsc × exact/twopass128，8 配置）
- `wsdq_ref.json`（原始）/ `wsdq_ref_tiu.json`（TIU 舍入对齐后重验）
- `wsdq_run.log` / `wsdq_run_tiu.log`：运行日志
- `qwen_emu.py` / `qwen_kal_ref.c`：舍入已对齐 TIU 公式
