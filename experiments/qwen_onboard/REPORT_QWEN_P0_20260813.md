# Qwen2.5-0.5B On-board P0 前置验证报告（Phase 4 / Option A）

日期：2026-08-13
作者：推理引擎工程师
状态：**纯 host 验证，未上板、未接引擎、未提交 git**（experiments/ 全 untracked，master 未动）
范围：Qwen2.5-0.5B-Instruct on-board 最小可运行路径的 P0 前置判定

---

## 1. 结论摘要

**固定配置（INT4 G32 存储 → 片上 per-channel INT8 计算）在 host 仿真下判定 P0 FAIL（0/3）。**

根因双轴：权重轴（per-channel INT8 无法表达 INT4 组 scale 结构）+ 激活轴（现引擎 per-tensor 单激活 scale 过粗）。
两条修法均已量化验证，其中 **路径 A（K-aligned INT4 per-group dequant + per-row 激活）质量侧 3/3 通过**，等 TPU 底层工程师确认 bmk1822 可行性后由 CEO 裁定 A / B。

## 2. 已完成与验证

1. **转换器** convert_qwen.py：bf16 safetensors → 全部 on-board 权重文件（~339MB）。格式：config.bin / embed_i8.bin / embed_scales.f32 / layerN_i4.bin / layer_scales.bin / **layerN_bias.f32（新增）** / final_rms.f32。
2. **修复隐藏 bug**：Qwen2.5 q/k/v 投影 **带 bias**（q:896, k:128, v:128；o_proj 与 MLP 无 bias）。此前转换器/仿真器只取 weight，导致 q_proj 输出偏 10x。已补 bias 提取与写盘、仿真器加回 bias。
3. **仿真器数值正确性**：qwen_emu.py 与 torch 差分测试——同一套 per-channel INT8 量化权重分别喂 torch 与仿真器，L0 q_proj corr=0.99998（norm 404 一致），整网 next_token 一致（差分测试通过）。

## 3. P0 验证数据（3 个固定 prompt）

| prompt | bf16 | host INT4 G32 (bf16 dequant) | 片上预测① fp32act | 片上预测② 全INT8act |
|---|---|---|---|---|
| 中国的首都是 | 2130 '____' | **2130 ✓** | 20412 '是' ✗ | 89455 'dba' ✗ |
| The capital of France is | 12095 ' Paris' | **12095 ✓** | 60119 'afür' ✗ | 30709 '小' ✗ |
| 今天天气很好… | 99366 '玩' | **99366 ✓** | 26288 '大' ✗ | 28691 'precedented' ✗ |

- host INT4（bf16 dequant of flat INT4 G32）：3/3 与 bf16 一致（bf16 gap 0.375/0.875/0.125）。
- 片上预测②为真实引擎语义（per-tensor INT8 激活 + rshift，与 smollm2_pool_demo.c 的 compute_scale_sym/matmul_rshift 完全一致）。
- **固定配置 0/3 判负。**

## 4. 根因

### 4.1 权重轴：per-channel INT8 无法表达 INT4 组 scale 结构
- INT4 G32 组 scale 在单输出通道内中位动态范围 ~80x。
- per-channel INT8 对 INT4-dequant 重建误差仅 +0.96% relrms（总量 11.8%≈host INT4），但误差模式不同，足以翻转 razor-thin 的 bf16 决策（gap 0.125–0.875）。
- 已试 5 种 per-channel scale 启发式（max/127、rms/127、2rms/127、rms/40、asymmetric）全部 0/3 → **非启发式可修**。

### 4.2 激活轴：per-tensor 单激活 scale 过粗
- 现引擎 compute_scale_sym = max/127 覆盖整个 seq×D 张量。
- 即使权重换成直接 per-channel INT8（fp32act 时 3/3 通过），per-tensor INT8 激活仍 0/3。

## 5. 完整质量矩阵

| 权重表示 | 激活表示 | next_token（3 prompt） | 备注 |
|---|---|---|---|
| INT4 flat → per-ch INT8（原固定配置） | per-tensor INT8 | 0/3 ✗ | 引擎现状 |
| 直接 per-ch INT8（bf16） | fp32 | 3/3 ✓ | 仅权重侧 |
| 直接 per-ch INT8（bf16） | per-tensor INT8 | 0/3 ✗ | |
| K-aligned INT4 per-group | per-tensor INT8 | 0/3 ✗ | |
| **K-aligned INT4 per-group** | **per-row INT8** | **3/3 ✓** | 路径 A |

路径 A（K-aligned per-group + per-row 激活）3 个 prompt 的 gap：1.366 / 1.312 / 0.962，比 host INT4（1.250/0.750/0.750）更稳。

## 6. 关键发现：组布局陷阱

现有 INT4 文件由 q4.py 的 group_quantize 产生，是 **FLAT 布局**（对展平的 [K,N] 矩阵按每 G=32 连续元素分组）。在引擎 [K,N] 布局下（N=896>G=32），每组实际是「单行内 32 个连续 N 列的段」，不是 K 维切片。
→ **与 K 对齐的 per-group matmul 不兼容**（输出列跨组，组 scale 无法按 K 切片分解）。
→ 路径 A 必须改转换器为 **K-aligned 布局**（scale[K/G,N]，每组=同一输出列的 K 切片）。体积不变（scale 数量同为 K*N/G），质量已验证保持 3/3。

## 7. 待定决策（A / B）

| | 路径 A：K-aligned INT4 per-group | 路径 B：直接 per-ch INT8（bf16） |
|---|---|---|
| 配置 | 保留 INT4 G32（~338MB，SD 8.39MB/layer/token） | 改配置（~495MB，SD 14.9MB/layer/token，~1.8x） |
| 质量 | 3/3 已验证 | 3/3（仅权重侧；需 per-row 激活） |
| matmul | 需新 per-group dequant（bmk1822 ps32 可行性 TBD） | 复用现有 per-ch INT8 路径 |
| ION 缓冲 | INT4 layer 8.39MB → 双缓冲 16.8MB **FIT 24MB** | INT8 layer 14.9MB → 双缓冲 29.8MB **>24MB，需子矩阵拆分** |
| 激活 | per-row（必需） | per-row（必需） |
| 主要风险 | TPU per-group matmul 性能（submit 数：down K=4864=152 组/层） | ION 拆分复杂 + SD 更大 |

## 8. 产物清单（均 untracked）

- `convert_qwen.py` / `qwen_emu.py` / `weights/`（含 layerN_bias.f32）
- `p0_refs.json`（bf16 / host INT4 / emu per-ch / emu int8act 的 3 prompt 结果）
- 验证脚本：/tmp/host_ref3.py、/tmp/dbg_l0c.py、/tmp/dbg_emu_vs_torch.py、/tmp/scale_test.py、/tmp/pc8_from_bf16.py、/tmp/emu_pathA2.py

---

# Phase 5 补充（2026-08-13 下午）：ps32-free 两遍法 → A/B 双路 3/3 通过

## 1. 决定性结论

**ps32（int32 累加输出）不是必需的。** 发现 ps32-free 两遍法：
per-chunk K 切分 matmul + 两遍 rshift（pass1 安全 rshift 读 int8 输出 → CPU 算
per-chunk 真实 max → pass2 精化 rshift）→ CPU fp32 累加。
**Path A（INT4 G32 K-aligned）与 Path B（per-ch INT8）用此法均达 3/3。**

这直接回答了 spike 的担忧：即便 ps32 不可读，Qwen 3/3 仍可达。

## 2. 实验链（全部 host，Path B = 直接 per-ch INT8 from bf16）

| 配置 | next_token 3 prompt | gap 余量 |
|---|---|---|
| bf16 参考 | 3/3（基线） | 0.125–0.875（razor-thin） |
| 固定配置（per-tensor act，全 K） | 0/3 | — |
| Path B + per-row act + 全 K 单 rshift | 1/3 | — |
| Path B + per-chunk K=32 + per-row 精确 max rshift | **3/3** | 1.327/0.722/0.524 |
| Path B + per-chunk K=32 + 固定 rshift | 2/3 | — |
| Path B + 确定性 bound rshift（无读回） | 1–2/3 | — |
| **Path B + per-chunk K=128 + shared rshift（两遍法）** | **3/3** | 1.129/0.704/0.676 |
| Path A per-group INT8 + 两遍法 | **3/3** | 0.848/0.833/0.468 |
| **Path A INT4 G32 K-aligned + 两遍法** | **3/3** | **1.923/1.105/0.485** |

关键机理：
- 输出 int8 rounding 是 razor-thin logits 的根本瓶颈（hilo_rs=0/3，hilo_exact=3/3）。
- per-chunk 切分 + **真实 max 的 per-chunk rshift** 是解药；确定性上界（matmul_rshift
  类 bound）过粗（1-2/3）→ 必须两遍读回，或 per-row 拆分。
- KG=128 两遍法从 pass1 量化输出估 r_opt（±1 bit）仍 3/3；KG=256 退化 2/3 → 取 KG=128。
- 两遍法对 decode（M=1）与 prefill（M>1，shared max over rows）统一成立。

## 3. A/B 对比（两遍法下，质量侧已同为 3/3）

| | Path A：INT4 G32 K-aligned + 两遍法 | Path B：per-ch INT8 + 两遍法 |
|---|---|---|
| 质量（host） | 3/3（gap 1.923/1.105/0.485） | 3/3（gap 1.129/0.704/0.676） |
| SD / token | 24×8.39MB ≈ 201MB（**1.78x 少**） | 24×14.9MB ≈ 358MB |
| ION 双缓冲 | 8.39×2=16.8MB **FIT 24MB** | 14.9×2=29.8MB **>24MB 需子矩阵拆分** |
| TIU 提交 | KG=32=G，K=896 切 28 块 ×2 遍 | KG=128，K=896 切 7 块 ×2 遍 |
| 额外工作 | K-aligned converter 重写 + CPU INT4→INT8 dequant + per-(g,col) scale 累加 | 复用 per-ch INT8；per-col scale 累加 |
| 主要风险 | dequant CPU 开销（~180M op/token，~1-2% 余量） | ION 拆分复杂 + SD 更大 |

## 4. 推荐：Path A（INT4 K-aligned + 两遍法）

理由：SD 带宽是主导瓶颈（SmolLM2 7.2s/token 中 48% 是 SD 读）。Path A 带宽少 1.78x、
ION 双缓冲 FIT、gap 余量最大（razor-thin prompt 1.923 vs bf16 0.125）。工程增量集中在
converter K-aligned 重写 + 两遍法 matmul 重构。

## 5. 工作量估算更新（对照 2-3 人日预算）

| 项 | 估算 | 说明 |
|---|---|---|
| per-row 激活 scale（公共项） | 0.5d | host 已验证；引擎集成 |
| 两遍法 per-chunk matmul 重构（KG=32） | 1–1.5d | pass1 读 max + pass2 精化 |
| converter K-aligned 重写（FLAT→K-aligned INT4） | 0.5–1d | 体积不变，布局改 |
| CPU per-group dequant + per-(g,col) scale 累加 | 0.5–1d | 替代 per-col 累加 |
| 集成/验证 | 0.5d | 3 prompt 扩样本 |
| **合计** | **3–4 人日** | 略超 2-3 预算，无 ps32 风险 |

## 6. 产物（均 untracked，master 未动）

- `emu_perrow.py` / `emu_hilo.py` / `emu_chunk8.py` / `emu_chunk_sweep.py`
- `emu_chunk_bound.py` / `emu_chunk_shared.py` / `emu_chunk_twopass.py`
- `emu_group_twopass.py`（Path A per-group INT8）/ `emu_group_int4.py`（Path A INT4 最终配置）
- `chunk8_ref.json` / `chunk_sweep.json` / `chunk_bound.json` / `chunk_shared.json`
  / `chunk_twopass.json` / `group_twopass.json` / `group_int4.json`

---

# Phase 5b（2026-08-13 晚）：Path A 开工 — converter K-aligned 已交付并字节级验证

## 已完成
1. **convert_qwen_kal.py**：K-aligned INT4 G32 转换器（RTN 对称，每 (K-block, 输出列)
   fp16 scale）。产物 weights_kal/：**201.4MB，8.39MB/layer**（与 Path A 预估一致）。
   布局：每 matmul 按 K-block-major 存 [N][16] packed nibble + [N] fp16 gscale，
   引擎可按 block 直接读+反量化。
2. **verify_kal_roundtrip.py**：从 layerN_kal.bin 字节级回读 → 两遍法 forward
   **3/3**（gap 1.283/1.470/0.050，含 on-board embed 量化）。
3. **pack/unpack 无损**：q 值完全一致；gscale fp16 舍入 ≤5.2e-5。
4. **DESIGN_PATH_A_TWOPASS_20260813.md**：两遍法 matmul 微内核 + per-group 反量化 +
   ION 布局 + 流水线重叠设计。

## 设计要点（给 TPU 工程师闸口评估）
- ION 用 **per-matrix 双缓冲**（非 per-layer）：最大矩阵 2.45MB → 权重缓冲仅 4.9MB，
  全引擎 ≈7.2MB < 24MB（余量充足）。
- TIU 提交：**≈2,064/layer、≈49,500/token**（含两遍×2 与 N-tile 细化）——比之前估的
  640/layer 高，需闸口②确认可被 SD 读（~6.9s/token 主瓶颈）掩盖。
- CPU 关键路径是 INT4→INT8 反量化（~20-25G op/token ≈2-3s 顺序），必须与 SD 读重叠；
  若实测成瓶颈，降级选项为 SD 直存 per-group INT8（16.8MB/layer）换带宽。
- **LM Head flag**：embed 136MB >> ION，每 token 流式读 136MB ≈4.7s（29MB/s），
  decode 预算升至 ~11.6s/token。需另立方案（top-k 两段式 / 热区缓存）。
