# per-row 激活 scale 启动状态 — A/B 共通项（Phase 4→5 衔接）

日期：2026-08-13 | 作者：推理引擎工程师 | 状态：**host + C906B 骨架双层 3/3 闭环，转引擎接入**
范围：CEO「确认 + 并行授权」第 2a 项（A/B 共通项 per-row 激活 scale，0.5-1d）
护栏：全在 experiments/（untracked），不碰 master。

---

## 0. 结论摘要

**per-row 激活 scale 共通项质量判定：A' 与 B 双路径均 3/3 通过（engine-realistic 两遍法）。**
启动所需 host 侧验证与 C906B 骨架（qwen_kal_ref.c）已全部就绪，下一步即引擎接入
（C906B TIU 微内核 per-row quant + 两遍法），属 Phase 5 引擎接入范畴。

**同时发现并修正一处误导性 emulation 伪影**：`emu_perrow.py` 的单遍固定 rshift
（rshift=matmul_rshift(K)-5）模型精度过低（A' 0/3、B 1/3，见 perrow_ref.json），
**不代表引擎真实质量**。引擎实际用两遍法数据自适应 rshift，该路径下 A'、B 均 3/3。

## 1. 质量判定数据（engine-realistic = per-row 激活 + 两遍法 + per-chunk 自适应 rshift）

| 路径 | 权重来源 | per-row 激活 | matmul | verdict | gap |
|---|---|---|---|---|---|
| **A'** | K-aligned INT4 G32 → per-ch INT8（weights_kal） | ✓ | 两遍法 KG=128 | **3/3** | 1.794/0.961/1.010 |
| **B** | bf16 直存 per-ch INT8 | ✓ | 两遍法 KG=128 | **3/3**（本轮新增） | 0.338/0.688/0.046 |
| A' 单遍（伪影） | 同上 | ✓ | 单遍 rshift=12 | 0/3 | — |
| B 单遍（伪影） | 同上 | ✓ | 单遍 rshift=12 | 1/3 | — |

- A' 两遍法数据：`wsdq_ref_tiu.json`（kal_nat / kal_bf16lsc，均已入 A' 裁定 5085dec）。
- **B 两遍法为本轮新验证**：`emu_b_perrow_twopass.py` + `b_perrow_twopass_ref.json`，
  复用 emu_wsdq 的 forward_wsdq（两遍法语义）+ emu_perrow.load_weights_B（bf16 直存 per-ch INT8）。
- C906B 骨架（`qwen_kal_ref.c`，per-row + 两遍法）本轮编译重验 **3/3**：
  NEXT 2130/12095/99366，gap 0.689/0.680/0.260（与 REPORT_WS_DQ_VETO §9 一致）。

## 2. 伪影根因（为何单遍 1/3 不可信）

`emu_perrow.py::matmul_i8_perrow` 用固定保守 rshift：
`rsh = max(matmul_rshift(D)-5, 8)` = 12（D=896）。理论累加上限 ~14.4M >> 12 ≈ 3527，
远超 int8 上限 → `int8_round_div` 大面积饱和 → logit 塌缩 → 0/3、1/3。

两遍法则不同：pass1 用安全 rsafe 估真实 max → pass2 用数据自适应 r =
ceil(log2(est/127))，把真实累加映射到 int8 满幅（相对误差 ~0.4%）→ 3/3。

**含义**：per-row 激活 scale 本身精度充足；关键是 matmul 必须走两遍法
（A'、B 共用，PS32_PER_GROUP_VERDICT / GATE_A_SIGNOFF 已锚定该语义）。

## 3. 启动状态与下一步（0.5-1d 映射）

已就绪（本任务完成）：
- [x] host 质量判定：A' 3/3（既有）+ B 3/3（本轮新增）→ per-row 共通项双路径成立。
- [x] C906B 骨架 qwen_kal_ref.c：per-row quant + 两遍法，3/3（本轮编译重验）。
- [x] 伪影修正：emu_perrow.py docstring 标注单遍不可用；新增 b_perrow_twopass_ref.json。

待引擎接入（Phase 5，CEO 09:35:37 已派发）：
- [ ] C906B TIU 微内核：per-row quant（M=1 时即单标量 max|h|/127）+ KG=128 两遍法
      rshift 提交（pass1 rsafe → DDR 回读 max → pass2 r_opt）。
- [ ] pf_worker 解包路径 per-ch INT8 入 ION + per-row scale 缓冲。
- [ ] 实测 C906B 反量化吞吐与 TIU 两遍延迟，确认与 SD 读 289ms/layer 重叠。

## 4. 产物（均 untracked）

- `emu_b_perrow_twopass.py`：B + per-row + 两遍法 host 测试
- `b_perrow_twopass_ref.json`：结果（3/3，gap 0.338/0.688/0.046）
- `emu_perrow.py`（docstring 已修正单遍伪影说明）
- `qwen_kal_ref.c`（既有，本轮编译重验 3/3）
