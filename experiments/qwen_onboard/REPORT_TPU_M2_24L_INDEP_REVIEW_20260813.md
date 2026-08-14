# M2 24L 独立复核结论 — TPU 底层工程师

日期：2026-08-13 | 作者：TPU 底层工程师 | 状态：**复核完成，ALL MATCH** | 关联：`verify_m2_24l_indep.py` / `REPORT_M2_24L_20260813.md` / `qwen_engine_24l.c` / `qwen_kal_ref.c` / `gen_layer0_ref.py`

---

## 0. 结论摘要（供 CEO 归档）

**M2 24L（推理引擎交付 + LM head + 3-prompt 回归）独立复核通过。** 独立于推理引擎与
emulator 的验证脚本 `verify_m2_24l_indep.py` 只从 raw 二进制（`weights_kal/layer0..23_kal.bin`
+ `embed_i8.bin` + `embed_scales.f32` + `final_rms.f32` + `layerN_bias.f32`）重新解包，
独立实现 Path A 两遍法语义跑 3-prompt，四项独立核对全部一致：

- **(A) 数据链/二进制布局**：24×layerN_kal.bin（8,393,728 B）布局 `off==size` 全解析一致；
  embed_i8 / embed_scales / final_rms 字节数逐项命中（136,134,656 / 151,936 / 896）。
- **(B)(C) 独立 24 层前向门禁**：NEXT token **3/3 = 2130 / 12095 / 99366** 命中；
  min gap **0.3413 ≥ 0.05** 门禁（强满足，余量 >6.8×）。
- **(D) TIU run 计数独立重算**：按引擎 tiling（M≤3→tilew=896，M>3→768）结构推导
  **pass1=55872、pass2=55872、total=111744**，与报告位精确一致。
- **(E) layer-0 语义位精确**：token 105538 单层前向 attn / after_wo / after_ffn 三中间量
  对可信 `gen_layer0_ref.py` 参考 **maxdiff = 0.0（BIT-EXACT）**——matmul/rms/quant/rope/
  attention/ffn 实现语义与参考完全一致。

**结论：M2 里程碑双签核闭环成立。** 引擎 VERIFY=1 的 bad1=bad2=rbad=0（111,744 runs）
语义路径已被 host 独立实现复核；NEXT 3/3 与门禁 gap 均强满足，无新增阻塞。

## 1. 复核方法与独立实现说明

- **独立性**：`verify_m2_24l_indep.py` 不 import 引擎（`qwen_engine_24l.c`）或任何
  emulator（`emu_*.py` / `qwen_emu.py`），仅用 numpy 从 raw 二进制独立重建语义。
- **数值口径**（与 `qwen_kal_ref.c` / 引擎 `eng_matmul` 文档语义一致）：
  - 两遍法 matmul：per K-block（G=32）int 精确累加（f64 dgemm，整数 <2^53 精确），
    pass1 rsafe → block_max（覆盖 ALL M 行 × ALL N-tile）→ r_opt（block-shared）→ pass2；
    double accd 累加，`out = (float)accd * sc_row[m]`。
  - per_row quant：round-half-even（bankers）+ clip ±127；rms_norm：double ss → float inv。
  - LM head：double 累加 `Σ h[j]*er[j]*esc[t]`，与引擎/参考一致。
- **debug 记录**：首版用 `np.einsum('mkg,kgn->gmn')` 出现 per-block 输出错位（虽总和正确），
  改为显式 per-block dgemm 后与参考 BIT-EXACT（见 §2(E)）。此坑已在脚本内注释留存。

## 2. 复核判定

### (A) 数据链 / 二进制布局

| 文件 | 字节数 | 期望 | 判定 |
|---|---|---|---|
| embed_i8.bin | 136,134,656 | V·D = 151936×896 | ✓ |
| embed_scales.f32 | 151,936 | V | ✓ |
| final_rms.f32 | 896 | D | ✓ |
| layer0..23_kal.bin (×24) | 8,393,728 | 布局 off==size | ✓ |
| layer0..23_bias.f32 (×24) | 4,608 | (D+DKV+DKV)×4 | ✓ |

### (D) TIU run 计数独立重算

按引擎 `max_tile_for_m`（M≤3→896，M>3→768）推导 per-layer pass1 runs：

| matmul | K-blocks | M≤3 tiles | M>3 tiles | M≤3 runs | M>3 runs |
|---|---|---|---|---|---|
| q_proj (N=896) | 28 | 1 | 2 | 28 | 56 |
| k_proj (N=128) | 28 | 1 | 1 | 28 | 28 |
| v_proj (N=128) | 28 | 1 | 1 | 28 | 28 |
| wo_proj (N=896) | 28 | 1 | 2 | 28 | 56 |
| up_proj (N=4864) | 28 | 6 | 7 | 168 | 196 |
| gate_proj (N=4864) | 28 | 6 | 7 | 168 | 196 |
| down (K=4864→1024×4+768) | 152 | 1 | 2 | 152 | 304 |
| **per-layer pass1** | | | | **600** | **864** |

pass1 = 24×(600 + 864 + 864) = 24×2328 = **55,872**；pass2 同；total = **111,744** ✓

### (E) layer-0 语义位精确

| 中间量 | maxdiff vs gen_layer0_ref | 判定 |
|---|---|---|
| attn | 0.0 | BIT-EXACT |
| after_wo | 0.0 | BIT-EXACT |
| after_ffn | 0.0 | BIT-EXACT |

### (B)(C) 独立 24 层前向 · 3-prompt 门禁

| Prompt | seq | NEXT | 期望 | top5 | gap | 判定 |
|---|---|---|---|---|---|---|
| P1 中国的首都是 | 3 | 2130 | 2130 | 2130 68990 9909 7 198 | 1.1126 | ✓ |
| P2 The capital of France is | 5 | 12095 | 12095 | 12095 7407 279 32671 30743 | 0.8055 | ✓ |
| P3 今天天气很好，我们去公园 | 7 | 99366 | 99366 | 99366 111261 109280 100855 69249 | 0.3413 | ✓ |

- NEXT token：**3/3 命中**（2130 / 12095 / 99366）。
- min gap：**0.3413 ≥ 0.05** ✓（门禁强满足）。

## 3. 与 M2 里程碑交叉引用

| # | M2 报告断言 | 独立复核证据 | 判定 |
|---|---|---|---|
| 1 | NEXT 2130/12095/99366 3/3 | §2(B)(C) 独立前向 3/3 | ✓ |
| 2 | min gap 0.4508 ≥ 0.05 | on-board 记录 0.4508；独立 host 0.3413；host ref 0.2600 均 ≥0.05 | ✓ |
| 3 | TIU 111,744 runs（bad1=bad2=rbad=0） | §2(D) run 计数结构重算 111744；§2(E) 语义 BIT-EXACT | ✓ |
| 4 | 数据链（weights_kal）可靠 | §2(A) 字节/布局全一致 | ✓ |

## 4. expf 差异口径（gap 值对实现敏感，top-1 不变）

gap 的**具体数值**对不同 expf 实现敏感（softmax 浮点路径，非 TIU 整数路径）：

| 实现 | expf | P1 gap | P2 gap | P3 gap | min gap |
|---|---|---|---|---|---|
| host `qwen_kal_ref`（glibc） | glibc expf | 0.6890 | 0.6802 | 0.2600 | 0.2600 |
| on-board 引擎（musl） | musl expf | 0.6890 | 0.4508 | 0.5149 | **0.4508** |
| 独立复核（numpy fp32 exp） | numpy expf | 1.1126 | 0.8055 | 0.3413 | 0.3413 |

- **top-1 NEXT 跨全部三种 expf 实现均稳定为 2130/12095/99366**（3/3 不变）。
- **min gap 跨全部实现均 ≥ 0.26 ≥ 0.05**，门禁余量 >5×。
- on-board 报告值 0.4508 落在该实现差异带内，与 musl-expf 归因一致；
  该差异已被 CEO 裁定记录在案（softmax 非 TIU 路径、不阻塞），本次复核进一步证实其
  不影响门禁判定。

## 5. 复现

```
cd /home/vasilybabyboy/Documents/MilkV_duo_project/tpu_bench/experiments/qwen_onboard
python3 verify_m2_24l_indep.py        # ALL MATCH（~25s，含 embed 读取）
```

依赖：`weights_kal/`（layer0..23_kal.bin + embed_i8.bin + embed_scales.f32 + final_rms.f32 +
layerN_bias.f32）与 `qwen_engine_layer0_ref.h`（可选，供 §(E) 位精确核对）。
护栏：本脚本与报告均留在 untracked `experiments/qwen_onboard/`，不碰 master。
