# Step ① per-row 激活 scale — 引擎接入集成报告

日期：2026-08-13 | 作者：推理引擎工程师 | 关联：`per_row_quant.c` + `test_per_row_quant.py`
状态：**引擎可复用模块 + 语义锁定 3/3 通过**（C selftest / 随机 BIT-EXACT / 真实激活 BIT-EXACT）
护栏：全部 untracked（experiments/qwen_onboard/），不碰 master。

---

## 0. 结论摘要

CEO 裁决实施顺序第 ① 步（per-row 激活 scale，公共项，0.5d）**已完成集成**：

- 将 `qwen_kal_ref.c::quant_per_row()` 隔离为独立引擎模块 `per_row_quant.c`，提供
  `per_row_quant()`（M≥1，prefill）与 `per_row_quant_decode()`（M=1 快速路径，decode）双入口。
- 语义锁定（与 host C 参考 / numpy 逐位一致）：
  `sc[m] = max|x[m,:]|/127（下限 1e-12）`；`q[m,k] = clamp(round_bankers(x/sc), -128, 127)`，
  其中 `round_bankers = round-half-to-even`（numpy np.round parity，非 round-half-away）。
- 独立验证 **3/3**：① C `--selftest` 边界用例（零行 / 半整数 / 饱和 / decode fastpath）；
  ② 随机 [1×896, 3×896, 10×896, 1×1024, 7×4864] 与 numpy **BIT-EXACT**；③ Qwen layer-0
  rms_attn 真实激活（P0 prompt 1，3×896）与 numpy **BIT-EXACT**。
- 该模块为 A/B 共通项，C906B TIU 微内核可直接 include/编译，无需改语义。

## 1. 引擎接入点（调用位置）

| 位置 | 左操作数 | M | K | 说明 |
|---|---|---|---|---|
| post rms_attn → q/k/v/up/gate | h | M=1(dec)/≥1(prefill) | 896 | 公共 quant 一次，7 个 matmul 复用 |
| post attn → wo | attn | 同 | 896 | 公共 quant 一次 |
| post SiLU → down | mid 切片 | 同 | 1024（per K-chunk） | 每 chunk 一次 |

- decode（M=1）走 `per_row_quant_decode`：免 m 循环，单标量 scale，896 元素量化 ~1-2µs 级。
- prefill（M≤10 chunk）走 `per_row_quant`，行间互不依赖，可 RVV 化（max 归约 + 除/round 循环）。

## 2. 验证数据

```
== step ① per_row_quant integration ==
[per_row_quant selftest] PASS (RC=0)
   ok  round_bankers(2.5)=2  3.5->4  4.5->4  5.5->6  -2.5->-2  -3.5->-4   （half-even）
   ok  sat +1000->127 / -1000->-127
   ok  63.5 half->even(64)
   ok  decode fastpath == generic (M=1)
[1/3] C selftest PASS
[2/3] random [1x896,3x896,10x896,1x1024,7x4864] BIT-EXACT vs numpy
[3/3] real rms_attn layer0 [3x896] BIT-EXACT vs numpy
== step ① per_row_quant integration: PASS 3/3 ==
```

## 3. 关键语义决策（防引擎回归）

- **round-half-to-even**：与 numpy np.round 一致（C ref 已按此实现并经 3/3 全链验证）。
  若引擎误用 round-half-away，`x/sc` 恰好落在 .5 的值会差 1 LSB；虽然对 logit 影响微小，
  但会破坏与 C ref 的逐位对照——故显式锁定。
- **scale 下限 1e-12**：全零行避免除 0（引擎不应假设激活恒非零）。
- **饱和 clamp ±127**：`x/sc` 理论 ≤127（sc 由行 max 定义），但 float 舍入可能越界 1，
  clamp 兜底。

## 4. 产物（untracked）

- `per_row_quant.c`：引擎可复用模块（host gcc 与 riscv64-unknown-linux-musl-gcc 同源编译）。
- `test_per_row_quant.py`：3/3 验证驱动（`python3 test_per_row_quant.py`）。
- `DESIGN_PATH_A_TWOPASS_20260813.md` §9a/§9b：CEO 两处设计确认（ION 布局 / SD 掩盖）。

## 5. 下一步（step ②）

两遍法 per-chunk matmul（KG=32）C906B 微内核重构：以 `qwen_kal_ref.c::chunk_matmul_twopass`
为语义骨架，映射到 TIU per-chunk 两遍提交（pass1 rsafe → 回读 max → pass2 r_opt），
配合 6a/9b 的 DQ_BUF 双缓冲与 cmdbuf 预建预算。待 TPU 工程师闸口对齐回执。
