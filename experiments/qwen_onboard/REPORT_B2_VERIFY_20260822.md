# Report — Qwen Path A dequant Design B2 上板验证（新鲜复跑）

日期：2026-08-22 | 作者：TPU 底层工程师 | 状态：**验证通过**
关联：`DESIGN_QWEN_INT4_B_ION_20260822.md`（设计）/ `46eeb78`（B2 实施 commit）
序列：B2（本报告）→ A1（推理引擎，待 CEO 另行指派）

---

## 0. 结论摘要

B2（nib16[16][N] 重排 + 连续 vle8 dequant 内核）在板卡空闲窗口内完成**新鲜独立复跑**，全部验收项通过：

1. **正确性 / 位精确**：3-prompt NEXT 3/3、decode 链 2130→198→32、CHAT 75-token 链与基线逐 token 一致，TIU bit-exact 全 0。
2. **性能**：单调用 dequant 吞吐 N=896 **1.75×**、N=4864 **2.02×**（微基准实测）；端到端 CHAT 75-token prefill **↓28%**、decode **↓32%**。
3. **内存**：ION 归零、无 watchdog、无孤儿进程。

---

## 1. 验证对象与可复现性

| 项 | 值 |
|---|---|
| 实施 commit | `46eeb78`（HEAD，基线 bd5be99 之上） |
| 二进制 | `qwen_engine_int8kv_b2` sha256 `bd286ccd…` |
| 重建验证 | 从 HEAD `git archive` 重建 → sha256 与板上部署**逐字节一致** |
| 权重 | `weights_kal_b2/` 24 层（nib16 布局），板上 `layer{k}_kal.bin` sha256 与本地一致 |
| 板卡状态 | 验证前/后 ION used=0，无残留进程 |

内核签名：`dequant_kal_rvv(const uint8_t *nib, int N, int col_off, int ncols, int8_t *w)`，3 调用点（eng_matmul pass1 / pass2 / eng_matmul_merged）全部统一 nib16 基址。符号扩展沿用 `vsra(vsll(v,4),4)/vsra(v,4)`（raw int4 8..15 = -8..-1），bit-exact 由构造保持。

---

## 2. Run 1 — 回归 + decode 链 + dequant PROFILE（标准配置）

配置：`DECODE=1 DECODE_STEPS=2 PROFILE=1 RSH=1`，`VERIFY=1`（默认），LW_READ=mmap bare。

```
PROMPT 1 expected_next=2130  OK
PROMPT 2 expected_next=12095  OK
PROMPT 3 expected_next=99366  OK
total wall = 159.50s (all 3 prompts)
==== 24L regression: expected_next 3/3 OK ====
==== 24L regression: TIU internal BIT-EXACT ====

DECODE pos=3 tok_in=2130 next=198   (gap=0.4164)
DECODE pos=4 tok_in=198  next=32    (gap=4.5688)
==== decode bit-exact: bad1=0 bad2=0 r_opt=0 rsh=0 ====
==== PROFILE: eng_matmul breakdown (decode M=1) ====
eng_matmul wall : 39.456s (528 calls)
  dequant_rvv   :  0.944s ( 2.4%)  [CPU dequant INT4->INT8]
```

- **decode 链 2130→198→32 与 FP32 基线 / INT8 KV 门禁前缀一致**。
- decode 段 dequant 0.944s（2.4%），与实施 commit 记录的 0.963s（2.4%）一致，落于设计目标（dequant 占比 → ~30% 以下）之内。

---

## 3. Run 2 — CHAT 75-token（基线同配置）

配置：`CHAT=1 DECODE_STEPS=2 PROFILE=1 RSH=1 VERIFY=0`（与既有 75-token 基线 a2_base_full75 同 VERIFY 档位），输入 `/data/qwen/input_tokens.bin`（75 tokens）。

```
==== CHAT: n_tokens=75 + 2 decode steps =====
  CHAT prefill done: kv_len=75 next=97639 (219.59s)
DECODE pos=75 tok_in=97639 next=96555   (10.85s, majflt=1647)
DECODE pos=76 tok_in=96555 next=100798  (11.11s, majflt=1613)
CHAT: 97639 96555 100798
==== CHAT decode avg per-token = 10.98s over 2 steps =====
==== CHAT TIU runs: pass1=243456 pass2=243456 total=486912 ====
==== CHAT bit-exact: bad1=0 bad2=0 r_opt=0 rsh=0 ====
```

- **CHAT 链 97639→96555→100798 与基线 a2_base_full75 逐 token 一致**（bit-exact 构造保持的直接证据）。
- 端到端对比（基线 = a2_base_full75，同 VERIFY=0 / RSH 档位）：

| 指标 | 基线 | B2 | 变化 |
|---|---|---|---|
| CHAT prefill 75-token | 304.52s | 219.59s | **↓28%** |
| CHAT decode avg/step | 16.25s | 10.98s | **↓32%** |

> 注：该 VERIFY=0/RSH=1 配置下 PROFILE 的 dequant 桶显示 57.8%（11.34s），为**计时归属伪影**——RSH=1 跳过 wmax 预扫后 "other-rest" 桶（基线 RSH=0 时 67.2%）折叠，dequant 计时吸收层文件 page-fault/SD 读延迟；端到端时间（上表）为净收益且与标准配置（Run 1）一致。dequant 内核真实吞吐见 §4 微基准。

---

## 4. 单调用 dequant 吞吐（`dequant_kal_bench_b2` 板载实测）

```
correctness RVV vs scalar: N=896 bad=0  N=4864 bad=0  ALL EXACT
-- N=896  --  rvv_old (gather): 307.0 MB/s   rvv_b2 (vle8): 538.2 MB/s   (1.75×)
-- N=4864 --  rvv_old (gather): 373.2 MB/s   rvv_b2 (vle8): 753.1 MB/s   (2.02×)
```

- 单调用成本下降 **43%（1/1.75×）~ 50%（1/2.02×）**，落入预期 ↓40-50%。
- 全块 + 窗口（col_off/ncols）shape 下 RVV vs scalar 逐元素全 EXACT。

---

## 5. 验收对照（CEO 门禁）

| 项 | 要求 | 结果 |
|---|---|---|
| 3-prompt NEXT | 3/3 | ✅ 2130 / 12095 / 99366 |
| decode 链 | 2130→198→32 | ✅ 逐 token 一致 |
| CHAT 75-token | bit-exact | ✅ 链 97639→96555→100798 与基线一致 |
| TIU bit-exact | 全 0 | ✅ bad1=0 bad2=0 r_opt=0 rsh=0（回归/decode/CHAT 三段） |
| dequant 单调用 | ↓40-50% | ✅ 实测 1.75×~2.02×（↓43-50%） |
| ION / watchdog | 无残留 / 无触发 | ✅ used=0，零 WATCHDOG |

---

## 6. 附注：工作区并发改动观测（需 CEO 知悉）

验证期间发现 `qwen_engine_lmhead2.c` 工作区被**并发加入 A1 相关改动**（`ION_A1_EST`、`g_use_a1`、`mpool_build` 签名扩展等，+220/-78 行），文件 mtime 03:50。该文件按 CEO 指令在 B2 期间不应被推理引擎触碰。本验证二进制系 B2 干净 HEAD 构建，**未受此改动影响**；本报告 commit 亦**不包含**该 A1 改动（A1 归属推理引擎，建议 CEO 协调其提交时机与范围）。

---

## 7. 物证

- `b2_verify_reg.log`（Run 1 全量输出，37587 B）
- `b2_verify_chat.log`（Run 2 全量输出，36851 B）
- `dequant_kal_bench_b2` 板载输出（见 §4）

## 8. 结论

B2 达到设计目标且可复现：bit-exact 全保持、单调用 dequant 降本 ~50%、端到端 CHAT 提速 28-32%。建议 CEO 放行序列至 A1。
