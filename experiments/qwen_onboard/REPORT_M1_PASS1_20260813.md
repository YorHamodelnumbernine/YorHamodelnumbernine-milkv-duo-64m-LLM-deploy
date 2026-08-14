# M1 里程碑 — 上板 pass1 回读对齐（真实 Qwen Path A 数据）

日期：2026-08-13 | 作者：推理引擎工程师 | 关联：`qwen_m1_chunk.c` / `qwen_m1_data.h` / `gen_m1_data.py`
状态：**5/5 真实块上板 ALL PASS（P1/P2 逐元素 bad=0），M1 达成**
护栏：全部 untracked（experiments/qwen_onboard/），不碰 master。

---

## 0. 结论摘要

**M1（pass1 回读对齐）达成**：在 CV1800B 上，用**真实 Qwen2.5-0.5B layer-0 INT4 K-aligned
权重块 + 真实激活**跑通完整 Path A 数据链：

```
weights_kal INT4 K-aligned  →  CPU per-group dequant(原始 int4 -8..7)
  →  per_row 激活 quant（Step ①）
  →  TIU 两遍法 KG=32 N=896：
        pass1 rshift=rsafe → l2g 回读 → CPU max → r_opt
        pass2 rshift=r_opt  → l2g 回读
  →  fp32 累加 p2×2^r×gsc×sc_row
```

**5/5 块 pass1/pass2 逐元素精确（bad=0）**，fp32 累加 vs fp64 金标 maxdiff ~1e-9..2e-8。

## 1. 上板实测原始数据（Duo, rc=0）

| 用例 | 块 | wmax | rsafe | P1 ok | P1 bad | maxabs | r_opt | P2 ok | P2 bad | sat8 | fp32maxdiff | outmax |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| q0 | q_proj g=0 | 7 | 5 | 896/896 | **0** | 7 | 1 | 896/896 | **0** | 0 | 7.0e-09 | 0.158 |
| q13 | q_proj g=13 | 7 | 5 | 896/896 | **0** | 9 | 2 | 896/896 | **0** | 0 | 2.4e-08 | 0.662 |
| q27 | q_proj g=27 | 7 | 5 | 896/896 | **0** | 7 | 1 | 896/896 | **0** | 0 | 7.5e-09 | 0.284 |
| dn0 | down_proj g=0 | 7 | 5 | 896/896 | **0** | 6 | 1 | 896/896 | **0** | 0 | 1.9e-09 | 0.037 |
| up0 | up_proj g=0 | 7 | 5 | 896/896 | **0** | 7 | 1 | 896/896 | **0** | 0 | 9.2e-10 | 0.028 |

- lmem=29600 B（right[32,896]=28KB + res 0.875KB + left 32B），FORCED 线性 shape。
- r_opt 实测区间 **[1,2]**（decode M=1，单块；与 prefill M>1 的 4–5 不同，符合数据自适应预期）。
- pass2 小 rshift（1–2）仍逐元素精确 → 两遍法精度鲁棒。

## 2. 覆盖的数据链（四步实施项首次合流）

| 实施项 | 位置 | 验证 |
|---|---|---|
| ③ converter K-aligned | 权重块直接读自 weights_kal/layer0_kal.bin | 解包尺寸/布局正确 |
| ④ CPU per-group dequant | `dequant_block()` nib→原始 int8 | 喂入 TIU 的 right 与 host 完全一致 |
| ① per_row 激活 | `per_row_quant()`（round-half-even） | 左操作数 x_i8 与 host 一致 |
| ② 两遍法 KG=32 | `pass1→max→r_opt→pass2`，同 LMEM 不重载 | P1/P2 全 bad=0 |

## 3. TIU 关键细节（复用 TPU 闸口裁决）

- **FORCED 线性 shape** `{n,c=1,w=full,col=full}`：`bmk1822_matrix_lmem_default_shape` 的
  c-split 布局无法被 `l2g_matrix_copy` 解交织——沿用 gate1_mrow_check 裁决。
- **pass1/pass2 同 LMEM 不重载**：cmdbuf2 仅重注册 lmem 布局，left/right 已在 LMEM，
  只改 `rshift_bits`（rsafe→r_opt），符合「per-call rshift_bits 变更」闸口。
- 回读路径：`tdma_l2g_matrix_copy` + `MemInvld`，每 pass 独立 cmdbuf。

## 4. 复现

```
cd experiments/qwen_onboard
python3 gen_m1_data.py          # 重新生成 qwen_m1_data.h（5 块）
make qwen_m1_chunk              # 或手动 riscv64 交叉编译（见 qwen_m1_chunk.c 头注释）
python3 ~/Documents/MilkV_duo_project/duo_run.py qwen_m1_chunk   # → rc=0 ALL PASS
```

## 5. 下一步（M2 方向）

1. **多块连跑**：q_proj 全 28 块 + 全 7 矩阵，验证 per-block 循环 + r_opt 注入稳定性。
2. **N-tile 批处理**：up/gate N=4864 → 6 N-tile/block 合并为一次 pass 的 cmdbuf。
3. **SD 流式 + ION 双缓冲**：per-matrix/layer SD 读替换嵌入式数据（DESIGN §9a 布局）。
4. **cmdbuf 预建**：16 rshift 变体模板化（闸口②硬约束）。

已同步 TPU 工程师独立复核（数据同源）。
