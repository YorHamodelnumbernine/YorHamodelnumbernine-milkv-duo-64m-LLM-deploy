# M1 独立复核结论 — TPU 底层工程师（seq 206）

日期：2026-08-13 | 作者：TPU 底层工程师 | 状态：**复核完成，ALL MATCH** | 关联：`verify_m1_indep.py` / `REPORT_M1_PASS1_20260813.md` / `REPORT_M2_TIU_CORE_20260813.md`

---

## 0. 结论摘要（供 CEO 归档）

**M1（pass1 回读对齐）独立复核通过**。独立于推理引擎的验证脚本 `verify_m1_indep.py`
从 `layer0_kal.bin` 原始二进制重新解包 5 个 M1 case，三项独立核对全部一致：

- **(A) 数据链**：`qwen_m1_data.h` 内嵌 nib/gsc/act 与 `weights_kal/layer0_kal.bin` 原始数据
  **逐字节一致**（nib 5×14336、gsc 5×896，activation maxdiff 2.4e-07 < 1e-6）。
- **(B) host 独立两遍法语义**：rsafe / maxabs / r_opt / p2 / sat8 五项指标与
  `REPORT_M1_PASS1` 上板实测表 **5/5 全 OK**。
- **(C) fp32 累加 maxdiff 复算**：与 `qwen_m1_chunk.c` 同顺序，量级 ~1e-9..2.5e-8，
  与报告一致。

## 1. 与推理引擎 M2 里程碑交叉引用

CEO 下达的三项硬约束已确认落进 C906B 微内核实现（IMPL_C906B_MATRIX / PLAN_DEPLOY）：

| # | 硬约束 | 落点核验 | 证据 |
|---|---|---|---|
| 1 | FORCED 线性布局 | `{n,c=1,w=full,col=full}`，c-split 布局无法被 `l2g_matrix_copy` 解交织 | M1 报告 §3；verify (A) |
| 2 | TIU round-half-up CPU 反量化匹配 | pass1/pass2 回读 vs host `int8_round_div` bit-exact（bad=0/25088） | M2 报告 §1；verify (B) |
| 3 | 区域 MemInvld | 每 pass 独立 cmdbuf + `tdma_l2g_matrix_copy` + `MemInvld`；M2 已实测 0.052ms/block 并列为区域化优化项 | M1 §3 / M2 §1,4 |

## 2. 复核判定

```
== (A) 数据链:   ALL BYTE-IDENTICAL ==
== (B)(C) host 两遍法 vs 上板实测: ALL MATCH ==
q0  (7,5,7,1,0) md=6.6e-09  outmax=0.158
q13 (7,5,9,2,0) md=2.5e-08  outmax=0.662
q27 (7,5,7,1,0) md=7.3e-09  outmax=0.284
dn0 (7,5,6,1,0) md=1.7e-09  outmax=0.037
up0 (7,5,7,1,0) md=9.2e-10  outmax=0.028
```

**结论：M1 里程碑独立复核通过，与推理引擎 M2 TIU 核心产出互相印证，无冲突项。**
闸口①/④b 签核闭环后，M2 方向（N-tile=896 形状池、区域化 MemInvld、全层集成）无新增阻塞。
