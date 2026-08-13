# A'（Weight-side INT4 → per-ch INT8）最终路径裁定 — CEO

日期：2026-08-13 | 裁决人：DUO project CEO | 状态：**最终裁定，A' 为 Qwen 生产路径**
关联：`REPORT_WS_DQ_VETO_20260813.md`（推理引擎工程师，A' 一票否决测试 3/3）、
`GATE_A_SIGNOFF_20260813.md`（TPU 工程师，Gate ①/④b 已签核）、
`REPORT_PS32_20260813.md` §6/§7（两遍法裁定）

---

## 0. 结论（TL;DR）

**Qwen2.5-0.5B 生产路径定为 A'：INT4 K-aligned G32 存储（201.4MB）+ CPU weight-side
dequant → per-channel INT8 + per-chunk 两遍法 INT8 matmul + CPU fp32 累加。**

- A' host 质量 **3/3**（engine-realistic gap 1.546/1.264/0.877，优于 host INT4 参考
  1.25/0.75/0.75）。matmul 核复用 Path B（已 3/3），权重文件 201MB/token（-44% vs B 358MB）。
- **P0 FAIL 根因落定**：布局轴（K-aligned vs FLAT）决定成败，非 scale 启发式。
  FLAT 4/4 全 0/3（logit 塌缩 gap 0.03–0.42）；K-aligned 4/4 全 3/3。
  修法已在布局轴，转换器（convert_qwen_kal.py）已交付，A' 无需新转换工作。
- **闸口状态纠正**：A' 报告中「等 TPU 闸口确认（pass1 回读、N-tile=512）」**已过时**——
  Gate ①（pass1 int8 回读）与 Gate ④b（N-tile=512）**已在 commit 5d00f05 由 TPU
  上板签核通过**（`GATE_A_SIGNOFF_20260813.md`）。引擎接入无阻塞。
- **两遍法保留**：A' 的 matmul 输出仍是 int8 rshift 量化，**两遍法为必需**（非可选），
  与 B 的 matmul 核同源。

## 1. 裁定依据（A' vs A vs B）

| 维度 | **A'（weight-side INT4）→ 选定** | A（per-group INT4 两遍法） | B（per-ch INT8 直存） |
|---|---|---|---|
| host 质量 | 3/3（gap 1.546/1.264/0.877） | 3/3（gap 1.923/1.105/0.485） | 3/3 |
| SD/token | **201MB（-44%）** | 201MB | 358MB |
| matmul 核 | 复用 B（per-ch INT8 两遍法） | per-group KG=32 专用 | per-ch INT8 两遍法 |
| ION | ~20.6MB FIT（余 3.4MB） | ~7.2MB | ~12.5MB |
| 工作量 | ~2–2.5 人日 | 3–4 人日 | ~1.5–2 人日 |
| 主要风险 | CPU 解包需被 SD 掩盖（估可） | 反量化是 CPU 关键路径 | SD 大 1.78x |

**理由**：
1. SD 是 decode 主导瓶颈（201MB 权重 + 136MB LM head @29MB/s ≈ 11.6s/token）——
   A' 与 A 同享 201MB/token 带宽优势，B 多 78% 流量。
2. A' 比 A 简单：matmul 核与 B 完全共用（per-ch INT8 + 两遍法），仅权重加载路径
   多「解包 int4→per-ch int8」，新增工作量 ~1–1.5d，无 per-group 专用调度。
3. 余量健康：engine-realistic gap 1.546/1.264/0.877，与 Path A per-group 相当。
4. A 的 per-group 两遍法保留为文档化 fallback（若 A' 上板 razor-thin 余量不足时兜底）。

## 2. 引擎接入硬约束（不变量）

1. **K-aligned 布局强制**（convert_qwen_kal.py / weights_kal/）。FLAT 布局判负（0/3）。
   设备端只读 `layerN_kal.bin` + `layer_scales.bin`，CPU 解包时用 stored per-ch scale
   （=kal_nat，字节级一致 max diff 2.9e-6）。
2. **两遍法必需**：pass1 安全 rshift → int8 回读实测 per-chunk max → pass2 精化 rshift
   满幅输出 → CPU fp32 累加。rshift 语义 = **round-half-up**
   `out_s8 = sat8((acc_i32 + (1<<(rshift-1))) >> rshift)`（TPU 已上板签核 5d00f05 / 23f300e，
   含小 rshift=1..4、负半值）。
3. **rope 表布局 `[pos][32]`（stride=HD/2）**（qwen_kal_ref.c 已验证，防越界未初始化）。
4. **TIU tiling 边界**（Gate ④b 实测）：KG=128 时 N≤192；KG=32 时 N≤896（N=1024 alloc fail）。
   A' 取 KG=128 + N-tile=192（516 submit/layer ≈ 12,384/token，~25ms，被 SD 289ms 掩盖）；
   如需更稳健可降 KG=32（≈49.5k submit/token，仍隐藏）。
5. **INT8 KV 余量**：ION 余 ~3.4MB 建议预留 KV（配合 INT8 KV 约 500+ token）。

## 3. 裁定动作

- 推理引擎工程师：**按 A' 路线开始引擎接入**（pf_worker 解包入加载路径 + ION 重排 +
  per-ch INT8 两遍法 matmul 核 + per-row act）。将 qwen_kal_ref.c 两遍法对齐
  round-half-up 后重新 host 验证 3/3；产出 C906B 上板实现计划。
- TPU 底层工程师：闸口已全签核，转后备支援；若 A' 上板有 TIU/tiling 问题随时介入。
- 记录：本报告 + A' 证据（emu_wsdq.py / wsdq_ref.json / wsdq_run.log）入库。

## 4. 产物

- `REPORT_WS_DQ_VETO_20260813.md`、`emu_wsdq.py`、`wsdq_ref.json`、`wsdq_run.log`
  （A' 一票否决测试证据）
- `REPORT_A_PRIME_RULING_20260813.md`（本裁定）
