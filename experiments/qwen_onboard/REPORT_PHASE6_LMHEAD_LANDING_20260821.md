# Phase 6 · LM Head 两段式落地报告（正式立项落地）

日期：2026-08-21 | 作者：推理引擎工程师 | 状态：**落地完成，待 TPU 独立复核握手**
关联：CEO 2026-08-21 立项「LM head 两段式落地」/ `qwen_engine_lmhead2.c`（本次改动）/ `REPORT_PHASE6_LMHEAD_TOPK_20260821.md`（spike）
改动：`qwen_engine_lmhead2.c` +95/-27（centroid ION 常驻自适应放置 + Stage1 双路径 + 清理）

---

## 0. 结论摘要

1. **落地范围 1/3 完成（离线 + 内存），引擎集成沿用既有 CPU Stage1 + pread Stage2。**
   离线 cluster-major 产物齐全且 host 复现 11/11；centroid fp16 1.84MB 已实现 **ION FREE 常驻**（自适应放置，余量不足回退 DDR fp32+mlock），上板实测无 OOM、无泄漏。
2. **上板 3-prompt NEXT 3/3 + min gap≥0.05 红线通过（ION 路径）。** P2 gap 0.4508 / P3 gap 0.6824，bit-exact 0 项。
3. **关键事实修正：CEO「ION 余 5.16MB」与实际不符。** 出货配置 GSC_ION=1 运行期实测 free=3.77 MiB（非 5.16），放 centroid 后 free=1.85 MiB、ION 94%。**1.84MB 可容纳但余量偏薄**，故落地为自适应（CENT_ION=0 可强制回退 DDR）。
4. **延迟目标 10.0-10.4s/token 未达（实测 11.3-12.2s）。** 非 LM head 瓶颈——decode 由 t_layers（SD-bound，~10.5-11.8s）主导，LM head 仅 0.5-0.7s。ION 路径相对 DDR 的增量仅 stage1 +15ms（f16→f32）。
5. **Stage1 TPU matmul 未实现（建议延后）。** CPU dot 仅 24ms（ION 路径）/9ms（DDR 路径），TPU 化省 ~15-20ms（0.2%），风险高。FORCED 线性布局铁律已记录，作为未来 TPU 化前置约束。

---

## 1. 落地范围逐项核对

| 项 | CEO 范围 | 状态 | 证据 |
|---|---|---|---|
| 1 | 离线 embed_i8 cluster-major 重排 + 8KB 索引表 | **✅ 完成** | `embed_i8_cl.bin` 136MB / `centroid_f16.bin` 1.84MB / `clust_idx.bin` 8KB / `row_to_tok_cl.bin` 均在 `weights_kal/`；host `lmhead_twostage_c` 复现 11/11 |
| 2 | 引擎 Stage1 (h·centroid) + Stage2 (span 读) 集成 | **⚠️ 部分（Stage1 为 CPU dot）** | Stage2 pread span 既有；Stage1 为 CPU dot（ION 路径 24ms），TPU matmul 未做，理由见 §6 |
| 3 | centroid fp16 1.84MB 常驻 ION FREE | **✅ 完成（自适应）** | `lmhead2_cent_place()`：free_ion≥阈值 → ION fp16；否则 DDR fp32+mlock。上板实测 ION 路径无 OOM |

## 2. ION 布局实测（CEO「余 5.16MB」核对）

| 项 | 值 | 备注 |
|---|---|---|
| carveout 总容量 | 28,102,656 B (26.8 MiB) | debugfs 实测 |
| GSC_ION=1 运行期 used（未放 centroid） | 24,330,240 B (23.2 MiB) | mem 0.25 + gsc 21.33 + pools 1.70 |
| **运行期 free（未放 centroid）** | **3,772,416 B (3.6 MiB)** | **≠ CEO 假定的 5.16MB** |
| 放 centroid 后 used | 26,165,248 B (24.95 MiB, 94%) | +1,835,008 B |
| 放 centroid 后 free | 1,937,408 B (1.85 MiB) | 可容纳，余量薄 |
| ION 峰值（历史累计） | 26,226,688 B (25.0 MiB) | 启动时已存在，非本 run 贡献；未超 carveout |
| 进程退出后 used | 0 B | 无泄漏 |

**口径说明（应 TPU 独立复核建议补充）**：表中「运行期 free（未放 centroid）3,772,416 B (3.6 MiB)」为 **pool/mp 已建后**的稳态账面（used 含 pools ~1.70MB + mp ~0.26MB）；而 `lmhead2_cent_place()` 的 placement 决策发生在 pool_build **之前**，其比较的即时余量为实测 `free_ion=5,476,352 B (5.22 MiB)`（≥ 阈值 need 4.98MB → 走 ION）。两口径不冲突：3.77MB 已扣除 pools 账面，5.22MiB 是放置时点即时余量，阈值判断基于后者。

**结论**：CEO「余 5.16MB 应容纳」的方向正确（1.84MB 确实放得下），但**余量数字需修正为 3.77MB→1.85MB**。为此落地采用**自适应放置**：仅当 `free_ion ≥ CENT_ION_SZ + pools_est + mp_est`（4.98MB）时走 ION，否则回退 DDR fp32+mlock（Phase 6 既有行为，stage1 9ms）。`CENT_ION=0` 可强制回退做 A/B。

**硬约束（TPU 复核有条件项，须记录）**：放 centroid 后 ION 余量 1.85MiB = 6.9%，且历史峰值（26,226,688 B）仅比当前 used 高 61KB——任何新增 ION 消费者必须先回收 centroid（`CENT_ION=0`）或收缩 gsc，不得直接追加。

## 3. recall@1 复现（host，离线）

`./lmhead_twostage_c weights_kal 128` → **recall_all=11/11，P0=3/3，P0_min_cand_gap=0.4386，PASS**
`./lmhead_twostage_c weights_kal 64` → **recall_all=10/11（量子计算 100152→2130 miss），P0=3/3，FAIL**（边界确认）

与 spike `lmhead_mdp_kcurve.json` 完全一致：C=1024 主工作点 Kc=128（1.6× 边界余量），Kc=64 不可用。

## 4. 上板回归（GSC_ION=1 出货配置）

### 4.1 3-prompt NEXT + 门禁 gap

| Prompt | 预期 next | ION 路径 | gap | DDR A/B 路径 | 判定 |
|---|---|---|---|---|---|
| P1 中国的首都是 | 2130 | 2130 ✅ | 0.6890 | 2130 ✅ | ✓ |
| P2 The capital of France is | 12095 | 12095 ✅ | 0.4508 | 12095 ✅ | ✓ |
| P3 今天天气很好，我们去公园 | 99366 | 99366 ✅ | 0.6824 | 99366 ✅ | ✓ |

- **NEXT 3/3，min gap 0.4508 ≥ 0.05 红线**；TIU bit-exact（bad1=bad2=rbad=0）。
- decode 流 2130→198→32 两路径一致。

### 4.2 延迟两套数据（2 步 decode，P1 prefill seq=3）

| 指标 | 基线 run1 (DDR+mlock) | ION run2 | DDR A/B run3 |
|---|---|---|---|
| decode avg /token | **11.30s** | **12.21s** | **13.21s** |
| t_layers step1/step2 | 10.91 / 10.55 | 11.77 / 11.31 | 12.74 / 12.47 |
| t_head step1/step2 | 0.61 / 0.53 | 0.60 / 0.73 | 0.61 / 0.60 |
| stage1 s1 | 0.009s | **0.024s** | 0.009s |
| stage2 s2 | 0.52-0.60 | 0.58-0.70 | 0.60 |

**SD 漂移归因**：三连跑 t_layers 单调递增 10.91→11.77→12.74s，与 centroid 改动无关（DDR A/B 比 ION 更慢）。真实 ION 增量 = **stage1 s1 +15ms/token**（f16→f32 逐元素转换），占 decode 0.15%，可忽略。

## 5. 内存效果

| 项 | DDR+mlock（原） | ION fp16（新） |
|---|---|---|
| DDR 匿名占用 | 3.67MB（fp32 g_cent）+ mlock | **0**（释放 3.67MB DDR） |
| ION carveout | 0 | 1.84MB（fp16，94%→余 1.85MB） |
| swap 抖动风险 | mlock 防换出 | 天然常驻（carveout） |
| stage1 | 9ms（预转 fp32） | 24ms（逐元素 f16→f32） |

## 6. Stage1 TPU matmul 评估（未做，建议延后）

- **现状**：Stage1 为 CPU dot（`score[c]=h·centroid[c]`），ION 路径 24ms / DDR 路径 9ms。
- **TPU 化收益**：~15-20ms/token ≈ decode 的 0.2%，且需新增 fp16 TIU matmul 路径（现引擎 matmul 为 INT8 two-pass），引入新代码面 + 破坏 bit-exact 链风险。
- **FORCED 线性布局铁律**：已记录为未来 Stage1 上 TPU 的前置约束（bm1822 硬约束）。当前 CPU 路径无布局约束，不受影响。
- **建议**：本里程碑保持 CPU dot；TPU 化仅当后续出现「Stage1 成为瓶颈」时才值得做。

## 7. 遗留与建议

1. **ION 余量薄（1.85MB）**：若后续 gsc 缓存需扩容，需先回收 centroid 或降级 gsc 层数。自适应回退（DDR+mlock）为保险。
2. **延迟目标 10.0-10.4s 的杠杆在 t_layers，不在 LM head**：现 decode 由层权重 SD 读主导（21.5MB/s 天花板 → 9.37s 地板 + ~1s 重叠损耗 + LM head 0.6s ≈ 11s）。LM head 两段式已把其贡献压到 0.5-0.7s，进一步需 B-2 per-matrix 流水或权重压缩（均已收口评估）。
3. **recall 泛化**：11 prompt 标定；真实对话 gold top-1 落出 top-Kc 簇无兜底。建议后续加 full-scan fallback（候选 gap 低时触发），Kc=256 保留为保守档。

## 8. 复现

```
# host recall@1
cd experiments/qwen_onboard
./lmhead_twostage_c weights_kal 128   # PASS 11/11
./lmhead_twostage_c weights_kal 64    # FAIL 10/11 (边界)

# 上板 (GSC_ION=1 出货配置, ION 路径默认)
# 构建: riscv64-unknown-linux-musl-gcc ... -o qwen_lh2_landing qwen_engine_lmhead2.c -lcviruntime -lcvikernel
LW_READ=mmap GSC_ION=1 VERIFY=0 RSH=1 DECODE=1 DECODE_STEPS=2 PROFILE=1 ./qwen_lh2_landing
# 回退 A/B: 追加 CENT_ION=0
```

板上日志：`/data/qwen/lh2_baseline.log`（run1 基线）/ `lh2_ion.log`（run2 ION）/ `lh2_ddr.log`（run3 DDR A/B）。

---

## 附：原始数据摘要（板上）

```
run2 ION 路径:
  [centroid] ION fp16 resident: 1.75 MB (free_ion=5476352 B); stage1 f16->f32 on-the-fly
  ION after: used=26165248 B (94%), free=1937408 B
  P1 NEXT=2130 gap=0.6890 | P2 NEXT=12095 gap=0.4508 | P3 NEXT=99366 gap=0.6824
  DECODE pos=3 total=12.38s (layers 11.77 + head 0.60, s1 0.024)
  DECODE pos=4 total=12.04s (layers 11.31 + head 0.73, s1 0.024)
  decode avg = 12.21s/token over 2 steps | bit-exact 0
run3 DDR A/B (CENT_ION=0):
  [centroid] DDR fp32+mlock: 3.50 MB resident
  DECODE pos=3 total=13.35s | pos=4 total=13.06s | decode avg = 13.21s/token
run1 基线 (DDR+mlock, 首次): decode avg = 11.30s/token
```
