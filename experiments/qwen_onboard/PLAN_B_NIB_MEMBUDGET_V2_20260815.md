# B 方向（nib-only + gsc 跨区缓存）内存预算复核文档 — V2

日期：2026-08-15 | 作者：TPU 底层工程师（bmk1822 / CV1800B）
状态：**待 CEO 复核后放行实现**
依据：CEO 2026-08-15 指令「nib-only + gsc 缓存重构，先交具体内存预算」；上级验收线 ≤9.5s PASS / 9.5~10s SOFT-PASS / >10s FAIL
前置：`PLAN_APRIME_MEMBUDGET_20260815.md`（V1） / `REPORT_PHASE8_CLOSE_20260815.md` / `REPORT_ION_DOUBLEBUF_FEASIBILITY_20260814.md`
设备实测：本机 SSH 直读（MemTotal 28MB、ION carveout 28,102,656B、swap 64MB on）

---

## 0. 一句话结论

**CEO 的「ION 余 ~10.4MB + DDR 放 ~11.3MB → gsc 21.33MB 跨 ION+DDR 全缓存恰好闭合」不成立，闭合失败。**
- ION 侧实测余量 ~11.1–12.1 MiB，per-layer nib 双缓冲（2×7.46MB）形态下 **ION 最多缓存 12 层 gsc（10.67 MiB）**，margin 仅 0.84 MiB（须释放 prefill pools）。
- DDR 侧运行期（kernel ~10 + 引擎匿名 ~10.6 + rms 0.16）只剩 **~5–6 MiB 安全余量**，被 Phase 8 B-2 实测证实（7 层=6.5 MiB DDR gsc 即 thrash，accum 2.44s/step）。
- **ION 12 层 + DDR 5–6 层 = 17–18/24 层缓存**，**6–7 层 gsc 每 token 仍需从 SD 读**。gsc 全缓存不可达。
- 全量 gsc 缓存唯一内存可行形态是 per-matrix SD_BUF（B-2），**已被 Phase 8 实测关闭（11.29~11.42s，FAIL）**。
- 可行档位（C-1 12/24、C-2 17/24）预估 per-token **~10.2–11.0s**，落在 SOFT-PASS 边缘到 FAIL，**9.5s PASS 线结构性不可达**。

---

## 1. 硬件实测基线（2026-08-15 本机 SSH 直读）

| 项 | 值 | 来源 |
|---|---|---|
| MemTotal | **28 MB** | `free -m`（used 12 / free 5 / buff/cache 12 / available 14，空闲态） |
| ION carveout | **28,102,656 B（26.80 MiB）** | `/sys/kernel/debug/ion/cvi_carveout_heap_dump/summary` |
| ION 当前占用 | 0 B（干净） | 同上 |
| **ION 历史 peak** | **28,073,984 B（99.9%！余 28 KB）** | 同上（B-2 运行峰值） |
| Swap | 64 MB 已启用（/swap, 2M used） | `swapon --show` |
| SD 顺序读物理上限 | 20.5–21.5 MiB/s（9 种读法收敛） | layer_read_bench.log / conn check |

**关键风险注记**：ION 历史 peak 28,073,984 B 说明上一轮 B-2 已把 carveout 用到 99.9%。任何新布局必须在 peak 序列上留足 margin，不能只算稳态。

## 2. 层文件字节构成（parse_layer，8,393,728 B/layer）

| 区段 | B/layer | MiB/layer | ×24 层 |
|---|---|---|---|
| rms_attn + rms_ffn | 7,168 | 0.007 | 0.16 MiB |
| nib（7 矩阵 int4，K-aligned G32） | **7,454,720** | 7.11 | **170.62 MiB** |
| gsc（7 矩阵 fp16 per-group scale） | **931,840** | 0.89 | **21.33 MiB** |
| **合计** | 8,393,728 | 8.005 | 192.12 MiB |

- nib 矩阵分解：q 401,408 / k 57,344 / v 57,344 / wo 401,408 / up 2,179,072 / gate 2,179,072 / down 2,179,072。
- nib-only per-token SD = **179.17 MB（170.62 MiB）**，@21.5 MiB/s 地板 **8.33s**，@20.5 地板 8.74s。
- 7 个 nib 区段文件偏移（512 对齐已验证）：{3584, 455168, 519680, 584192, 1035776, 3487232, 5938688}，O_DIRECT 可直接 span 读。

## 3. ION 逐块预算（CEO 要求的 B 形态：per-layer nib 双缓冲 + gsc 缓存）

SD_BUF = nib-only 每槽 7,454,720 B，双缓冲 2 槽。

| 分配项 | B | MiB | 说明 |
|---|---|---|---|
| nib SD_BUF_A / B | 14,909,440 | 14.22 | 2 × 7,454,720（nib-only，gsc/rms 移除） |
| neuron（NEURON_SZ: ACTQ/DQ/P1/P2） | 262,144 | 0.25 | TPU 工作区，固定 |
| pools decode 态（ps1/ps3/mp，ps5/ps7 已释放） | ~865,075 | ~0.83 | 实测全 pools=1,703,936，decode 前释放 M=5/7 ~0.8 MiB |
| **基础合计** | **16,036,659** | **15.29** | |
| gsc ION 缓存（**12 层**） | 11,182,080 | 10.67 | 12 × 931,840 |
| **C-1 稳态合计** | **27,218,739** | **25.96** | |
| **余量** | **883,917** | **0.84** | 3.1% |

**关键结论 ION**：
- **最多 12 层 gsc（10.67 MiB）**。第 13 层需 931,840 B > 余量 883,917 B，差 47,923 B，放不下。
- 若**不释放 ps5/ps7**（pools=1,703,936）：合计 28,057,600，余量仅 45,056 B（0.04%）→ **必释放 prefill pools**，且须在分配序列上控制 peak（gsc→SD_BUF→pools 顺序 + decode 前 free ps5/ps7）。
- 若再压缩 pools（只保留 rsafe 实际用到的 rshift 档，~0.5 MiB），ION 最多 **13 层**（margin 0.28 MiB），仍不够全缓存。

## 4. DDR 预算（decode 运行期，MemTotal 28 MB）

| 占用项 | MiB | 可回收性 |
|---|---|---|
| kernel + 系统 | ~10.0 | — |
| 引擎匿名（不可回收） | ~10.6 | — |
| ├ g_cent（fp32，mlock 常驻） | 3.50 | 不可回收（mlock） |
| ├ esccl / tokcl / esc（LM head） | ~1.75 | 匿名 |
| ├ bounce（O_DIRECT 2 MiB） | 2.00 | 匿名 |
| ├ KV cache + statics | ~1.70 | 匿名 |
| └ libc / malloc 开销 | ~1.5 | — |
| rms 全层缓存（g_rms_all） | 0.16 | 匿名（小） |
| embed_cl mmap 驻留 | ~1.8 | **文件背衬可回收** |
| **运行期合计（含 page cache）** | **~26.4–27.7** | |
| **可额外容纳（DDR 安全余量）** | **~5–6 MiB** | |

**实证锚点**：Phase 8 B-2 把 gsc 7 层（6.5 MiB）放 DDR mmap → **accum 2.44s/step 页错误**，即 6.5 MiB 已触发换页/回收抖动。anon malloc 版 gsc 常驻**实测 crash（RC=139）**。

**结论 DDR**：安全放 ~5–6 层 gsc（4.7–5.6 MiB），**不是 11.3 MiB**。即便把引擎匿名极限压缩（g_cent→fp16 省 1.75、LM head 三缓冲→mmap 省 ~1.75、bounce 2→1 省 1.0，合计 ~4.5 MiB，全部有 bit-exact/回归风险），DDR 余量也只到 ~10.6 MiB，仍让 gsc 11.3 MiB 顶在 28 MB 上限 → 必然 thrash/swap。

## 5. 闭合性检查（回答 CEO 核心问题）

**gsc 21.33 MiB 跨 ION+DDR 全缓存 → 不闭合。**

| 项 | CEO 前提 | 实测/推导 |
|---|---|---|
| ION 余量 | ~10.4 MiB | **11.13 MiB（pools 未释放）~ 12.07 MiB（释放后）**，可放 12 层（10.67 MiB） |
| DDR 余量 | ~11.3 MiB | **~5–6 MiB**（B-2 6.5 MiB 即 thrash 的实测硬约束） |
| gsc 需 DDR 补足 | 21.33 − 10.4 ≈ 10.9 MiB | 21.33 − 10.67 = **10.67 MiB ≫ 5–6 MiB** |
| **闭合判定** | 恰好闭合 | **FAIL（缺 ~5 MiB）** |

- 缺口 ~5 MiB 来自 **DDR 侧**：引擎匿名 ~10.6 MiB 是硬占用，B-2 实测 6.5 MiB DDR gsc 即抖动。
- 即便 ION 压到 13 层（压缩 pools），DDR 仍需 11 层 ≈ 9.78 MiB，仍超安全 5–6 MiB。
- 因此 **per-layer nib 双缓冲形态下，gsc 全缓存不可达**；0 gsc SD 读的理论最优 ~8–9s 无法成立。

## 6. 可行档位（降级）与 per-token 预估

> 预估参数：SD 地板用 20.5（保守）/21.5（乐观）MiB/s；双缓冲重叠损耗取 A' 实测 ~1.0s/step（2 槽浅流水 + 层0 冷同步）；LM head 0.5–1.0s（A' 实测 0.59–0.60s，峰值 1.04s）。纯 TIU+CPU 计算地板 ~4.0s < SD 地板，SD-bound 成立。

### 档 C-1：gsc 12/24 全 ION（CEO 定义的 C）

- ION：27.22 MiB（表 §3），margin 0.84 MiB。**须释放 ps5/ps7。**
- 每 token SD = nib 170.62 + 12 层 gsc 10.67 = **181.3 MiB**
- SD 地板：8.43s（@21.5）/ 8.84s（@20.5）
- t_layers ≈ 地板 + ~1.0s 重叠损耗 ≈ **9.4–9.8s**
- **per-token ≈ 10.0–10.9s（含 head）→ FAIL/SOFT-PASS 边缘**
- 风险：12 层未缓存 gsc 若走冷 mmap 小区域读（~8MB/s）→ accum 增加 ~1.3s/step（回到 GSC_ION=0 的老问题）；必须并入预读线程顺序 span 读（+0.5s 地板）才可控。

### 档 C-2：gsc 12 ION + 5 DDR = 17/24（DDR 安全上限）

- ION：27.22 MiB（同 C-1）；DDR：+5 层 = 4.66 MiB（DDR 运行期 ~25.4 MiB < 28，留 2.6 MiB 呼吸，接近但不超 B-2 抖动线）。
- 每 token SD = nib 170.62 + 7 层 gsc 6.23 = **176.85 MiB**
- SD 地板：8.22s（@21.5）/ 8.63s（@20.5）
- t_layers ≈ **9.2–9.6s**
- **per-token ≈ 9.8–10.6s（含 head）→ SOFT-PASS 边缘，全场最佳 PASS 机会，但不保证 ≤9.5s**
- 风险：DDR 5 层接近 B-2 抖动线（6.5 MiB thrash），需 drop_caches 冷启动实测；若 accum 回升则该档回退 C-1。

### 档 C-3：gsc 12 ION + 6 DDR = 18/24

- DDR +6 层 = 5.59 MiB（DDR ~26.3 MiB，紧）。每 token SD = 176.0 MiB，地板 8.19–8.59s。
- **风险高于 C-2（逼近 6.5 MiB thrash 线），per-token 收益仅 ~0.03s → 不推荐**。

### 档 D（参考，已关闭）：per-matrix nib SD_BUF + gsc 22 ION + 2 DDR（B-2）

- 唯一内存可行的 gsc 全缓存形态，Phase 8 已实测 **11.29–11.42s（FAIL）**：ION peak 99.9%、DDR 7 层 thrash、结构气泡（sd_wait ~3.7s + noslot ~1.6s）。**不再立项。**

### 回退档 A：现状收尾（mmap + gsc 全 ION 基线）

- decode **11.29s/token**（锚点 decode_e1_v0.log），锁定出货。>10s，但 bit-exact 全绿、一键回滚。

### 汇总

| 档 | gsc 缓存 | 每 token SD | per-token 预估 | 判定 |
|---|---|---|---|---|
| B-full（CEO 提议） | 24/24 全缓存 | 170.62 MiB | 9.4–10.3s（理想 8.9） | **内存不闭合，不可实现** |
| C-1（12/24） | 12 ION | 181.3 MiB | 10.0–10.9s | FAIL / SOFT-PASS 边缘 |
| **C-2（17/24）** | 12 ION + 5 DDR | 176.85 MiB | **9.8–10.6s** | **最佳，SOFT-PASS 边缘** |
| C-3（18/24） | 12 ION + 6 DDR | 176.0 MiB | 9.8–10.6s | 风险高，不推荐 |
| D（per-matrix B-2） | 24/24 全缓存 | 170.62 MiB | 11.29–11.42s（实测） | 已关闭 FAIL |
| A（回退） | 24 ION（现状） | 170.62 MiB（mmap 实测有效 ~27.9） | **11.29s（锚点）** | 出货基线 |

## 7. OOM 缓解与红线（各档通用）

1. **swap-on 强制**：64MB /swap 已启用（实测）。大分配前 `swapon /swap` 幂等确保。
2. **保守 pool 配置**：decode 前释放 ps5/ps7（~0.8 MiB）；优先只 build rsafe 实际用到的 rshift 档（可再省 ~1 MiB，给 ION 多 1 层 gsc 余量）。pools 全量 1,703,936 B 为峰值上限。
3. **ION 分配序列**：gsc 缓存 → SD_BUF → pools 大块优先；分配顺序与释放时序（decode 前 free ps5/ps7）必须显式控制，避免复现 ION peak 99.9%。
4. **红线（不越）**：
   - **不碰副核 SDHCI**（G-ION-1 整板死锁实证，`CMD_MHA_SD_TAKE_OWNER` 协议不可用）。
   - **不 push 到 kernel lockup**：DDR gsc 用 mmap 文件背衬（可回收）而非 anon malloc（已实测 crash RC=139）；运行期盯 `VmSwap` 稳态与 `majflt/step`，超出即回退档位。
   - **一键回滚**：引擎默认即 mmap 路径（gsc 全 ION 11.29s 锚点），新形态由环境变量门控，`sh /data/qwen/run_clean.sh --clean <bin>` 后可秒级切回，零重建。
5. **实现前置验证**：C-2 放行前先复测 ION peak 分布（避免 99.9% 复现）+ DDR 5 层 gsc 冷启动 accum 实测（>0.3s/step 则降 C-1）。

## 8. 结论与建议

- **CEO 前提「ION 10.4 + DDR 11.3 → gsc 21.33 闭合」不成立**：ION 侧成立（12 层），DDR 侧只安全 ~5–6 MiB，缺口 ~5 MiB。**B-full 形态不可实现，0 gsc SD 读的理论 ~8–9s 不可达。**
- **建议档 C-2（12 ION + 5 DDR = 17/24 缓存）**：per-token 预估 **10.2–10.6s**，是内存安全内最低 SD 流量、最佳 PASS 机会，但仍 **SOFT-PASS 边缘**（>9.5s 概率大，<10s 有希望）。
- **诚实结论**：9.5s PASS 线位于「SD 20.5–21.5 MiB/s 物理上限 + ION 26.8 MiB + DDR 28 MB」三重约束的结构性极限之外。**建议 CEO 同步准备 SOFT-PASS（9.5–10s）验收口径，或接受 11.29s 出货基线收尾。**
- 放行条件：CEO 对「走 C-2，接受 SOFT-PASS 概率」批复后，TPU 侧按 §7 红线实施（nib-only 预读线程 + gsc 跨区缓存 + 释放 pools + 回归）。

---
*关联归档：本 V2 预算 + V1（PLAN_APRIME_MEMBUDGET_20260815.md）+ Phase 8 收口（REPORT_PHASE8_CLOSE_20260815.md）*
