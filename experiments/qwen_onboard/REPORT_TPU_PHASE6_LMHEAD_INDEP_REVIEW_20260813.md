# Phase 6 · 两段式 LM head 上板落地独立复核 — TPU 底层工程师

日期：2026-08-13 | 作者：TPU 底层工程师 | 状态：**复核通过（含 3 处表述修正建议）**
关联：`REPORT_PHASE6_LMHEAD_LANDING_20260813.md` / `qwen_engine_lmhead2.c` / `lmhead_twostage_c.c` / `verify_phase6_lmhead_indep.py`（本次新增）
护栏：本报告与复核脚本均留在 untracked `experiments/qwen_onboard/`，不碰 master。

---

## 0. 结论摘要（供 CEO 归档）

**Phase 6 两段式 LM head 上板落地独立复核通过。** 独立跑了一轮完整 3-prompt 回归
（板上 `qwen_engine_lmhead2` 全量重跑）+ host 侧独立 numpy 实现 + 数据链 md5 一致性核验。
四项验收点全部复现：

| # | 验收点 | 报告断言 | 独立复测 | 判定 |
|---|---|---|---|---|
| 1 | TIU bit-exact | bad1=bad2=r_opt=0，111744 runs | **bad1=0 bad2=0 r_opt=0，pass1=55872+pass2=55872=111744** | ✓ |
| 2 | 候选 gap min≥0.05 | 0.689/0.451/0.682，min 0.451 | **板上 0.6890/0.4508/0.6824（min 0.4508）；host 独立 numpy 1.1126/0.8055/0.4386（min 0.4386）** | ✓ |
| 3 | 性能 | LM head 0.56–0.73s；stage1 15ms；墙钟 277→164s | **0.584/0.721/0.554s；stage1 16ms；3-prompt 墙钟 163.94s** | ✓ |
| 4 | 内存 | VmSwap 7MB→1.6MB；mlock(g_cent) 3.5MB；稳定 | **VmSwap 1608/1664/1740kB；mlock'd 3.50MB；全程无 swap 抖动** | ✓ |

**结论：Phase 6 Milestone 收口背书成立。** 附带 3 处"表述级"修正建议（§4），
均不影响验收判定（top-1 3/3 + min gap 门禁在全部实现口径下强满足）。

## 1. 独立复核方法

- **独立实现**：新增 `verify_phase6_lmhead_indep.py`——不 import 引擎
  （`qwen_engine_lmhead2.c`）、不 import 任何 emulator/构建脚本（`lmhead_*.py`），
  仅从 raw 二进制独立重建两段式语义。
- **三路对照**：
  1. 板上全量引擎重跑（`qwen_engine_lmhead2`，VERIFY=1，3-prompt 回归）；
  2. host C 验证 `lmhead_twostage_c weights_kal 128`（既有，PASS）；
  3. host 独立 numpy（本次新增，f64 口径，与 host C 逐项核对一致）。
- **数据链**：host `weights_kal/` 与板上 `/data/qwen/` 逐文件 md5 一致性；
  cluster-major 排列合法性独立校验。

## 2. 验收点逐项证据

### (1) TIU bit-exact — 复现 ✓

板上全量回归输出（本次重跑）：

```
==== P1/P2 bit-exact: bad1=0 bad2=0  r_opt mismatches=0 ====
==== TIU runs: pass1=55872 pass2=55872 total=111744 ====
==== 24L regression: expected_next 3/3 OK ====
==== 24L regression: TIU internal BIT-EXACT ====
```

- 引擎 `VERIFY=1` 对每个 K-block 用 host int32 参考逐位核对 TIU P1/P2 输出与 r_opt
  （bad1/bad2/rbad），全部 111,744 runs 零失配。
- run 计数（55872+55872）与报告及 M2 独立复核的结构推导一致。

### (2) 候选 gap 验收线 — 可信 ✓

| Prompt | 板上引擎（f32 acc） | host C / numpy（f64 acc） | min |
|---|---|---|---|
| P1 | 0.6890 | 1.1126 | ≥0.05 |
| P2 | 0.4508 | 0.8055 | ≥0.05 |
| P3 | 0.6824 | 0.4386 | ≥0.05 |

- 三路实现 min gap 均 ≥ **0.43**，为 0.05 验收线的 **>8×**。
- top-1（NEXT 2130/12095/99366）跨 f32/f64、host/板上全部一致。
- gap 具体数值对 h 输入敏感（板上 TIU h vs host 模拟 h），但门禁判定不敏感。

### (3) 性能 — 复现 ✓（本次板上全量重跑）

| Prompt | LMHEAD2 | stage1 | stage2 sd/cpu | cand_rows | per_token | 报告对照 |
|---|---|---|---|---|---|---|
| P1 (seq3) | 0.584s | 16ms | 438/128ms | 6574 | 13.25s | 0.70s / 15ms / 520/164 / 6574 / 13.67s |
| P2 (seq5) | 0.721s | 16ms | 552/150ms | 8488 | 11.10s | 0.73s / 15ms / 556/153 / 8488 / 11.21s |
| P3 (seq7) | 0.554s | 16ms | 417/118ms | 6083 | 9.79s | 0.56s / 15ms / 417/120 / 6083 / 9.53s |

- LM head 单 token **0.554–0.721s**，落在报告 0.56–0.73s 区间（±SD 抖动）。
- Stage1 15–16ms，Stage2 由 SD 主导（sd >> cpu），cand_rows 与报告**逐值一致**。
- 3-prompt 总墙钟 **163.94s**（报告 163.96s）。t_layers 39.17/54.77/67.95s
  （报告 40.31/55.32/66.15s），差异为 SD/热态抖动，方向一致。
- 注：本次为冷页面缓存首跑（板子上次空闲），仍复现 164s，说明 mmap 层加载对
  冷/热 cache 均稳健。

### (4) 内存方案 — 成立 ✓

- **mlock(g_cent) 3.50 MB**：引擎启动日志 `[lmhead2_load] mlock'd 3.50 MB (g_cent only)`
  与报告一致。
- **VmSwap ~1.6MB**：3 prompt 的 LM head 阶段实测 VmSwap = **1608/1664/1740 kB**。
- **稳定性**：全程无 swap 抖动迹象；t_layers 三 prompt 单调递增（39→55→68s，
  与 seq 3→5→7 一致），无 0.25–0.5s 级的 stage1 swap-in 尖峰（stage1 恒定 15-16ms）。
- 报告"7MB→1.6MB"的"7MB 前值"为历史基线（malloc 8MB 匿名 layer buffer 方案），
  本次无法在不改代码的情况下直接复测；但"后值 1.6MB"与机制（mmap→page cache，
  clean 可回收不占 swap）均已确认。

## 3. 数据链 / 排列合法性（host 独立 numpy）

| 校验 | 结果 |
|---|---|
| row_to_tok_cl.bin 是 0..V-1 排列 | ✓ PASS |
| embed_i8_cl[perm 反查] == embed_i8 | ✓ PASS（逐位） |
| embed_scales_cl[perm 反查] == embed_scales | ✓ PASS |
| clust_idx span sum==V 且连续覆盖 [0,V) | ✓ PASS |
| host↔板 md5：centroid/clust_idx/esc_cl/tok_cl/embed_i8/embed_i8_cl/embed_scales/final_rms/layer0_kal | ✓ 全部一致 |
| 两段式语义 recall_all 11/11，P0 3/3 | ✓ PASS |
| f32 vs f64 累加 top-1 稳定（P0 3 prompt） | ✓ PASS |

## 4. 需修正的表述项（不影响验收判定）

1. **"host 与板上 logits/gap 逐位一致"表述不严谨。** gap 具体数值对 h 输入敏感：
   板上 TIU h → 0.689/0.451/0.682；host 模拟 h（`lmhead_h_cache.npz`）→ 1.113/0.806/0.439，
   并非逐位一致。复现成立的是：**top-1 NEXT 3/3 一致 + 候选 gap 门禁（≥0.05）在全部
   口径下强满足（min≥0.43）**。建议报告措辞改为"top-1 与候选 gap 门禁 host/板上一致"。
2. **"sorted pread 17–35MB/s"未能复测到上限。** 独立跑 `sd_random_read_bench`（128 spans，
   冷 cache）：random 9.77 / sorted 10.67 / sequential 17.69 MB/s。引擎实际 Stage2
   有效吞吐 ≈ 13–14MB/s（cand×896B ÷ sd 时间），结论"SD 带宽主导"成立，但报告数值偏高
   （可能为热 cache 下测得）。
3. **§2.1 stage1 f32 dot "7.5ms"微基准未复现。** 独立跑 `stage1_bench`：f32 scalar
   16.78ms / f64 15.08ms / unroll4 10.23ms。引擎实际 stage1 = 15–16ms（与报告
   LMHEAD2 表一致）。根因归因（stage1 慢的根因是 swap-in 而非 CPU/TIU/SD）仍成立。

## 5. 复现

```
# 1) 板上全量回归（~164s，冷首跑）
cd ~/Documents/MilkV_duo_project
python3 duo_run.py tpu_bench/experiments/qwen_onboard/qwen_engine_lmhead2 --timeout 300

# 2) host C 验证
cd tpu_bench/experiments/qwen_onboard && ./lmhead_twostage_c weights_kal 128   # PASS

# 3) host 独立 numpy（本次新增）
python3 verify_phase6_lmhead_indep.py                                          # PASS

# 4) 微基准
python3 duo_run.py tpu_bench/experiments/qwen_onboard/stage1_bench --timeout 120
python3 duo_push.py tpu_bench/experiments/qwen_onboard/sd_random_read_bench /tmp/sd_random_read_bench
python3 duo_run.py --cmd "chmod +x /tmp/sd_random_read_bench && /tmp/sd_random_read_bench /data/qwen/embed_i8_cl.bin /data/qwen/clust_idx.bin 128" --timeout 120
```

> 备注：复核期间板上出现队友（推理引擎工程师）并发复跑同一引擎，本复核的主回归
> 在其启动前已干净完成，时序数据不受干扰。复核过程中曾因 `duo_run.py` 参数误用
> 覆盖板上 `embed_i8_cl.bin`，已立即用 `deploy_qwen_weights.py --only embed_i8_cl.bin`
> 恢复并 md5 验证一致（ac5e6855...），后续所有核验均在恢复后的正确数据链上进行。
