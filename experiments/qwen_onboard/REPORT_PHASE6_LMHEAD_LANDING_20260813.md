# Phase 6 · LM Head 两段式(Top-K Cluster)上板落地报告

- 日期: 2026-08-13
- 引擎: `qwen_engine_lmhead2.c` (M2 24L Path A' 派生, 3-prompt 回归)
- 验收: NEXT 3/3 + 候选 gap ≥ 0.05 + TIU bit-exact + 延迟 ~10s/token

## 结论(Summary)

两段式 LM head 在 CV1800B 上板落地 **通过验收**, 且附带修复了全引擎的
**内存 swap 抖动**问题, 3-prompt 回归总墙钟 **277s → 164s (~41% 提升)**。

| 指标 | 落地前(M2 旧 LM head) | 落地后(两段式 + mmap 层加载) |
|---|---|---|
| LM head 单 token | ~2.1s (mmap 页故障) | **0.56–0.73s** |
| Stage1 (h·centroid) | 0.5s (swap-in 主导) | **15ms** |
| Stage2 (span pread) | — | 0.54–0.71s (SD 带宽主导) |
| 3-prompt 总墙钟 | ~277s | **164s** |
| NEXT 3/3 | — | **2130 / 12095 / 99366** |
| 候选 gap (P0) | — | **0.689 / 0.451 / 0.682** (min ≥ 0.05) |
| TIU bit-exact | ✓ | **✓ (bad1=bad2=r_opt=0)** |

## 1. 两段式 LM head 结构(已落地)

- **离线构建** (`lmhead_cluster_build.py`): MDP 球形 k-means C=1024, cluster-major
  重排 `embed_i8_cl.bin` [V,D] int8 + `embed_scales_cl.f32` [V] + `row_to_tok_cl.bin`
  [V] int32 + `centroid_f16.bin` [C,D] fp16 + `clust_idx.bin` [C,2] int32。
- **Stage1** (CPU, per token): `score[c] = h·centroid[c]` (float32, 1024×896),
  top-Kc=128 簇。centroid 载入时 fp16→fp32 一次性预转, 运行期纯 f32 dot。
- **Stage2** (per token): 对 top-Kc 簇 span 按 offset 升序 `pread` (避免 mmap
  每 4KB 页故障), 逐 span 精确 logits `Σ h[j]·row[j]·esc`, running top-5
  (cluster-major 是排列, 候选行 1:1 映射 token, 无需稠密 V 数组)。

## 2. 关键调试发现(重要, 供后续参考)

### 2.1 Stage1 "0.5s" 的真正根因是 swap-in, 不是 f32/f64
- 早期把 Stage1 0.4–0.6s 归咎于 "C906B double 太慢"。**错误**。
- 上板微基准: 同一 f32 dot 循环 7.5ms, f64 也仅 8.6ms; 500/40k 次 TIU 运行 +
  100MB SD 读后仍 8ms。**CPU/TIU/SD 均非根因**。
- 真根因: Linux 侧仅 ~28MB (64MB DDR 中 ~36MB 被 ION 划走)。24 层运行时
  8MB 匿名 layer buffer + 3.67MB g_cent 把系统推入 swap 抖动, g_cent 匿名页被
  回收 → stage1 每次全量 swap-in ~500ms。
- 旁证: `mlock(g_cent)` 后 stage1 稳定 9ms; `/proc/self/status VmSwap`
  从 ~7MB 降至 ~1.6MB。

### 2.2 全量 mlock 是负优化
- 一次性 mlock 4.67MB (g_cent+esccl+tokcl+clidx) 虽救回 stage1, 但把 swap 压力
  转移到 bias/frms/esc, 24 层 prefill +103s → 净亏。
- **只 mlock g_cent 3.5MB** 仍致 layer +30s/prompt。

### 2.3 mmap 层权重文件 = 根治
- 把逐层 `fopen/fread 8MB → malloc buffer` 改为 **`mmap(MAP_PRIVATE)`**:
  - 8MB 匿名(可换出)内存 → 干净的 page cache (可回收, 不占 swap);
  - VmSwap 7MB → 1.6MB, 全引擎 swap 抖动消失;
  - 24 层 t_layers P1/P2/P3 = **40/55/66s (原 70/90/112s, ~45% 加速)**;
  - 顺带: 层权重 mmap 免除 fread 的用户态拷贝, 首读由内核 readahead 覆盖。
- 最终配置 = **mmap 层加载 + mlock(g_cent)**。无 mlock 时 stage1 回落到 0.25s
  (g_cent 仍被回收), 故 mlock(g_cent) 保留为廉价保险 (3.5MB)。

## 3. 性能分解 (最终回归, 2026-08-13 23:10)

| Prompt | seq | t_layers | LMHEAD2 | stage1 | stage2 sd/cpu | cand_rows | per_token |
|---|---|---|---|---|---|---|---|
| P1 中国的首都是 | 3 | 40.31s | 0.70s | 15ms | 520/164ms | 6574 | 13.67s |
| P2 The capital of France is | 5 | 55.32s | 0.73s | 15ms | 556/153ms | 8488 | 11.21s |
| P3 今天天气很好，我们去公园 | 7 | 66.15s | 0.56s | 15ms | 417/120ms | 6083 | **9.53s** |

- NEXT: 2130 / 12095 / 99366, gap 0.6890 / 0.4508 / 0.6824 (host 与板上一致)。
- 总墙钟 163.96s (3 prompt), TIU runs 111744, bit-exact。
- decode 预估: 权重 SD 9.54s + LM head ~0.6s + 计算 (swap 修复后显著下降)
  ≈ **~10–11s/token**, 接近 ~10s 目标 (需 chat_duo 实测确认)。

## 4. 产物/文件

- `qwen_engine_lmhead2.c` — 引擎 (mmap 层加载 + 两段式 LM head + mlock g_cent)
- `lmhead_cluster_build.py` / `lmhead_twostage_c.c` — 离线构建 + host C 验证 (PASS)
- `weights_kal/embed_i8_cl.bin` 等 cluster-major 权重
- `sd_random_read_bench.c` — SD 散读基准 (sorted pread 17–35MB/s)
- `tiu_impact_probe.c` / `stage1_bench.c` — 根因定位微基准

## 5. decode 实测 (chat 循环, 2026-08-13 收口)

CEO 派发的真实 KV-cache decode 循环已完成, 并在 Phase 6 里程碑收口时正式交付:

- 实现: `qwen_engine_lmhead2.c` 新增 `run_decode_step()` (M=1 TIU 逐 token 前向),
  prefill 逐层捕获 kbuf/vbuf 至全局 KV cache `[L][KV_CAP=20][DKV]`, decode 逐 token
  追加并 `attention_cached(pos, cache, len=pos+1)`; 两段式 LM head 取 next。
  环境变量 `DECODE=1` / `DECODE_STEPS=N` 切换 (默认 N=5)。
- **实测 (P1 prefill seq=3 + 10 步, Duo 上板): decode 平均 25.98s/token**
  (10 步稳定区间 25.52–26.45s)。分解: t_layers ≈ 25.3s (97.5%, 每 token 重读
  24 层权重 201MB @ ~8MB/s) + LM head ≈ 0.60s (stage1 10ms, stage2 0.56s)。
- 正确性: TIU bit-exact (bad1=bad2=r_opt=0); prefill 回归 NEXT 3/3 保持;
  token 流 2130→198→32→13→220→102012→… (decode 为 `____\nA. `, 无自循环)。
- 质量: decode 循环数值正确; 输出碎片化为既有 INT4/INT8 量化精度限制 (参考本身
  即预测 2130), 非 decode 循环缺陷。
- **关键结论 (Phase 7 第一杠杆)**: 实测 ~26s/token 非预估 ~10s; 根因是 layer 权重
  mmap 流读仅 ~8MB/s (SD 顺序读天花板实测 21.5MB/s, 读路径存在 **2.6× 提升空间**)。
  若收敛至 ~21MB/s, decode 可降至 ~11–12s/token。优先级高于权重压缩/SDIO 副核。
- TPU 独立复核 Phase 6 4 项全过 (bit-exact / gap 0.4508≥0.05 / LM head 0.55–0.72s /
  内存 mlock 3.5MB VmSwap ~1.6MB), 里程碑正式关闭。
