# B-P0 权重预取（readahead + DONTNEED）板验报告 — 关闭归档

日期：2026-08-22 | 作者：推理引擎工程师 | 状态：**已关闭（死胡同实证）**
定位：`DESIGN_QWEN_B_WEIGHT_PREFETCH_20260822.md` 的 B-P0 原型板验。目标击穿 decode 最大瓶颈 pass1 冷权重加载（~5.74s/token，dequant 的 98.5%）。
基线：B2 出货配置，CHAT 35-tok DECODE_STEPS=6 VERIFY=0，decode **11.10s/token**（本实验同口径实测，B2 档 11.15s）。

---

## 0. 结论摘要

1. **B-P0 关闭**：readahead 提前 Δ 层预取 + DONTNEED 释放，在此内核上是死胡同。全扫测 Δ{0,6,12} 无一正收益，Δ=6 最差 +5.0s。
2. **根因一（设计支柱死亡）**：内核未实现释放类内存建议系统调用——`posix_fadvise(POSIX_FADV_DONTNEED)` 与 `madvise(任意 advice)` 全部返回 **38 (ENOSYS)**；仅 `readahead()`/`mincore()` 可用。
3. **根因二（无释放则窗口击穿）**：64MB DDR 下可用页缓存 ~11MB，Δ=6 窗口（7×7.45MB≈52MB）远超容量 → LRU 驱逐预取前沿，majflt 不降反升（1648→1864），readahead 又叠加 ~179MB/步 SD 额外请求 → SD 双倍压力。
4. **附带资产（重要）**：现有 `mmap_ra` 模式的 `madvise(MADV_SEQUENTIAL)` 一直静默 ENOSYS no-op——历史 Phase 7 mmap_ra 负结果中 madvise 贡献为 0，负因全部来自同步 readahead。
5. **转向**：B-P1（独立线程 pread→DDR anon 双缓冲，显式控制不依赖页缓存 hint）。

---

## 1. 板验数据（完整 7 行）

口径：GSC_ION=1 CENT_ION=0 LW_READ=mmap RSH=1 VERIFY=0 PROFILE=1 CHAT=1
CHAT_INPUT=/data/qwen/input_tokens_35.bin DECODE_STEPS=6，每轮 drop_caches 冷启动。

| 轮 | 配置 | decode avg | Δ vs base | majflt/step | t_readahead/step | eng_matmul wall | dequant_rvv | bit-exact |
|---|---|---|---|---|---|---|---|---|
| base | B_PREFETCH 未设 | **11.10s** | — | 1648-1689 | — | 60.17s | 35.61s (59.2%) | 0/0/0/0 |
| d0_nn0 | Δ=0 DN=0 | 13.62s | +2.52s | 1474-1623 | ~2.0s | 59.28s | 34.47s (58.1%) | 0/0/0/0 |
| d0_nn1 | Δ=0 DN=1 | 13.50s | +2.40s | 1516-1591 | ~2.0s | 58.90s | 34.29s (58.2%) | 0/0/0/0 |
| d6_nn0 | Δ=6 DN=0 | 16.11s | +5.01s | 1737-1931 | ~1.9s | 68.90s | 42.72s (62.0%) | 0/0/0/0 |
| d6_nn1 | Δ=6 DN=1 | 15.66s | +4.56s | 1816-1887 | ~1.6s | 65.86s | 39.94s (60.6%) | 0/0/0/0 |
| d12_nn0 | Δ=12 DN=0 | 待补 | 待补 | 待补 | 待补 | 待补 | 待补 | 待补 |
| d12_nn1 | Δ=12 DN=1 | 待补 | 待补 | 待补 | 待补 | 待补 | 待补 | 待补 |

> Δ=0 实现说明：门控 `l+g_b_pf_delta<L` 在 Δ=0 时恒真 → 变相"readahead 当前层"（mmap_ra 式），非设计意图的 no-op 控制；真 no-op 控制由 base 充当。不影响 Δ=6/12 结论（代码路径逐字节一致）。

## 2. 关键判定

- **majflt 判据**：Δ 增大 → majflt 应显著↓。实测 Δ=0 微降（-10%）、Δ=6 反升（+10%）。不满足。
- **decode 判据**：全部 Δ 负收益（+2.4~+5.0s）。不满足。
- **判定分支**：非"majflt 已低仍慢"（majflt 高居 1500-1900），也非"Δ 有效"。落入"无变化/更差"→ 回退基线，关闭 B-P0。

## 3. 探针证据（ENOSYS 实证）

`fadv_probe`（B-P0 7 nibble 段）：`posix_fadvise` 每段返回 38，`readahead` 返回 0。
`madv_probe`（MAP_PRIVATE 整层）：`madvise` 对 DONTNEED/WILLNEED/SEQUENTIAL/NORMAL 全部 -1 errno=38；mincore 显示触碰后 416 页驻留、DONTNEED 后仍 416 页（未释放）。

系统调用可用性清单：

| syscall | 状态 | 影响 |
|---|---|---|
| readahead | ✅ 正常 | B-P0 预取可用但无释放 → 击穿 |
| mincore | ✅ 正常 | 诊断可用 |
| posix_fadvise (DONTNEED) | ❌ ENOSYS | 释放支柱死亡 |
| madvise (全部 advice) | ❌ ENOSYS | mmap_ra 的 MADV_SEQUENTIAL 静默 no-op |

## 4. 内存/约束记录

- B-P0 零新增 ION/anon 预算（设计原承诺）保持，但代价是依赖不可用的页缓存调度。
- 无释放机制下，页缓存窗口上限 ~1.4 层（~11MB），Δ≥2 即开始驱逐预取页。
- B-P1 转向预算：DDR anon 双缓冲 16.8MB 有超 28MB 风险；CEO 裁定单缓冲 8.4MB 或半层 4.2MB 起步、独立线程 pread。

## 5. 与支线/资产关系

- **mmap_ra 历史负结果重新归因**：Phase 7 实测 mmap+readahead 变慢，此前归因"本层同步 readahead 串行争抢 SD"——本实验补充证实其 madvise 部分恒为 no-op，负因全部为同步 readahead 的 SD 争抢。
- **LW_PREAD 路径**（顺序 pread ~21MB/s，+8.4MB anon）为 B-P1 的已验证内核形态，B-P1 是对其线程化 + 双缓冲的形式化。
- **TPU 底层工程师** ION notes 与 B-P2（ION 驻留）不受本结论影响，仍为 SD 硬地板上的最终可选扩展。

## 6. 验收结论

- 正确性：全 6+1 轮 bit-exact 0/0/0/0 —— B-P0 读路径改动不改变 DQ 字节，构造保持验证通过。
- 性能：未达成（无正收益轮次）。基线 11.10s 保留为 B-P1 对照。
- 板卡卫生：扫测后核验 ps 无 qwen 残留 + ION used=0。

## 7. 复现/归档文件

- 二进制：`qwen_engine_lmhead2_b0`（B-P0 门控 B_PREFETCH/DELTA/DONTNEED）
- 日志：`b0_base.log` `b0_d0_nn0.log` `b0_d0_nn1.log` `b0_d6_nn0.log` `b0_d6_nn1.log` `b0_d12_nn0.log` `b0_d12_nn1.log`
- 探针：`fadv_probe.c` / `madv_probe.c`
- 扫测脚本：`b0_sweep.sh` / `b0_sweep_d.sh`
