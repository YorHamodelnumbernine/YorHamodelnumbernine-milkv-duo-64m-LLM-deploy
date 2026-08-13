# bmk1822 per-group dequant（ps32 硬件路径）可行性裁定 — TPU 底层工程师

日期：2026-08-13 | 作者：TPU 底层工程师 | 状态：上板实测结论，交付 CEO 与推理引擎工程师

---

## 1. 裁定结论

**bmk1822 ps32 per-group dequant 硬件可行（int32 部分和可读取、可正确计算），
但不是最优实现路径：submit 数（~58k/token 层内、~80k 含 lm_head）比推理引擎工程师
两遍法（49.5k/token）还多，且需 7-12 人日全新重构（预算 2-3 人日的 3-4x），无质量增益。**

推荐：**主路径采用 Path A 两遍法（KG=128 ps32-free，推理引擎工程师已 3/3 验证）**。
ps32 硬件路径作为已文档化的 fallback。

## 2. 上板实测证据（probe5/6/18/19/20/21）

| # | 结论 | 证据 |
|---|---|---|
| 1 | **ps32 int32 部分和可读取** | ps32_mode=2 + res_is_int8=1 + lmem_alloc_ps32_matrix。byte-plane 布局：byte b of col n 在 `b*N+n`。probe5（acc=288→0x20/0x01）、probe6（[1,8] 8/8）、probe18（forced c=1, N=16/32/64/112 全对） |
| 2 | **fp32 输出不可读** | res_is_int8=0 / FMT_BF16 alloc → 全零（probe5）。必须 CPU 侧 fp32 累加 |
| 3 | **TIU 输出宽度硬件上限 N=192/次** | forced c=1 宽度扫描（probe20）：N=128/144/160/192 全对（192/192），N=224/256 失败（0）。g2l 加载在全部宽度完整（0 bad），限制在 TIU 侧。此前 default_shape 的 w 上限（16/32/64/112）只是 shape 映射产物，非硬件上限 |
| 4 | **tK=32 容量可行** | group-major：W[32,896]=28KB + ps32 res[1,896]=3.8KB + left[1,32]=32B ≈ 29.7KB < 32KB。4864 宽矩阵按 N=192 分块（[32,192]=6KB/块） |
| 5 | **批量 per-op 成本** | probe21（N=192 forced-c1）：build ~1.9μs/op + run ~1.8μs/op ≈ 3.7μs/op，正确性 192/192。cmdbuf 容量 ~2000 op/256KB（超出触发 `cur_nr_desc < max_nr_desc` 断言） |
| 6 | **QDM 无可用路径** | probe4：qdm 输出 byte0=0x00；SUMMARY.md：mul_qdm 全零、matmul_qdm shape 冲突 |

## 3. submit 数核算（Qwen2.5-0.5B，G=32，N-chunk=192）

G=32 强制每个 matmul 只含一个 K-group（组 scale 无法从 K 内拆出）→
submit = (K/32) × ceil(N/192)。

| 矩阵 | 形状 | 组数 | N-chunk | submit/层 |
|---|---|---|---|---|
| Wq | [896,896] | 28 | 5 | 140 |
| Wk | [896,128] | 28 | 1 | 28 |
| Wv | [896,128] | 28 | 1 | 28 |
| Wo | [128,896] | 4 | 5 | 20 |
| up | [896,4864] | 28 | 26 | 728 |
| gate | [896,4864] | 28 | 26 | 728 |
| down | [4864,896] | 152 | 5 | 760 |
| **小计/层** | | | | **2432** |
| ×24 层 | | | | **58368** |
| lm_head | [896,151936] | 28 | 792 | 22176 |
| **总计/token** | | | | **80544** |

## 4. 端到端性能估算（decode, M=1, batch 2000/cmdbuf）

- TIU：~58k × 1.8μs ≈ **105ms**（lm_head 另计 +40ms）
- cmdbuf build（CPU）：~58k × 1.9μs ≈ **110ms**
- 权重 g2l DMA：层内权重 ~15.4MB/层 ×24 ≈ 370MB/token ≈ **185ms**（部分可重叠）
- CPU per-group fp32 累加：~58k × 192 元素 ≈ 11M MACs ≈ **20-40ms**
- 总 ≈ **350-500ms/token → ~2-3 tok/s**

## 5. 工作量（对照 2-3 人日预算）

| 项 | 估算 |
|---|---|
| 新 bmk1822 forced-c1 K=G=32 ps32 matmul 调度 + int32 读回 + CPU fp32 per-group 累加（group-major tiling） | 5-8 人日 |
| per-row 激活 scale（M=1 单标量，公共项） | 1-2 人日 |
| converter K-aligned 布局（scale[K/G,N]；推理工程师已交付 weights_kal/） | 0.5-1 人日 |
| lm_head 151936 宽策略（top-k 两段式 / 热区缓存） | 1-2 人日 |
| **合计** | **7-12 人日（预算 3-4x，超支）** |

## 6. A/B/ps32 对比（合并推理引擎工程师数据）

| | 路径 A 两遍法（KG=128） | 路径 B per-ch INT8 两遍法 | ps32 per-group（本裁定） |
|---|---|---|---|
| 质量（host） | 3/3（gap 1.923/1.105/0.485） | 3/3 | 3/3（exact int32 部分和） |
| SD/token | ~201MB | ~358MB | ~201MB |
| ION 双缓冲 | FIT 16.8MB | >24MB 需拆分 | FIT |
| TIU submit/token | ~49,500 | ~更低 | ~58,368（层）+ ~22k（lm_head） |
| 工作量 | 3-4 人日 | 复用+拆分 | 7-12 人日 |
| 主要风险 | CPU INT4→INT8 反量化（需与 SD 重叠） | ION 拆分复杂 | submit 多 + 重构投入大 |

## 7. 裁定

ps32 硬件路径**可行但被两遍法支配**：submit 更多、工程更重、无质量增益。
主路径 = **Path A 两遍法**。ps32 路径保留为 fallback（若两遍法上板 quality 不达标
可回退：ps32 int32 部分和精确性可提供 razor-thin prompt 更大余量）。

## 8. 探针产物

`ps32_probe18.c`（forced c=1 全 N 列验证）、`ps32_probe19.c`（批量吞吐）、
`ps32_probe19b.c`（N≥256 诊断）、`ps32_probe20.c`（宽度上限扫描）、
`ps32_probe21.c`（批量 build/run 成本）。
