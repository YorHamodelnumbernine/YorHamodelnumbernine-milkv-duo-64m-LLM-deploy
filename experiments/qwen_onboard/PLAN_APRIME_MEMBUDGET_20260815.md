# A' nib-only 内存预算与可行性文档（CEO 复核用）

日期：2026-08-15
作者：TPU 底层工程师
状态：**待 CEO 复核后放行实现**
依据：CEO 2026-08-15 指令「nib-only + gsc 缓存，先交内存预算文档」
上级决策：整层读规格已关闭；≤9.5s PASS，9.5~10s SOFT-PASS，>10s FAIL

---

## 0. 一句话结论

**CEO 的「ION 余 ~10.4MB + DDR 余 ~11.3MB → gsc 21.33MB 跨 ION+DDR 恰好全缓存」前提不成立：DDR ~11.3MB 是空闲值，decode 运行期引擎自身匿名 ~11MB 会吃掉它。**
按精确预算，可行的完整 gsc 缓存只有「per-matrix SD_BUF」一型（ION 22 层 + DDR 2 层，DDR 仅 +1.8MB，安全）；per-layer SD_BUF 只能做到部分缓存（17/24 层），性能地板差 ~0.3s。两者都只能把 A' 推进到 **~9.3–10.5s（大概率 SOFT-PASS 区间）**，9.5s PASS 线位于物理极限边缘，不保证。

---

## 1. 实测基线（整层读 A'，smoke_iondb2.log，2026-08-15）

| 指标 | 值 | 备注 |
|---|---|---|
| decode avg | **11.72s/token**（10.73 / 12.71） | 未过 9.5s 验收线 |
| A' 指标 step1 | sd=9.167s, sd_wait=2.078s, err=0 | 23 层预取 |
| SD 有效速率 step1 | 21.08 MiB/s（193.2MB/9.167s） | 已达物理天花板 |
| compute/step（主线程） | ≈6.08s（TIU 3.47 + dequant 0.89 + accum 0.66 + 其他） | < SD，SD-bound 成立 |
| LM head/step | 0.5~1.4s（sd=0.43~1.04s） | 与层计算串行 |

**SD 天花板**：21.5 MiB/s（layer_read_bench.log 九种读法全 19.8~20.6；mmap=20.16，pread_odirect=20.57 最优，readahead 反而差）。

## 2. 层文件构成（parse_layer 顺序，8,393,728 B/layer）

| 区段 | 大小 B | 说明 |
|---|---|---|
| rms_attn / rms_ffn | 2 × 3,584 | 归一化权重 |
| nib（7 矩阵） | 7,458,304 | q 401,408 / k 57,344 / v 57,344 / o 401,408 / up 2,179,072 / gate 2,179,072 / down 2,179,072 |
| gsc（7 矩阵） | 931,840 | fp16 per (K-block=32, N)；24 层 = 22,364,160 B = **21.33 MiB** |

- 每 token nib-only 读 = 24 × 7,465,472 = **179,171,328 B（170.9 MiB）**
- 整层读 = 24 × 8,393,728 = **201,449,472 B（192.1 MiB）**
- nib-only 地板：179.17MB / 21.5 = **8.33s**（整层读地板 9.37s，省 1.04s/token）

## 3. ION 现状（carveout 28,102,656 B = 26.8 MiB，smoke 实测）

| 分配 | 大小 B | 备注 |
|---|---|---|
| mem（NEURON_SZ：ACTQ/DQ/P1/P2） | 262,144 | 256 KiB |
| SD_BUF_A/B（整层） | 2 × 8,396,800 = 16,793,600 | A' 现规格 |
| pools（ps1/3/5/7 + mp merged cmdbuf） | ≈1,703,936 | 实测推导 18,759,680−16,793,600−262,144 |
| **运行期使用合计** | **18,759,680（66.8%）** | |
| **空闲** | **9,342,976（8.9 MiB）** | |
| 峰值（init/运行期瞬时） | 25,300,992（90.0%） | 来源待查，作为 ION 余量风险 |

## 4. DDR 现状（28 MiB 系统，decode 运行期）

**引擎匿名 ≈ 11 MB**（代码精确清单）：
- 静态激活/累加缓冲 ≈ 1.23 MB（x/h/qbuf/attn 4×100KB、upb/gateb/mid 3×136KB、accd 272KB、cosb/sinb 18KB 等）
- heap：g_cent 3.67MB、g_esccl 0.61、g_tokcl 0.61、esc 0.61、KV 0.49、bounce 1.0、embed_cl 常驻 1.84
- libc/栈/malloc 开销 ≈ 1~2 MB

运行期：kernel ≈ 10 MB（空闲 `free` used=12MB 推导）+ 引擎 11 MB ≈ **21 MB** → 可额外容纳 ≈ **5~7 MB**（靠回收 page cache；若用 O_DIRECT 消除层读页缓存则更稳）。
**因此 DDR 只能安全放 ~5–6 层 gsc（4.7–5.6 MB），不是 11.3 MB。**

## 5. 候选设计内存预算

### 方案 B-1：per-layer nib-only SD_BUF + 部分 gsc 缓存（小改）

SD_BUF 压缩为 nib 连续区（rms+nib 顺序排布，去掉 gsc 孔洞）：`7,458,816 B/槽 × 2 = 14,917,632`。

| 项 | 大小 B |
|---|---|
| mem | 262,144 |
| SD_BUF_A/B（nib-only） | 14,917,632 |
| pools | 1,703,936 |
| gsc ION 缓存（11 层） | 10,250,240 |
| **ION 合计** | **27,133,952（96.6%，余 0.92 MiB）** |
| gsc DDR 缓存（6 层） | 5,591,040（DDR 总 26.3 MB < 28 ✓ 但紧） |
| gsc 未缓存 | 7 层 → 每 token 从 SD 读 6,522,880 B |
| **每 token SD** | 179.17MB + 6.52MB = **185.69MB（177.1 MiB）→ 地板 8.64s** |

改量：A' 预读线程 1 处（改读 9 段 span）+ parse_layer 偏移表。**保持 per-layer 双缓冲不变。**

### 方案 B-2：per-matrix nib-only SD_BUF + 全量 gsc 缓存（大改）

SD_BUF 按矩阵粒度双缓冲，槽位按最大矩阵 nib（up/gate/down = 2,179,072 B）定尺寸：`2,179,072 × 2 = 4,358,144`。

| 项 | 大小 B |
|---|---|
| mem | 262,144 |
| SD_BUF_A/B（per-matrix） | 4,358,144 |
| pools | 1,703,936 |
| gsc ION 缓存（22 层） | 20,500,480 |
| **ION 合计** | **26,824,704（95.5%，余 1.22 MiB）** |
| gsc DDR 缓存（2 层） | 1,863,680（DDR 总 22.8 MB < 28 ✓ 舒适） |
| gsc 未缓存 | 0（全量缓存） |
| **每 token SD** | **179.17MB（170.9 MiB）→ 地板 8.33s** |

改量：预读线程按矩阵 span 预取 + run_prompt/run_decode_step 每次 eng_matmul 前按矩阵等待槽位。**推翻 per-layer 双缓冲，重新做矩阵级流水。**

## 6. 性能预估（诚实区间，含实测重叠损耗）

| 方案 | SD 地板 | 预估 t_layers | 预估 total（+LM head 0.5~1.4s） | 判定 |
|---|---|---|---|---|
| 现状整层 A' | 9.37s | 10.1~12.1（实测） | **11.7s** | FAIL |
| B-1（部分缓存） | 8.64s | 9.1~10.1 | **9.6~11.5s** | SOFT-PASS 边缘 |
| B-2（全量缓存） | 8.33s | 8.8~9.8 | **9.3~11.2s** | 最佳机会，PASS 不保证 |

> 重叠损耗取整层 A' 实测的 t_layers−SD ≈ 0.97s（step1）。**9.5s PASS 线在物理极限边缘**：即便 B-2 完美实现，也可能落在 SOFT-PASS（9.5~10s）。CEO 需接受这一现实，或同步准备 SOFT-PASS 验收口径。

## 7. 附加优化（捆绑实现，压低重叠损耗）

1. **O_DIRECT 读 nib**：layer_read_bench 实测 20.57 MiB/s 最优，且**不污染 page cache**（给 DDR gsc 让出内存）。所有 nib/rms span 偏移与尺寸均 512 对齐（已验证）。
2. bounce 1 MiB → 2 MiB：减少 pread 次数。
3. 重叠策略：prefetch 线程在最后一层算完后（层读已结束）可预读下一 token 的 embed span，尝试把 LM head sd 与首层计算重叠（后续验证，非必须）。
4. 释放 prefill 专用 pools（ps5/ps7，M=5/7）在进入 decode 前：ION 再省 ~0.8 MiB，提高 B-1/B-2 余量。

## 8. 红线与风险

- **swap-on 强制**（standing instruction，不变）
- **不碰副核 SDHCI / 不改内核**；一键回滚 LW_READ=mmap GSC_ION=1（11.29s 锚点）不变
- ION 峰值 25.3MB 来源未定位——放行前先复测一次 ION 峰值分布（B-1/B-2 稳态余量 0.9~1.2 MiB，若峰值叠加会超限）
- DDR：B-1 运行期 26.3MB（偏紧），B-2 22.8MB（舒适）——**B-2 在内存安全性上反而更优**
- 正确性保持：nib/gsc 字节与现文件同源，仅文件内位置/指针重映射 → bit-exact 由构造保持（需 VERIFY=1 回归确认）

## 9. 建议

**首选 B-2**：唯一能把 gsc 全量缓存的设计，DDR 安全（+1.8MB），地板最低（8.33s），并附 §7 三项优化。代价是矩阵级流水重写（改动面大于 B-1）。
**备选 B-1**：改动小、风险低，但地板 +0.3s，DDR 偏紧，且只到 SOFT-PASS 边缘。

请 CEO 批复：(a) 走 B-2；(b) 走 B-1；(c) 先复测 ION 峰值再定。批复后放行实现。
