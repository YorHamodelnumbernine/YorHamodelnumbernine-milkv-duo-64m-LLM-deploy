# ION 双缓冲可行性预研 — ION 布局 / DMA 路径 / 收益估算

日期：2026-08-14 | 作者：TPU 底层工程师 | 性质：**决策输入，不落地生产代码** | 关联：`DESIGN_PATH_A_TWOPASS §9a`、`GATE_A_SIGNOFF`、`decode_rsh1_v0.log`、`layer_read_bench.log`

---

## 0. 结论先行（TL;DR）

1. **ION 布局：能容纳 2×8.4MB 层双缓冲。** 24 MiB carveout 内 SD_BUF_A/B = 2×8.4 MiB +
   其余工作区，合计 ~19.7 MiB，余 ~4.3 MiB。**但 §9a 现布局的 "8.0 MiB SD_BUF" 比真实
   layer（8,393,728 B = 8.005 MiB）短 5,120 B** —— 这是既有规划里的一个尺寸 bug，必须
   修正为 8.4 MiB。
2. **DMA/g2l 路径：可行，且副核 SDIO 直读固件已就绪。** 副核 C906L（FreeRTOS）已内置
   `CMD_MHA_SD_TAKE_OWNER / SD_READ_LAYER / SD_RELEASE`（mbox 0x30–0x32）+ `sdhci_cv180x.c`
   ADMA2 读驱动，可把 SD 块直接 DMA 到任意物理地址（含 ION PA）。**但 Linux 侧无调用者、
   带宽从未实测** —— 这是本方案最大的不确定性。低风险替代：Linux O_DIRECT pread → ION
   （`layer_read_bench` 已实测 20.57MB/s）。
3. **收益：13.02 → ~9.5–10.5 s/token（约 -25%），不是 CEO 估的 7.7s。** 原因：全量权重
   201MB（192.1 MiB）@ ~20.5–21.5 MiB/s 的 SD 地板是 **8.9–9.4s**，而 TIU+CPU 计算地板仅
   ~5.2–5.5s。SD 读（9.4s）**藏不进**计算（5.5s）后面，双缓冲只能让流水线变成 SD-bound，
   无法把 SD 读从关键路径上完全移除。
4. **是否立项：值得，但有前置条件。** 先做两个 Gate 测量（副核 SDHCI 带宽 / O_DIRECT→ION
   验证），≥18MB/s 再 commit。总工作量 ~3.5–5.5 人日。

---

## 1. ION 布局核查

### 1.1 真实 layer 字节构成（逐字段，parse_layer 验证）

| 分量 | B/layer | MiB/layer | ×24 |
|---|---|---|---|
| INT4 nib（K-aligned G32） | 7,454,720 | 7.11 | 170.6 MiB |
| fp16 gsc（per-group scale） | 931,840 | 0.89 | 21.3 MiB |
| rms 表 | 7,168 | 0.007 | 0.17 MiB |
| **合计** | **8,393,728** | **8.005** | **192.1 MiB（=201.4 MB 十进制）** |

gsc 占 SD 流量 11.1% —— 冷读 gsc 与 nib 同源，ION 双缓冲一并掩盖，无额外冷读。

### 1.2 ION 布局预算（24 MiB carveout）

| 区域 | 大小 | 角色 |
|---|---|---|
| SD_BUF_A | 8.4 MiB（8,397,312 B 对齐） | 当前层 layerL INT4 nib+gsc，C906L/O_DIRECT DMA 写入 |
| SD_BUF_B | 8.4 MiB | 下一层 layerL+1（双缓冲，读与消费重叠） |
| DQ_BUF_A / B | 2 × 160 KiB | [32,4864] int8 解包块（TIU g2l 读源，CPU 双缓冲） |
| P1 / P2 | 2 × 64 KiB | pass1/pass2 输出 |
| ACT | 32 KiB | 激活 staging |
| KV/INT8 预留 | 1.0 MiB | KV cache / 杂项 |
| **合计** | **~19.7 MiB** | **< 24 MiB，余 ~4.3 MiB** |

**结论：能容纳。** 相比 §9a 现布局（18.84 MiB 用 / 5.16 MiB 余）只多占 ~0.8 MiB（8.0→8.4
MiB ×2）。KV 余量从 5.16 → ~4.3 MiB，仍可支撑 INT8 KV ≥500 token 或 LM head 热区。

**§9a 既有 bug**：`SD_BUF 8.0 MiB`（8,388,608 B）比 layer 文件 8,393,728 B 短 5,120 B ——
真实现布局必须用 8.4 MiB，否则 DMA 越界。

---

## 2. DMA/g2l 路径

### 2.1 现有路径（答复"g2l 是 DDR→ION 还是 SD→ION"）

```
SD ──mmap/pread──▶ page cache(DDR) ──CPU dequant──▶ ION DQ ──TDMA g2l──▶ LMEM
```
- **TDMA g2l 读的是 ION 全局内存**（DQ 缓冲），不是 DDR、也不是 SD。
- 目前 **没有 SD→ION 直接 DMA**：SD 读走 Linux VFS（mmap demand-paging 或 pread），
  CPU 解包时写入 ION DQ。
- mmap 冷页 fault 与 TIU 计算串行是当前 13s 的结构性浪费（fault 等待 ~5.3s 落在 dequant/
  other 桶）。

### 2.2 ION 双缓冲的两条 DMA 实现路径

**选项 A（低风险，推荐先验）— Linux O_DIRECT pread → ION SD_BUF**
- `layer_read_bench` 已实测 `pread_odirect = 20.57MB/s`（24 种机制里最快之一）。
- MMC 控制器 bus-master DMA 直写 ION carveout 物理页（carveout 固定 PA、非 swapable、
  天然 pin），**无 copy_to_user**，C906B 后台线程几乎零 CPU。
- 与 TIU 重叠：异步线程发 pread（阻塞在内核等 DMA），主线程照常跑 TIU/解包。
- 改动面：引擎层读模式加一个 `LW_READ=ion` 分支 + 一个预读线程 + ION SD_BUF 双缓冲。
- 风险：O_DIRECT 要求 4KB 对齐 + 扇区对齐（ION 满足）；vfat 对 O_DIRECT 的支持需实测。

**选项 B（高天花板，高风险）— 副核 C906L SDHCI 直读 → ION**
- 固件**已就绪**：`comm_main.c` 有 `CMD_MHA_SD_TAKE_OWNER`(0x30)/`SD_READ_LAYER`(0x31)/
  `SD_RELEASE`(0x32)；`sdhci_cv180x.c` 用 CMD18+ADMA2 把 `num_blocks` 读到任意 `dst_paddr`。
- 副核被占时做同步 DMA（polling），**与主核 TIU 完全并行** —— 架构上最优。
- **约束**：① 按 LBA 寻址（Linux 侧需 FIBMAP/FIEMAP 把文件偏移→LBA 映射，含非连续 extent）；
  ② `MAX_ADMA_DESCS=64`、每 desc ≤64KB → **单次最多 4MB**，8.39MB 层需 **2 次 mbox 调用**
  （每次 ~4.2MB）；③ DMA 后副核已 `inv_dcache_range`，主核仍需对 ION SD_BUF 做 MemInvld。
- **未实测带宽**：同一 SDIO 控制器（4-bit、high-speed），物理上限与 Linux 路径同量级
  （~20–22MB/s），但 polling 无 readahead，可能略低 —— **必须 Gate 实测**。

### 2.3 副核 SDIO 可用性评估（直接回答）

- **固件能力：有。** SDHCI 驱动 + mbox 命令已内置在 FreeRTOS 副核。
- **当前使用状态：未接入。** `common/mha_descriptor.h` 定义了 0x30–0x32，但整个 tpu_bench
  无 Linux 侧调用者；带宽/正确性从未上板实测。
- **落地成本**：LBA 映射（Linux 侧）+ mbox 异步封装 + ≤4MB 分块 + 缓存一致性处理，
  估计 1–1.5 人日固件/驱动面验证 + 引擎集成。

---

## 3. 收益估算

### 3.1 现状分解（decode_rsh1_v0 实测，VERIFY=0 / RSH=1，per-token）

| 分量 | 耗时 | 备注 |
|---|---|---|
| t_head（LM head 两段式） | ~0.6 s | 已优化 |
| runcmdbuf（TIU） | ~4.6 s | 28,800 calls @0.161ms |
| dequant（含 SD nib fault） | ~4.0 s | 纯 RVV 数学仅 ~1.5–2s，余为 fault |
| other（accum+blockmax+SD gsc fault） | ~3.1 s | 纯数学 ~1–1.5s，余为 fault |
| flush/invld/copy_act | ~0.8 s | g2l/l2g 缓存维护 |
| **t_layers 合计** | **~12.4 s** | +t_head ≈ **13.0–14.0 s/token** |

CEO 的 "~5.3s 落在 dequant 桶" = dequant(4.0) + other 内 fault 份额 ≈ SD 读串行化部分。

### 3.2 理论上限（关键修正）

| 场景 | per-token | 依据 |
|---|---|---|
| 现状（CEO 口径） | 13.02 s | 实测 |
| 隐藏 5.3s fault（CEO 假设） | ~7.7 s | **不成立**：SD 读并未消失 |
| **ION 双缓冲（可行上限）** | **~9.6–10.5 s** | 流水线 → SD-bound |

- **SD 读地板 = 192.1 MiB / (20.5–21.5 MiB/s) = 8.9–9.4 s**，物理不可再压（同控制器/卡）。
- 计算地板 = max(TIU 4.6, CPU 解包+累加 ~3–3.5) + lmhead 0.6 ≈ **5.2–5.5 s**。
- 双缓冲后 per-token = max(SD 9.4, 计算 5.5) + 首尾/同步气泡 ≈ **9.6–10.5 s**。
- **SD 读（9.4s）藏不进计算（5.5s）后面**：除非同时把权重字节降到 ~165MB 以下（INT4 已
  是下限，无空间）或 SD 带宽物理翻倍（不可能），7.7s 不可达。

### 3.3 结论

- **可立项**：省 ~2.5–3.5 s/token（~20–27%），是当前唯一能动的最大单项。
- **上限是 SD-bound ~9.5s**，不是计算-bound 7.7s。立项目标应锚定 9.5–10.5s。

---

## 4. 工作量 / 风险

### 4.1 工作量（~3.5–5.5 人日）

| 项 | 人日 | 说明 |
|---|---|---|
| Gate 测量（前置） | 0.5–1 | ① 副核 SDHCI 带宽（选项 B）② O_DIRECT→ION 验证（选项 A） |
| ION 布局重构 + DQ 双缓冲 + 软件流水 | 1–1.5 | SD_BUF_A/B + DQ_A/B，dequant g+1 与 TIU g 重叠 |
| LBA 映射（vfat FIBMAP/FIEMAP） | 0.5–1 | 非连续 extent 分块 |
| mbox SD 读集成 / O_DIRECT 线程 | 0.5–1 | TAKE_OWNER→READ_LAYER(≤4MB×2)→RELEASE + 异步 poll + MemInvld |
| 回归（3/3 + bit-exact + 红线） | 0.5 | 复用既有回归框架 |

### 4.2 风险

| 风险 | 等级 | 缓解 |
|---|---|---|
| **副核 SDHCI 带宽未实测**（<15MB/s 则选项 B 无收益甚至回退） | **高** | 前置 Gate；低于阈值直接切选项 A（O_DIRECT 已证 20.57） |
| vfat LBA 映射（FIBMAP 在 vfat 的可用性 / 碎片多 extent） | 中 | Gate 验证；极端可做 raw 权重 blob/分区 |
| 副核 DMA 后 ION 缓存一致性（错则权重静默损坏） | 中 | 副核 inv_dcache + 主核 MemInvld SD_BUF 后置；bit-exact 回归兜底 |
| ADMA 4MB 上限 → 每层 2 次 mbox 同步开销 | 低–中 | 2 次/层开销 ~µs 级，可忽略；同步语义已现成 |
| O_DIRECT 对齐（vfat/扇区/ION 页对齐） | 低–中 | 层文件偏移 %512==0；ION 页对齐 |
| §9a SD_BUF 8.0 MiB 尺寸 bug（短 5,120B） | 低 | 本设计已用 8.4 MiB |

---

## 5. 建议（决策输入）

1. **立项值得**，但把收益锚定 **~9.5–10.5s/token（-25%）**，不要按 7.7s 设验收线。
2. **先跑两个前置 Gate 再 commit**：
   - G-ION-1：副核 `CMD_MHA_SD_READ_LAYER` 裸读实测带宽（≥18MB/s 才走选项 B）。
   - G-ION-2：Linux O_DIRECT pread → ION carveout 实测（作为选项 A 兜底）。
3. **默认路线建议**：选项 A（O_DIRECT→ION）先行，选项 B（副核 SDHCI）作为二期增强；
   两者共用同一 ION 布局 / DQ 双缓冲 / 流水线骨架，改动面不重复。
4. 实施由推理引擎工程师主导引擎侧，TPU 侧负责 Gate 测量 + bmk1822/g2l 缓冲一致性签核。
