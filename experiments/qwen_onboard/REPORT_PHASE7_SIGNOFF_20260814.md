# Phase 7 全链路优化签核报告（里程碑收口）

日期：2026-08-14 | 作者：推理引擎工程师
状态：**正式签核 / 里程碑收口**（CEO 口径确认，decode 生产口径 VERIFY=0 基线锚定 **11.29s/token**）
关联：`qwen_engine_lmhead2.c` / `dequant_kal.c` / `merge_diag.c` / `phase7e_run.sh` / `phase7e_deploy_run_host.sh`
日志：`phase7b2_run_board.log` / `decode_rsh1_v0.log` / `decode_e0_v0.log` / `decode_e1_v0.log` / `decode_e1_v1.log`
衔接：`REPORT_GATE_ION_20260814.md` / `REPORT_ION_DOUBLEBUF_FEASIBILITY_20260814.md`

---

## 0. 签核结论（TL;DR）

1. **decode（VERIFY=0 生产口径）累计 19.97 → 11.29 s/token，-8.68 s（-43%），12s 目标达成。**
2. 全链四阶段全部验收：**7b 负结果收口（基线锚定）→ 7c rsafe 离线预标定 → 7d TIU 提交合并 → 7e ION gsc 缓存**。
3. 验收红线全过：**NEXT 回归 3/3、TIU bit-exact（bad1=bad2=r_opt=0）、rsafe 表 vs 运行时扫描 rsh=0、decode min_gap ≥0.05、VmSwap 稳态**。
4. 当前基线：**11.29 s/token**（`decode_e1_v0.log`，gsc 全 24 层常驻 ION）。
5. 剩余主体为 SD nib 单遍读（~6.1s/step）+ TIU 提交（~2.8s/step），已无可算法杠杆；
   <10s 路径已立项 **A' 路线（ION double-buffer，Linux 读 + memcpy 入 ION）**，验收线 **9.5 s**（预期 6.9–9.1 s/token）。

| 阶段 | 杠杆 | decode | 累计 | 备注 |
|---|---|---|---|---|
| 7b 前 | — | **19.97 s/token** | — | mmap 裸读基线（生产口径 VERIFY=0） |
| 7c | rsafe 离线预标定 | **→13.99 s** | -5.98 s（-30%） | 去 wmax 预扫双读（RSH=1 查表） |
| 7d | TIU 提交合并 | **→13.02 s** | -0.97 s（-6.9%） | up/gate 合并 8×608 单 cmdbuf |
| 7e | ION gsc 缓存 | **→11.29 s** | -1.73 s（-13.3%） | 全 24 层 gsc 21.33MiB 常驻 ION |

> 注：7d 阶段 `decode_e0_v0.log` 单次记录 13.11s；CEO 口径取多轮稳态中值 **13.02 s**。7e A/B 在同一基线（13.11/13.02）上对比，结论不受该 ±0.09s 运行方差影响。

---

## 1. 里程碑范围与数据链总览

Phase 7 目标：击穿 decode 两个结构性大头——**SD 层权重读** 与 **TIU 提交开销**。数据链按
「负结果收口 → 三项正向杠杆」推进，全部在 Duo 板上实测（drop_caches 冷启动，6 步稳定区间，
生产口径 VERIFY=0 PROFILE=1）。

| 日志 | 对应阶段 | 关键改动 | decode 实测 |
|---|---|---|---|
| `phase7b2_run_board.log` | 7b 负结果收口 | mmap 裸读（Phase 6 基线） | **19.97 s/token** |
| `decode_rsh1_v0.log` | 7c | + rsafe 离线表（RSH=1 skip 预扫） | **13.99 s/token** |
| `decode_e0_v0.log` | 7d | + up/gate TIU 提交合并 | **13.11 s/token**（CEO 口径 13.02） |
| `decode_e1_v0.log` | 7e | + gsc 全量 ION 缓存 | **11.29 s/token** |

---

## 2. Phase 7b — 负结果收口与基线锚定（19.97s）

mmap 系跨层双缓冲被硬件/OS 硬约束否定，三条根因成立：

1. **单核可见**：Linux 仅见 C906B（cpu0），C906L 为 FreeRTOS，不可用于主控侧双缓冲。
2. **页缓存不足**：页缓存 ~13.9MB < 2×8.39MB 层文件 → 预取层被逐出放大。
3. **同步 readahead 阻塞**：整层预取在单核上阻塞消费。

**关闭此方向。** 基线锚定更新为 VERIFY=0 **19.97 s/token**（mmap 裸读）。
同期读路径归因（`REPORT_PHASE7_READPATH_20260814.md`）：裸 mmap demand-paging 仅 ~8MB/s
（页 fault 串行、readahead 跟不上），顺序 pread 可达 ~21.5MB/s——为 7e 之前各阶段的
结构性差距埋下归因基础。

---

## 3. Phase 7c — rsafe 离线预标定（19.97 → 13.99）

- **离线表**：`rsafe.bin`（264 B，24 层 × 11 项：q/k/v/wo/up/gate 各 1 + down 5 个 K-chunk），
  转换期对静态权重逐 tile 预计算；引擎启动载入，运行时 wmax 预扫改查表（`RSH=1`）。
- **关键发现**：对称 INT4（SYM_QMAX=7, G=32）下 24×11 项 wmax≡7 → rsafe≡5 恒定——
  **wmax 预扫是架构级冗余**，已彻底消除。rsafe 纯静态、无激活相关分量，无残余运行时成本。
- **rsafe 正确性验证**：`rsh(scan-vs-table)=0`（VERIFY=1 回归保留运行时扫描作 QA 安全网）。
- **收益机制**：去除 wmax 预扫双读（每 token ~201MB 二次读中的一次）≈ 本阶段主节省来源。
- **实测**：`decode_rsh1_v0.log` decode avg = **13.99 s/token**（6 步）。

---

## 4. Phase 7d — TIU 提交合并（13.99 → 13.02）

- **改动**：up/gate 的 N-tile 提交合并为单 cmdbuf（**8×608**，`MTILEW=608`, `MNT=8`），
  每 K-block pass 从 8 次提交 → 1 次。
- **提交数**：decode 36,000 → 22,560 次/token（**-47%**）；硬件 op 数不变（93,312/step），只砍提交开销。
- **成本**：8-tile 合并 cmdbuf 单次 0.183ms vs 逐 tile 0.115ms，净省 0.07ms/提交。
- **A/B**（6 步，drop_caches 冷启动）：13.99 → **13.02 s/token**（`decode_e0_v0.log` 单次记录 13.11s）。
- **位精确**：merged pass1/pass2 与 per-tile 在相同 DQ 数据下逐 n 位一致（`merge_diag.c` 已证），
  模型级 bad1=bad2=r_opt=rsh=0。

---

## 5. Phase 7e — ION gsc 缓存（13.02 → 11.29）

### 5.1 动机（"other" 桶拆分）
对 `other` 插桩子计时（`g_t_bm`/`g_t_acc`/`g_t_fin`）后定位最大子项：

| 子项 | 基线（GSC_ION=0） | 说明 |
|---|---|---|
| **accum（gsc 冷 mmap 读）** | **18.0 s/6 步** | 每 token 21.4MB gsc @ ~8MB/s（272KB 小区域无法拉满 SD readahead） |
| blockmax | ~1.07 s | 收集块最大值 |
| final | ~0.04 s | 输出缩放 |
| other-rest | ~0 s | RSH=1 已去 wmax 预扫 |

### 5.2 实施
- 全 24 层 × 7 个 gsc 区（q/k/v/wo/up/gate/down，共 **21.33 MiB**）启动时 `pread` 入单一 ION
  缓冲（CPU 缓存路径 ~2200MB/s，载入 1.345s），逐层 `LayerRef` gsc 指针改指 ION。
- **约束合规**：ION-only（无匿名 malloc 双缓冲）；数据缓冲、不重复 LoadDmabuf，
  单 LoadDmabuf 规则未触碰。
- gsc 区文件偏移与 `parse_layer` 布局逐字节吻合（`GSC_*_OFF/SZ` 宏镜像同一偏移序列）。

### 5.3 结果（6 步稳定区间，drop_caches 冷启动，`decode_e1_v0.log`）

| 项 | GSC_ION=0 | GSC_ION=1 | 变化 |
|---|---|---|---|
| **decode per-token** | 13.11 s | **11.29 s** | **-1.82 s（-13.9%）** |
| accum（冷 gsc 读） | 18.0 s/6步 | 3.2 s/6步 | -14.8 s/6步 |
| dequant | 5.29 s/step | 6.13 s/step | +0.84 s（gsc 缺页停顿消失后 SD 预取同步化，净收益仍正） |
| up matmul | 135.4 ms/call | 104.9 ms/call | -22.5% |
| gate matmul | 135.7 ms/call | 122.4 ms/call | -9.8% |

---

## 6. 验收红线与回归（VERIFY=1 复验 + 生产口径）

| 红线 | 结果 | 证据 |
|---|---|---|
| **3-prompt 回归 NEXT 3/3** | OK（2130 / 12095 / 99366） | decode_e1_v0.log |
| **TIU bit-exact** bad1=bad2 | 0 | P1/P2 回读逐字节比对 |
| **r_opt mismatch** | 0 | 两遍法 pass 间优化路径一致 |
| **rsafe 正确性**（离线表 vs 运行时扫描 rsh） | 0 | VERIFY=1 回归保留扫描作安全网 |
| **decode min_gap** | 0.50 / 4.87 / 10.20 / 1.42 / 1.10 / 5.85（均 ≥0.05） | 6 步 decode |
| **TIU 运行计数** | pass1=116352 pass2=116352 total=232704 | decode 段 6 步 |
| **VmSwap** | 稳态 ~2.4–2.85 MB | 无新增匿名缓冲（ION-only） |

- 全链回归在 `decode_e1_v1.log`（VERIFY=1）复验，bad1=bad2=0 / r_opt=0 / rsh=0。
- 位精确由构造保持：7c/7d/7e 均不改变 dequant 数值路径（gsc 字节同源、merged 与 per-tile
  逐 n 一致、rsafe 表由同布局静态推导），回归项全部通过。

---

## 7. 当前基线结构分解（11.29s/token，PROFILE ÷6）

`decode_e1_v0.log` PROFILE 插桩（eng_matmul 61.408s / 1584 calls + t_head）：

| 分量 | s/token | 性质 |
|---|---|---|
| dequant_rvv | 6.12 | ~纯 SD nib 读等待（RVV 数学仅 ~0.02s） |
| runcmdbuf(TIU) | 2.82 | 纯计算（92,160 calls @0.183ms） |
| accum（CPU fp32） | 0.54 | 纯计算 |
| flush/invld/copy_act | 0.43 | g2l/l2g DMA 维护 |
| blockmax | 0.18 | 纯计算 |
| t_head（LM head 两段式） | 0.50 | 已优化 |
| 计时 gap | 0.11 | 开销/未计入 |
| **合计** | **~10.7** | ≈11.29 实测 |

**计算地板（非 SD）≈ 4.1 s/layer 流程 + 0.5 s head ≈ 4.6 s/token**；剩余 ~6.1s 为 SD nib
读（170.6 MiB @ ~27.9MiB/s 实测有效带宽）。**已无可算法杠杆**，<10s 依赖 ION 双缓冲把
SD 读藏进计算。

---

## 8. 资源与内存

| 项 | 值 |
|---|---|
| ION 占用 | 引擎 ~1.2MB + gsc 21.33MiB ≈ 22.5MB |
| ION carveout | 26.8MB（debugfs 实测）；余 ~4.3MB |
| 匿名 buffer | 未新增（ION gsc 为唯一新增驻留） |

---

## 9. 衔接标注：A' 路线已立项（ION double-buffer，验收线 9.5s）

前置 Gate 结论（`REPORT_GATE_ION_20260814.md`，上板实测）已定，A' 立项无阻塞：

| Gate | 结果 | 裁决 |
|---|---|---|
| **G-ION-1**（副核 SDHCI 直读→ION） | `CMD_MHA_SD_TAKE_OWNER` 软复位+时钟重配与 Linux mmc 冲突，**整板内核死锁** | **搁置**（需 boot 期独占/交接架构，超出本项目） |
| **G-ION-2**（O_DIRECT 零拷贝→ION） | ION 为 VM_PFNMAP，`get_user_pages` 无法 pin → **pread EFAULT (14)** | **判死** |

**A' 路线（已立项）**：`SD ──缓冲/readahead 读──▶ 对齐 bounce ──CPU memcpy──▶ ION SD_BUF_A/B ──dequant──▶ DQ ──g2l──▶ TIU`
- memcpy 7.11 MiB/layer @ ~753MB/s = **9.4ms/层 ≈ 0.23s/token**，可与读重叠、可忽略。
- **预期收益**：11.29 → **~6.9–9.1 s/token**（乐观 6.9 / 保守 9.1，中枢 ~8s；三档全部 <10s）。
- **验收线：9.5 s/token（留余量）**。工作量 ~2.5–4 人日（预读线程 + ION SD_BUF_A/B 双缓冲 +
  软件流水 + DQ 双缓冲 + 回归），无固件改动。
- 实施由推理引擎工程师主导引擎侧；TPU 侧负责 ION SD_BUF/DQ 布局与 g2l 一致性签核。
- **板子状态**：G-ION-1 死锁后 192.168.42.1 不可达，**等待物理断电重启**；重启后先做
  连通性回归（ping/SSH/`layer_read_bench` 基线）再开 A' 实现。

---

## 10. 交付物清单

| 文件 | 说明 |
|---|---|
| `qwen_engine_lmhead2.c` | 引擎（RSH 查表 + merged cmdbuf + ION gsc 缓存 + 子计时） |
| `qwen_engine_lmhead2_phase7e` | riscv64 二进制 |
| `merge_diag.c` | merged vs per-tile 位一致性诊断 |
| `bench_ion.c` / `bench_gsc.c` / `bench_dequant.c` | ION/gsc/dequant 微基准 |
| `phase7e_run.sh` / `phase7e_deploy_run_host.sh` | 板上 A/B + 主机部署编排 |
| `phase7b2_run_board.log` | 7b 基线（19.97s）原始日志 |
| `decode_rsh1_v0.log` | 7c 实测（13.99s）原始日志 |
| `decode_e0_v0.log` | 7d 实测（13.11s）原始日志 |
| `decode_e1_v0.log` | 7e 实测（**11.29s，当前基线**）原始日志 |
| `decode_e1_v1.log` | VERIFY=1 全链复验日志 |
| `REPORT_GATE_ION_20260814.md` / `REPORT_ION_DOUBLEBUF_FEASIBILITY_20260814.md` | A' 前置 Gate 与可行性（衔接） |
| `REPORT_PHASE7_SIGNOFF_20260814.md` | 本报告（里程碑收口） |

---

## 附：数据链对照（CEO 口径 vs 日志记录）

| 阶段 | CEO 口径 | 日志记录 | 日志 |
|---|---|---|---|
| 7b 基线 | 19.97 | 19.97 | phase7b2_run_board.log |
| 7c rsafe | 13.99 | 13.99 | decode_rsh1_v0.log |
| 7d TIU 合并 | **13.02** | 13.11 | decode_e0_v0.log |
| 7e gsc ION | **11.29** | 11.29 | decode_e1_v0.log |
