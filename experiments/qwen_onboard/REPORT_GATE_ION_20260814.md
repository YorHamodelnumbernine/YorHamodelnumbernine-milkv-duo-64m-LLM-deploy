# ION 双缓冲前置 Gate 实测报告 — G-ION-1 / G-ION-2 + 11.29s 基线收益重估

日期：2026-08-14 | 作者：TPU 底层工程师 | 性质：**Gate 决策输入（上板实测）** | 关联：`REPORT_ION_DOUBLEBUF_FEASIBILITY_20260814.md`、`GATE_A_SIGNOFF`、`decode_e1_v0.log`

> **板子状态警告**：G-ION-1 实测过程中副核 SDHCI 接管导致**整板内核级死锁**，当前
> 192.168.42.1 不可达（USB NCM 仍枚举、ICMP/ARP 无响应），**需物理断电重启**。
> 详见 §4。本报告数据在死锁前已取得（G-ION-2 完整、G-ION-1 的"接管即死锁"本身就是结论）。

---

## 0. 结论先行（TL;DR）

1. **G-ION-2（O_DIRECT→ION）：纯零拷贝路线判死。** O_DIRECT 直接 `pread` 到 ION
   carveout VA **失败 EFAULT**（errno=14）——ION 是 VM_PFNMAP 型内存，`get_user_pages`
   无法 pin，内核 direct-IO 不支持直写。对齐约束本身正常（错位 offset→EINVAL，已证）。
   → **选项 A 必须降级为 A'**：O_DIRECT/缓冲读 → 对齐 bounce → CPU memcpy → ION。
   memcpy 成本 ~10ms/层（ION 写 753MB/s），可忽略；SD 带宽不变。
2. **G-ION-1（副核 SDHCI 接管）：按现协议直接判不可用。** `CMD_MHA_SD_TAKE_OWNER`
   在 Linux 仍持有 SD（rootfs 在其上）时触发 **整板死锁**。副核 `sdhci_cv180x_init`
   对共享控制器做软复位+时钟重配，与 Linux mmc 驱动冲突。→ **选项 B 需先解决控制器
   交接（boot 期独占 or Linux mmc 静默），非现成可用；且其带宽上限与 A' 相同
   （同控制器 ~20–22MB/s）。**
3. **收益重估（11.29s 基线，CEO 关键问题）：<10s 成立。** 双缓冲把 SD nib 读
   （6.1–8.3s）藏进计算（~4.1s）后，decode 上限 **≈ 6.9–9.1s/token**（乐观 6.9 /
   保守 9.1，中枢 ~8s）。比旧可行性报告（13.02→9.5–10.5s）更好——因为 gsc 已 ION
   缓存，剩 SD 读仅 nib。
4. **立项建议：值得，但路线改为「A'（Linux 读 + memcpy 入 ION）为主，B 搁置待交接方案」。**
   省 ~2.3–4.4s/token（20–39%），是当前唯一能动的最大单项。

---

## 1. 前置：当前 11.29s 基线分解（decode_e1_v0.log，gsc ION 缓存后）

decode 6 步平均 11.29s/token。per-token 分量（PROFILE ÷6）：

| 分量 | s/token | 性质 |
|---|---|---|
| dequant_rvv | **6.12** | ~纯 SD nib 读等待（RVV 数学仅 ~0.02s，见 bench_gsc 8938MB/s） |
| runcmdbuf(TIU) | 2.82 | 纯计算（92,160 calls @0.183ms） |
| accum(CPU fp32) | 0.54 | 纯计算 |
| blockmax | 0.18 | 纯计算 |
| flush/invld/copy_act | 0.43 | DMA 维护 |
| t_head(LM head) | 0.50 | 已优化 |
| 计时 gap | 0.11 | 开销/未计入 |
| **合计** | **~10.7** | ≈11.29 实测 |

- **计算地板（非 SD）≈ 4.1s/layer 流程 + 0.5s head ≈ 4.6s/token。**
- **SD nib 读 = 170.6MiB**：
  - 现引擎 mmap+readahead 有效带宽 = 170.6MiB/6.12s ≈ **27.9MiB/s**（当前实测值）；
  - O_DIRECT 1MiB 块 = **20.5MB/s**（layer_read_bench 实测）；
  - SD 物理顺序上限 ≈ **20.5–21.5MB/s**（同控制器 4-bit high-speed）。

---

## 2. G-ION-2：O_DIRECT→ION（上板实测，gate_ion2_odirect）

**结果：纯零拷贝判死。**

```
G-ION-2 O_DIRECT->ION  file=/data/qwen/layer0_kal.bin size=8393728
ION alloc OK va=0x3fd350f000 pa=0x82473000  (page-align va=0 off=0)
open(O_DIRECT) OK
  [align-probe] misaligned offset=1: r=-1 errno=22 (EINVAL)  OK: O_DIRECT 强制对齐
  pread(O_DIRECT) off=0 n=1048576 r=-1 errno=14 (EFAULT)     <- 致命
```

- **对齐**：ION VA 页对齐 ✓；错位 offset 返回 EINVAL ✓（对齐约束正常生效）。
- **DMA 后一致性**：无法进入该阶段——第一个对齐的 1MiB pread 就 EFAULT。
- **原因**：CVI_RT 的 ION carveout VA 是 **VM_PFNMAP/特殊映射**，其页无法被内核
  direct-IO 的 `get_user_pages` 固定（pin）→ EFAULT。这是 ION 内存的固有性质，
  不是对齐或驱动 bug。

**降级路径 A'（可行）**：
```
SD ──O_DIRECT/缓冲 pread──▶ 对齐 bounce(DMA 可达) ──CPU memcpy──▶ ION SD_BUF ──dequant──▶ DQ ──g2l──▶ TIU
```
- bounce 读带宽 = 20.5MB/s（O_DIRECT）或更高（缓冲/readahead）；
- memcpy 7.11MiB/layer @ ~753MB/s（bench_dequant ION 写实测）= **9.4ms/层** ≈ 0.23s/token，可忽略且可与读重叠。

---

## 3. G-ION-1：副核 SDHCI 裸读（上板实测，gate_ion1_sdhci）

### 3.1 已取得的部分（死锁前打印）

```
G-ION-1 SDHCI->ION  file=/data/qwen/layer0_kal.bin size=8393728 dev=0xb302 part_start_sector=262145 fsblk=1024
  extents: 4
    [0] logical=0      phys_sector=5638146 bytes=1048576  -> LBA=5900291
    [1] logical=1048576 phys_sector=5648386 bytes=1048576  -> LBA=5910531
    [2] logical=2097152 phys_sector=5656578 bytes=6291456  -> LBA=5918723
    [3] logical=8388608 phys_sector=5672962 bytes=5120     -> LBA=5935107
```

- **FIEMAP→LBA 映射验证通过**：文件 4 个 extent，物理扇区连续（5638146→5648386→5656578
  →5672962），加分区起始 262145 得绝对 LBA，映射正确。

### 3.2 死锁（决定性结论）

- 程序在打印上述映射后继续执行：O_DIRECT 参考读（8.4MB，正常）→ ION alloc →
  **`CMD_MHA_SD_TAKE_OWNER`（副核 `sdhci_cv180x_init`）**。
- 接管后数秒内**整板无响应**：ICMP/ARP 全丢，USB NCM 仍枚举（SoC 有电）但内核冻结。
  SSH/串口均不可达，需物理断电。
- **A/B 对照**：修改前版本（只跑 FIEMAP+O_DIRECT，不碰 SDHCI）板子持续存活、后续
  多轮探测正常；本版进入 TAKE_OWNER 后冻结 → **归因副核 SDHCI 接管**。

**根因分析**（读 `sdhci_cv180x.c`）：`sdhci_cv180x_init` 直接操作共享 SDHCI 控制器
寄存器——**软复位（SDHCI_RESET_ALL）+ 时钟门控/分频重配 + PAD 复用改写**。Linux 侧
mmc 驱动（rootfs 就在这卡上）同时持有该控制器，接管导致双方状态冲突 → mmc 请求
永久挂起 → 内核级级联死锁（含网络栈）。**这不是带宽问题，是控制器交接协议不安全。**

### 3.3 对选项 B 的裁决

| 项 | 结论 |
|---|---|
| 固件命令存在 | ✓（0x30/0x31/0x32 + sdhci_cv180x.c ADMA2） |
| Linux 侧可安全调用 | **✗ 现协议死锁** |
| 带宽实测 | 未能取得（接管即死锁）；但同控制器上限已知 ~20–22MB/s |
| 是否值得继续 | **搁置**，除非做「boot 期独占」或「Linux mmc 静默」交接方案（工程量>原估） |

---

## 4. 板子恢复与安全重测计划

- **现状**：192.168.42.1 不可达，需**物理断电重启**（我侧无 root/sudo、无串口、USB
  设备节点 root-only，无法软件复位）。
- **重启后安全 G-ION-1 变体（可选）**：若 CEO 仍要副核 SDHCI 带宽数字，唯一安全路径是
  **先让 Linux 释放 SD**——但 rootfs 在 SD 上，Linux 无法卸载。因此现实选项是：
  (a) 在 U-Boot/early-initramfs 阶段（Linux mmc 绑定前）由副核独占 SD 预读权重，
      与「boot-from-SD」冲突，需改启动架构；
  (b) 放弃选项 B 的带宽实测，直接采用其物理上限（~20–22MB/s）作为规划输入。
- **建议**：不阻塞主项目，按 §6 走选项 A'。G-ION-1 带宽作为已闭合项（上限已知、
  协议不可用）归档，不追加投入。

---

## 5. 收益重估（CEO 关键问题：是否 <10s？）

**答案：成立。11.29 → ~6.9–9.1s/token（全部 <10s，中枢 ~8s）。**

双缓冲后（背景线程预读 layer L+1 nib 入 ION SD_BUF_B，主线程处理 layer L）：

```
per-token t_layers ≈ max(SD_nib_read, 计算地板) + 流水气泡(~0.3s)
per-token t_total  ≈ t_layers + t_head 0.5s
```

| 场景 | SD nib 读 | 计算地板 | t_layers | **t_total** |
|---|---|---|---|---|
| 乐观（预读达现 27.9MiB/s） | 6.1s | 4.1s | 6.4s | **6.9s** |
| SD 物理上限 21.5MiB/s | 7.9s | 4.1s | 8.2s | **8.7s** |
| 保守（O_DIRECT 式 20.5MiB/s） | 8.3s | 4.1s | 8.6s | **9.1s** |

- **关键修正（相对旧可行性报告）**：旧报告 13.02→9.5–10.5s。现在 gsc 已 ION 缓存、
  基线降到 11.29s，且剩 SD 读仅 nib 170.6MiB（而非 201MB），故上限收窄到 **6.9–9.1s**。
- **SD-bound 不变**：SD 读（6.1–8.3s）仍 > 计算地板（4.1s），但双缓冲使其与计算重叠，
  per-token 逼近 max() 而非串行和。**7.7s 级"纯隐藏"仍不可达**（那是假设 SD 读不花时间），
  但 <10s 在全部三档都成立。
- **为什么比 mmap 双缓冲（Phase 7b 判负）好**：Phase 7b 依赖 14MB page cache 装不下
  预读层；ION carveout 是专用固定物理内存（SD_BUF_A/B 2×8.4MiB），不占 page cache，
  预读层必然驻留。

---

## 6. 推荐路线与工作量修正

### 6.1 路线（修改自旧报告"默认选项 A"）

| 路线 | 状态 | 说明 |
|---|---|---|
| **A'：Linux 读 + bounce memcpy → ION**（推荐） | 立即可做 | 缓冲/readahead 读（现引擎已证 27.9MiB/s）+ memcpy 入 ION；无固件改动 |
| ~~A：O_DIRECT 零拷贝→ION~~ | **判死** | G-ION-2 EFAULT |
| ~~B：副核 SDHCI 直读→ION~~ | **搁置** | G-ION-1 接管死锁；需 boot 期独占交接方案 |

### 6.2 工作量（A' 主路）

| 项 | 人日 | 说明 |
|---|---|---|
| 引擎加预读线程 + ION SD_BUF_A/B 双缓冲 + 软件流水 | 1.5–2.5 | 背景 pread 下一层 nib→bounce→memcpy→ION，与 TIU 重叠 |
| DQ 双缓冲 + g2l 源切换（读 ION 而非 mmap） | 0.5–1 | 复用既有 ION DQ 机制 |
| 回归（3/3 + bit-exact + 红线） | 0.5 | 复用既有框架 |
| **合计** | **~2.5–4** | 比旧估（3.5–5.5）略降：砍掉 LBA 映射 + mbox SD 集成 |

### 6.3 风险表更新

| 风险 | 等级（更新） | 缓解 |
|---|---|---|
| O_DIRECT 直接入 ION（原低-中） | **关闭**（G-ION-2 判死） | 改用 A' bounce+memcpy |
| 副核 SDHCI 带宽未实测（原高） | **升级为「协议死锁」** | 选项 B 搁置；带宽上限取同控制器 20–22MB/s |
| §9a SD_BUF 8.0MiB 尺寸 bug | 低（未变） | 实现用 8.4MiB |
| 预读线程与 TIU 争 CPU | 低 | pread 阻塞在内核，CPU 占用近零；memcpy ~10ms/层 |

---

## 7. 建议（决策输入）

1. **立项，目标锚定 6.9–9.1s/token（-20~39%），验收线设 9.5s（留余量）。**
2. **路线走 A'（Linux 读 + memcpy 入 ION），不依赖固件；B（副核 SDHCI）搁置**，除非
   CEO 另行拍板做 boot 期独占架构。
3. **请物理断电重启 Duo** 以恢复 G-ION-1 后续实验环境；重启后我侧先做一次连通性回归
   （ping/SSH/`layer_read_bench` 基线），再按 A' 开实现。
4. 实施由推理引擎工程师主导引擎侧（预读线程+流水），TPU 侧负责 ION SD_BUF/DQ 布局
   与 g2l 一致性签核。

---

## 附：产物

- `gate_ion2_odirect.c` / `gate_ion1_sdhci.c`（本 Gate 上板探针，已推送 /data/qwen）
- `bench_gsc` / `bench_dequant`（复用既有带宽工具，引用数据）
- `decode_e1_v0.log`（11.29s 基线，PROFILE 分解）
