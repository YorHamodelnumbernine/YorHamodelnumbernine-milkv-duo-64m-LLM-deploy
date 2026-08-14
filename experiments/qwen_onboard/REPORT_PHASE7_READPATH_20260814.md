# Phase 7 首任务 · 层权重读路径带宽归因与修复 — 实施就绪 + 上板阻塞

日期：2026-08-14 | 作者：推理引擎工程师 | 关联：`qwen_engine_lmhead2.c`（LW_READ 三档读模式）/ `layer_read_bench.c` / `decode10.log`
状态：**主机侧完成（代码/构建/部署脚本就绪），上板实测被硬件断开阻塞（Duo USB 于 08-14 00:33 被拔出）**

---

## 0. 结论摘要

1. **根因确认（机制层面）**：层权重裸 `mmap` demand-paging 是 ~8MB/s 的直接原因——每 4KB
   页 fault 在单核 C906B 上串行（trap→页分配→SD I/O 提交→等→建 PTE），fault 间隙 SD 空闲，
   内核按需 readahead（默认 ~128KB 窗口）跟不上消费节奏 → SD 只有 ~8MB/s 有效吞吐，而
   顺序 `pread/read` 一次性提交大块顺序 I/O 可跑满 ~21.5MB/s。
2. **与 Stage2 LM head 先例同一机制**：embed span 裸 mmap 每页 fault = 3.1MB/s；改 offset
   升序 pread = 17–35MB/s（~2–10×）。层权重读大概率同因（8 vs 21.5，~2.7×）。
3. **已实施修复（`qwen_engine_lmhead2.c`）**：新增 `LW_READ` 三档读模式开关——
   - `LW_READ=mmap`（默认）：Phase 6 基线，行为与改前逐位一致；
   - `LW_READ=mmap_ra`（主修复）：mmap + `readahead()` 整层预取（512KB 分片循环，规避
     单次上限）+ `madvise(MADV_SEQUENTIAL)`——**swap-safe（不引入匿名 buffer）**，目标
     把读吞吐从 ~8 收敛到 ~21.5MB/s；
   - `LW_READ=pread`（对照）：顺序 pread 整层入复用 buffer（proven 21.5MB/s，但 +8.4MB
     匿名，有 swap 抖动风险，仅作对照）。
4. **bit-exact 由构造保持**：三种模式读到的字节完全一致（同一文件），parse_layer 各 tensor
   偏移全为 16 的倍数（验证：24 个 layer 文件尺寸 8393728B 与布局计算逐字节吻合），mmap 基址
   4096 对齐 / malloc 基址 16 对齐下指针对齐相同 → 数值/TIU/LM head 路径不变。
5. **上板阻塞**：Duo 板 USB 被物理拔出（`journalctl -k`: `usb 1-3: USB disconnect 08-14 00:33:22`），
   本机无法 ping/SSH。全部上板验收（per-token 前后对比、归因表、3/3 回归）需人工重插 USB 后跑
   `phase7_deploy_run_host.sh` 一键完成。

---

## 1. 问题与基线（decode10.log 实测）

| 项 | 值 | 备注 |
|---|---|---|
| decode per-token（10 步平均） | **25.98 s/token** | decode10.log |
| 其中 t_layers（24 层权重读 + 计算） | ≈25.3 s | 201.4MB ÷ 25.3s ≈ **7.96 MB/s** |
| 层权重总量（24×layerN_kal.bin） | 201.4 MB | 每 token 全量重读 |
| SD 顺序读天花板（Phase 6 实测） | **21.5 MB/s** | 4k/64k/1M 块不敏感 |
| 层权重读地板（@21.5MB/s） | ≈9.4 s/token | 目标收敛值 |
| 预期 decode 收敛 | ~11–12 s/token | 9.4 + LM head 0.6 + 计算残差 |

---

## 2. 归因：为什么 mmap 流读 ~8MB/s 而顺序读 21.5MB/s

### 2.1 机制
引擎逐层 `mmap(layerN_kal.bin)` 后不主动读，由 `eng_matmul` 逐 K-block 解量化时**增量解引用**
→ 页按需 fault。单核 C906B 上每个 page fault 串行：trap → 页分配 → block layer 提交 SD I/O
→ 等 I/O 完成 → 建 PTE。fault 间隙 SD 设备空闲。

内核 readahead 默认窗口 ~128KB：消费者触到当前窗口末尾才触发下一批预取，于是 SD 以
「128KB 突发 → 消费者计算 → 128KB 突发」方式工作。设 128KB 突发耗时 T_io≈6ms（@21.5MB/s），
实测 8MB/s ⇒ 每突发周期 ≈16ms ⇒ 计算间隙 ≈10ms。**SD 大部分时间空转。**

### 2.2 同机制先例（代码内实测）
`qwen_engine_lmhead2.c` Stage2 注释：embed span **mmap 每 4KB 页 fault = 3.1MB/s**；改
**offset 升序 pread 整 span = 17–35MB/s**。层权重读（8 vs 21.5，~2.7×）是同一现象，只是
页尺寸/readahead 组合不同。

### 2.3 候选对照（`layer_read_bench` 8+1 种机制）
板上 `layer_read_bench` 对真实 layer0..23_kal.bin 冷读 24 层（201MB/pass）输出归因表：

| # | 机制 | 预期 |
|---|---|---|
| 1 | mmap_bare（Phase 6 基线） | ~8 MB/s（复现瓶颈） |
| 2 | mmap + madvise(SEQUENTIAL) | 待测 |
| 3 | mmap + madvise(WILLNEED) | 待测 |
| 4 | mmap + readahead 全层（主修复等价） | 目标 ~21.5 |
| 5 | mmap + fadvise(WILLNEED) | 待测 |
| 6 | pread 1MB 顺序入 buffer | ~21.5（proven） |
| 7 | read() 256KB（smollm2 路径） | ~21.5 |
| 8 | mmap + 引擎 tensor/K-block 顺序触页（无 TIU） | 分离「读机制」vs「访问序」 |
| 9 | pread O_DIRECT（绕过 page cache） | 对照 |

---

## 3. 实施（已就绪，`qwen_engine_lmhead2.c`）

### 3.1 改动（纯增量，默认行为不变）
1. `#define _GNU_SOURCE`（`readahead` 需要）。
2. 新增 `LayerIO` 抽象 + `layer_io_begin()/layer_io_end()`：封装 mmap / mmap+readahead / pread 三种
   层装载。`run_prompt`（prefill）与 `run_decode_step`（decode）两处 `open/mmap/close/munmap`
   替换为 begin/end。
3. `main()` 解析 `LW_READ` env；`pread` 模式分配一次复用 `g_layer_buf`（lsz=8393728B）。
4. `mmap_ra` 下 `readahead(fd,off,n)` 以 512KB 分片循环整层（规避单次 readahead 上限），
   `madvise(MADV_SEQUENTIAL)`；readahead 失败计数 `g_ra_err` 在 decode 摘要打印。

### 3.2 为何不动数值路径
三种模式喂给 `parse_layer` 的字节完全一致（同一文件）；所有 tensor 偏移 `%16==0`（已验证），
mmap 基址（页对齐）与 malloc 基址（16 对齐）下指针对齐相同；dequant/TIU/ION/LM head 均未改。
⇒ **bit-exact 由构造保持**，板上回归只需确认（bad1=bad2=r_opt=0、NEXT 3/3、min gap 与基线一致）。

### 3.3 内存/swap 红线
- `mmap_ra`：层字节留在 page cache（clean、可回收），**无新增匿名 buffer**，VmSwap 应维持
  ~2.6MB 稳态，不引入 swap 抖动。
- `pread`：+8.4MB 匿名（Phase 6 已证明此类压力风险大），故只作对照，不作为主修复。

### 3.4 独立复核（TPU 底层工程师，静态通过）
- **设计通过（静态）**：`mmap_ra` 主修复 / `pread` 仅对照的定位认可；三档字节同源 → bit-exact
  由构造保持（TPU 独立手算层布局 8,393,728B 逐字节吻合）。
- 采纳的非阻塞增强：
  1. **mincore 驻留探针**：`mmap_ra` 在 readahead 提交后、compute 前对每层做 `mincore`，
     累计驻留页占比并在 decode 摘要打印——把「readahead 生效 vs 部分预取/被回收」变成直接
     观测（`g_ra_err==0` 不足以证明整层驻留）。
  2. **run_clean 600s 超时风险**：原 `run_clean.sh` 全局 600s 可能在 pread 对照 swap 抖动时
     SIGKILL；已改为板上脚本内建 900s 每档 timeout（BusyBox 兜底），并前置一次
     `run_clean.sh --clean` 清 ION。
- 保留的复核点：板子重插跑完后由 TPU 独立核验归因表（mode4≈mode6 为修复生效核心证据）。

---

## 4. 上板验收（被硬件断开阻塞）

### 4.1 已就绪脚本（主机侧一键）
```bash
cd ~/Documents/MilkV_duo_project/tpu_bench/experiments/qwen_onboard
sh phase7_deploy_run_host.sh 10
# 构建 → push 二进制/脚本到 /data/qwen → 跑 phase7_readfix_run.sh → pull 4 份 log
```
板上 `phase7_readfix_run.sh 10` 依次执行：
1. `layer_read_bench` 9 机制归因表（reps=2, cold）；
2. `LW_READ=mmap   DECODE=1 DECODE_STEPS=10`（基线，内含 3-prompt 回归）；
3. `LW_READ=mmap_ra DECODE=1 DECODE_STEPS=10`（主修复）；
4. `LW_READ=pread  DECODE=1 DECODE_STEPS=10`（对照）。
每步 ≥5 步稳定区间；回归与 bit-exact 内嵌于每次引擎运行输出。

### 4.2 交付物清单（全部就绪）
| 文件 | 说明 |
|---|---|
| `qwen_engine_lmhead2_phase7` | riscv64 引擎（LW_READ 三档 + mincore 探针，VERIFY=1） |
| `layer_read_bench` | riscv64 读路径归因基准（9 机制） |
| `phase7_readfix_run.sh` | 板上 A/B 编排（900s/档 timeout + ION 清理前置） |
| `phase7_deploy_run_host.sh` | 主机部署/运行/pull 编排（2400s SSH 上限） |
| `phase7_analyze_logs.py` | log 解析 → 归因表 + decode 对比 + 红线回归表 |
| `qwen_engine_lmhead2.c` | 改动源码（增量，默认 mmap 不变） |
| `REPORT_PHASE7_READPATH_20260814.md` | 本报告 |

### 4.3 阻塞
Duo 板 USB 于 **2026-08-14 00:33:22** 被物理拔出（`journalctl -k`），本机 `lsusb` 无 Cvitek
设备、ping 192.168.42.1 不通、无串口。**需人工重插 USB**（CDCEther 驱动已加载，插上即
`enxe206575f3db8`），随后 `sh phase7_deploy_run_host.sh` 即可完成全部验收。

---

## 5. 预期结果与风险

- 预期：`mmap_ra` 将 t_layers 从 ~25.3s 收敛到 ~9.4–10s（SD 读地板 + 计算残差），
  decode ≈ **11–12 s/token**（与任务锚定一致）。
- 风险：若板上 `readahead()` 在 vendor 内核失效（`g_ra_err>0`），则 mmap_ra 自动回落
  mmap 速度，此时以 `pread`（模式 6，proven 21.5）为修复备选（需先确认 VmSwap 不抖动）。
- 风险：`--cold` 依赖 `/proc/sys/vm/drop_caches` 可写；不可写时归因表为半冷态（201MB 自然
  驱逐仍近似冷），对比仍有效。
