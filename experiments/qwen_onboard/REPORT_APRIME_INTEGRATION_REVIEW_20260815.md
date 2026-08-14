# A' 集成评审报告 + 回滚预案（Phase 8 最终版）

日期：2026-08-15 | 作者：推理引擎工程师（集成/验证协同）
评审对象：`PLAN_APRIME_IMPL_20260814.md` + 已落地实现 `qwen_engine_lmhead2.c` LW_ION_DB（2026-08-15 00:13 状态）
工作方式：**与 TPU 并行，未触碰引擎文件**；host 侧 A/B 平台独立交付。
关联：`REPORT_GATE_ION_20260814.md` / `REPORT_ION_DOUBLEBUF_FEASIBILITY_20260814.md` / `REPORT_PHASE7_SIGNOFF_20260814.md`

---

## Phase 8 最终签核（CEO 收口，2026-08-15）

**关闭决定：Phase 8 关闭。** A'（整层 ion_db 预取）与 B-2（per-matrix ion_db 预取）两型
**均未过 9.5s 验收线**，硬件天花板确认：

| 路径 | decode avg/token | 判定 |
|---|---|---|
| A'（整层 pread，smoke_iondb2.log 实测） | **11.72s**（10.73/12.71） | FAIL（+3.8% vs 基线，负优化） |
| B-2（per-matrix O_DIRECT，源码 02:37 落地） | **未过 9.5s** | FAIL（天花板边缘，见 PLAN_APRIME_MEMBUDGET §6） |
| **出货配置：mmap + gsc ION 基线** | **11.29s**（decode_e1_v0.log 锚点） | **锁定出货** |

**根因（闭环）**：SD 20.2MiB/s（实测 19.86–20.57）+ ION 28.1MB carveout 顶 →
nib-only 地板 8.33s + 重叠损耗 ~0.97s + LM head 0.5–1.4s ≈ **9.45s 理论地板**。
该地板需更快存储或更大 ION 才可达，**硬件现状不可达** → A'/B-2 结构性无法突破 9.5s。
优化预算已转移至 gsc 压缩 / SD 块读放大 / 更大存储介质（二期）。

**死锁结论（本报告 §2.2）**：`ion_prefetch_wait` 死锁在 **ion_db 实验路径**内，
**出货配置（mmap）不执行该代码路径，已排除出出货范围**。代码保留在
`qwen_engine_lmhead2.c`（LW_READ=ion_db 门控），若二期复用它需先做
§2.2 的最小修复（wait 循环失败 break + 统一走冷同步兜底）。

**回归工具链（已标注）**：`phase7f_aprime_run.sh`（板上三档 A/B）+ `phase7_analyze_logs.py`
（host 对比表/验收线/回归清单）已标注为 **Phase 8 回归工具链**。出货配置任何改动后
按以下流程复测：
```sh
sh phase7_deploy_run_host.sh aprime 6      # 板上三档: B 基线 mmap / A perf ion_db / A corr VERIFY=1
python3 phase7_analyze_logs.py             # 验收线 9.5s 自动判定 + 回归清单
```

**一键回滚（出货配置 = 引擎默认，零重建）**：见 §5。

---

## 0. TL;DR

1. **主循环集成整体正确（已下板实测通过）**：slot=l&1 切换点、warmup 层0 同步装载、跨步冷槽补载、
   LM head 衔接、gsc 旁路与 bit-exact 构造保持、fd/ION 生命周期，六项均评估通过。
2. **[HIGH] 性能预算不一致 — 已被实测证实，A' 不达验收线**：A' 每步实际 SD 读 = **整层文件 201MB**
   （nib+gsc+rms，实现 `lsz_global` 全量 pread），而计划 §1.1 / G-ION §5 的 6.9–9.1s 预估值按
   **nib-only 170.6MiB** 计算。**板上实测（2 步 decode，冷页缓存）：decode avg = 11.72s/token，
   相对 11.29s 基线 +3.8%，验收线 9.5s FAIL**（步骤 10.73/12.71s；sd 读 9.17/10.88s/步；
   TIU 计算地板 ~6.1s/步）。释放 gsc ION 缓存使每步多读 30.4MB 为结构性代价，A' 净效应为负。
3. **[HIGH] 预读失败死锁**：`ion_prefetch_wait` 在预读线程报错（`g_sd_slot_layer[slot]=-1`）时
   永久阻塞；`layer_io_begin` 中"预读失败兜底 sync 补载"是 **dead code**。需先修再长跑。
4. **次要**：prefetch_n 期望应为 `23×N`（L-1/步）而非 `24×N`；A' metrics 为累计值；
   跨步冷槽 0.4s/步可用 wrap-around 预取消除（二期）。
5. **host 侧 A/B 平台已交付并实测自检通过**：`phase7_deploy_run_host.sh aprime` +
   `phase7f_aprime_run.sh` + `phase7_analyze_logs.py`（对比表 / 验收线 9.5s 自动判定 / 回归清单）。
   分析器已用真实 ion_db log 验证：正确识别 11.72s → 验收线 FAIL，回归项全绿。

---

## 1. 主循环集成逐项评估

### 1.1 slot = l&1 切换点 — 正确（可执行）

`layer_io_begin` LW_ION_DB 分支（当前 L905-921）：

- `slot = l & 1` ping-pong，`if (l + 1 < L) ion_prefetch_issue(l + 1, slot ^ 1)` 预读下一层到对侧槽。
- 实现比计划骨架**更稳健**：新增 `g_ion_expected[slot]`（主线程期望层）区分"流水线等待"与
  "冷槽同步补载"，避免计划骨架用 `g_sd_slot_layer[slot] != l` 判断时，在预读未完成瞬间把
  "wait 流水线"误判为"冷槽 sync"导致重复装载/竞写。
- fd 处理：L909 在 ion_db 分支立即 `close(lfd)`，无 fd 泄漏（计划骨架未显示 close，落地已修正）。

### 1.2 warmup 层0 同步装载 — 正确（可执行）

main() L1631-1652：

- 顺序正确：`lsz_global = lsz` → `pthread_create(ion_prefetch_thread)` → `ion_prefetch_sync(0,0)`。
- 防御完整：`lsz > SD_BUF_SZ` 运行时复核（L1632）、SD_BUF ION alloc 失败显式 return 2（L1640）。
- 启动打印 carveout debugfs（L1651）供 host 侧复核 ION 布局余量。
- warmup 与 P1 层0 的衔接：warmup 后 `g_ion_expected[0]==0`，P1 层0 走 `l>0` 跳过等待、
  `slot_layer[0]==0` 跳过重载，直接 issue(1→slot1)，无重复装载。✓

### 1.3 跨步冷槽补载气泡 — 正确（0.4s/步），建议二期 wrap-around

- 每 decode 步结构：**层0 必为冷 sync（~0.4s 串行）**，层1-23 走预读流水线。总气泡 ≈0.4s/步，
  与计划 §2.2 "~0.4s/步" 一致（实现用 `g_ion_expected` 避免了计划骨架中"层1 也可能误 sync"，
  实际为 1 次 sync/步，非 2 次）。
- **可选优化（计划 §6 二期 "跨步 circular prefetch"）**：层 L-1（l=23）结束时
  `ion_prefetch_issue(0, slot^1)`，用 LM head ~0.5s 隐藏层0 预读，消除每步 0.4s 串行气泡。
  若验收线 9.5s 边缘紧张，建议优先落地（低风险，只改 issue 条件 + wrap 边界 expected 处理）。

### 1.4 与 LM head 路径衔接 — 正确（可执行）

- LM head（L1409-1424）在层循环之后，两段式读 frms + embed 簇，与层权重读路径解耦。
- 层循环结束（l=23 无后继 issue）→ 预读线程自然 idle，不与 LM head 争 SD/ION。✓
- LM head 的 embed SD 读（~0.44s/步）不被双缓冲隐藏，已含在"计算地板 4.6s"预算。✓

### 1.5 gsc 旁路与 bit-exact — 正确（构造保持）

- L1534-1535：`LW_READ=ion_db` 跳过 `gsc_ion_load`（打印 "gsc ION cache skipped"）。
- `g_gsc_ion=NULL` → `gsc_ion_apply()` L711 首行 no-op；`parse_layer` 的 gsc 指针直指 SD_BUF
  同偏移字节（文件布局不变）→ 数值/TIU/LM head 路径字节级不变，bit-exact 由构造保持。✓

### 1.6 fd / ION 生命周期 — 正确

- 清理 L1715-1724：`g_ion_shutdown=1` + broadcast + join 预读线程 + `CVI_RT_MemFree` SD_BUF_A/B
  + `free(g_bounce)`。ION 无泄漏路径完整。

---

## 2. 关键风险

### 2.1 [HIGH] 性能预算不一致：A' 每步 SD 读 201MB（整层），非计划 170.6MiB（nib-only）

**证据链（已含 2026-08-15 板上实测）：**

| 项 | 值 | 来源 |
|---|---|---|
| 层文件 | 8,393,728 B（nib 7,454,720 + gsc 931,840 + rms 7,168） | 板上 stat / 计划 §1.3 |
| 每步 SD 读（A'） | 24 × 8.39MB = **201.4MB 全量**（实现 `lsz_global` 全文件 pread） | `ion_prefetch_sync` / thread L853 |
| 每步 SD 读（基线 7e） | **170.6MiB nib-only**（gsc 已 ION 缓存，page cache 命中） | 计划 §1.1 / G-ION §5 |
| 实测 SD 顺序读 | ~20.0–21.3 MiB/s（cat 19.86-20.07；pread_1m 20.34） | conn check / layer_read_bench |
| **A' decode avg（实测）** | **11.72s/token（2 步；step1=10.73s, step2=12.71s）** | `smoke_iondb2.log` |
| A' sd 读/步（实测） | 9.167s / 10.883s（A' metrics sd 累计差） | 同上 |
| A' TIU 计算地板 | eng_matmul wall 12.165s/2步 ≈ **6.08s/步**（runcmdbuf 57% + dequant 15% + DMA 9% + accum 11%） | 同上 PROFILE |

**实测结论（2026-08-15 02:xx，smoke_iondb2，DECODE_STEPS=2）：**

```
A' decode avg 11.72s/token  vs  基线 11.29s  =>  +3.8%  FAIL(≥9.5 验收线)
  step1 total=10.73s  sd=9.167s  t_layers=10.14s  t_head=0.60s
  step2 total=12.71s  sd=10.883s t_layers=12.12s  t_head=0.59s
  TIU 计算地板 ≈6.1s/步 → SD 读 (~10s) 为绝对主导, 流水线已被预读线程拉满
  正确性: NEXT 3/3 OK, decode bit-exact 0/0/0/0, VmSwap 稳态, prefetch_n=23×N err=0
```

**结论：A' 现实 = 11.72s/token（实测），相对 11.29s 基线 +3.8%，验收线 9.5s 明确 FAIL**。
计划 §1.1 的 6.9–9.1s 依据的字节数（170.6MiB）在释放 gsc 缓存后不再成立；且 A' 每步多读
~30.4MB（gsc）为结构性代价，净效应为**负优化**（-0.43s/token）。

**建议：**
1. **实测已裁决：A' 不达验收线，且为负优化（11.72s vs 11.29s）**。建议与 CEO 对齐：
   - 方向 A（推荐，恢复 7e 基线路径为生产）：**放弃 ion_db 整层预取，回到 mmap + gsc ION 缓存
     路径（11.29s）**；把优化预算转向 gsc 压缩 / SD 块读放大等，目标是让"每步只读 nib"成立。
   - 方向 B（若仍要双缓冲）：**gsc 压缩（INT8 gsc 或 per-K 共享 scale）**——把 gsc 21.33MiB
     减半/减至 ~10MB，腾出 ION 空间同时装 SD_BUF（nib-only 2×7.46MB）+ gsc ION 缓存，恢复
     "每步只读 nib 170.6MiB"路径 → SD 预算回到 8.4s/步，A' 才有正收益可能。
   - 方向 C：接受 11.72s 为 A' 收口（相对基线 +3.8%，无收益），不推荐。
2. **wrap-around 预取**（1.3 节）在 ion_db 保留期内仍建议做（白拿 ~0.4s/步），但不足以扭转
   验收线结论。
3. **修 2.2 死锁**仍必须做——ion_db 保留期间防长跑挂死；即便切回 mmap，代码路径保留待复用。

### 2.2 [HIGH] 预读失败死锁：`ion_prefetch_wait` 永久阻塞

`ion_prefetch_wait`（L884-891）：
```c
while (!g_ion_done[slot] || g_sd_slot_layer[slot] != l)
    pthread_cond_wait(&g_ion_cv, &g_ion_mtx);
```
预读线程 pread 失败时（L864）置 `g_sd_slot_layer[slot] = -1` 再置 `g_ion_done[slot]=1`。
此时 `!done`=false、`slot_layer(-1) != l`=true → 条件恒真 → **主线程永久阻塞**。
`layer_io_begin` 中 L913-914 的"预读失败兜底 sync 补载"在 wait 之后，**永远不可达（dead code）**。

触发概率低（SD pread 正常不报错），但单次 SD 抖动即挂死整步，且该兜底分支本意就是处理此
场景——属实现缺陷。**建议先修**（小改）：
```c
static void ion_prefetch_wait(int slot, int l) {
    double tw = now();
    pthread_mutex_lock(&g_ion_mtx);
    while (!g_ion_done[slot]) pthread_cond_wait(&g_ion_cv, &g_ion_mtx);
    pthread_mutex_unlock(&g_ion_mtx);
    g_t_sd_wait += now() - tw;
}
```
返回后由调用方 `if (g_sd_slot_layer[slot] != l) ion_prefetch_sync(l, slot)` 统一走兜底。

---

## 3. 次要发现

| # | 项 | 说明 |
|---|---|---|
| 3.1 | **prefetch_n 期望 = 23×N**（非 24×N） | `if (l+1 < L)` 每步只 issue L-1=23 次（层 L-1 无后继）；层0 冷 sync 不计数。任务/计划口径"24×steps"需对齐为 23×steps。host 分析器已按 23×N 判定。 |
| 3.2 | **A' metrics 为累计值** | `A' metrics:` 每步打印，计数器仅 profile_reset（prefill 后）清零 → 每步行是累计。分析器取最后一行。若要每步独立值需在 run_decode_step 内做增量（非必须）。 |
| 3.3 | **g_sd_slot_layer 无锁读** | 主线程 L911 `g_ion_expected[slot]` 无锁读 volatile；线程写 `g_sd_slot_layer`（L865 未持锁）。volatile+后续 mutex wait 保证可见性（当前正确），严格属 data-race 语义，嵌入式可接受。 |
| 3.4 | **sd_wait 语义** | SD-bound（SD 9.9s > 计算 4.1s）时 sd_wait≈SD-重叠计算 属正常，非"预读没追上"。host 分析器已改为 sanity 判定（非"必须小"红线）。 |

---

## 4. host 侧 A/B 平台交付（本次）

| 文件 | 作用 |
|---|---|
| `phase7_deploy_run_host.sh`（升级） | 新增 `aprime` 模式：构建 `qwen_engine_lmhead2_aprime` → push → 跑 A/B → pull 4 份 log。legacy readfix 模式保留。 |
| `phase7f_aprime_run.sh`（新） | 板上 3 档：B 基线 mmap(GSC_ION=1) / A perf ion_db(V=0) / A corr ion_db(V=1)；每档 drop_caches + run_clean；跑后存 ion_after.log。 |
| `phase7_analyze_logs.py`（升级） | ion_db vs mmap 对比表（锚点 11.29s + 本轮 B + A perf + A corr）、验收线 9.5s 自动判定、回归清单（NEXT 3/3 / bad1=bad2=r_opt=rsh=0 / min_gap≥0.05 / VmSwap 稳态 / prefetch_n=23×N err=0 / sd_wait sanity / ION 无泄漏）。 |

**用法**：`sh phase7_deploy_run_host.sh aprime 6` → 完成后 `python3 phase7_analyze_logs.py`。

---

## 5. 回滚预案（ion_db 不达标 → 一键切回 11.29s 基线）

A'/B-2 均为**新增实验模式**（`LW_READ=ion_db`），引擎默认仍走 `LW_READ=mmap GSC_ION=1`
（7e 基线，出货配置）。因此回滚零重建、秒级生效：

**方案 A — 同二进制环境变量切换（推荐，最快）**
```sh
sh /data/qwen/run_clean.sh --clean qwen_engine_lmhead2_aprime   # 释放 A' 的 ION SD_BUF
cd /data/qwen && LW_READ=mmap GSC_ION=1 VERIFY=0 RSH=1 DECODE=1 DECODE_STEPS=6 PROFILE=1 \
  ./qwen_engine_lmhead2_aprime | tee decode_rollback_mmap.log
```
引擎默认分支 mmap 未被 A' 改动 → 应复现 11.29s。

**方案 B — 已知良好二进制（最稳）**
板上已有 `qwen_engine_lmhead2_phase7e`（7e 生产二进制，产出 11.29s 锚点）：
```sh
sh /data/qwen/run_clean.sh --clean qwen_engine_lmhead2_aprime
cd /data/qwen && GSC_ION=1 VERIFY=0 RSH=1 DECODE=1 DECODE_STEPS=6 PROFILE=1 \
  ./qwen_engine_lmhead2_phase7e | tee decode_rollback_mmap.log
```
无需 host push、无需重编。

**回归确认**：`python3 phase7_analyze_logs.py` 检查 rollback log：avg≈11.29±0.5、NEXT 3/3、
bit-exact 4 项=0。若 7e 二进制也不达标 → 环境异常（SD 速率/ION/供电），先跑
`aprime_conn_check.sh` 排查。

**切换纪律**：任何模式切换前先 `run_clean.sh --clean <binary>`，避免 A' 的 16.8MB SD_BUF ION
残留污染基线；每档前 drop_caches 保证冷启动公平对比。

---

## 6. 结论与建议（最终版）

1. **集成评审通过 + 实测通过（正确性）**：主循环六项正确；板上实测 NEXT 3/3 OK、
   decode bit-exact 0/0/0/0、ION 无泄漏、VmSwap 稳态。
2. **实测裁决：A'（ion_db 整层预取）不达验收线，且为负优化**——decode avg **11.72s** vs 基线
   11.29s（+3.8%），验收线 9.5s **FAIL**。根因即 §2.1：释放 gsc ION 缓存后每步 SD 读 201.4MB
   （vs 基线 nib-only 170.6MiB），SD ~10s/步为绝对主导。B-2（per-matrix 全量 gsc 缓存）虽将
   SD 流量收敛回 nib-only 179.17MB，但 **9.5s PASS 线位于物理极限边缘（地板 8.33s + 重叠损耗
   ~0.97s + LM head ≈ 9.45s），实测仍过不了** —— 硬件天花板确认。
3. **Phase 8 关闭，出货配置锁定 mmap + gsc ION 基线（11.29s/token）**，为引擎默认
   （`g_lw_mode=LW_MMAP` + `GSC_ION` 默认开），一键回滚零重建（§5）。
4. **死锁（§2.2）排除出出货范围**：ion_db 实验路径已收口且不达验收线，出货配置（mmap）
   不执行预读代码；代码保留于 `LW_READ=ion_db` 门控下，二期复用前需先做最小修复。
5. host A/B 平台已标注为 **Phase 8 回归工具链**（对比表 + 验收线 + 回归清单 + 回滚），已用
   真实 ion_db log 自检通过（正确识别 FAIL + 回归全绿），供出货配置后续改动回归复用。

---

## 附：实测记录（2026-08-15）

`smoke_iondb2.log`（LW_READ=ion_db VERIFY=0 RSH=1 DECODE=1 DECODE_STEPS=2 PROFILE=1）：

- 预填回归：PROMPT 1/2/3 expected_next=2130/12095/99366 **3/3 OK**；TIU internal **BIT-EXACT**。
- decode（M=1）：avg **11.72s**/token；step1=10.73s（sd=9.17s, t_head=0.60s）、step2=12.71s
  （sd=10.88s, t_head=0.59s）；A' metrics prefetch_n=46/2步（=23×N ✓）err=0；
  eng_matmul wall=12.165s/2步（TIU runcmdbuf 6.95s=57%、dequant_rvv 1.77s=15%、DMA 1.12s=9%、
  accum 1.31s=11%）→ 计算地板 ≈6.1s/步。
- ION：carveout 28,102,656B，run 中 used 18,759,680B（67%），peak 25,300,992B < carveout ✓；
  run 退出后 used=0（分析器 PASS）。

---

## 附：A/B 平台自测

- **真实 ion_db log**（`smoke_iondb2.log` → decode_a_iondb.log）：avg 11.72s → 验收线 **FAIL**
  （正确判定）、NEXT 3/3 ✓、bit-exact 0/0/0/0 ✓、VmSwap 稳态 ✓、prefetch_n=23×N ✓ err=0、
  ION 无泄漏 PASS。分析器对真实 log 端到端可用。
- 合成 A perf avg=9.80s → 验收线 **FAIL**（正确识别 SD-bound 边缘场景）。
- 合成 A corr avg=10.00s，回归清单：NEXT 3/3 ✓、bit-exact 4 项 ✓、min_gap ✓、VmSwap ✓、
  prefetch_n=92=23×4 ✓ err=0、sd_wait sanity ✓（隐含 SD-bound floor≈9.11s/step）、ION 无泄漏 ✓。
- 分析器对 legacy phase7 log（mmap/mmap_ra/pread）向后兼容（bit-exact 无 rsh 字段时忽略）。
