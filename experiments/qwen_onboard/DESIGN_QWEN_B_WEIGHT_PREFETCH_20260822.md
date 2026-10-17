# Qwen B-design：权重驻留/预取（pass1 冷权重加载消除）

日期：2026-08-22 | 作者：推理引擎工程师 | 状态：**设计交付物（待 CEO 设计评审与实现门控）**
定位：承接 A 系列关闭（A1/A2 shelve）后的战略转向——攻击 decode 最大剩余瓶颈 **pass1 冷权重加载**。
关联：`REPORT_A2_CLEAN_VERIFY_20260822.md`（根因实验）/ `DESIGN_QWEN_INT4_B_ION_20260822.md`（TPU 支线，B2 nib16 已落地）/ `qwen_engine_lmhead2.c` eng_matmul/读路径
基线（B2 出货配置，VERIFY=0，CHAT 35-tok DECODE_STEPS=6）：decode **11.15s/token**，eng_matmul wall 9.97s/token。

---

## 0. 结论摘要

1. **瓶颈已定位（A2 实验证伪"双解包"后留下的事实）**：decode eng_matmul wall 9.97s/token 中，dequant 计时 5.83s（58.5%），其中 **pass1 冷权重加载 ~5.74s/token（dequant 的 98.5%），pass2 热解包仅 ~0.09s**。dequant_rvv 计时本质上 = **权重加载（mmap 页故障/SD 读 stall）+ 解包计算**，加载占绝对主导。
2. **现状**：B2 nib16 vle8 已把单调用解包提速至 538-753MB/s（热），gsc 已 24 层 ION 预热、rms 已全量 pread——**唯一剩余冷读 = nibble 权重（7,454,720 B/层，~178MB 总）**。
3. **设计方向（三个递进杠杆）**：
   - **B-P0（推荐先做，零 ION/anon 预算）**：`readahead()`/`posix_fadvise` 提前预取下一批层 nibble + `DONTNEED` 释放已消费层 → 把 pass1 的同步页故障变成重叠的顺序 SD 读。
   - **B-P1（DDR 层双缓冲）**：显式 pread 下一层 nibble 到 DDR anon 双缓冲 → 预算 ~16.8MB，与 28MB anon 紧张，需先证 P0 收益。
   - **B-P2（ION 部分驻留）**：原 B1，carveout 余量仅 1.85MiB，仅容 up/gate/down 之一，收益边际；依赖 TPU 支线 ION notes。
4. **实现门控前置（低风险实证）**：先上 B-P0 原型（读路径提前量 + fadvise，~30 行），板验 `majflt/step` 与 decode avg。若 majflt 显著下降且 decode 改善 → 立项完整 B-P1；若 majflt 已低但仍慢 → 证明 SD 带宽是硬地板（此时只能 ION 驻留，B-P2）。**不空谈方案，先用 5 分钟测量定方向。**

---

## 1. 根因量化（A2 实验直接产出）

### 1.1 dequant 计时 = 加载 + 解包

A2（pass2 解包→memcpy 复用）板验的意外结果，精确定量了 pass1/pass2 不对称：

| 项 | 值 | 说明 |
|---|---|---|
| decode dequant_rvv（B2） | 34.993s / 6 tok = **5.83s/token** | eng_matmul wall 58.5% |
| A2 移除 pass2 dequant 后节省 | 0.54s / 6 tok = **0.09s/token** | pass2 重读同一 nib（pass1 刚触热）→ 缓存驻留 |
| pass1 冷权重加载（= dequant 余项） | **~5.74s/token** | dequant 总量的 98.5% |
| 解包热速率（B2 内核，单调用） | 538-753 MB/s | 已远高于加载 |
| decode 有效解包速率 | 253MB÷5.83s ≈ **43MB/s** | 仅热速率的 ~8% → ~92% 时间在 stall |

> 253MB/token = 各矩阵 nibble 触摸量（q 19.3 + k/v 各 2.8 + wo 19.3 + up 52.3 + gate 52.3 + down 104.6 MB，见 §1.2）。pass1 冷加载 ~22MB/s 有效——与 SD/mmap 页故障 stall 一致。

### 1.2 每 token 权重触摸量（decode M=1）

| 矩阵 | nibble 触摸/层（2 passes 或 merged 单） | ×24 层 |
|---|---|---|
| q / wo | 0.80MB ×2 | 各 19.3MB |
| k / v | 0.115MB ×2 | 各 2.8MB |
| up / gate（merged 单） | 2.18MB ×1 | 各 52.3MB |
| down（5 分块） | 4.36MB ×2 | 104.6MB |
| **合计** | | **~253MB/token** |

### 1.3 读路径现状（已有资产）

- gsc：24 层 ION 预热（21.3MiB，`GSC_ION=1`）+ B-2 混合 DDR mmap + `madvise(MADV_WILLNEED)` 预热 → **decode 期 0 页错误**。
- rms：全量 pread 到 `g_rms_all`（L*2*D floats）→ 0 冷读。
- nibble：`mmap` 裸读（mmap bare 模式，Phase 6 基线）→ **懒加载，pass1 首次触摸页故障**。SD_BUF 双槽预取机制存在（ION 4.16MB），但 mmap 模式未用于 nibble 全量。

---

## 2. 设计空间

### B-P0（推荐先做）：mmap 顺序预取 + 释放（零预算，~30 行）

```
在解码层循环（layer L 计算开始处）：
  readahead(fd, off(L+Δ), len(L+Δ))    /* 提前 Δ 层预取 nibble 入 page cache */
  posix_fadvise(fd, off(L-K), len(L-K), POSIX_FADV_DONTNEED)  /* 释放已消费层 */
```

- **机制**：页故障由"同步 stall"变"异步顺序读"。SD 顺序读比随机页故障吞吐高（~2-3MB/s→可能更高）。
- **Δ 提前量**：按 decode per-layer 计算窗口调优（11.15s/24层 ≈ 0.46s/layer；8.4MB/layer 在 ~3MB/s 下需 ~2.8s → 提前量 ≥ 6 层才够完全隐藏）。
- **预算**：0 新增 ION / 0 anon（page cache 可回收）；仅需保持 layer fd 打开的 offset 管理。
- **风险**：DONTNEED 过早释放会破坏 pass2 热复用（pass2 需同层 nib）→ 释放窗口必须 ≥ 当前层之后。
- **预期**：若 majflt/step 从 ~1500-2000 显著下降，pass1 冷加载 5.74s → 接近热解包 0.5-1s，**decode 有望 11.15 → ~6-8s/token**（受 SD 带宽上限约束）。

### B-P1（完整方案）：DDR 层双缓冲（显式 pread）

- 与 P0 互补：P0 依赖 page cache 调度，P1 用 `pread` 显式填 DDR anon 双缓冲，dequant 从固定地址读。
- 预算：2 × 8.39MB = **16.8MB DDR anon**。28MB anon 已含引擎数据结构 + swap，**超预算风险高** → 需 P0 实证后按 1 层缓冲（8.4MB）或半层起步。
- 收益上限与 P0 同源（隐藏 SD 读），P0 先行则 P1 边际小，除非 P0 证明页缓存调度不可控。

### B-P2（ION 部分驻留，依赖 TPU 支线）

- 原 `DESIGN_QWEN_INT4_B_ION` §2.2 B1：carveout 余量仅 1.85MiB，DQ/SD_BUF 已 4.32MB。
- 可行子集：up/gate/down 之一（各 2.18MB nib）——单个也超余量，需 A 系 pool 回收后评估。
- **结论**：P2 为 P0/P1 之外的可选扩展；收益受 SD 带宽硬地板约束，只有 P0 实证显示"缓存命中足够但 SD 顺序读是最后瓶颈"时才立项。等待 TPU 底层工程师 ION 可行性 notes 汇入后评审。

---

## 3. 内存/约束预算

| 方案 | ION | DDR anon | page cache | 风险 |
|---|---|---|---|---|
| B-P0 | 0 | 0 | ~8-16MB（可回收） | 释放窗口误伤 pass2 热复用 |
| B-P1 | 0 | 16.8MB（双缓冲） | 0 | 超 28MB anon / 换页顶爆 |
| B-P2 | ~2.2MB（需回收） | 0 | 0 | carveout 余量不足 |

上板实现前自查沿用既有约定：ps 无 qwen/smollm 残留 + ION used=0。

---

## 4. 实现门控：先做 5 分钟实证（B-P0 原型）

**不直接立项完整方案。** 第一步上板原型（~30 行，只改读路径，不动 matmul/dequant 逻辑）：

1. 板验档（VERIFY=0，CHAT 35-tok DECODE_STEPS=6，同 B2 口径）：
   - 指标：`majflt/step`（现状 ~1500-2000）、decode avg（基线 11.15s）、PROFILE dequant_rvv 占比。
2. 三档提前量 Δ = {0（基线）, 6, 12} × DONTNEED 开/关，各跑 1 轮。
3. 判定：
   - **Δ 增大 → majflt 显著↓ 且 decode avg ↓** → 立项 B-P1 完整化（预算按实测调）。
   - **majflt 已低（<300）但 decode 仍慢** → 证明 stall 非页故障，转查 SD 顺序带宽/缓存命中率 → B-P2 ION 驻留。
   - **无变化** → 读路径非瓶颈（与 58.5% dequant 计时矛盾时复查计时归属），回退基线。

**验收口径（立项后全量）**：
1. 正确性：3-prompt NEXT 3/3 + decode 流 + CHAT 75-token，bit-exact 0/0/0 —— 与 B2 基线一致（读路径改动不改变 DQ 字节，构造可保）。
2. 性能：decode avg **< 11.15s（B2 基线）**，目标 -20%+；majflt/step 显著下降；PROFILE dequant_rvv 占比 58% → ~30%。
3. 内存：carveout used 不超基线；anon 增量 = 0（P0）或按实测（P1）。
4. 回归：`qwen_engine_int8kv`（master 9ae95c5）完整跑通，无 watchdog。

---

## 5. 与支线/资产关系

- **TPU 底层工程师**（agt_336c00b3abbaa226）：正在产出 ION 可行性 notes（进行中）。B-P2 立项依赖其产出；P0/P1 与其无重叠（只碰 DDR/page-cache 读路径，不碰 ION 布局）。
- **B2 nib16 vle8（46eeb78）已落地**：P0/P1 建立在现有 nib16 布局上，内核零改动。
- **gsc ION 预热 + rms pread（已上线）**：P0 复用同一 `layer{l}_kal.bin` fd 的 offset 管理，不改文件格式。

---

## 6. 落地顺序（建议）

1. **P0 实证**（我实现，~30 行读路径）→ 板验 3 档 Δ → 数据回报 CEO。
2. 依据实测：P1 立项（预算按实测量身）或 P2 立项（等 TPU ION notes 汇入）。
3. 与 TPU 支线合并设计评审后，CEO 做实现门控裁决。

---

## 7. 复现基线

```
B2 出货配置（master 9ae95c5，46eeb78 源）：
  decode avg           : 11.15s/token（CHAT 35-tok, DECODE_STEPS=6, VERIFY=0）
  eng_matmul wall      : 9.97s/token（1584 calls）
  dequant_rvv          : 5.83s/token（58.5%）= pass1 冷加载 ~5.74s + pass2 热 ~0.09s
  runcmdbuf(TIU)       : 2.80s/token（0.182ms/call）
  majflt/step          : ~1500-2000（mmap 裸读）
  nibble/层            : 7,454,720 B（nib16[KG,16,N]）
  板载输入             : /data/qwen/input_tokens_35.bin（35-tok CHAT）
```
