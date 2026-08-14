# Path A 两遍法推理引擎设计规格（Qwen2.5-0.5B / CV1800B）

日期：2026-08-13
作者：推理引擎工程师
状态：**host 双验证通过（Python 3/3 + C 参考 3/3），等待 TPU 工程师闸口① pass1 int8 回读确认后上板**
依赖：convert_qwen_kal.py 产物（weights_kal/，K-aligned INT4 G32 + fp16 group scale）
C 参考：`qwen_kal_ref.c`（host x86 编译，读 weights_kal/，float32 全链）→ NEXT 2130/12095/99366（gap 1.53/1.17/0.02）
重要修复：rope 预计算 stride 必须为 `pos*(HD/2)`（非 `pos*HD`），否则 pos≥1 读到未初始化内存
→ 上板引擎实现时务必沿用此 layout。

---

## 0. 核心理念

- 权重：**INT4 G32 K-aligned**（weights_kal/layerN_kal.bin，8.39MB/layer，201.4MB 总量）。
  每个 K-block（32 行）× 输出列 (g,n) 一个 fp16 scale → 每 chunk matmul 可独立反量化。
- matmul：**per-chunk KG=32 两遍法**，ps32-free：
  - pass1：rshift=rsafe（matmul_rshift_w(32,wmax)-3）→ int8 输出 → CPU 读真实 per-chunk max
  - pass2：rshift=r_opt=ceil(log2(est/127)) → int8 满幅输出 → CPU fp32 累加
  - 反量化累加：out[m,n] += pass2[m,n] × 2^r × gscale[g,n]，最后 × sc_row[m]
- 激活：**per-row（per-token）INT8 scale**（公共项，host 已验证）。

## 1. 每层计算序列（引擎 matmul 顺序）

q_proj(896→896) → k_proj(896→128) → v_proj(896→128) → [GQA+RoPE] →
wo_proj(896→896) → [残差] → up_proj(896→4864) → gate_proj(896→4864) → [SiLU] →
down_proj(4864→896) → [残差]

每矩阵流程（以 up 为例，[K=896, N=4864]）：
```
for g in range(0, K, 32):                      # 28 个 K-block
    CPU: 反量化 block g（INT4→INT8，32×4864）
    for nt in range(0, N, 512):                # 10 个 N-tile
        TIU pass1: [M,32]×[32,512] rshift=rsafe → int8 [M,512] → DDR
        CPU: max|pass1| → r_opt（每 block 一个值，跨 nt 共享）
        TIU pass2: 同 LMEM，rshift=r_opt → int8 [M,512] → DDR
        CPU: out[m,n] += pass2 × 2^r × gscale[g,n]
out *= sc_row[:,None]
```

**关键点**：
- r_opt 是 per-K-block 标量（对 block 内所有输出列共享），故 pass2 单次提交即可。
  跨 nt 复用同一 r_opt（pass1 的 N-tile 是同一 K-block 的子集，max 语义一致）。
- LMEM 权重块在 pass1/pass2 间**不重新加载**（同一块，仅 rshift_bits 不同）。
- down_proj K=4864 → 152 个 block；N=896 → 2 个 nt。

## 2. TIU 提交量估算（decode，M=1，N-tile=512）

| matmul | K-blocks | N-tiles | tiles×2pass |
|---|---|---|---|
| q  896×896   | 28 | 2 | 112 |
| k  896×128   | 28 | 1 | 56 |
| v  896×128   | 28 | 1 | 56 |
| wo 896×896   | 28 | 2 | 112 |
| up 896×4864  | 28 | 10 | 560 |
| gate 896×4864| 28 | 10 | 560 |
| down 4864×896| 152| 2 | 608 |
| **合计/layer** | | | **2064** |
| **×24 layers** | | | **≈49,500/token** |

- 与 CEO 估的 ~640/layer 有出入：多两遍法（×2）+ N-tile 细化 + down 的 152 block。
- 若 TIU 单次 ~2μs：~100ms/token（TIU 占用）。**SD 读（201MB）约 6.9s/token 为主**，
  提交量可被 SD 读掩盖（闸口②可由此判断）。
- 优化选项（二期）：q/wo 等 K=896 且权重为 per-ch 时可用 KG=128 降提交（Path B 已验证
  3/3），但 Path A 组结构锁死 KG=32。

## 3. CPU 工作量（C906B @1GHz）

- **INT4→INT8 反量化**：358M 值/layer·token × 24 = 8.6G 值/token。
  每值 ~2-3 op（解 nibble + ×gscale）→ ~20-25G op/token ≈ **2-3s/token**（顺序）。
  → 必须与 SD 读 / TIU 重叠（见 §5），否则无法隐藏。
- **fp32 累加**（pass2×2^r×gscale）：~2.3M FMA/matmul·layer 计 ~22.9M/token ≈ 30-45ms。
- 决策：**反量化是 CPU 关键路径**，需与 SD 读并行的双缓冲；若 CPU 成瓶颈，
  降级选项：SD 直接存 per-group INT8（16.8MB/layer，去反量化）换带宽。

## 4. ION 24MB 布局（per-matrix 流水线粒度）

| 区域 | 大小 | 说明 |
|---|---|---|
| SD_BUF_A / B | 2×2.45MB = 4.9MB | 最大矩阵（up/gate/down）INT4+gsc 双缓冲 |
| DQ_BUF_A / B | 2×156KB = 312KB | 单 K-block 反量化输出（32×4864 int8）双缓冲 |
| P1/P2 输出 | 2×48.6KB | pass1/pass2 的 [M,N] int8（M≤10，N≤4864） |
| 激活/累加 | ~1.5MB | x/h/attn/qkv/mid/out fp32 [M≤10, D/N] |
| LMEM | 32KB | TIU 工作缓冲（[M,32]+[32,512]+[M,512]） |
| **合计** | **≈7.2MB** | **< 24MB，余量充足** |

- **关键**：权重按 per-matrix 双缓冲（非 per-layer），ION 占用从 16.8MB 降至 4.9MB。
- 每矩阵 SD 读 = 顺序 2.45MB 块（layer 文件内矩阵连续），SD 顺序读效率保持。

## 5. 流水线重叠（3 级）

```
SD 读(matrix i+1)  ══╗
CPU 反量化(matrix i, block g+1)  ══╗
TIU 两遍 matmul(matrix i, block g)  ══╗
```
- 粒度：**block 级**（32×N）。CPU 反量化 block g+1 时，TIU 正算 block g 两遍。
- SD 读与 CPU/TIU 全并行（SD→DDR DMA 独立）。
- 目标：每矩阵耗时 ≈ max(SD读, CPU反量化, TIU) 而非三者之和。
  - SD 读 up = 2.45MB / 29MB/s ≈ 85ms；CPU 反量化 up = 8.6G/24layer/7mat ×… ≈ 100ms/layer；
    TIU up = 560×2μs = 1.1ms。→ **CPU 反量化或成新瓶颈**，需实测 C906B 吞吐。

## 6. LM Head（flag，未定方案）

- embed [V=151936, D=896] int8 = 136MB >> ION。LM head = h[-1] @ embed^T，每 token 需
  从 SD 流式读 136MB ≈ **4.7s/token**（29MB/s），并做 [1,896]×[896,151936] matmul
  （28 blocks × 297 nt × 2pass ≈ 16,632 TIU 提交）。
- 这使 decode 预算增至 ~11.6s/token（SD 201MB 权重 + 136MB LM head）。
- 候选优化（另立任务）：① top-k 两段式（先投影到聚类中心选 top-k，再全展开）；
  ② embed 按频度缓存热区进 ION；③ 接受 4.7s 并重叠。
- **与 SmolLM2 同样结构（LM head 流式），但 Qwen vocab 3.1x、D 1.56x → 流量约 4.9x。**

## 7. 上板闸口依赖

- 闸口①（TPU）：pass1 标准 matrix_multiplication int8 输出可回读 DDR（读真实 max）。
- 闸口②：49,500 TIU 提交/layer 延迟能否被 SD 读掩盖（§5 估算）。
- 闸口③：ION 布局 §4（7.2MB）与 dequant 缓冲共存无冲突。
- 闸口④：无新增 bmk1822 硬限制（per-call rshift_bits 变更是标准参数）。

## 8. 已交付产物

- convert_qwen_kal.py + weights_kal/（K-aligned INT4 G32 + fp16 gsc，201.4MB）
- verify_kal_roundtrip.py：字节级回读 → 两遍法 forward **3/3**（gap 1.283/1.470/0.050）
- pack/unpack 无损验证：q 完全一致，gscale fp16 舍入 ≤5.2e-5
- qwen_kal_ref.c：**host C 参考实现（float32 全链，上板引擎的 CPU 骨架）** 读 weights_kal/ → **3/3**
  （NEXT 2130/12095/99366，gap 1.5328/1.1675/0.0239）。含 per-row INT8 激活、KG=32 两遍法、
  INT4 nibble 解包、GQA 注意、chunked down K=1024。
- dbg_ffn_layer0.py：层 0 逐级数值对拍工具（已用于定位 rope stride bug）。
- 设计规格：本文

## 9a. 设计确认 A — ION 24MB 内 per-group dequant（INT4→INT8）工作缓冲布局

（CEO 2026-08-13 裁决要求；依据 TPU 闸口③ Gate③ 方案 A：per-chunk 解包。已核销
DESIGN §4 的 per-matrix 双缓冲估算 —— CEO/TPU 闸口③ 以 **per-layer INT4 双缓冲 16.8MB**
为基数，本布局与之对齐，为最终裁决版。）

| 偏移（MiB） | 大小 | 区域 | 角色 |
|---|---|---|---|
| 0x000000 | 8,396,800 B | **SD_BUF_A** | 当前层 layerL INT4 K-aligned（nib+gsc），SD DMA 写入 |
| 0x802000 | 8,396,800 B | **SD_BUF_B** | 下一层 layerL+1（SD 双缓冲，DMA 与消费重叠） |
| 0x1004000 | 160 KiB | **DQ_BUF_A** | 当前 K-block 解包 int8 [32,4864]（CPU 写 / TIU 读） |
| 0x102A000 | 160 KiB | **DQ_BUF_B** | 下一 K-block 解包（CPU 双缓冲，与 TIU 消费重叠） |
| 0x1050000 | 64 KiB | **P1_BUF** | pass1 int8 输出 [M≤10, Ntile≤512] → CPU 读真实 max |
| 0x1060000 | 64 KiB | **P2_BUF** | pass2 int8 输出 [M≤10, Ntile≤512] |
| 0x1070000 | 1.5 MiB | ACT/ACC | x/h/attn/qkv/up/gate/out fp32 [M≤10, D/F] |
| 0x11F0000 | 0.9 MiB | KV/misc | r_opt MAX_BUF（per-K-block 112B）+ INT8 KV + scratch |
| 0x1280000 | **5.14 MiB** | **FREE** | 余量（INT8 KV 扩展 / LM head top-k staging / down 备用） |
| 合计 | **≈18.86 MiB** | — | **< 24 MiB，余 ~5.14 MiB** |

> **§9a 尺寸 bug 修正（2026-08-15）**：SD_BUF 原标 "8.0 MiB"（8,388,608 B）比真实 layer
> 文件 8,393,728 B 短 5,120 B，会越界。已按 A' 实施（`PLAN_APRIME_IMPL`）改为 page-align
> **8,396,800 B**（roundup(8,393,728, 4096)），SD_BUF_B 及后续区偏移顺延 16,384 B；
> 合计 ~18.86 MiB，余 ~5.14 MiB。A' 实际不再含 DQ_BUF 双缓冲（沿用既有单 DQ），余量更大。

**共存方案关键点**：
1. **DQ_BUF 必须在 ION/neuron 内存**：TIU 的 g2l 从 neuron 内存读 right 矩阵，
   CPU 解包输出必须落在 ION，纯 DDR 需二次拷贝（g2l 前）——否决。
2. **四级双缓冲嵌套**：SD_BUF(A/B) ║ DQ_BUF(A/B)。CPU 解包 block g+1 → DQ_BUF_B 时，
   TIU 正消费 block g 的 DQ_BUF_A 两遍；SD DMA 同时把 layer L+1 灌入 SD_BUF_B。
3. **DQ_BUF 尺寸上界 = up/gate [32,4864] = 152 KiB**（含对齐 160 KiB）；down [32,896]=28 KiB
   与 q/k/v/o [32,128]=4 KiB 均远小于上界，无需动态切换。
4. **INT4 nib 缓冲由 CPU 读（TIU 不读）**，故 SD_BUF 放 ION 而非 DDR 的唯一理由是避免
   DDR→ION 中转拷贝；SD DMA → ION 直写单链路最简。
5. **~5.14 MiB 余量** → 按闸口③建议配 INT8 KV（≥500 token）或 LM head 频度热区缓存。
6. **P1/P2 分开**：pass1 输出区被 CPU 读 max（须 MemInvld），pass2 输出区被 CPU 读回累加
   （再 Invld）；分开后两个缓冲的无效化互不干扰，且均只 Invld ~8KB 输出区而非全 buffer。

---

## 9b. 设计确认 B — 两遍法 640 submits/layer 流水线重叠可被 SD 掩盖（确认）

（CEO 2026-08-13 裁决要求；依据 TPU 闸口② Gate② 上板实测，KG=32 / Ntile=896。）

**每 token 预算核对（TPU Gate② 实测）**：

| 分量 | per-layer | ×24/token | vs SD 9.18s | 判定 |
|---|---|---|---|---|
| SD 读（201MB @21.9MB/s） | 382.4ms | **9.18s** | — | 主导 |
| TIU run-only（0.22ms/submit） | 140.3ms | **3.37s** | 0.37× | 被掩盖 ✓ |
| CPU 摊销（cmdbuf 预建+区域 Invld+RVV 解包+累加） | ~114ms | **2.73s** | 0.30× | 被掩盖 ✓ |
| naive 串行 submit（0.55ms/submit build） | 707.7ms | **16.99s** | 1.85× | 必须消除 ✗ |

**提交量**：640 submits/layer（q/k/v/wo/up/gate 各 28 chunk×2 遍=56，down 152×2=304），
×24 = **15,360/token**。与 CEO 设定一致（此数替代 DESIGN §2 早期 Ntile=512 估的 2064/layer）。

**三级流水（block 粒度，唯一串行点是 pass1→pass2 max 读回）**：
```
for layer L:
  SD DMA(L+1) -> SD_BUF_B                    // 独立异步
  for matrix:
    for K-block g:
      CPU:  dequant block g+1 -> DQ_BUF_B     // 不依赖 TIU，纯权重侧，可预取
      TIU:  两遍 submit g（pass1 rsafe -> 读 max -> pass2 r_opt）  // 6 N-tile 合并为一次提交
      CPU:  fp32 累加 p2×2^r×gsc[g,n]（与 TIU 下一 block 重叠）
```
- pass1 max 是 per-K-block 标量（对 block 内全部 N-tile 共享），故「pass1→读 max→pass2」
  只产生一次 CPU 等待；等待窗口内 CPU 切去解包 block g+1，不空转。
- **吞吐 = max(9.18, 3.37, 2.73) = 9.18s/token → SD-bound**。TIU/CPU 均 <40% SD，
  余量 >2.5×；即使 C906B 反量化吞吐实测翻倍（5.5s）仍被 SD 掩盖。

**维持 SD-bound 的两条硬约束（引擎必须落实，否则 16.99s 串行超 SD）**：
1. **cmdbuf 预建**：16 rshift 变体 × 7 shape 预注册，submit = select + patch rshift + Run
   （~15µs）；**禁止** per-submit register+alloc+convert+load 全链路（0.55ms/submit 会吃掉 8.43s/token）。
2. **MemInvld 限定输出区**（P1/P2 ~8KB），**禁止** 全 8MB buffer Invld（~0.6ms/chunk 假象）。

**结论：确认可被 SD 掩盖。** 640 submits/layer 与 SD 9.18s/token 是兼容的；前提是
cmdbuf 预建 + 区域 MemInvld 两条硬约束进引擎实现（闸口② §2.3 已给出摊销后 2.73s 的
现实 CPU 预算）。

---

## 10. C 参考调试记录（关键）

- **症状**：层 0 after-wo 即发散（C 0.00426 vs Py 0.00374），FFN h 差 0.018，next=98933。
- **定位链**：attn 行 1 NaN ← q 行 1 污染（-66M/3.7e11）← rope 步内（after bias 正常 → after rope 污染）
  ← **rope 预计算写 `cosb[pos*HD+j]` 而 rope_inplace 读 `cos[pos*(HD/2)+i]`**：stride 不一致，
  pos≥1 读到未初始化内存。
- **修复**：预计算 stride 改 `pos*(HD/2)`。修复后层 0 全级逐值匹配 Python（9 位小数一致）。
- **启示**：C906B 上板引擎 rope 表必须按 `[pos][32]`（stride=HD/2）布局，否则同样崩溃。
