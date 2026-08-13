# Path B per-chunk INT8 matmul 上板验证报告（ps32-free 出口 / 闸口①③）

日期：2026-08-13
调查：TPU 底层工程师（CV1800B 上板实测，`pathb_chunk_probe.c`）
联动：推理引擎工程师（host 扫参 KG=32~256 3/3）
记录：DUO project CEO
状态：**闸口① 数据流 + 闸口③ 无新增硬限制 = 通过**，可转 Path B 引擎实现

---

## 0. 结论（TL;DR）

CEO 交办的 ps32-free 出口（**per-chunk INT8 matmul 多 submit + 自适应 per-row rshift + CPU fp32 累加**）
在 CV1800B 上板**完全可实现**：

1. **① 数据流可行**：KG=256（及 128）两遍法（pass1 安全 rshift 回读 → r_opt → pass2）
   在 q/k/v/o/up/gate/down 全部生产形状上，**与 host 两遍法仿真逐位一致（BIT-EXACT，
   maxrel=0.00e+00）**。TIU 舍入= `sat8((acc + 2^(r-1))>>r)` 完全复现 host 假设。
2. **② 延迟可被 SD 读完全掩盖**：per-chunk 串行 116ms/layer → **2.79s/token（最坏）**；
   TIU run-only 35.8ms/layer → **0.86s/token**。均远小于 Path B SD 读 680ms/layer（358MB/token
   @21.9MB/s ≈ 16.3s/token）。**decode 上界不变**。对比全 K 基线：单矩阵快 3~13x，但
   **down（K=4864）全 K 根本放不进 lmem → 不可行**，且全 K 质量 1/3 已判负。
3. **③ 无新增硬限制**：KG=256 rshift=1..15 全部逐元素精确（含饱和）；N-tile 上限纯 lmem 约束
   （KG=256 → N_tile≤112，KG=128 → ≤224，KG=64 → ≤448，KG=32 → ≤896）。
4. **KG/ION 对齐建议**：KG=256（N_tile=112）成本最低（down 47ms vs KG=128 的 87ms）；
   若质量余量不足退回 KG=128（N_tile=192）。ION 缓冲 ≈ 10.3MB < 24MB，**Path B 无
   INT4→INT8 反量化**（权重直接 per-channel INT8），无 Path A 的 CPU 反量化瓶颈。

---

## 1. ① 数据流可行性：两遍法 BIT-EXACT（闸口① 通过）

`pathb_chunk_probe.c` 的 P2 对 Qwen 生产形状逐矩阵实测（M=1 decode，synthetic 数据
|left|≤80 / |right|≤30，rsafe=14/13 保证 pass1 不饱和）：

| 矩阵 | [K,N] | KG | N_tile | chunks | r_opt 区间 | vs host 两遍法 |
|---|---|---|---|---|---|---|
| q/wo | 896×896 | 256 | 112 | 4 | 14 | **BIT-EXACT** (896/896, maxrel=0) |
| q/wo | 896×896 | 128 | 192 | 7 | 13 | **BIT-EXACT** (896/896, maxrel=0) |
| k/v | 896×128 | 256 | 112 | 4 | 14 | **BIT-EXACT** (128/128, maxrel=0) |
| k/v | 896×128 | 128 | 192 | 7 | 13 | **BIT-EXACT** (128/128, maxrel=0) |
| up/gate | 896×4864 | 256 | 112 | 4 | 14 | **BIT-EXACT** (4864/4864, maxrel=0) |
| up/gate | 896×4864 | 128 | 192 | 7 | 13 | **BIT-EXACT** (4864/4864, maxrel=0) |
| down | 4864×896 | 256 | 112 | 19 | 14 | **BIT-EXACT** (896/896, maxrel=0) |
| down | 4864×896 | 128 | 192 | 38 | 13 | **BIT-EXACT** (896/896, maxrel=0) |

- 上板 acc_fp32（pass2×2^r 跨 chunk 累加 → ×sc_row×lsc_col）与 host 两遍法仿真**逐位一致**。
- 即：给定相同的 r_opt，TIU 输出与 host `int8_round_div` 完全一致 → **host 3/3 的算法语义
  在硬件上原样成立**。
- 注：synthetic 数据 r_opt 恒为 14（数据接近理论界），未覆盖真实 Qwen 的 r_opt 8~12 变化区间；
  但 P4 已对 rshift=1..15 全覆盖验证，机制层面无差别。

---

## 2. ② 提交延迟 / 每 token 开销 / 对比全 K 基线

### 2.1 每 chunk 延迟（KG=256，串行 = 最坏）

| 矩阵 | 每 chunk 总 | build | run | read+cpu |
|---|---|---|---|---|
| q [896×896] | 2.54ms | 1.20ms | 0.67ms | 0.67ms |
| k [896×128] | 2.30ms | 1.16ms | 0.45ms | 0.69ms |
| up [896×4864] | 3.74ms | 1.20ms | 1.83ms | 0.71ms |
| down [4864×896] | 2.40ms | 1.09ms | 0.66ms | 0.65ms |

- run（TIU+TDMA 实际执行）是硬成本；build/read+cpu 为 CPU 侧，引擎三级流水（SD║CPU║TIU）
  可重叠。
- read+cpu 含每次**全 8MB MemInvld 的固定开销 ~0.6ms**（探针人为，引擎可只 invld 输出区规避）。

### 2.2 每 token 总开销（KG=256，串行上界）

| 分量 | 每层 | ×24/token |
|---|---|---|
| build | 51.2ms | 1.23s |
| run（TIU+TDMA） | 35.8ms | **0.86s** |
| read+cpu | 29.0ms | 0.70s |
| **串行合计** | **116ms** | **2.79s** |

- **关键**：Path B SD 读 = 358MB/token @21.9MB/s ≈ **16.3s/token**（=680ms/layer）。
  per-chunk 串行 116ms/layer、run-only 35.8ms/layer 均 ≪ 680ms → **完全可被 SD 读掩盖**。
  → **decode 上界（SD 受限 ~16.3s/token）不受 per-chunk 提交量影响**（与 REPORT_PS32 §7 结论一致）。

### 2.3 对比当前 INT8 全 K 基线

| 矩阵 | 全 K 基线（rshift=12） | per-chunk 2-pass | 倍率 | 说明 |
|---|---|---|---|---|
| q | 1.21ms | 10.18ms | 8.4x | 全 K N_tile=32→28 tiles |
| k | 0.72ms | 9.18ms | 12.7x | |
| up | 4.55ms | 14.96ms | 3.3x | |
| down | **INFEASIBLE** | 45.5ms | — | K=4864 右矩阵连 N_tile=5 都放不进 32KB lmem |

- **全 K 基线对 down 不可行**（`[fullk] lmem alloc FAIL K=4864 Ntile=5`）：
  4864×5+4864+5≈29KB 虽可放，但 bmkernel 对 w=5 的最小 lane 宽度对齐使其超 32KB。
  → **down 必须 K 切分**，per-chunk 是唯一可行布局（非可选优化）。
- 即便可行矩阵，全 K 单 rshift 是 Qwen 质量 1/3 判负的根因（REPORT_PS32 §6）。per-chunk
  的 3~13x 计算成本是换 3/3 质量的代价，且被 SD 掩盖。

---

## 3. ③ 无新增硬限制（rshift 精度 / 饱和 / N-tile）

### 3.1 rshift 精度与饱和（KG=256，P4）

| rshift | 正确性 | 饱和数 |
|---|---|---|
| 1 | 112/112 bad=0 | 111 |
| 2 | 112/112 bad=0 | 106 |
| 4 | 112/112 bad=0 | 99 |
| 8 | 112/112 bad=0 | 1 |
| 12 | 112/112 bad=0 | 0 |
| 15 | 112/112 bad=0 | 0 |

- 覆盖自适应 r_opt 真实区间（8~12）与极端饱和（1/2/4）。TIU 语义 `sat8((acc+2^(r-1))>>r)`
  **对负半值、饱和均成立**（延续 rshift_check.c 的 K=32/128 结论，扩展到 KG=256）。
- **CPU 侧反量化必须用同一 round-half-up 公式**，否则系统性偏差 1（已在 GATE_A_SIGNOFF 记录）。

### 3.2 N-tile lmem 上限（P1，bmkernel raw layout）

| KG | 最大 N_tile | 右矩阵 lmem |
|---|---|---|
| 32 | 896 | 28KB |
| 64 | 448 | 28KB |
| 128 | 224 | 28KB |
| 256 | **112** | 28KB |
| 512 | 48 | 24KB |

- 上限纯由 32KB lmem 容量决定（右矩阵 + 左 + res 的原始字节），非 TIU 硬件上限。
- 引擎实现若用 cvikernel 的成对打包布局，N_tile 可能到 ~2x（待引擎工程师实现时确认）；
  本报告给出 bmkernel 原始布局的保守界，**按此界可保证可行**。

### 3.3 其他

- **M=1（decode）per-row rshift = 每 chunk 单一标量**，与 host 仿真逐位一致。
- **M>1（prefill）per-row rshift 需注意**：TIU 单 matmul 只有一个 rshift，多行需 M 次 pass2
  提交（或共享 rshift 损失自适应收益）。decode 主路径无影响，prefill 留待引擎定方案。

---

## 4. KG 取值与 ION 缓冲方案（与推理引擎工程师对齐）

### 4.1 KG 建议

- **优先 KG=256**：down 19 chunk（vs KG=128 的 38 chunk）→ 45.5ms vs 87ms，成本最低；
  N_tile=112。若 host 两遍法 KG=256 质量余量不足（REPORT_PS32 §6 曾记 KG=256 退化 2/3），
  **退回 KG=128**（N_tile=192，q/up 提交数 ~1.5x，down ~1.9x）。
- 上板数据对两种 KG 均为 BIT-EXACT，硬件层无偏好 → **最终由 host 质量扫参决定**。

### 4.2 ION 缓冲（Path B，24MB 预算内）

| 区 | 大小 | 说明 |
|---|---|---|
| 权重双缓冲（最大矩阵 up/gate/down 4.36MB×2） | 8.72MB | per-matrix 双缓冲 |
| lsc per-channel scale | ~50KB | fp16，可常驻 |
| P1/P2 int8 输出 [1,N] | 2×4.9KB | 每 chunk 复用 |
| acc fp32 [M,N] | ~20KB | CPU 累加 |
| 激活/中间量 | ~1.5MB | x/h/attn/qkv/mid/out |
| **合计** | **≈10.3MB** | **< 24MB，余量充足** |

- **Path B 无 INT4→INT8 反量化**（权重直接 per-channel INT8）→ **不存在 Path A 的 CPU
  反量化新瓶颈**（IMPL_C906B §5）。换取的是 SD 带宽 358MB/token（vs Path A 201MB）。
- 三级流水沿用 DESIGN_PATH_A §5：SD 读(matrix i+1) ║ CPU 累加(matrix i) ║ TIU 两遍(matrix i)。

---

## 5. 验证方式（可复现）

- `pathb_chunk_probe.c`（本提交入库）：P1 N-tile 扫描 / P2 两遍法逐形状 BIT-EXACT /
  P3 延迟 + 全 K 基线 / P4 rshift 精度。CV1800B 上板实测，8MB neuron buffer，bmkernel 原始布局。
- 数据：synthetic INT8（|left|≤80, |right|≤30），rsafe=ceil(log2(KG·80·30/127))+1。
- host 参照：`int8_round_div`（round-half-up + sat8）模拟 pass1→r_opt→pass2 两遍法，逐位比对。

---

## 6. 裁决与下一步

1. **闸口①（per-chunk 两遍法数据流）通过**：所有 Qwen 形状 BIT-EXACT。
2. **闸口③（rshift 精度/饱和、N-tile）通过**：无新增硬限制；down 必须 per-chunk（全 K 不可行）。
3. **提交量非瓶颈**：2.79s/token（串行）与 0.86s/token（run-only）均被 SD 读（16.3s/token）掩盖。
4. **对齐输出**：建议 KG=256/N_tile=112（fallback KG=128/N_tile=192）；ION ≈10.3MB。
5. 待推理引擎工程师：确认 KG（质量扫参）+ 落 ION 布局 + 出正式引擎 REPORT；TPU 侧 Path B 无遗留闸口。
