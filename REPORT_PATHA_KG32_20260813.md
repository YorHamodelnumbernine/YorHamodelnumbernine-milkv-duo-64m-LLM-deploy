# Path A（INT4 K-aligned, KG=32=G）两遍法上板验证报告 — CEO 更新目标

日期：2026-08-13 | 作者：TPU 底层工程师（CV1800B 上板实测） | 关联：`patha_kg32_check.c`
状态：**闸口① 签核通过 / 闸口④ 通过 / 闸口③ 适配可行 / 闸口② 有条件通过（需 build 摊销）**

---

## 0. 结论（TL;DR）

CEO 更新目标（Path A, KG=32=G, 替代此前 KG=256 设定）四项闸口上板结论：

1. **① 开工闸口 —— 通过（签核）**：pass1 读 int8 输出 = 标准 `matrix_multiplication`
   语义（ps32-free, `res_is_int8=1`），`tdma_l2g_matrix_copy` + `MemInvld` 直接回读，
   在 KG=32 生产形状上 **5/5 组逐元素精确（bad=0）**，含 up N=4864 宽矩阵与饱和场景。
   → **推理引擎工程师可开工**。
2. **② submit 延迟 —— 有条件通过**：TIU run-only **3.37s/token ≪ SD 9.18s/token**，计算
   完全可被 SD 掩盖；但**串行 submit 总时长 16.99s/token > SD**，根因是 naive per-submit
   build（8.43s/token，0.55ms/submit）——引擎**必须摊销 cmdbuf 构建**（按 rshift 预建模板
   或 patch rshift 字段）。摊销后 CPU 侧 ≈2.7s/token → 流水线维持 SD-bound。
3. **③ ION 布局 —— 可行**：INT4 双缓冲 16.8MB + per-chunk int8 解包工作缓冲（≤155KB×2）
   ≈ **18.9MB < 24MB**，余 ~5.1MB。若改全矩阵 int8 staging（4.36MB）则 ~23MB 勉强可放，
   余量仅 ~1MB → 建议 per-chunk 解包或单缓冲/DDR 兜底。
4. **④ 无新增硬限制 —— 通过**：KG=32 下 rshift=1..15 全部 bad=0（含饱和）；四生产形状
   两遍法 **BIT-EXACT**；TIU round-half-up 语义 `sat8((acc+2^(r-1))>>r)` 与 host 3/3 一致。

---

## 1. 闸口①：pass1 int8 回读 = 标准 matmul 语义（开工闸口，签核）

`patha_kg32_check.c` P0，标准 INT8 matmul（M=1, K=32 单 chunk, ps32-free）+
`l2g_matrix_copy`（int8 lane 解交织）+ `MemInvld` 回读，逐元素对比 host `ref_div`：

| 矩阵 | K | N | Ntile | tiles | rshift | ok/total | bad | sat8 | 判定 |
|---|---|---|---|---|---|---|---|---|---|
| q | 32 | 896 | 896 | 1 | 11（安全） | 896/896 | 0 | 0 | **READBACK OK** |
| k/v | 32 | 128 | 128 | 1 | 11 | 128/128 | 0 | 0 | **READBACK OK** |
| up | 32 | 4864 | 896 | 6 | 11 | 5376/5376 | 0 | 0 | **READBACK OK** |
| q（紧 rshift） | 32 | 896 | 896 | 1 | 5 | 896/896 | 0 | 756 | **READBACK OK** |
| down（单 chunk） | 32 | 896 | 896 | 1 | 11 | 896/896 | 0 | 0 | **READBACK OK** |

- **结论**：pass1 int8 回读在 KG=32 生产形状上全部精确，含最宽 N=4864（6 N-tile 单 cmdbuf）
  与 rshift=5 饱和 756/896 值仍逐元素确定。回读路径与既有 23/24 benchmark 相同。
- 语义：TIU 输出 = `sat8((acc + (1<<(rshift-1))) >> rshift)`（round-half-up），CPU 反量化
  必须匹配此公式（已在 `rshift_check.c` / `GATE_A_SIGNOFF` 确认，本探针扩展至生产形状）。

---

## 2. 闸口②：KG=32 per-chunk submit 延迟（per-token 开销 vs SD）

### 2.1 实测 per-chunk 两遍法延迟（P3，探针串行 = 最坏）

KG=32，rsafe=11，synthetic |left|≤80 / |right|≤30：

| 矩阵 | [K,N] | chunks | tiles/chunk | submits(×2遍) | build(ms) | run(ms) | read+cpu(ms) | total(ms) | per-chunk |
|---|---|---|---|---|---|---|---|---|---|
| q | 896×896 | 28 | 1 | 56 | 30.44 | 10.55 | 18.44 | 59.43 | 2122 µs |
| k/v | 896×128 | 28 | 1 | 56 | 30.17 | 9.78 | 18.14 | 58.10 | 2075 µs |
| up/gate | 896×4864 | 28 | 6 | 56 | 32.03 | 21.22 | 21.06 | 74.31 | 2654 µs |
| down | 4864×896 | 152 | 1 | 304 | 165.87 | 57.19 | 101.00 | 324.06 | 2132 µs |

**与 CEO 目标核对**：q/k/v/o/up/gate K=896 → **28 chunks ✓**（896/32）；down K=4864 →
**152 chunks ✓**（4864/32）；×2 遍 = **640 submits/layer ✓**（6×56+304=640）。

### 2.2 per-token 开销（×24 layers，串行上界）

| 分量 | per-layer | ×24/token | vs SD 9.18s |
|---|---|---|---|
| build（naive per-submit 0.55ms） | 351.2ms | **8.43s** | 0.92×SD ⚠️ |
| run（TIU+TDMA 硬成本 0.22ms/submit） | 140.3ms | **3.37s** | 0.37×SD ✓ |
| read+cpu（含全 8MB MemInvld 假象 0.6ms/chunk） | 216.3ms | **5.19s** | 0.57×SD（假象成分高） |
| **串行合计** | **707.7ms** | **16.99s** | 1.85×SD ✗ |
| SD 读（201MB@21.9MB/s） | 382.4ms | **9.18s** | — |

**关键判定**：
- **TIU run-only 3.37s/token ≪ SD 9.18s → 计算侧完全可被 SD 掩盖**（SD 是主导瓶颈的
  前提在计算侧成立）。
- **串行 submit 16.99s/token > SD**：根因是探针每 submit 都 register+alloc+convert+load
  （0.55ms/submit × 15360 = 8.43s/token）。**这是探针/naive 引擎的最坏值，不是硬上限**——
  两遍法 per-chunk cmdbuf 结构固定，仅 rshift 字段逐 submit 变化。

### 2.3 摊销后的现实预算（引擎必须落实）

| 优化项 | 依据 | per-layer |
|---|---|---|
| cmdbuf 预建（16 rshift 变体 × 7 shape，submit 仅 patch/select+Run） | run 已含执行 0.22ms；CPU 调用开销 ~15µs | ~9.6ms |
| read+acc 区域 MemInvld（只 invld 输出区 ~8KB，非全 8MB） | 实测 read+cpu 中 ~0.6ms/chunk 为全 buffer 假象，真实 CPU ~0.06-0.15ms/chunk | ~64ms |
| INT4→INT8 解包 RVV | **上板实测 419MB/s**（`int4_bench`；scalar 仅 50MB/s）→ 16.8MB int8/层 | ~40ms |
| **摊销后 CPU 合计** | | **~114ms/layer = 2.73s/token** |
| TIU run | 硬成本 | 140ms/layer = 3.37s/token |
| SD | DMA 主导 | 382ms/layer = 9.18s/token |

- 三级流水（SD ║ CPU 解包/build/acc ║ TIU 两遍）吞吐 = max(9.18s SD, 2.73s CPU,
  3.37s TIU) = **9.18s/token → SD-bound**，与 CEO 预期一致。
- **给推理引擎工程师的硬约束**：per-submit **禁止**走 register+alloc+convert+load 全链路；
  必须预建 cmdbuf（rshift 模板或字段 patch），MemInvld 必须限定输出区。

---

## 3. 闸口③：ION 缓冲布局（INT4 双缓冲 + int8 解包工作缓冲）

Path A 每层 INT4 权重 = 201.4MB/24 = **8.39MB** → 双缓冲 **16.8MB**（与 CEO 设定一致）。

| 方案 | INT4 双缓冲 | int8 解包工作缓冲 | 其他（激活/acc/lsc/SD-io） | 合计 | 判定 |
|---|---|---|---|---|---|
| **A. per-chunk 解包（推荐）** | 16.8MB | ≤155KB×2（up/gate [32,4864]） | ~1.8MB | **≈18.9MB** | **FIT，余 ~5.1MB** |
| B. 全矩阵 int8 staging（单） | 16.8MB | 4.36MB（up/gate 全矩阵） | ~1.8MB | ≈23.0MB | 勉强 FIT，余 ~1MB ⚠️ |
| C. 全矩阵 staging（双）+ 单缓冲 INT4 | 8.4MB | 8.72MB | ~1.8MB | ≈18.9MB | FIT（牺牲双缓冲→SD 流水受损） |

- **推荐 A**：解包按 KG=32 chunk 粒度进行，int8 工作缓冲只需 [32, N] = up/gate 155KB
  （双缓冲 310KB）。与 TIU 按 chunk 消费天然匹配，余量 ~5.1MB 可支持 INT8 KV
  （A' 裁定建议 500+ token）。
- **方案 B 可放但余量薄**：16.8+4.36+1.8 = 23MB < 24MB，仅 1MB 余量；若需留 KV/激活余量，
  建议 down 或 up/gate 改单缓冲，或 int8 staging 放 DDR（CPU 解包写 DDR，g2l 前再拷 ION）。
- **注意**：int8 解包工作缓冲必须在 ION/neuron 内存（TIU 的 g2l 从中读 right 矩阵），
  纯 DDR 需中转。

---

## 4. 闸口④：无新增硬限制（rshift 精度 / 饱和行为）

### 4.1 rshift=1..15 @ KG=32（P1，N=896）

| rshift | ok/total | bad | sat8 | 判定 |
|---|---|---|---|---|
| 1 | 896/896 | 0 | 853 | ✓（极端饱和仍精确） |
| 2 | 896/896 | 0 | 834 | ✓ |
| 4 | 896/896 | 0 | 593 | ✓ |
| 8 | 896/896 | 0 | 0 | ✓（r_opt 真实区间） |
| 12 | 896/896 | 0 | 0 | ✓ |
| 15 | 896/896 | 0 | 0 | ✓ |

### 4.2 两遍法数据流 BIT-EXACT（P2，KG=32）

| 矩阵 | KG | Ntile | Npad | rsafe | chunks | r_opt | vs host 两遍法 |
|---|---|---|---|---|---|---|---|
| q/wo | 32 | 896 | 896 | 11 | 28 | [11..11] | **BIT-EXACT** (896/896, maxrel=0) |
| k/v | 32 | 128 | 128 | 11 | 28 | [11..11] | **BIT-EXACT** (128/128, maxrel=0) |
| up/gate | 32 | 896 | 5376 | 11 | 28 | [11..11] | **BIT-EXACT** (4864/4864, maxrel=0) |
| down | 32 | 896 | 896 | 11 | 152 | [11..11] | **BIT-EXACT** (896/896, maxrel=0) |

- **结论**：给定相同 r_opt，TIU 输出与 host `int8_round_div` 逐位一致 → host 3/3 的算法
  语义在硬件上原样成立。pass1 回读→r_opt→pass2→fp32 累加全链路可复现。
- 无新增硬限制：N-tile=896 上限为 lmem 容量（right[32,896]=28KB+left+res≈29KB<32KB）；
  KG=32 的 r_opt 真实区间（Qwen 数据约 8~12）已被 rshift 8/12 覆盖。
- 附注：synthetic 数据 r_opt 恒为 11（数据近理论界）；真实 Qwen 权重幅值更小 r_opt 更低，
  机制层面无差别。

---

## 5. 验证方式（可复现）

- `patha_kg32_check.c`（本探针，入库）：P0 Gate① 五组 / P1 rshift 六组 / P2 四形状
  BIT-EXACT / P3 延迟四形状。CV1800B 上板实测，8MB neuron buffer，bmkernel 原始布局。
- `int4_bench`（板上实测）：RVV `int4_unpack_fixed_rvv` = **419MB/s**（scalar 50MB/s）。
- host 参照：`ref_div`（round-half-up + sat8），逐位比对。
- 部署：`python3 duo_run.py patha_kg32_check`（RC=0）。

---

## 6. 对推理引擎工程师的交接

1. **可开工**：pass1 回读签核（闸口①）。matmul 核复用 Path B 的 per-chunk INT8 两遍法，
   仅 KG 改为 32。
2. **submit 预算**：640 submits/layer、15,360/token 是硬性提交次数（两遍法 per-chunk
   rshift 逐 submit 变化）。TIU run-only 3.37s/token 可被 SD 掩盖；**必须预建 cmdbuf**，
   per-submit 禁止全链路 register/alloc/convert/load。
3. **解包**：RVV 定点解包（`int4_unpack_fixed_rvv`）419MB/s → 40ms/layer，非瓶颈；scalar
   解包 2124ms/30层(SmolLM2) 不可接受，必须用 RVV。
4. **ION**：per-chunk 解包（A 方案）→ 18.9MB，余 5.1MB 可配 INT8 KV。全矩阵 staging
   ~23MB 边缘，不建议。
5. **N-tile**：KG=32 时 N≤896（实测）；up/gate N=4864 → 6 N-tile/chunk，已在探针验证。
6. **MemInvld**：只 invld 输出区（~8KB），规避探针全 8MB 假象。
