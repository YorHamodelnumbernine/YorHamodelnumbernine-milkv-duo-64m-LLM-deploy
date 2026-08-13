# Path A 两遍法闸口签核 — 上板实证（Gate ① pass1 int8 回读 + Gate ④b N-tile）

日期：2026-08-13 | 作者：TPU 底层工程师 | 状态：**已签核，上板实测** | 关联：`gate_a_check.c`

---

## 0. 签核结论（TL;DR）

- **Gate ①（pass1 int8 回读）签核通过**：标准 INT8 matmul（ps32-free、
  `res_is_int8=1`）+ `tdma_l2g_matrix_copy` + `MemInvld` 回读，在
  K=32/128、N=192、rshift=8/5 下 **全部逐元素精确**（bad=0）。
  含饱和场景（rshift=5，部分值 sat8 到 ±127）仍确定可复现。
- **Gate ④b（N-tile=512）签核通过**：标准 INT8 两遍法**不受 ps32 N=192
  硬件上限约束**，仅受 lmem 容量约束。K=32 下 **N=512 实测 512/512 正确**
  （right[32,512]=16KB）；K=32 最大 N=896（28KB，N=1024 因对齐越界 alloc fail）。
  KG=128 下最大 N=192（right[128,192]=24KB）。
- **TIU 舍入语义补订（含小 rshift + 负数，CEO 追问项）**：INT8 matmul
  rshift>0 时为 **round-half-up**（11440>>8 → 45，非截断 44），**对所有
  rshift=1..4、含负数半值均成立**（`rshift_check.c` 上板：6 组 bad=0，
  负/正半值命中 32/38、17/16、4/8、5/2、6/10、4/3）。即 signed 值向 +∞ 舍入：
  `out = sat8((acc + (1<<(rshift-1))) >> rshift)`。CPU 侧反量化必须匹配此语义，
  否则两遍法 pass2 结果系统性偏差 1。

## 1. Gate ①：pass1 int8 回读（ps32-free 标准 INT8 matmul + rshift）

| K | N | rshift | res lmem | 正确性 | 备注 |
|---|---|---|---|---|---|
| 32 | 192 | 8 | 6176 | **192/192 bad=0** | KG=32 组 |
| 128 | 192 | 8 | 24704 | **192/192 bad=0** | KG=128 块（设计主档） |
| 128 | 192 | 5 | 24704 | **192/192 bad=0** | 部分饱和→±127，仍确定 |
| 256 | 192 | 8 | — | alloc fail | right[256,192]=48KB>32KB（文档化） |

- 回读路径与既有 23/24 benchmark 相同：`l2g_matrix_copy`（int8 res 为
  lane-interleaved，矩阵拷贝自动解交织）+ `MemInvld`。
- 判定：**pass1 int8 回读可正式签核**。rshift 只需保证 pass1 不饱和
  （安全 rshift），pass2 用每 chunk 实测 max 精化。

## 2. Gate ④b：N-tile 宽度扫描（标准 INT8 matmul，两遍法 tiling）

| K | N | lmem_r (bytes) | 正确性 | 判定 |
|---|---|---|---|---|
| 32 | 192 | 6144 | 192/192 | ✓ |
| 32 | 256 | 8192 | 256/256 | ✓ |
| 32 | 384 | 12288 | 384/384 | ✓ |
| 32 | **512** | 16384 | **512/512** | **N-tile=512 可行** |
| 32 | 896 | 28672 | 896/896 | 最大实用宽度 |
| 32 | 1024 | — | alloc fail | 32768+对齐 > 32KB |
| 128 | 192 | 24576 | 192/192 | ✓（设计主档） |
| 128 | 256 | — | alloc fail | 32768+对齐 > 32KB |
| 128 | 384 | — | alloc fail | 48KB > 32KB |

- **关键结论**：两遍法（ps32-free）的 N-tile 上限是 **lmem 容量**而非 TIU
  硬件（ps32 per-group 路径才有 N=192 TIU 上限）。设计杠杆：
  - **KG=128 + N-tile=192**：7 K-chunk，right 24KB → 当前设计。
  - **KG=32 + N-tile=512**：28 K-chunk（submit 更多），right 16KB，但每个
    N-tile 更宽（Wq 每 chunk 2 次而非 5 次）→ submit 折衷可调。
- 对齐开销说明：right[K,N]=32768（恰 32KB）仍 alloc fail，因矩阵 lmem
  分配含 bank 对齐/元数据；实际容量上限 ~31KB。

## 3. 对推理引擎实现的直接输入

1. pass1/pass2 的 CPU 反量化公式（K=128, N=192 chunk）：
   `out_s8 = sat8((acc_i32 + (1<<(rshift-1))) >> rshift)`，rshift>0 时必加
   round-half-up 偏置。
2. pass1 安全 rshift 建议：先按 chunk 内理论 max 保守取（如 +0.5 倍余量），
   pass2 按 pass1 实测 max 精化；饱和路径已验证确定（rshift=5 全对）。
3. 无需受 ps32 N=192 限制——两遍法 tiling 自由选择（KG=128/N=192 或
   KG=32/N=512）。
4. alloc fail 边界已实测：KG=128 时 N 不得 ≥256；KG=32 时 N 不得 ≥1024。

## 4. 产物

- `gate_a_check.c`（本次闸口探针：Gate ① 三组 + Gate ④b 九组宽度扫描）
- `dbg_mm.c`（已知数据调试件，确定 round-half-up 语义：11440>>8→45）
- `rshift_check.c`（小 rshift + 负数半值确认探针：rshift=1..4、K=32/128，6 组 bad=0）
- `rs_noreload_check.c`（2026-08-13 晚补充探针：Gate ④a 同 LMEM 权重块、per-call rshift_bits、pass1/pass2 不重载）

## 5. 补充签核（2026-08-13 晚，CEO 闸口核对 → 新增 ④a 上板验证）

Path A 两遍法设计的 ④a 要求：pass1(rsafe) 与 pass2(r_opt) 用**同一 LMEM 权重块**、
仅 rshift_bits 不同、**不重载**。此前 `patha_kg32_check.c` P2 两遍间实际重载了 right；
本次 `rs_noreload_check.c` 上板直接验证不重载路径（RC=0）：

| 测试 | 结构 | rshift | 结果 |
|---|---|---|---|
| Test A | 单 cmdbuf：g2l left/right 一次，matmul(11)→l2g P1，matmul(5)→l2g P2（同一 ml_r，无 right g2l） | 11 / 5 | 512/512 + 512/512 bad=0（pass2 含 408 个 sat8 值仍精确） |
| Test B | 真实两遍：cmdbuf1 g2l+mm(rsafe=11)→读 P1 算 r_opt；cmdbuf2 fresh ctx 同 alloc 布局仅 mm(r_opt)→l2g P2（无 right 重载） | 11→r_opt=11 | 512/512 bad=0 |

- **结论：bmk1822 支持同块不重载**。rshift_bits 是 TIU 指令字段（非权重编码），
  matrix_multiplication 不修改输入 → 权重块在 pass1/pass2 间驻留复用安全。
  该结论直接支撑引擎 cmdbuf 预建模板（pass1/pass2 共享 right 的 g2l，省一次带宽）。
- Test A 同时复证 ④b（K=32/N=512 right=16KB + res 512B + left 32B alloc 成功）与 ④c
  （标准 INT8 matmul 全 512 列正确，含饱和）。

## 5b. 补充签核（2026-08-13，CEO 闸口①复核 → 新增 M>1 两遍法上板验证）

CEO 闸口①复核要求匹配 `qwen_kal_ref.c::chunk_matmul_twopass` 语义（M>1 行）的
轻量上板测试。新增 `gate1_mrow_check.c`：完全复刻参考实现的 rsafe 公式
（`matmul_rshift_w(32,wmax)-3`，clamp≥4）、pass1 int8 回读取 per-chunk max、
r_opt 精化、pass2 重算，全部 bmk1822 标准 INT8 matmul（ps32-free, res_is_int8=1）。

**重要布局结论**：必须用**线性 FORCED 布局** `{n,c=1,w=全宽,col=全宽}`（同
`gate_a_check`），不能用 `bmk1822_matrix_lmem_default_shape`（其返回 c-split
布局如 `r{n=32,c=8,w=64,col=512}`，bmk1822 `l2g_matrix_copy` 无法正确解交织，
实测全错）。线性布局 + l2g_matrix_copy 是唯一已验证的回读路径。

| M | K | N | lmem(线性) | P1 bad | maxabs | r_opt | P2 bad | sat8 | 判定 |
|---|---|---|---|---|---|---|---|---|---|
| 16 | 32 | 512 | 25088 | **0/8192** | 128* | 10 | **0/8192** | 0 | ✓ ④b N=512+M>1 |
| 32 | 32 | 256 | 17408 | **0/8192** | 128* | 10 | **0/8192** | 0 | ✓ 更大 M |
| 64 | 32 | 128 | 14336 | **0/8192** | 127 | 9 | **0/8192** | 2 | ✓ MAX_SEQ=64 |
| 1  | 32 | 512 | 16928 | **0/512** | 107 | 9 | **0/512** | 0 | ✓ ④b decode 复核 |

\* maxabs=128 表示 pass1 实测含 -128（rsafe=9 在 |x|,|w|≤100 最坏数据下饱和），
回读仍逐元素精确 → 复证"饱和路径确定性"（同 §1 rshift=5 结论）。r_opt 由
`est=maxabs<<rsafe` 求出（M=16/32 时 est=65536→r_opt=10），P2 仍 0 bad。
数值累加路径 `p2*(1<<r_opt)*gsc[n]*sc_row[m]` 与 fp64 参考最大偏差 ~1.8e-06（fp32 舍入级）。

**对引擎的 ④b 设计输入（M>1 时的 lmem 预算）**：N-tile=512 在 M=1（decode，
16.5KB）与 M=16（24.5KB）均可容纳；M=32 时 [32,32]×[32,512] 需 res 16KB + right
16KB + left 1KB = 33KB 超限 → prefill 大 M 需按 M 或 N 切 tile（如 M=16/N=512 或
M=32/N=256）。无 TIU 侧新限制（标准 INT8 matmul 不受 ps32 N=192 上限约束）。

### 提交量口径备忘（CEO 要求标注两种 tiling）

| tiling | submits/layer | submits/token | 依据 |
|---|---|---|---|
| KG=32 + N-tile=896（patha_kg32 P3 实测） | 640（6×56 + down 304） | **15,360** | REPORT_PATHA_KG32 §2.1 |
| KG=32 + N-tile=512（设计默认，k/v 等窄矩阵仍 1 tile） | 2,064 | **≈49,500** | DESIGN_PATH_A_TWOPASS §2 |

- 两口径均 SD-bound（TIU 183–200ms/token ≪ SD 9.2s），不影响 decode 上界；
  引擎 spec 需固定实际 tiling 并自洽（如 N-tile=512 时 49.5k/token）。
