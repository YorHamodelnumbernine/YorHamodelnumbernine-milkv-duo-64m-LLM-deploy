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
- **TIU 舍入语义补订**：INT8 matmul rshift>0 时为 **round-half-up**
  （11440>>8 → 45，非截断 44）。CPU 侧反量化必须匹配此语义，否则两遍法
  pass2 结果会系统性偏差 1。

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
