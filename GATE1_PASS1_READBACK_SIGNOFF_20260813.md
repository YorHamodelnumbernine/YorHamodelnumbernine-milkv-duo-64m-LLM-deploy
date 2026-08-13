# 闸口① 正式签核 — pass1 INT8 matmul 输出回读 DDR（含 ④b N-tile=512 复核）

日期：2026-08-13 | 作者：TPU 底层工程师 | 状态：**正式签核，全部上板实测** | 关联：`gate1_mrow_check.c`（含 fresh run rc=0）、`gate_a_check.c`、`rshift_check.c`、`rs_noreload_check.c`、`GATE_A_SIGNOFF_20260813.md`

---

## 0. 签核结论（TL;DR，供归档）

**闸口① —— PASS（正式签核）**。标准 bmk1822 `matrix_multiplication` INT8 输出
（ps32-free、`res_is_int8=1`）经 `tdma_l2g_matrix_copy` + `MemInvld` **可逐字节精确回读
DDR**。两遍法 pass1 以安全 rshift（rsafe）输出 INT8 部分和，CPU 可直接读真实 per-chunk max
并据此精化 r_opt，供 pass2 满幅重算。回读路径与既有 SmolLM2 引擎 23/24 benchmark 完全同构。

**闸口④b —— PASS（复核确认）**。N-tile=512 的标准 INT8 matmul `[M,32]×[32,512]` 在
LMEM 32KB 内可容纳：**M=1（decode）16.5KB、M=16 24.5KB 均实测正确**（512/512 bad=0）。
M=32 需 33KB 超限 → prefill 大 M 按 M-tile（M=16）或 N-tile（256）切分即可，无 TIU 侧新限制。

---

## 1. 闸口①：pass1 int8 回读（正式签核内容）

### 1.1 断言

两遍法引擎中，pass1 使用 `matmul_rshift_w(K,wmax)-3`（clamp≥4）的安全 rshift，TIU 输出
INT8 部分和 `p1 = sat8(round((acc + (1<<(rshift-1))) >> rshift))`，随后：
`tdma_l2g_matrix_copy` 写 DDR → `CVI_RT_MemInvld` → CPU 逐元素读真实 max。此链路必须逐字节精确，
否则 r_opt 精化（闸口①核心）失效。

### 1.2 上板实测证据（2026-08-13 当日 fresh run，rc=0 ALL PASS）

`gate1_mrow_check.c` 完全复刻 `qwen_kal_ref.c::chunk_matmul_twopass` 语义：pass1 回读 →
CPU per-chunk max → `est=maxabs<<rsafe` → `r_opt=ceil(log2(est/127))` → pass2 同 LMEM 不重载重算。

| M | K | N | lmem(线性 B) | P1 bad | maxabs | r_opt | P2 bad | sat8 | fp32maxdiff | 判定 |
|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 32 | 512 | 25088 (24.5KB) | **0/8192** | 128* | 10 | **0/8192** | 0 | 1.85e-06 | ✓ ④b N=512+M>1 |
| 32 | 32 | 256 | 17408 (17KB) | **0/8192** | 128* | 10 | **0/8192** | 0 | 1.12e-06 | ✓ 更大 M |
| 64 | 32 | 128 | 14336 (14KB) | **0/8192** | 127 | 9 | **0/8192** | 2 | 1.75e-06 | ✓ MAX_SEQ=64 |
| 1  | 32 | 512 | 16928 (16.5KB) | **0/512** | 107 | 9 | **0/512** | 0 | 1.10e-07 | ✓ ④b decode 复核 |

\* maxabs=128 表示 pass1 实测含 -128（rsafe=9 在 |x|,|w|≤100 最坏数据下饱和），回读仍逐元素
精确 → **饱和路径确定性**。P2 数值累加 `p2×(1<<r_opt)×gsc[n]×sc_row[m]` 与 fp64 金标最大偏差
~1.1e-07..1.85e-06（fp32 舍入级），非量化损失。

补充证据（同一结论，不同侧重）：
- `gate_a_check.c`：K=32/128、N=192、rshift=8/5 全 bad=0（含饱和 rshift=5）——KG=128 主档。
- `rshift_check.c`：rshift=1..4 含负数半值，round-half-up 语义 6 组 bad=0。
- `rs_noreload_check.c`：pass1/pass2 同 LMEM 权重块、仅 rshift_bits 不同、不重载，512/512 bad=0。
- 推理引擎工程师 M1 里程碑（`REPORT_M1_PASS1_20260813.md`）：真实 Qwen layer-0 权重块 5/5
  P1/P2 逐元素 bad=0，fp32maxdiff ~1e-9..2e-8。

### 1.3 三条硬性实现约束（引擎必须沿用，均为上板实证）

1. **线性 FORCED 布局**：`{n, c=1, w=全宽, col=全宽}`。`bmk1822_matrix_lmem_default_shape`
   返回 c-split 布局（如 `r{n=32,c=8,w=64,col=512}`），`l2g_matrix_copy` 无法解交织 → 全错。
   线性布局 + l2g_matrix_copy 是**唯一已验证**的回读路径。
2. **TIU 舍入语义 = round-half-up**（rshift>0）：`sat8((acc + (1<<(rshift-1))) >> rshift)`，
   signed 值向 +∞ 舍入（11440>>8 → 45 非 44）。**CPU 反量化必须匹配此语义**，否则系统性偏差 1。
3. **MemInvld 限定输出区**：pass1 回读前仅 Invld P1 区（~8KB），禁止全 buffer 无效化。

---

## 2. 闸口④b：N-tile=512 LMEM 预算（复核确认）

`[M,32]×[32,512]` 三块线性矩阵 lmem 占用（实测）：

| M | left [M,32] | right [32,512] | res [M,512] | 合计 | 判定 |
|---|---|---|---|---|---|
| 1  | 32 B | 16 KB | 512 B | **16.5 KB** | ✓ decode 主路径 |
| 16 | 512 B | 16 KB | 8 KB | **24.5 KB** | ✓ prefill 可用 |
| 32 | 1 KB | 16 KB | 16 KB | **33 KB** | ✗ 超限 → 切 M=16 或 N=256 |

- N-tile=512 上限由 **lmem 容量**决定，非 TIU 硬件（标准 INT8 matmul 不受 ps32 per-group 路径
  的 N=192 上限约束）。
- 设计默认 M≤10（`DESIGN_PATH_A_TWOPASS` §4）→ N-tile=512 在 decode 与 prefill 均安全。
- 对齐注记：right[K,N] 恰 32KB（K=32,N=1024）仍 alloc fail，因矩阵 lmem 含 bank 对齐/元数据；
  实际可用上限 ~31KB。KG=128 时最大 N=192（24KB）。

---

## 3. 交付判定

- 闸口①（pass1 int8 回读供 CPU 读真实 per-chunk max）：**签核通过**。
- 闸口④b（N-tile=512 `[M,32]×[32,512]` LMEM 32KB 可容纳）：**签核通过**（M≤16；M>16 切 tile）。
- 附带确认：④a（pass1/pass2 同 LMEM 权重块、per-call rshift_bits、不重载）此前已签核。
- **推理引擎工程师可全面进入上板实现。** 唯一实现侧需锁定的是 §1.3 三条约束与 N-tile/M-tile
  选择（M>16 场景）。

## 4. 产物与复现

- 探针：`gate1_mrow_check.c`（本签核主证据）、`gate_a_check.c`、`rshift_check.c`、
  `rs_noreload_check.c`（均在 master 提交）。
- 复现：`make gate1_mrow_check && duo_run.py gate1_mrow_check` → `rc=0 (ALL PASS)`。
