# Path A 两遍法闸口签核 — 上板实证（Gate ①②③④a/④b/④c 合并记录）

日期：2026-08-13 | 作者：TPU 底层工程师 | 状态：**已签核，上板实测（CEO 四项闸口核对验收通过）** | 关联：`gate_a_check.c`、`rs_noreload_check.c`、`gate1_mrow_check.c`

---

## 0. 签核结论（TL;DR）

- **Gate ①（pass1 int8 回读）签核通过**：标准 INT8 matmul（ps32-free、
  `res_is_int8=1`）+ `tdma_l2g_matrix_copy` + `MemInvld` 回读，在
  K=32/128、N=192、rshift=8/5 下 **全部逐元素精确**（bad=0）。
  含饱和场景（rshift=5，部分值 sat8 到 ±127）仍确定可复现。
- **Gate ②（提交量 / 批量成本）确认通过**：批量 per-op 成本 **3.7μs**（PS32 probe21：
  build ~1.9μs + run ~1.8μs，192/192 正确，cmdbuf ~2000 op/256KB）。A' 实际 tiling
  15,360/token ≈ 57ms/token、保守 per-group 80,544/token ≈ 0.30s/token，对 SD 9.18s/token
  余量 ≥30×（CEO 核对 ≥34×）。**build 摊销为硬前提**：naive per-submit 0.55ms 不可接受，
  必须预建 cmdbuf（详见 §6 提交量口径）。
- **Gate ③（ION/LMEM）无冲突**：ION 布局 A ≈18.9MB < 24MB（余 ~5.1MB）；lmem 界
  KG=32→N≤896（28KB）/ KG=128→N≤224，两遍法 tiling 自由，无 TIU 硬件上限。
- **Gate ④a（同块不重载）签核通过**：pass1(rsafe)/pass2(r_opt) 同一 LMEM 权重块、仅
  rshift_bits 不同、**无 right g2l 重载**，`rs_noreload_check.c` Test A/B 512/512 bad=0
  （pass2 含 408 个 sat8 值仍精确）。
- **Gate ④b（N-tile=512）签核通过**：标准 INT8 两遍法**不受 ps32 N=192
  硬件上限约束**，仅受 lmem 容量约束。K=32 下 **N=512 实测 512/512 正确**
  （right[32,512]=16KB）；K=32 最大 N=896（28KB，N=1024 因对齐越界 alloc fail）。
  KG=128 下最大 N=192（right[128,192]=24KB）。
- **Gate ④c（全 N 列）签核通过**：标准 INT8 matmul 全 N 列逐元素正确（含饱和）——
  K=32/N=512 全 512 列、K=32/N=896 全 896 列（patha_kg32 P0 生产形状 5/5 bad=0）。
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

## 6. ② 提交量口径（CEO 要求合并标注，三个数字勿混用）

**批量 per-op 成本（PS32 probe21，N=192 forced-c1）**：build ~1.9μs + run ~1.8μs
≈ **3.7μs/op**，192/192 正确；cmdbuf 容量 ~2000 op/256KB。**build 摊销为硬前提**——
naive per-submit（register+alloc+convert+load+Run）0.55ms/submit 不可接受
（REPORT_PATHA_KG32 §2.2）；预建 cmdbuf 后 Run+Invld 0.14ms、完整两遍法
0.44ms/block → C 流水 ≈3.4s/token < SD 9.18s（REPORT_SUBMIT_BUDGET §1）。

| tiling 口径 | runs/layer | runs/token | 批量成本 @3.7μs/op | 依据 |
|---|---|---|---|---|
| **A' 实际（引擎口径，2026-08-13 M2 修正）**：KG=32 + N-tile=896，engine per-tile RunCmdbufEx（q/k/v/wo 1 tile、up/gate 6 tile） | **1200** pass runs（q/k/v/wo 224 + up 336 + gate 336 + down 304）+ **600 g2l**（④a no-reload） | **28,800**（pass runs；含 g2l 43,200） | pass-only ~107ms/token（余量 ~86×）；含 g2l ~160ms/token（余量 ~57×） | REPORT_M2_TIU_CORE §3 |
| 设计默认：KG=32 + N-tile=512（k/v 等窄矩阵仍 1 tile） | 2,064 | **≈49,500** | ~183ms/token（余量 ~50×） | DESIGN_PATH_A_TWOPASS §2 |
| 保守：G=32 + N-chunk=192（PS32 per-group 口径，**含 lm_head**） | — | **80,544** | ~0.30s/token（余量 ≥30×，CEO ≥34×） | PS32_PER_GROUP_VERDICT §3 |

> **提交量口径修正（2026-08-13，M2 引擎上板定稿）**：原"A' 实际 640/layer（15,360/token）"
> 为 patha_kg32 P3 探针**合并口径**——单 cmdbuf 内顺序 6×(g2l tile, mm, l2g)。引擎实现按
> **每 tile 一次 RunCmdbufEx**（[32,4864] 右矩阵 155KB > 32KB LMEM 无法整块驻留，须逐 tile
> g2l+mm），真实 **1200 pass runs/layer**（+600 g2l no-reload = 1800 runs/layer）。
> 批量成本：pass-only ~107ms/token、含 g2l ~160ms/token，余量 ≥57×，**仍 SD-bound**。
> M2 上板实测 per-RunCmdbufEx=0.145ms → pass-only 4.17s/token、全 runs 6.26s/token，
> 流水线 max(SD 9.18, ~7.98) → **余量 1.15×**。结论不变：**build 摊销（预建 cmdbuf 池）为硬前提**。

- **各口径均 SD-bound**：批量成本 0.11–0.30s/token ≪ SD 9.18s/token，不影响 decode 上界。
- **勿混用**：各数字对应不同 tiling 与 run 口径（N-tile=896 per-tile / 512 / 192、是否含 g2l、
  是否含 lm_head）。引擎 spec 必须固定实际 tiling 并自洽；引用时必须带口径前缀
  （如"N-tile=896 per-tile 时 28,800 pass runs/token、含 g2l 43,200"）。

## 7. 合并与验收记录（CEO 裁定 2026-08-13）

四项闸口核对验收通过，本记录为合并版（Gate ①②③④a/④b/④c）：
- **② 提交量正式确认**：批量 3.7μs/op，余量 ≥34×，build 摊销为硬前提（并入引擎 IMPL spec）。
- **④a 同块不重载新验证通过**：`rs_noreload_check.c` Test A/B 512/512 bad=0，
  rshift 变更无 g2l 重载（并入引擎 IMPL spec）。
- **④b N-tile=512 签核**（K=32 上限 N=896）、**④c 全 N 列签核**（含饱和）。
- **③ ION/LMEM 无冲突**：ION 布局 A 18.9MB < 24MB；lmem 界见 §2/§0。
- **提交量口径**：见 §6 三口径表，勿混用。
