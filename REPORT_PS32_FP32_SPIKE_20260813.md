# ps32_mode==2 fp32 输出决定性 spike — CV1800B 上板实证（A/B 一锤定音）

日期：2026-08-13 | 作者：TPU 底层工程师 | 状态：**决定性结论，上板实测**
目的：裁决 CEO 最高优先级问题 —— ps32_mode=1/2 上板实测（127/236）与
推理引擎工程师 API 层分析（cvikernel.h / bm1880v2.h 注释 ps32_mode==2→fp32）的直接冲突。

---

## 0. 结论（TL;DR）

**CV1800B（bmk1822）ps32_mode==2 不输出 fp32，且结构上不可能输出 fp32。这是硬件限制，
不是配置/回读问题。**

- ① **fp32 不可读**：ps32_md 寄存器已确认正确置 2（cmdbuf 逐字节验证），
  res 用 BF16 分配 + `res_is_int8=0` → 输出 int8 饱和值复制（0x7f），整个 32KB 可读区
  扫描 fp32(320000.0)=0 命中；res 用 F32 分配 → matmul 算子本身断言拒绝。
- ② **element_wise 不支持 F32**：F32 张量分配被 `check_tiu_tensor` 断言拒绝
  （仅允许 I8/U8/BF16）；BF16 mul/mac 算术产生垃圾（mul 2/4 正确、mac 全错）；
  仅 I8 可靠。**「int32 × fp32 scale → fp32 累加」的 LMEM 闭环不可实现**。
- ③ **Qwen 实测 submit 数**：每层 2432（Wq140/Wk28/Wv28/Wo20/up728/gate728/down760），
  ×24=58,368；lm_head 22,176；**合计 ~80,544/token**。per-op build ~1.7-1.9μs、
  run ~1.8μs（K=32, N=192 批测）。
- ④ **根因定性**：**硬件限制**。ps32_md=2 寄存器已正确配置，但 bm1822 TIU 无 fp32
  累加/输出通路；「fp32 if ps32_mode==2」注释只存在于 bm1880v2.h 与 cvikernel.h
  （BM1880 家族继承），**bmk1822.h 无此注释**，CV1800B 不实现该语义。

**裁决映射**：变体1（纯 TPU fp32 累加）**不可用**；int32 部分和可读 → **变体2
（CPU 回环 fp32 累加）**为 ps32 路径唯一可行形态，但 submit 更多、需重构，
仍被 Path A 两遍法（ps32-free, 49.5k submit）支配。

---

## 1. 决定性实验设计（ps32_dec_spike.c）

- 形状：K=32, M=1, N=32, left=right=100 → host 部分和 = 320000 (0x0004E200)。
- 手法：forced-c1 全列形状（probe18 已证可行）+ ps32_matrix 分配 +
  **tdma_l2g_general_copy 写 neuron（与 int8 回读同一 DMA 路径）+ CPU MemInvld 回读**，
  读回完整 32KB lmem 区。
- 解码：int32 连续 / int32 byte-plane / fp32 连续 / fp32 byte-plane / bf16 连续，
  外加**全 32KB 扫描** int32(00 E2 04 00)、fp32(00 40 9C 48)、bf16(9C 48) 特征字节。
- 附带：cmdbuf 反汇编验证 ps32_md 寄存器实际写入值；fork 隔离断言型配置。

## 2. 上板实测证据

| Case | ps32_mode | res_is_int8 | res 格式 | ps32_md 寄存器 | 输出 | 判定 |
|---|---|---|---|---|---|---|
| C0_ctrl | 0 | 1 | I8 | 0 | 0x7f (127) | int8 饱和 ✓ 对照 |
| C1 | 1 | 0 | BF16 | **1** | 全 0x7f | 无 fp32 / 无 int32 |
| C2 | 2 | 0 | BF16 | **2** | 全 0x7f | **无 fp32** |
| C3 | 2 | 1 | I8 | **2** | int32_bp=32/32 = 320000 | **int32 部分和可读** |
| C6 | 1 | 1 | I8 | 1 | 全 0x7f | 非 int32（仅 mode=2 出 int32） |
| C7 | 2 | 0 | **F32** | 2 | **matmul 算子断言 abort** | F32 res 被拒绝 |

- **ps32_md 寄存器验证**：cmdbuf 中 TIU 描述符 len=112、tsk=2、ps32_md = ps32_mode 值
  （0/1/2 全部逐字节确认）→ **配置正确，排除配置错误**。
- **pattern 扫描**（C1/C2/C3，全 32KB）：fp32(320000.0) = **0 命中**；
  bf16(0x489C) = 0 命中；仅 C3 的 int32 byte-plane 布局命中（00/E2/04/00）。
- 原始 ps32_test（CEO 引用数据点）复现：`res_is_int8=0` 路径 C1/C2 输出 127（0x7f）、
  B1/C3 输出 237（0xED 垃圾，预期 50）——**均非 fp32**，与决定性 probe 一致。

## 3. element_wise F32/BF16 实测

| op | 格式 | 结果 |
|---|---|---|
| mul | I8 | 2,4,6,8 ✓（对照） |
| mul | BF16 | 3,5,垃圾,垃圾（2/4）→ BF16 算术**损坏** |
| mul | F32 | **分配断言 abort**（check_tiu_tensor 仅允 I8/U8/BF16） |
| mac | I8 | 需 16-bit 高位/低位对，输出饱和（本测数据布局非标准） |
| mac | BF16 | 全 3.0（预期 16/26/36/46）→ **损坏** |
| mac | F32 | **分配断言 abort** |

- bmk1822 头文件：add/sub「must all be 16-bit」、mac「res 必须 16-bit（res_high 非空）」、
  rshift/lshift int 语义。**全 API 无 F32/BF16 算术变体**（仅
  `bmk1822_tiu_bf16_element_wise_ge` 比较与 bf16 pooling 存在）。
- 分配器只允许 I8/U8/BF16 → F32 连张量都建不出来。

## 4. Qwen submit 实测数（K=32/组, N=192/次硬件上限）

| 矩阵 | 形状 [K,N] | K 组 (K/32) | N 块 (⌈N/192⌉) | **submit/层** |
|---|---|---|---|---|
| Wq | [896,896] | 28 | 5 | **140** |
| Wk | [896,128] | 28 | 1 | **28** |
| Wv | [896,128] | 28 | 1 | **28** |
| Wo | [128,896] | 4 | 5 | **20** |
| up | [896,4864] | 28 | 26 | **728** |
| gate | [896,4864] | 28 | 26 | **728** |
| down | [4864,896] | 152 | 5 | **760** |
| **层小计** | | | | **2432** |
| ×24 层 | | | | **58,368** |
| lm_head | [896,151936] | 28 | 792 | **22,176** |
| **总计/token** | | | | **80,544** |

批测成本（K=32, N=192 chunk）：build **1.7-1.9μs/op**（稳态 ~1.8）、run **~1.8μs/op**、
cmdbuf ~112B/op。
→ 80,544 × 3.6μs ≈ **290ms/token**（TIU+cmdbuf build），与既有裁定一致。

## 5. 根因定性：硬件限制（非配置/回读）

1. **寄存器正确**：ps32_md=2 已由 cmdbuf 反汇编逐字节确认 → 非配置错误。
2. **回读路径正确**：tdma_l2g general_copy + MemInvld 与 int8 回读同路径，
   C3 的 int32 部分和已在该路径逐字节精确 → 非回读问题。
3. **F32 结构性不可达**：res 分配 F32 → matmul 算子 `check_tiu_tensor` 断言；
   res 分配 BF16 → 输出 int8 饱和复制而非 fp32；全 32KB 无 fp32 特征位。
4. **SDK 证据链**：「output fp32 if ps32_mode==2」注释仅存在于
   `bmk1880v2.h:868`（matrix_multiplication）与 `cvikernel.h:754`
   （depthwise_pt_convolution），属 **BM1880 家族 API 继承**；**bmk1822.h 无此注释**。
   CV1800B（bm1822）TIU 不实现 fp32 累加/输出，ps32_md=2 落回 int8 饱和路径。
5. 交叉印证：element_wise 全系无 F32/BF16 算术（F32 分配拒绝、BF16 算术损坏），
   与「TIU 无 fp datapath」一致。

## 6. 对 CEO 决策的裁决

- **变体1（纯 TPU fp32 累加）→ 判负**。ps32_mode==2 不产 fp32；element_wise 无 fp32
  算术；matmul add_result 仍被 `assert(!add_result)` 禁止 → LMEM 内
  `acc_fp32 += partial_int32*scale` 闭环**结构上不存在**。
- **变体2（CPU 回环）→ 可行**。ps32_mode=2 + res_is_int8=1 + ps32_matrix 可精确导出
  int32 部分和（N=192/次、byte-plane 布局，probe18/19/20/21 已证），CPU 侧 fp32
  scale + 累加。成本：80,544 submit × ~3.6μs ≈ 290ms + int32 回读 + CPU 累加
  ≈ **+0.5-1s/token**（与 CEO 预估一致）。
- **路径选择不变**：Path A 两遍法（ps32-free，49.5k submit、int8 回读 1/4 数据量）
  仍支配 ps32 变体2。本 spike 不改变主路径裁定，但**彻底关闭「fp32 白嫖」幻想**，
  变体2 作为文档化 fallback 的成本边界已实测锚定。

## 7. 产物

- `ps32_dec_spike.c`（本次决定性探针，fork 隔离 + cmdbuf 反汇编 + 布局解码 + 批测）
- `REPORT_PS32_FP32_SPIKE_20260813.md`（本报告）
- 关联：`PS32_PER_GROUP_VERDICT_20260813.md` §9 补充裁定与本报告一致。
