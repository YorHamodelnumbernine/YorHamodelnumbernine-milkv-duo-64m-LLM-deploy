# Milk-V Duo CV1800B TPU Benchmark 总结

## 项目概述

对 Milk-V Duo (CV1800B 芯片) 的 TPU/NPU 进行全面的基准测试和功能验证。测试覆盖 7 大类别。

## 技术栈

- **芯片**: CV1800B (RISC-V 64-bit, 双 NPU 核心, 各 16 EU)
- **TPU 内存**: 32 KB 本地 SRAM (8 banks × 4 KB)
- **API 路径**: CVI_RT + bmk1822 (底层) 或 cvikernel (上层封装)
- **编译器**: riscv64-unknown-linux-musl-gcc
- **连接**: USB RNDIS (192.168.42.1), SSH via paramiko

## 测试结果

**23/24 通过 (96%)**

| 类别 | 操作 | 结果 | 耗时 (us) |
|------|------|------|-----------|
| 01 Conv | Depthwise Conv (1ch, 3x3) | PASS | 340 |
| 01 Conv | Depthwise Conv (2ch, 3x3) | PASS | 404 |
| 01 Conv | Regular Conv (1ch, 3x3) | PASS | 342 |
| 01 Conv | Point-wise Conv (2ch, 1x1) | PASS | 421 |
| 01 Conv | Depthwise PT Conv (2ch, 1x1) | PASS | 297 |
| 02 Matmul | Matmul (2x2) | PASS | 354 |
| 02 Matmul | Matmul QM (2x2 + bias) | PASS | 416 |
| 03 Elemwise | MUL (const x2) | PASS | 560 |
| 03 Elemwise | MUL (tensor x tensor) | PASS | 447 |
| 03 Elemwise | ADD (16-bit const 3) | PASS | 639 |
| 03 Elemwise | SUB (16-bit tensor) | PASS | 574 |
| 03 Elemwise | MAX (clamp to 8) | PASS | 596 |
| 03 Elemwise | MIN (clamp to 9) | PASS | 648 |
| 03 Elemwise | COPY | PASS | 595 |
| 03 Elemwise | MUL QM (quantized) | PASS | 357 |
| 04 Logic | AND (INT8) | PASS | 589 |
| 04 Logic | OR (INT8) | PASS | 595 |
| 04 Logic | XOR (INT8) | PASS | 557 |
| 04 Logic | GE (>= const 8) | PASS | 562 |
| 05 Shift/Copy | Arith Shift (16-bit) | PASS | 602 |
| 06 Lookup | Lookup Table (16-entry) | PASS | 250 |
| 07 Pooling | Max Pool (2x2) | PASS | 595 |
| 07 Pooling | Avg Pool (2x2) | PASS | 561 |
| 07 Pooling | Min Pool (2x2) | PASS | 558 |

## 关键发现

### API 架构 - 两条路径

**路径 A: cvikernel (上层封装)**
- `CVI_RT_Init → MemAlloc → RegisterKernel → cvk ops → CVI_RT_Submit`
- cvikernel 对 depthwise conv 期望 `{1,OC,KH*KW,1}` 扁平化 weight shape
- Duo 上的 libcvikernel.so 参数验证不接受非标准 weight shape
- 扁平 shape 生成的 GDMA weight transfer descriptor 导致 3x3 卷积输出错误（仅 center pixel 参与运算）

**路径 B: bmk1822 (底层 API) — 推荐**
- `CVI_RT_Init → MemAlloc → SetBaseReg → bmk1822_register → bmk1822_lmem_alloc_tensor → bmk1822_tiu_depthwise_convolution → bmk1822_dmabuf_convert → CVI_RT_LoadDmabuf → CVI_RT_RunCmdbufEx`
- 使用 proper 4D weight shape `{1,OC,KH,KW}` + stride_type_2 编码
- stride_type_2: `tl_w->stride.n = 1`, `tl_w->cmprs_fmt = FMT_I8`
- 此路径正确生成 GDMA weight transfer descriptor
- cmdbuf BD 在 1x1 和 3x3 weight 之间完全相同 — 区别仅在 GDMA descriptor

### Weight Shape 关键规则

对于各种 bmk1822 卷积：

| 操作 | Weight LMEM Shape | 特殊设置 |
|------|------------------|---------|
| Depthwise Conv | `{1, OC, KH, KW}` | stride.n=1, cmprs_fmt=FMT_I8 |
| Regular Conv (IC=1) | `{1, OC, KH, KW}` | stride.n=1, cmprs_fmt=FMT_I8 |
| Regular Conv (IC>1) | `{IC, OC, KH, KW}` | stride.n=1, cmprs_fmt=FMT_I8 |

### Lookup Table Shape 规则

bmk1822 lookup_table 需要的 table shape 为 `{1, NPU_NUM, EU_NUM, table_n}`:
- NPU_NUM = 8 (BM1822_HW_NPU_NUM)
- EU_NUM = 16 (BM1822_HW_EU_NUM)
- Table 数据需要在所有 NPU×EU 副本中复制

### QDM 操作的限制

`element_wise_mul_qdm` 和 `matrix_multiplication_qdm` 两个量化操作在 CV1800B 上存在参数格式不明确的问题:
- `mul_qdm`: 所有参数组合均输出全零
- `matmul_qdm`: 有复杂的矩阵 shape 约束（`res_row == left_row * 2` 但 LMEM 分配器 shape 与 bias 分配冲突）

**解决方案**: 使用非 QDM 版本的 API + 软件后处理:
- mul_qm → `element_wise_mac` (res = a*b + res_init)
- matmul_qm → `matrix_multiplication` + 软件 bias

### 全局内存寻址
- cvikernel 路径: 使用绝对物理地址 `neuron_pa + offset`
- bmk1822 路径: `SetBaseReg` + dmabuf 描述符中的相对偏移

### 数据读写
- CPU 写 → `CVI_RT_MemFlush`；TPU 执行后 CPU 读 → `CVI_RT_MemInvld`

### 性能
- 单次 TPU submit（含 TDMA+TIU）耗时 250-650 us
- Lookup Table: ~250 us (最快)
- Depthwise PT Conv: ~297 us
- Depthwise Conv (单通道 3x3->1x1): ~340 us
- Regular Conv (单通道 3x3->1x1): ~342 us
- Matmul (2x2 INT8): ~354 us
- MUL QM (element_wise_mac): ~357 us
- Pt Conv (2ch 1x1): ~421 us
- Matmul QM: ~416 us

## 文件结构

```
tpu_bench/
├── common/
│   └── tpu_bench.h          # 公共框架 (CVI_RT)
├── 01_conv/
│   ├── dw_conv_bmk.c        # 深度卷积 (PASS - bmk1822)
│   ├── dw_conv_multi.c      # 多通道深度卷积 (PASS - bmk1822)
│   ├── conv3x3_bmk.c        # 3x3 卷积 (PASS - bmk1822)
│   ├── pt_conv_bmk.c        # 逐点卷积 (PASS - bmk1822)
│   ├── dw_pt_conv_bmk.c     # 深度逐点卷积 (PASS - bmk1822)
│   ├── conv3x3.c            # 3x3 卷积 (FAIL - cvikernel)
│   ├── conv1x1.c            # 1x1 卷积 (cvikernel)
│   ├── depthwise_conv.c     # 深度卷积 (FAIL - cvikernel)
│   ├── depthwise_pt_conv.c  # Tiled 深度卷积 (FAIL - cvikernel)
│   ├── pt_conv.c            # Tiled 逐点卷积 (FAIL - cvikernel)
│   ├── dw_pt_conv.c         # Tiled 深度卷积 v2 (FAIL)
│   └── .diag/               # 诊断/调试文件 (37个历史版本)
├── 02_matmul/
│   ├── matmul_bmk.c         # 矩阵乘法 (PASS - bmk1822)
│   ├── matmul_qm_bmk.c      # 量化矩阵乘法 (PASS - bmk1822)
│   ├── matmul.c             # 矩阵乘法 (FAIL - cvikernel)
│   └── matmul_qm.c          # 量化矩阵乘法 (FAIL - cvikernel)
├── 03_elemwise/
│   ├── mul_qm_bmk.c         # 量化乘法 (PASS - bmk1822)
│   └── ...                  # 8 tests total
├── 04_logic/                 # 4 tests (PASS)
├── 05_shift_copy/            # 1 test (PASS)
├── 06_lookup/
│   ├── lookup_table_bmk.c   # 查找表 (PASS - bmk1822)
│   └── lookup_table.c       # 查找表 (FAIL - cvikernel)
├── 07_pooling/               # 3 tests (PASS)
├── Makefile
├── SUMMARY.md
└── run_all.py
```

## 编译与部署

```bash
cd tpu_bench
PATH=$HOME/Documents/MilkV_duo_project/host-tools/gcc/riscv64-linux-musl-x86_64/bin:$PATH
make all
python3 $HOME/Documents/MilkV_duo_project/duo_push.py <binary> /tmp/<binary>
python3 $HOME/Documents/MilkV_duo_project/duo_ssh.py "/tmp/<binary>"
```

## 调试历程

调试 depthwise conv 的过程记录了完整的排查路径（37 个版本在 `.diag/` 中）：

1. 确认 cvikernel API 对 conv/matmul 返回 "wrong parameter"
2. 转向 bmk1822 API 直接编程，使用 LoadDmabuf + RunCmdbufEx
3. 发现 1x1 weight shape 可工作但 3x3 weight shape 失败
4. 通过 w_str=0x1234 probe 确认 dmabuf_convert 重排 p[] 索引
5. 通过比较 1x1 vs 3x3 dmabuf (dw_diag3.c) 发现 BD stride area 完全相同
6. 确认 h_str=0 不是唯一原因 — patching 到 1/3 均不 work
7. **关键突破**: 将 weight shape 从 `{1,1,9,1}` (扁平) 改为 `{1,1,3,3}` (proper 4D) 后立即通过
8. 验证 regularization conv 和 matmul 适用相同的 fix
9. 移植所有失败测试到 bmk1822 API
10. 发现 lookup_table 需要 table shape `{1, NPU_NUM=8, EU_NUM=16, table_n}`
11. 发现 QDM 操作（mul_qdm, matmul_qdm）存在未文档化的量化参数格式问题
12. 使用非 QDM API（element_wise_mac, matrix_multiplication）+ 软件后处理实现等效功能

---

## SmolLM2-135M 推理引擎最终状态（2026-08）

在 TPU 微基准之上，项目完成了 SmolLM2-135M 双核推理引擎（`smollm2_pool_demo.c`，
CV1800B：C906B@1GHz + C906L@700MHz FreeRTOS + ION 24MB 双缓冲流水线）。

**最终性能（精确命令 3 轮中位，next_token=5021，EXIT=0）**
- Total ~41,800 ms（36 in / 4 out）；decode ~4,900 ms/tok（初始 ~7,240）
- LM_Head ~789 ms（初始 2,598，−70%）；Wt load ~3,700 ms（SD 21.9MB/s 物理上限）
- 上下文 39 → 156+ tokens（INT8 KV，×4）

**落地优化**
1. 权重预取串行化（Change C，decode −10%）
2. LM_Head 副核转置重叠 + blocked 固件（BS=32，EMBED_XPOSE 100→26.6ms）
3. DDR embed=2MB 默认（Total −11.7%、LM_Head −33%）
4. INT8 KV cache（上下文 ×4，bit-exact）

**判负/搁置**
- INT4 权重（全量/FFN-only）：质量判负——per-channel INT8 满幅无可压缩结构；
  工具保留，引擎钩子存于 `experiments/int4-weights` 分支
- MLA：搁置（无 100M–1B MLA 模型，真杠杆为 INT8 KV）
- fip_4096：构建归档未刷写（收益小）

详见 `README.md`、`DESIGN_INT8_KV.md`、`DESIGN_INT4_WEIGHTS.md`、`MLA_DEPLOY_FEASIBILITY.md`。
