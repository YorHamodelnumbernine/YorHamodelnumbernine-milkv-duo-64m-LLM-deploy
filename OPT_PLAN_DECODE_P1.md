# Decode 延迟优化 · 第一期 — 静态分析与实施计划

日期：2026-08-08
基线（Duo 实测，36-token prompt，3 decode tokens，3+3 pipeline）：
- Decode: 7,609 ms/token
- Wt load: 2,033 ms | LM_Head: 2,787 ms | FFN_up: 505 ms | FFN_dn: 407 ms | QKV: 264 ms

两项瓶颈合计 4,820 ms / 7,609 ms = **63%**（SD 权重读取 + LM Head）。

---

## 1. 静态代码分析

### 1.1 LM Head 路径（`sm_forward_pool()` ~L1632-1770）

当前实现：
- 48 chunks × 1024 列，每 chunk 一次 `tpu_matmul_build`(1×576 × 576×1024)。
- 每 chunk 内部：读 embed chunk（DDR>ION>SD 三级缓存）→ 转置（mbox 或 CPU）→ matmul → `CVI_RT_Submit` + `CVI_RT_MemInvld` → dequant。
- **嵌入缓存命中率 50%**（3+3 模式：DDR 12MB + ION ~2MB = 14MB / 28MB）。
  未命中部分（~25 chunks，约 14MB/token）从 SD 随机读。

发现的低效点：
1. **next chunk 读取是同步的**（L1711 `LM_CACHE_READ`），SD 读延迟完全暴露在关键路径上。
   已有 `ef_job_t` 异步读线程基础设施（L801-846）但 LM Head 未使用。
2. **每次 matmul 独立 Submit+MemInvld**，48 次。chunk 越大，固定开销摊得越薄。
3. **SD 读块太小**：576KB/chunk（1024 列）。顺序读 SD 实测 16-19MB/s，
   小块随机读效率低。
4. **转置后矩阵在 ION**：`tpu_matmul_build` 中 `r_is_nm=false`，每 tile 都要把
   右矩阵从 ION `memcpy` 进 neuron memory + `MemFlush`（768 tiles × 36KB = 28MB
   拷贝 + 768 次全 1MB MemFlush）。这个开销随 CHUNK 变化不大（总 tile 数不变）。

### 1.2 SD 权重读取路径（`sm_forward_pool()` 3+3 双缓冲）

- 每 batch 用 **3 个并发 pthread** 分别读 3 个 layer 文件（`pf_worker`）。
- 实测并发读 8-10MB/s vs 顺序读 16-19MB/s —— **并发读比顺序读慢 ~2 倍**。
- decode 单 token 时每 batch TPU 计算仅 ~140ms，SD 预取 ~560-1100ms，
  流水线被 SD 读主导。

## 2. 优化方案（按收益/风险排序）

### 改动 A（低风险，高收益）：LM Head CHUNK 1024 → 4096（动态，受 ION 空间约束）
- 48 submits → 12 submits，固定开销 /4。
- SD 读块 576KB → 2.36MB，顺序读带宽更好。
- 转置调用 48 → 12（总转置工作量不变，但函数调用/命令开销 /4）。
- 内存安全：xpose 4 缓冲 = 4×576×4096 = 9.4MB < weight 槽区
  （3+3=21.3MB, 2+2=14.2MB）。1+1 模式自动降到 2048。
- 预期收益：LM_Head 省 ~500-900ms。

### 改动 B（低风险，高收益）：LM Head next-chunk 异步 SD 预取
- 用已有 `ef_job_t` 把 SD chunk 读放入后台线程，与当前 chunk 的
  转置+matmul+dequant 重叠。
- 预期收益：LM_Head 中 SD 读延迟基本隐藏，省 ~400-800ms。
- 风险：与 mbox 转置并发（副核 DMA vs 主核 SD 读）互不冲突；缓存命中 chunk 仍同步 memcpy。

### 改动 C（中风险，后续期）：权重预取 3 并发 → 单线程顺序读
- 3 个并发 layer 读合并为 1 个线程顺序读 3 个文件，SD 吞吐 ~2 倍。
- 需重排 `pf_worker`/jobs 结构，涉及 pipeline 同步，风险中等。
- 预期收益：Wt load 2000ms → 1000-1400ms（若读确实被暴露）。

### 改动 D（后续期）：LM Head matmul 右矩阵直接放 neuron memory（r_is_nm=true）
- 消除 768 次 tile 拷贝 + 768 次 MemFlush。
- CHUNK 需 ≤1024（转置块 589KB 才能进 1MB neuron mem），与改动 A 冲突，
  需要实测对比确定哪个更优。

## 3. 本期实施范围

- 实施改动 A + B（1-2 个低风险高收益改动）。
- 实测改动前后 decode 耗时对比。
- 改动 C/D 作为下一期建议。

## 4. 预期合计

LM_Head 2,787ms → ~1,200-1,800ms（省 ~1-1.6s）。
Decode 7,609ms → ~6,000-6,600ms/token（-13%~-21%）。

---

## 5. 实测结果（2026-08-08，Duo 上机验证）

已实施改动 A（CHUNK 动态放大）+ B（next-chunk 异步 SD 预取），编译并推送至 Duo，
用与基线相同的命令实测（36-token prompt、max_new=3、force_mode=3、eos=2）：

```
[LM_Head] CHUNK=2048 (24 chunks, 1.12 MB/chunk, weight_region=20763 KB)
[LM_Head] 13/24 chunks from SD
Prefill: 25482 ms
Decode:  27617 ms (4 tok, 6904 ms/tok)
Per-step avg (n=7): Wt load 2118 | QKV 279 | FFN_up 510 | FFN_dn 400 | LM_Head 2201
Tokens: 5021 1308 6990 154 (logit gap 1.0-5.8, 无自循环)
```

### 对比基线

| 指标 | 基线 | 实测 | 变化 |
|------|------|------|------|
| Decode | 7,609 ms/tok | 6,904 ms/tok | **-9.3%** |
| LM_Head | 2,787 ms | 2,201 ms | **-586ms (-21%)** |
| Wt load | 2,033 ms | 2,118 ms | +85ms（SD 抖动） |
| QKV | 264 ms | 279 ms | ~持平 |
| FFN_up / FFN_dn | 505/407 ms | 510/400 ms | ~持平 |

### 结论

- 改动 B（异步预取）完整生效；改动 A 因 mbox 副核 EMBED_XPOSE 上限 cur_v≤2048，
  CHUNK 被限制在 2048（而非 4096），故只获得部分收益。
- LM_Head 2,787→2,201ms（-21%），符合改动 B 的预期（400-800ms），低于 A+B 合计预期
  （1.2-1.7s），主要受 mbox 2048 上限约束。
- Wt load 未改动，仍为 decode 第一大瓶颈（2,118ms / 6,904ms = 31%）。

### 下一期（Phase 2）候选

1. **突破 mbox EMBED_XPOSE cur_v≤2048**（TPU 底层）：改 comm_main.c 副核限制，
   或 A/B 测试关闭 mbox 走 CPU 转置 + CHUNK=4096，量化 CHUNK=4096 的边际收益。
2. ~~**改动 C**~~（推理引擎）→ **已实施并实测通过，见 §6**
3. **改动 D**（TPU 底层）：tpu_matmul_build r_is_nm=true，消除 768 次 tile 拷贝 + MemFlush。

---

## 6. 改动 C 实测（2026-08-08，推理引擎工程师，同日交错 A/B）

实现：`pf_job_t` 支持整批 n 个连续 layer 顺序读，`sm_forward_pool` 由
`jobs[batch_slots]`（3 并发 pthread）改为单 `pf_batch` + `pf_inflight`，
保持 3+3 双缓冲同步语义。交叉编译通过，MD5 校验后推送 Duo。

### 实测数据（同条件，交错 A/B 控制噪声）

| 指标 | 并发读（基线） | 单线程顺序读（改动 C） | 变化 |
|------|--------------|----------------------|------|
| SD 读吞吐 | 7.88 MB/s | 13.06 MB/s | **1.66×** |
| Decode | 34,231/34,887/35,015 ms | 31,081/31,166/31,344 ms | **~10%** |
| Prefill | — | — | ~3.6% |
| 输出质量 | — | tokens 正常、无自循环、logit gap 2.1-5.8 | 无损 |

### 关键结论

- **诊断显示常规状态下预取已被 TPU 计算完全隐藏**（Wt load≈530ms 纯初始加载），
  收益并非来自减少暴露的 SD 读取，而是**消除了 3 个并发读线程对主核计算的
  调度/中断干扰**。
- 设备噪声大（±15%），故用交错 A/B 保证结论可靠。
- 注：本组绝对数值（~8.5s/tok 基线）与 §5 的 6,904ms/tok 存在差异，因测试
  条件不同（prompt 长度影响 3+3 vs 2+2 pipeline 模式），相对增益以交错 A/B 为准。

### 与 CEO 实测的 CHUNK=4096 A/B 交叉验证

- CEO 侧实验（禁 mbox 走 CPU 转置 + CHUNK=4096）：LM_Head 2,201→6,615ms，decode
  →13,109ms/tok，**严重回退**。确认副核 mbox EMBED_XPOSE 转置卸载对 LM_Head 至关重要。
- 因此 CHUNK=4096 的收益只能通过**突破副核 cur_v≤2048 上限**获得（Phase 2 候选 1），
  不能靠关闭 mbox 用 CPU 转置换取。

### Phase 2 综合状态

- 改动 A+B（LM_Head CHUNK 放大 + 异步预取）：CEO 已验证，decode -9.3%。
- 改动 C（权重预取串行化）：推理引擎工程师已验证，decode ~-10%。
- 待办：突破 mbox 2048 上限（TPU 底层）、改动 D（r_is_nm=true）。

---

## 7. 干净联合复测（2026-08-08，CEO，交错 A/B ×3，设备空闲）

在独立 worktree 构建原始基线（bb7e0cb，A/B/C 之前），与 A+B+C 交错各跑 3 次，
同命令 `/root/smollm2_pool_demo{,base} /root/smollm2_instruct/ /root/input_tokens.bin 3 3 2`：

| Run | 基线 Decode | A+B+C Decode |
|-----|------------|--------------|
| 1 | 37,642 ms | 31,320 ms |
| 2 | 38,638 ms | 31,203 ms |
| 3 | 38,463 ms | 31,340 ms |
| **平均** | **38,248 ms (9,562 ms/tok)** | **31,288 ms (7,822 ms/tok)** |

- **总提速 -18.2%**（基线内波动 ±1.3%，A+B+C 内波动 ±0.3%，结论可靠）。
- LM_Head：基线 ~3,275ms → A+B+C ~2,559ms（**-22%**）。
- 与预估一致：C(~10%) 与 A+B(~9%) 叠加 ≈ 1 − 0.9×0.82 ≈ 18%。
- 注：绝对数值高于 §5 历史基线（7,609/6,904ms/tok），为设备漂移（设备连续运行 7d22h、
  load 常态 3.0），**相对提升以本组交错 A/B 为准**。

---

## 8. TPU 底层工程师 Phase 2 调查结论（2026-08-08）

### 8.1 真正瓶颈：副核 EMBED_XPOSE 转置本身（~11.5MB/s）

- 微基准拆分（`bench_lmhead_cost.c`，Duo 实测）：LM_Head 每 chunk ~112ms 中，
  **转置占 80%**（mbox CHUNK=2048 ≈ 100-106ms，带宽线性受限 11.5MB/s）。
- 副核转置是 naive 双重跨 64B cache line 循环（comm_main.c L320-322），缓存极不友好。
- 大核 blocked(BS=32) 转置实测 57ms vs naive 500ms（**~8× 加速**），可移植到副核。

### 8.2 `cur_v≤2048` 来源与突破

- **纯软件校验**：comm_main.c L310-311 `if (... || D>2048 || cur_v>2048)`。
  非硬件/DMA/内存限制（rows/cols 为 uint32、size 2.36MB 无溢出、cache 操作为 64B 行）。
  参考同文件 CMD_MHA_TRANSPOSE 已允许 >4096。
- 突破 4096 可行：comm_main.c L311 + smollm2_pool_demo.c L1662 各 1 行。
  ION 9.4MB 在 2+2/3+3 下安全，1+1 自动降档。

### 8.3 收益修正

| 项 | 原估计 | 修正后 | 原因 |
|----|--------|--------|------|
| 改动 A (CHUNK 4096) | 500-900ms | **~100-300ms** | 转置带宽受限，总工作量 V×D 恒定 |
| 改动 D (r_is_nm=true) | 省 0.65-0.9s | **单独≈0** | 与改动 A 冲突（强制 CHUNK≤1024），省拷贝被转置次数翻倍吃掉 |

### 8.4 推荐路线（Phase 3，按收益/风险排序）

| 序 | 优化 | 改动点 | 预估收益 | 风险 |
|----|------|--------|---------|------|
| ① | **blocked 副核转置** | comm_main.c EMBED_XPOSE 改 BS=16/32 分块 | 转置 2.4s→0.4-0.8s（省 **1.6-2.0s**） | 中（固件，改动局部） |
| ② | **转置与 matmul 重叠** | demo 侧 async send + 延后 wait（现有双缓冲） | 省 ~0.7s | 低 |
| ③ | 改动 A (CHUNK 4096) | comm_main.c + demo 各 1 行 | ~0.1-0.3s | 低 |
| ④ | 改动 D (r_is_nm=true) | demo 侧，CHUNK≤1024 | 单独~0；叠 ① 后 +0.65s | 低-中 |
| ⑤ | TPU TDMA 硬件转置 | 研究 `tdma_g2l_matrix_copy_row_col_transposed` | 潜在消灭转置 | 高 |

- **结论**：下一步优先 ①（blocked 副核转置，固件）+ ②（转置/matmul 重叠，demo）。
  改动 A 顺手做（便宜），改动 D 只在 ① 完成后评估。
- 已提交 `bench_lmhead_cost.c` 微基准工具（`make bench_lmhead_cost`）供后续验证。
