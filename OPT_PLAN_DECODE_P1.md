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

---

## 9. 改动 ② 实测（2026-08-08，CEO 实施，转置/matmul 重叠 · Change B2）

### 9.1 实现（demo 侧，无固件改动）

把 LM_Head 的 mbox EMBED_XPOSE 转置从「同步 start+立即 poll」改为异步软件流水，
迭代 i 内：
- (a) 确认 chunk i+1 的 SD 读完成 → 异步启动 mbox 转置(i+1) 写入 xpose_dst[nxt]
- (b) 启动 chunk i+2 的 SD 读（复用已释放的 xpose_src[cur]）
- (c) matmul(i) + dequant(i)（读 xpose_dst[cur]）
- (d) poll 转置(i+1)

转置(i+1) 因此与 matmul(i)+dequant(i) 重叠，隐藏 ~100ms/chunk 的 EMBED_XPOSE
延迟（带宽受限 11.5MB/s）。安全性：LM_Head matmul 的 neuron scratch 用量
（M=1,K=576,tile_n=96 → 最高 ~56KB；prefill M≤36 → ~57KB）始终低于
MHA_OFF_DMA_DESC=0x1F800，in-flight 描述符不会被 matmul 覆盖；mbox slot 0
不与 DDR_TO_ION 并发。

实现中发现并修复一个 off-by-one：step (b) 原写 `v2=nxt_v_start`（=chunk i+1），
应为 `v2=nxt_v_start+CHUNK`（=chunk i+2）。该 bug 使 matmul(i) 读到转置(i-1)
的数据（chunk≥2 全部错误），并跳过 chunk 23 的 SD 读（SD chunk 计数 13→12），
表现为 prefill next_token 从 5021 漂移到 1582。修复后 prefill next_token 与
SD chunk 计数均与基线一致。

### 9.2 实测（Duo，与 §7 相同命令 `/root/smollm2_pool_{base,b2} ... 3 3 2`，
交错 A/B ×3，设备常态 load 3.0）

| Run | 基线 Decode | B2 Decode | 基线 LM_Head | B2 LM_Head |
|-----|------------|-----------|--------------|-----------|
| 1 | 31,416 ms | 27,273 ms | 2,586 ms | 1,833 ms |
| 2 | 31,151 ms | 27,402 ms | 2,543 ms | 1,831 ms |
| 3 | 31,310 ms | 27,305 ms | 2,537 ms | 1,825 ms |
| **平均** | **31,292 ms (7,823/tok)** | **27,327 ms (6,832/tok)** | **2,555 ms** | **1,830 ms** |

- **Decode -12.7%**；**LM_Head -725ms (-28.4%)**，与 §8.4 预估 ~0.7s 一致。
- 基线 7,823 ms/tok 与 §7 干净复测的 7,822 ms/tok 几乎完全一致；B2 组内波动
  ±0.3%，结论可靠。
- 注：观测到 **transformer 本身的非确定性（预存在，与 B2 无关）**：基线自身
  两次运行在第 3 个 decode token 处分歧（...6990 1291 vs ...6990 9391），
  与 Change C 的权重预取/mbox 时序相关。A/B 以 decode 耗时为准。

### 9.3 与 §8 路线对照

- ②（转置/matmul 重叠，demo）✅ 完成并验证：LM_Head 2,555→1,830ms。
- ①（blocked 副核转置，固件）→ 烧写路径评估完成，见 §10。
- ③ ④ ⑤ 维持 §8.4 结论（④ 仅在 ① 完成后评估）。

---

## 10. Phase 3 · 优化①（blocked 副核转置）烧写路径评估（2026-08-08，TPU 底层工程师）

### 10.1 烧写/部署路径（已核实，可逆）

- **产物链**：`comm_main.c` → `freertos/cvitek/install/bin/cvirtos.bin`（=BLCP_2ND）
  → `fsbl/plat/cv180x/fiptool.py genfip` → `fip.bin` → 设备 `/boot/fip.bin`
  （vfat, mmcblk0p1）→ FSBL 把 BLCP_2ND 加载到 `0x83f40000` 运行（C906L@700MHz）。
- **重建**：`freertos/cvitek` 下 `export MV_BOARD=milkv-duo-sd; ./build_cv180x.sh`
  （增量 2-5 分钟；必须保持 `DDR_64MB_SIZE=y` 与现有 build 目录，切 128MB 会起不来）。
- **打包（已实测 byte-exact）**：
  `python3 fsbl/plat/cv180x/fiptool.py genfip --OLD_FIP=<当前 fip> --BLCP_2ND=<新 cvirtos.bin> --BLCP_2ND_RUNADDR=0x83f40000 --MONITOR_RUNADDR=0x80000000 --compress=lzma /tmp/fip_blocked.bin`
  用未改的 cvirtos 重打包，输出与当前 fip.bin sha256 完全一致 → 只替换 RTOS 组件。
- **部署**：备份 `/boot/fip.bin` → `cat > /boot/fip.bin` → `sync` → `reboot`。
- **回滚**：设备已有多个 `.bak`；最稳是拔 SD 卡在宿主机重写 `/boot/fip.bin`
  （vfat，永远可用）；串口 U-Boot `fatload mmc 0:1` 次之；maskrom USB 为终极手段。
- **风险**：**LOW-MEDIUM**（只动 BLCP_2ND 一个 case 的内层循环，不碰
  BL2/u-boot/opensbi/DDR 参数；最坏启动 hang/转置错误，均可经 SD 卡回滚）。

### 10.2 关键实测数据（TPU 工程师，Duo）

| 转置方式 | @2048 实测 |
|---|---|
| 副核 mbox naive（当前） | 100.3 ms |
| 大核 blocked BS=32（cacheable 缓冲） | 57.6 ms |
| 副核 mbox + `SEND_WAIT` 校验 | 100.3 ms（bad=0） |

### 10.3 CEO 决策：走固件路径，不做大核零烧写方案

以 B2 实际 LM_Head=1,830ms ≈ 24 chunk × 76ms/chunk（转置即关键路径）推演：

| 方案 | 大核 | 副核 | 每 chunk |
|---|---|---|---|
| 现状 mbox B2 | build+dequant ≈40ms | 转置 ≈76ms（重叠） | **≈76ms** |
| 大核 blocked（零烧写） | build+转置+dequant 串行 ≈30+58+10 | — | **≈98ms（回归 ~530ms）** |
| 副核 blocked（固件①） | build+dequant ≈40ms | 转置 ≈55-65ms（重叠） | **≈55-65ms（省 ~250-500ms）** |

大核方案把转置搬上大核、与 build/dequant 串行化，预计**回归**，故不实施；
直接走固件①，预期 LM_Head 1,830 → ~1,300-1,500ms。

### 10.4 下一步

- TPU 工程师实现 `comm_main.c` blocked 转置（BS=32，输出布局/描述符语义不变，
  diff 见其 §3.1）+ 重建 FreeRTOS + genfip 打包 + 备份准备（**不烧写**）。
- CEO 确认后：备份 `/boot/fip.bin` → 部署 → `bench_lmhead_cost` 验证
  （预期 @2048 从 100ms → 55-75ms，bad=0）→ `smollm2_pool_demo` 回归
  （LM_Head 1,830ms → 预期 ~1,300-1,500ms）。

---

## 11. Phase 3 · 优化① 部署与实测结果（2026-08-09，CEO 确认后执行）

### 11.1 部署（可逆，已验证）

- 固件产物：`/tmp/fip_blocked.bin`（sha256 `2693d03a…`，322560B）；存档
  `fsbl/build/cv1800b_milkv_duo_sd/fip_blocked_phase3.bin`。
- 部署前备份：`/boot/fip.bin → /boot/fip_pre_blocked_phase3.bak`
  （`a000efa9…`，与烧写前完全一致）。
- 烧写后 `/boot/fip.bin` = `2693d03a…`，重启 ~20s 恢复，无 hang。
- 唯一源码变更：`comm_main.c` L320-326 EMBED_XPOSE naive 循环 → **blocked BS=32**
  （`EMBED_XPOSE_BS` 宏）；L311 `cur_v>2048` 上限未动。输出布局/描述符语义不变。

### 11.2 微基准（bench_lmhead_cost，设备实测）

| 项 | naive（烧写前） | blocked（烧写后） | 加速 |
|---|---|---|---|
| [3] busy-poll cur_v=1024 | 58.36 ms | 21.41 ms | 2.7× |
| [3] busy-poll cur_v=2048 | 106.45 ms | 29.79 ms | 3.6× |
| [3b] SEND_WAIT cur_v=2048 | 100.31 ms（bad=0） | **26.58 ms（bad=0）** | **3.8×** |

验收标准（@2048 55-75ms、bad=0）**远超达标**：26.58ms、bad=0（比最优预估快 ~2 倍）。

### 11.3 生产回归（交错 A/B ×2，§9.2 精确命令
`/root/smollm2_pool_{base,b2} /root/smollm2_instruct/ /root/input_tokens.bin 3 3 2`）

| 指标 | 旧固件 §9.2 | 新固件（blocked）平均 | 变化 |
|---|---|---|---|
| base LM_Head | 2,555 ms | **1,343 ms** | **-1,212 ms (-47%)** |
| b2 LM_Head | 1,830 ms | **1,242 ms** | **-588 ms (-32%)** |
| base Decode | 31,292 ms | **22,588 ms** | **-8,704 ms (-28%)** |
| b2 Decode | 27,327 ms | **22,030 ms** | **-5,297 ms (-19%)** |

- next_token=5021、13/24 chunks SD 与基线完全一致（正确性完好）。
- **b2 LM_Head 1,830→1,242ms**：低于 §10.3 预期 1,300-1,500ms，转置从关键路径
  （~76ms/chunk）降到 ~30ms/chunk，每 chunk 关键路径转移到 TPU build+dequant（~40ms）。
- **base 收益（-47%）> b2 收益（-32%）**：base 无转置重叠，blocked 直接释放其串行关键路径；
  b2 已把转置藏在 matmul 后，提速被重叠吸收，符合 §10.3 推演。

### 11.4 教训：回归必须用精确同命令

- TPU 工程师首轮回归误用 `/root/smollm2_pool /root/test_tokens.bin 20`
  （权重目录/token 文件/max_new 20/force_mode 0 全错），得 LM_Head 2,248ms，
  一度误判「反而变慢」。改用 §9.2 精确命令（`smollm2_instruct/` + `input_tokens.bin` +
  `3 3 2`）后 LM_Head 1,242ms，结论反转。
- **教训：任何 A/B 必须复用基线精确命令与输入；测试工程师不得凭记忆拼参数。**

### 11.5 结论与下一步

- **固件① 已验证有效并保留**（系统稳定、转置正确、生产显著提速，无回滚理由）。
- 转置已不再是 LM_Head 瓶颈；下一步候选（按预估收益）：
  1. **改动 D 重新评估**（`r_is_nm=true`，neuron 单缓冲）：转置提速后，每 chunk 的
     ION→neuron tile 拷贝+全量 MemFlush（28-37ms/chunk）成为主要残余开销；去掉后可再省
     ~300-600ms。需验证单缓冲与 B2 双缓冲重叠的取舍（§8.3 曾估叠加快转置后 +0.65s）。
  2. **Wt load 3.3s**（Change C 后续）：设备 load 下 SD 权重读取仍是全局最大单项。
  3. **cur_v=4096**（L311 上限 + demo CHUNK）：独立正交，收益 ~100-300ms。
- 回滚预案保持：`/boot/fip_pre_blocked_phase3.bak` 可一键恢复。
