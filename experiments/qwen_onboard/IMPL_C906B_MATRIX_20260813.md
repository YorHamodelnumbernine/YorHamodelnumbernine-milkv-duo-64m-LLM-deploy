# C906B 逐矩阵实现建议（Path A 两遍法 / Qwen2.5-0.5B）— 正式版

日期：2026-08-13
作者：推理引擎工程师
状态：**正式（闸口①签核 / fp32 spike 关闭 / M2 24L 数值骨架已验证后转正）**
依赖：`qwen_kal_ref.c`（host C 参考，3/3 数值骨架）· `weights_kal/`（K-aligned INT4 G32 + fp16 gsc）
配套：`DESIGN_PATH_A_TWOPASS_20260813.md`（§4 逐矩阵粒度 / §9a 最终 ION 布局 / §10 rope 修复）

---

## 0. 目标与状态

**目标**：把 `qwen_kal_ref.c` 的 `chunk_matmul_twopass` 语义映射到 C906B，使 SD(201MB)
主导的 decode 中 **SD 读 / CPU 反量化 / TIU 两遍** 三者全并行，每矩阵耗时
≈ max(SD, CPU, TIU) 而非三者之和。

**本版转正依据（4 条）**：
1. **fp32 spike 关闭**（commit f6b014c）：CV1800B `ps32_mode==2` 不输出 fp32（硬件限制），
   变体1 判负；变体2（CPU 回环）成本 +0.5-1s/token。**主路径 Path A 两遍法不变**，
   C 参考仍是唯一主路径数值骨架。
2. **闸口① 签核**（`patha_kg32_check` 5/5）：pass1 int8 回读在 KG=32 生产形状上逐元素
   精确（含 up N=4864 与 rshift=5 饱和），tdma_l2g + MemInvld 路径可用。
3. **M2 24L 里程碑**：24 层 + LM head + 3-prompt 回归通过，NEXT 3/3、min gap 0.4508、
   111,744 TIU runs 全 bit-exact、VERIFY=0 延迟 **14.55 s/token**。
4. **KG=32=G 锁定**（`REPORT_KG_SWEEP`）：KG=64/128 均 2/3（组 scale 变粗伤重建精度），
   submit 不能靠放大 KG 降 ~4x，块 KG 必须等于 scale 组 G。

---

## 1. ION 布局映射（§4 逐矩阵粒度 → §9a 最终裁决版）

**设计演进说明**：§4 给出**逐矩阵双缓冲**（SD_BUF 2×2.45MB，合计 ~7.2MB）作为流水线
粒度概念；闸口③ 以 **per-layer INT4 双缓冲 16.8MB** 为基数核定最终内存布局（§9a，
≈18.84MB，余 5.16MB）。逐矩阵实现建议以 **§9a 最终裁决版**为 ION 映射依据；§4 的
「SD 双缓冲 + CPU dequant 并行」逐矩阵流水线概念完整保留。

**最终 ION 布局（§9a，本建议的内存映射）：**

| 偏移（MiB） | 大小 | 区域 | 逐矩阵角色 |
|---|---|---|---|
| 0x000000 | 8.0 MiB | **SD_BUF_A** | 当前层 layerL 全部 7 矩阵 INT4(nib+gsc)；CPU/TIU 消费 |
| 0x800000 | 8.0 MiB | **SD_BUF_B** | 下一层 layerL+1（SD DMA 双缓冲，与消费重叠） |
| 0x1000000 | 160 KiB | **DQ_BUF_A** | 当前 K-block 全宽解包 int8 [32,4864]（CPU 写 / TIU g2l 读） |
| 0x1028000 | 160 KiB | **DQ_BUF_B** | 下一 K-block（CPU 双缓冲，与 TIU 消费重叠） |
| 0x1050000 | 64 KiB | **P1_BUF** | pass1 int8 [M≤10, Nt≤896]（CPU 读真实 max） |
| 0x1060000 | 64 KiB | **P2_BUF** | pass2 int8 [M≤10, Nt≤896]（CPU 读回 fp32 累加） |
| 0x1070000 | 1.5 MiB | ACT/ACC | x/h/attn/qkv/mid/out fp32 [M≤10, D/F] |
| 0x11F0000 | 0.9 MiB | KV/misc | r_opt MAX_BUF + INT8 KV + scratch |
| 0x1280000 | 5.16 MiB | FREE | 余量（KV 扩展 / LM head top-k staging） |
| 合计 | **≈18.84 MiB** | — | **< 24 MiB，余 5.16 MiB** |

**逐矩阵映射规则**：
- SD_BUF_A 内 7 矩阵按层文件布局连续（rms_attn→q→k→v→wo→up→gate→down→rms_ffn），
  CPU 依 offset 定位各矩阵 nib/gsc 段；
- **DQ_BUF 尺寸上界 = up/gate/down [32,4864] = 155 KiB（对齐 160 KiB）**；q/k/v/wo
  [32,896]=28 KiB、[32,128]=4 KiB 均远小于上界，无需动态切换；
- P1/P2 输出 [M≤10, Nt≤896] ≤ 8.96 KiB，实际 MemInvld 只刷输出区 ~8 KiB（勿全 buffer）；
- DQ_BUF 必须在 ION/neuron 内存（TIU g2l 从 neuron 读 right 矩阵），纯 DDR 需二次拷贝——否决。

---

## 2. 每矩阵参数表（M=1 decode，tilew=896）

| matmul | [K,N] | KG | N-tiles | nib+gsc 字节 | SD 读 @21.1MB/s | pass runs(×2) |
|---|---|---|---|---|---|---|
| q | 896×896 | 28 | 1 | 451,584 | 21.4 ms | 56 |
| k | 896×128 | 28 | 1 | 64,512 | 3.1 ms | 56 |
| v | 896×128 | 28 | 1 | 64,512 | 3.1 ms | 56 |
| wo | 896×896 | 28 | 1 | 451,584 | 21.4 ms | 56 |
| up | 896×4864 | 28 | 6 | 2,451,456 | 116.2 ms | 336 |
| gate | 896×4864 | 28 | 6 | 2,451,456 | 116.2 ms | 336 |
| down | 4864×896 | 152 | 1 | 2,451,456 | 116.2 ms | 304 |
| **层合计** | | | | **8,393,728** | **397.8 ms** | **1200** |
| **×24 /token** | | | | **201.4 MB** | **9.55 s** | **28,800** |

- **SD 读主导**：9.55s/token（权重）+ LM head 6.12s（embed_i8 136MB@22.2MB/s）=
  **15.7 s/token 实测地板**（对应 M2 24L VERIFY=0 的 14.55s，含未完全重叠残差）。
- **TIU run-only**：1200 × 0.145ms × 24 ≈ **4.18 s/token**（pass-only），≪ SD 9.55s → 可被掩盖。
- **CPU 摊销**（RVV 解包 + 区域 Invld + fp32 累加）≈ **1.7 s/token**，≪ SD → 可被掩盖。

---

## 3. SD/CPU 流水线（3 级，K-block 粒度）

```
for layer L:
  SD DMA(L+1 -> SD_BUF_B)                        # 独立异步，8.39MB@21.1MB/s ≈ 398ms
  for matrix m in {q,k,v,wo,up,gate,down}:
    for K-block g (q/k/v/wo/up/gate: 28；down: 152):
      CPU:  dequant block g+1 全宽 -> DQ_BUF_B   # RVV ~384MB/s，纯权重侧，不依赖 TIU
      TIU:  two-pass block g（消费 DQ_BUF_A）    # 唯一串行点 = pass1 max 读回
      CPU:  fp32 累加 pass2（与 TIU 下一 block 重叠）
```

- **CPU 反量化是纯权重侧（INT4→INT8），天然可预取**：TIU 消费 block g 时 CPU 解 block g+1。
- **pass1→读 max→pass2 每个 K-block 只产生一次 CPU 等待**；等待窗口内 CPU 切去解
  block g+1，不空转。
- **吞吐 = max(SD 9.55s, CPU ~1.7s, TIU 4.18s) = SD-bound**，余量 >2×。

**关键设计点（比 M2 引擎省一半解包）**：
- **反量化按 K-block 全宽 [32,N] 一次完成**（非按 N-tile 分片、非 pass1/pass2 各做一次）。
  DQ 内容在整个 block 的 6 N-tile × 2 pass 周期内驻留 DQ_BUF。
- M2 引擎（`qwen_engine_24l.c`）当前 per-(tile,pass) 重解包 → 1200 次/layer；
  本建议降为 **320 次/layer**（q/k/v/wo/up/gate 各 28 + down 152），消除 pass1/pass2
  的重复解包与重复 g2l（Gate 4a「same-LMEM weight, no reload between passes」的 ION 侧实现）。
- DQ_BUF_A/B 双缓冲使解包（写 B）与 TIU 消费（读 A）并行，四级嵌套：
  SD_BUF(A/B) ║ DQ_BUF(A/B)。

---

## 4. 两遍法微内核 TIU 调度

**单次提交 = 合并 cmdbuf**：`g2l(left M×32)` + `g2l(right [32,Nt] @ DQ_OFF+tile)` +
`TIU matmul(rshift)` + `l2g(out [M,Nt] -> P1/P2)`。一次 `RunCmdbufEx` 完成一 tile 一遍。

**per K-block g 微流程**：
```
1. CPU:  RVV 解 block g 全宽 [32,N] -> DQ_BUF（toggle A/B）     # up/gate ~0.4ms
2. TIU:  pass1 各 N-tile，rshift=rsafe，输出 P1
3. CPU:  区域 Invld(P1 ~8KB) + 扫 P1 更新 block_max（跨全部 M 行 × 全部 tile）
         r_opt = ceil(log2((block_max<<rsafe)/127))            # block-shared 标量
4. TIU:  pass2 各 N-tile，rshift=r_opt，输出 P2
5. CPU:  区域 Invld(P2 ~8KB) + accd[m,n] += p2 × 2^r_opt × gsc[g,n]
6. 块结束: out[m,n] = accd[m,n] × sc_row[m]
```

**cmdbuf 池（预建，禁止 per-submit 全链路）**：
- 池维度：**16 rshift 变体 × 7 shape（128/256/384/512/768/896）× 2 dest（P1/P2）**；
  decode M=1 只需 tilew=896（LMEM left[1,32]+right[32,896]+out[1,896] ≈ 29KB < 32KB）。
- 每 cmdbuf **恰好 LoadDmabuf 一次**，源 dmabuf 按 **psize-only** 分配
  （`enable_pmu=false`，单条 ~1KB；勿带固定 1.05MB pmu，33 条即 34MB 撞 26MB ION headroom）。
- submit = select + patch rshift + Run（~15µs）。**禁止** register+alloc+convert+load
  全链路（0.55ms/submit × 28,800 = 8.4s/token 假象）。

---

## 5. 硬约束（引擎铁律，违反则串行 16.99s 超 SD）

1. **cmdbuf 预建**：16 rshift × shape 预注册；submit 仅 patch rshift + Run。
2. **区域 MemInvld**：只 invld P1/P2 输出区（~8KB），禁止全 buffer Invld（0.6ms/chunk 假象）。
3. **RVV 解包**：`dequant_kal_rvv`（上板 377.9/384.0 MB/s，scalar 仅 33-50 MB/s）；
   列分块 CB=256 使 down 从 130→384MB/s。
4. **psize-only dmabuf**：`LoadDmabuf(enable_pmu=false)`，源按 psize 分配。
5. **N-tile 池精确匹配**：g2l right 不得越过 dequant tile 边界（tilew 精确命中池 shape，
   否则 g2l 读到 DQ 区外）。
6. **DQ 全宽驻留**：同一 K-block 的 pass1/pass2 复用同一 DQ 内容，不重解包、不重写 DQ。

---

## 6. 数值骨架保持（上板对照基线）

- **`chunk_matmul_twopass` 语义原样保留**（`qwen_kal_ref.c` 146-198 行，上板引擎基线）：
  - 两遍法 per K-block：pass1 `rshift=rsafe` → `est=max|p1|<<rsafe` → `r_opt` → pass2。
  - TIU 舍入 = **`sat8((acc + (1<<(rshift-1))) >> rshift)`**（round-half-up toward +inf，
    GATE_A_SIGNOFF 确认），CPU 反量化必须用 `int8_round_div` 匹配，否则 pass2 与硬件不一致。
  - fp32 累加：`accd[m,n] += p2 × 2^r × gsc[g,n]`，最终 `out = accd × sc_row[m]`。
  - per-row 激活量化：**round-half-even（bankers）+ clip ±127**，`sc[m]=max/127`。
- **r_opt 语义**：**block-shared**（每 K-block 跨全部 M 行 × 全部 N-tile 收集 max）；
  decode M=1 退化为单行；prefill M=seq 跨行收集（M2 已验证）。
- **rope 表布局（关键，M2 已修复）**：cosb/sinb 必须按 **`[pos][HD/2]`（stride = pos×(HD/2)）**
  预计算，`rope_inplace` 读 `cos[pos*(HD/2)+i]`。历史 bug：预计算写 `cosb[pos*HD+j]` 而
  读 `cos[pos*(HD/2)+i]`，stride 不一致导致 pos≥1 读到未初始化内存、层 0 发散。
  **上板引擎 rope 表必须 [pos][32] 布局，否则同样崩溃。**

---

## 7. 上板验证清单（闸口①已签核；本建议落地后执行）

1. `[M=1, 1×32]×[32,N]` 最小两遍闭环：pass1 rsafe 回读 max → r_opt → pass2，比对 host。
2. **up/gate N=4864 全宽 DQ [32,4864] + 6 N-tile 两遍时序**（验证 DQ 驻留、不重解包）。
3. down K-chunk 1024（152 block）+ 块内 G=32 解包流水。
4. decode 1 token 全层与 `qwen_kal_ref` 对照 top-5（门禁 min gap ≥ 0.05，NEXT 3/3）。

---

## 8. 开放问题 / 二期（不影响本建议落地）

- **LM head 流式 136MB → 6.12s/token**：top-k 两段式（Phase 6 最高优先，单项可把
  decode 地板 15.7s → ~10s 以下）。
- **SD 控制器 21.1 vs 理论 29MB/s**：核查 HS/UHS 模式；若可提速，权重+LM head 全链路
  ×1.35 受益（15.7s → ~11.6s）。
- **双核 offload**：`/proc/self/pagemap` 不可用（SmolLM2 已知）→ 默认主核单核执行；
  DDR→ION offload 需物理地址解析可行路径（另立任务）。
- **重叠优化已近饱和**：ION 双缓冲 / TDMA g2l 追加重叠最多回收 ≤0.5-0.7s（Phase 6 不投）。

---

## 9. 复现 / 关联产物

- host 数值骨架：`qwen_kal_ref <weights_dir> <tok...>`（gcc -O2，3/3）。
- 上板验证里程碑：`qwen_engine_24l.c`（M2 24L，VERIFY=1 全 bit-exact）。
- RVV 解包：`dequant_kal.c`（scalar + `dequant_kal_rvv`，CB=256）。
- 闸口证据：`REPORT_PATHA_KG32_20260813.md` / `REPORT_PS32_FP32_SPIKE_20260813.md` /
  `REPORT_KG_SWEEP_20260813.md` / `GATE_A_SIGNOFF_20260813.md`。
