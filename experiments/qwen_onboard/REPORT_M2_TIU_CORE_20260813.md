# M2 引擎 TIU 核心上板验证完成（qwen_engine_tiu）

日期：2026-08-13 | 作者：推理引擎工程师 | 关联：`qwen_engine_tiu.c` / `ion_probe{2}.c` / `ion_pmu_test.c` / `gen_engine_tiu_data.py`
状态：**真实 layer0 q_proj 28 个 K-block 上板通过——P1/P2 bit-exact、r_opt 与 fp64 gold 全一致、fp32 累加 maxrel 5.5e-08**（M2 骨架 TIU 核心达成）
护栏：全部 untracked（experiments/qwen_onboard/），不碰 master。

---

## 0. 结论摘要

1. **M2 TIU 核心上板闭环**：预建 cmdbuf 池（单次 LoadDmabuf，psize-only）+ 区域 Invld +
   真实权重流式 → 28 块两遍法 **P1/P2 bad=0/25088、r_opt 0/28 一致、累加 maxrel=5.5e-08**。
2. **关键内存发现（解锁预建池）**：`LoadDmabuf(enable_pmu=false)` 可用 **psize-only 源 dmabuf**
   ——每 cmdbuf ION 成本从 ~1.05MB（pmu）降到 ~1KB。此前 submit_budget 按 psize+pmu 分配是
   浪费；33-cmdbuf 池若带 pmu ≈34MB 会直接撞 26MB ION headroom OOM。
3. **计时**：per-RunCmdbufEx **0.145ms**；真实提交量（up/gate 需 N-tile）1800 runs/layer →
   TIU 全 runs ~6.25s/token（pass-only ~4.17s），CPU+invld ~1.4s，RVV dequant ~0.31s
   → **流水线保持 SD-bound（9.18s/token），1.15× 余量**。
4. **修正设计 §9b 提交量**：原 640 submits/layer 低估——up/gate [32,4864] 无法单次 LMEM
   承载（155KB>32KB），必须 N-tile（Ntile=896 → 6 片）。真实 pass 提交 1200/layer。

## 1. 上板实测（CV1800B，真实 layer0 q_proj，N=896，KG=32，M=1）

| 指标 | 实测 | 判定 |
|---|---|---|
| P1 回读 vs host int8_round_div | bad=0/25088 | ✅ bit-exact |
| P2 回读 vs host int8_round_div | bad=0/25088 | ✅ bit-exact |
| r_opt（TIU pass1 max → 计算） vs fp64 gold | 0/28 不一致 | ✅ |
| fp32 累加 vs fp64 gold | maxrel=5.5e-08, maxabs=3.7e-07 | ✅ |
| per-block wall | 0.931 ms | — |
| submit（g2l+pass1+pass2） | 0.434 ms/block → **0.145 ms/RunCmdbufEx** | ✅ 快路径 |
| MemInvld（全 256KB） | 0.052 ms/block | 区域化可再降 |
| RVV dequant（dequant_kal_rvv） | 0.041 ms/block | ~0.31s/token |
| CPU（max+累加，含 host 参考） | 0.133 ms/block | 真实引擎更小 |

**外推（24 层，真实 runs/layer=1800）**：
- TIU 全 runs（g2l+pass）：1800 × 0.145ms × 24 ≈ **6.25s/token**
- TIU pass-only：1200 × 0.145ms × 24 ≈ **4.17s/token**
- RVV dequant：320 blk × 0.041ms × 24 ≈ **0.31s/token**
- invld+cpu：320 × 0.174ms × 24 ≈ **1.42s/token**
- **SD 读地板 9.18s/token → max(SD, TIU, CPU) = SD 主导，余量 1.15×** ✅

## 2. 根因与修复：ION OOM 与 psize-only 突破

**症状**：qwen_engine_tiu 首版 `CVI_RT_MemAlloc(psize+pmu)` 在池内第 2 个 cmdbuf 即
`ion ioctl fail: Out of memory` + `mem_alloc_raw: 451` 断言。

**定位链**：
1. `ion_probe2` 实测 `CVI_RT_Init` 后 ION headroom ≈ **26MB**（26×1MB 分配成功，第 27 失败）。
2. `bmk1822_dmabuf_size` 对每个 cmdbuf 报 pmu=**1052672 B**（1.05MB）——与 cmdbuf 本体
   （psize 仅 640-960 B）无关的固定 pmu 区。
3. 我的池 33 cmdbuf × ~1.05MB ≈ 34MB > 26MB → OOM。submit_budget 只建 17 cmdbuf ≈18MB，
   恰好能过。

**修复**：`ion_pmu_test` 证明 `LoadDmabuf(enable_pmu=false)` 配 **psize-only** 源 dmabuf
正常执行（rc=0，matmul MATCH）。改后 33 cmdbuf ≈ **33KB**，整个预建池无内存压力。

**引擎铁律补充（写入引擎清单）**：
- 源 dmabuf 一律按 `psize` 分配（勿加 pmu），`LoadDmabuf(enable_pmu=false)`；
- 每条 cmdbuf **恰好 LoadDmabuf 一次**（沿用 submit_budget 慢路径铁律）。

## 3. 真实提交量修正（对齐 DESIGN §9b）

DESIGN §9b 以「6 N-tile 合并一次提交」估 640 submits/layer，但 [32,4864] 右矩阵 155KB
> 32KB LMEM，**无法单次提交**。Ntile=896（[32,896]=28KB，M1 已验证）需 6 片/块：

| matmul | K-blocks | tiles | pass runs | g2l runs |
|---|---|---|---|---|
| q/k/v/wo (×4) | 28 | 1 | 224 | 112 |
| up (N=4864) | 28 | 6 | 336 | 168 |
| gate (N=4864) | 28 | 6 | 336 | 168 |
| down (K=4864) | 152 | 1 | 304 | 152 |
| **合计/layer** | | | **1200** | **600** |

TIU pass-only 4.17s/token 仍 < 9.18s SD 地板，但余量由设计假设的 2.7× 收紧到 1.15×。

## 4. 下一步

1. **S2/S3 形状池**：N=128（k/v）与 N=4864→Ntile=896（up/gate，尾片 384）的
   g2l+pass 预建池，上板验证 bit-exact + 计时（补足 1800 runs/layer 外推）。
2. **区域化 MemInvld**：`CVI_RT_MemPreAlloc` 子视图限定 P1/P2 ~8KB，把 0.052ms/block
   的整 256KB Invld 进一步压低。
3. **M2 层前向集成**：rms_norm→quant→7 matmul→attention→residual→ffn 的完整层，
   M=1 增量 decode + KV cache，跑 3-prompt 回归（门禁 min gap<0.05）。

## 5. 复现

```
cd experiments/qwen_onboard
python3 gen_engine_tiu_data.py          # 生成 qwen_engine_tiu_data.h（真实层0激活）
# riscv64 交叉编译（见 qwen_engine_tiu.c 头注释）
python3 ~/Documents/MilkV_duo_project/duo_run.py qwen_engine_tiu
# 依赖：/data/qwen/layer0_kal.bin 已部署（deploy_qwen_weights.py）
# 注意：同 boot 多次跑 ION 程序会令运行时退化（submit 0.14→1.1ms），测时前 reboot
```
