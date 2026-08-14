# ②④ Submit 预算阶段性结果 + ③ 权重部署完成（Qwen Path A → Duo）

日期：2026-08-13 | 作者：推理引擎工程师 | 关联：`submit_budget.c` / `submit_trigger{2,3}.c` / `deploy_qwen_weights.py`
状态：**submit 预算闭环（快路径 0.44ms/block 实测）+ 324MB 权重部署完成（md5 全过，SD 22.2MB/s）**
护栏：全部 untracked（experiments/qwen_onboard/），不碰 master。

---

## 0. 结论摘要

1. **cmdbuf 预建摊销达成 10-18x**：naive per-submit 2.50ms → prebuilt Run+Invld 0.140ms。
2. **完整两遍法（含 g2l+回读+CPU r_opt）实测 0.44ms/block**（KG=32, N=896），
   换算 per-token CPU+TIU ≈ **3.4s < SD 9.18s 地板** ✅ 硬约束成立。
3. **发现并定位一个关键运行时陷阱**：对已加载 dmabuf 二次 `LoadDmabuf`
   会永久翻转运行时到 ~22ms/submit 慢路径（44ms/block）。**每个 cmdbuf 必须恰好加载一次**。
4. **权重部署完成**：`/data/qwen/` 54 文件 324MB，逐文件 md5 全过，SD 顺序读 22.2MB/s。
5. **遗留瓶颈**：scalar Path A dequant 实测 44-55MB/s → **3.2-4.0s/token**，
   需要 RVV 化（参考 int4_common.c RVV 419MB/s → 预期 ~0.4s/token）才能建立充裕余量。

## 1. Submit 预算实测（CV1800B 上板，KG=32, N=896, M=1）

| 指标 | 实测 | 换算 per-token | 判定 |
|---|---|---|---|
| A naive per-submit（register+alloc+g2l+mm+l2g+acquire+convert+load+Run） | 2.50 ms | 38.4 s | ❌ 违反 SD 地板 |
| B1 prebuilt Run+Invld | 0.140 ms | 2.14 s | ✅ |
| B2 prebuilt Run-only（floor） | 0.115 ms | 1.76 s | ✅ |
| B3 Invld-only（256KB 整块） | 0.020 ms | 0.31 s | ✅ |
| **C 两遍法整块（g2l+pass1+inv+CPU max+pass2+inv）** | **0.44 ms/block** | **~3.4 s** | ✅ < 9.18s |
| D scalar PathA dequant | 44-55 MB/s | 3.2-4.0 s | ⚠️ 需 RVV |

**C 分解/block**：g2l 0.128 + pass1 0.120 + inv1 0.027 + CPU r_opt 0.014 + pass2 0.123 + inv2 0.024 ≈ 0.44ms。
（24 层 × 320 blocks/层 × 0.44ms ≈ 3.4s/token。）

## 2. 慢路径根因：二次 LoadDmabuf 是致命陷阱

定位过程：
- `submit_budget.c` 首版 C 段 44ms/block（p1/p2 各 21.9ms），与 B 段 0.14ms 矛盾。
- `submit_trigger.c`：孤立 g2l/pass/回读均保持快态（0.11ms）→ 排除「g2l 本身」。
- `submit_trigger2.c`：干净环境完整 C 循环 28 块仅 **0.42ms/block**，状态恒 FAST。
- `submit_trigger3.c`：逐步复现 B→C 序列，**X1（对同一 var_mem 二次 LoadDmabuf）后立即 SLOW 22.2ms**。
- 修复（单次加载，B/C 共用 p_ld）后 `submit_budget` 恢复 **0.44ms/block** ✅。

**引擎铁律（写入 DESIGN）**：所有 cmdbuf 经 `LoadDmabuf` 恰好一次，保留 loaded handle
全程复用；任何「每层/每矩阵重建或重载」都会触发 ~22ms/submit 慢路径。

## 3. 权重部署完成（③ 前置条件满足）

```
/data/qwen/  54 文件  324MB
  embed_i8.bin       136.1 MB
  24× layerN_kal.bin 各 8.39 MB
  embed_scales/bias/config/final_rms 等
```
- 传输：`deploy_qwen_weights.py`（paramiko 管道，逐文件 md5）→ **deployed 54/54**。
- SD 顺序读：`dd` 实测 layer0 22.2 MB/s、embed 22.3 MB/s（与设计假设 21.9MB/s 吻合）。
- 已具备 M2 上板 3-prompt 回归的权重流式读取条件。

## 4. 下一步

1. **Path A dequant RVV 化**（D 段 3.2-4s → 目标 0.4s）：适配 `int4_unpack_fixed_rvv`
   (419MB/s) 到 Qwen K-aligned 布局（nib[N][16] → [32,N] K-major），消除 CPU 瓶颈余量风险。
2. **M2 引擎骨架**：SD 流式 + ION 双缓冲 + 单次 LoadDmabuf 预建 cmdbuf 池，跑上板 3-prompt 回归
   （门禁 min gap<0.05）。
3. 区域 MemInvld（B3 整块 256KB 0.31s/token 可再降）。
4. 同步 TPU 工程师：慢路径铁律 + submit 预算闭环数据。

## 5. 复现

```
cd experiments/qwen_onboard
make submit_budget submit_trigger2 submit_trigger3   # 或手动 riscv64 交叉编译
python3 ~/Documents/MilkV_duo_project/duo_run.py submit_budget
python3 ~/Documents/MilkV_duo_project/duo_run.py submit_trigger3
```
