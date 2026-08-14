# Phase 6 预研 · SD 带宽实测 + 重叠优化空间评估 — TPU 底层工程师

日期：2026-08-13 | 作者：TPU 底层工程师 | 状态：**预研完成（B 项）** | 关联：`REPORT_M2_24L_20260813.md` §5b / `qwen_engine_24l.c`
护栏：本报告留在 untracked `experiments/qwen_onboard/`，不碰 master。

---

## 0. 结论摘要

1. **SD 顺序读天花板实测 = ~21-22 MB/s**（块大小 4k/64k/1M 不敏感），低于 CEO 锚定的
   理论 29 MB/s。这是当前硬件实际天花板，非软件可再榨取的常规带宽。
2. **decode 每 token SD 地板（实测速率）≈ 15.7 s/token**：权重 201.4MB → 9.54s +
   LM head 136.1MB → 6.12s。高于 CEO 锚定的 11.6s（该值基于 29MB/s 理论）。
3. **重叠优化（ION 双缓冲 / TDMA g2l）空间已近枯竭**：引擎已 ~95% 与 SD 重叠；
   计算侧（TIU+CPU+dequant ≈ 7.97s）< SD 侧（≈15.7s），完美重叠也仅省 ≤0.5-0.7s。
4. **Phase 6 真杠杆 = 压 SD 字节数，不是重叠**：LM head top-k 两段式（136MB→降 1-2
   数量级）是最大单项；SD 控制器速度核查次之；ION 权重驻留仅 5.16MB 余量，无法承载
   201MB 权重。

## 1. SD 顺序读实测（Duo 上板，2026-08-13 20:5x）

| 测试 | 数据量 | 耗时 | 速率 | 备注 |
|---|---|---|---|---|
| layer0_kal.bin bs=1M | 8.4MB | 0.399s | **21.0 MB/s** | 单文件大块 |
| layer0_kal.bin bs=4k | 8.4MB | 0.404s | **20.8 MB/s** | 小块不敏感 |
| 全 24 层 layer*_kal.bin 顺序 cat | 201.4MB | 9.53s | **21.1 MB/s** | = 每 token 权重加载地板 |
| embed_i8.bin bs=1M | 136.1MB | 6.12s | **22.2 MB/s** | = LM head 流式地板 |
| SD 卡 | SD64G | | | mmcblk0 64GB |

- 结论：顺序读天花板 ~21-22 MB/s，与 M2 报告引用的 22.2MB/s（上板 dd）一致。
- 块大小 4k→1M 无差异 → 权重文件已充分顺序，无"更大块提带宽"空间。

## 2. decode 每 token SD 地板（按实测速率）

| 项 | 字节 | 实测地板 | 说明 |
|---|---|---|---|
| 24×layerN_kal.bin | 201.4MB | **9.54 s/token** | 每 decode 步全量重读 |
| embed_i8.bin（LM head 流式） | 136.1MB | **6.12 s/token** | V=151936 全量 top-5 |
| **SD 合计地板** | 337.5MB | **≈15.7 s/token** | 实测速率 |
| CEO §5b 锚定上界 | 337.5MB | 11.6 s/token | 基于理论 29MB/s |

> 差异来源：29 vs 21-22 MB/s 的 SD 速率假设。11.6s 是"理论带宽"上界，
> 15.7s 是"当前硬件"上界。两者之差即 SD 速度缺口，是 Phase 6 首要核查项之一。

## 3. 重叠优化空间评估（ION 双缓冲 / TDMA g2l）

引擎流水线现状（M2 §5b）：
- SD 权重读 9.54s + LM head 读 6.12s = **SD 侧 15.7s/token**
- TIU runs 6.26s + CPU/invld 1.4s + RVV dequant 0.31s = **计算侧 7.97s/token**
- M2 VERIFY=0 实测 14.55s/token（prefill per-token，SD 权重被 seq 摊销）

**判定**：计算侧 < SD 侧（7.97 < 15.7），引擎已将计算重叠在 SD 读之下，逼近 SD-bound。
- 理论上完美重叠 → per-token = SD 地板（~15.7s，decode 口径）。
- ION 双缓冲 / TDMA g2l 追加重叠最多回收"当前未隐藏的计算残差" ≤ **0.5-0.7s**，
  且无法突破 SD 地板。
- **结论：重叠优化不是 Phase 6 真杠杆**（边际 <5%），不建议优先投入。

## 4. Phase 6 真杠杆排序（按收益/风险）

| # | 杠杆 | 节省（估） | 依据/风险 |
|---|---|---|---|
| 1 | **LM head top-k 两段式** | ~4-6 s/token（136MB→1-2 数量级↓） | 当前每 token 固定 6.12s 读 embed；先投影聚类中心选 top-k 再全展开，M2 §5b 已裁定 |
| 2 | **SD 控制器速度核查** | 全局 ×~1.35（若 21→29MB/s） | 确认 SD 是否跑满协议上限（SD64G @ 21MB/s 偏低，可能非 UHS/HS 模式）；若可提速，SD 地板 15.7→11.6s |
| 3 | 权重驻留/热区缓存（ION） | 受限 | ION headroom 仅 5.16MB（M2 §5b），201MB 权重无法承载；仅能缓存极小热区，收益 <1s |
| 4 | INT8 KV | 仅降 KV 带宽 | decode 权重读（9.54s）主导，KV 非主导；收益有限 |
| 5 | ION 双缓冲/TDMA g2l 重叠 | ≤0.5-0.7s | 已近饱和，非优先 |

## 5. 建议（Phase 6 立项口径）

1. **最高优先：LM head top-k 两段式**——把 136MB/6.12s 的固定成本压到 ~0.1-0.5s，
   单项可把 decode 地板从 ~15.7s 拉到 ~10s 以下。
2. **核查 SD 控制器模式**：确认 SD 卡工作在 HS/UHS 模式、时钟配置是否到位；
   若能把 21→29MB/s，则全链路（权重+LM head）同时受益，地板 ~11.6s。
3. **重叠优化不再投**：ION 双缓冲/TDMA g2l 已近上限，省下的 <0.7s 不值得复杂度。
4. 权重驻留仅作"热区缓存"小步试，不作为主路径。

## 6. 复现

```
cd /home/vasilybabyboy/Documents/MilkV_duo_project
python3 duo_run.py --cmd 'dd if=/data/qwen/layer0_kal.bin of=/dev/null bs=1M'   # ~21MB/s
python3 duo_run.py --cmd 'time cat /data/qwen/layer*_kal.bin > /dev/null'      # 9.53s
python3 duo_run.py --cmd 'dd if=/data/qwen/embed_i8.bin of=/dev/null bs=1M'    # 22.2MB/s
```
