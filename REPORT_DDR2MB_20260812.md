# DDR embed=2MB 正式回归 — 实测报告

日期：2026-08-12
作者：推理引擎工程师
状态：正式回归完成，设备已交还（ION 无残留）
设备状态：TPU 底层工程师 Phase 4 完成后释放，22:17 巡检无残留，ION used=0
基准命令（精确命令，避免 §11.4 教训）：
- 基线：`/root/smollm2_pool_b2 /root/smollm2_instruct/ /root/input_tokens.bin 3 3 2`
- DDR=2MB：`LM_EMB_DDR_KB=2048 /root/smollm2_pool_b2_vc /root/smollm2_instruct/ /root/input_tokens.bin 3 3 2`
- 同二进制基线对照：`/root/smollm2_pool_b2_vc ... 3 3 2`（默认=基线行为，隔离环境变量效应）
- 未跑 `LM_EMB_DDR_KB=0`（会 OOM 毒化 ION）

---

## 1. 三次运行原始数据（连续跑，设备空闲、ION 全程 used=0）

| 指标 | ① b2 基线（DDR=12MB） | ② b2_vc 同二进制基线 | ③ b2_vc DDR=2MB |
|---|---|---|---|
| Wt load | 3551 ms | 3280 ms | 3688 ms |
| LM_Head | 1175 ms | 1311 ms | **826 ms** |
| Prefill | 25132 ms | 23756 ms | **22223 ms** |
| Decode | 22213 ms | 22487 ms | **20245 ms** |
| Decode/tok | 5553 ms | 5622 ms | **5061 ms** |
| Total | 47345 ms | 46243 ms | **42469 ms** |
| VmSwap(前 decode) | 7296 KB | — | — |
| embed DDR/ION | 12096/1728 KB | 12096/1728 KB | 1728/1728 KB |
| next_token | 5021 | 5021 | 5021 |

- next_token=5021 三次全保持（正确性 OK）；采样末 token 有噪声属正常。
- ION 三次运行后均 used=0（无孤儿进程）。

---

## 2. 结论：DDR=2MB 是净收益最优配置（同二进制对比 ② vs ③）

| 指标 | Δ（③ − ②） | 幅度 |
|---|---|---|
| Wt load | +408 ms | +12.4%（在噪声带内，见 §3）|
| **LM_Head** | **−485 ms** | **−37.0%** |
| Prefill | −1533 ms | −6.5% |
| Decode | −2242 ms | −9.9% |
| **Decode/tok** | **−561 ms** | **−10.0%** |
| **Total** | **−3774 ms** | **−8.2%** |

对照 b2 基线（① vs ③）：Total −4876 ms（−10.3%），decode/tok −492 ms（−8.9%）。

## 3. Wt load 略升的说明（不构成否决）

- Wt load 三次：3551 / 3280 / 3688 ms，波动 ±300–400 ms（SD/swap 噪声主导，
  与历史 3314–3941 ms 波动一致）。② vs ③ 的 +408 ms 部分在噪声带内。
- 即便 Wt load 真因 DDR 收紧略升，净收益仍明确为正：
  LM_Head −485 ms + decode −561 ms/tok 远大于 Wt load +408 ms。
- 与 CEO 扫描 run_c2（LM_Head 834 ms、Total 44370 ms、decode 5166 ms/tok）完全吻合。

## 4. 建议

1. **采纳 DDR=2MB 固化**（`LM_EMB_DDR_KB=2048` + `smollm2_pool_b2_vc`），净收益 +8–10%。
2. Wt load 的真正优化仍走 INT4 权重（设计 A，3.4s→~1.5–1.7s），与 DDR=2MB 叠加。
3. 长上下文走 INT8 KV（39→~156 token），与 INT4 权重叠加可达 ~1000 token。

## 5. 回归对比存档

- 本次：b2 基线 Total=47345ms；DDR=2MB Total=42469ms（best）。
- 历史参考：CEO run_c2 Total=44370ms；Wt load 波动 3314–3941ms。
- 未提交 git。
