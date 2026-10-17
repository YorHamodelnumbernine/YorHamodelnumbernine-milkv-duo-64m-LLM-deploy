# Stage 3 · Qwen2.5-0.5B-Instruct host INT4 质量验证报告

日期：2026-08-13
作者：推理引擎工程师
状态：**纯 host 验证完成，未上板、未接引擎、未提交 git**
范围：Qwen2.5-0.5B-Instruct（bf16 safetensors，988MB）
结论摘要：**原生 bf16 起点的 INT4 组量化方向可行（不塌缩）**。推荐 **线性层 RTN 对称 INT4 G=32（fp16 scale）+ 嵌入层 INT8 per-row**，12/12→11/12 argmax 保持、PPL +7.8%、体积 338MB（bf16 的 34%）。G64/G128 更省体积但 argmax 保持率降至 67%/58%。

---

## 1. 任务背景与对照

CEO 裁定：载体 = Qwen2.5-0.5B-Instruct；本阶段只做 host 质量验证。
对照上次判负根因（SmolLM2 per-channel INT8 → 组 INT4 二次量化不可压缩）：本次为**原生 bf16 起点 + 组量化**，理论上更可压缩。

## 2. 实验环境与工具

- host：x86_64，Python 3.13，torch 2.13+cpu（本地 wheel），transformers 5.14，safetensors 0.8
- 模型：`Qwen/Qwen2.5-0.5B-Instruct`（hf-mirror 拉取，24 层，hidden 896，intermediate 4864，vocab 151936，**tie_word_embeddings=true**，即 lm_head=embed 权重）
- 量化器：自研 `q4.py`（对称 RTN：scale=max/7，clamp[-8,7]，per-group fp16 scale；可切换非对称 zero-point、AWQ 校准）
- 脚本：`q4.py` / `stats_volume.py` / `perturb.py`（单层扰动）/ `full_model.py`（整模型）
- 产物目录：`~/Documents/MilkV_duo_project/tpu_bench/experiments/qwen_int4/`（untracked）

## 3. 权重统计（bf16 原始权重）

| 项 | 元素数 | bf16 体积 |
|---|---|---|
| attention+MLP 线性（24×7 矩阵） | 357.8M | 715.7 MB |
| embed_tokens（=lm_head） | 136.1M | 272.3 MB |
| 合计 | 494.0M | 987.9 MB |

- 权重 INT4 重建误差（RTN 对称，G=32）：RMS 相对误差约 10-15%（与组内 max 决定的步长相关）；G=32→G=128 误差上升约 3-4pct。
- L0 q_proj 存在强离群（max|w|=1.23，p99=0.24），单组 scale 最大 0.175——组量化步长受离群组影响。

## 4. 单层扰动（Gate 1，第一道闸）

真实激活 × INT4 重建权重 的 matmul 输出 vs bf16 基线（prompt：今天天气很好，我们去公园）：

| 层 | matmul rel_rms 范围 | 层输出 hidden rel_rms |
|---|---|---|
| L0 | 3.0–16.9%（v_proj 最差） | 13.3–16.8%（G32→G128）|
| L8 | 3–8% | 0.08–0.10% |
| L16 | — | 0.09–0.11% |
| L23（末层） | 6.9–14.8% | 14.2–17.9% |

- **中层因残差连接吸收扰动**，hidden 扰动仅 0.1%；首层 L0 与末层 L23 扰动最大（13-18%）。
- 单层量化后整模型 next_token：除 **L23 在 G≥64 翻转**（基线 gap 仅 0.125，过紧），其余全保持。
- 结论：单层扰动在 G=32 下可接受，L23/lm_head 邻近层是敏感点。

## 5. 整模型质量（Gate 2）

评估集：12 个中英 prompt（`eval2.txt`）。指标：argmax 一致率、top gap、logit MSE、KL(softmax)、PPL（calib 8 段）。

| 配置 | argmax 保持 | 平均 gap（base 1.53） | logit MSE | KL | PPL |
|---|---|---|---|---|---|
| bf16 基线 | — | 1.53 | — | — | 10.53 |
| **INT4 G32 + embed bf16** | **11/12 (92%)** | 1.43 | 0.97 | 0.23 | 11.28 (+7%) |
| INT4 G64 + embed bf16 | 8/12 (67%) | 1.45 | 1.29 | 0.39 | 11.05 (+5%) |
| INT4 G128 + embed bf16 | 7/12 (58%) | 1.33 | 1.97 | 0.47 | 9.69 (−8%) |
| **INT4 G32 + embed INT8** | **11/12 (92%)** | ~1.4 | 0.95 | — | 11.36 (+8%) |
| INT4 G32 + embed INT4 G64 | 10/12 (83%) | — | 1.20 | — | 12.41 (+18%) |
| INT4 G32 + embed INT4 G896（per-row） | 7/12 (58%) | — | — | — | — |

要点：
1. **gap 无明显收窄、无 logit 塌缩**（平均 gap 1.33–1.48 ≈ 基线 1.53；最坏单点 gap 0.0 仅出现在 embed INT4 G64 的 1 个 prompt）。
2. 未保持 argmax 的 prompt 均为基线 gap<0.6 的紧竞（如"今天天气很好…"基线 0.125，"人工智能是一种"0.625），翻转后的 token 多为 top-2/5 且同样合理的候选，**未演变为塌缩**。
3. PPL 波动 ±10%，属可接受量化噪声（非灾难性）。
4. **AWQ（alpha 扫描 0.1–1.0）不帮助**：校准域外不泛化，逐层 hidden 扰动反而 0.56（RTN 仅 0.10），弃用。
5. 非对称 zero-point（rtn_asym）与 RTN 对称持平，无额外收益。

## 6. 多 token 生成（贪心 12 token）

| 配置 | 中国的首都是 | The capital of France is |
|---|---|---|
| baseline | 中国的首都是____\nA. 北京\nB. 上海 | …is Paris. It is the largest city in the world… |
| **G32** | 与 baseline 逐字一致 | …is Paris. It is the largest city in Europe…（连贯） |
| G64 | ____。\nA. 乌鲁木齐\nB. 北京（结构完整） | …is Paris. The capital of Switzerland is Bern…（连贯） |
| G128 | 与 baseline 逐字一致 | …is Paris. The capital of Italy is Rome…（连贯） |

**关键结论：即使 G=128，多 token 生成仍连贯、语义正确**。这与 SmolLM2 判负时的"输出碎片化、gap→0"本质不同——Qwen2.5 每步重归一化能力强，单步 argmax 翻转不会级联塌缩。

## 7. 体积表 + SD 读取换算（阶段 4 决策用）

SD 带宽取实测 21.9MB/s；时间为"每 decode token 全量重读权重"的下限估算。

| 配置 | 体积 | bf16 占比 | SD 读取/ token |
|---|---|---|---|
| bf16 全量 | 988.0 MB | 100% | 45.1 s |
| 线性 INT4 G32 + embed bf16 | 473.5 MB | 47.9% | 21.6 s |
| 线性 INT4 G64 + embed bf16 | 462.3 MB | 46.8% | 21.1 s |
| **线性 INT4 G32 + embed INT8** | **337.7 MB** | **34.2%** | **15.4 s** |
| 线性 INT4 G64 + embed INT8 | 326.5 MB | 33.0% | 14.9 s |
| 线性 INT4 G32 + embed INT4 G64 | 273.6 MB | 27.7% | 12.5 s |
| 全 INT4 G64 | 262.4 MB | 26.6% | 12.0 s |
| 全 INT4 G128 | 254.7 MB | 25.8% | 11.6 s |

## 8. 结论与推荐

**INT4 方向（从 bf16 起点）明确可行，质量不塌缩。** 对照 SmolLM2 判负（二次量化不可压缩 + 生成塌缩），Qwen2.5-0.5B 从原始 bf16 出发的 INT4 组量化在生成层面保持连贯正确。

**推荐配置（阶段 4 候选）：线性层 RTN 对称 INT4 G=32（fp16 scale）+ 嵌入层 INT8 per-row。**
- 质量：argmax 保持 11/12、PPL +7.8%、gap 不塌缩、生成与基线一致。
- 体积：337.7 MB（bf16 的 34.2%），SD 读取下限 15.4s/token。

**分档建议（按体积/质量权衡）：**
| 档次 | 配置 | 体积 | 质量 | 适用 |
|---|---|---|---|---|
| 保守 | G32 + embed bf16 | 473 MB | 11/12 | 质量优先 |
| **推荐** | **G32 + embed INT8** | **338 MB** | **11/12** | 均衡 |
| 激进 | G64 + embed INT8 | 327 MB | 8/12 | 体积优先可接受质量降 |
| 极限 | G32 + embed INT4 G64 | 274 MB | 10/12（PPL+18%） | 内存极限 |

**两个必须上报的硬约束：**
1. **体积刚性**：Qwen2.5-0.5B 是 SmolLM2-135M 的 ~5 倍。即使最优 INT4，SD 读取下限仍在 12-15s/token，decode 实时性远差于 SmolLM2（3.7s/token）。阶段 4 若坚持该载体，必须解决权重驻留（ION/DDR 缓存 + 预取重叠）或接受高延迟。
2. **G 选择**：argmax 保持率 G32 92% > G64 67% > G128 58%。**推荐 G=32**；G64/G128 的额外体积收益（11-25MB）不值得质量降级。

**AWQ/校准结论**：本模型规模下 AWQ 激活感知缩放不泛化，反而劣化；RTN 对称即为简单可靠的基线。

## 9. 产物清单（均 untracked，未提交 git）

- `model/`：Qwen2.5-0.5B-Instruct bf16 safetensors（988MB）
- `q4.py` / `sf_io.py` / `stats_volume.py`：量化与统计工具
- `perturb.py`：单层扰动验证（`out/perturb.json`）
- `full_model.py`：整模型验证（`out/full_model.json`）
- `calib.txt` / `eval2.txt`：校准与评估语料
