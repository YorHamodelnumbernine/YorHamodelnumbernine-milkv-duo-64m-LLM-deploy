# SmolLM2-135M(-Instruct) on Milk-V Duo — 优化记录

## 模型规格

| 参数 | 值 |
|------|-----|
| 模型 | SmolLM2-135M / SmolLM2-135M-Instruct |
| d_model (D) | 576 |
| n_heads (H) | 9 |
| n_kv_heads (Kvh) | 3 |
| head_dim (d) | 64 |
| n_layers (L) | 30 |
| ffn_hidden (F) | 1536 |
| vocab_size (V) | 49152 |
| max_seq | 64 |
| 权重精度 | BF16 → per-tensor INT8（对称量化） |
| 总权重大小 | ~101 MB (30 layers × 3.38 MB/layer) |

## 内存布局 (Duo 64MB DRAM)

```
┌──────────────────────────────────────┐  高地址
│  Linux 系统 + 用户态 (~40 MB)        │
├──────────────────────────────────────┤
│  ION Pool (24 MB)                    │  ← CVI_RT_MemAlloc
│  ┌────────────────────────────────┐  │
│  │  KV Cache (L × 2 × max_seq×dkv)│  │  ← ION 顶部，向下增长
│  │  e.g. 30×2×40×192 = 450 KB    │  │
│  ├────────────────────────────────┤  │
│  │  Embed Cache (ION, 可变)       │  │  ← LM Head 加速
│  │  0 ~ 8.6 MB                    │  │
│  ├────────────────────────────────┤  │
│  │  Layer Weight Slots            │  │
│  │  Bank A: slots 0~2 (10.1 MB)   │  │  ← TPU 计算活跃区
│  │  Bank B: slots 3~5 (10.1 MB)   │  │  ← SD 预取目标
│  │  (3+3 mode, 2+2 则更少)        │  │
│  ├────────────────────────────────┤  │
│  │  LM Head Transpose Buffers     │  │  ← 双缓冲 (4×576×1024≈2.3MB)
│  └────────────────────────────────┘  │
├──────────────────────────────────────┤
│  DDR Pool (12 MB, malloc)            │  ← Embed 缓存 (DDR)
│  前 ~12,288 KB 的 embedding 行       │
└──────────────────────────────────────┘  低地址
```

### Embedding 三级缓存

```
Token ID → 行偏移 off = tid × 576

   off < embed_ddr_bytes (12 MB)
     → DDR 命中 (~14,200 行, 最频繁)
   
   off ∈ [embed_ddr_bytes, embed_ddr_bytes + embed_ion_bytes)
     → ION 命中 (~0~5,000 行, 次频繁)
   
   off >= embed_ddr_bytes + embed_ion_bytes
     → SD 卡读取 (~30,000 行, 稀有)
```

## 自适应 Pipeline：3+3 → 2+2 → 1+1

### 原理

ION 24MB 中 KV Cache 从顶部向下分配，layer weights + embed cache 从底部向上使用。随着 prompt 长度增长，KV Cache 增大，留给 layer weight slots 的空间减少，pipeline 自动降级。

### pool_calc_pipeline_mode() 决策逻辑

```
ion_free = ION_POOL_SZ (24MB) - kv_bytes
slots = (ion_free - 2MB_embed_reserve) / layer_sz (3.38MB)

  slots >= 6  →  mode = 3  (3+3 pipeline, 6 weight slots)
  slots >= 4  →  mode = 2  (2+2 pipeline, 4 weight slots)  
  slots <  4  →  mode = 1  (1+1 pipeline, 2 weight slots)
```

实测表现：
- max_seq=4 (prompt 短) → ion_free≈24MB → 3+3 mode (6 slots, embed≈3.6MB)
- max_seq=40 (prompt 长) → ion_free≈22.8MB → 2+2 mode (4 slots, embed≈8.9MB)

### 双缓冲流水线流程

```
   Bank A = slots [0, batch_slots)     (TPU 计算用)
   Bank B = slots [batch_slots, 2×batch_slots)  (SD 预取用)

   Step 1: SD read layers 0~2   → ION Bank A   (同步)
   Step 2: pthread SD read layers 3~5 → ION Bank B   (异步启动)

   循环 10 个 batch (30 层 / 3 slots):
   ┌─────────────────────────────────────────────────────────┐
   │  TPU compute layers 0~2 (Bank A)  │  SD→Bank B (并行)  │
   │  swap banks (A↔B)                                      │
   │  TPU compute layers 3~5 (Bank A)  │  SD→Bank B (并行)  │
   │  ...                                                    │
   │  TPU compute layers 27~29 (Bank A)│  (最后一个 batch)   │
   └─────────────────────────────────────────────────────────┘

   fallback: 如果 ION slots < 2×batch_slots，降级为串行 (SD read → TPU → repeat)
```

关键实现细节：
- SD 通过 pthread worker 直接读到 ION va（无 DDR staging，无 memcpy）
- 每个 batch 计算前 `CVI_RT_MemFlush` 确保 TPU 看到新数据
- `pf_job_t` 使用 mutex + cond 同步

## Chunked Prefill（分块预填充）

TPU 单次最多处理 10 个 token 的 attention（neuron memory 限制）。

```
prompt_len=37 的 chat prompt:
  Chunk 0: pos 0~9,  10 tokens, 无 LM Head
  Chunk 1: pos 10~19, 10 tokens, 无 LM Head
  Chunk 2: pos 20~29, 10 tokens, 无 LM Head
  Chunk 3: pos 30~36, 7 tokens,  LM Head → next_token
```

每个 chunk 之间 KV Cache 累积在 ION 中，后续 chunk 可以 attend 到所有之前的位置。

## LM Head：流式 Embedding Chunk 转置

最后一层输出 x(seq×576) × Embed^T(576×49152) = logits(seq×49152)

```
  49152 vocab → 48 个 chunk × 1024 列

  双缓冲在 ION 中:
    xpose_src[0/1]: 存放从 DDR/ION/SD 读取的 embed chunk (576×1024 INT8)
    xpose_dst[0/1]: 存放转置后的 embed chunk (1024×576 INT8)

  对每个 chunk:
    1. LM_CACHE_READ: DDR→ION 或 ION→ION 或 SD→ION
    2. CPU 转置 (或 mailbox 卸载到副核)
    3. tpu_matmul_build: x_final_i8(seq×576) × xpose_dst(576×1024)
    4. 与下一 chunk 的 I/O 重叠
```

LM_CACHE_READ 宏使用三级缓存：DDR > ION embed > SD fallback。

## rshift 修正（关键 Bug 修复）

### 问题背景

TPU INT8 matmul 使用 `res_is_int8=1` + `rshift_bits`，输出公式：
```
output_int8 = (dot_product_accum >> rshift)  clamped to [-128, 127]
dequant:     output_float = output_int8 × scale_left × scale_right × (1 << rshift)
```

`matmul_rshift(K)` 按最坏情况 K × 127 × 127 计算安全 rshift，防止 INT8 溢出：
- matmul_rshift(576) = 17  (K=576: 576×127×127=9.3M, >>17=70.9<127)
- matmul_rshift(1536) = 18 (K=1536: 1536×127×127=24.8M, >>18=94.5<127)

### Bug #1: LM Head 全零输出

**位置**: `sm_forward_pool()` line 1549

最终 hidden state 的 INT8 值很小（~[-8, 7]），实际点积远小于 9.3M。rshift=17 时：
- 典型点积 10,000 >> 17 = 0 → 输出全零

**修复**: `rshift_lm = matmul_rshift(D) - 5 = 12`
- 10,000 >> 12 = 2 (保留!)
- 最大安全值: 127 × 4096 = 520,192，实际点积远低于此值

### Bug #2: 单 token Decode 层间信号消失

**位置**: `sm_layer_forward()` lines 469-606

单 token decode 的 hidden state 值范围小（[-0.4, 0.4]），原有 rshift 导致 attention 和 FFN 的 INT8 matmul 输出全部舍入为零，只有残差连接 x = x + 0 + 0 通过。30 层后 L29 输出 = 输入 embedding，模型完全忽略上下文。

**修复**: 所有 rshift 统一减少 5：
```
rshift_qkv      = matmul_rshift(D)    - 5 = 12  (was 17)
rshift_scores   = matmul_rshift(d)    - 5 = 8   (was 13)
rshift_attn     = matmul_rshift(kv)   - 5 = 8   (was 13, varies)
rshift_ffn_up   = matmul_rshift(D)    - 5 = 12  (was 17)
rshift_ffn_down = matmul_rshift(F)    - 5 = 13  (was 18)
```
均 clamp 到 `>= 8`。

## 性能数据

| 阶段 | 耗时 | 占比 |
|------|------|------|
| Prefill (37 tokens) | 25,630 ms | 47% |
| Decode (per token) | 7,240 ms | - |

| 操作 (per-step avg) | 耗时 |
|---------------------|------|
| Wt load (SD read) | 3,451 ms |
| LM Head | 2,598 ms |
| FFN up | 523 ms |
| FFN down | 419 ms |
| QKV | 250 ms |
| SM (softmax) | 36 ms |
| Scores | 31 ms |
| Attn | 25 ms |

瓶颈：SD 卡权重读取 (48%) + LM Head (36%) = 84% 的总计算时间。

## Per-Channel 量化（v3 — 解决自循环）

### 问题分析

per-tensor INT8 量化导致两个致命问题：
1. **LM Head bias**：整个 49152×576 embed 矩阵共享一个 scale，某些行（如 token 965 " then"）的动态范围远大于其他行。在 per-tensor 下，这些行的 INT8 表示占满 [-127,127]，而大部分行只用到 [-10,10]。logit 计算时大动态范围的行天然占优。
2. **层权重噪声累积**：每层的 QKV/FFN 权重矩阵各用单一 scale。不同 output channel 的数值范围差异可达 10×，per-tensor 量化使小范围 channel 精度损失 90%+。30 层累积后，hidden state 被噪声淹没。

### 修复方案

**Per-row Embed 量化**（LM Head + Embedding lookup）：
- embed 矩阵 (49152×576) 的每一行独立量化，49152 个 per-row scale
- 文件：`embed_scales.f32`（192 KB）
- 每个 vocab token 的 embedding 和 logit dequantization 使用该 token 对应的 scale

**Per-channel 层权重量化**（QKV/FFN 所有矩阵）：
- 每个权重矩阵的每一列（output channel）独立量化
- Wq: 576 scales, Wk: 192, Wv: 192, Wo: 576, FFN up: 1536, gate: 1536, down: 576
- 每层 5184 scales / 20.7 KB，30 层共 155,520 scales
- 文件：`layer_scales.bin`（607 KB）
- dequantization 公式：`Y_f32[j] = Y_i8[j] × sc_in × scale[j] × (1 << rshift)`

### 效果对比

| 指标 | per-tensor (v2) | per-channel (v3) | 改善 |
|------|----------------|-------------------|------|
| Prefill logit gap | 2.4 | 1.5 | -38% |
| Decode step 1 logit gap | **44** | **1.4** | **-97%** |
| 自循环 | token 7133 × N | **无** (全部不同) | 解决 |
| Decode tokens | 2592,7133,7133,7133... | 30847,39444,26704,202,3427... | 多样化 |
| Weight load time | 3,450 ms | 2,418 ms | -30% |

### 当前输出示例

```
Input:  "Hello, how are you?"
Output: "abre unsuscomings...strugg therape..."
```

模型不再自循环，每个 decode token 都不同，logit gap 保持在 1-10 的健康范围。输出仍然是碎片化的词片段（"abre", "unsus", "comings", "strugg", "therap"），说明 30 层 INT8 量化仍然损失了大量语义信息，但已从「死锁」进化到「能生成多样化但低质量的文本」。

## 已知限制

1. ~~**Token 965 不动点**~~ → 已通过 per-row embed 量化解决
2. ~~**Token 7133 自循环**~~ → 已通过 per-channel 层权重量化解决
3. **输出质量**：per-channel 量化后模型不再自循环，但输出仍为碎片化词片段，非连贯自然语言。这是 135M 模型在 INT8 精度下的根本限制。要获得可用的输出质量，可能需要：
   - Per-token 激活量化（目前使用统一的 sc_x）
   - 更小的模型（< 50M 参数，层数少，量化噪声累积少）
   - INT4/INT8 混合精度 + calibration-based 量化
   - 更大的 TPU 内存以支持 INT16 中间结果

4. **SD 卡带宽瓶颈**：每层权重 3.38MB，30 层共需读取 ~101MB/token，SD 卡 random read 是最大延迟来源。

## 量化架构总结

```
量化层级（v3）：

  Embedding:  per-row INT8    (49152 scales, 192 KB)
  LM Head:    per-row INT8    (复用 embed scales)
  QKV matmul: per-channel INT8 (Wq 576 + Wk 192 + Wv 192 scales/layer)
  Wo matmul:  per-channel INT8 (576 scales/layer)
  FFN up:     per-channel INT8 (1536 scales/layer)
  FFN gate:   per-channel INT8 (1536 scales/layer)
  FFN down:   per-channel INT8 (576 scales/layer)
  Activations: per-tensor INT8 (sc_x 统一)
  RMS Norm:   FP32
  KV Cache:   FP16
  Attention:  INT8 matmul (per-tensor)
  Softmax:    FP32

总 scale 文件: 607 KB (layer_scales.bin) + 192 KB (embed_scales.f32) + 848 B (scales.bin)
```

## 文件索引

| 文件 | 用途 |
|------|------|
| `smollm2_pool_demo.c` | 主推理程序（pool + pipeline + LM Head） |
| `convert_smollm2.py` | 权重下载 + BF16→INT8 转换 |
| `chat_duo.py` | Python 交互式聊天客户端 |
| `common/tpu_bench.h` | TPU 初始化 + tpu_matmul 封装 |
| `Makefile` | 交叉编译（`make smollm2_pool_demo`） |

## 部署命令

```bash
# 编译
export PATH="$HOME/Documents/MilkV_duo_project/host-tools/gcc/riscv64-linux-musl-x86_64/bin:$PATH"
cd ~/Documents/MilkV_duo_project/tpu_bench
make smollm2_pool_demo

# 部署
sshpass -p milkv scp -O smollm2_pool_demo root@192.168.42.1:/root/

# 运行
sshpass -p milkv ssh root@192.168.42.1 \
  '/root/smollm2_pool_demo /root/smollm2_instruct/ /root/input_tokens.bin <max_new> <force_mode> <eos_id>'

# chat_duo.py 交互式聊天
cd ~/Documents/MilkV_duo_project/tpu_bench
python3 chat_duo.py --max-new 32
```
