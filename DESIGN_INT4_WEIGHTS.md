# INT4 权重原型设计 — Wt load 主力优化

日期：2026-08-12
作者：推理引擎工程师
状态：host 侧设计 + 原型框架，未接生产路径、未上板（设备由 TPU 工程师占用）
目标：Wt load 3.4s → ~1.5s（×2.2），配合 INT8 KV 可同时放大上下文

---

## 1. 现状与目标

- 权重以 `layerN.bin` 单文件存放（merge_layers.c 合并），每层 3,543,552 B：
  rms_attn(f32,2304) + Wq(i8,D·D) + Wk/Wv(i8,D·dkv) + Wo(i8,D·D) +
  rms_ffn(f32,2304) + ffn_up/ffn_gate/ffn_down(i8,D·F)。
- 7 个 INT8 矩阵合计 3,538,944 B/层，是 SD 读取主体。
- Wt load 组成：每批读 10.1MB≈462ms @21.9MB/s，计算仅 133ms，暴露 329ms/批×10≈3.4s。
- **INT4 目标：SD 读字节减半 → 读 260ms/批；解包（CPU）隐藏到计算下 → 暴露 ~130ms/批 → Wt load ~1.5s。**

---

## 2. 打包格式（设计）

### 2.1 文件格式 `layerN_i4.bin`
保持与 `layerN.bin` 相同的组件顺序，只把 7 个 INT8 矩阵换成 INT4 打包块：

```
[ rms_attn f32        : D*4            ]  2304  B   （不量化）
[ rms_ffn  f32        : D*4            ]  2304  B   （不量化）
[ Wq  i4 包           : (D*D)/2 + S    ]
[ Wk  i4 包           : (D*dkv)/2 + S  ]
[ Wv  i4 包           : (D*dkv)/2 + S  ]
[ Wo  i4 包           : (D*D)/2 + S    ]
[ ffn_up  i4 包       : (D*F)/2 + S    ]
[ ffn_gate i4 包      : (D*F)/2 + S    ]
[ ffn_down i4 包      : (F*D)/2 + S    ]
```

### 2.2 组内量化（per-group symmetric）
- 组大小 `G=32`（每 32 个连续 int8 值 1 个 scale；32 与 C906 128-bit 向量宽度对齐）。
- 组内 scale：`s = max(|v_0..v_31|) / 7`（int4 对称范围 [-7,7]，留 -8 备用或也允许）。
- 量化：`q = clamp(round(v / s), -8, 7)`（对称；如用 [-7,7] 则 clamp(-7,7)）。
- 反量化：`v' = q * s`，四舍五入到 int8。

### 2.3 字节布局（每个矩阵）
```
[ packed_nibbles : N/2 字节 ]   ← 低半字节=值2i, 高半字节=值2i+1（行优先连续）
[ scales        : (N/G)*4 字节 ] ← float32, 每个组 1 个
```
- N=矩阵元素数。scale 开销：每 32 值 4B → 12.5% 的 int4 数据量。
- 用 fp32 起步（简单、可精确定位），后续可改 fp16 省一半 scale 字节。

### 2.4 层大小估算（D=576, dkv=192, F=1536）
| 项 | INT8 现值 | INT4 打包 |
|---|---|---|
| rms (f32) | 4,608 | 4,608 |
| 7 矩阵 int8 | 3,538,944 | — |
| packed_nibbles | — | 1,769,472 |
| scales (G=32, fp32) | — | 3,538,944/32×4 = 442,368 |
| **层总计** | **3,543,552** | **2,216,448**（**-37%**）|

> 注：G=32 + fp32 scale 使 SD 字节只省 37%（非 50%）。若改 **G=128 + fp16 scale**：
> scales = 3,538,944/128×2 = 55,296 → 层 1,829,376（**-48%**，接近 50%）。
> 推荐：**G=64 + fp16 scale**（scales=110,592，层 1,884,672，-47%，精度/开销平衡）。
> 精度与开销的权衡见 §4。

---

## 3. 数据流与解包位置

### 3.1 设计 A（推荐先做）：SD 侧 INT4，ION 槽保持 INT8
```
pf_worker（预取线程）：
  读 layerN_i4.bin 到 io_buf（256KB 分块）        ← SD 读，字节减半
  → 逐块解包 int4→int8，直接写入 ION 权重槽        ← CPU 解包
主线程：计算 Bank A 时，预取线程同时做上面的读+解包到 Bank B
```
- ION 槽仍是 int8（3.54MB/槽），**ION 占用不变**（20.3MB，6 槽）。
- 预取线程总耗时 = 读(3×~0.63MB@21.9=260ms) + 解包(3×~10ms=30ms) ≈ 290ms。
- 暴露 = 290 - 133(计算) = ~157ms/批 → Wt load ≈ 首批 290ms + 9×157 ≈ **1.7s**。
- 若解包向量化达 ~500MB/s，解包 ~15ms/批 → 暴露 142ms → **~1.55s**。
- 优点：改动集中在 pf_worker + 首载路径；无需重排 ION；风险最低。
- 缺点：不解 ION，无上下文增益。

### 3.2 设计 B（Phase 2）：ION 侧 INT4 + 每层 JIT 解包
```
- ION 槽存 int4（1.77MB/槽 + scales），6 槽 = ~11.4MB → 释放 ~9MB ION
- 释放空间 = 1 个 int8 解包工作槽（3.54MB）+ ~5.5MB 额外 KV
- 主线程每层 matmul 前：把该层 int4 从槽解包到 int8 工作槽（~10ms/层）
```
- 计算变 133+30=163ms/批 > 读 260ms → 流水线变计算受限，读完全隐藏 → Wt load ~首批 290ms + 少量 ≈ **~0.4-0.6s**。
- 收益：Wt load 更低 + **KV 预算 +5.5MB** → 上下文大幅提升（配合 INT8 KV）。
- 代价：解包进入关键路径（30 层×10ms=300ms/step 增加计算）；需重排 ION 布局与 `sm_setup_ptrs`/bank 逻辑；复杂度高。
- 建议：先落地 A 拿 Wt load 收益，B 作为下一期（需 TPU 工程师协助验证 neuron/ION 重排）。

---

## 4. 精度评估（INT4 量化误差）

- 输入是**已 INT8 量化的权重**（值域 -128..127），再量化为 int4+组 scale。
- 每值相对误差 ≈ scale/2 / |v|；组内最坏 ~1/16，RMS 约 1-2%（依赖分布）。
- G 越小误差越小、开销越大；G=64+fp16 是精度/开销合理点。
- 对 SmolLM2 这类 FP32 精度需求不高的 135M 模型，INT4 权重误差通常可接受
  （参考社区 INT4 部署普遍无损可用）。需在板端用 next_token/困惑度做验收：
  **next_token=5021 保持、logits top-5 相对顺序不变、gap 无明显收窄**。
- 备选：若 INT4 质量不达标，可退 INT4/INT8 混合（敏感矩阵如 Wo/rms 相关保持 INT8，
  FFN 用 INT4），灵活调。

---

## 5. 原型代码框架（host 侧，`int4_proto.c`）

独立程序，不依赖 Duo/TPU，用于：
1. 生成随机 int8 矩阵 → 打包 int4 → 解包 → 校验最大/RMS 误差。
2. 测量解包吞吐（x86 代理值；板上 C906 需上板后测）。
3. 提供 `int4_pack()` / `int4_unpack()` 两个可直接移植到引擎的纯函数。

接口草案：
```c
/* 打包：把 row-major int8 矩阵压缩为 int4 包
 * out 容量 = n/2 + (n/G)*4 字节 */
void int4_pack(const int8_t *src, int n, int G,
               uint8_t *out_nib, float *out_scale, int *out_scales);
/* 解包：恢复为 int8（反量化）
 * src_nib / src_scale 由 pack 产出 */
void int4_unpack(const uint8_t *src_nib, const float *src_scale,
                 int n, int G, int8_t *dst);
```

引擎侧接入点（后续，设计 A）：
- `pf_worker()`（L784）：读 `layer%d_i4.bin`，每 256KB 块 `int4_unpack` 到 ION 槽。
- 首载路径 `pool_load_embed_and_init_layers()`（L1296）：同样改读 i4 + unpack。
- 需新增一个层文件选择开关（如 `WT_INT4=1` 环境变量），默认仍走 int8，先不接生产。

需要 TPU 工程师配合的算子：
- bmk1822 无原生 INT4 matmul。方案一（设计 A）：**解包用 CPU 向量化**（C906 vector，
  由我实现，不需要 TPU 算子）。方案二：若希望解包走 TPU，可用 **elemwise**
  （nibble 拆分 + 移位 + 乘 scale）实现，但 CPU 解包更简单且已足够快。
  → 建议：CPU 解包，TPU 侧无需新算子。

---

## 6. 验收口径（上板后）
- 精确命令回归：`/root/smollm2_pool_b2 ... 3 3 2`，对比 Wt load。
- 正确性：next_token=5021、tokens 序列合理、EXIT=0。
- 性能：Wt load 3.4s → ≤1.7s；decode/tok 相应下降。
- 记录前后数据回报 CEO，不提交 git。
