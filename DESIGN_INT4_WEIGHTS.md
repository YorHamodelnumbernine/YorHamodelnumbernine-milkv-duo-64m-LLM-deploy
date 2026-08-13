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

### 5.1 真实权重验证（2026-08-12，host，layer0.bin）

工具：`convert_i4.c`（layerN.bin → layerN_i4.bin，复用 `int4_common.c` 同一套
pack/unpack）。用设备上的真实 `layer0.bin`（3,543,552 B，md5 41a3b489…）验证：

| 项 | 结果 |
|---|---|
| 7 矩阵 packed 字节 | **1,880,064 B**（=1,769,472 nibble + 110,592 fp16 scale）✓ |
| 层总字节（含 rms f32） | **1,884,672 B**（较 INT8 3,543,552 B 省 47%）✓ |
| rms_attn / rms_ffn 透传 | 完全一致 ✓ |
| round-trip max_err | 9（全 7 矩阵）|
| round-trip RMS | 3.2 – 4.2（Wq 3.2 / Wk 3.3 / Wv 3.6 / Wo 3.2 / up 4.2 / gate 4.2 / down 3.8）|

> 说明：真实权重是满幅 per-channel INT8（值域近 ±127），组内动态范围大，
> G=64 下每值重建 RMS ~3-4% 属预期；下游 matmul 输出误差会随 N 平均，
> 实际 logits 扰动远小于权重 RMS。若验收 next_token 漂移，可退 G=32（scale 开销
> 升至 12.5%）或 INT4/INT8 混合（敏感矩阵保 INT8）。

---

## 6. 验收口径（上板后）
- 精确命令回归：`/root/smollm2_pool_b2 ... 3 3 2`，对比 Wt load。
- 正确性：next_token=5021、tokens 序列合理、EXIT=0。
- 性能：Wt load 3.4s → ≤1.7s；decode/tok 相应下降。
- 记录前后数据回报 CEO，不提交 git。

---

## 7. 最终结论（2026-08-13）：INT4 压缩路线判负 + convert bug 修正记录

**状态：实现并上板验证 → 质量不达标，正式搁置。生产路径维持 INT8。**

### 7.1 重要：convert_i4.c 布局 bug（先修后验）

设备端独立校验发现，首版 `layerN_i4.bin` 的 FFN 3 矩阵全部错位 **2304 字节**。
根因：convert 连续读取 7 个矩阵，但源 `layerN.bin` 布局在 `Wo` 与 `ffn_up` 之间
内嵌 `rms_ffn`（f32,2304B），导致第 4 个矩阵把 `[rms_ffn + ffn_up[:882432]]` 当作
`ffn_up` 打包，gate/down 依次错位；`rms_ffn` 被写到文件末尾而非中间。
attn 4 矩阵不受影响（rms 3.2），FFN 全为垃圾（rms 55）。

> ⚠️ **此前所有基于未修复 convert 的 INT4 质量结论（含 Design A 的 "next_token=45401 /
> gap 0.3"）均建立在该 bug 上，作废。** 已修复（按 INT8 布局读写 rms_ffn），全部 30 层
> 重新生成，设备端独立校验 FFN rms 降至 3-4。

### 7.2 修正后回归数据（同一二进制、同一输入，3 轮）

| 配置 | next_token | top 间隙 | Wt load | Total | 文件/层 |
|---|---|---|---|---|---|
| INT8 基线 | 5021 ✓ | 5.8 | 3741 ms | 42106 ms | 3,543,552 B |
| full-INT4 G64 | 3625（5021 为 #3 候选）| 3.4→0.6 塌缩 | 1283 ms | 26906 ms (−36%) | 1,884,672 B (−47%) |
| FFN-only G64 | 6271 ✗ | 4.2（后步）| 2212 ms | 31943 ms (−24%) | 2,299,392 B (−35%) |
| FFN-only G16 | 34346 ✗ | 2.4–2.7 | 3694 ms | 43299 ms (+3%) | 2,548,224 B (−28%) |

### 7.3 判负结论与根因

1. **FFN-only 判负**：G=16（计划内最小分组、误差最低）仍 next_token=34346≠5021、
   gap 2.4<3，且 G16 的 scale 开销使字节节省缩水到 28%，Wt load 无收益（+3% 劣化）。
2. **full-INT4 修正后仍判负**：有效权重下并非灾难性崩塌（5021 进入首步 top-3、gap 3.4），
   但后续 decode 仍塌缩（gap 0.6）、next_token=3625≠5021。
3. **根因（理论印证）**：per-channel INT8 权重已满幅（std 30-38），组 INT4 量化步长
   ≈max/7≈18，对 FFN 输出扰动过大；G 64→16 仅把误差 4.2→3.2，仍不足以稳定 next_token=5021。
   INT4 量化噪声在该模型（INT8 基线 gap 仅 5.8，本已边缘）上不可接受。
4. 生产路径 INT8 未受影响（next_token=5021 ✓ 验证），WT_INT4 开关保留但默认关闭。

### 7.4 对未来的启示

- **per-channel INT8 满幅权重 → 组 INT4 二次量化不可行**（无局部可压缩结构）。
  若要 INT4，需从原始 fp32 权重出发 + calibration/更细激活量化，且须先验证单层
  logit 扰动而非仅权重 RMS。
- **host 转换工具与设备端需交叉校验**：布局/尺寸语义（内嵌 rms_ffn 这类"段间非矩阵
  数据"）是易错点，转换器应按源布局结构逐段读写，而非连续矩阵扫描。
- Wt load 的压缩路径封死后，SD 读取（21.9MB/s 物理上限 + 101MB/token 刚性）已近
  理论最优（3.7s vs 理想 5.6s 的预取重叠），后续收益仅来自驻留缓存等边际优化。
