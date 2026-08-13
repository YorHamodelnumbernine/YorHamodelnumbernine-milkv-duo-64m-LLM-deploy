# INT8 KV 设计 — 上下文 39→~156 token（×4）

日期：2026-08-12
作者：推理引擎工程师
状态：**已实现**于 `feat/int8-kv` 分支（基于 96d1e5c），交叉编译通过，未上板
目标：把 KV cache 从 FP32 改为 INT8（每 token/层 1536B → 384B），
    在**相同 ION 预算**下把最大上下文从 ~39 提到 ~156 token（×4.0），
    与 INT4 权重叠加后可达 ~1000 token（见 §6）。

---

## 0. 结论先行（TL;DR）

- **改动点集中在 4 处**：`sm_kv_cache_t` 结构、`sm_kv_store_contig`/`sm_kv_load_contig`
  两个 helper、`sm_kv_alloc_ion` 的每层字节、attention 的 K/V 读取（3 处调用点）。
- **量化方案：per-token（每行）对称 INT8 + 1 个 float scale/token/KV**。
  - 写入：每 token 一行 dkv 个值，`sc = max(|行|)/127`，`q = round(v/sc)`。
  - 读取：`v' = q * sc`。
  - **不随时间累积误差**：每个 token 独立量化/反量化，无反馈回路。
- **误差：相对现状增加约 1 次 INT8 往返（~0.5–1% RMS）**。K/V 本源就是 INT8 matmul
  输出反量化而来，精度天花板本来就是 INT8；再往下游 attention 本来就要按组重新量化
  成 INT8 打分，INT8 KV 的实际精度损失可忽略（详见 §5）。
- **无需新增 TPU 算子**：量化/反量化全在 CPU（C906）上做，每 token 仅 dkv 次乘加，开销 ~µs 级。
- 兼容现有 `compute_scale_sym`/`quantize_i8_sym` helper，直接复用。

---

## 1. 现状 KV 布局（要改的基准）

`smollm2_pool_demo.c` 当前：

- `sm_kv_cache_t`（L494-496）：`float *K[30], *V[30]`，全部放 ION。
- `sm_kv_alloc_ion`（L1226）：每层 `per_layer = max_seq * dkv * sizeof(float)`，
  K、V 各一份 → 30 层合计 `2*max_seq*dkv*4` 字节。
  - `max_seq=39` 时 `kv_total = 1,797,120 B ≈ 1.71 MB`。
- 写入（L601-604）：`K_f32`（已 dequant + RoPE）→ `memcpy` 到 cache。
- 读取（L622-627 / L654-657）：从 cache `memcpy` 到 `Kh_f32/Vh_f32`，
  再按组 `compute_scale_sym` + `quantize_i8_sym` 打成 INT8 供打分/输出 matmul。

每 token/层字节：`2 × 192 × 4 = 1536 B` → 30 层 = **46,080 B/token** →
1.7MB 预算 ≈ **39 token**（实测 max_seq=39）。

---

## 2. INT8 KV 存储设计

### 2.1 布局

```
[ K_int8[30] : 每层 max_seq * dkv 字节 ]   ← ION（连续，256B 对齐）
[ V_int8[30] : 每层 max_seq * dkv 字节 ]   ← ION
[ K_scale[30]: 每层 max_seq * 4 字节  ]   ← heap（DDR），每 token 1 个 float
[ V_scale[30]: 每层 max_seq * 4 字节  ]   ← heap（DDR）
```

- **scale 放 DDR heap 而不是 ION**：体积小（max_seq*2*4 B/层，156 token 时全层仅 37KB），
  不占 ION 预算；读写在 CPU 侧，DDR 更快更简单。
- ION 只放 INT8 K/V：`per_layer = max_seq * dkv * 1`，K+V 两层。

### 2.2 写入（新增 `kv_store_i8`）

```c
/* 替换 L506 sm_kv_store_contig */
static void kv_store_i8(int8_t *cache, float *scale, const float *new_data,
                        int seq, int pos, int dkv) {
    for (int s = 0; s < seq; s++) {
        const float *row = new_data + s * dkv;
        float sc = compute_scale_sym(row, dkv);          /* 复用现有 helper */
        quantize_i8_sym(cache + (pos+s) * dkv, row, dkv, sc);
        scale[pos + s] = sc;
    }
}
```

调用点（L603-604）改为：
```c
kv_store_i8(kv->K[layer], kv->K_s[layer], K_f32, seq, pos, dkv);
kv_store_i8(kv->V[layer], kv->V_s[layer], V_f32, seq, pos, dkv);
```

### 2.3 读取（新增 `kv_load_i8`，反量化）

attention 里两处读：

- K 读（L622-627），每组 g 取列区间 `[g*d, g*d+d)`：
```c
for (int s = 0; s < kv_len; s++) {
    float sc = kv->K_s[layer][s];
    const int8_t *src = kv->K[layer] + s * dkv + g * d;
    for (int c = 0; c < d; c++) Kh_f32[s*d + c] = (float)src[c] * sc;
}
```
- V 读（L654-657）：同样方式从 `kv->V` / `kv->V_s` 填 `Vh_f32`。

（`sm_kv_load_contig` 若无其他调用点可直接删除。）

### 2.4 结构体与分配（L494-496、L1226、L500）

```c
typedef struct {
    int8_t *K[30], *V[30];   /* ION: int8 */
    float  *K_s[30], *V_s[30]; /* heap: per-token scale */
} sm_kv_cache_t;
```

`sm_kv_alloc_ion`：
- `int per_layer = max_seq * dkv * 1;`（INT8）
- K/V 指针偏移不变（仍是 `l*2*per_layer_aligned` / `(l*2+1)*per_layer_aligned`）。
- 在 heap 上 `calloc` 两组 scale 数组：`2 * n_layers * max_seq * sizeof(float)`。

`sm_kv_free`（L500）：在 `free(kv)` 前释放 4 组 scale 数组（K_s/V_s × 30）。

### 2.5 ION 预算变化

| max_seq | FP32 现状 | INT8 KV | 差值 |
|---|---|---|---|
| 39 | 1,797,120 B（1.71MB）| 449,280 B（0.43MB）| **释放 1.35MB** |
| 156 | 7,194,240 B（6.86MB）→ 超预算 | 1,797,120 B（1.71MB）| 与现状 39 相同 |

→ **不增加 ION 占用即可把 max_seq 提到 ~156**；`kv_start` 更小、`ion_free` 更大，
给权重槽/embed 留出空间（mode 3 判定更从容，见 `pool_calc_pipeline_mode` L1378）。

---

## 3. 代码改动点清单（精确到行）

| # | 位置 | 改动 |
|---|---|---|
| 1 | `sm_kv_cache_t`（L494-496）| `float *K/V[30]` → `int8_t *K/V[30]` + `float *K_s/V_s[30]` |
| 2 | `sm_kv_store_contig`（L506）| 替换为 `kv_store_i8`（每行 scale + 量化）|
| 3 | `sm_kv_load_contig`（L509）| 替换为 `kv_load_i8`（按行 scale 反量化）或删 |
| 4 | `sm_kv_alloc_ion`（L1226-1247）| `per_layer` 去掉 `*sizeof(float)`；heap 分配 scale 数组 |
| 5 | `sm_kv_free`（L500-505）| free scale 数组 |
| 6 | 写入调用（L603-604）| 改调 `kv_store_i8` |
| 7 | K 读取（L622-627）| 反量化循环代替 memcpy |
| 8 | V 读取（L654-657）| 反量化循环代替 memcpy |
| 9 | `main()` max_seq 决策（L2122-2124）| 放开 `c.max_seq` 到目标长度（见 §6）|

**不需要改**：`sm_forward_pool` 签名、TPU matmul、ION 分配主体、权重布局。

---

## 4. 量化精度评估

### 4.1 误差来源

- K/V 在写入 cache 前是 **FP32**（由 INT8 QKV matmul 输出按 per-channel `lsc` 反量化而来，
  并已过 RoPE）。误差的"原生精度"本来就是 INT8 QKV 路径决定的。
- INT8 KV 额外引入 **1 次对称 INT8 量化/反量化**（per-token scale）。

### 4.2 误差量级

- 每 token 行：`sc = max(|行|)/127`，量化步长 = sc，误差 ∈ [-sc/2, sc/2]，
  RMS ≈ `sc/√12 ≈ 0.29·sc`。
- 若行内典型值 ~0.3·max，则相对 RMS ≈ `0.29·sc / (0.3·max) = 0.29/(0.3·127) ≈ 0.8%`。
- 典型 SmolLM2 K/V 分布近似零中心高斯，per-token 行动态范围稳定 → **RMS ~0.5–1%**。

### 4.3 为什么不累积

per-token scale 是**每个 token 独立量化+独立反量化**，读出来就是写进去的量化值×固定
scale，没有反馈、没有跨 token 依赖。**kv_len 从 39 涨到 156/1000，单 token 误差不变**，
不会像某些低秩/累积方案那样随长度漂移。

### 4.4 与现有 attention 的关系

下游 `sc_kh`/`sc_vh` 本来就把 K/V 按组重打成 INT8 做打分/输出 matmul。
INT8 KV 只是让这一步骤的输入从"精确 FP32"变成"±1% 的 INT8 反量化"，对
softmax 打分的影响远小于打分本身已有的 INT8 量化噪声 → **实际精度损失可忽略**。
验收仍以 next_token=5021 保持、top-5 相对顺序不变为准。

---

## 5. 工作量与风险

| 项 | 评估 |
|---|---|
| 工作量 | **小（~1 天）**：结构体+2 helper+2 读调用，无新算子 |
| 风险 | **低**：不碰 ION 分配主体、不碰 matmul、不碰权重 |
| CPU 开销 | 写：每 token `dkv` 次；读：每 token 每层 `dkv` 次乘加 → decode 每步每层 ~384 次，µs 级 |
| 副作用 | `ion_free` 增大 → `pool_calc_pipeline_mode` 可能自动升 mode；需回归确认 mode 判定 |
| 回归命令 | `/root/smollm2_pool_b2 /root/smollm2_instruct/ /root/input_tokens.bin 3 3 2` |

---

## 6. 上下文收益汇总

| 配置 | 每 token/层 | 30 层/token | KV 预算 | 最大上下文 |
|---|---|---|---|---|
| FP32 KV（现状）| 1536 B | 46,080 B | 1.7MB | ≈39 |
| **INT8 KV（本方案）** | **384 B** | **11,520 B** | 1.7MB | **≈156（×4）** |
| INT8 KV + INT4 权重（设计 B） | 384 B | 11,520 B | ~11.7MB | **≈1000** |

> INT4 权重（`DESIGN_INT4_WEIGHTS.md`）把 ION 权重槽 20.3MB → ~10.6MB，
> 释放 ~10MB 给 KV → INT8 KV 预算变 ~11.7MB → 上下文 ~1000 token。
> 两步叠加即"大幅提升上下文"的现实路径（见 MLA 调研结论）。

---

## 7. 验收口径（上板后）

1. 精确命令回归：`/root/smollm2_pool_b2 ... 3 3 2`，Wt load / decode 不劣化。
2. 正确性：next_token=5021、tokens 序列合理、EXIT=0。
3. 长上下文：`max_seq` 调到 ~150，喂 100+ token prompt，能正常 decode 且 top-1 稳定。
4. 记录前后数据回报 CEO，不提交 git。
