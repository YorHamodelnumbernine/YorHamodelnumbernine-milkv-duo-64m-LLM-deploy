# Qwen INT4 设计 B — ION 侧 INT4 驻留 + JIT 解包 / 布局重排（dequant 单调用成本削减）

日期：2026-08-22 | 作者：TPU 底层工程师 | 状态：**离线设计草稿（未上板、未改生产代码）**
定位：与推理引擎工程师「设计 A（dequant 调用减半）」互补；本设计针对 **dequant 每次调用成本**（stride-16 gather）与 **ION 侧 INT4 布局**。
关联：`dequant_kal.c`（RVV 内核）/ `DESIGN_QWEN_INT4_A_DEQUANT_20260822.md`（设计 A）/ `qwen_engine_lmhead2.c` eng_matmul
硬约束沿用：ION 余量 1.85MiB；**设计不新增 ION 消费者**（除非复用既有 carveout 区）。

---

## 0. 结论摘要

1. **现状瓶颈**：`dequant_kal_rvv` 占 eng_matmul wall 58.2%（136.8s/235.0s，6336 calls，CHAT run，设计 A §0 实测）。单次调用成本高，结构根因是 **INT4 源布局 `nib[N][16]`（列主序）迫使 RVV 用 `vlse8` stride-16 gather** 跨列取字节，cache-line 利用率低（128B 负载触碰 ~32 条 cache line）。
2. **方案 B2（推荐）**：将 INT4 nibble 源布局重排为 **`nib16[16][N]`（行对主序，row-pair-major）**——每 K-block 内 16 个 row-pair 平面，每平面 N 字节连续。解包时输入改 `vle8` 连续载入，输出保持 K-major INT8（写 DQ 不变）。
   - **位精确性构造保持**：DQ 中 [32,N] K-major INT8 逐字节不变，仅源字节排列改变 → TIU 右操作数不变 → 全回归 bit-exact 可保。
   - **不新增 ION**：DQ 区复用；源文件换布局为宿主侧一次性转换（`layer{l}_kal.bin` 重排），板上只读。
   - 预期：单次 dequant 成本降（gather→load，cache 触碰 ~16×↓），与设计 A（调用减半）相乘 → dequant 总占比 58% → ~15-25%。
3. **方案 B1（次选，有界扩展）**：在 ION 余量内驻留**少量高频矩阵的 INT4**（复用 SD_BUF/释放 DQ 区），削 SD 重读。受 carveout 26.8MB 近满限制，仅对 ~2-3 个大矩阵可行，收益边际；与 B2 布局兼容（驻留即重排后布局）。
4. **决策点（待 CEO）**：B2 是否需要同时换 `layer{l}_kal.bin` 物理布局（宿主重转 201MB，一次性），还是运行时 JIT 转置（板上每层一次，~ms 级，存 DDR 临时区）。

---

## 1. 现状与根因

### 1.1 权重格式与读路径

- 层文件 `layer{l}_kal.bin`：**8,393,728 B/层**（×24 ≈ 201MB）。
  - INT4 nibble：**7,454,720 B**（Wq/Wk/Wv/Wo/up/gate/down 七矩阵，G=32；早前 7,444,720 为笔误，经 `convert_qwen_kal_b2.py` 逐段复核修正）。
  - gsc（fp16 scale）：931,840 B/层。
  - norms/bias：~15KB。
- 读路径：nibble 走 SD（mmap/pread/ion_db 双槽 SD_BUF 预取）→ `dequant_kal_rvv` 解包到 DQ（ION，`DQ_OFF` 4096，容量 159,744B）→ TIU matmul 右操作数 `{32,N} FMT_I8` K-major。
- gsc 走 ION 缓存（Tier 2：22 层 ION + 2 层 DDR，`GSC_ION_LAYERS=22`）。

### 1.2 dequant 单调用成本根因

当前布局 `nib[N][16]`（每输出列 16 字节，字节 j = K 行 2j/2j+1 的 raw int4）：

```c
/* dequant_kal_rvv 主循环（每 256 列块、每 j∈0..15）：*/
v = vlse8_v_i8m8(src + off*16, 16, vl);   /* stride-16 gather：byte j 跨列 */
lo = vsra(vsll(v,4),4); hi = vsra(v,4);
vse8(w + 2j*N + off, lo, vl);             /* 输出：行 2j 连续 store */
vse8(w + (2j+1)*N + off, hi, vl);
```

- **输入**：`vlse8` stride 16。128B 负载散布在 256 列 × 16B = 4KB 区间，约 32 条 64B cache line（每 line 命中 4 字节）。gather 指令吞吐低于连续 load，且 cache 利用率 4/64。
- **输出**：`vse8` 连续（行内 stride 1），无问题。
- 因此单调用成本集中在**输入 gather**。

---

## 2. 设计空间

### 2.1 B2：源布局重排（row-pair-major）— 推荐

**新布局**：每 K-block（32 行）内，`nib16[16][N]`（16 个 row-pair 平面，每平面 N 字节连续）：

```
旧 nib[N][16]：  col n 的 16 字节连续（字节 j = 行 2j/2j+1）
新 nib16[16][N]：plane j 的 N 字节连续（列 n 的低半字节 = 行 2j，高半字节 = 行 2j+1）
```

**解包内核改动**（`dequant_kal_rvv` 内层两行 + 签名扩展）：
```c
/* 新签名: nib = nib16 基址 (K-block g 起点 = nib + g*16*N), 列窗 [col_off, col_off+ncols). */
void dequant_kal_rvv(const uint8_t *nib, int N, int col_off, int ncols, int8_t *w);
for (int j = 0; j < 16; j++) {
    v = vle8_v_i8m8(nib + j*N + col_off + off, vl); /* 连续 load，128B → 2 cache line */
    lo = vsra(vsll(v,4),4); hi = vsra(v,4);          /* 符号扩展与旧内核完全相同 */
    vse8(w + 2j*N + off, lo, vl);
    vse8(w + (2j+1)*N + off, hi, vl);                /* 输出不变：K-major INT8 */
}
```
- **符号扩展必须沿用 `vsra(vsll(v,4),4)`/`vsra(v,4)`**：`v & 0xF` 会给出 0..15 而非 -8..-1（raw int4 8..15 表示 -8..-1），会破坏位精确性。设计早前草稿的 `v & 0xF` 为错误写法，已更正。
- **签名需扩展 (N, col_off)**：per-tile dequant（A2/A1 之前的形态）处理 `[32, tn]` 列块，而新布局平面跨全 N，列窗需由 `col_off` 定位。调用点改 `dequant_kal_rvv(nib + g*16*N, N, toff[t], tn[t], (int8_t*)(va+DQ_OFF))`。A1 merged 后 col_off=0、ncols=N（整 K-block），签名前向兼容。
- 位精确性：DQ 输出 `w[32][N]` 逐字节与旧布局解包结果**完全相同**（同一 raw int4 值、同一符号扩展、同一目标地址）。TIU 右操作数、g2l、cmdbuf 全部不变 → 全回归 bit-exact 构造保持。

**宿主侧转换（工具已完成并离线验证）**：`convert_qwen_kal_b2.py`（新独立文件，2026-08-22）。
- 输入 `weights_kal/layer{l}_kal.bin` → 输出 `weights_kal_b2/layer{l}_kal_b2.bin`，24 层 / 13.5s，同大小 8,393,728 B/层。
- 重排 = 纯 numpy 转置 `nib[KG,N,16].transpose(0,2,1) → nib16[KG,16,N]`；gsc/rms 逐字节透传。
- 校验（全 24 层通过）：① 非 nibble 段 gsc/rms 与输入逐字节 cmp==0；② 每 nibble 段旧/新布局 scalar-dequant 逐元素相等；③ 向量化解包另经与 `dequant_kal.c` 同语义的标量参考独立复核（layer0 全 7 种形状 KG=28/152 × N=128/896/4864，全部 True）。
- 板上文件与宿主源 sha256 逐字节一致（layer0/12/23 抽查）→ 转换源权威。
- 板上只读新布局，无 ION 增量。实施阶段：push `weights_kal_b2/layer{l}_kal_b2.bin` → 板端 `layer{l}_kal.bin` 原位替换（保留原文件备份）。

**层文件精确偏移表**（`parse_layer` 顺序；本工具逐段复核基准）：

| 段 | KG | N | nib 偏移 | nib 大小 | gsc 偏移 | gsc 大小 |
|---|---|---|---|---|---|---|
| rms_attn | — | — | 0 | 3,584 | — | — |
| Wq | 28 | 896 | 3,584 | 401,408 | 404,992 | 50,176 |
| Wk | 28 | 128 | 455,168 | 57,344 | 512,512 | 7,168 |
| Wv | 28 | 128 | 519,680 | 57,344 | 577,024 | 7,168 |
| Wo | 28 | 896 | 584,192 | 401,408 | 985,600 | 50,176 |
| up | 28 | 4,864 | 1,035,776 | 2,179,072 | 3,214,848 | 272,384 |
| gate | 28 | 4,864 | 3,487,232 | 2,179,072 | 5,666,304 | 272,384 |
| down | 152 | 896 | 5,938,688 | 2,179,072 | 8,117,760 | 272,384 |
| rms_ffn | — | — | 8,390,144 | 3,584 | — | — |
| 合计 | | | | 7,454,720 | | 931,840 |

**预期收益**：gather→load，cache 触碰 ~16×↓；RVV 0.7 无 gather 指令仿真开销。单调用成本预计 ↓~40-50%（需 `dequant_kal_bench.c` 实测校准）。

### 2.2 B1：ION 侧 INT4 驻留（有界）

- 目标：削 SD 重读（decode SD-bound ~9.37s floor 的最大项）。
- 约束：carveout 余量 1.85MiB，且 DQ/SD_BUF 已占 4.32MB。
- 可行子集：up/gate（各 2,179,072B nib）为最大矩阵，SD_BUF 已双槽预取。B1 仅在**合并 A1 merged cmdbuf + 释放 per-tile pool**后才有空间驻留 1-2 个整层 INT4。
- 结论：**单独 B1 收益边际**；建议作为 B2 之后的可选扩展（若 A1 释放 pool ION ≥1.7MB）。

### 2.3 B3：与设计 A 协同（调用减半 × 单调用降价）

- A1/A2 将每 K-block dequant 从 2×ntile → 1 次；B2 将每次成本降 ~40-50%。
- 相乘：dequant 总占比 58% → ~15-25%；CHAT 长 prompt prefill 整体 -10~20%（含 submit 降）。
- 落地顺序（CEO 2026-08-22 裁定）：**A2 → B2 → A1**。B2 宿主重排工具已先行完成（离线验证）；B2 内核改动（`dequant_kal_rvv` 签名 + 内层 2 行）在 A2 验证 + commit、`qwen_engine_lmhead2.c` 编辑权交接后进行。A1 复用 B2 的内核签名（col_off=0/ncols=N）。互不阻塞、不碰对方冻结文件。

---

## 3. ION / 内存约束

- B2：0 新增 ION（DQ 复用，159,744B 不变；源文件布局替换，大小不变 7,454,720B/层）。
- B1：需 carveout 释放后评估（A1 pool 回收 + CENT_ION=0 回退），预计 ≤2MB 驻留窗口。
- 上板实现前自查：ps 无 qwen/smollm 残留 + ION used=0（沿用 §6 既有约定）。

---

## 4. 风险与验收

**风险**
- B2 内核改动位精确性：构造保证，但仍须 `dequant_kal_bench.c` 随机数据 RVV vs scalar 逐元素校验（N=896/4864，含新签名 col_off/ncols）。
- **内核签名扩展**：`dequant_kal_rvv` 增加 (N, col_off) 参数 → `eng_matmul`/`eng_matmul_merged` 全部调用点需同步改（3 处）；属生产代码改动，实施须在 A2 交接后（CEO 序列 A2→B2→A1）。
- 宿主重排工具正确性：**已离线闭环**——offset 表逐段复核、gsc/rms 逐字节 cmp、旧/新 dequant 逐元素相等（对照 `dequant_kal.c` 标量参考），全 24 层通过。
- 板上文件替换：push `_kal_b2.bin` 前须备份原 `layer{l}_kal.bin`；替换后 gsc/rms 段 sha256 与基线一致。
- 若选 JIT 转置（不重排磁盘文件）：每层一次转置 ~ms，DDR 需 7.1MB 临时区（余量 ~5-6MiB，需确认）——CEO 已裁定走宿主重排，此项仅留档。

**验收口径（上板后）**
1. 正确性：3-prompt NEXT 3/3 + decode 流 + CHAT 75-token，TIU bad1/bad2/rbad=0 —— 与 INT8 KV 基线一致。
2. 性能：`dequant_kal_bench.c` 单调用吞吐提升；PROFILE 中 dequant 占 eng_matmul wall 58% → 目标 ~30%（B2）/~20%（B2+A）。
3. 内存：carveout used 不超基线；DDR anon 增量 = 0（B2）或 JIT 转置临时区（可选）。
4. 回归：`qwen_engine_int8kv`（HEAD b4e5358 之后）完整跑通，无 watchdog。

---

## 5. 待 CEO 决策

1. **B2 源布局落地方式**：宿主一次性重排 `layer{l}_kal.bin`（磁盘 201MB，读路径零开销） vs 板上 JIT 转置（不动磁盘，但每层 ~ms + 7.1MB DDR 临时区）。
2. **是否同步 A1**：B2 先（低风险）还是 A 先（设计 A 已定案）——建议 B2 先，因内核改动最小、可独立验收。
3. **B1 是否立项**：需 A1 释放 pool 后再评估，建议列为 B2 之后的增量。

---

## 6. 附：现状基线（复现）

```
PROFILE (CHAT M=1, int8kv 基线):
  eng_matmul wall : 234.973s (6336 calls)
  dequant_rvv     : 136.777s ( 58.2%)  [CPU dequant INT4->INT8]
  layer file      : 8,393,728 B/层 ×24 ≈ 201MB（nibble 7,454,720B + gsc 931,840B）
  ION carveout    : 26,810,? B，free ~1.85MB（gsc 22 层 19.6MB + SD_BUF 4.16MB + pools 1.7MB + centroid 1.75MB + DQ 0.16MB）
```
