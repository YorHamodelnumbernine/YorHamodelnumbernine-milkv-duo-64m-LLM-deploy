# Step ② 两遍法 per-chunk matmul（KG=32）引擎向重构 — 集成报告

日期：2026-08-13 | 作者：推理引擎工程师 | 关联：`twopass_matmul.c` + `test_twopass_matmul.py`
状态：**host 侧微内核重构完成，整数两遍语义 BIT-EXACT + 真实权重 fp32 累加验证通过**
护栏：全部 untracked（experiments/qwen_onboard/），不碰 master。

---

## 0. 结论摘要

CEO 裁决实施顺序第 ② 步（两遍法 per-chunk matmul 重构，KG=32）**已完成 host 侧重构**：

- 将 `qwen_kal_ref.c::chunk_matmul_twopass()` 拆分为**引擎 1:1 微原语**，映射到 C906B TIU
  cmdbuf 与 CPU 回读/累加：
  - `block_pass1()` → TIU pass1 cmdbuf（全 N-tile 批处理，rshift=rsafe）→ P1_BUF
  - `rshift_from_pass1()` → CPU 读 max|p1| → est → **r_opt（per-block 全局标量）**
  - `block_pass2()` → TIU pass2 cmdbuf（同 LMEM 复用，rshift=r_opt）→ P2_BUF
  - `block_accum()` → CPU fp32 累加 `out += p2 × 2^r × gsc[g,n]`
- **提交结构对齐 TPU Gate②**：同一 K-block 的**全部 N-tile 合并为一次 pass**（pass1 一批 → 读
  max → pass2 一批），r_opt 为 block 全局标量 → 640 submits/layer（q/k/v/wo/up/gate 各
  28 block×2 遍=56，down 152×2=304），与 CEO/TPU 设定一致。
- **接口按 CEO 指示设计**：M 统一（decode M=1 → 单标量 r_opt 注入 pass2 cmdbuf；prefill M>1
  同 per-block 标量语义，无需改接口）。

## 1. 验证数据（两关全过）

### 1a. 整数两遍语义（r_opt）全生产形状 BIT-IDENTICAL

| 形状 | r_opt | 判定 |
|---|---|---|
| [1,896]×[896,896] q | IDENTICAL | ✓ |
| [1,896]×[896,128] k/v | IDENTICAL | ✓ |
| [1,896]×[896,4864] up/gate | IDENTICAL | ✓ |
| [1,4864]×[4864,896] down | IDENTICAL | ✓ |
| M=3 / M=10 prefill | IDENTICAL | ✓ |
| 边界 [1,32]×[32,512] / [1,64]×[64,1] | IDENTICAL | ✓ |

→ **重构未改变两遍法整数语义**（rsafe / est / r_opt / pass1 / pass2 全同）。

### 1b. 真实 Qwen layer-0 权重（7 形状）fp32 累加 vs fp64 金标

| 矩阵 | 形状 | rsafe | r_opt 范围 | maxrel | absmax |
|---|---|---|---|---|---|
| q_proj | 896×896 | 5 | 4–5 | 5.8e-08 | 5.4e-07 |
| k_proj | 896×128 | 5 | 4–5 | 5.4e-08 | 3.5e-07 |
| v_proj | 896×128 | 5 | 4–5 | 5.8e-08 | 3.0e-08 |
| o_proj | 896×896 | 5 | 4–5 | 5.8e-08 | 5.9e-08 |
| up_proj | 896×4864 | 5 | 4–5 | 5.9e-08 | 1.1e-07 |
| gate_proj | 896×4864 | 5 | 4–5 | 5.9e-08 | 1.7e-07 |
| down_proj | 4864×896 | 5 | **0–3** | 5.9e-08 | 1.3e-07 |

- **真实数据 fp32 累加 maxrel ≈ 6e-8**（远低于 adversarial 随机最坏 ~2e-3；Qwen 权重动态
  范围小、r_opt 4–5 稳定）。相对 logit gap（需 >0.5）可忽略。
- **down r_opt 区间 [0,3] 实测有效**：部分 block pass1 max 极小 → r=0（`int8_round_div` r=0
  时 half=0 退化为 clip 语义），真实数据覆盖了该边界，未出错。

## 2. 引擎 submit 预算确认（与闸口②一致）

- 640 submits/layer（Ntile=896 批处理），×24 = 15,360/token。
- cmdbuf 预建：16 rshift 变体 × 7 shape；submit = select + patch rshift + Run（~15µs）。
- MemInvld 限定 P1/P2 输出区（~8KB）。
- SD 9.18s/token 掩盖 TIU run 3.37s + CPU 2.73s（见 DESIGN §9b）。

## 3. 关键决策

- **N-tile 批处理 per pass**：pass1 覆盖 block 全部 N-tile → r_opt 是 block 全局 max（非
  per-N-tile）。若误拆 per-N-tile 两遍，r_opt 变局部、pass2 可能饱和 → 与 reference 不一致。
  本重构锁死「per-block 全局 r_opt」语义。
- **fp32 累加**：引擎按 DESIGN §0 用 fp32；reference 用 fp64 做金标。已验证真实数据误差
  ~6e-8，无需改引擎为 fp64。
- **M 统一接口**：`twopass_matmul(ctx, r_opt_out)`，r_opt_out 可选输出（供测试/调试），
  C906B 版可丢弃。

## 4. 产物（untracked）

- `twopass_matmul.c`：引擎向微内核（host gcc 与 riscv64-unknown-linux-musl-gcc 同源）。
- `test_twopass_matmul.py`：两关验证（`python3 test_twopass_matmul.py`）。
- `REPORT_STEP1_PERROW_20260813.md`（Step ①，已完成）。

## 5. 下一步（step ③/④ 状态）

- **Step ③ converter K-aligned 重写**：`convert_qwen_kal.py` 已存在且产出字节级验证的
  weights_kal/（roundtrip 3/3、layer0 尺寸校验 OK）。待形式化：明确其为唯一 K-aligned 转换器
  入口、补充 k-aligned 布局自检断言。
- **Step ④ CPU per-group dequant + per-(g,col) scale 累加**：`block_accum`（per-(g,col) 累加）
  已在 twopass_matmul.c 落地验证；INT4→INT8 解包（host: unpack_mat / on-board: RVV
  int4_unpack_fixed_rvv 419MB/s）为引擎 build-out 项，待 C906B 上板。
