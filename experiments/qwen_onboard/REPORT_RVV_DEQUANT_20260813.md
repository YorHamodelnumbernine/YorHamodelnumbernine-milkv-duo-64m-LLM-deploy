# Path A 反量化 RVV 化完成（dequant_kal）

日期：2026-08-13 | 作者：推理引擎工程师 | 关联：`dequant_kal.c` / `dequant_kal_bench.c`
状态：**RVV 内核 bit-exact + 上板实测 378-384MB/s，per-token 0.47s**（CEO 裁决①高优项达成）
护栏：untracked（experiments/qwen_onboard/），不碰 master。

---

## 0. 结论

| 项 | scalar | RVV | 加速比 |
|---|---|---|---|
| N=896（q/k/v/o/up/gate） | 49.7 MB/s → 3.60 s/token | **377.9 MB/s → 0.47 s/token** | 7.6x |
| N=4864（down_proj） | 33.7 MB/s → 5.31 s/token | **384.0 MB/s → 0.47 s/token** | 11.4x |
| 正确性（RVV vs scalar） | — | **ALL EXACT（bad=0）** | — |

**合并预算**：C 流水 3.4s + RVV dequant 0.47s ≈ **3.9s/token CPU+TIU，SD 9.18s 地板余量 5.3s**。

## 1. 内核设计（dequant_kal.c）

- 布局：`nib[N][16]`（列 n 的 16 字节含 K=2j 低 / K=2j+1 高 nibble），输出 `w[32][N]` K-major。
- 策略：对每个 j（行对），**stride=16 的 `vlse8` 跨列 gather** byte j，双 nibble 符号扩展
  （`(v<<4)>>4` 低、`v>>4` 高，与 int4_common.c 语义一致），**连续 store** 到 2j/2j+1 行。
- LMUL=8（VLEN=128bit → 128 int8/向量）。
- **列分块 CB=256**：大 N 时全宽 gather 跨 77KB（down_proj）打爆 L2 → 分块后工作集 4KB 常驻，
  N=4864 从 130→384MB/s。
- 与 scalar `dequant_block`（qwen_m1_chunk.c）逐元素一致，可直接作为 M2 引擎 dequant 内核。

## 2. 调试记录（两个坑）

1. **LMUL 混淆**：`vint8m1_t` 在 VLEN=128bit 下仅 16 元素，误用循环步长 128 → 每 128 中 112 个
   漏写（r=0）。改 `vint8m8_t`（128 元素）后 bit-exact。
2. **N=4864 gather 超缓存**：stride=16 全宽 gather 跨 77KB → 列分块修复。

## 3. 复现

```
cd experiments/qwen_onboard
riscv64-unknown-linux-musl-gcc -mcpu=c906fdv -march=rv64imafdcv0p7xthead -mcmodel=medany \
  -mabi=lp64d -O3 -std=gnu11 -fsigned-char -o dequant_kal_bench dequant_kal_bench.c -lm -s
python3 ~/Documents/MilkV_duo_project/duo_run.py dequant_kal_bench
```

## 4. 下一步

- `dequant_kal_rvv` 合入 M2 引擎（SD 流式 → RVV dequant → ION → g2l → TIU 两遍法）。
- N-tile 大块（down_proj）已无瓶颈；LM Head 大矩阵（V=151936）后续单独验证。
