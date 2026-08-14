# M2 层前向集成 · layer0 全层上板位精确通过（qwen_engine_layer0）

日期：2026-08-13 | 作者：推理引擎工程师 | 关联：`qwen_engine_layer0.c` / `gen_layer0_ref.py` / `qwen_engine_layer0_ref.h` / `dequant_kal.c`
状态：**真实 layer0 全层 forward 上板三检查点全部 maxrel=0.000e+00、P1/P2 全块 bit-exact、r_opt 0 失配**（M2 层前向集成达成）
护栏：全部 untracked（experiments/qwen_onboard/），不碰 master。

---

## 0. 结论摘要

1. **M2 层前向闭环**：rms_attn→quant→q/k/v→bias→rope(0)→GQA attn→wo→residual→rms_ffn→
   quant→up/gate→SiLU→down(1024-chunk)→residual，全部矩阵走 TIU 两遍法，**三检查点
   （ref_attn / ref_after_wo / ref_after_ffn）maxrel=0.000e+00（位精确）**。
2. **up/gate N=4864 多 N-tile + block-shared r_opt 上板验证**：6 tile（5×896+384），
   pass1 跨全部 tile 收集 block_max → r_opt → pass2，与 host 参考位精确一致。
3. **down K-chunk 1024 + 块内 G=32 反量化流水**：5 chunk（4×1024+768），逐 chunk per-row
   量化 mid 切片，上板通过。
4. **发现并修复 1 个真实 bug**：down gsc 指针 `(kc/G)*D*2`（uint16_t 指针上多乘 2）→
   越过 layer 缓冲区末尾读 → SIGSEGV。修为 `(kc/G)*D`（半字单位），全层通过。

## 1. 验证内容（layer0, token 105538 = P0 prompt1 首 token）

```
x = embed[105538] * esc[105538]                (流式读单行, 不加载全 embed)
h = rms_norm(x, rms_attn)                       vs tq_act maxrel=3.652e-06 (fp32 排序差异, 量化吸收)
xi, sc_row = per_row_quant(h)
q,k,v = TIU two-pass [896x896 / 896x128 / 896x128] + bias
attn = GQA head map (seq=1, kvh=hh//7)          vs ref_attn maxrel=0
ai, sc_a = per_row_quant(attn)
wo = TIU two-pass [896x896]; x += wo            vs ref_after_wo maxrel=0
h2 = rms_norm(x, rms_ffn); x2i, sc2 = quant
up, gate = TIU two-pass [896x4864 x6tile]; mid = up*silu(gate)
out = Σ_chunk TIU down [1024/768 x 896]; x += out   vs ref_after_ffn maxrel=0
```

## 2. 上板实测原始数据（CV1800B, real layer0 weights）

| 阶段 | 结果 | 耗时 |
|---|---|---|
| QKV TIU | bad1=0 bad2=0 rbad=0 | 0.147 s |
| attn vs ref_attn | **maxrel=0.000e+00 maxabs=0.000e+00** | — |
| wo TIU | bad1=0 bad2=0 rbad=0 | 0.078 s |
| after_wo vs ref | **maxrel=0.000e+00 maxabs=0.000e+00** | — |
| up/gate TIU | bad1=0 bad2=0 rbad=0 | 0.558 s |
| down TIU (5 chunk) | bad1=0 bad2=0 rbad=0 | 0.269 s |
| after_ffn vs ref | **maxrel=0.000e+00 maxabs=0.000e+00** | — |
| P1/P2 bit-exact | bad1=0 bad2=0（全块）r_opt 失配=0 | — |

**示例值**：q[0..3]=-0.034317 0.006528 -0.168114 -0.046644 · k[0..3]=-8.352914
-2.933614 -6.299231 0.897643 · v[0..3]=0.007584 0.011139 -0.023518 -0.024841 ·
sc_row=0.017697。

注：耗时含 host 逐块 acc 参考核对（校验用），非引擎流水线代表值；仅作对照。

## 3. 关键实现决策与 bug

### 3.1 up/gate block-shared r_opt（N-tile 6 片）
与 emu_group_int4 参考的 r_opt 定义一致：r_opt 由该 K-block 在**全部 N 列**上的
pass1 max 决定（跨 tile 收集），非 per-tile。per-tile r_opt 与参考不位精确。
上板 r_opt 失配=0 证实该决策正确。

### 3.2 down gsc 指针 bug（已修）
`down_gsc` 是 `uint16_t*`，chunk 起始指针原写为 `+(kc/G)*D*2` → 半字单位下多乘 2，
使 kc≥2048 的 chunk 读越过 layer 缓冲区（8,393,728 B）末尾 → SIGSEGV。
修为 `+(kc/G)*D`（半字）。nib 指针（uint8_t）`+(kc/G)*D*16` 正确，未动。

### 3.3 形状池
3 个预建池（Nshape=128/384/896），各含 g2l + 16×2 pass cmdbuf，psize-only
单 LoadDmabuf（沿用 REPORT_M2_TIU_CORE 铁律）。k/v→128、up/gate 尾片→384、
其余→896。

## 4. 与设计文档对照

- IMPL_C906B_MATRIX §4：up/gate N-tile=896（6 片）实际档 → 上板通过，1800 runs/layer
  结构与真实提交量一致（q/k/v/wo 单 tile、up/gate 6 tile、down 152 K-block）。
- DESIGN §9b 两遍法：pass1 rsafe → 回读 max → r_opt → pass2，逐矩阵一致。
- 上板无 ps32、无 per-group INT8 降级，纯 A' INT4 G32 → CPU RVV 反量化 → TIU 两遍。

## 5. 下一步（M2 门禁路径）

1. **24 层 + LM head 流式 embed + 3-prompt 回归**：layer0 骨架平铺到 layer0..23，
   decode 增量 + KV cache；门禁 min gap<0.05 + per-token ≈15.4s。
2. LM head 两段式 top-k 留 Phase 6（CEO 已裁定不阻塞 Phase 5）。
3. 回归比对 host qwen_kal_ref（C 参考）top-5 + 附原始 logits 走 TPU 独立复核。

## 6. 复现

```
cd experiments/qwen_onboard
python3 gen_layer0_ref.py                     # 重新生成 qwen_engine_layer0_ref.h
# riscv64 交叉编译（见 qwen_engine_layer0.c 头注释）
python3 ~/Documents/MilkV_duo_project/duo_run.py qwen_engine_layer0
# 依赖：/data/qwen/layer0_kal.bin + embed_i8.bin + embed_scales.f32 + layer0_bias.f32 已部署
```
