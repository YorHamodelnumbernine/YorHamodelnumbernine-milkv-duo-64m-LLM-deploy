# TPU Benchmark & SmolLM2-135M Inference on Milk-V Duo

CV1800B (C906B @1GHz + C906L @700MHz) TPU/NPU benchmark toolkit and
SmolLM2-135M dual-core inference engine.  Built on the
[cvitek-tdl-sdk](https://github.com/milkv-duo/duo-buildroot-sdk).

## Directory Layout

```
tpu_bench/
├── common/                     # Shared headers
│   ├── tpu_bench.h             #   TPU init / matmul / utility wrappers
│   ├── mha_descriptor.h        #   Multihead-attention shared layout
│   └── rtos_cmdqu.h            #   Mailbox command-queue userspace defs
├── smollm2_pool_demo.c         # ★ Main inference: dual-core pipelined
├── smollm2_demo.c              #   Legacy single-core inference
├── convert_smollm2.py          # Convert HF weights → flat INT8 binary
├── deploy_v2.py                # Deploy weights & binary to Duo over SSH
├── smollm2_tokenize.py         # Tokenizer helper
├── gen_rand_weights.c          # Generate random weights for testing
├── gen_scales.py               # Generate quantization scales
├── gen_report.py               # HTML report generator
├── run_all.py                  # Run all micro-benchmarks
├── Makefile                    # Cross-compilation rules
├── transformer_demo.c          # FP32 transformer reference
├── transformer.h               # Transformer block definitions
├── mha_*.c                     # MHA attention variants (SD-card, LUT, etc.)
├── 01_conv/ .. 07_pooling/     # TPU micro-benchmarks by operation
└── SUMMARY.md                  # Historical development notes
```

## SmolLM2-135M Inference (`smollm2_pool_demo`)

### Model config

| Param | Value |
|-------|-------|
| d_model | 576 |
| n_heads | 9 |
| n_kv_heads | 3 |
| head_dim | 64 |
| n_layers | 30 |
| ffn_hidden | 1536 |
| vocab_size | 49152 |
| Weight total | ~101 MB (INT8 + FP32 norms) |

### Architecture

```
┌──────────────────────────────────────────────────────┐
│                   Linux C906B @1GHz                   │
│  ┌─────────┐   ┌──────────┐   ┌──────────────────┐   │
│  │ SD Card │──▶│ DDR pool │──▶│ ION carveout     │   │
│  │ weights │   │ 8 MB     │   │ 24 MB (TPU-visible)│   │
│  │ embed   │   │ dual-buf │   │ embed cache +     │   │
│  └─────────┘   │ staging  │   │ layer slots       │   │
│                └──────────┘   └───────┬────────────┘   │
│                                       │ TPU matmul     │
│  ┌──────────────────────────────────┐ │                │
│  │        Mailbox (0x01900000)      │ │                │
│  └──────────────┬───────────────────┘ │                │
└─────────────────┼─────────────────────┼────────────────┘
                  │                     │
┌─────────────────┼─────────────────────┼────────────────┐
│          FreeRTOS C906L @700MHz       │                │
│  ┌──────────────▼─────────────────────▼──────────────┐ │
│  │  CMD_MHA_EMBED_XPOSE (0x28)                       │ │
│  │  uint8 row-major → int8 col-major transpose       │ │
│  └───────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────┘
```

### Memory layout (64 MB DDR)

```
Physical address space:
  0x80000000 ──────────────────── DDR start (64 MB)
  0x82473000 ──────────────────── neuron memory (1 MB, TPU work buffer)
  0x82573000 ──────────────────── ION carveout (24 MB, TPU-visible)
  0x83F40000 ──────────────────── FreeRTOS firmware (768 KB)
  0x84000000 ──────────────────── DDR end

ION layout (24 MB):
  [embed cache: ~17.6 MB] [layer slot 0: 3.38 MB] [layer slot 1: 3.38 MB]
   ↑ preserved across decode    ↑ alternating TPU compute targets
   
DDR layout (8 MB):
  [ddr_buf[0]: ~3.5 MB] [ddr_buf[1]: ~3.5 MB] [io_buf: 256 KB]
   ↑ current layer             ↑ next layer (SD pre-read)
```

### Dual-core pipeline

```
Layer loop (30 layers per token):
  ┌─ SD read layer N+1 → ddr_buf[next]  ─┐  parallel
  └─ memcpy ddr_buf[cur] → ION slot      ─┘
  ┌─ TPU compute layer N                 ─┐  serial after DDR→ION
  └─ swap cur ↔ next                     ─┘

LM Head (48 vocab chunks of 1024 tokens):
  ┌─ Pre-read chunk i+1 from ION embed / SD  ─┐
  └─ CMD_MHA_EMBED_XPOSE (sec-core transpose) ─┘ parallel
  ┌─ TPU matmul(dst_buf[cur])                ─┐
  └─ Dequantize logits                       ─┘
```

### Secondary-core commands

| Cmd ID | Name | Status | Description |
|--------|------|--------|-------------|
| 0x20 | MEMCPY | working | Memory copy in shared space |
| 0x21 | MEMSET | working | Memory set |
| 0x22 | CACHE_FLUSH | working | Cache flush |
| 0x23 | CACHE_INVLD | working | Cache invalidate |
| 0x24 | QUANTIZE | working | FP32→INT8 quantization |
| 0x25 | TRANSPOSE | working | Matrix transpose |
| 0x26 | DEQUANTIZE | working | INT8→FP32 dequantization |
| **0x27** | **DDR_TO_ION** | **disabled** | DDR→ION async memcpy (needs DDR phys addr — pagemap unavailable) |
| **0x28** | **EMBED_XPOSE** | **working** | Embedding chunk transpose (uses ION phys addr) |

## Build

### Prerequisites

- Cross-compiler: `riscv64-unknown-linux-musl-gcc` (from [duo-examples](https://github.com/milkv-duo/duo-examples))
- TPU SDK: `cvitek-tdl-sdk-cv180x` (from [duo-buildroot-sdk](https://github.com/milkv-duo/duo-buildroot-sdk))

### Compile

```bash
cd tpu_bench
# Edit Makefile: set SDK_ROOT and CROSS_COMPILE paths
make smollm2_pool_demo       # Main inference binary
make smollm2_demo            # Legacy single-core binary
make                         # All benchmarks
```

### Rebuild FreeRTOS firmware (if modifying secondary-core commands)

```bash
cd duo-buildroot-sdk/freertos/cvitek
export PATH="/path/to/riscv64-elf-x86_64/bin:$PATH"
export MV_BOARD="milkv-duo-sd"
export BUILD_PATH="/path/to/duo-buildroot-sdk/build"
export BUILD_ENV_PATH="$BUILD_PATH"
export DDR_64MB_SIZE="y"
./build_cv180x.sh

# Assemble fip.bin
cd duo-buildroot-sdk/fsbl
python3 plat/cv180x/fiptool.py -v genfip \
    build/cv1800b_milkv_duo_sd/fip.bin \
    --MONITOR_RUNADDR="0x80000000" \
    --BLCP_2ND_RUNADDR="0x83f40000" \
    --BLCP_2ND="/path/to/cvirtos.bin" \
    ... (see fip_v2.mk for full args)

# Flash to Duo
cat fip.bin | ssh root@192.168.42.1 "cat > /boot/fip.bin && sync && reboot"
```

## Convert & Deploy Weights

### Convert SmolLM2-135M weights

```bash
pip install safetensors numpy
python3 convert_smollm2.py \
    --model "HuggingFaceTB/SmolLM2-135M-Instruct" \
    --out ./smollm2_weights
```

This produces per-layer combined `layerN.bin` files and `embed.i8`.

### Merge individual weight files into combined layers (on Duo)

```bash
# Compile merge tool
riscv64-unknown-linux-musl-gcc -O3 -s -o merge_layers merge_layers.c
# Deploy and run on Duo
cat merge_layers | ssh root@192.168.42.1 "cat > /tmp/merge_layers && chmod +x /tmp/merge_layers"
ssh root@192.168.42.1 "/tmp/merge_layers /root/smollm2 /root/smollm2_pool"
```

### Deploy binary & weights

```bash
# Push binary
cat smollm2_pool_demo | ssh root@192.168.42.1 "cat > /root/smollm2_pool_demo && chmod +x /root/smollm2_pool_demo"

# Push weights (if not already on SD card)
scp -r smollm2_weights root@192.168.42.1:/root/smollm2_pool/
```

## Run Inference

```bash
# On Duo (via SSH)
ssh root@192.168.42.1

# Format: smollm2_pool_demo <weight_dir> <token_ids.bin> <max_new_tokens>
/root/smollm2_pool_demo /root/smollm2_pool /root/test_tokens.bin 20
```

### Create test tokens

```python
from smollm2_tokenize import tokenize
token_ids = tokenize("Hello, how are you?")
# Write as int32 binary
import struct
with open("test_tokens.bin", "wb") as f:
    for t in token_ids:
        f.write(struct.pack("<i", t))
```

## Performance (CV1800B, Aug 2026 — final)

Precise regression: `/root/smollm2_pool_demo <weights> <tokens.bin> 3 3 2`

| Metric | Value | Notes |
|--------|-------|-------|
| Prefill (36 tok) | ~22,200 ms | chunked prefill |
| Decode per token | ~4,900 ms | 30 layers, dual-core pipelined |
| LM Head | ~789 ms | DDR=2MB embed + async sec-core transpose overlap |
| Weight load | ~3,700 ms | SD @ 21.9 MB/s (physical cap, near optimal) |
| Total (36 in / 4 out) | ~41,800 ms | 3-round median, next_token=5021, EXIT=0 |
| Context length | 39 → 156+ tokens | INT8 KV cache (×4) |

### Landed optimizations (Phase 3-4)

1. **DDR embed=2MB default** (variant c): Total −11.7%, LM_Head −33% (1175→789 ms).
2. **INT8 KV cache**: context 39→156+ tokens, bit-exact vs FP32 baseline, no perf regression.
3. **Sec-core blocked transpose firmware** (BS=32, flashed): EMBED_XPOSE 100→26.6 ms.
4. **LM_Head sec-core transpose overlap**: 2555→1830 ms (then →789 ms with DDR=2MB).
5. **Weight prefetch serialization** (Change C): decode −10%.

### Explored & rejected

- **INT4 weights** (Design A): quality-gated — saturated per-channel INT8 leaves no compressible group structure; full-INT4 G64 and FFN-only G16/64 both fail next_token stability. Tools kept in `convert_i4.c` / `int4_common.*`; engine hook on `experiments/int4-weights` branch.
- **MLA deployment**: rejected — no 100M–1B MLA model exists; the real context lever was INT8 KV.
- **fip_4096**: built & archived (`fip_archive/`), not flashed (small gain, deferred).

### Current bottlenecks

1. **SD card bandwidth** (21.9 MB/s physical cap): 101 MB/token re-read ⇒ Wt load ~3.7 s is near theoretical optimum.
2. **LM Head**: embed (28 MB) > ION (24 MB), partial chunk SD reads remain.
3. **DDR→ION secondary-core offload disabled**: `/proc/self/pagemap` unavailable ⇒ DDR phys addrs unresolved.

### Optimization roadmap

- [x] Serialize weight prefetch (Change C)
- [x] LM_Head sec-core transpose overlap + blocked firmware
- [x] DDR embed=2MB default
- [x] INT8 KV cache (context ×4)
- [ ] Resolve DDR physical addresses (kernel patch `CONFIG_PROC_PAGE_MONITOR` / ION mmap)
- [ ] Hot-layer weight residency in DDR/ION (cut per-token SD traffic)
- [ ] SDIO access from secondary core (C906L)

## Phase 8 出货配置与回滚（Qwen 24L 引擎，2026-08-15 签核）

Phase 8 收口决定：A'（整层 ion_db 预取 11.72s）与 B-2（per-matrix ion_db 预取）**均未过
9.5s 验收线**，SD 20.2MiB/s + ION 28.1MB 硬件天花板确认（理论地板 ~9.45s 不可达）。

**出货配置（引擎默认，无需任何环境变量）**：
- 读路径：`LW_READ=mmap`（`g_lw_mode=LW_MMAP`，未设置环境变量时即默认）
- gsc：`GSC_ION=1`（默认开启，24 层 gsc 全 ION 缓存）
- 实测 decode = **11.29s/token**（`experiments/qwen_onboard/decode_e1_v0.log` 锚点）
- `LW_READ=ion_db` 为已收口实验路径（负优化），仅回归对照用，默认不启用。

**一键回滚命令（出货基线 = 默认，零重建）**：
```sh
# 若曾跑过 ion_db 实验，先释放其 ION SD_BUF
sh /data/qwen/run_clean.sh --clean qwen_engine_lmhead2_aprime
# 以 mmap + gsc ION 出货配置跑 decode（无需设 LW_READ/GSC_ION，缺省即出货）
cd /data/qwen && VERIFY=0 RSH=1 DECODE=1 DECODE_STEPS=6 PROFILE=1 \
  ./qwen_engine_lmhead2 | tee decode_rollback_mmap.log
# 或直接用已知良好二进制（无需重编）
cd /data/qwen && GSC_ION=1 VERIFY=0 RSH=1 DECODE=1 DECODE_STEPS=6 PROFILE=1 \
  ./qwen_engine_lmhead2_phase7e | tee decode_rollback_mmap.log
```
回归确认：`avg≈11.29±0.5`、NEXT 3/3、bit-exact 4 项=0。详见
`experiments/qwen_onboard/REPORT_APRIME_INTEGRATION_REVIEW_20260815.md`（Phase 8 最终版）。

**Phase 8 回归工具链**（标注在 `experiments/qwen_onboard/`）：
```sh
sh phase7_deploy_run_host.sh aprime 6     # 板上三档: B 基线 mmap / A perf ion_db / A corr VERIFY=1
python3 phase7_analyze_logs.py            # 对比表 + 验收线 9.5s 自动判定 + 回归清单
```

## License

MIT
