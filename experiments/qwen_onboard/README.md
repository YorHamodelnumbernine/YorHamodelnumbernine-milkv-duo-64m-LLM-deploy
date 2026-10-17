# Qwen2.5-0.5B-Instruct on CV1800B (Milk-V Duo 64M)

On-board INT4 inference engine for **Qwen2.5-0.5B-Instruct**, running on the
CV1800B TPU (bmk1822) with the C906B Linux core driving weights from SD card.
This directory contains the shipping engine, converters, host chat tooling and
all phase reports.

Last updated: 2026-10-17 · shipping state: B2 dequant + INT8 KV + two-stage LM head

Latest status summary (bilingual):
[English](STATUS_20261017_en.md) · [中文](STATUS_20261017_zh.md)

## Model / engine configuration

| Param | Value |
|---|---|
| Model | Qwen2.5-0.5B-Instruct |
| Layers | 24 |
| hidden / heads / KV heads / head_dim | 896 / 14 / 2 / 64 |
| FFN hidden | 4864 |
| Vocab | 151,936 |
| Weights | INT4 G32, K-aligned, `nib[N][16] → nib16[16][N]` (B2 layout) |
| Embedding | INT8 (136 MB) + FP32 scales |
| KV cache | INT8 per-token symmetric (int8 + per-token scale), `KV_CAP=192` |
| On-SD footprint | ~324 MB (embed 136 MB + 24 × ~8.1 MB layers) |
| Context | 192 tokens (was 48 before INT8 KV, ×4 in the same memory) |
| Board runtime | `/data/qwen`, shipped binary `qwen_engine_lmhead2_b2` |

## Shipping run configuration

```sh
cd /data/qwen
# decode regression (mmap + GSC_ION=1 are engine defaults)
VERIFY=0 RSH=1 DECODE=1 DECODE_STEPS=6 PROFILE=1 ./qwen_engine_lmhead2_b2

# chat (host side)
experiments/qwen_int4/.venv/bin/python3 chat_qwen.py "你好"
```

Guaranteed output contract: 3-prompt NEXT = 2130 / 12095 / 99366,
TIU internal bit-exact (bad1=bad2=r_opt=rsh=0).

## Performance (final, 2026-08-22)

| Metric | Value | Notes |
|---|---|---|
| Decode | **11.10–11.19 s/token** | 24 layers, SD-bound |
| CHAT prefill, 35 tok | 101.7 s | |
| CHAT prefill, 75 tok | 219.6 s | ↓28% vs 304.5 s pre-B2 |
| Decode (75-token context) | 10.98 s/token | ↓32% vs 16.25 s pre-B2 |
| LM head | 0.5–0.7 s | stage1 CPU dot 24 ms, stage2 span pread |
| Dequant throughput | 538 MB/s (N=896), 753 MB/s (N=4864) | 1.75× / 2.02× vs gather kernel |
| TIU runs / decode step | 299,520 (pass1+pass2) | PROFILE probe |

Hardware ceilings: SD sequential read ~20.2 MiB/s, ION carveout 28.1 MB,
per-token weight traffic ~201 MB. The implied decode floor is ~9.45 s; 11.1 s
is the engineering limit under these constraints.

## Landed work (timeline)

| Phase | Change | Result |
|---|---|---|
| P0–P4 | Weight conversion, K-aligned INT4, on-board P0/P1 bring-up | engine runs, q/k/v bias bug fixed |
| 6 | Two-pass INT4 matmul, RPMsg/TPU path, LM head two-stage | recall@1 11/11; head 0.5–0.7 s |
| 7 | mmap read path, rsafe pre-calibration (RSH=1), TIU submit merge, gsc resident in ION | decode 26.0 → 11.29 s |
| 7.5 | INT8 KV cache | context 48 → 192, bit-exact |
| 8 | ion_db prefetch A/A'/B-2 | FAIL vs 9.5 s target; shipping config locked to `mmap + GSC_ION=1` |
| 6 (LM head) | centroid fp16 1.84 MB adaptive ION residency + span pread | head cost flat, no OOM |
| Watchdog | init-window coverage + atomic heartbeat + forced `_exit(1)` | no hangs, no ION orphans, zero false kills |
| B2 | `nib16[16][N]` re-layout + contiguous `vle8` RVV dequant | dequant 1.75–2.02×, prefill ↓28%, decode ↓32% |
| B-P0 | readahead + DONTNEED prefetch | **closed (dead end)**: `madvise`/`posix_fadvise(DONTNEED)` = ENOSYS, all negative |

## Correctness evidence

- 3-prompt NEXT regression: expected 2130 / 12095 / 99366 → 3/3 OK.
- Decode chain: 2130 → 198 → 32.
- CHAT 75-token chain identical token-by-token to the pre-B2 baseline.
- TIU internal bit-exact counters all zero on every regression run.
- ION usage returns to 0 after each run; watchdog tests force-exit with rc=1.

## Key files

| File | Purpose |
|---|---|
| `qwen_engine_lmhead2.c` | Shipping engine (INT4 W, INT8 KV, two-stage LM head, watchdog) |
| `convert_qwen.py` | HF safetensors → on-board FP32/INT8 layout |
| `convert_qwen_kal.py` / `convert_qwen_kal_b2.py` | INT4 G32 K-aligned / B2 nib16 conversion |
| `deploy_qwen_weights.py` | Weight deployment to `/data/qwen` |
| `chat_qwen.py` | Host-side tokenize → push → run → detokenize chat wrapper |
| `qwen_emu.py` | NumPy reference emulator |
| `REPORT_PHASE8_CLOSE_20260815.md` | SD/ION ceiling + shipping config |
| `REPORT_PHASE6_LMHEAD_LANDING_20260821.md` | LM head landing and ION budget |
| `REPORT_B2_VERIFY_20260822.md` | B2 on-board verification |
| `REPORT_ION_WATCHDOG_VERIFY_20260822.md` | Watchdog validation |
| `REPORT_B0_PREFETCH_20260822.md` | B-P0 prefetch closure |

Large artifacts (`weights/`, `weights_kal/`, logs, cross-compiled binaries) are
not tracked; regenerate with the converters and `git archive` builds.
