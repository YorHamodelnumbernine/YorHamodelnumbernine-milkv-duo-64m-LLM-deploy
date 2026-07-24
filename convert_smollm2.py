#!/usr/bin/env python3
"""Download SmolLM2-135M-Instruct and convert weights to flat INT8 binary files
for Milk-V Duo SD card streaming inference.

Uses safetensors (no torch needed) to parse model weights.
Weight files are flat binary: INT8 for matmul weights, FP32 for norms.

Output layout (in --out dir):
  config.bin             model config (8×int32)
  embed.i8              vocab×d_model INT8 (weight-tied with lm_head)
  layer{N}_rms_attn.f32  d_model FP32
  layer{N}_Wq.i8         d_model×d_model INT8
  layer{N}_Wk.i8         d_model×d_kv INT8
  layer{N}_Wv.i8         d_model×d_kv INT8
  layer{N}_Wo.i8         d_model×d_model INT8
  layer{N}_rms_ffn.f32   d_model FP32
  layer{N}_ffn_up.i8     d_model×ffn_hidden INT8
  layer{N}_ffn_gate.i8   d_model×ffn_hidden INT8
  layer{N}_ffn_down.i8   ffn_hidden×d_model INT8
  final_rms.f32          d_model FP32
"""

import os, sys, struct, argparse, json
import numpy as np
from pathlib import Path

def quantize_i8(data: np.ndarray):
    """Symmetric per-tensor INT8 quantization. Returns (i8_data, scale)."""
    absmax = float(np.max(np.abs(data)))
    if absmax < 1e-10:
        absmax = 1.0
    scale = absmax / 127.0
    q = np.clip(np.round(data / scale), -128, 127).astype(np.int8)
    return q, scale

def write_f32(path, data: np.ndarray):
    data.astype(np.float32).tofile(path)
    print(f"  {Path(path).name:30s}  {str(data.shape):20s}  {data.nbytes:>8d} B")

def write_i8(path, data: np.ndarray, scale: float = 0):
    data.astype(np.int8).tofile(path)
    print(f"  {Path(path).name:30s}  {str(data.shape):20s}  {data.nbytes:>8d} B  sc={scale:.6f}")

def write_layer_bin(path, parts):
    """Write merged layer.bin: concatenate list of (tensor, dtype) tuples."""
    flat_parts = [p.astype(dt).ravel() for p, dt in parts]
    merged = np.concatenate(flat_parts)
    merged.tofile(path)
    total = sum(p.nbytes for p, _ in parts)
    print(f"  {Path(path).name:30s}  {'merged 9 tensors':20s}  {total:>8d} B")

def download_model(model_id, cache_dir):
    """Download all safetensors files from HF hub without torch."""
    from huggingface_hub import hf_hub_download, list_repo_files

    files = list_repo_files(model_id)
    safetensors_files = [f for f in files if f.endswith('.safetensors')]
    config_file = 'config.json' if 'config.json' in files else None

    print(f"  Found {len(safetensors_files)} safetensors files, config={config_file}")

    downloaded = {}
    for f in safetensors_files:
        path = hf_hub_download(model_id, f, cache_dir=cache_dir)
        downloaded[f] = path
        print(f"  Downloaded: {f}")

    if config_file:
        cfg_path = hf_hub_download(model_id, config_file, cache_dir=cache_dir)
        downloaded['config.json'] = cfg_path

    return downloaded

def bf16_bytes_to_f32(byte_data, shape):
    """Convert bfloat16 bytearray to float32 numpy array."""
    # BF16 is stored as 2 bytes per element (little-endian uint16)
    # To convert to F32: place the 16 BF16 bits in the upper 16 bits of F32
    raw = np.frombuffer(byte_data, dtype=np.uint16)
    f32 = np.zeros(len(raw), dtype=np.float32)
    f32.view(np.uint32)[:] = raw.astype(np.uint32) << 16
    return f32.reshape(shape)

def load_safetensors(path):
    """Load a safetensors file, returns dict of {key: numpy_array}.
    Handles BF16, F16, F32 dtypes from safetensors 0.8.0 API."""
    import safetensors
    with open(path, 'rb') as f:
        data = f.read()
    result = safetensors.deserialize(data)

    # safetensors 0.8.0 returns list of (name, dict) tuples
    # where dict has: {'shape': [...], 'dtype': 'BF16'|'F32', 'data': bytearray}
    state = {}
    for name, info in result:
        dtype = info['dtype']
        shape = info['shape']
        raw = info['data']
        if dtype == 'BF16':
            state[name] = bf16_bytes_to_f32(raw, shape)
        elif dtype == 'F16':
            arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
            state[name] = arr.reshape(shape)
        elif dtype in ('F32', 'FLOAT32'):
            arr = np.frombuffer(raw, dtype=np.float32)
            state[name] = arr.reshape(shape)
        elif dtype == 'I64':
            arr = np.frombuffer(raw, dtype=np.int64)
            state[name] = arr.reshape(shape)
        else:
            raise ValueError(f"Unknown dtype {dtype} for {name}")
    return state

def main():
    parser = argparse.ArgumentParser(description="Convert SmolLM2-135M to flat INT8 for Duo")
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct",
                        help="HF model ID")
    parser.add_argument("--out", default="/tmp/smollm2",
                        help="Output directory for weight files")
    parser.add_argument("--cache", default=None,
                        help="HF cache directory")
    parser.add_argument("--max-seq", type=int, default=64,
                        help="Maximum sequence length")
    parser.add_argument("--local", default=None,
                        help="Local directory with safetensors files (skip download)")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # --- Load model ---
    print(f"[1/4] Loading model: {args.model}")

    if args.local:
        # Load from local directory
        local = Path(args.local)
        state = {}
        for sf in sorted(local.glob("*.safetensors")):
            print(f"  Loading: {sf.name}")
            state.update(load_safetensors(str(sf)))
        # Load config
        cfg_path = local / "config.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = json.load(f)
        else:
            print("  WARNING: no config.json found, using defaults for SmolLM2-135M")
            cfg = {"hidden_size": 576, "num_attention_heads": 9,
                   "num_key_value_heads": 3, "num_hidden_layers": 30,
                   "intermediate_size": 1536, "vocab_size": 49152}
    else:
        cache_dir = args.cache or os.path.expanduser("~/.cache/huggingface/hub")
        files = download_model(args.model, cache_dir)

        # Load config
        if 'config.json' in files:
            with open(files['config.json']) as f:
                cfg = json.load(f)
        else:
            cfg = {"hidden_size": 576, "num_attention_heads": 9,
                   "num_key_value_heads": 3, "num_hidden_layers": 30,
                   "intermediate_size": 1536, "vocab_size": 49152}

        # Merge all safetensors
        state = {}
        for name, path in files.items():
            if name.endswith('.safetensors'):
                state.update(load_safetensors(path))

    d_model = cfg["hidden_size"]
    n_heads = cfg["num_attention_heads"]
    n_kv_heads = cfg["num_key_value_heads"]
    head_dim = d_model // n_heads
    n_layers = cfg["num_hidden_layers"]
    ffn_hidden = cfg["intermediate_size"]
    vocab_size = cfg["vocab_size"]
    max_seq = args.max_seq

    print(f"  d_model={d_model}, n_heads={n_heads}, n_kv_heads={n_kv_heads}, head_dim={head_dim}")
    print(f"  n_layers={n_layers}, ffn_hidden={ffn_hidden}, vocab_size={vocab_size}")
    print(f"  weight keys: {len(state)}")

    # Write config
    cfg_bin = struct.pack("iiiiiiii", d_model, n_heads, n_kv_heads, head_dim,
                          n_layers, ffn_hidden, vocab_size, max_seq)
    with open(out / "config.bin", "wb") as f:
        f.write(cfg_bin)
    print(f"  -> config.bin ({len(cfg_bin)} bytes)")

    # --- Extract and quantize ---
    print(f"\n[2/4] Extracting and quantizing weights...")
    total_bytes = 0

    # Embedding (weight-tied with lm_head in SmolLM2)
    # HF format: [vocab, d_model], we keep this layout
    embed_key = "model.embed_tokens.weight"
    embed_w = state[embed_key]
    print(f"  {embed_key}: {embed_w.shape}  dtype={embed_w.dtype}")
    embed_i8, embed_sc = quantize_i8(embed_w.astype(np.float32))
    write_i8(str(out / "embed.i8"), embed_i8, embed_sc)
    total_bytes += embed_i8.nbytes

    for l in range(n_layers):
        prefix = f"model.layers.{l}"

        # RMS Norm (input_layernorm)
        rms_attn = state[f"{prefix}.input_layernorm.weight"].astype(np.float32)

        # QKV weights — HF stores as [out_features, in_features]
        # Our matmul convention: x[seq, in] × W[in, out], so we transpose: W_hf^T
        wq = state[f"{prefix}.self_attn.q_proj.weight"].astype(np.float32).T
        wk = state[f"{prefix}.self_attn.k_proj.weight"].astype(np.float32).T
        wv = state[f"{prefix}.self_attn.v_proj.weight"].astype(np.float32).T
        wo = state[f"{prefix}.self_attn.o_proj.weight"].astype(np.float32).T

        wq_i8, _ = quantize_i8(np.ascontiguousarray(wq))
        wk_i8, _ = quantize_i8(np.ascontiguousarray(wk))
        wv_i8, _ = quantize_i8(np.ascontiguousarray(wv))
        wo_i8, _ = quantize_i8(np.ascontiguousarray(wo))

        # RMS Norm (post_attention_layernorm)
        rms_ffn = state[f"{prefix}.post_attention_layernorm.weight"].astype(np.float32)

        # FFN weights
        w_up = state[f"{prefix}.mlp.up_proj.weight"].astype(np.float32).T
        w_gate = state[f"{prefix}.mlp.gate_proj.weight"].astype(np.float32).T
        w_down = state[f"{prefix}.mlp.down_proj.weight"].astype(np.float32).T

        up_i8, _ = quantize_i8(np.ascontiguousarray(w_up))
        gate_i8, _ = quantize_i8(np.ascontiguousarray(w_gate))
        down_i8, _ = quantize_i8(np.ascontiguousarray(w_down))

        # Write merged layerN.bin: rms_attn|Wq|Wk|Wv|Wo|rms_ffn|up|gate|down
        write_layer_bin(str(out / f"layer{l}.bin"), [
            (rms_attn, np.float32),
            (wq_i8, np.int8), (wk_i8, np.int8), (wv_i8, np.int8), (wo_i8, np.int8),
            (rms_ffn, np.float32),
            (up_i8, np.int8), (gate_i8, np.int8), (down_i8, np.int8),
        ])
        total_bytes += rms_attn.nbytes + rms_ffn.nbytes
        total_bytes += wq_i8.nbytes + wk_i8.nbytes + wv_i8.nbytes + wo_i8.nbytes
        total_bytes += up_i8.nbytes + gate_i8.nbytes + down_i8.nbytes

    # Final RMS Norm
    final_rms = state["model.norm.weight"].astype(np.float32)
    write_f32(str(out / "final_rms.f32"), final_rms)
    total_bytes += final_rms.nbytes

    print(f"\n[3/4] Summary:")
    print(f"  Weight files: {len(list(out.glob('*')))}")
    print(f"  Total INT8+FP32: {total_bytes:,} B ({total_bytes/1024/1024:.1f} MB)")
    total_disk = sum(f.stat().st_size for f in out.glob('*'))
    print(f"  Total on disk: {total_disk:,} B ({total_disk/1024/1024:.1f} MB)")

    # Verify
    print(f"\n[4/4] Verification:")
    embed_check = np.fromfile(out / "embed.i8", dtype=np.int8)
    ok = embed_check.shape[0] == vocab_size * d_model
    print(f"  embed.i8: {embed_check.shape[0]} elements {'OK' if ok else 'FAIL'}")
    assert ok, f"Expected {vocab_size * d_model}, got {embed_check.shape[0]}"

    dkv = n_kv_heads * head_dim
    expected_layer_sz = (
        d_model * 4           # rms_attn
        + d_model * d_model   # Wq
        + d_model * dkv * 2   # Wk + Wv
        + d_model * d_model   # Wo
        + d_model * 4         # rms_ffn
        + d_model * ffn_hidden * 2  # up + gate
        + ffn_hidden * d_model      # down
    )
    layer0_check = np.fromfile(out / "layer0.bin", dtype=np.int8)
    ok = layer0_check.shape[0] == expected_layer_sz
    print(f"  layer0.bin: {layer0_check.shape[0]} elements (expected {expected_layer_sz}) {'OK' if ok else 'FAIL'}")
    assert ok

    print(f"\nDone! Files in {out}/")


if __name__ == "__main__":
    main()
