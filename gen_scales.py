#!/usr/bin/env python3
"""Compute per-tensor symmetric INT8 scales from SmolLM2 safetensors.
Generates scales.bin — a flat float32 binary file with scales in order:
  [embed_scale] [for each layer: Wq,Wk,Wv,Wo,up,gate,down] [final_rms(1.0)]
Total: 1 + 30*7 + 1 = 212 float32 values ≈ 848 bytes.
"""
import sys, struct, json
import numpy as np
from pathlib import Path

def bf16_bytes_to_f32(byte_data, shape):
    raw = np.frombuffer(byte_data, dtype=np.uint16)
    f32 = np.zeros(len(raw), dtype=np.float32)
    f32.view(np.uint32)[:] = raw.astype(np.uint32) << 16
    return f32.reshape(shape)

def load_safetensors(path):
    import safetensors
    with open(path, 'rb') as f:
        data = f.read()
    result = safetensors.deserialize(data)
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
    return state

def quant_scale(data: np.ndarray):
    """Return symmetric INT8 scale = absmax / 127.0"""
    absmax = float(np.max(np.abs(data)))
    if absmax < 1e-10:
        absmax = 1.0
    return absmax / 127.0

def main():
    model_dir = sys.argv[1] if len(sys.argv) > 1 else \
        str(Path.home() / ".cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/12fd25f77366fa6b3b4b768ec3050bf629380bac")

    # Load config
    with open(Path(model_dir) / "config.json") as f:
        cfg = json.load(f)

    d_model = cfg["hidden_size"]
    n_layers = cfg["num_hidden_layers"]

    # Load safetensors
    sf_path = Path(model_dir) / "model.safetensors"
    state = load_safetensors(str(sf_path))
    print(f"Loaded {len(state)} tensors from {sf_path}", file=sys.stderr)

    scales = []

    # Embedding scale (weight-tied with lm_head)
    embed_w = state["model.embed_tokens.weight"].astype(np.float32)
    sc = quant_scale(embed_w)
    scales.append(sc)
    print(f"embed: {sc:.6f}", file=sys.stderr)

    # Per-layer scales
    for l in range(n_layers):
        prefix = f"model.layers.{l}"

        wq = state[f"{prefix}.self_attn.q_proj.weight"].astype(np.float32).T
        wk = state[f"{prefix}.self_attn.k_proj.weight"].astype(np.float32).T
        wv = state[f"{prefix}.self_attn.v_proj.weight"].astype(np.float32).T
        wo = state[f"{prefix}.self_attn.o_proj.weight"].astype(np.float32).T
        up = state[f"{prefix}.mlp.up_proj.weight"].astype(np.float32).T
        gate = state[f"{prefix}.mlp.gate_proj.weight"].astype(np.float32).T
        down = state[f"{prefix}.mlp.down_proj.weight"].astype(np.float32).T

        sc_wq = quant_scale(np.ascontiguousarray(wq))
        sc_wk = quant_scale(np.ascontiguousarray(wk))
        sc_wv = quant_scale(np.ascontiguousarray(wv))
        sc_wo = quant_scale(np.ascontiguousarray(wo))
        sc_up = quant_scale(np.ascontiguousarray(up))
        sc_gate = quant_scale(np.ascontiguousarray(gate))
        sc_down = quant_scale(np.ascontiguousarray(down))

        scales.extend([sc_wq, sc_wk, sc_wv, sc_wo, sc_up, sc_gate, sc_down])
        if l < 3:
            print(f"layer{l}: Wq={sc_wq:.6f} Wk={sc_wk:.6f} Wv={sc_wv:.6f} Wo={sc_wo:.6f} "
                  f"up={sc_up:.6f} gate={sc_gate:.6f} down={sc_down:.6f}", file=sys.stderr)

    # Final RMS norm doesn't need a scale (it's FP32), but we add a placeholder for index alignment
    # Placeholder = 0.0
    scales.append(0.0)

    # Write scales.bin
    data = struct.pack(f"<{len(scales)}f", *scales)
    out_path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/smollm2/scales.bin"
    with open(out_path, "wb") as f:
        f.write(data)
    print(f"\nWrote {len(scales)} scales ({len(data)} bytes) to {out_path}", file=sys.stderr)
    print(f"Scale range: {min(scales):.6f} - {max(scales):.6f}", file=sys.stderr)

if __name__ == "__main__":
    main()
