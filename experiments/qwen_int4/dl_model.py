#!/usr/bin/env python3
"""Download Qwen2.5-0.5B-Instruct via hf-mirror into experiments/qwen_int4/model."""
import os, sys, time
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import snapshot_download

REPO = "Qwen/Qwen2.5-0.5B-Instruct"
DST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
os.makedirs(DST, exist_ok=True)

t0 = time.time()
path = snapshot_download(
    repo_id=REPO,
    local_dir=DST,
    allow_patterns=[
        "config.json", "generation_config.json",
        "tokenizer.json", "tokenizer_config.json",
        "vocab.json", "merges.txt", "special_tokens_map.json",
        "model.safetensors",
    ],
)
print(f"download done in {time.time()-t0:.1f}s -> {path}", flush=True)
# quick size check
for f in sorted(os.listdir(DST)):
    fp = os.path.join(DST, f)
    print(f"  {f}: {os.path.getsize(fp)/1e6:.1f} MB", flush=True)
