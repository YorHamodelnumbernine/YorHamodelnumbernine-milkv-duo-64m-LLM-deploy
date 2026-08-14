#!/usr/bin/env python3
"""qwen_tokenize.py — encode fixed prompts to Qwen BPE token IDs for on-board P0 test.

Writes input_tokens.bin: [n_tokens int32][token ids int32] (matches smollm2 flow),
plus a human-readable mapping for verifying next_token against host reference.

Usage:
  ../qwen_int4/.venv/bin/python3 qwen_tokenize.py --prompts "中国的首都是" \
      "The capital of France is" "今天天气很好，我们去公园" --out input_tokens.bin
"""
import os, sys, struct, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PRE = os.path.join(HERE, "..", "qwen_int4")
DEFAULT_MODEL = os.path.join(PRE, "model")
DEFAULT_PROMPTS = ["中国的首都是", "The capital of France is", "今天天气很好，我们去公园"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--prompts", nargs="*", default=DEFAULT_PROMPTS)
    ap.add_argument("--out", default=os.path.join(HERE, "input_tokens.bin"))
    ap.add_argument("--json", default=os.path.join(HERE, "p0_inputs.json"))
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    all_ids = []
    meta = {}
    for p in args.prompts:
        ids = tok.encode(p, add_special_tokens=False)
        all_ids.extend(ids)
        meta[p] = {"tokens": ids, "n": len(ids),
                   "text": [tok.decode([t]) for t in ids]}
        print(f"[tok] {p!r} -> n={len(ids)} {ids[:12]}...")
    arr = np.array([len(all_ids)] + all_ids, dtype=np.int32)
    arr.tofile(args.out)
    print(f"[tok] wrote {args.out} ({arr.nbytes} B, {len(all_ids)} ids)")
    with open(args.json, "w") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
    print(f"[tok] wrote {args.json}")


if __name__ == "__main__":
    main()
