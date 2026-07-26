#!/usr/bin/env python3
"""
Interactive streaming chat with SmolLM2-135M on Milk-V Duo.

Usage:
  python3 chat_duo.py [--max-new 30] [--model /root/smollm2_pool/]

Flow:
  1. You type text → tokenized locally
  2. Token IDs uploaded to Duo
  3. Inference runs, tokens stream back via SSH stdout
  4. Each token decoded and printed in real time
"""

import sys, struct, subprocess, os, tempfile, time, argparse

DUO_HOST  = "root@192.168.42.1"
TOKENIZER = "HuggingFaceTB/SmolLM2-135M-Instruct"  # cached locally, same vocab as base

# Chat template for SmolLM2-Instruct (includes system prompt)
CHAT_TEMPLATE = (
    "<|im_start|>system\n"
    "You are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n"
    "<|im_start|>user\n"
    "{text}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

# ── Tokenizer (lazy load) ──────────────────────────────────────────
_tokenizer = None

def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        print("[loading tokenizer...]", file=sys.stderr)
        _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
        eos_endoftext = _tokenizer.convert_tokens_to_ids('<|endoftext|>')
        eos_im_end = _tokenizer.convert_tokens_to_ids('<|im_end|>')
        print(f"[tokenizer OK] vocab={_tokenizer.vocab_size} "
              f"<|endoftext|>={eos_endoftext} <|im_end|>={eos_im_end} "
              f"(eos_token_id={_tokenizer.eos_token_id})", file=sys.stderr)
    return _tokenizer

# ── Duo remote control ─────────────────────────────────────────────
def duo_upload(local_path, remote_path="/root/input_tokens.bin"):
    cmd = ["sshpass", "-p", "milkv", "scp", "-O",
           "-o", "StrictHostKeyChecking=no", local_path, f"{DUO_HOST}:{remote_path}"]
    subprocess.run(cmd, check=True, capture_output=True)

def duo_run(remote_bin, model_dir, remote_tokens, max_new, eos_id, force_mode=0):
    """Run inference on Duo, yield token IDs as they arrive (line-buffered)."""
    cmd = ["sshpass", "-p", "milkv", "ssh",
           "-o", "StrictHostKeyChecking=no", DUO_HOST,
           f"{remote_bin} {model_dir} {remote_tokens} {max_new} {force_mode} {eos_id}"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            bufsize=0, text=False)

    for line_bytes in proc.stdout:
        line = line_bytes.decode().strip()
        if not line:
            continue
        try:
            yield int(line)
        except ValueError:
            pass  # skip non-numeric lines (shouldn't happen on stdout)

    proc.wait()
    # Print stderr diagnostics
    err = proc.stderr.read().decode()
    for l in err.splitlines():
        if any(kw in l for kw in ("Prefill:", "Decode:", "step ", "EOS",
                                   "tok/s", "swap", "RESULTS")):
            print(f"  [diag] {l}", file=sys.stderr)

# ── Main chat loop ─────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SmolLM2-135M Duo Chat")
    parser.add_argument("--max-new", type=int, default=64, help="Max new tokens")
    parser.add_argument("--model", default="/root/smollm2_instruct/", help="Model dir on Duo")
    parser.add_argument("--bin", default="/root/smollm2_pool_demo", help="Binary path on Duo")
    parser.add_argument("--force-mode", type=int, default=0,
                        help="Pipeline mode: 0=auto, 1=1+1, 2=2+2, 3=3+3")
    parser.add_argument("--eos-id", type=int, default=None,
                        help="EOS token ID (auto-detect from tokenizer if not set)")
    parser.add_argument("--no-chat", action="store_true",
                        help="Disable instruct chat template (raw text mode)")
    args = parser.parse_args()

    tok = get_tokenizer()
    # Instruct model uses <|im_end|> (id=2) as EOS
    im_end_id = tok.convert_tokens_to_ids('<|im_end|>')
    eos_id = args.eos_id if args.eos_id is not None else im_end_id
    print(f"[eos_id={eos_id} <|im_end|>={im_end_id}]", file=sys.stderr)

    print("\n" + "=" * 50, file=sys.stderr)
    print("  SmolLM2-135M on Milk-V Duo — Interactive Chat", file=sys.stderr)
    print("  Type 'quit' or Ctrl+C to exit", file=sys.stderr)
    print("  " + "=" * 50 + "\n", file=sys.stderr)

    while True:
        try:
            text = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\n[bye]", file=sys.stderr)
            break

        if text.lower() in ("quit", "exit", "q"):
            break
        if not text.strip():
            continue

        # Tokenize input — apply chat template for Instruct model
        if not args.no_chat:
            text = CHAT_TEMPLATE.format(text=text)
            print(f"[chat template applied: {len(text)} chars]", file=sys.stderr)
        ids = tok.encode(text, add_special_tokens=False)
        if not ids:
            print("(empty input)", file=sys.stderr)
            continue
        print(f"[tokens: {len(ids)}] ", file=sys.stderr, end="")

        # Write token IDs as binary int32
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(struct.pack(f"{len(ids)}i", *ids))
            tmp_path = f.name

        try:
            duo_upload(tmp_path)
            os.unlink(tmp_path)

            print("Duo: ", end="", flush=True, file=sys.stderr)
            t0 = time.time()
            first = True

            for token_id in duo_run(args.bin, args.model, "/root/input_tokens.bin",
                                     args.max_new, eos_id, args.force_mode):
                if first:
                    # Skip the first token (it's the last prompt token / BOS echo)
                    # Actually for base model we want all tokens
                    first = False

                if token_id == eos_id:
                    print("<EOS>", file=sys.stderr)
                    break

                decoded = tok.decode([token_id])
                print(decoded, end="", flush=True)

            elapsed = time.time() - t0
            print(f"\n[{elapsed:.0f}s]", file=sys.stderr)

        except subprocess.CalledProcessError as e:
            print(f"\n[SSH error: {e}]", file=sys.stderr)
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

if __name__ == "__main__":
    main()
