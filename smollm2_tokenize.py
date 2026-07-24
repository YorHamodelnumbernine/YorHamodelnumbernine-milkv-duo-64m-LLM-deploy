#!/usr/bin/env python3
"""Host-side tokenizer for SmolLM2-135M on Milk-V Duo.

Encode: python3 smollm2_tokenize.py encode "Hello world" > tokens.bin
Decode: python3 smollm2_tokenize.py decode 1234 5678 910
Pipe:  tail -1 duo_output.txt | grep TOKENS | python3 smollm2_tokenize.py decode -

Output format: raw int32 little-endian binary (for encode).
Input format for decode: space-separated token IDs, or "-" to read from stdin (expects "TOKENS: id id ..." line).

Requires: transformers, torch (or use --model-path to point at local safetensors dir)
"""

import sys, os, struct, argparse
import numpy as np

def get_tokenizer(model_id="HuggingFaceTB/SmolLM2-135M-Instruct"):
    """Load the tokenizer from HF hub."""
    from transformers import AutoTokenizer
    print(f"Loading tokenizer: {model_id}...", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(model_id)
    print(f"  vocab_size={tok.vocab_size}, bos={tok.bos_token_id}, eos={tok.eos_token_id}",
          file=sys.stderr)
    return tok

def cmd_encode(args):
    tok = get_tokenizer(args.model)
    text = args.text
    if args.file:
        with open(args.file) as f:
            text = f.read()

    # Apply chat template if requested
    if args.chat:
        messages = [{"role": "user", "content": text}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        print(f"Chat template applied: {text[:100]}...", file=sys.stderr)

    ids = tok.encode(text, add_special_tokens=not args.no_special)
    print(f"Encoded: {len(ids)} tokens", file=sys.stderr)
    print(f"Token IDs: {ids[:20]}{'...' if len(ids) > 20 else ''}", file=sys.stderr)

    # Output raw int32 little-endian
    data = struct.pack(f"<{len(ids)}i", *ids)
    if args.output:
        with open(args.output, 'wb') as f:
            f.write(data)
        print(f"Wrote {len(data)} bytes to {args.output}", file=sys.stderr)
    else:
        sys.stdout.buffer.write(data)

def cmd_decode(args):
    tok = get_tokenizer(args.model)

    # Read token IDs
    if args.tokens == '-':
        # Read from stdin, expect "TOKENS: id id ..." format
        line = sys.stdin.readline().strip()
        if line.startswith("TOKENS:"):
            line = line[7:].strip()  # remove "TOKENS:"
        ids = [int(x) for x in line.split() if x.lstrip('-').isdigit()]
    elif args.binary:
        # Read raw int32 binary
        with open(args.binary, 'rb') as f:
            data = f.read()
        ids = list(struct.unpack(f"<{len(data)//4}i", data))
    else:
        ids = [int(x) for x in args.tokens]

    # Decode
    text = tok.decode(ids, skip_special_tokens=args.skip_special)
    print(text)

def cmd_info(args):
    tok = get_tokenizer(args.model)
    print(f"Model: {args.model}")
    print(f"Vocab size: {tok.vocab_size}")
    print(f"BOS token: {tok.bos_token} (id={tok.bos_token_id})")
    print(f"EOS token: {tok.eos_token} (id={tok.eos_token_id})")
    print(f"PAD token: {tok.pad_token} (id={tok.pad_token_id})")
    if hasattr(tok, 'chat_template') and tok.chat_template:
        print(f"Chat template: yes")


def main():
    parser = argparse.ArgumentParser(description="SmolLM2 tokenizer for Duo")
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct",
                        help="HF model ID")

    sub = parser.add_subparsers(dest="cmd", help="Commands")

    p = sub.add_parser("encode", help="Encode text to token IDs")
    p.add_argument("text", nargs="?", default="Hello, how are you?",
                   help="Text to encode")
    p.add_argument("--file", help="Read text from file instead")
    p.add_argument("--output", "-o", help="Output file (default: stdout binary)")
    p.add_argument("--chat", action="store_true",
                   help="Apply chat template")
    p.add_argument("--no-special", action="store_true",
                   help="Don't add special tokens")

    p = sub.add_parser("decode", help="Decode token IDs to text")
    p.add_argument("tokens", nargs="*", default=[],
                   help="Token IDs (space-separated, or '-' for stdin TOKENS format)")
    p.add_argument("--binary", "-b", help="Read raw binary int32 file")
    p.add_argument("--skip-special", action="store_true", default=True,
                   help="Skip special tokens when decoding")

    p = sub.add_parser("info", help="Show tokenizer info")

    args = parser.parse_args()
    if args.cmd == "encode":
        cmd_encode(args)
    elif args.cmd == "decode":
        cmd_decode(args)
    elif args.cmd == "info":
        cmd_info(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
