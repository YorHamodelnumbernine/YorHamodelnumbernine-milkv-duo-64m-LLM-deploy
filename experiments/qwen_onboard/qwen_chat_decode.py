#!/usr/bin/env python3
"""qwen_chat_decode.py — host 侧 CHAT 输出解码器 (token id 流 -> 文本).

用法:
  python3 qwen_chat_decode.py <logfile>          # 从日志提取 CHAT: 行并解码
  python3 qwen_chat_decode.py --ids "2130 198 32"  # 直接解码 id 串
  python3 qwen_chat_decode.py --bin input_tokens.bin  # 显示 prompt token 文本

CHAT: 行格式: "CHAT: <prefill_next> <decode_0> ... <decode_{n-1}>"
"""
import os, sys, argparse, struct

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "..", "qwen_int4", "model")


def get_tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)


def decode_ids(tok, ids):
    parts = [tok.decode([i]) for i in ids]
    return "".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?", help="log file containing CHAT: line")
    ap.add_argument("--ids", help="space-separated token ids")
    ap.add_argument("--bin", help="input_tokens.bin to show prompt text")
    args = ap.parse_args()

    tok = get_tok()

    if args.bin:
        with open(args.bin, "rb") as f:
            data = f.read()
        n = struct.unpack("<i", data[:4])[0]
        ids = list(struct.unpack(f"<{n}i", data[4:4 + 4 * n]))
        print(f"[prompt] n={n} ids={ids}")
        print(f"[prompt] text={decode_ids(tok, ids)!r}")

    if args.ids:
        ids = [int(x) for x in args.ids.split()]
        print(f"[gen] ids={ids}")
        print(f"[gen] text={decode_ids(tok, ids)!r}")
        return

    if args.log:
        chat_line = None
        with open(args.log) as f:
            for line in f:
                line = line.rstrip("\n")
                if line.startswith("CHAT:"):
                    chat_line = line
        if chat_line is None:
            print(f"[err] no CHAT: line in {args.log}")
            sys.exit(1)
        ids = [int(x) for x in chat_line[5:].strip().split()]
        print(f"[gen] {chat_line}")
        print(f"[gen] ids={ids}")
        print(f"[gen] text={decode_ids(tok, ids)!r}")


if __name__ == "__main__":
    main()
