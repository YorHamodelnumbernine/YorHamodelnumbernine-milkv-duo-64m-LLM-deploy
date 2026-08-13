#!/usr/bin/env python3
"""Offline ByteLevel BPE tokenizer for SmolLM2-135M-Instruct using cached tokenizer.json.

No numpy/transformers/torch required.  Used to build a 100+ token prompt file for
the INT8 KV long-context regression on the Duo.

Usage:
  python3 offline_tokenize.py encode "text..." -o /tmp/prompt_tokens.bin
  python3 offline_tokenize.py encode --file prompt.txt -o /tmp/prompt_tokens.bin
  python3 offline_tokenize.py decode --binary /tmp/prompt_tokens.bin
"""
import json, re, struct, sys, os

CACHE = os.path.expanduser(
    "~/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M-Instruct/"
    "snapshots/12fd25f77366fa6b3b4b768ec3050bf629380bac/tokenizer.json")

def load():
    d = json.load(open(CACHE))
    vocab = d["model"]["vocab"]
    merges = d["model"]["merges"]
    rank = {tuple(m.split()): i for i, m in enumerate(merges)}
    return vocab, rank

def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~")+1)) + \
         list(range(ord("¡"), ord("¬")+1)) + \
         list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256+n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))

byte_encoder = bytes_to_unicode()
byte_decoder = {v: k for k, v in byte_encoder.items()}

# Approximate GPT-2 ByteLevel pre-tokenizer regex in Python `re` syntax.
# [[:alpha:]] -> [^\W\d_] (unicode letters), [[:digit:]] -> \d, [[:punct:]] -> [^\w\s]
PAT = r"""'s|'t|'re|'ve|'m|'ll|'d| ?[^\w\s]| ?[^\W\d_]+| ?\d+|\s+(?!\S)|\s+"""

def pretokenize(text):
    s = text.encode("utf-8")
    s = "".join(byte_encoder[b] for b in s)
    return re.findall(PAT, s)

def bpe(token, rank):
    word = tuple(token)
    if len(word) <= 1:
        return token
    while True:
        pairs = [(word[i], word[i+1]) for i in range(len(word)-1)]
        if not pairs:
            break
        best = min(pairs, key=lambda p: rank.get(p, float("inf")))
        if best not in rank:
            break
        a, b = best
        out, i = [], 0
        while i < len(word):
            if i < len(word)-1 and word[i] == a and word[i+1] == b:
                out.append(a+b); i += 2
            else:
                out.append(word[i]); i += 1
        word = tuple(out)
    return " ".join(word)

def encode(text, vocab, rank):
    ids = []
    for tok in pretokenize(text):
        for sub in bpe(tok, rank).split(" "):
            if sub in vocab:
                ids.append(vocab[sub])
            else:
                # byte_fallback: split into UTF-8 bytes -> <0xXX> tokens
                for b in "".join(byte_decoder.get(c, ord(c)) for c in sub).encode("utf-8"):
                    key = f"<0x{b:02X}>"
                    if key in vocab:
                        ids.append(vocab[key])
                    else:
                        raise KeyError(f"byte fallback missing {key!r}")
    return ids

def decode(ids, vocab, rank):
    inv = {v: k for k, v in vocab.items()}
    toks = []
    for i in ids:
        t = inv.get(i)
        if t is None:
            toks.append("<UNK>"); continue
        if t.startswith("<0x") and t.endswith(">"):
            toks.append(chr(int(t[3:-1], 16)))
        else:
            toks.append(t)
    s = "".join(toks)
    return s.encode("utf-8", errors="ignore").decode("utf-8", errors="replace")

def main():
    import argparse
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    pe = sub.add_parser("encode")
    pe.add_argument("text", nargs="?", default="")
    pe.add_argument("--file")
    pe.add_argument("-o", "--output", required=True)
    pd = sub.add_parser("decode")
    pd.add_argument("--binary")
    args = ap.parse_args()

    vocab, rank = load()
    if args.cmd == "encode":
        text = open(args.file).read() if args.file else args.text
        ids = encode(text, vocab, rank)
        with open(args.output, "wb") as f:
            f.write(struct.pack(f"<{len(ids)}i", *ids))
        print(f"encoded {len(ids)} tokens -> {args.output}", file=sys.stderr)
        print(f"ids[:10]={ids[:10]} ids[-5:]={ids[-5:]}", file=sys.stderr)
        print(f"in-range={all(0 <= i < 49152 for i in ids)}", file=sys.stderr)
    elif args.cmd == "decode":
        data = open(args.binary, "rb").read()
        ids = list(struct.unpack(f"<{len(data)//4}i", data))
        print(decode(ids, vocab, rank))
    else:
        ap.print_help()

if __name__ == "__main__":
    main()
