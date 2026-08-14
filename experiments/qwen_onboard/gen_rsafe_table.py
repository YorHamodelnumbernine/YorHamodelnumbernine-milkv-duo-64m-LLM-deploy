#!/usr/bin/env python3
"""gen_rsafe_table.py — Phase 7c: 离线 rsafe 预标定表 (rsafe.bin) 生成器.

从 layerN_kal.bin 读回每个 (layer, matrix[, down K-chunk]) 的 packed INT4 nib,
按引擎 matmul_rshift_w 语义算 rsafe (对称 INT4 SYM_QMAX=7 => wmax 恒 7 => rsafe 恒 5),
写入 rsafe.bin (每层 11 字节: q,k,v,wo,up,gate 各 1 + down 5 个 K-chunk).

引擎 (qwen_engine_lmhead2.c) 在 RSH=1 下启动载入此表并跳过运行时 wmax 预扫;
VERIFY=1 回归会做一次运行时扫描与表比对, 证明位精确一致.

用法: python3 gen_rsafe_table.py [dir=weights_kal] [out=rsafe.bin]
"""
import os, sys

D, L, F, G = 896, 24, 4864, 32
DKV = 128
DCHUNK = 1024  # down K-chunk (引擎硬编码 kc += 1024)
NDOWN = (F + DCHUNK - 1) // DCHUNK  # 5


def matmul_rshift_w(wmax):
    md = 32 * 127 * wmax  # K=G=32
    r = 0
    while (md >> r) > 127:
        r += 1
    return r


def rsafe_of(wmax):
    r = matmul_rshift_w(wmax) - 3
    return r if r >= 4 else 4


def wmax_of(nib):
    w = 0
    for b in nib:
        lo = b & 0xF
        hi = b >> 4
        if lo > 7:
            lo -= 16
        if hi > 7:
            hi -= 16
        a = -lo if lo < 0 else lo
        bb = -hi if hi < 0 else hi
        if a > w:
            w = a
        if bb > w:
            w = bb
    return w


# nib 字节数, 与 parse_layer 布局一致
MATS = [  # name, nib_bytes
    ("q", (D // G) * D * 16),
    ("k", (D // G) * DKV * 16),
    ("v", (D // G) * DKV * 16),
    ("wo", (D // G) * D * 16),
    ("up", (D // G) * F * 16),
    ("gate", (D // G) * F * 16),
    ("down", (F // G) * D * 16),
]


def main():
    d = "weights_kal"
    out = "rsafe.bin"
    if len(sys.argv) > 1 and sys.argv[1].startswith("dir="):
        d = sys.argv[1].split("=", 1)[1]
    if len(sys.argv) > 2 and sys.argv[2].startswith("out="):
        out = sys.argv[2].split("=", 1)[1]

    tbl = bytearray()
    for l in range(L):
        p = os.path.join(d, f"layer{l}_kal.bin")
        data = open(p, "rb").read()
        off = D * 4  # rms_attn
        row = []
        for name, nsz in MATS:
            nib = data[off:off + nsz]
            off += nsz + (nsz // 16) * 2  # nib + gsc fp16 (每 16B nib 配 2B gsc)
            if name == "down":
                chunk = DCHUNK // G * D * 16  # 每 K-chunk 的 nib 字节
                for c in range(NDOWN):
                    w = wmax_of(nib[c * chunk:(c + 1) * chunk])
                    row.append(rsafe_of(w))
            else:
                w = wmax_of(nib)
                row.append(rsafe_of(w))
        assert off + D * 4 == len(data), f"layer{l} layout mismatch"
        tbl.extend(bytes(row))
        print(f"layer{l:2d}: {row}")

    with open(out, "wb") as f:
        f.write(tbl)
    distinct = sorted(set(tbl))
    print(f"wrote {out}: {len(tbl)} B (24 layers x 11) | distinct rsafe values = {distinct}")
    assert len(tbl) == L * (6 + NDOWN), "size mismatch"


if __name__ == "__main__":
    main()
