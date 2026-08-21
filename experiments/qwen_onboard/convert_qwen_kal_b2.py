#!/usr/bin/env python3
"""convert_qwen_kal_b2.py — Qwen K-aligned INT4 nibble 重排: nib[N][16] -> nib16[16][N].

Design B2 (row-pair-major) host-side re-layout, CEO-approved (2026-08-22).
  - 输入: layer{l}_kal.bin (既有 K-aligned G32 布局, convert_qwen_kal.py 生成,
          parse_layer 顺序, nibble 段为 nib[KG][N][16]).
  - 输出: layer{l}_kal_b2.bin (同大小 8,393,728 B, nibble 段重排为 nib16[KG][16][N],
          其余段 gsc/rms 逐字节不变).
  - 板上引擎改动(dequant_kal_rvv)由 B2 实施阶段做; 本工具只改磁盘布局.

重排映射 (每 K-block g, 每列 n, 每 row-pair j):
    new[g*16*N + j*N + n] = old[g*N*16 + n*16 + j]
等价 numpy: nib_old.reshape(KG,N,16).transpose(0,2,1) -> nib_new[KG,16,N].

位精确性构造保持: 同一 raw int4 字节, 仅源排列改变 -> dequant 输出 w[32][N] K-major
逐字节不变 -> TIU 右操作数不变 -> 全回归 bit-exact 可保.

校验 (--check):
  [1] 非 nibble 段 (rms_attn/rms_ffn + 7×gsc) 输出 vs 输入逐字节 cmp == 0.
  [2] 每 nibble 段 scalar-dequant(旧) vs scalar-dequant(新) 逐元素相等 (bit-exact).

用法:
  python3 convert_qwen_kal_b2.py                 # 转换全部 24 层到 <in>/../weights_kal_b2
  python3 convert_qwen_kal_b2.py --in DIR --out DIR --layers 0 1 2
  python3 convert_qwen_kal_b2.py --check-only    # 只做 bit-exact 校验, 不写文件
"""
import os, sys, argparse, hashlib
import numpy as np

# ---- Qwen geometry (mirrors convert_qwen_kal.py / qwen_engine_lmhead2.c) ----
D = 896
DKV = 128
F = 4864
G = 32

# (name, KG, N) in parse_layer order.  KG = K/G; down K=F -> KG=F/G=152.
MAT_GEOM = [
    ("q",     D // G, D),
    ("k",     D // G, DKV),
    ("v",     D // G, DKV),
    ("wo",    D // G, D),
    ("up",    D // G, F),
    ("gate",  D // G, F),
    ("down",  F // G, D),
]
RMS_BYTES = D * 4
LAYER_BYTES = 8393728


def layer_seg_offsets():
    """返回 [(name, KG, N, nib_off, nib_sz, gsc_off, gsc_sz)] 按 parse_layer 顺序."""
    off = RMS_BYTES          # rms_attn
    segs = []
    for name, KG, N in MAT_GEOM:
        nib_sz = KG * N * 16
        gsc_sz = KG * N * 2
        segs.append((name, KG, N, off, nib_sz, off + nib_sz, gsc_sz))
        off += nib_sz + gsc_sz
    off += RMS_BYTES          # rms_ffn at very end
    assert off == LAYER_BYTES, f"layer seg layout mismatch: {off} != {LAYER_BYTES}"
    return segs


SEGS = layer_seg_offsets()
NIB_TOTAL = sum(s[4] for s in SEGS)
GSC_TOTAL = sum(s[6] for s in SEGS)
assert NIB_TOTAL == 7454720, NIB_TOTAL     # 7,454,720 B (文档原 7,444,720 为笔误)
assert GSC_TOTAL == 931840, GSC_TOTAL      # 931,840 B


def relayout_nibble(nib_old: bytes, KG: int, N: int) -> bytes:
    """nib[KG,N,16] -> nib16[KG,16,N] (纯转置, 位不变)."""
    arr = np.frombuffer(nib_old, dtype=np.uint8).reshape(KG, N, 16)
    return np.ascontiguousarray(arr.transpose(0, 2, 1)).tobytes()


def dequant_scalar(nib: np.ndarray, KG: int, N: int) -> np.ndarray:
    """解包为 w[KG,32,N] K-major INT8, 全 numpy 向量化 (逐元素语义与标量参考一致).
    nib 形状: 旧 (KG,N,16) -> lo/hi 各 [KG,N,16], 转置成 [KG,16,N] 后写入偶/奇行;
              新 (KG,16,N) -> 直接写入偶/奇行."""
    assert nib.ndim == 3 and nib.shape[0] == KG
    lo4 = nib & 0x0F
    hi4 = (nib >> 4) & 0x0F
    lo = np.where(lo4 > 7, lo4 - 16, lo4).astype(np.int8)   # 符号扩展
    hi = np.where(hi4 > 7, hi4 - 16, hi4).astype(np.int8)
    if nib.shape[1] == N:          # 旧 (KG,N,16): [n][j] -> [j][n]
        lo = np.ascontiguousarray(lo.transpose(0, 2, 1))
        hi = np.ascontiguousarray(hi.transpose(0, 2, 1))
    elif nib.shape[2] != N:
        raise AssertionError(f"unexpected nib shape {nib.shape} (KG={KG}, N={N})")
    w = np.empty((KG, 32, N), dtype=np.int8)
    w[:, 0::2, :] = lo
    w[:, 1::2, :] = hi
    return w


def convert_layer(layer: bytes) -> bytes:
    """整层重排: nibble 段转置, gsc/rms 逐字节透传."""
    out = bytearray(LAYER_BYTES)
    for name, KG, N, nib_off, nib_sz, gsc_off, gsc_sz in SEGS:
        nib16 = relayout_nibble(layer[nib_off:nib_off + nib_sz], KG, N)
        out[nib_off:nib_off + nib_sz] = nib16
        # gsc / rms_attn(已写) / rms_ffn: 透传 (byte-identical)
    # rms_attn (0..RMS_BYTES) 与 rms_ffn (LAYER_BYTES-RMS_BYTES..) 原样拷贝
    out[0:RMS_BYTES] = layer[0:RMS_BYTES]
    out[LAYER_BYTES - RMS_BYTES:LAYER_BYTES] = layer[LAYER_BYTES - RMS_BYTES:LAYER_BYTES]
    # gsc 段原样拷贝
    for name, KG, N, nib_off, nib_sz, gsc_off, gsc_sz in SEGS:
        out[gsc_off:gsc_off + gsc_sz] = layer[gsc_off:gsc_off + gsc_sz]
    return bytes(out)


def check_layer(layer: bytes, tag: str) -> None:
    """bit-exact 校验: gsc/rms 透传 cmp + 每 nibble 段 dequant(旧)==dequant(新)."""
    b2 = convert_layer(layer)
    # [1] 非 nibble 段逐字节一致
    for name, KG, N, nib_off, nib_sz, gsc_off, gsc_sz in SEGS:
        assert layer[gsc_off:gsc_off + gsc_sz] == b2[gsc_off:gsc_off + gsc_sz], \
            f"{tag} {name} gsc mismatch"
    assert layer[0:RMS_BYTES] == b2[0:RMS_BYTES], f"{tag} rms_attn mismatch"
    assert layer[LAYER_BYTES - RMS_BYTES:] == b2[LAYER_BYTES - RMS_BYTES:], \
        f"{tag} rms_ffn mismatch"
    # [2] 每 nibble 段 dequant 等价
    for name, KG, N, nib_off, nib_sz, gsc_off, gsc_sz in SEGS:
        old = np.frombuffer(layer[nib_off:nib_off + nib_sz], dtype=np.uint8).reshape(KG, N, 16)
        new = np.frombuffer(b2[nib_off:nib_off + nib_sz], dtype=np.uint8).reshape(KG, 16, N)
        wo = dequant_scalar(old, KG, N)
        wn = dequant_scalar(new, KG, N)
        assert np.array_equal(wo, wn), f"{tag} {name} dequant mismatch"
    print(f"[check] {tag}: gsc/rms byte-identical + dequant bit-exact OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights_kal"))
    ap.add_argument("--out", dest="out_dir",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights_kal_b2"))
    ap.add_argument("--layers", type=int, nargs="*", help="layers to convert (default: 0..23)")
    ap.add_argument("--check-only", action="store_true", help="只校验, 不写输出")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    layers = args.layers if args.layers is not None else list(range(24))
    sha_in, sha_out = hashlib.sha256(), hashlib.sha256()
    for l in layers:
        inp = os.path.join(args.in_dir, f"layer{l}_kal.bin")
        with open(inp, "rb") as f:
            layer = f.read()
        if len(layer) != LAYER_BYTES:
            print(f"[skip] layer{l}: size {len(layer)} != {LAYER_BYTES} (非 kal 格式?)")
            continue
        check_layer(layer, f"layer{l}")          # 转换前先证位精确构造
        b2 = convert_layer(layer)
        check_layer(b2, f"layer{l}_b2")          # 转换后再证一次 (幂等性)
        if not args.check_only:
            outp = os.path.join(args.out_dir, f"layer{l}_kal_b2.bin")
            with open(outp, "wb") as f:
                f.write(b2)
            print(f"[conv ] layer{l}: {inp} -> {outp}")
        sha_in.update(layer); sha_out.update(b2)

    if args.check_only:
        print(f"[ok] bit-exact check passed for {len(layers)} layers")
    else:
        print(f"[ok] wrote {len(layers)} layers -> {args.out_dir}")
        print(f"     sha256(layer-in) = {sha_in.hexdigest()}")
        print(f"     sha256(layer_b2) = {sha_out.hexdigest()}")
        print(f"     nibble total = {NIB_TOTAL} B | gsc total = {GSC_TOTAL} B | layer = {LAYER_BYTES} B")


if __name__ == "__main__":
    main()
