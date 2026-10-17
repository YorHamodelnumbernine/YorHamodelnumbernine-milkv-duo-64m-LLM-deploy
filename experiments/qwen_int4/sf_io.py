#!/usr/bin/env python3
"""Minimal safetensors reader that decodes BF16 -> float32 via raw bytes.
Avoids safetensors.numpy's inability to resolve 'bfloat16' dtype on this host.
"""
import json, struct
import numpy as np


class SF:
    def __init__(self, path):
        self.path = path
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            self.header = json.loads(fh.read(n))
            self.data = fh.read()

    def keys(self):
        return [k for k in self.header if k != "__metadata__"]

    def get(self, name):
        info = self.header[name]
        start, end = info["data_offsets"]
        raw = self.data[start:end]
        shape = info["shape"]
        dt = info["dtype"]
        if dt == "BF16":
            u = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
            u = u << 16
            return u.view("<f4").reshape(shape)
        if dt == "F32":
            return np.frombuffer(raw, dtype="<f4").reshape(shape)
        if dt == "F16":
            u = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
            # fp16 -> fp32
            return _fp16_to_fp32(u).reshape(shape)
        if dt == "I64":
            return np.frombuffer(raw, dtype="<i8").reshape(shape)
        raise NotImplementedError(dt)


def _fp16_to_fp32(u):
    u = u.astype(np.uint32)
    s = (u >> 15) & 0x1
    e = (u >> 10) & 0x1F
    m = u & 0x3FF
    out = np.zeros_like(u)
    # normal
    nz = (e != 0) & (e != 31)
    out[nz] = (s[nz] << 31) | ((e[nz] + 112) << 23) | (m[nz] << 13)
    # subnormal
    sn = (e == 0) & (m != 0)
    if sn.any():
        # convert to normal by shifting mantissa
        m2 = m[sn].astype(np.uint64)
        e2 = np.full(m2.shape, 113, np.uint64)
        while (m2 & 0x400) == 0:
            m2 = m2 << 1
            e2 -= 1
        m2 = m2 & 0x3FF
        out[sn] = (s[sn] << 31) | (e2 << 23) | (m2 << 13)
    # zero
    z = (e == 0) & (m == 0)
    out[z] = s[z] << 31
    # inf/nan
    inf = (e == 31) & (m == 0)
    out[inf] = (s[inf] << 31) | 0x7F800000
    nan = (e == 31) & (m != 0)
    out[nan] = (s[nan] << 31) | 0x7FC00000
    return out.view("<f4")
