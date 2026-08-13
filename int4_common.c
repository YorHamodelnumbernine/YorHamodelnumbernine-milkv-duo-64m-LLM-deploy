/* int4_common.c — shared INT4 pack/unpack implementation (host-side tools).
 * Single source of truth for int4_proto.c and convert_i4.c. */
#include "int4_common.h"
#include <string.h>
#include <math.h>

/* ---------- tiny fp16 helpers (host) ---------- */
uint16_t float_to_fp16(float x) {
    uint32_t u; memcpy(&u, &x, 4);
    uint32_t sign = (u >> 16) & 0x8000;
    int32_t  exp  = (u >> 23) & 0xff;
    uint32_t mant = u & 0x7fffff;
    if (exp == 0xff) return (uint16_t)(sign | 0x7c00 | (mant ? 0x200 : 0)); /* inf/nan */
    int32_t e = exp - 127 + 15;
    if (e >= 0x1f) return (uint16_t)(sign | 0x7c00);       /* overflow -> inf */
    if (e <= 0) {
        /* subnormal */
        if (e < -10) return (uint16_t)sign;
        mant |= 0x800000;
        int shift = 14 - e;
        uint32_t half = mant >> shift;
        if (half & 0x200) half += 0x400;                    /* round */
        return (uint16_t)(sign | (half >> 3));
    }
    uint32_t h = sign | ((uint32_t)e << 10) | (mant >> 13);
    if (mant & 0x1000) h += 1;                              /* round */
    return (uint16_t)h;
}
float fp16_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    uint32_t u;
    if (exp == 0) {
        if (mant == 0) u = sign;
        else { /* subnormal */
            exp = 1;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3ff;
            u = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 0x1f) {
        u = sign | 0x7f800000 | (mant << 13);
    } else {
        u = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f; memcpy(&f, &u, 4);
    return f;
}

/* ---------- pack: int8 row-major -> nibbles + per-group scale ---------- */
int int4_pack(const int8_t *src, int n, int G, int scale_type,
              uint8_t *out_nib, uint8_t *out_scale) {
    int nsc = n / G;
    for (int g = 0; g < nsc; g++) {
        const int8_t *blk = src + (size_t)g * G;
        int maxv = 0;
        for (int i = 0; i < G; i++) { int a = blk[i]; if (a < 0) a = -a; if (a > maxv) maxv = a; }
        float s = (float)maxv / 7.0f;
        if (s < 1e-9f) s = 1.0f;
        if (scale_type == 0) memcpy(out_scale + (size_t)g * 4, &s, 4);
        else { uint16_t h = float_to_fp16(s); memcpy(out_scale + (size_t)g * 2, &h, 2); }
        uint8_t *nb = out_nib + (size_t)g * G / 2;
        for (int i = 0; i < G; i += 2) {
            int q0 = (int)lrintf((float)blk[i] / s);     if (q0 < -8) q0 = -8; if (q0 > 7) q0 = 7;
            int q1 = (int)lrintf((float)blk[i+1] / s);   if (q1 < -8) q1 = -8; if (q1 > 7) q1 = 7;
            nb[i/2] = (uint8_t)((q0 & 0xF) | ((q1 & 0xF) << 4));
        }
    }
    return nsc;
}

/* ---------- unpack: nibbles + scale -> int8 (dequantized, clamped) ---------- */
void int4_unpack(const uint8_t *src_nib, const uint8_t *src_scale,
                 int n, int G, int scale_type, int8_t *dst) {
    int nsc = n / G;
    for (int g = 0; g < nsc; g++) {
        float s;
        if (scale_type == 0) memcpy(&s, src_scale + (size_t)g * 4, 4);
        else { uint16_t h; memcpy(&h, src_scale + (size_t)g * 2, 2); s = fp16_to_float(h); }
        const uint8_t *nb = src_nib + (size_t)g * G / 2;
        int8_t *d = dst + (size_t)g * G;
        for (int i = 0; i < G; i += 2) {
            uint8_t b = nb[i/2];
            int q0 = (int)(b & 0xF); if (q0 >= 8) q0 -= 16;   /* 8..15 -> -8..-1 */
            int q1 = (int)(b >> 4);  if (q1 >= 8) q1 -= 16;
            int v0 = (int)lrintf((float)q0 * s); if (v0 < -128) v0 = -128; if (v0 > 127) v0 = 127;
            int v1 = (int)lrintf((float)q1 * s); if (v1 < -128) v1 = -128; if (v1 > 127) v1 = 127;
            d[i]   = (int8_t)v0;
            d[i+1] = (int8_t)v1;
        }
    }
}

/* fixed-point unpack (faster on device; s_fixed = round(s * 2^Q), Q=8) */
void int4_unpack_fixed(const uint8_t *src_nib, const int16_t *src_scale,
                       int n, int G, int8_t *dst) {
    int nsc = n / G;
    for (int g = 0; g < nsc; g++) {
        int16_t sf = src_scale[g];
        const uint8_t *nb = src_nib + (size_t)g * G / 2;
        int8_t *d = dst + (size_t)g * G;
        for (int i = 0; i < G; i += 2) {
            uint8_t b = nb[i/2];
            int q0 = (int)(b & 0xF); if (q0 >= 8) q0 -= 16;
            int q1 = (int)(b >> 4);  if (q1 >= 8) q1 -= 16;
            int v0 = (q0 * sf + 128) >> 8; if (v0 < -128) v0 = -128; if (v0 > 127) v0 = 127;
            int v1 = (q1 * sf + 128) >> 8; if (v1 < -128) v1 = -128; if (v1 > 127) v1 = 127;
            d[i] = (int8_t)v0; d[i+1] = (int8_t)v1;
        }
    }
}

#if defined(__riscv_vector)
#include <riscv_vector.h>
/* RVV 0.7.1 vectorized fixed-point unpack (C906, VLEN=128).
 * Same math as int4_unpack_fixed:  q = sign-extended 4-bit nibble,
 * v = clamp((q*sf + 128) >> 8).  One group (G=64 elements = 32 nibble
 * bytes) per iteration; 32-lane m2 vectors.
 *
 * Notes:
 *  - q*sf is evaluated in int32: |q|<=8, sf = round(s*256) <= ~4645,
 *    so |q*sf| <= ~37152 overflows int16.
 *  - vnclip applies the vxrm RNU rounding constant (1<<(shift-1)) = 128
 *    internally, i.e. vnclip(x,8) == (x+128)>>8 then saturate to i8 —
 *    exactly the scalar math.  We therefore do NOT add 128 ourselves.
 *  - Interleave via 16-bit stores: low byte = even element (masked to
 *    8 bits), high byte = odd element (sign-extension bits shifted out). */
void int4_unpack_fixed_rvv(const uint8_t *src_nib, const int16_t *src_scale,
                           int n, int G, int8_t *dst) {
    int nsc = n / G;
    size_t vl = G / 2;   /* one group (G elems = G/2 nibble bytes) per iteration */
    for (int g = 0; g < nsc; g++) {
        int16_t s = src_scale[g];
        const int8_t *nb = (const int8_t *)(src_nib + (size_t)g * G / 2);
        int8_t *d = dst + (size_t)g * G;

        vint8m2_t v  = vle8_v_i8m2(nb, vl);
        vint8m2_t lo = vsra_vx_i8m2(vsll_vx_i8m2(v, 4, vl), 4, vl);  /* low nibble */
        vint8m2_t hi = vsra_vx_i8m2(v, 4, vl);                        /* high nibble */

        /* dequant even (low nibble) and odd (high nibble).
         * q*sf in int32 avoids overflow; saturate i32->i16 (shift 0) does not
         * change the final int8 clamp result (shown by range analysis); then
         * vnclip i16->i8 applies the RNU rounding (+128) with >>8. */
        vint32m8_t lo32 = vwmul_vx_i32m8(vwadd_vx_i16m4(lo, 0, vl), s, vl);
        vint16m4_t lo16 = vnclip_wx_i16m4(lo32, 0, vl);
        vint8m2_t  lo8  = vnclip_wx_i8m2(lo16, 8, vl);
        vint32m8_t hi32 = vwmul_vx_i32m8(vwadd_vx_i16m4(hi, 0, vl), s, vl);
        vint16m4_t hi16 = vnclip_wx_i16m4(hi32, 0, vl);
        vint8m2_t  hi8  = vnclip_wx_i8m2(hi16, 8, vl);

        /* interleave: 32 x i16[lo8[i] | hi8[i]<<8] -> 64 contiguous bytes */
        vint16m4_t c = vor_vv_i16m4(
            vand_vx_i16m4(vwadd_vx_i16m4(lo8, 0, vl), 0xFF, vl),
            vsll_vx_i16m4(vwadd_vx_i16m4(hi8, 0, vl), 8, vl), vl);
        vse16_v_i16m4((int16_t *)d, c, vl);
    }
}
#endif /* __riscv_vector */
