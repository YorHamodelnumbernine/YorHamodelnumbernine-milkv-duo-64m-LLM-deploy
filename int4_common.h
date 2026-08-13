#ifndef INT4_COMMON_H
#define INT4_COMMON_H
/* Shared INT4 pack/unpack — single source of truth for int4_proto.c
 * and convert_i4.c (host-side tools).  Ported to device later for pf_worker.
 *
 * Formats (see DESIGN_INT4_WEIGHTS.md):
 *   nibble byte:  low nibble = element 2i, high nibble = element 2i+1
 *   scales:       one per G elements, fp32 (4B) or fp16 (2B)
 */
#include <stdint.h>

uint16_t float_to_fp16(float x);
float    fp16_to_float(uint16_t h);

/* Pack row-major int8 -> nibbles + per-group scale.
 * out_nib: n/2 bytes; out_scale: (n/G) scales (fp32 or fp16).
 * scale_type: 0=fp32, 1=fp16.  Returns number of scales written. */
int  int4_pack(const int8_t *src, int n, int G, int scale_type,
               uint8_t *out_nib, uint8_t *out_scale);

/* Unpack nibbles + scale -> dequantized int8 (clamped to int8 range). */
void int4_unpack(const uint8_t *src_nib, const uint8_t *src_scale,
                 int n, int G, int scale_type, int8_t *dst);

/* Fixed-point unpack (faster on device; s_fixed = round(s * 2^Q), Q=8). */
void int4_unpack_fixed(const uint8_t *src_nib, const int16_t *src_scale,
                       int n, int G, int8_t *dst);

#if defined(__riscv_vector)
/* RVV 0.7.1 vectorized fixed-point unpack (C906).  Same math as
 * int4_unpack_fixed; only available when the vector extension is on. */
void int4_unpack_fixed_rvv(const uint8_t *src_nib, const int16_t *src_scale,
                           int n, int G, int8_t *dst);
#endif

#endif
