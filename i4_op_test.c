/* i4_op_test.c — isolate each RVV op used in int4_unpack_fixed_rvv. */
#include <stdio.h>
#include <stdint.h>
#include <riscv_vector.h>

#define VL 32

static void dump8(const char *tag, const int8_t *a, int n, const char *fmt) {
    printf("%s: ", tag);
    for (int i = 0; i < n; i++) printf(fmt, a[i]);
    printf("\n");
}

int main(void) {
    int8_t src[VL];      /* nibble bytes */
    int8_t hi_out[VL], lo_out[VL];
    int16_t prod[VL];
    int8_t clip[VL];
    int8_t inter[VL*2];

    /* known nibble bytes: byte0=0x2e (lo=0xe->-2, hi=0x2->2), byte1=0xf1 (lo=1, hi=0xf->-1), byte2=0x80 (lo=0,hi=8->-8) */
    src[0]=0x2e; src[1]=0xf1; src[2]=0x80;
    for (int i=3;i<VL;i++) src[i]=0x00;
    size_t vl = 32;
    vint8m2_t v = vle8_v_i8m2(src, vl);

    /* TEST hi: arithmetic shift right 4 */
    vint8m2_t hi = vsra_vx_i8m2(v, 4, vl);
    vse8_v_i8m2(hi_out, hi, vl);
    printf("--- vsra(v,4) high-nibble sign-ext ---\n");
    dump8("hi", hi_out, 3, "%d ");

    /* TEST lo: (v<<4)>>4 */
    vint8m2_t lo = vsra_vx_i8m2(vsll_vx_i8m2(v, 4, vl), 4, vl);
    vse8_v_i8m2(lo_out, lo, vl);
    dump8("lo", lo_out, 3, "%d ");

    /* TEST vwadd (sign-extend i8->i16) + vmul */
    int16_t s = 2926;
    vint16m4_t hi16 = vmul_vx_i16m4(vwadd_vx_i16m4(hi, 0, vl), s, vl);
    vse16_v_i16m4(prod, hi16, vl);
    printf("--- vwadd+vmul(q*2926) ---\n");
    dump8("prod", (int8_t*)prod, 6, "%d ");

    /* TEST vadd+vnclip */
    vint16m4_t h = vadd_vx_i16m4(hi16, 128, vl);
    vint8m2_t cl = vnclip_wx_i8m2(h, 8, vl);
    vse8_v_i8m2(clip, cl, vl);
    printf("--- vadd(128)+vnclip(,8) -> i8 ---\n");
    dump8("clip", clip, 3, "%d ");

    /* TEST interleave: even from lo8, odd from hi8 via 16-bit store */
    vint8m2_t cllo = vnclip_wx_i8m2(vadd_vx_i16m4(vmul_vx_i16m4(vwadd_vx_i16m4(lo,0,vl), s, vl),128,vl), 8, vl);
    vint16m4_t c = vor_vv_i16m4(
        vwadd_vx_i16m4(cllo, 0, vl),
        vsll_vx_i16m4(vwadd_vx_i16m4(cl, 0, vl), 8, vl), vl);
    vse16_v_i16m4((int16_t*)inter, c, vl);
    printf("--- interleave [lo0,hi0,lo1,hi1,...] ---\n");
    for (int i=0;i<6;i++) printf("  out[%d]=%d ", i, inter[i]);
    printf("\n");
    return 0;
}
