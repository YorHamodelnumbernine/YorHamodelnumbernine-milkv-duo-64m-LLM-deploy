/* per_row_quant.c — engine-ready per-row (per-token) INT8 activation quant.
 *
 * Step ① of CEO-adopted Path A implementation order.  Isolates the per-row
 * activation scale from qwen_kal_ref.c::quant_per_row() into a standalone,
 * verifiable unit so the C906B TIU microkernel can consume it directly.
 *
 * Semantics (locked to qwen_kal_ref.c, host 3/3):
 *   sc[m] = max_k |x[m,k]| / 127.0     (floor 1e-12 to avoid div0)
 *   q[m,k] = clamp(round_bankers(x[m,k] / sc[m]), -128, 127)
 *   round_bankers = round-half-to-even, i.e. numpy np.round parity.
 *   (NOT round-half-away; must match host so on-board logits match C ref.)
 *
 * Engine call sites (Qwen2.5-0.5B Path A decode, M=1):
 *   - post rms_attn : q/k/v/up/gate left operand  (h [1,896])
 *   - post attn     : wo left operand             (attn [1,896])
 *   - post SiLU     : down per K-chunk            (mid slice [1,1024])
 *
 * Build (host x86): gcc -O2 -o per_row_quant per_row_quant.c -lm
 * Deploy/C906B:     riscv64-unknown-linux-musl-gcc -O2 -static ... (same source)
 *
 * Usage:
 *   per_row_quant --selftest                     built-in edge-case checks (RC=0)
 *   per_row_quant <M> <K> < in.f32               read M*K fp32, stdout q (M*K i8)
 *                                                 then sc (M f32)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

/* numpy np.round parity: round-half-to-even (banker's). */
static inline int32_t round_bankers(float v) {
    float f = floorf(v);
    float d = v - f;
    if (d > 0.5f) return (int32_t)(f + 1.0f);
    if (d < 0.5f) return (int32_t)f;
    return ((int32_t)f % 2 == 0) ? (int32_t)f : (int32_t)(f + 1.0f);
}

/* Quantize rows of x [M,K] to int8 q [M,K] with per-row scale sc [M].
 * sc[m] = max|x|/127 (floor 1e-12).  This is THE engine entry point. */
void per_row_quant(const float *x, int M, int K, int8_t *q, float *sc) {
    for (int m = 0; m < M; m++) {
        const float *xr = x + (size_t)m * K;
        float mx = 0.0f;
        for (int k = 0; k < K; k++) { float a = fabsf(xr[k]); if (a > mx) mx = a; }
        float s = mx / 127.0f;
        if (s < 1e-12f) s = 1e-12f;
        sc[m] = s;
        int8_t *qr = q + (size_t)m * K;
        for (int k = 0; k < K; k++) {
            float v = xr[k] / s;
            int32_t ri = round_bankers(v);
            if (ri > 127) ri = 127;
            if (ri < -128) ri = -128;
            qr[k] = (int8_t)ri;
        }
    }
}

/* M=1 decode fast path: single scalar scale, avoids m-loop overhead.
 * Same semantics; engine calls this for decode, per_row_quant for prefill M>1. */
static inline void per_row_quant_decode(const float *x, int K, int8_t *q, float *sc) {
    float mx = 0.0f;
    for (int k = 0; k < K; k++) { float a = fabsf(x[k]); if (a > mx) mx = a; }
    float s = mx / 127.0f; if (s < 1e-12f) s = 1e-12f;
    sc[0] = s;
    for (int k = 0; k < K; k++) {
        float v = x[k] / s;
        int32_t ri = round_bankers(v);
        if (ri > 127) ri = 127;
        if (ri < -128) ri = -128;
        q[k] = (int8_t)ri;
    }
}

/* ---------------- selftest ---------------- */
static int nfail = 0;
static void check_i8(const char *tag, int8_t got, int8_t exp) {
    if (got != exp) { printf("  FAIL %-24s got %d exp %d\n", tag, (int)got, (int)exp); nfail++; }
    else            printf("  ok   %-24s %d\n", tag, (int)got);
}
static void check_f32(const char *tag, float got, float exp, float tol) {
    if (fabsf(got - exp) > tol) { printf("  FAIL %-24s got %.6f exp %.6f\n", tag, got, exp); nfail++; }
    else                         printf("  ok   %-24s %.6f\n", tag, got);
}

/* Fixed vectors: x row -> expected q (per_row_quant semantics).
 * sc = max/127; q = round_bankers(x/sc) clamp [-128,127]. */
static void selftest(void) {
    printf("[per_row_quant selftest]\n");
    /* Row: [0,0,0] -> sc=1e-12, q all 0 */
    {
        float x[3] = {0, 0, 0}; int8_t q[3]; float sc[1];
        per_row_quant(x, 1, 3, q, sc);
        check_f32("zero row sc", sc[0], 1e-12f, 1e-18f);
        check_i8("zero row q0", q[0], 0);
    }
    /* Row: [63.5, 64, 127, -127] -> mx=127, sc=1.0
     * 63.5/1 = 63.5 -> half-way -> round-half-even -> 64 (64 even)
     * 64/1 = 64, 127/1=127, -127/1=-127 */
    {
        float x[4] = {63.5f, 64.0f, 127.0f, -127.0f}; int8_t q[4]; float sc[1];
        per_row_quant(x, 1, 4, q, sc);
        check_f32("sc=127 row sc", sc[0], 1.0f, 1e-7f);
        check_i8("63.5 half->even(64)", q[0], 64);
        check_i8("64 exact", q[1], 64);
        check_i8("127 max", q[2], 127);
        check_i8("-127 min", q[3], -127);
    }
    /* Row: [62.5] -> mx=62.5, sc=62.5/127.  x/sc = 127.0 exactly -> q=127.
     * Verify scale math: x/sc = 62.5 / (62.5/127) = 127.0 */
    {
        float x[1] = {62.5f}; int8_t q[1]; float sc[1];
        per_row_quant(x, 1, 1, q, sc);
        check_f32("sc=62.5/127 sc", sc[0], 62.5f / 127.0f, 1e-6f);
        check_i8("62.5->127", q[0], 127);
    }
    /* Round-half-even edge: x = [0.5, 1.5, 2.5, 3.5] with mx=3.5, sc=3.5/127.
     * x/sc = [18.14, 54.43, 90.71, 127].  None exactly half.  Use mx=2:
     * x=[1.5, 2.5] mx=2.5 sc=2.5/127; x/sc = [76.2, 127].  Not clean.
     * Direct rounding unit test below on raw values instead. */
    /* Round-half-even raw: v=2.5->2, 3.5->4, 4.5->4, 5.5->6 (banker's) */
    check_i8("round_bankers(2.5)=2", (int8_t)round_bankers(2.5f), 2);
    check_i8("round_bankers(3.5)=4", (int8_t)round_bankers(3.5f), 4);
    check_i8("round_bankers(4.5)=4", (int8_t)round_bankers(4.5f), 4);
    check_i8("round_bankers(5.5)=6", (int8_t)round_bankers(5.5f), 6);
    check_i8("round_bankers(6.5)=6", (int8_t)round_bankers(6.5f), 6);
    check_i8("round_bankers(-2.5)=-2", (int8_t)round_bankers(-2.5f), -2);
    check_i8("round_bankers(-3.5)=-4", (int8_t)round_bankers(-3.5f), -4);
    /* Saturation: x = [1000, -1000] mx=1000 sc=1000/127; x/sc = ±127 -> q ±127 */
    {
        float x[2] = {1000.0f, -1000.0f}; int8_t q[2]; float sc[1];
        per_row_quant(x, 1, 2, q, sc);
        check_i8("sat +1000->127", q[0], 127);
        check_i8("sat -1000->-127", q[1], -127);
    }
    /* decode fast path == per_row_quant (M=1) on same row */
    {
        float x[6] = {-3.2f, 0.0f, 5.5f, -5.5f, 1.0f, 2.0f};
        int8_t qa[6], qb[6]; float sca[1], scb[1];
        per_row_quant(x, 1, 6, qa, sca);
        per_row_quant_decode(x, 6, qb, scb);
        int ok = 1;
        for (int i = 0; i < 6; i++) if (qa[i] != qb[i]) ok = 0;
        if (!ok || sca[0] != scb[0]) { printf("  FAIL decode fastpath mismatch\n"); nfail++; }
        else printf("  ok   decode fastpath == generic (M=1)\n");
    }
    printf("[selftest] %s (RC=%d)\n", nfail ? "FAIL" : "PASS", nfail ? 1 : 0);
}

int main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "--selftest")) { selftest(); return nfail ? 1 : 0; }
    if (argc != 3) { fprintf(stderr, "usage: %s --selftest | <M> <K> < in.f32\n", argv[0]); return 2; }
    int M = atoi(argv[1]), K = atoi(argv[2]);
    size_t n = (size_t)M * K;
    float *x = malloc(n * sizeof(float));
    if (fread(x, sizeof(float), n, stdin) != n) { fprintf(stderr, "short input\n"); return 2; }
    int8_t *q = malloc(n);
    float *sc = malloc((size_t)M * sizeof(float));
    if (M == 1) per_row_quant_decode(x, K, q, sc);
    else        per_row_quant(x, M, K, q, sc);
    fwrite(q, 1, n, stdout);
    fwrite(sc, sizeof(float), M, stdout);
    free(x); free(q); free(sc);
    return 0;
}
