/* twopass_matmul.c — engine-oriented per-chunk KG=32 two-pass matmul.
 *
 * Step ② of CEO-adopted Path A implementation order.  Restructures
 * qwen_kal_ref.c::chunk_matmul_twopass() into the exact sequence the C906B
 * TIU will execute, so the microkernel maps 1:1 to on-board cmdbufs.
 *
 * ENGINE SUBMIT SCHEDULE (per matrix, per token):
 *   wmax = max|W| over whole matrix            -> rsafe = mm_rshift(32,wmax)-3 (>=4)
 *   for g in 0..K/G-1:                          // KG=32 blocks
 *     TIU pass1 cmdbuf(g): ALL N-tiles, rshift=rsafe   -> p1 [M,N] -> P1_BUF
 *     CPU:  max|p1| over [M,N] x 2^rsafe -> est -> r_opt(g)   // single scalar
 *     TIU pass2 cmdbuf(g): ALL N-tiles, rshift=r_opt  -> p2 [M,N] -> P2_BUF
 *     CPU:  out[m,n] += p2[m,n] * 2^r_opt * gsc[g,n]
 *   out[m,:] *= sc_row[m]
 *
 * One cmdbuf covers ALL N-tiles of a block for one pass => r_opt is the block's
 * GLOBAL scalar (max over all output cols), matching TPU Gate② "56 submits per
 * chunk-matrix = 28 blocks x 2 passes" and the qwen_kal_ref reference exactly.
 *
 * API is unified over M (decode M=1 + prefill M>1): rshift_from_pass1() reduces
 * over all rows, so M=1 yields a single scalar r_opt (engine patches it into the
 * pass2 cmdbuf rshift field).  M>1 keeps the same semantics (per-block scalar)
 * so prefill needs no interface change.
 *
 * Usage:
 *   twopass_matmul --selftest
 *   twopass_matmul <M> <K> <N> < stdin  (x_i8 M*K, w_i8 K*N, gsc f32 K/G*N, sc_row f32 M)
 *                                        stdout: out fp32 M*N (caller-zeroed input not needed)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#define G 32

/* TIU round-half-up toward +inf: sat8((acc + 2^(r-1)) >> r).  Locked on-board
 * (GATE_A_SIGNOFF / patha_kg32_check).  r=0 -> half=0 (acc, clamped). */
static inline int8_t int8_round_div(int32_t acc, int rshift) {
    int32_t scale = 1 << rshift;
    int32_t half = scale >> 1;
    int32_t r = (acc + half) >> rshift;
    if (r > 127) r = 127;
    if (r < -128) r = -128;
    return (int8_t)r;
}
static int32_t max_i8(const int8_t *a, size_t n) {
    int32_t m = 0; for (size_t i = 0; i < n; i++) { int32_t v = a[i]; if (v < 0) v = -v; if (v > m) m = v; } return m;
}
static int matmul_rshift_w(int K, int wmax) {
    int r = 0; long long md = (long long)K * 127 * wmax;
    while ((md >> r) > 127) r++;
    return r;
}

typedef struct {
    int M, K, N;
    const int8_t *x;       /* [M,K] per-row int8 activations */
    const int8_t *w;       /* [K,N] raw int4-dequant values (engine: DQ_BUF) */
    const float *gsc;      /* [K/G,N] fp16->fp32 group scales */
    const float *sc_row;   /* [M] per-row activation scale */
    float *out;            /* [M,N] fp32 accumulate (engine: ACT/ACC) */
} TwopassCtx;

/* pass1: p1[m,n] = sat8((x[m,gG..] . w[gG..,n] + 2^(rsafe-1)) >> rsafe) */
static void block_pass1(const TwopassCtx *c, int g, int rsafe, int8_t *p1) {
    const int8_t *xs = c->x;
    const int8_t *ws = c->w + (size_t)g * G * c->N;
    int N = c->N;
    for (int m = 0; m < c->M; m++) {
        const int8_t *xr = xs + (size_t)m * c->K + (size_t)g * G;
        int8_t *p1r = p1 + (size_t)m * N;
        for (int n = 0; n < N; n++) {
            int32_t s = 0;
            const int8_t *wc = ws + n;
            for (int k = 0; k < G; k++) s += (int32_t)xr[k] * (int32_t)wc[(size_t)k * N];
            p1r[n] = int8_round_div(s, rsafe);
        }
    }
}
/* r_opt from pass1: est = max|p1| * 2^rsafe ; smallest r with est <= 127*2^r.
 * Single scalar per (block).  M=1 -> decode fast path (same code path). */
static int rshift_from_pass1(const int8_t *p1, int M, int N, int rsafe) {
    int32_t est = 0;
    for (int m = 0; m < M; m++) {
        const int8_t *p1r = p1 + (size_t)m * N;
        for (int n = 0; n < N; n++) {
            int32_t v = p1r[n]; if (v < 0) v = -v; if (v > est) est = v;
        }
    }
    est = est * (1 << rsafe);
    int r = 0; while (est > (127 << r)) r++;
    return r;
}
/* pass2: p2[m,n] = sat8((acc + 2^(r-1)) >> r) */
static void block_pass2(const TwopassCtx *c, int g, int r, int8_t *p2) {
    const int8_t *xs = c->x;
    const int8_t *ws = c->w + (size_t)g * G * c->N;
    int N = c->N;
    for (int m = 0; m < c->M; m++) {
        const int8_t *xr = xs + (size_t)m * c->K + (size_t)g * G;
        int8_t *p2r = p2 + (size_t)m * N;
        for (int n = 0; n < N; n++) {
            int32_t s = 0;
            const int8_t *wc = ws + n;
            for (int k = 0; k < G; k++) s += (int32_t)xr[k] * (int32_t)wc[(size_t)k * N];
            p2r[n] = int8_round_div(s, r);
        }
    }
}
/* accumulate: out[m,n] += p2[m,n] * 2^r * gsc[g,n] */
static void block_accum(TwopassCtx *c, int g, int r, const int8_t *p2) {
    int N = c->N;
    float f = (float)(1 << r);
    for (int m = 0; m < c->M; m++) {
        const int8_t *p2r = p2 + (size_t)m * N;
        float *or_ = c->out + (size_t)m * N;
        const float *gr = c->gsc + (size_t)g * N;
        for (int n = 0; n < N; n++)
            or_[n] += (float)p2r[n] * f * gr[n];
    }
}

/* Unified two-pass matmul.  Engine maps block_pass1/block_pass2 to TIU submits
 * and rshift_from_pass1/block_accum to CPU reads; this host version runs the
 * same schedule.  Integer semantics (r_opt, pass1/pass2 int8) are BIT-EXACT with
 * qwen_kal_ref.c::chunk_matmul_twopass; the fp32 accumulate is the engine's
 * deliberate precision choice (reference uses fp64 gold; divergence <~1e-3 rel). */
void twopass_matmul(TwopassCtx *c, int *r_opt_out /* NULL ok, [K/G] */) {
    int KG = c->K / G;
    int wmax = max_i8(c->w, (size_t)c->K * c->N);
    int rsafe = matmul_rshift_w(G, wmax) - 3; if (rsafe < 4) rsafe = 4;
    int8_t *p1 = malloc((size_t)c->M * c->N);
    int8_t *p2 = malloc((size_t)c->M * c->N);
    if (!p1 || !p2) { fprintf(stderr, "oom\n"); exit(1); }
    memset(c->out, 0, (size_t)c->M * c->N * sizeof(float));
    for (int g = 0; g < KG; g++) {
        block_pass1(c, g, rsafe, p1);                 /* TIU pass1 -> P1_BUF */
        int r = rshift_from_pass1(p1, c->M, c->N, rsafe); /* CPU read max -> r_opt */
        if (r_opt_out) r_opt_out[g] = r;
        block_pass2(c, g, r, p2);                     /* TIU pass2 -> P2_BUF */
        block_accum(c, g, r, p2);                     /* CPU accumulate */
    }
    for (int m = 0; m < c->M; m++) {
        float sc = c->sc_row[m];
        float *or_ = c->out + (size_t)m * c->N;
        for (int n = 0; n < c->N; n++) or_[n] *= sc;
    }
    free(p1); free(p2);
}

/* ---------------- selftest: BIT-EXACT vs qwen_kal_ref reference ---------------- */
static void ref_chunk_twopass(const int8_t *x, int M, int K, const int8_t *w, int N,
                              const float *gsc, const float *sc_row, float *out) {
    int KG = K / G;
    int wmax = max_i8(w, (size_t)K * N);
    int rsafe = matmul_rshift_w(G, wmax) - 3; if (rsafe < 4) rsafe = 4;
    double *accd = calloc((size_t)M * N, sizeof(double));
    int32_t *acc = malloc((size_t)M * N * sizeof(int32_t));
    for (int g = 0; g < KG; g++) {
        const int8_t *xs = x; const int8_t *ws = w + (size_t)g * G * N;
        for (int m = 0; m < M; m++) {
            const int8_t *xr = xs + (size_t)m * K + (size_t)g * G;
            int32_t *ar = acc + (size_t)m * N;
            for (int n = 0; n < N; n++) {
                int32_t s = 0; const int8_t *wc = ws + n;
                for (int k = 0; k < G; k++) s += (int32_t)xr[k] * (int32_t)wc[(size_t)k * N];
                ar[n] = s;
            }
        }
        int32_t est = 0;
        for (int m = 0; m < M; m++) { int32_t *ar = acc + (size_t)m * N;
            for (int n = 0; n < N; n++) { int32_t v = int8_round_div(ar[n], rsafe); if (v < 0) v = -v; if (v > est) est = v; } }
        est = est * (1 << rsafe); int r = 0; while (est > (127 << r)) r++;
        for (int m = 0; m < M; m++) { int32_t *ar = acc + (size_t)m * N; double *dr = accd + (size_t)m * N;
            const float *gr = gsc + (size_t)g * N;
            for (int n = 0; n < N; n++) { int32_t p2 = int8_round_div(ar[n], r); dr[n] += (double)p2 * (double)(1 << r) * (double)gr[n]; } }
    }
    for (int m = 0; m < M; m++) { const double *dr = accd + (size_t)m * N; float *or_ = out + (size_t)m * N;
        float sc = sc_row[m]; for (int n = 0; n < N; n++) or_[n] = (float)dr[n] * sc; }
    free(accd); free(acc);
}

static int nfail = 0;
/* Reference r_opt list (fp64-gold), to verify the restructure's integer
 * two-pass semantics are BIT-IDENTICAL regardless of accumulate precision. */
static void ref_rshift_list(const int8_t *x, int M, int K, const int8_t *w, int N, int *rout) {
    int KG = K / G;
    int wmax = max_i8(w, (size_t)K * N);
    int rsafe = matmul_rshift_w(G, wmax) - 3; if (rsafe < 4) rsafe = 4;
    int32_t *acc = malloc((size_t)M * N * sizeof(int32_t));
    for (int g = 0; g < KG; g++) {
        const int8_t *xs = x; const int8_t *ws = w + (size_t)g * G * N;
        for (int m = 0; m < M; m++) {
            const int8_t *xr = xs + (size_t)m * K + (size_t)g * G;
            int32_t *ar = acc + (size_t)m * N;
            for (int n = 0; n < N; n++) {
                int32_t s = 0; const int8_t *wc = ws + n;
                for (int k = 0; k < G; k++) s += (int32_t)xr[k] * (int32_t)wc[(size_t)k * N];
                ar[n] = s;
            }
        }
        int32_t est = 0;
        for (int m = 0; m < M; m++) { int32_t *ar = acc + (size_t)m * N;
            for (int n = 0; n < N; n++) { int32_t v = int8_round_div(ar[n], rsafe); if (v < 0) v = -v; if (v > est) est = v; } }
        est = est * (1 << rsafe); int r = 0; while (est > (127 << r)) r++;
        rout[g] = r;
    }
    free(acc);
}
static void run_case(int M, int K, int N, unsigned seed) {
    srand(seed);
    int KG = K / G;
    int8_t *x = malloc((size_t)M * K), *w = malloc((size_t)K * N);
    float *gsc = malloc((size_t)KG * N * sizeof(float));
    float *srow = malloc(M * sizeof(float));
    float *o1 = malloc((size_t)M * N * sizeof(float)), *o2 = malloc((size_t)M * N * sizeof(float));
    int *r_new = malloc(KG * sizeof(int)), *r_ref = malloc(KG * sizeof(int));
    for (int i = 0; i < M * K; i++) x[i] = (int8_t)(rand() % 255 - 127);
    for (int i = 0; i < K * N; i++) w[i] = (int8_t)(rand() % 15 - 7);   /* int4 range */
    for (int i = 0; i < KG * N; i++) gsc[i] = 0.001f + (float)(rand() % 1000) / 100000.0f;
    for (int i = 0; i < M; i++) srow[i] = 0.001f + (float)(rand() % 1000) / 10000.0f;
    TwopassCtx c = { M, K, N, x, w, gsc, srow, o1 };
    twopass_matmul(&c, r_new);
    ref_chunk_twopass(x, M, K, w, N, gsc, srow, o2);
    ref_rshift_list(x, M, K, w, N, r_ref);
    int r_bad = 0; for (int g = 0; g < KG; g++) if (r_new[g] != r_ref[g]) r_bad++;
    double maxrel = 0;
    for (int i = 0; i < M * N; i++) {
        double rel = o2[i] != 0 ? fabs((double)o1[i] - o2[i]) / fabs((double)o2[i]) : fabs((double)o1[i]);
        if (rel > maxrel) maxrel = rel;
    }
    const char *ropt = (r_bad == 0) ? "r_opt IDENTICAL" : "r_opt DIFF";
    const char *acc = (maxrel < 1e-2) ? "fp32-acc OK" : "fp32-acc OVER-TOL";
    printf("  [%dx%d x %d] %s | %s (maxrel=%.3e)\n", M, K, N, ropt, acc, maxrel);
    if (r_bad) { nfail++; printf("    r_opt mismatch sample: %d vs %d\n", r_new[0], r_ref[0]); }
    /* r_opt must be exact; fp32-vs-fp64 accumulate tolerance is loose here because
     * this is an adversarial random case (full-scale act + spread gsc).  The real
     * Qwen-data bound is checked in test_twopass_matmul.py (step 2). */
    if (maxrel >= 1e-2) nfail++;
    free(x); free(w); free(gsc); free(srow); free(o1); free(o2); free(r_new); free(r_ref);
}

int main(int argc, char **argv) {
    if (argc >= 2 && !strcmp(argv[1], "--selftest")) {
        printf("[twopass_matmul selftest] KG=32 BIT-EXACT vs qwen_kal_ref reference\n");
        run_case(1, 896, 896, 1);      /* q */
        run_case(1, 896, 128, 2);      /* k/v */
        run_case(1, 896, 4864, 3);     /* up/gate */
        run_case(1, 4864, 896, 4);     /* down */
        run_case(3, 896, 4864, 5);     /* up M=3 (prefill reserved) */
        run_case(10, 896, 896, 6);     /* q M=10 prefill */
        run_case(3, 4864, 896, 7);     /* down M=3 */
        run_case(1, 32, 512, 8);       /* single block boundary */
        run_case(1, 64, 1, 9);         /* N=1 (min width) */
        printf("[selftest] %s (RC=%d)\n", nfail ? "FAIL" : "PASS", nfail ? 1 : 0);
        return nfail ? 1 : 0;
    }
    if (argc != 4) { fprintf(stderr, "usage: %s --selftest | <M> <K> <N> < stdin\n", argv[0]); return 2; }
    int M = atoi(argv[1]), K = atoi(argv[2]), N = atoi(argv[3]);
    int8_t *x = malloc((size_t)M * K), *w = malloc((size_t)K * N);
    float *gsc = malloc((size_t)(K / G) * N * sizeof(float));
    float *srow = malloc(M * sizeof(float)), *out = malloc((size_t)M * N * sizeof(float));
    if (fread(x, 1, (size_t)M * K, stdin) != (size_t)M * K) return 2;
    if (fread(w, 1, (size_t)K * N, stdin) != (size_t)K * N) return 2;
    if (fread(gsc, 4, (size_t)(K / G) * N, stdin) != (size_t)(K / G) * N) return 2;
    if (fread(srow, 4, M, stdin) != (size_t)M) return 2;
    TwopassCtx c = { M, K, N, x, w, gsc, srow, out };
    twopass_matmul(&c, NULL);
    fwrite(out, 4, (size_t)M * N, stdout);
    return 0;
}
