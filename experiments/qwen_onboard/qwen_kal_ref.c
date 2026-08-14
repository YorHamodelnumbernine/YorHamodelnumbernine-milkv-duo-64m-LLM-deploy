/* qwen_kal_ref.c — host-side C reference for Path A two-pass inference.
 *
 * Reads weights_kal/ (K-aligned INT4 G32 + fp16 group scales) produced by
 * convert_qwen_kal.py and runs the full Qwen2.5-0.5B forward with:
 *   - per-row (per-token) INT8 activation
 *   - per-chunk KG=32 matmul, TWO-PASS rshift (pass1 safe -> read max ->
 *     pass2 refined), TIU semantics emulated on CPU (exact int32 acc then
 *     int8 round), fp32 accumulate with per-(group,col) gscale.
 * This is the numeric reference / CPU-side skeleton for the on-board engine.
 *
 * Usage: qwen_kal_ref <weights_dir> <tok1> <tok2> ... [<prompt_id>]
 * Prints next-token top-5.  Must reproduce Python emulator 3/3 for the 3 P0
 * prompts (tok ids: 105538,59975,100132 / 785,6722,315,9625,374 /
 * 100644,104307,101243,3837,97639,85336,102077).
 *
 * Build (host, x86-64): gcc -O2 -o qwen_kal_ref qwen_kal_ref.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#define D 896
#define H 14
#define KVH 2
#define HD 64
#define L 24
#define F 4864
#define V 151936
#define G 32
#define DKV 128
#define GROUPS 7
#define ROPE_THETA 1000000.0
#define EPS 1e-6f
#define MAX_SEQ 64

/* ---------------- small utils ---------------- */
static float fp16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t man = h & 0x3ff;
    uint32_t f;
    if (exp == 0) {
        if (man == 0) f = sign;
        else { /* subnormal */ exp = 127 - 15 + 1; while (!(man & 0x400)) { man <<= 1; exp--; } man &= 0x3ff; f = sign | (exp << 23) | (man << 13); }
    } else if (exp == 31) {
        f = sign | 0x7f800000u | (man << 13);
    } else {
        f = sign | ((exp + (127 - 15)) << 23) | (man << 13);
    }
    float out; memcpy(&out, &f, 4); return out;
}
static int32_t max_i8(const int8_t *a, size_t n) {
    int32_t m = 0; for (size_t i = 0; i < n; i++) { int32_t v = a[i]; if (v < 0) v = -v; if (v > m) m = v; } return m;
}
/* minimal r with K*127*wmax >> r <= 127 */
static int matmul_rshift_w(int K, int wmax) {
    int r = 0; long long md = (long long)K * 127 * wmax;
    while ((md >> r) > 127) r++;
    return r;
}
/* TIU-style: round(acc / 2^rshift), clamp int8.
 * Confirmed on-board (commit 23f300e / GATE_A_SIGNOFF_20260813.md):
 * round-half-up toward +inf, formula sat8((acc + 2^(r-1)) >> r).
 * For negative exact half-way ties this rounds toward zero (e.g. -16>>5 -> 0),
 * NOT round-half-away-from-zero.  CPU must match TIU so pass2 matches on-board. */
static inline int8_t int8_round_div(int32_t acc, int rshift) {
    int32_t scale = 1 << rshift;
    int32_t half = scale >> 1;
    int32_t r = (acc + half) >> rshift;   /* arithmetic shift on signed = floor */
    if (r > 127) r = 127;
    if (r < -128) r = -128;
    return (int8_t)r;
}

/* ---------------- weights ---------------- */
typedef struct { int K, N; int8_t *q; float *gsc; } Mat;
typedef struct {
    float rms_attn[D], rms_ffn[D];
    Mat Wq, Wk, Wv, Wo, up, gate, down;
    float bias[D + DKV + DKV];
} Layer;

static void load_f32(FILE *f, float *dst, size_t n) { if (fread(dst, 4, n, f) != n) { fprintf(stderr, "short f32 read\n"); exit(1); } }
static int16_t load_i16(FILE *f) { uint8_t b[2]; if (fread(b, 1, 2, f) != 2) { fprintf(stderr, "short i16\n"); exit(1); } return (int16_t)(b[0] | (b[1] << 8)); }

static void unpack_mat(FILE *f, Mat *m, int K, int N) {
    m->K = K; m->N = N;
    int KG = K / G;
    m->q = malloc((size_t)K * N);
    m->gsc = malloc((size_t)KG * N * sizeof(float));
    if (!m->q || !m->gsc) { fprintf(stderr, "oom mat\n"); exit(1); }
    uint8_t *nib = malloc((size_t)KG * N * G / 2);
    if (fread(nib, 1, (size_t)KG * N * G / 2, f) != (size_t)KG * N * G / 2) { fprintf(stderr, "short nib\n"); exit(1); }
    /* nib[g,n,j] = q[g,2j,n] | q[g,2j+1,n]<<4 ; q -> [K,N] row-major (k-major) */
    for (int g = 0; g < KG; g++) {
        for (int n = 0; n < N; n++) {
            const uint8_t *base = nib + ((size_t)g * N + n) * (G / 2);
            for (int j = 0; j < G / 2; j++) {
                uint8_t b = base[j];
                int lo = b & 0xF, hi = b >> 4;
                lo = lo > 7 ? lo - 16 : lo; hi = hi > 7 ? hi - 16 : hi;
                m->q[(size_t)(g * G + 2 * j) * N + n] = (int8_t)lo;
                m->q[(size_t)(g * G + 2 * j + 1) * N + n] = (int8_t)hi;
            }
        }
    }
    free(nib);
    for (int i = 0; i < KG * N; i++) m->gsc[i] = fp16_to_f32((uint16_t)load_i16(f));
}

static int8_t *g_embed; static float *g_esc, *g_frms;

static Layer *load_layers(const char *dir) {
    char path[512];
    g_embed = malloc((size_t)V * D); g_esc = malloc(V * sizeof(float)); g_frms = malloc(D * sizeof(float));
    snprintf(path, sizeof path, "%s/embed_i8.bin", dir); FILE *f = fopen(path, "rb"); if (!f) { perror(path); exit(1); }
    if (fread(g_embed, 1, (size_t)V * D, f) != (size_t)V * D) { fprintf(stderr, "embed read\n"); exit(1); } fclose(f);
    snprintf(path, sizeof path, "%s/embed_scales.f32", dir); f = fopen(path, "rb"); load_f32(f, g_esc, V); fclose(f);
    snprintf(path, sizeof path, "%s/final_rms.f32", dir); f = fopen(path, "rb"); load_f32(f, g_frms, D); fclose(f);

    Layer *layers = calloc(L, sizeof(Layer));
    for (int l = 0; l < L; l++) {
        snprintf(path, sizeof path, "%s/layer%d_kal.bin", dir, l);
        f = fopen(path, "rb"); if (!f) { perror(path); exit(1); }
        load_f32(f, layers[l].rms_attn, D);
        unpack_mat(f, &layers[l].Wq, D, D);
        unpack_mat(f, &layers[l].Wk, D, DKV);
        unpack_mat(f, &layers[l].Wv, D, DKV);
        unpack_mat(f, &layers[l].Wo, D, D);
        unpack_mat(f, &layers[l].up, D, F);
        unpack_mat(f, &layers[l].gate, D, F);
        unpack_mat(f, &layers[l].down, F, D);
        load_f32(f, layers[l].rms_ffn, D);
        fclose(f);
        snprintf(path, sizeof path, "%s/layer%d_bias.f32", dir, l);
        f = fopen(path, "rb"); load_f32(f, layers[l].bias, D + DKV + DKV); fclose(f);
    }
    return layers;
}

/* ---------------- matmul: per-chunk two-pass ---------------- */
/* x_i8 [M,K], w_i8 [K,N] (raw int4 values -8..7), gsc [K/G,N], sc_row [M]
 * -> out [M,N] fp32 (already * sc_row).  K must be divisible by G. */
static void chunk_matmul_twopass(const int8_t *x_i8, int M, int K,
                                 const int8_t *w_i8, int N, const float *gsc,
                                 const float *sc_row, float *out) {
    int KG = K / G;
    int wmax = max_i8(w_i8, (size_t)K * N);
    int rsafe = matmul_rshift_w(G, wmax) - 3; if (rsafe < 4) rsafe = 4;
    double *accd = calloc((size_t)M * N, sizeof(double));
    int32_t *acc = malloc((size_t)M * N * sizeof(int32_t));
    for (int g = 0; g < KG; g++) {
        const int8_t *xs = x_i8;               /* [M, K], chunk uses cols g*G..g*G+G-1 */
        const int8_t *ws = w_i8 + (size_t)g * G * N;   /* [G, N] */
        /* acc = x[:,g*G:(g+1)*G] @ w[g*G:(g+1)*G,:] */
        for (int m = 0; m < M; m++) {
            const int8_t *xr = xs + (size_t)m * K + (size_t)g * G;
            int32_t *ar = acc + (size_t)m * N;
            for (int n = 0; n < N; n++) {
                int32_t s = 0;
                const int8_t *wc = ws + n;    /* column n of weight chunk */
                for (int k = 0; k < G; k++) s += (int32_t)xr[k] * (int32_t)wc[(size_t)k * N];
                ar[n] = s;
            }
        }
        /* pass1: int8 round with rsafe, find max |p1| */
        int32_t est = 0;
        for (int m = 0; m < M; m++) {
            int32_t *ar = acc + (size_t)m * N;
            for (int n = 0; n < N; n++) {
                int32_t v = int8_round_div(ar[n], rsafe);
                if (v < 0) v = -v;
                if (v > est) est = v;
            }
        }
        est = est * (1 << rsafe);
        int r = 0; while (est > (127 << r)) r++;
        /* pass2 + fp32 accumulate with gscale */
        for (int m = 0; m < M; m++) {
            int32_t *ar = acc + (size_t)m * N;
            double *dr = accd + (size_t)m * N;
            const float *gr = gsc + (size_t)g * N;
            for (int n = 0; n < N; n++) {
                int32_t p2 = int8_round_div(ar[n], r);
                dr[n] += (double)p2 * (double)(1 << r) * (double)gr[n];
            }
        }
    }
    for (int m = 0; m < M; m++) {
        const double *dr = accd + (size_t)m * N;
        float *or_ = out + (size_t)m * N;
        float sc = sc_row[m];
        for (int n = 0; n < N; n++) or_[n] = (float)dr[n] * sc;
    }
    free(accd); free(acc);
}

/* ---------------- layers ---------------- */
static void rms_norm(const float *x, int seq, const float *g, int n, float *out) {
    for (int m = 0; m < seq; m++) {
        const float *xr = x + (size_t)m * n;
        float *or_ = out + (size_t)m * n;
        double ss = 0; for (int i = 0; i < n; i++) ss += (double)xr[i] * xr[i];
        float inv = (float)(1.0 / sqrt(ss / n + EPS));
        for (int i = 0; i < n; i++) or_[i] = xr[i] * inv * g[i];
    }
}
static inline float silu(float x) { return x / (1.0f + expf(-x)); }
/* np.round parity: round-half-to-even (banker's) */
static inline int32_t round_bankers(float v) {
    float f = floorf(v);
    float d = v - f;
    if (d > 0.5f) return (int32_t)(f + 1.0f);
    if (d < 0.5f) return (int32_t)f;
    return ((int32_t)f % 2 == 0) ? (int32_t)f : (int32_t)(f + 1.0f);
}
static void quant_per_row(const float *x, int M, int K, int8_t *q, float *sc) {
    for (int m = 0; m < M; m++) {
        const float *xr = x + (size_t)m * K;
        float mx = 0; for (int k = 0; k < K; k++) { float a = fabsf(xr[k]); if (a > mx) mx = a; }
        float s = mx / 127.0f; if (s < 1e-12f) s = 1e-12f;
        sc[m] = s;
        for (int k = 0; k < K; k++) {
            float v = xr[k] / s;
            int32_t ri = round_bankers(v); if (ri > 127) ri = 127; if (ri < -128) ri = -128;
            q[(size_t)m * K + k] = (int8_t)ri;
        }
    }
}
static void rope_inplace(float *q, int pos, const float *cos, const float *sin) {
    int half = HD / 2;
    for (int i = 0; i < half; i++) {
        float x0 = q[i], x1 = q[half + i];
        float c = cos[pos * half + i], s = sin[pos * half + i];
        q[i] = x0 * c - x1 * s;
        q[half + i] = x0 * s + x1 * c;
    }
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <weights_dir> <tok1> <tok2>...\n", argv[0]); return 2; }
    const char *dir = argv[1];
    int seq = argc - 2;
    int *toks = malloc(seq * sizeof(int));
    for (int i = 0; i < seq; i++) toks[i] = atoi(argv[2 + i]);

    Layer *layers = load_layers(dir);

    /* precompute rope up to seq+8 */
    float *cosb = malloc((size_t)(seq + 8) * HD * sizeof(float));
    float *sinb = malloc((size_t)(seq + 8) * HD * sizeof(float));
    for (int pos = 0; pos < seq + 8; pos++)
        for (int j = 0; j < HD / 2; j++) {
            float freq = powf(ROPE_THETA, -(float)j / (HD / 2));
            cosb[pos * (HD / 2) + j] = cosf(pos * freq);
            sinb[pos * (HD / 2) + j] = sinf(pos * freq);
        }

    /* x [seq, D] = embed[t] * esc[t] */
    float *x = malloc((size_t)seq * D * sizeof(float));
    for (int i = 0; i < seq; i++) {
        int t = (toks[i] < 0 || toks[i] >= V) ? 0 : toks[i];
        const int8_t *er = g_embed + (size_t)t * D;
        float s = g_esc[t];
        for (int k = 0; k < D; k++) x[(size_t)i * D + k] = (float)er[k] * s;
    }

    float *h = malloc((size_t)seq * D * sizeof(float));
    int8_t *xi = malloc((size_t)seq * D);
    float *scr = malloc(seq * sizeof(float));
    float *qbuf = malloc((size_t)seq * D * sizeof(float));
    float *kbuf = malloc((size_t)seq * DKV * sizeof(float));
    float *vbuf = malloc((size_t)seq * DKV * sizeof(float));
    float *attn = malloc((size_t)seq * D * sizeof(float));
    float *mid = malloc((size_t)seq * F * sizeof(float));
    float *oout = malloc((size_t)seq * D * sizeof(float));

    for (int l = 0; l < L; l++) {
        Layer *ly = &layers[l];
        /* qkv */
        rms_norm(x, seq, ly->rms_attn, D, h);
        quant_per_row(h, seq, D, xi, scr);
        chunk_matmul_twopass(xi, seq, D, ly->Wq.q, D, ly->Wq.gsc, scr, qbuf);
        chunk_matmul_twopass(xi, seq, D, ly->Wk.q, DKV, ly->Wk.gsc, scr, kbuf);
        chunk_matmul_twopass(xi, seq, D, ly->Wv.q, DKV, ly->Wv.gsc, scr, vbuf);
        const float *bq = ly->bias, *bk = ly->bias + D, *bv = ly->bias + D + DKV;
        for (int i = 0; i < seq; i++) {
            for (int j = 0; j < D; j++) qbuf[(size_t)i * D + j] += bq[j];
            for (int j = 0; j < DKV; j++) { kbuf[(size_t)i * DKV + j] += bk[j]; vbuf[(size_t)i * DKV + j] += bv[j]; }
        }
        /* rope */
        for (int i = 0; i < seq; i++) {
            for (int hh = 0; hh < H; hh++) rope_inplace(qbuf + (size_t)i * D + (size_t)hh * HD, i, cosb, sinb);
            for (int hh = 0; hh < KVH; hh++) { rope_inplace(kbuf + (size_t)i * DKV + (size_t)hh * HD, i, cosb, sinb); }
        }
        /* attention */
        float *att = attn;
        for (int hh = 0; hh < H; hh++) {
            int kvh = hh / GROUPS;
            for (int m = 0; m < seq; m++) {
                const float *qm = qbuf + (size_t)m * D + (size_t)hh * HD;
                const float *km = kbuf + (size_t)kvh * HD;    /* head kvh, row s at +s*DKV */
                const float *vm = vbuf + (size_t)kvh * HD;
                /* logits[m][s] = qm . km[s] / sqrt(HD) */
                float lgm[MAX_SEQ], mx = -1e30f;
                for (int s = 0; s < seq; s++) {
                    float v = (s > m) ? -1e30f : 0.0f;
                    if (s <= m) {
                        const float *ks = km + (size_t)s * DKV;
                        v = 0; for (int j = 0; j < HD; j++) v += qm[j] * ks[j];
                        v *= 1.0f / sqrtf((float)HD);
                    }
                    lgm[s] = v; if (v > mx) mx = v;
                }
                float sum = 0;
                for (int s = 0; s < seq; s++) { lgm[s] = expf(lgm[s] - mx); sum += lgm[s]; }
                float *attrow = att + (size_t)m * D + (size_t)hh * HD;
                for (int j = 0; j < HD; j++) {
                    float acc = 0;
                    for (int s = 0; s < seq; s++) acc += lgm[s] / sum * vm[(size_t)s * DKV + j];
                    attrow[j] = acc;
                }
            }
        }
        /* wo */
        quant_per_row(attn, seq, D, xi, scr);
        chunk_matmul_twopass(xi, seq, D, ly->Wo.q, D, ly->Wo.gsc, scr, oout);
        for (int i = 0; i < seq; i++) for (int j = 0; j < D; j++) x[(size_t)i * D + j] += oout[(size_t)i * D + j];

        /* ffn */
        rms_norm(x, seq, ly->rms_ffn, D, h);
        quant_per_row(h, seq, D, xi, scr);
        float *upb = malloc((size_t)seq * F * sizeof(float));
        float *gateb = malloc((size_t)seq * F * sizeof(float));
        chunk_matmul_twopass(xi, seq, D, ly->up.q, F, ly->up.gsc, scr, upb);
        chunk_matmul_twopass(xi, seq, D, ly->gate.q, F, ly->gate.gsc, scr, gateb);
        for (int i = 0; i < seq; i++) for (int j = 0; j < F; j++)
            mid[(size_t)i * F + j] = upb[(size_t)i * F + j] * silu(gateb[(size_t)i * F + j]);
        free(upb); free(gateb);
        /* down: K-chunked 1024, per-chunk per-row quant */
        memset(oout, 0, (size_t)seq * D * sizeof(float));
        for (int kc = 0; kc < F; kc += 1024) {
            int kcn = (F - kc < 1024) ? F - kc : 1024;
            /* quant per row of mid[:, kc:kc+kcn] */
            float *mch = malloc((size_t)seq * kcn * sizeof(float));
            int8_t *mch_i8 = malloc((size_t)seq * kcn);
            float *mch_sc = malloc(seq * sizeof(float));
            for (int i = 0; i < seq; i++) memcpy(mch + (size_t)i * kcn, mid + (size_t)i * F + kc, kcn * sizeof(float));
            quant_per_row(mch, seq, kcn, mch_i8, mch_sc);
            const int8_t *wq = ly->down.q + (size_t)kc * D;
            const float *gsc = ly->down.gsc + (size_t)(kc / G) * D;
            float *sub = malloc((size_t)seq * D * sizeof(float));
            chunk_matmul_twopass(mch_i8, seq, kcn, wq, D, gsc, mch_sc, sub);
            for (int i = 0; i < seq; i++) for (int j = 0; j < D; j++) oout[(size_t)i * D + j] += sub[(size_t)i * D + j];
            free(mch); free(mch_i8); free(mch_sc); free(sub);
        }
        for (int i = 0; i < seq; i++) for (int j = 0; j < D; j++) x[(size_t)i * D + j] += oout[(size_t)i * D + j];
        fprintf(stderr, "[ref] layer %d/%d done\n", l + 1, L);
    }

    /* final rms + lm head */
    rms_norm(x + (size_t)(seq - 1) * D, 1, g_frms, D, h);
    double *logits = malloc((size_t)V * sizeof(double));
    for (int t = 0; t < V; t++) {
        const int8_t *er = g_embed + (size_t)t * D;
        double s = 0; for (int j = 0; j < D; j++) s += (double)h[j] * (double)er[j] * (double)g_esc[t];
        logits[t] = s;
    }
    int top[5]; double tv[5];
    for (int i = 0; i < 5; i++) { top[i] = -1; tv[i] = -1e300; }
    for (int t = 0; t < V; t++) {
        for (int i = 0; i < 5; i++) if (logits[t] > tv[i]) {
            for (int j = 4; j > i; j--) { top[j] = top[j - 1]; tv[j] = tv[j - 1]; }
            top[i] = t; tv[i] = logits[t]; break;
        }
    }
    printf("NEXT_TOKEN: %d\n", top[0]);
    printf("TOP5: ");
    for (int i = 0; i < 5; i++) printf("%d ", top[i]);
    printf("\n");
    printf("GAP: %.4f\n", (double)(tv[0] - tv[1]));
    for (int i = 0; i < 5; i++) printf("TOPVAL[%d]=%.4f\n", top[i], tv[i]);
    free(logits); return 0;
}
