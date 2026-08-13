/* convert_i4.c — host tool: layerN.bin (INT8) -> INT4 weights
 *
 * Two modes:
 *   default : layerN_i4.bin    — all 7 int8 matrices -> INT4 (G=64 fp16)
 *   -f      : layerN_ffni4.bin — FFN-only: attn 4 mats + rms f32 pass through
 *                                INT8 unchanged; only ffn_up/gate/down -> INT4.
 *                                (CEO Direction-2: save ~35% bytes, keep attn exact.)
 *
 * INT8 layer layout (matches merge_layers.c):
 *   [ rms_attn f32 : D*4 ]
 *   [ Wq i8 : D*D ]  [ Wk i8 : D*dkv ]  [ Wv i8 : D*dkv ]  [ Wo i8 : D*D ]
 *   [ rms_ffn f32 : D*4 ]
 *   [ ffn_up i8 : D*F ]  [ ffn_gate i8 : D*F ]  [ ffn_down i8 : F*D ]
 *
 * INT4 output format (DESIGN_INT4_WEIGHTS.md):
 *   - rms f32 blocks passed through unchanged
 *   - each int8 matrix -> packed_nibbles (n/2 B) + scales ((n/G)*2 B, fp16)
 *   - G=64, fp16 scales (default): 7 matrices packed = 1,880,064 B; layer total = 1,884,672 B
 *   - FFN-only: attn 4 mats (884,736 B) + rms (4,608 B) passthrough + FFN i4 (1,410,048 B)
 *               = 2,299,392 B/layer (-35%)
 *
 * Build (host): gcc -O3 -o convert_i4 convert_i4.c int4_common.c -lm
 * Run:          ./convert_i4 <layerN.bin> [out.bin] [-f]
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "int4_common.h"

#define D    576
#define DKVD 192
#define FFN  1536
#define G    64
#define ST   1            /* scale type: 1 = fp16 */

/* matrix geometry in layer order (after rms_attn): rows x cols */
static const int mat_rows[] = { D, D, D, D, D, D, FFN };
static const int mat_cols[] = { D, DKVD, DKVD, D, FFN, FFN, D };
static const int N_MAT = 7;
static const int N_ATTN = 4;   /* Wq Wk Wv Wo */

static long long read_full(FILE *f, void *buf, size_t n) {
    return (long long)fread(buf, 1, n, f);
}

int main(int argc, char **argv) {
    int ffn_only = 0;
    int Gp = G;                 /* group size, default 64; -g N overrides */
    const char *inp = NULL;
    char outbuf[256] = {0};
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-f")) ffn_only = 1;
        else if (!strcmp(argv[i], "-g") && i + 1 < argc) Gp = atoi(argv[++i]);
        else if (!inp) inp = argv[i];
        else if (!outbuf[0]) snprintf(outbuf, sizeof(outbuf), "%s", argv[i]);
        else { fprintf(stderr, "extra arg: %s\n", argv[i]); return 1; }
    }
    if (!inp) { fprintf(stderr, "Usage: %s <layerN.bin> [out.bin] [-f] [-g N]\n", argv[0]); return 1; }
    if (Gp < 8 || Gp > 256 || (3538944 % Gp)) { fprintf(stderr, "bad G=%d\n", Gp); return 1; }
    if (!outbuf[0]) {
        snprintf(outbuf, sizeof(outbuf), "%s", inp);
        char *dot = strrchr(outbuf, '.');
        if (dot && !strchr(dot, '/')) snprintf(dot, 12, "%s.bin", ffn_only ? "_ffni4" : "_i4");
        else strncat(outbuf, ffn_only ? "_ffni4.bin" : "_i4.bin", sizeof(outbuf)-strlen(outbuf)-1);
    }

    FILE *fin = fopen(inp, "rb");
    if (!fin) { perror(inp); return 1; }
    FILE *fout = fopen(outbuf, "wb");
    if (!fout) { perror(outbuf); fclose(fin); return 1; }

    uint8_t *rms = (uint8_t*)malloc(D*4);       /* 2304 B, passthrough */
    long long packed_bytes = 0, total_in = 0;

    /* Source INT8 layer layout (must match merge_layers.c / sm_setup_ptrs):
     *   [rms_attn f32][Wq][Wk][Wv][Wo][rms_ffn f32][ffn_up][ffn_gate][ffn_down]
     * Output layout is IDENTICAL (engine expects it that way).  We therefore
     * read/write in this exact order: rms_attn, attn mats, rms_ffn, ffn mats.
     * (Earlier bug: 7 mats were read contiguously after rms_attn, which
     *  shifted all FFN mats by the embedded 2304-byte rms_ffn.) */
    if (read_full(fin, rms, D*4) != D*4) { fprintf(stderr, "short read rms_attn\n"); return 1; }
    fwrite(rms, 1, D*4, fout); total_in += D*4;

    /* attn mats 0..3: Wq Wk Wv Wo (pass through INT8 in FFN-only mode) */
    for (int m = 0; m < N_ATTN; m++) {
        int n = mat_rows[m] * mat_cols[m];
        int8_t *src = (int8_t*)malloc(n);
        if (read_full(fin, src, n) != n) { fprintf(stderr, "short read attn mat %d\n", m); return 1; }
        total_in += n;

        if (ffn_only) {
            if (fwrite(src, 1, n, fout) != (size_t)n) { fprintf(stderr, "write fail\n"); return 1; }
            printf("  mat %d %-8s PASS-THROUGH INT8 (%dx%d, n=%d, %d B)\n",
                   m, "(attn)", mat_rows[m], mat_cols[m], n, n);
        } else {
            size_t nib_sz = n/2, sc_sz = (size_t)(n/Gp)*2;
            uint8_t *nib = (uint8_t*)malloc(nib_sz);
            uint8_t *sc  = (uint8_t*)malloc(sc_sz);
            int8_t  *chk = (int8_t*)malloc(n);
            int nsc = int4_pack(src, n, Gp, ST, nib, sc);
            if ((long long)nsc * Gp != n) { fprintf(stderr, "n not multiple of G\n"); return 1; }
            fwrite(nib, 1, nib_sz, fout);
            fwrite(sc,  1, sc_sz,  fout);
            packed_bytes += nib_sz + sc_sz;
            int4_unpack(nib, sc, n, Gp, ST, chk);
            long long maxe = 0; double ss = 0;
            for (int i = 0; i < n; i++) {
                long long e = (long long)src[i] - chk[i]; if (e<0)e=-e; if(e>maxe)maxe=e; ss += (double)e*e;
            }
            printf("  mat %d %-8s PACKED i4   (%dx%d, n=%d): %zu B  max_err=%lld  rms_err=%.3f\n",
                   m, "(attn)", mat_rows[m], mat_cols[m], n, nib_sz+sc_sz, maxe, sqrt(ss/n));
            free(nib); free(sc); free(chk);
        }
        free(src);
    }

    /* rms_ffn f32 — in the MIDDLE (between attn and ffn), matching INT8 layout */
    if (read_full(fin, rms, D*4) != D*4) { fprintf(stderr, "short read rms_ffn\n"); return 1; }
    fwrite(rms, 1, D*4, fout); total_in += D*4;

    /* ffn mats 4..6: up gate down */
    for (int m = N_ATTN; m < N_MAT; m++) {
        int n = mat_rows[m] * mat_cols[m];
        int8_t *src = (int8_t*)malloc(n);
        if (read_full(fin, src, n) != n) { fprintf(stderr, "short read ffn mat %d\n", m); return 1; }
        total_in += n;

        {
            size_t nib_sz = n/2, sc_sz = (size_t)(n/Gp)*2;
            uint8_t *nib = (uint8_t*)malloc(nib_sz);
            uint8_t *sc  = (uint8_t*)malloc(sc_sz);
            int8_t  *chk = (int8_t*)malloc(n);
            int nsc = int4_pack(src, n, Gp, ST, nib, sc);
            if ((long long)nsc * Gp != n) { fprintf(stderr, "n not multiple of G\n"); return 1; }
            fwrite(nib, 1, nib_sz, fout);
            fwrite(sc,  1, sc_sz,  fout);
            packed_bytes += nib_sz + sc_sz;
            int4_unpack(nib, sc, n, Gp, ST, chk);
            long long maxe = 0; double ss = 0;
            for (int i = 0; i < n; i++) {
                long long e = (long long)src[i] - chk[i]; if (e<0)e=-e; if(e>maxe)maxe=e; ss += (double)e*e;
            }
            printf("  mat %d %-8s PACKED i4   (%dx%d, n=%d): %zu B  max_err=%lld  rms_err=%.3f\n",
                   m, "(ffn)", mat_rows[m], mat_cols[m], n, nib_sz+sc_sz, maxe, sqrt(ss/n));
            free(nib); free(sc); free(chk);
        }
        free(src);
    }

    /* verify no trailing data */
    int extra = 0; uint8_t c;
    while (fread(&c, 1, 1, fin) == 1) extra++;

    fclose(fin); fclose(fout); free(rms);

    printf("layer  : %s (%lld B in)\n", inp, total_in);
    printf("out    : %s\n", outbuf);
    if (ffn_only) {
        printf("mode   : FFN-only (attn INT8 + FFN INT4, G=%d)\n", Gp);
        printf("packed : %lld B (3 FFN mats, G=%d fp16)  <-- expect %lld\n",
               packed_bytes, Gp, 3LL*((D*FFN)/2 + ((D*FFN)/Gp)*2));
        printf("total  : %lld B (incl attn 884736 + rms 4608)\n",
               packed_bytes + 884736 + 2*D*4);
    } else {
        printf("mode   : full INT4 (7 mats, G=%d)\n", Gp);
        printf("packed : %lld B (7 matrices, G=%d fp16)\n", packed_bytes, Gp);
        printf("total  : %lld B (incl rms f32)\n", packed_bytes + 2*D*4);
    }
    if (extra) printf("WARN   : %d trailing bytes in source\n", extra);
    return 0;
}
