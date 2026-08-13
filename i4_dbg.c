/* i4_dbg.c — pinpoint scalar vs RVV unpack mismatch on device. */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "int4_common.h"

#define G 64
#define D 576

int main(int argc, char **argv) {
    /* Use real layer0 Wq (n=331776) from disk */
    const char *dir = (argc > 1) ? argv[1] : "/root/smollm2_instruct";
    char path[256];
    snprintf(path, sizeof(path), "%s/layer0_i4.bin", dir);
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return 1; }

    int n = D * D;  /* Wq */
    int nib = n/2, sc = (n/G)*2;
    uint8_t *nibb = (uint8_t*)malloc(nib);
    uint8_t *scp  = (uint8_t*)malloc(sc);
    int8_t  *ref  = (int8_t*)malloc(n);
    int8_t  *out  = (int8_t*)malloc(n);
    uint8_t *rms  = (uint8_t*)malloc(D*4);

    fread(rms, 1, D*4, f);          /* skip rms_attn */
    fread(nibb, 1, nib, f);
    fread(scp, 1, sc, f);
    fclose(f);

    int16_t *scf = (int16_t*)malloc(sc);
    for (int g = 0; g < n/G; g++) {
        uint16_t h; memcpy(&h, scp + (size_t)g*2, 2);
        scf[g] = (int16_t)lrintf(fp16_to_float(h) * 256.0f);
    }

    int4_unpack_fixed(nibb, scf, n, G, ref);
    int4_unpack_fixed_rvv(nibb, scf, n, G, out);

    int nerr = 0;
    printf("group | idx | nibble | q(scl) | q(rvv) | sf | ref | rvv\n");
    for (int i = 0; i < n; i++) {
        if (ref[i] != out[i]) {
            int g = i / G;
            int idx = i % G;
            uint8_t b = nibb[g*G/2 + idx/2];
            int qsc = (idx & 1) ? (int)((b >> 4) & 0xF) : (int)(b & 0xF);
            if (qsc >= 8) qsc -= 16;
            int qrv = (idx & 1) ? (int)((int8_t)(b << 4) >> 4) : 0; /* not used */
            printf("%5d | %3d | 0x%02x | %3d | %3d | %5d | %4d | %4d\n",
                   g, idx, b, qsc, (idx&1)? qsc : qsc, scf[g], ref[i], out[i]);
            if (++nerr >= 20) break;
        }
    }
    printf("total mismatches: %d / %d\n", nerr, n);
    free(nibb); free(scp); free(ref); free(out); free(scf); free(rms);
    return 0;
}
