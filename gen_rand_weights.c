/* gen_rand_weights.c — Generate random INT8 weight files for SmolLM2-135M testing.
   Writes large files in chunks to avoid OOM on 28MB Duo.
   Build: make gen_rand_weights
   Run on Duo: ./gen_rand_weights [--out /data/smollm2]  */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <math.h>

/* SmolLM2-135M config */
#define D_MODEL   576
#define N_HEADS   9
#define N_KV_HEADS 3
#define HEAD_DIM  64
#define N_LAYERS  30
#define FFN_HIDDEN 1536
#define VOCAB_SIZE 49152
#define MAX_SEQ   64
#define DKV       (N_KV_HEADS * HEAD_DIM)  /* 192 */

#define CHUNK    65536   /* 64KB write buffer to stay safe on 28MB Duo */

static int write_file(const char *path, const void *data, int sz) {
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) { perror(path); return -1; }
    if (write(fd, data, sz) != sz) { perror("write"); close(fd); return -1; }
    close(fd);
    return 0;
}

/* Write INT8 matrix row-by-row to an open file descriptor */
static int write_i8_to_fd(int fd, int rows, int cols) {
    int8_t *row = (int8_t *)malloc(cols);
    if (!row) { perror("malloc row"); return -1; }
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) row[c] = (int8_t)((rand() % 255) - 128);
        if (write(fd, row, cols) != cols) { perror("write row"); free(row); return -1; }
    }
    free(row);
    return 0;
}

/* Write a large INT8 matrix row-by-row to avoid OOM */
static int write_i8_chunked(const char *path, int rows, int cols) {
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) { perror(path); return -1; }
    int8_t *row = (int8_t *)malloc(cols);
    if (!row) { perror("malloc row"); close(fd); return -1; }
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) row[c] = (int8_t)((rand() % 255) - 128);
        if (write(fd, row, cols) != cols) { perror("write row"); free(row); close(fd); return -1; }
        if (r % 4096 == 0) printf("    row %d/%d\r", r, rows);
    }
    free(row); close(fd);
    printf("    row %d/%d done\n", rows, rows);
    return 0;
}

/* Allocate and fill if small enough, otherwise use chunked */
static int write_i8_smart(const char *path, int rows, int cols) {
    int sz = rows * cols;
    if (sz <= CHUNK * 2) {
        int8_t *buf = (int8_t *)malloc(sz);
        if (!buf) goto fallback;
        for (int i = 0; i < sz; i++) buf[i] = (int8_t)((rand() % 255) - 128);
        int ret = write_file(path, buf, sz);
        free(buf);
        return ret;
    }
fallback:
    return write_i8_chunked(path, rows, cols);
}

int main(int argc, char **argv) {
    const char *out = "/data/smollm2";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--out") && i+1 < argc) out = argv[++i];
    }

    mkdir(out, 0755);
    char path[512];
    int ret = 0;

    srand(42);  /* fixed seed for reproducibility */

    printf("Generating random SmolLM2-135M weights in %s/\n", out);
    printf("Memory: chunked write for large files (> %d KB)\n", CHUNK/1024);

    /* Config */
    int cfg[8] = {D_MODEL, N_HEADS, N_KV_HEADS, HEAD_DIM, N_LAYERS,
                  FFN_HIDDEN, VOCAB_SIZE, MAX_SEQ};
    snprintf(path, sizeof(path), "%s/config.bin", out);
    ret |= write_file(path, cfg, sizeof(cfg));

    /* Generate random scales (1 + 30*7 + 1 = 212) */
    float scales[212];
    for (int i = 0; i < 212; i++) scales[i] = (float)(rand() % 10000) / 500000.0f + 0.001f;
    snprintf(path, sizeof(path), "%s/scales.bin", out);
    ret |= write_file(path, scales, sizeof(scales));

    /* Embedding: [V, D] INT8 = 28.3 MB — must use chunked */
    printf("  embed.i8 (%d x %d = %.1f MB)...\n", VOCAB_SIZE, D_MODEL,
           (double)VOCAB_SIZE*D_MODEL/1024/1024);
    snprintf(path, sizeof(path), "%s/embed.i8", out);
    ret |= write_i8_chunked(path, VOCAB_SIZE, D_MODEL);

    /* Per-layer weights — all 9 tensors merged into single layerN.bin */
    for (int l = 0; l < N_LAYERS; l++) {
        snprintf(path, sizeof(path), "%s/layer%d.bin", out, l);
        int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) { perror(path); ret |= 1; continue; }

        float rms[D_MODEL];
        for (int i = 0; i < D_MODEL; i++) rms[i] = 1.0f;

        /* Layout: rms_attn | Wq | Wk | Wv | Wo | rms_ffn | ffn_up | ffn_gate | ffn_down */
        write(fd, rms, sizeof(rms));                    /* rms_attn: D*4 */
        ret |= write_i8_to_fd(fd, D_MODEL, D_MODEL);    /* Wq: D*D */
        ret |= write_i8_to_fd(fd, D_MODEL, DKV);        /* Wk: D*dkv */
        ret |= write_i8_to_fd(fd, D_MODEL, DKV);        /* Wv: D*dkv */
        ret |= write_i8_to_fd(fd, D_MODEL, D_MODEL);    /* Wo: D*D */
        write(fd, rms, sizeof(rms));                    /* rms_ffn: D*4 */
        ret |= write_i8_to_fd(fd, D_MODEL, FFN_HIDDEN); /* ffn_up: D*F */
        ret |= write_i8_to_fd(fd, D_MODEL, FFN_HIDDEN); /* ffn_gate: D*F */
        ret |= write_i8_to_fd(fd, FFN_HIDDEN, D_MODEL); /* ffn_down: F*D */

        close(fd);
        if (l % 5 == 0) printf("  layer %d/%d done\n", l, N_LAYERS);
    }

    /* Final RMS */
    {
        float rms[D_MODEL];
        for (int i = 0; i < D_MODEL; i++) rms[i] = 1.0f;
        snprintf(path, sizeof(path), "%s/final_rms.f32", out);
        ret |= write_file(path, rms, sizeof(rms));
    }

    if (ret == 0)
        printf("All weights generated successfully in %s/\n", out);
    else
        printf("Some files failed!\n");

    return ret;
}
