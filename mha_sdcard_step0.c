/* Measure Step 0 with SD card file I/O.
   Step 0 = read weights/input from SD card + quantize (FP32→INT8).
   This reflects real deployment where weights are stored on SD card.

   Build: make mha_sdcard_step0
   Run:   ./mha_sdcard_step0 [d_model] [n_heads] [seq_len]
*/
#include "common/tpu_bench.h"
#include "common/mha_descriptor.h"
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

static double tick(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

static float compute_scale(const float *data, int n, int *zp_out) {
    float min = data[0], max = data[0];
    for (int i = 1; i < n; i++) {
        if (data[i] < min) min = data[i];
        if (data[i] > max) max = data[i];
    }
    float s = (max - min) / 255.0f;
    if (s < 1e-10f) s = 1.0f;
    *zp_out = (int)(-128.0f - min / s);
    if (*zp_out < -128) *zp_out = -128;
    if (*zp_out > 127)  *zp_out = 127;
    return s;
}

static void quantize_i8(int8_t *dst, const float *src, int n, float scale, int zp) {
    float inv = 1.0f / scale;
    for (int i = 0; i < n; i++) {
        int q = (int)(src[i] * inv + zp + 0.5f);
        if (q > 127) q = 127; else if (q < -128) q = -128;
        dst[i] = (int8_t)q;
    }
}

static int read_file(const char *path, void *buf, int expected_sz) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "  open(%s) failed\n", path); return -1; }
    int n = read(fd, buf, expected_sz);
    close(fd);
    if (n != expected_sz) {
        fprintf(stderr, "  read(%s) got %d, expected %d\n", path, n, expected_sz);
        return -1;
    }
    return 0;
}

static int write_file(const char *path, const void *buf, int sz) {
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) { fprintf(stderr, "  create(%s) failed\n", path); return -1; }
    int n = write(fd, buf, sz);
    close(fd);
    if (n != sz) { fprintf(stderr, "  write(%s) %d != %d\n", path, n, sz); return -1; }
    return 0;
}

int main(int argc, char **argv) {
    setvbuf(stderr, NULL, _IONBF, 0);
    int d_model = argc > 1 ? atoi(argv[1]) : 512;
    int n_heads = argc > 2 ? atoi(argv[2]) : 8;
    int seq_len = argc > 3 ? atoi(argv[3]) : 32;

    int D = d_model, H = n_heads, S = seq_len, d = D / H;
    int total = S * D, w_sz = D * D;
    int input_bytes = total * (int)sizeof(float);
    int weight_bytes = w_sz * (int)sizeof(float);

    fprintf(stderr, "\n===== SD Card Step 0 Benchmark  D=%d H=%d d=%d S=%d =====\n",
            D, H, d, S);
    fprintf(stderr, "  Weights (4x): %d KB each,  Input: %d KB\n",
            weight_bytes / 1024, input_bytes / 1024);

    /* ---- Phase 1: Generate data and write to SD card ---- */
    fprintf(stderr, "\n--- Phase 1: Generate & write to SD card ---\n");
    double t_gen = tick();

    fprintf(stderr, "  malloc weights...\n");
    float *W_f32[4];
    for (int w = 0; w < 4; w++) {
        W_f32[w] = (float *)malloc(weight_bytes);
        if (!W_f32[w]) { fprintf(stderr, "  OOM W_f32[%d]\n", w); return 1; }
        for (int i = 0; i < w_sz; i++)
            W_f32[w][i] = (float)(rand() % 256 - 128) / 1000.0f;
    }
    fprintf(stderr, "  malloc input...\n");
    float *x_f32 = (float *)malloc(input_bytes);
    if (!x_f32) { fprintf(stderr, "  OOM x_f32\n"); return 1; }
    for (int i = 0; i < total; i++)
        x_f32[i] = (float)(rand() % 256 - 128) / 200.0f;

    fprintf(stderr, "  write files...\n");
    double t_wr = tick();
    const char *base = "/tmp/mha_bench";
    mkdir(base, 0755);

    char path[256];
    snprintf(path, sizeof(path), "%s/Wq.f32", base);
    write_file(path, W_f32[0], weight_bytes);
    snprintf(path, sizeof(path), "%s/Wk.f32", base);
    write_file(path, W_f32[1], weight_bytes);
    snprintf(path, sizeof(path), "%s/Wv.f32", base);
    write_file(path, W_f32[2], weight_bytes);
    snprintf(path, sizeof(path), "%s/Wo.f32", base);
    write_file(path, W_f32[3], weight_bytes);
    snprintf(path, sizeof(path), "%s/input.f32", base);
    write_file(path, x_f32, input_bytes);

    double t_wr_done = tick();
    fprintf(stderr, "  Generate: %.1f us,  Write to SD: %.1f us\n",
            t_wr - t_gen, t_wr_done - t_wr);

    /* Free write buffers after files are on disk to save RAM */
    for (int w = 0; w < 4; w++) free(W_f32[w]);
    free(x_f32);

    fprintf(stderr, "  sync...\n");
    sync();

    /* ---- Phase 2: Read from SD card (warm, as in real deployment) ---- */
    fprintf(stderr, "\n--- Phase 2: Read from SD card (realistic warm path) ---\n");

    /* 2a: Read & quantize weights one at a time to minimize RAM */
    fprintf(stderr, "  reading & quantizing weights from SD (one at a time)...\n");
    int8_t *W_i8[4];
    int wp_zp[4]; float wp_sc[4];
    double t_read_weights = 0, t_quant_weights = 0;

    double t_rw_start = tick();
    for (int w = 0; w < 4; w++) {
        float *wbuf = (float *)malloc(weight_bytes);
        if (!wbuf) { fprintf(stderr, "  OOM wbuf[%d]\n", w); return 1; }
        W_i8[w] = (int8_t *)malloc(w_sz);
        if (!W_i8[w]) { fprintf(stderr, "  OOM W_i8[%d]\n", w); return 1; }

        snprintf(path, sizeof(path), "%s/W%c.f32", base, "qkvo"[w]);
        double t1 = tick();
        read_file(path, wbuf, weight_bytes);
        double t2 = tick();
        t_read_weights += (t2 - t1);

        wp_sc[w] = compute_scale(wbuf, w_sz, &wp_zp[w]);
        quantize_i8(W_i8[w], wbuf, w_sz, wp_sc[w], wp_zp[w]);
        double t3 = tick();
        t_quant_weights += (t3 - t2);

        free(wbuf);
        fprintf(stderr, "    W%c: read %.0f us, quant %.0f us\n",
                "qkvo"[w], t2 - t1, t3 - t2);
    }
    double t_qw_done = tick();

    fprintf(stderr, "  weights read+quant OK\n");

    /* 2c: Read and quantize input */
    fprintf(stderr, "  reading input from SD...\n");
    float *x_rd = (float *)malloc(input_bytes);
    int8_t *x_i8 = (int8_t *)malloc(total);
    if (!x_rd || !x_i8) { fprintf(stderr, "  OOM input buf\n"); return 1; }

    snprintf(path, sizeof(path), "%s/input.f32", base);
    double t_ri_start = tick();
    if (read_file(path, x_rd, input_bytes) != 0) return 1;
    double t_ri_done = tick();
    double t_read_input = t_ri_done - t_ri_start;

    int x_zp; float x_sc = compute_scale(x_rd, total, &x_zp);
    quantize_i8(x_i8, x_rd, total, x_sc, x_zp);
    double t_qi_done = tick();
    double t_quant_input = t_qi_done - t_ri_done;

    fprintf(stderr, "  Read  input   (%d KB):     %8.1f us (%.1f MB/s)\n",
            input_bytes / 1024, t_read_input,
            (double)input_bytes / t_read_input);
    fprintf(stderr, "  Quant input:               %8.1f us\n", t_quant_input);

    double t_step0_total = t_qi_done - t_rw_start;
    fprintf(stderr, "\n  Step 0 TOTAL (read+quant): %8.1f us\n", t_step0_total);
    fprintf(stderr, "    = read_weights(%.0f) + quant_weights(%.0f)"
            " + read_input(%.0f) + quant_input(%.0f)\n",
            t_read_weights, t_quant_weights,
            t_read_input, t_quant_input);

    /* ---- Phase 3: Multi-batch simulation ---- */
    int n_batches = 6;
    fprintf(stderr, "\n--- Phase 3: %d-batch Step 0 cost projection ---\n", n_batches);

    /* Generate 5 more input files */
    fprintf(stderr, "  generating %d more inputs...\n", n_batches - 1);
    for (int b = 1; b < n_batches; b++) {
        float *tmp = (float *)malloc(input_bytes);
        if (!tmp) { fprintf(stderr, "  OOM batch %d\n", b); return 1; }
        for (int i = 0; i < total; i++)
            tmp[i] = (float)(rand() % 256 - 128) / 200.0f;
        snprintf(path, sizeof(path), "%s/input_b%d.f32", base, b);
        write_file(path, tmp, input_bytes);
        free(tmp);
    }
    sync();
    fprintf(stderr, "  done\n");

    /* Measure per-batch read + quantize */
    double t_br[6], t_bq[6];
    for (int b = 0; b < n_batches; b++) {
        float *rbuf = (float *)malloc(input_bytes);
        int8_t *qbuf = (int8_t *)malloc(total);
        if (!rbuf || !qbuf) { fprintf(stderr, "  OOM read batch %d\n", b); return 1; }

        double t0 = tick();
        if (b == 0)
            snprintf(path, sizeof(path), "%s/input.f32", base);
        else
            snprintf(path, sizeof(path), "%s/input_b%d.f32", base, b);
        if (read_file(path, rbuf, input_bytes) != 0) {
            fprintf(stderr, "  read_file failed batch %d\n", b); return 1;
        }
        double t1 = tick();

        int zp; float sc = compute_scale(rbuf, total, &zp);
        quantize_i8(qbuf, rbuf, total, sc, zp);
        double t2 = tick();

        t_br[b] = t1 - t0;
        t_bq[b] = t2 - t1;

        free(rbuf); free(qbuf);
    }

    fprintf(stderr, "\n  %-6s %12s %12s %12s\n",
            "Batch", "Read(us)", "Quant(us)", "Step0(us)");
    double sum_read = 0, sum_quant = 0;
    for (int b = 0; b < n_batches; b++) {
        sum_read += t_br[b]; sum_quant += t_bq[b];
        fprintf(stderr, "  %4d   %12.1f %12.1f %12.1f\n",
                b, t_br[b], t_bq[b], t_br[b] + t_bq[b]);
    }
    fprintf(stderr, "  %-6s %12.1f %12.1f %12.1f\n",
            "AVG", sum_read / n_batches, sum_quant / n_batches,
            (sum_read + sum_quant) / n_batches);

    /* ---- Summary ---- */
    fprintf(stderr, "\n===== Step 0 Cost Breakdown =====\n");
    fprintf(stderr, "  %-24s %12s\n", "Component", "Time(us)");
    fprintf(stderr, "  %-24s %12.1f\n", "Read weights (4x, once)", t_read_weights);
    fprintf(stderr, "  %-24s %12.1f\n", "Quant weights (4x, once)", t_quant_weights);
    fprintf(stderr, "  %-24s %12.1f\n", "Read input (per batch)",
            sum_read / n_batches);
    fprintf(stderr, "  %-24s %12.1f\n", "Quant input (per batch)",
            sum_quant / n_batches);
    fprintf(stderr, "  %-24s %12.1f\n", "Step 0 / batch",
            (sum_read + sum_quant) / n_batches);

    double step0_per_batch = (sum_read + sum_quant) / n_batches;
    fprintf(stderr, "\n  Weight read (once): %.1f ms  → amortized over %d batches: %.1f us/batch\n",
            t_read_weights / 1000.0, n_batches, t_read_weights / n_batches);

    double mbox_cost = 4000.0;
    fprintf(stderr, "\n  Dual-core offload potential:\n");
    fprintf(stderr, "    Step 0 cost/batch:            %.1f us (SD read + quant)\n",
            step0_per_batch);
    fprintf(stderr, "    Small-core CAN do:             quantize (CMD_MHA_QUANTIZE)\n");
    fprintf(stderr, "    Small-core CANNOT do:          SD card file read (no FS)\n");
    fprintf(stderr, "    If SD read done by main core, quant offloaded to small:\n");
    fprintf(stderr, "      Main-core SD read:           %.0f us\n",
            sum_read / n_batches);
    fprintf(stderr, "      Small-core quant (mailbox):  %.0f us\n", mbox_cost);
    fprintf(stderr, "      Total:                       %.0f us\n",
            sum_read / n_batches + mbox_cost);
    fprintf(stderr, "    vs Main-core does everything:  %.0f us\n",
            step0_per_batch);
    if (mbox_cost > sum_quant / n_batches) {
        fprintf(stderr, "    VERDICT: mailbox(%.0f) > quant_savings(%.0f) → quant offload = NET LOSS\n",
                mbox_cost, sum_quant / n_batches);
    }
    fprintf(stderr, "    The SD read IS the bottleneck, but small core cannot help with it.\n");
    fprintf(stderr, "    Dual-core helps ONLY if: small-core work > mailbox overhead (~%.0f us)\n",
            mbox_cost);

    /* Cleanup (W_f32, x_f32, and wbuf already freed) */
    for (int w = 0; w < 4; w++) free(W_i8[w]);
    free(x_rd); free(x_i8);

    fprintf(stderr, "\nDone.\n");
    return 0;
}
