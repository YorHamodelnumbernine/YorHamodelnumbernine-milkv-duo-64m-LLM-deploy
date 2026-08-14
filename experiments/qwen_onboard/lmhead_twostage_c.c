/* Phase 6 · LM head 两段式 host C 验证 (MDP C=1024, Kc 参数化).
 *
 * 与 Python 标定 (lmhead_cluster_build.py) 完全同语义:
 *   Stage1: score[c] = h·centroid[c] (centroid fp16 -> double 累加), top-Kc 簇
 *   Stage2: 读 top-Kc 簇 span (embed_i8_cl + embed_scales_cl), 精确 logits
 *           s = Σ h[j]*er[j]*esc[t] (double), 经 row_to_tok 映射回原 token
 * 门禁: P0 NEXT 3/3 + min gap >= 0.05.
 *
 * 用法: lmhead_twostage_c <weights_dir> <Kc>
 * 产物读取: embed_i8_cl.bin / embed_scales_cl.f32 / row_to_tok_cl.bin /
 *          centroid_f16.bin / clust_idx.bin / lmhead_h_cache.bin + lmhead_h_meta.json
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#define D 896
#define V 151936
#define NPROMPTS 11

static float *rd_f32(const char *path, size_t n, const char *tag) {
    FILE *f = fopen(path, "rb"); if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
    float *p = malloc(n * sizeof(float));
    if (fread(p, sizeof(float), n, f) != n) { fprintf(stderr, "short %s\n", tag); exit(2); }
    fclose(f); return p;
}
static uint16_t *rd_f16(const char *path, size_t n, const char *tag) {
    FILE *f = fopen(path, "rb"); if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
    uint16_t *p = malloc(n * sizeof(uint16_t));
    if (fread(p, sizeof(uint16_t), n, f) != n) { fprintf(stderr, "short %s\n", tag); exit(2); }
    fclose(f); return p;
}
static int8_t *rd_i8(const char *path, size_t n, const char *tag) {
    FILE *f = fopen(path, "rb"); if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
    int8_t *p = malloc(n);
    if (fread(p, 1, n, f) != n) { fprintf(stderr, "short %s\n", tag); exit(2); }
    fclose(f); return p;
}
static int32_t *rd_i32(const char *path, size_t n, const char *tag) {
    FILE *f = fopen(path, "rb"); if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
    int32_t *p = malloc(n * sizeof(int32_t));
    if (fread(p, sizeof(int32_t), n, f) != n) { fprintf(stderr, "short %s\n", tag); exit(2); }
    fclose(f); return p;
}

static float f16_to_f32(uint16_t h) {
    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
    uint32_t bits;
    if (e == 0) { if (m == 0) bits = s << 31; else { int k = 0; uint32_t mm = m; while (!(mm & 1024)) { mm <<= 1; k++; } bits = (s << 31) | ((127 - 15 - k) << 23) | ((mm & 1023) << 13); } }
    else if (e == 31) bits = (s << 31) | 0x7f800000u | (m << 13);
    else bits = (s << 31) | ((e + 112) << 23) | (m << 13);
    float out; memcpy(&out, &bits, 4); return out;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <weights_dir> <Kc>\n", argv[0]); exit(2); }
    char wd[512]; snprintf(wd, sizeof wd, "%s", argv[1]);
    int Kc = atoi(argv[2]);
    int C = 1024;

    char p[512];
    /* h 缓存在本目录, 权重在 weights_dir */
    float *h_all = rd_f32("lmhead_h_cache.bin", NPROMPTS * D, "h cache");
    snprintf(p, sizeof p, "%s/centroid_f16.bin", wd); uint16_t *cent = rd_f16(p, (size_t)C * D, "centroid");
    snprintf(p, sizeof p, "%s/clust_idx.bin", wd);   int32_t *clidx = rd_i32(p, (size_t)C * 2, "clust_idx");
    snprintf(p, sizeof p, "%s/embed_i8_cl.bin", wd); int8_t  *emb = rd_i8(p, (size_t)V * D, "embed_cl");
    snprintf(p, sizeof p, "%s/embed_scales_cl.f32", wd); float *esc = rd_f32(p, V, "esc_cl");
    snprintf(p, sizeof p, "%s/row_to_tok_cl.bin", wd); int32_t *tok = rd_i32(p, V, "row_to_tok");

    /* 期望 NEXT 数组 */
    int expect[NPROMPTS] = {2130, 12095, 99366, 7407, 6825, 1304, 362, 2130, 3837, 100152, 2130};
    const char *nm[NPROMPTS] = {
        "P1 中国的首都是", "P2 The capital of France is", "P3 今天天气很好，我们去公园",
        "The capital of Japan is", "Python is a programming language used for",
        "The largest ocean on Earth is the", "Who invented the telephone?",
        "人工智能的未来发展方向是", "上海是中国最大的城市", "量子计算的基本原理是", "光合作用的过程是"};

    int all_hit = 0, p0_hit = 0; double p0_gap_min = 1e300;

    for (int pi = 0; pi < NPROMPTS; pi++) {
        const float *h = h_all + (size_t)pi * D;
        /* ---- Stage1: score[c] = h·centroid[c], top-Kc ---- */
        static double sc[1024];
        for (int c = 0; c < C; c++) {
            const uint16_t *cd = cent + (size_t)c * D;
            double s = 0;
            for (int j = 0; j < D; j++) s += (double)h[j] * (double)f16_to_f32(cd[j]);
            sc[c] = s;
        }
        int sel[1024];
        for (int i = 0; i < Kc; i++) sel[i] = -1;
        /* 简单选择: 取 top-Kc */
        for (int i = 0; i < Kc; i++) {
            int best = -1; for (int c = 0; c < C; c++) if (sc[c] > (best < 0 ? -1e300 : sc[best])) best = c;
            sel[i] = best; sc[best] = -1e300;
        }
        /* ---- Stage2: 精确 logits over 候选簇 span, 映射回 token ----
         * 关键性质: cluster-major 重排是排列, 每 reordered 行唯一对应一个 token,
         * 候选行互不重复 -> 直接对候选行维护运行态 top-5 (无需 V 大小稠密数组). */
        int top1 = -1, top2 = -1; double tv1 = -1e300, tv2 = -1e300;
        size_t cand_rows = 0;
        for (int i = 0; i < Kc; i++) {
            int c = sel[i], o = clidx[c * 2 + 0], cnt = clidx[c * 2 + 1];
            if (cnt <= 0) continue;
            for (int r = 0; r < cnt; r++) {
                const int8_t *er = emb + (size_t)(o + r) * D;
                double s = 0;
                for (int j = 0; j < D; j++) s += (double)h[j] * (double)er[j] * (double)esc[o + r];
                int t = tok[o + r];
                cand_rows++;
                if (s > tv1) { tv2 = tv1; top2 = top1; tv1 = s; top1 = t; }
                else if (s > tv2) { tv2 = s; top2 = t; }
            }
        }
        double gap = tv1 - tv2;
        int hit = (top1 == expect[pi]);
        all_hit += hit;
        if (pi < 3) { p0_hit += hit; if (gap < p0_gap_min) p0_gap_min = gap; }
        printf("[%2d] %-36s g1=%7d top1=%7d hit=%d cand_rows=%zu gap=%.4f\n",
               pi, nm[pi], expect[pi], top1, hit, cand_rows, gap);
    }
    printf("\nrecall_all=%d/%d  P0=%d/3  P0_min_cand_gap=%.4f\n", all_hit, NPROMPTS, p0_hit, p0_gap_min);
    int ok = (all_hit == NPROMPTS && p0_hit == 3 && p0_gap_min >= 0.05);
    printf("判定: %s\n", ok ? "PASS" : "** FAIL **");
    return ok ? 0 : 1;
}
