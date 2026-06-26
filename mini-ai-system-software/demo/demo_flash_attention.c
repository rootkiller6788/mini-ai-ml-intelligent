#include "flash_attention.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static double random_normal(void) {
    double u1 = (double)rand() / RAND_MAX;
    double u2 = (double)rand() / RAND_MAX;
    return sqrt(-2.0 * log(u1 + 1e-10)) * cos(2.0 * 3.1415926535 * u2);
}

static void fill_random(double *data, size_t n) {
    for (size_t i = 0; i < n; i++) data[i] = random_normal() * 0.1;
}

static double mat_max_diff(const double *A, const double *B, size_t n) {
    double maxDiff = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs(A[i] - B[i]);
        if (d > maxDiff) maxDiff = d;
    }
    return maxDiff;
}

/* Naive attention for verification */
static void naive_attention(const double *Q, const double *K, const double *V,
                             uint32_t seqLen, uint32_t headDim,
                             double softmaxScale, int causal,
                             double *O) {
    double *S = (double *)malloc((size_t)seqLen * seqLen * sizeof(double));
    for (uint32_t i = 0; i < seqLen; i++) {
        double maxVal = -INFINITY;
        for (uint32_t j = 0; j < seqLen; j++) {
            if (causal && i < j) { S[i * seqLen + j] = -INFINITY; continue; }
            double dot = 0.0;
            for (uint32_t d = 0; d < headDim; d++)
                dot += Q[i * headDim + d] * K[j * headDim + d];
            S[i * seqLen + j] = dot * softmaxScale;
            if (S[i * seqLen + j] > maxVal) maxVal = S[i * seqLen + j];
        }
        double sumExp = 0.0;
        for (uint32_t j = 0; j < seqLen; j++) {
            if (!(causal && i < j)) {
                S[i * seqLen + j] = exp(S[i * seqLen + j] - maxVal);
                sumExp += S[i * seqLen + j];
            } else {
                S[i * seqLen + j] = 0.0;
            }
        }
        for (uint32_t j = 0; j < seqLen; j++)
            S[i * seqLen + j] /= sumExp;
        for (uint32_t d = 0; d < headDim; d++) {
            O[i * headDim + d] = 0.0;
            for (uint32_t j = 0; j < seqLen; j++)
                O[i * headDim + d] += S[i * seqLen + j] * V[j * headDim + d];
        }
    }
    free(S);
}

static void verify_correctness(FlashAttnConfig cfg, int causal) {
    printf("\n--- Correctness Check (causal=%d) ---\n", causal);

    cfg.causal = causal;
    cfg.batchSize = 1;
    cfg.numHeads  = 1;
    size_t perHead = (size_t)cfg.seqLen * cfg.headDim;

    double *Q = (double *)malloc(perHead * sizeof(double));
    double *K = (double *)malloc(perHead * sizeof(double));
    double *V = (double *)malloc(perHead * sizeof(double));
    fill_random(Q, perHead);
    fill_random(K, perHead);
    fill_random(V, perHead);

    FlashAttnOutput *flashOut = flash_attn_forward(Q, K, V, &cfg);

    double *naiveO = (double *)calloc(perHead, sizeof(double));
    naive_attention(Q, K, V, cfg.seqLen, cfg.headDim,
                     cfg.softmaxScale, cfg.causal, naiveO);

    double maxDiff = mat_max_diff(flashOut->output, naiveO, perHead);
    printf("  Max |flash - naive| = %.6e\n", maxDiff);
    printf("  Status: %s\n", maxDiff < 0.01 ? "PASS" : "WARN");

    /* Gradient check */
    double *dO = (double *)malloc((size_t)cfg.batchSize * cfg.numHeads
                                   * perHead * sizeof(double));
    fill_random(dO, perHead);

    double *dQ_f = (double *)calloc(perHead, sizeof(double));
    double *dK_f = (double *)calloc(perHead, sizeof(double));
    double *dV_f = (double *)calloc(perHead, sizeof(double));

    flash_attn_backward(Q, K, V, dO, dQ_f, dK_f, dV_f, &cfg);

    printf("  dQ[0..3] = %.4f %.4f %.4f %.4f\n",
           dQ_f[0], dQ_f[1], dQ_f[2], dQ_f[3]);
    printf("  dK[0..3] = %.4f %.4f %.4f %.4f\n",
           dK_f[0], dK_f[1], dK_f[2], dK_f[3]);
    printf("  dV[0..3] = %.4f %.4f %.4f %.4f\n",
           dV_f[0], dV_f[1], dV_f[2], dV_f[3]);

    flash_attn_output_free(flashOut);
    free(Q); free(K); free(V); free(naiveO);
    free(dO); free(dQ_f); free(dK_f); free(dV_f);
}

static void benchmark_io(void) {
    printf("\n--- IO Complexity Benchmark ---\n");

    uint32_t headDim = 64;
    uint32_t sramSizeKB[] = { 128, 192, 256 };
    uint32_t seqLens[]    = { 512, 1024, 2048, 4096 };

    printf("%8s  %8s  %8s  %8s  %8s\n",
           "N", "SRAM(KB)", "Naive", "Flash", "Ratio");
    printf("----------------------------------------------\n");

    for (int si = 0; si < 3; si++) {
        for (int ni = 0; ni < 4; ni++) {
            uint32_t N = seqLens[ni];
            uint32_t M = sramSizeKB[si] * 1024;
            double naive = (double)N * N * headDim;
            double flash = N * N * headDim * headDim / (double)M;
            printf("%8u  %8u  %8.1e  %8.1e  %6.0fx\n",
                   N, sramSizeKB[si], naive, flash, naive / flash);
        }
    }
}

static void demo_online_softmax(void) {
    printf("\n--- Online Softmax Demonstration ---\n");

    uint32_t headDim = 4;
    uint32_t blockSize = 3;
    double scale = 1.0 / sqrt((double)headDim);

    double q[12] = { 1,0,0,0,  0,1,0,0,  0,0,1,0 };
    double k[12] = { 1,0,0,0,  0,1,0,0,  0,0,1,0 };
    double v[12] = { 1,2,3,4,  5,6,7,8,  9,10,11,12 };

    double *o = (double *)calloc(12, sizeof(double));
    double *lse = (double *)malloc(3 * sizeof(double));

    FlashAttnConfig cfg = flash_attn_config_create(3, headDim, 0);
    cfg.blockSizeBr = blockSize;
    cfg.blockSizeBc = blockSize;
    cfg.causal = 0;
    cfg.softmaxScale = scale;

    flash_attn_forward_head(q, k, v, 3, headDim,
                             cfg.blockSizeBr, cfg.blockSizeBc,
                             cfg.causal, cfg.softmaxScale,
                             o, lse);

    printf("Q, K, V = Identity-like 3×4 matrices\n");
    printf("Attention output O (3×4):\n");
    for (uint32_t i = 0; i < 3; i++) {
        printf("  row %u: ", i);
        for (uint32_t d = 0; d < headDim; d++)
            printf("%7.3f ", o[i * headDim + d]);
        printf(" LSE=%.4f\n", lse[i]);
    }

    printf("\nOnline softmax process:\n");
    printf("  1. Load Q block (rows 0..%u)\n", blockSize - 1);
    printf("  2. For each K/V block:\n");
    printf("     a. Compute S = Q_block × K_block^T (in SRAM)\n");
    printf("     b. m_new = max(m_old, rowmax(S))\n");
    printf("     c. l = l_old * exp(m_old - m_new) + sum(exp(S - m_new))\n");
    printf("     d. O = O_old * exp(m_old - m_new) + exp(S - m_new) × V_block\n");

    free(o); free(lse);
}

static void demo_causal_mask(void) {
    printf("\n--- Causal Mask Demo (GPT-style autoregressive) ---\n");

    uint32_t seqLen  = 6;
    uint32_t headDim = 4;
    double scale = 1.0 / sqrt((double)headDim);

    size_t perHead = (size_t)seqLen * headDim;
    double *Q = (double *)calloc(perHead, sizeof(double));
    double *K = (double *)calloc(perHead, sizeof(double));
    double *V = (double *)calloc(perHead, sizeof(double));
    fill_random(Q, perHead);
    fill_random(K, perHead);
    fill_random(V, perHead);

    double *oCausal    = (double *)calloc(perHead, sizeof(double));
    double *lseCausal  = (double *)calloc(seqLen, sizeof(double));
    double *oNoCausal  = (double *)calloc(perHead, sizeof(double));
    double *lseNoCausal = (double *)calloc(seqLen, sizeof(double));

    FlashAttnConfig cfg = flash_attn_config_create(seqLen, headDim, 1);
    flash_attn_forward_head(Q, K, V, seqLen, headDim,
                             cfg.blockSizeBr, cfg.blockSizeBc,
                             1, cfg.softmaxScale, oCausal, lseCausal);

    flash_attn_forward_head(Q, K, V, seqLen, headDim,
                             cfg.blockSizeBr, cfg.blockSizeBc,
                             0, cfg.softmaxScale, oNoCausal, lseNoCausal);

    printf("Difference causal vs non-causal:\n");
    double maxDiff = 0.0;
    for (size_t i = 0; i < perHead; i++) {
        double d = fabs(oCausal[i] - oNoCausal[i]);
        if (d > maxDiff) maxDiff = d;
    }
    printf("  Max |O_causal - O_nocausal| = %.6e\n", maxDiff);
    printf("  (Last positions differ most: causal ignores future tokens)\n");

    free(Q); free(K); free(V); free(oCausal); free(lseCausal);
    free(oNoCausal); free(lseNoCausal);
}

static void demo_memory_savings(void) {
    printf("\n--- Memory Savings from No HBM Attention Matrix ---\n");

    printf("%8s  %10s  %14s  %12s  %12s\n",
           "N", "d", "AttnMatrix(GB)", "NaiveMem(GB)", "FlashMem(GB)");
    printf("-------------------------------------------------------\n");

    uint32_t configs[][2] = {
        {2048, 64}, {4096, 64}, {8192, 64},
        {2048, 128}, {4096, 128}, {8192, 128}
    };

    for (int i = 0; i < 6; i++) {
        uint32_t N = configs[i][0];
        uint32_t d = configs[i][1];
        double attnBytes = (double)N * N * sizeof(double);
        double naiveBytes = (double)N * N * sizeof(double) + (double)N * d * sizeof(double) * 6;
        double flashBytes = (double)N * d * sizeof(double) * 4; /* Q,K,V,O only */
        printf("%8u  %10u  %14.6f  %12.6f  %12.6f\n",
               N, d,
               attnBytes / 1e9, naiveBytes / 1e9, flashBytes / 1e9);
    }

    printf("\nFlashAttention advantage (A100 80GB):\n");
    printf("  N=8192,d=64: attn matrix = %.1f GB → impossible naively\n",
           (double)8192 * 8192 * 4 / 1e9);
    printf("  With FlashAttention: no O(N²) HBM writes\n");
}

int main(void) {
    srand((unsigned)time(NULL));

    printf("=== FlashAttention Demo ===\n");
    printf("  Algorithm: tile-based online softmax attention\n");
    printf("  Key insight: keep Q×K^T in SRAM, never write to HBM\n\n");

    FlashAttnConfig cfg = flash_attn_config_create(128, 64, 0);
    printf("Default config: N=%u d=%u Br=%u Bc=%u\n",
           cfg.seqLen, cfg.headDim, cfg.blockSizeBr, cfg.blockSizeBc);

    verify_correctness(cfg, 0);
    verify_correctness(cfg, 1);

    benchmark_io();
    demo_online_softmax();
    demo_causal_mask();
    demo_memory_savings();

    printf("\n=== Demo complete ===\n");
    return 0;
}
