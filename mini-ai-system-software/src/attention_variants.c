/*
 * attention_variants.c — MQA, GQA, KV-Cache, Sliding Window
 *
 * Knowledge points (L7-L8):
 *   L7: Multi-Query Attention (MQA) — K,V shared across all Q heads
 *        → KV-cache reduction factor H (for H query heads)
 *        → Paper: Shazeer, "Fast Transformer Decoding" (2019)
 *   L7: Grouped-Query Attention (GQA) — K,V shared per group
 *        → KV-cache reduction factor H/G
 *        → Paper: Ainslie et al., "GQA: Training Generalized
 *          Multi-Query Transformer Models" (2023)
 *   L7: KV-Cache management for autoregressive LLM inference
 *        → Memory-bandwidth tradeoff
 *   L8: Sliding Window Attention — O(N·W) complexity
 *        → Paper: Beltagy et al., "Longformer" (2020)
 *   L8: ALiBi position bias — linear position encoding
 *        → Paper: Press et al., "Train Short, Test Long" (ICLR 2022)
 */

#include "attention_variants.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* ─── Internal helpers ─── */

static inline double dot_product(const double *a, const double *b,
                                  uint32_t dim) {
    double s = 0.0;
    for (uint32_t i = 0; i < dim; i++) s += a[i] * b[i];
    return s;
}

/* Softmax over a row vector: in-place, returns max for numerical stability */
static void softmax_row(double *row, uint32_t len) {
    double maxVal = row[0];
    for (uint32_t i = 1; i < len; i++)
        if (row[i] > maxVal) maxVal = row[i];
    double sumExp = 0.0;
    for (uint32_t i = 0; i < len; i++) {
        row[i] = exp(row[i] - maxVal);
        sumExp += row[i];
    }
    if (sumExp > 0.0) {
        for (uint32_t i = 0; i < len; i++)
            row[i] /= sumExp;
    }
}

/* ─── Standard MHA ─── */

void attn_mha_forward(const double *Q, const double *K, const double *V,
                       const AttnDescriptor *desc, double *output) {
    if (!Q || !K || !V || !desc || !output) return;

    uint32_t H = desc->numQHeads;
    uint32_t Nq = desc->seqLenQ;
    uint32_t Nkv = desc->seqLenKV;
    uint32_t d = desc->headDim;
    double scale = desc->softmaxScale;

    printf("[MHA] %u heads, Q=(%u×%u), KV=(%u×%u), dim=%u\n",
           H, Nq, d, Nkv, d, d);

    double *scores = (double *)malloc(Nkv * sizeof(double));
    for (uint32_t h = 0; h < H; h++) {
        const double *qHead = Q + h * Nq * d;
        const double *kHead = K + h * Nkv * d;
        const double *vHead = V + h * Nkv * d;
        double *oHead = output + h * Nq * d;

        for (uint32_t qi = 0; qi < Nq; qi++) {
            for (uint32_t kj = 0; kj < Nkv; kj++) {
                if (desc->causal && qi < kj)
                    scores[kj] = -INFINITY;
                else
                    scores[kj] = dot_product(qHead + qi * d,
                                              kHead + kj * d, d) * scale;
            }
            softmax_row(scores, Nkv);
            for (uint32_t dd = 0; dd < d; dd++) {
                double sum = 0.0;
                for (uint32_t kj = 0; kj < Nkv; kj++)
                    sum += scores[kj] * vHead[kj * d + dd];
                oHead[qi * d + dd] = sum;
            }
        }
    }
    free(scores);
    printf("[MHA] Done\n");
}

/* ─── MQA: Multi-Query Attention ─── */

void attn_mqa_forward(const double *Q, const double *K, const double *V,
                       const AttnDescriptor *desc, double *output) {
    if (!Q || !K || !V || !desc || !output) return;

    uint32_t H = desc->numQHeads;
    uint32_t Nq = desc->seqLenQ;
    uint32_t Nkv = desc->seqLenKV;
    uint32_t d = desc->headDim;
    double scale = desc->softmaxScale;

    printf("[MQA] %u Q-heads, 1 KV-head, Q=(%u×%u), KV=(%u×%u)\n",
           H, Nq, d, Nkv, d);

    /* In MQA: all Q heads attend to the same K and V.
     * K and V are stored as (1 × Nkv × d) — no head dimension.
     * KV-cache size: Nkv·d (vs MHA: H·Nkv·d). */

    double *scores = (double *)malloc(Nkv * sizeof(double));
    for (uint32_t h = 0; h < H; h++) {
        const double *qHead = Q + h * Nq * d;
        /* All heads share the same K,V */
        const double *kShared = K;  /* (Nkv × d) */
        const double *vShared = V;  /* (Nkv × d) */
        double *oHead = output + h * Nq * d;

        for (uint32_t qi = 0; qi < Nq; qi++) {
            for (uint32_t kj = 0; kj < Nkv; kj++) {
                if (desc->causal && qi < kj)
                    scores[kj] = -INFINITY;
                else
                    scores[kj] = dot_product(qHead + qi * d,
                                              kShared + kj * d, d) * scale;
            }
            softmax_row(scores, Nkv);
            for (uint32_t dd = 0; dd < d; dd++) {
                double sum = 0.0;
                for (uint32_t kj = 0; kj < Nkv; kj++)
                    sum += scores[kj] * vShared[kj * d + dd];
                oHead[qi * d + dd] = sum;
            }
        }
    }

    /* KV-cache analysis: MQA vs MHA */
    size_t mhaKVBytes = (size_t)H * Nkv * d * sizeof(double) * 2;
    size_t mqaKVBytes = (size_t)Nkv * d * sizeof(double) * 2;
    printf("[MQA] KV-cache: %zu bytes vs MHA %zu bytes (%.1f× reduction)\n",
           mqaKVBytes, mhaKVBytes,
           mhaKVBytes > 0 ? (double)mhaKVBytes / (double)mqaKVBytes : 0.0);

    free(scores);
    printf("[MQA] Done\n");
}

/* ─── GQA: Grouped-Query Attention ─── */

void attn_gqa_forward(const double *Q, const double *K, const double *V,
                       const AttnDescriptor *desc, double *output) {
    if (!Q || !K || !V || !desc || !output) return;

    uint32_t H = desc->numQHeads;
    uint32_t G = desc->numKVHeads;  /* number of KV groups */
    uint32_t groupSize = H / G;
    uint32_t Nq = desc->seqLenQ;
    uint32_t Nkv = desc->seqLenKV;
    uint32_t d = desc->headDim;
    double scale = desc->softmaxScale;

    printf("[GQA] %u Q-heads / %u KV-groups = %u heads per group, "
           "Q=(%u×%u), KV=(%u×%u)\n",
           H, G, groupSize, Nq, d, Nkv, d);

    /* In GQA: Q heads grouped, each group shares one K,V pair.
     * Head h uses K[h / groupSize] and V[h / groupSize]. */

    double *scores = (double *)malloc(Nkv * sizeof(double));
    for (uint32_t h = 0; h < H; h++) {
        uint32_t kvIdx = h / groupSize;  /* which KV head to use */
        const double *qHead = Q + h * Nq * d;
        const double *kHead = K + kvIdx * Nkv * d;
        const double *vHead = V + kvIdx * Nkv * d;
        double *oHead = output + h * Nq * d;

        for (uint32_t qi = 0; qi < Nq; qi++) {
            for (uint32_t kj = 0; kj < Nkv; kj++) {
                if (desc->causal && qi < kj)
                    scores[kj] = -INFINITY;
                else
                    scores[kj] = dot_product(qHead + qi * d,
                                              kHead + kj * d, d) * scale;
            }
            softmax_row(scores, Nkv);
            for (uint32_t dd = 0; dd < d; dd++) {
                double sum = 0.0;
                for (uint32_t kj = 0; kj < Nkv; kj++)
                    sum += scores[kj] * vHead[kj * d + dd];
                oHead[qi * d + dd] = sum;
            }
        }
    }

    /* KV-cache comparison */
    size_t mhaBytes = (size_t)H * Nkv * d * sizeof(double) * 2;
    size_t gqaBytes = (size_t)G * Nkv * d * sizeof(double) * 2;
    printf("[GQA] KV-cache: %zu bytes vs MHA %zu bytes (%.1f× reduction)\n",
           gqaBytes, mhaBytes,
           mhaBytes > 0 ? (double)mhaBytes / (double)gqaBytes : 0.0);

    free(scores);
    printf("[GQA] Done\n");
}

/* ─── KV-Cache ─── */

KVCache *kv_cache_create(uint32_t batchSize, uint32_t numKVHeads,
                           uint32_t maxSeqLen, uint32_t headDim) {
    KVCache *cache = (KVCache *)malloc(sizeof(KVCache));
    memset(cache, 0, sizeof(*cache));
    cache->batchSize  = batchSize;
    cache->numKVHeads = numKVHeads;
    cache->maxSeqLen  = maxSeqLen;
    cache->headDim    = headDim;
    cache->currentLen = 0;

    size_t totalElems = (size_t)batchSize * numKVHeads * maxSeqLen * headDim;
    cache->cacheBytes = totalElems * sizeof(double) * 2; /* K + V */
    cache->kCache = (double *)calloc(totalElems, sizeof(double));
    cache->vCache = (double *)calloc(totalElems, sizeof(double));

    printf("[KV-CACHE] Created: %u batches × %u KV-heads × %u maxLen × %u dim "
           "= %.2f MB\n",
           batchSize, numKVHeads, maxSeqLen, headDim,
           (double)cache->cacheBytes / 1e6);
    return cache;
}

uint32_t kv_cache_append(KVCache *cache, const double *kNew,
                           const double *vNew, uint32_t numTokens) {
    if (!cache || !kNew || !vNew) return 0;

    uint32_t remaining = cache->maxSeqLen - cache->currentLen;
    uint32_t toAppend = (numTokens < remaining) ? numTokens : remaining;

    uint32_t B = cache->batchSize, G = cache->numKVHeads;
    uint32_t d = cache->headDim;
    size_t headSize = (size_t)d * sizeof(double);

    for (uint32_t b = 0; b < B; b++) {
        for (uint32_t g = 0; g < G; g++) {
            size_t kvOffset = ((size_t)b * G + g) * cache->maxSeqLen * d;
            for (uint32_t t = 0; t < toAppend; t++) {
                size_t dstPos = kvOffset + (size_t)(cache->currentLen + t) * d;
                size_t srcPos = ((size_t)b * G + g) * numTokens * d + t * d;
                memcpy(cache->kCache + dstPos, kNew + srcPos, headSize);
                memcpy(cache->vCache + dstPos, vNew + srcPos, headSize);
            }
        }
    }

    cache->currentLen += toAppend;
    printf("[KV-CACHE] Appended %u tokens → length %u/%u\n",
           toAppend, cache->currentLen, cache->maxSeqLen);
    return cache->currentLen;
}

void kv_cache_get_range(const KVCache *cache,
                          uint32_t batchIdx, uint32_t kvHeadIdx,
                          uint32_t start, uint32_t end,
                          double *kOut, double *vOut) {
    if (!cache || !kOut || !vOut) return;
    if (start >= end || end > cache->currentLen) return;

    uint32_t d = cache->headDim;
    size_t baseOffset = ((size_t)batchIdx * cache->numKVHeads + kvHeadIdx)
                       * cache->maxSeqLen * d;
    size_t rangeSize = (size_t)(end - start) * d * sizeof(double);

    memcpy(kOut, cache->kCache + baseOffset + (size_t)start * d, rangeSize);
    memcpy(vOut, cache->vCache + baseOffset + (size_t)start * d, rangeSize);
}

void kv_cache_free(KVCache *cache) {
    if (cache) {
        free(cache->kCache);
        free(cache->vCache);
        free(cache);
    }
}

void kv_cache_memory_compare(uint32_t numQHeads, uint32_t numKVHeads,
                               uint32_t seqLen, uint32_t headDim,
                               uint32_t numLayers,
                               double *mhaBytes, double *mqaBytes,
                               double *gqaBytes) {
    double perTokenKV = (double)headDim * sizeof(double) * 2; /* K + V */
    double mha  = (double)numLayers * numQHeads  * seqLen * perTokenKV;
    double mqa  = (double)numLayers * 1           * seqLen * perTokenKV;
    double gqaB = (double)numLayers * numKVHeads  * seqLen * perTokenKV;

    if (mhaBytes)  *mhaBytes  = mha;
    if (mqaBytes)  *mqaBytes  = mqa;
    if (gqaBytes)  *gqaBytes  = gqaB;

    printf("[KV-CACHE-MEM] seqLen=%u, layers=%u:\n", seqLen, numLayers);
    printf("  MHA (%u heads): %.2f GB\n", numQHeads, mha / 1e9);
    printf("  MQA (1 KV):     %.2f GB (%.1f× smaller)\n",
           mqa / 1e9, mha / mqa);
    printf("  GQA (%u KV):    %.2f GB (%.1f× smaller)\n",
           numKVHeads, gqaB / 1e9, mha / gqaB);
}

/* ─── Sliding Window Attention ─── */

void attn_sliding_window_forward(const double *Q, const double *K,
                                   const double *V,
                                   const AttnDescriptor *desc,
                                   const SlidingWindowConfig *sw,
                                   double *output) {
    if (!Q || !K || !V || !desc || !sw || !output) return;

    uint32_t H = desc->numQHeads;
    uint32_t Nq = desc->seqLenQ;
    uint32_t Nkv = desc->seqLenKV;
    uint32_t d = desc->headDim;
    double scale = desc->softmaxScale;
    uint32_t W = sw->windowSize;

    printf("[SLIDING-WINDOW] window=%u, causal=%d, Nq=%u, Nkv=%u\n",
           W, sw->causal, Nq, Nkv);

    /* Complexity: O(N·W·d) instead of O(N²·d).
     * For each query position i:
     *   causal: j = max(0, i-W+1) .. i
     *   non-causal: j = max(0, i-W) .. min(Nkv-1, i+W) */

    double *scores = (double *)malloc((size_t)(2 * W + 1) * sizeof(double));

    for (uint32_t h = 0; h < H; h++) {
        const double *qHead = Q + h * Nq * d;
        const double *kHead = K + (h % desc->numKVHeads) * Nkv * d;
        const double *vHead = V + (h % desc->numKVHeads) * Nkv * d;
        double *oHead = output + h * Nq * d;

        for (uint32_t qi = 0; qi < Nq; qi++) {
            uint32_t jStart, jEnd;
            if (sw->causal) {
                jStart = (qi >= W) ? qi - W + 1 : 0;
                jEnd = qi + 1;
            } else {
                jStart = (qi >= W) ? qi - W : 0;
                jEnd = (qi + W + 1 < Nkv) ? qi + W + 1 : Nkv;
            }

            uint32_t localLen = jEnd - jStart;
            /* Compute scores only for window */
            for (uint32_t jj = 0; jj < localLen; jj++) {
                uint32_t kj = jStart + jj;
                scores[jj] = dot_product(qHead + qi * d,
                                          kHead + kj * d, d) * scale;
            }
            softmax_row(scores, localLen);

            for (uint32_t dd = 0; dd < d; dd++) {
                double sum = 0.0;
                for (uint32_t jj = 0; jj < localLen; jj++) {
                    uint32_t kj = jStart + jj;
                    sum += scores[jj] * vHead[kj * d + dd];
                }
                oHead[qi * d + dd] = sum;
            }
        }
    }

    /* Complexity analysis */
    double fullOps = (double)Nq * (double)Nkv * (double)d * (double)H * 2.0;
    double slideOps = (double)Nq * (double)W * (double)d * (double)H * 2.0;
    printf("[SLIDING-WINDOW] Complexity: full=%.2e ops, sliding=%.2e ops "
           "(%.1f× reduction)\n",
           fullOps, slideOps, fullOps / slideOps);

    free(scores);
    printf("[SLIDING-WINDOW] Done\n");
}

/* ─── ALiBi Bias ─── */

void attn_alibi_bias(double *scores, uint32_t seqLenQ, uint32_t seqLenKV,
                       uint32_t numHeads, int causal) {
    if (!scores) return;

    printf("[ALIBI] %u heads, Q=%u, KV=%u, causal=%d\n",
           numHeads, seqLenQ, seqLenKV, causal);

    /* ALiBi slopes: for head h, slope = 2^(-8/h * (h+1))
     * where h is 1-indexed. Common: 2^{-8/numHeads}, 2^{-8/numHeads * 2},...
     * Exact formula from Press et al. (2022):
     *   For n heads: slopes = 2^{-8/n}, 2^{-8/n * 2}, ..., 2^{-8}
     *   For head h (0-indexed): m_h = pow(2, -8.0 * (h+1) / numHeads) */

    for (uint32_t h = 0; h < numHeads; h++) {
        double slope = pow(2.0, -8.0 * (double)(h + 1) / (double)numHeads);
        for (uint32_t qi = 0; qi < seqLenQ; qi++) {
            double *row = scores + (h * seqLenQ + qi) * seqLenKV;
            for (uint32_t kj = 0; kj < seqLenKV; kj++) {
                if (causal && qi < kj) continue;
                /* score -= slope * |qi - kj| (non-causal uses absolute) */
                double dist = (double)((int)qi - (int)kj);
                if (!causal) dist = fabs(dist);
                row[kj] -= slope * dist;
            }
        }
        if (h < 4 || h >= numHeads - 1)
            printf("[ALIBI] head %u slope = %.4f\n", h, slope);
    }
    printf("[ALIBI] Done\n");
}

/* ─── FLOPs Count ─── */

double attn_flops_count(const AttnDescriptor *desc, AttentionVariant variant) {
    if (!desc) return 0.0;

    uint32_t H = desc->numQHeads;
    uint32_t Kv = desc->numKVHeads;
    uint32_t Nq = desc->seqLenQ;
    uint32_t Nkv = desc->seqLenKV;
    uint32_t d = desc->headDim;

    double flops;
    /* Q·K^T: Nq × Nkv × d FLOPs per Q head */
    double qkFlops = 2.0 * (double)Nq * (double)Nkv * (double)d;
    /* softmax: ~5 ops per element */
    double smFlops = 5.0 * (double)Nq * (double)Nkv;
    /* attn·V: Nq × d × Nkv FLOPs */
    double avFlops = 2.0 * (double)Nq * (double)d * (double)Nkv;

    /* For MQA: same K,V for all heads, so only Kv=1 effective heads
     * For GQA: Kv effective heads
     * For MHA: H effective heads */
    switch (variant) {
    case ATTN_MHA:
        flops = (qkFlops + smFlops + avFlops) * (double)H;
        break;
    case ATTN_MQA:
        flops = qkFlops + smFlops + avFlops * (double)H;
        /* K is shared but attention values computed per head */
        break;
    case ATTN_GQA:
        flops = (qkFlops + smFlops) * (double)Kv + avFlops * (double)H;
        break;
    default:
        flops = 0.0;
    }

    const char *names[] = {"MHA", "MQA", "GQA"};
    printf("[ATTN-FLOPS] %s: %.2f GFLOPs\n",
           names[variant], flops / 1e9);
    return flops;
}

/* ─── Memory Bandwidth Estimate ─── */

double attn_memory_bandwidth_estimate(const AttnDescriptor *desc,
                                        double clockGHz,
                                        double busWidthBytes) {
    if (!desc) return 0.0;

    /* Peak bandwidth = clock × bus_width × 2 (DDR) for HBM.
     * Effective: depends on coalescing (not modeled here, use 80% of peak). */
    double peakBW = clockGHz * busWidthBytes * 2.0;  /* GB/s */

    /* Data movement per attention op:
     *   Load Q: Nq·d·numQHeads doubles
     *   Load K: Nkv·d·numKVHeads doubles
     *   Load V: Nkv·d·numKVHeads doubles
     *   Write O: Nq·d·numQHeads doubles
     * Total bytes moved */
    double bytesMoved = (double)desc->seqLenQ * desc->headDim
                       * (double)desc->numQHeads * sizeof(double)  /* Q */
                      + (double)desc->seqLenKV * desc->headDim
                       * (double)desc->numKVHeads * sizeof(double) * 2  /* K,V */
                      + (double)desc->seqLenQ * desc->headDim
                       * (double)desc->numQHeads * sizeof(double);  /* O */

    double effectiveBW = peakBW * 0.8;
    printf("[MEM-BW] Peak: %.1f GB/s, Effective: %.1f GB/s, "
           "Data: %.2f MB per attention\n",
           peakBW, effectiveBW, bytesMoved / 1e6);
    return effectiveBW;
}
