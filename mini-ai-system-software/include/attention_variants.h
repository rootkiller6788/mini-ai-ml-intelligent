#ifndef ATTENTION_VARIANTS_H
#define ATTENTION_VARIANTS_H

#include <stdint.h>
#include <stddef.h>

/* ───────────────────────────────────────────────────────────────
 * Attention Mechanism Variants — MQA, GQA, Sliding Window
 *
 * L7 — Applications:
 *   Multi-Query Attention (MQA): K,V shared across all heads
 *     → KV-cache reduced by factor H (number of heads)
 *     → Used in: PaLM, Gemini inference
 *   Grouped-Query Attention (GQA): K,V shared within groups
 *     → KV-cache reduced by factor H/G
 *     → Used in: LLaMA 2/3, Mistral
 *
 * L8 — Advanced Topics:
 *   Sliding Window Attention: each query attends to local window
 *     → O(N·W) complexity instead of O(N²)
 *   KV-Cache management for autoregressive decoding
 * ─────────────────────────────────────────────────────────────── */

/* --- MHA / MQA / GQA configuration --- */
typedef enum {
    ATTN_MHA = 0,  /* Multi-Head: 1 KV per head */
    ATTN_MQA = 1,  /* Multi-Query: 1 KV for all heads */
    ATTN_GQA = 2   /* Grouped-Query: 1 KV per group of heads */
} AttentionVariant;

/* --- Attention descriptor --- */
typedef struct {
    AttentionVariant variant;
    uint32_t batchSize;
    uint32_t numQHeads;       /* number of query heads */
    uint32_t numKVHeads;      /* number of K/V heads (1 for MQA, G for GQA) */
    uint32_t seqLenQ;         /* query sequence length */
    uint32_t seqLenKV;        /* key/value sequence length (cached) */
    uint32_t headDim;         /* dimension per head */
    uint32_t groupSize;       /* Q heads per KV head (for GQA) */
    double   softmaxScale;    /* 1/sqrt(headDim) */
    int      causal;          /* apply causal mask */
} AttnDescriptor;

/* ─── MHA (standard multi-head attention, for reference) ─── */

/* Standard MHA forward: each Q head attends to its own K,V head.
 * Q: batchSize × numHeads × seqLenQ × headDim
 * K,V: batchSize × numHeads × seqLenKV × headDim
 * output: same shape as Q */
void attn_mha_forward(const double *Q, const double *K, const double *V,
                       const AttnDescriptor *desc, double *output);

/* ─── MQA (Multi-Query Attention) ─── */

/* MQA forward: all Q heads share one K,V head pair.
 * Q: batchSize × numQHeads × seqLenQ × headDim
 * K,V: batchSize × 1 × seqLenKV × headDim (shared)
 * For each head h: attn(Q_h, K_shared, V_shared).
 * KV-cache memory: seqLenKV × headDim (vs MHA: H×seqLenKV×headDim).
 * Memory savings: H× reduction in KV-cache size. */
void attn_mqa_forward(const double *Q, const double *K, const double *V,
                       const AttnDescriptor *desc, double *output);

/* ─── GQA (Grouped-Query Attention) ─── */

/* GQA forward: Q heads grouped, each group shares one K,V head.
 * Q: batchSize × numQHeads × seqLenQ × headDim
 * K,V: batchSize × numKVHeads × seqLenKV × headDim
 * groupSize = numQHeads / numKVHeads
 * Head h accesses K[h / groupSize] and V[h / groupSize].
 * KV-cache memory: G×seqLenKV×headDim (vs MHA: H×seqLenKV×headDim). */
void attn_gqa_forward(const double *Q, const double *K, const double *V,
                       const AttnDescriptor *desc, double *output);

/* ─── KV-Cache ─── */

/* KV-Cache data structure for autoregressive decoding.
 * Stores past K,V activations to avoid recomputation.
 * For each new token: compute Q for that token, K,V for that token,
 *   append to cache, attend Q over all cached K,V. */
typedef struct {
    double *kCache;         /* batchSize × numKVHeads × maxSeqLen × headDim */
    double *vCache;
    uint32_t batchSize;
    uint32_t numKVHeads;
    uint32_t maxSeqLen;
    uint32_t currentLen;    /* tokens cached so far */
    uint32_t headDim;
    uint32_t cacheBytes;    /* total bytes allocated */
} KVCache;

/* Initialize a KV-cache with given capacity */
KVCache *kv_cache_create(uint32_t batchSize, uint32_t numKVHeads,
                           uint32_t maxSeqLen, uint32_t headDim);

/* Append new K,V tokens to the cache (in-place).
 * Returns new currentLen. */
uint32_t kv_cache_append(KVCache *cache, const double *kNew,
                           const double *vNew, uint32_t numTokens);

/* Retrieve slice [start, end) from cache for attention computation */
void kv_cache_get_range(const KVCache *cache,
                          uint32_t batchIdx, uint32_t kvHeadIdx,
                          uint32_t start, uint32_t end,
                          double *kOut, double *vOut);

/* Free KV-cache */
void kv_cache_free(KVCache *cache);

/* KV-cache memory comparison across MHA/MQA/GQA */
void kv_cache_memory_compare(uint32_t numQHeads, uint32_t numKVHeads,
                               uint32_t seqLen, uint32_t headDim,
                               uint32_t numLayers,
                               double *mhaBytes, double *mqaBytes,
                               double *gqaBytes);

/* ─── Sliding Window Attention ─── */

/* Sliding window attention: each query attends to ±windowSize tokens.
 * For causal: only look back windowSize tokens (including self).
 * Complexity: O(N·W) where W = windowSize, vs O(N²) for full attention. */
typedef struct {
    uint32_t windowSize;     /* half-window for non-causal, full window for causal */
    int      causal;         /* 1 = autoregressive (left-only window) */
    int      useAlibi;       /* 1 = add ALiBi position bias */
    uint32_t numAlibiHeads;  /* number of heads for ALiBi slopes */
} SlidingWindowConfig;

/* Sliding window attention forward pass.
 * Only computes attention within the sliding window.
 * For token i, attends to max(0, i-window+1)..i if causal,
 * or i-window..i+window if non-causal. */
void attn_sliding_window_forward(const double *Q, const double *K,
                                   const double *V,
                                   const AttnDescriptor *desc,
                                   const SlidingWindowConfig *sw,
                                   double *output);

/* ALiBi (Attention with Linear Biases) slope computation.
 * For head h, slope = 2^(-8/h * (h+1)).
 * Adds position bias: score += -slope * |i-j| for non-causal,
 * or -slope * (i-j) for causal (i≥j). */
void attn_alibi_bias(double *scores, uint32_t seqLenQ, uint32_t seqLenKV,
                       uint32_t numHeads, int causal);

/* ─── Analytical tools ─── */

/* Compute floating-point operations for given attention variant */
double attn_flops_count(const AttnDescriptor *desc, AttentionVariant variant);

/* Memory bandwidth estimate for attention (model weights + KV-cache) */
double attn_memory_bandwidth_estimate(const AttnDescriptor *desc,
                                        double clockGHz,
                                        double busWidthBytes);

#endif /* ATTENTION_VARIANTS_H */
