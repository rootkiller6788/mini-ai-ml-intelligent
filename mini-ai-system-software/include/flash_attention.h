#ifndef FLASH_ATTENTION_H
#define FLASH_ATTENTION_H

#include <stdint.h>
#include <stddef.h>

/* --- FlashAttention configuration --- */
typedef struct {
    uint32_t batchSize;
    uint32_t numHeads;
    uint32_t seqLen;       /* sequence length N */
    uint32_t headDim;      /* d, typically 64 or 128 */
    uint32_t blockSizeBr;  /* Q block size (rows per tile)   */
    uint32_t blockSizeBc;  /* K/V block size (cols per tile) */
    int      causal;       /* apply causal mask (1 = lower-triangular) */
    double   softmaxScale; /* 1/sqrt(d), precomputed */
} FlashAttnConfig;

/* --- Online softmax state for a block row --- */
typedef struct {
    double *m;       /* running max, length = blockSizeBr       */
    double *l;       /* running sum-exp, length = blockSizeBr   */
    double *O;       /* output accumulator, blockSizeBr × headDim */
    double *dO;      /* gradient of output (backward)            */
    double *dQ;      /* gradient of Q (backward)                 */
    double *dK;      /* gradient of K (backward)                 */
    double *dV;      /* gradient of V (backward)                 */
    uint32_t rows;
    uint32_t cols;
} FlashAttnState;

/* --- Tile (block) data: one block of Q, K, V --- */
typedef struct {
    double *data;     /* blockSize × headDim elements, row-major */
    uint32_t rows;
    uint32_t cols;
    uint32_t ld;      /* leading dimension */
} FlashTile;

/* --- Attention result --- */
typedef struct {
    double *output;   /* O: batchSize × numHeads × seqLen × headDim */
    double *lse;      /* log-sum-exp per query position */
    uint32_t batchSize;
    uint32_t numHeads;
    uint32_t seqLen;
    uint32_t headDim;
} FlashAttnOutput;

/* --- API --- */

/* Initialize FlashAttention config with sensible defaults */
FlashAttnConfig flash_attn_config_create(uint32_t seqLen, uint32_t headDim,
                                          int causal);

/* Forward pass: tile-based attention with online softmax.
 * Q, K, V are row-major (batchSize × numHeads × seqLen × headDim) each.
 * Returns dynamically allocated FlashAttnOutput (caller frees). */
FlashAttnOutput *flash_attn_forward(const double *Q, const double *K,
                                     const double *V,
                                     const FlashAttnConfig *cfg);

/* Backward pass: recompute softmax from SRAM (no HBM write of attention matrix).
 * dO is the gradient of the loss w.r.t. O (same shape as Q/K/V).
 * Outputs dQ, dK, dV (caller-allocated, same shape as Q/K/V). */
void flash_attn_backward(const double *Q, const double *K, const double *V,
                          const double *dO,
                          double *dQ, double *dK, double *dV,
                          const FlashAttnConfig *cfg);

/* Single-head forward (lower-level) for a given Q/K/V tile iteration. */
void flash_attn_forward_head(const double *q_head, const double *k_head,
                              const double *v_head,
                              uint32_t seqLen, uint32_t headDim,
                              uint32_t blockSizeBr, uint32_t blockSizeBc,
                              int causal, double softmaxScale,
                              double *o_head, double *lse);

/* Online softmax: given a new block of scores, update running max/sum/output */
void flash_online_softmax_block(FlashAttnState *state,
                                 const double *scoresBlock,
                                 const double *valuesBlock,
                                 uint32_t blockRows, uint32_t blockCols,
                                 uint32_t headDim);

/* IO complexity estimate: O(N²d² / M⁻¹) reduction vs naive O(N²). */
double flash_attn_io_complexity(uint32_t seqLen, uint32_t headDim,
                                 uint32_t sramSize);

/* Free output structure */
void flash_attn_output_free(FlashAttnOutput *out);

/* Free state */
void flash_attn_state_free(FlashAttnState *state);

#endif /* FLASH_ATTENTION_H */
