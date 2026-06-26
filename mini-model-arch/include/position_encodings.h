#ifndef POSITION_ENCODINGS_H
#define POSITION_ENCODINGS_H

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ── L1: Core Definitions ──
 * Modern position encoding methods for Transformer architectures.
 * References:
 *   - Su et al., "RoFormer: Rotary Position Embedding" (2021) — RoPE
 *   - Press et al., "Train Short, Test Long: ALiBi" (ICLR 2022)
 *   - Vaswani et al., "Attention Is All You Need" (2017) — sinusoidal
 */

/* ── L8: Rotary Position Embedding (RoPE) — Su et al., 2021 ──
 * Encodes position via rotation in complex space.
 * Used in LLaMA, Mistral, Gemma, Qwen.
 *
 * Theorem (RoPE):
 *   For positions m, n and query/key vectors q, k:
 *     (R_m q)^T (R_n k) = q^T R_{n-m} k
 *   Where R_θ rotates by mθ. This achieves relative encoding via absolute.
 */

typedef struct {
    int    dim;               /* head dimension (must be even) */
    int    max_seq_len;
    float *cos_cached;        /* cos(m * theta_i) for m=0..max_seq_len-1 */
    float *sin_cached;        /* sin(m * theta_i) */
    float  theta_base;        /* base frequency (default 10000.0) */
} RoPE;

/* ── L8: ALiBi — Attention with Linear Biases (Press et al., 2022) ──
 * Adds a static, non-learned bias to attention scores.
 * bias(i,j) = -m * |i - j| where m is head-specific slope.
 * Enables length extrapolation without position encodings.
 */

typedef struct {
    int     n_heads;
    float  *slopes;           /* geometric sequence of slopes per head */
} ALiBi;

/* ── L2: Core API ── */

/* RoPE */
RoPE *rope_create(int dim, int max_seq_len, float theta_base);
void  rope_free(RoPE *r);
void  rope_apply(RoPE *r, float *q, float *k, int seq_len, int start_pos);

/* ALiBi */
ALiBi *alibi_create(int n_heads);
void   alibi_free(ALiBi *a);
void   alibi_apply(const ALiBi *a, float *attn_scores, int seq_len);

/* ── L7: Sinusoidal (original Transformer, Vaswani et al. 2017) ── */
void sinusoidal_pe(float *pe, int max_len, int d_model);

/* ── L7: Learnable position embedding ── */
void learnable_pe_init(float *pe, int max_len, int d_model);
void learnable_pe_apply(const float *pe, float *x, int seq_len, int d_model);

#endif
