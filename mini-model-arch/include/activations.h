#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include <math.h>
#include <stdlib.h>

/* ── L1: Core Definitions ──
 * Modern activation functions for deep neural networks.
 * Reference: Dubey et al., "The Llama 3 Herd of Models" (2024)
 */

/* ── L2: GELU — Gaussian Error Linear Unit (Hendrycks & Gimpel, 2016) ──
 * Used in BERT, GPT-2, ViT.
 * Approx: x * Phi(x) where Phi is the standard normal CDF.
 */

float gelu(float x);
void  gelu_forward(const float *x, float *out, int n);

/* ── L2: SiLU / Swish (Ramachandran et al., 2017; Elfwing et al., 2018) ──
 * siLU(x) = x * sigmoid(x)
 * Self-gated; used in LLaMA, PaLM.
 */

float silu(float x);
void  silu_forward(const float *x, float *out, int n);

/* ── L2: LeakyReLU — prevents dying ReLU ── */
float leaky_relu(float x, float alpha);
void  leaky_relu_forward(const float *x, float *out, int n, float alpha);

/* ── L2: ELU — Exponential Linear Unit (Clevert et al., 2015) ──
 * Negative saturation for noise-robustness.
 */
float elu(float x, float alpha);
void  elu_forward(const float *x, float *out, int n, float alpha);

/* ── L8: SwiGLU — Swish-Gated Linear Unit (Shazeer, 2020) ──
 * Used in LLaMA, PaLM, Gemini. Replaces FFN in transformers.
 * FFN_SwiGLU(x) = (silu(x @ W1) * (x @ W2)) @ W3
 */
void swiglu_forward(const float *x, const float *W1, const float *W2, const float *W3,
                    float *out, int batch, int d_model, int d_ff);

/* ── L8: GEGLU — GELU-Gated variant ── */
void geglu_forward(const float *x, const float *W1, const float *W2, const float *W3,
                   float *out, int batch, int d_model, int d_ff);

/* ── L5: Mish (Misra, 2019) — self-regularized non-monotonic ── */
float mish(float x);
void  mish_forward(const float *x, float *out, int n);

/* ── L7: Softplus — smooth approximation of ReLU ── */
float softplus(float x);
void  softplus_forward(const float *x, float *out, int n);

#endif
