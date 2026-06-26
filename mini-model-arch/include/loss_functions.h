#ifndef LOSS_FUNCTIONS_H
#define LOSS_FUNCTIONS_H

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ── L1: Core Definitions ──
 * Loss functions and their gradients for neural network training.
 * Reference: Goodfellow, Bengio & Courville, "Deep Learning" (2016), Ch. 6
 */

typedef enum {
    LOSS_MSE = 0,
    LOSS_BCE,
    LOSS_CROSS_ENTROPY,
    LOSS_HUBER,
    LOSS_FOCAL,
    LOSS_KL_DIVERGENCE,
    LOSS_COSINE_EMBEDDING
} LossType;

/* ── L2: Loss function — forward (scalar value) ── */
float mse_loss(const float *pred, const float *target, int n);
float mse_loss_grad(const float *pred, const float *target, float *grad, int n);

/* ── L4: Binary Cross-Entropy — Information Theory ── */
float bce_loss(const float *pred, const float *target, int n);
float bce_loss_grad(const float *pred, const float *target, float *grad, int n);

/* ── L4: Categorical Cross-Entropy with softmax stability ── */
float cross_entropy_loss(const float *logits, const int *targets, int batch, int classes);
void  cross_entropy_grad(const float *logits, const int *targets,
                         float *grad, int batch, int classes);

/* ── L5: Huber Loss (smooth L1) — robust to outliers ── */
float huber_loss(const float *pred, const float *target, int n, float delta);
float huber_loss_grad(const float *pred, const float *target, float *grad, int n, float delta);

/* ── L8: Focal Loss (Lin et al., 2017) — for class imbalance ── */
float focal_loss(const float *logits, const int *targets, int batch, int classes,
                 float gamma, float alpha);
void  focal_loss_grad(const float *logits, const int *targets, float *grad,
                      int batch, int classes, float gamma, float alpha);

/* ── L4: KL Divergence — D_KL(P||Q) ── */
float kl_divergence(const float *p_log_prob, const float *q_log_prob, int n);

/* ── L7: Cosine embedding loss (face recognition, contrastive learning) ── */
float cosine_embedding_loss(const float *x1, const float *x2, int dim, float y, float margin);
void  cosine_embedding_grad(const float *x1, const float *x2, int dim, float y,
                            float margin, float *grad1, float *grad2);

/* ── L5: Softmax with numerical stability ── */
void  softmax_stable(float *x, int n);
void  log_softmax(const float *x, float *out, int n);

#endif
