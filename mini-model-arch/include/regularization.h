#ifndef REGULARIZATION_H
#define REGULARIZATION_H

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ── L1: Core Definitions ──
 * Regularization techniques to prevent overfitting.
 * References:
 *   - Srivastava et al., "Dropout" (JMLR 2014)
 *   - Wan et al., "DropConnect" (ICML 2013)
 *   - Szegedy et al., "Rethinking the Inception Architecture" (CVPR 2016)
 */

/* ── L2: Dropout — randomly zeroes activations during training ──
 * During inference: no dropout (or scale by p for inverted dropout).
 * Theorem: Approximates geometric model averaging (Srivastava et al.).
 */

typedef struct {
    float p;                 /* keep probability */
    int   inverted;          /* 1 = inverted dropout (scale at train, none at test) */
    int   *mask;             /* pre-allocated boolean mask */
    int   max_size;
} Dropout;

/* ── L5: Dropout create/apply/free ── */
Dropout *dropout_create(float keep_prob, int inverted, int max_size);
void     dropout_free(Dropout *d);
void     dropout_forward(Dropout *d, float *x, int n, int training);

/* ── L5: DropConnect — drops weights instead of activations ── */
void dropconnect_forward(const float *W, float *W_masked, int rows, int cols,
                         float keep_prob, int training);

/* ── L4: Label Smoothing (Szegedy et al., 2016) ──
 * Replace one-hot targets with soft targets:
 *   y_smooth = (1 - eps) * y_onehot + eps / K
 * Reduces overconfidence and improves calibration.
 */

void label_smoothing(const int *targets, float *smooth, int batch, int classes, float eps);

/* ── L5: Weight decay — L2 regularization in parameter space ── */
void weight_decay_apply(float *params, int n, float lr, float lambda);

/* ── L5: Early stopping heuristic ── */
typedef struct {
    float *best_params;
    float  best_loss;
    int    patience;
    int    patience_counter;
    int    param_count;
} EarlyStopping;

EarlyStopping *early_stop_create(int param_count, int patience);
void           early_stop_free(EarlyStopping *e);
int            early_stop_check(EarlyStopping *e, const float *params, float current_loss);

/* ── L7: Mixup data augmentation (Zhang et al., ICLR 2018) ── */
void mixup_augment(const float *x1, const float *x2, const float *y1, const float *y2,
                   float *x_out, float *y_out, int n, float alpha);
float mixup_beta_sample(float alpha);

#endif
