#include "regularization.h"

/*
 * L2: Dropout (Srivastava et al., JMLR 2014)
 * During training: randomly zeroes neurons with prob (1-p).
 * During inference: no dropout (inverted) or scale by p (standard).
 *
 * L4: Dropout approximates geometric model averaging over 2^n thinned
 *     networks. At test time, the full network approximates this ensemble.
 *
 * Inverted dropout: scale kept neurons by 1/p during training,
 * no scaling at test time (default in modern frameworks).
 */
Dropout *dropout_create(float keep_prob, int inverted, int max_size) {
    Dropout *d = (Dropout *)malloc(sizeof(Dropout));
    d->p = keep_prob;
    d->inverted = inverted;
    d->max_size = max_size;
    d->mask = (int *)malloc(max_size * sizeof(int));
    return d;
}

void dropout_free(Dropout *d) { free(d->mask); free(d); }

void dropout_forward(Dropout *d, float *x, int n, int training) {
    if (!training) {
        /* In inverted dropout: no scaling at test time.
         * In standard dropout: scale by p (already done or implied by weight scaling). */
        return;
    }

    if (n > d->max_size) {
        /* Simple fallback: compute mask on the fly */
        for (int i = 0; i < n; i++) {
            float r = (float)rand() / (float)RAND_MAX;
            if (r < d->p) {
                if (d->inverted) x[i] /= d->p;  /* scale kept neuron */
                /* else: keep as-is (will scale at test time) */
            } else {
                x[i] = 0.0f;  /* drop */
            }
        }
    } else {
        for (int i = 0; i < n; i++) {
            float r = (float)rand() / (float)RAND_MAX;
            d->mask[i] = (r < d->p) ? 1 : 0;
            if (d->mask[i]) {
                if (d->inverted) x[i] /= d->p;
            } else {
                x[i] = 0.0f;
            }
        }
    }
}

/*
 * L5: DropConnect (Wan et al., ICML 2013)
 * Drops individual weights instead of activations.
 * More fine-grained regularization; used as alternative to Dropout.
 */
void dropconnect_forward(const float *W, float *W_masked, int rows, int cols,
                         float keep_prob, int training) {
    int n = rows * cols;
    memcpy(W_masked, W, n * sizeof(float));
    if (!training) {
        /* Scale weights by keep_prob at test time */
        for (int i = 0; i < n; i++) W_masked[i] *= keep_prob;
        return;
    }
    for (int i = 0; i < n; i++) {
        float r = (float)rand() / (float)RAND_MAX;
        if (r > keep_prob) W_masked[i] = 0.0f;
        else W_masked[i] /= keep_prob;  /* inverted scaling */
    }
}

/*
 * L4: Label Smoothing (Szegedy et al., CVPR 2016)
 * Replaces one-hot target y with:
 *   y_smooth = (1 - eps) * y + eps / K
 * Prevents model from becoming overconfident.
 * Improves calibration (ECE) and generalization.
 *
 * Theorem: Label smoothing encourages features of same class to
 * cluster tightly with equal distances to other class centroids.
 */
void label_smoothing(const int *targets, float *smooth, int batch, int classes, float eps) {
    float uniform = eps / (float)classes;
    float confident = 1.0f - eps;
    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < classes; c++) {
            smooth[b * classes + c] = (c == targets[b]) ? confident + uniform : uniform;
        }
    }
}

/*
 * L5: Weight Decay / L2 Regularization
 * Applies L2 penalty directly to parameters.
 * theta_i -= lr * lambda * theta_i
 * Equivalent to adding lambda * ||theta||^2 to loss.
 */
void weight_decay_apply(float *params, int n, float lr, float lambda) {
    for (int i = 0; i < n; i++)
        params[i] -= lr * lambda * params[i];
}

/*
 * L5: Early Stopping
 * Monitors validation loss; saves best parameters.
 * Stops when no improvement for 'patience' epochs.
 */
EarlyStopping *early_stop_create(int param_count, int patience) {
    EarlyStopping *e = (EarlyStopping *)malloc(sizeof(EarlyStopping));
    e->best_params = (float *)malloc(param_count * sizeof(float));
    e->best_loss = 1e9f;
    e->patience = patience;
    e->patience_counter = 0;
    e->param_count = param_count;
    return e;
}

void early_stop_free(EarlyStopping *e) { free(e->best_params); free(e); }

int early_stop_check(EarlyStopping *e, const float *params, float current_loss) {
    if (current_loss < e->best_loss) {
        e->best_loss = current_loss;
        memcpy(e->best_params, params, e->param_count * sizeof(float));
        e->patience_counter = 0;
        return 0;  /* continue training */
    }
    e->patience_counter++;
    return (e->patience_counter >= e->patience);  /* 0=continue, 1=stop */
}

/*
 * L7: Mixup (Zhang et al., ICLR 2018)
 * Creates convex combinations of training examples:
 *   x_new = lambda * x1 + (1-lambda) * x2
 *   y_new = lambda * y1 + (1-lambda) * y2
 * lambda ~ Beta(alpha, alpha)
 * Improves robustness and calibration.
 */
float mixup_beta_sample(float alpha) {
    /* Approximate Beta(alpha,alpha) with Gamma method */
    if (alpha <= 0.0f) return 0.5f;
    float u1 = (float)rand() / (float)RAND_MAX;
    float u2 = (float)rand() / (float)RAND_MAX;
    /* Simple approximation for Beta(alpha,alpha) when alpha is small */
    if (u1 < 1e-6f) u1 = 1e-6f;
    if (u2 < 1e-6f) u2 = 1e-6f;
    float g1 = powf(-logf(u1), alpha);
    float g2 = powf(-logf(u2), alpha);
    return g1 / (g1 + g2);
}

void mixup_augment(const float *x1, const float *x2, const float *y1, const float *y2,
                   float *x_out, float *y_out, int n, float alpha) {
    float lam = mixup_beta_sample(alpha);
    for (int i = 0; i < n; i++) {
        x_out[i] = lam * x1[i] + (1.0f - lam) * x2[i];
        y_out[i] = lam * y1[i] + (1.0f - lam) * y2[i];
    }
}
