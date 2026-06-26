#include "linear_models.h"

#include <math.h>
#include <stdlib.h>

/* ──────────────────────────────────────────────
   helpers
   ────────────────────────────────────────────── */

/* ──────────────────────────────────────────────
   LinearModel
   ────────────────────────────────────────────── */

LinearModel lm_create(size_t n_features) {
    LinearModel m;
    m.n_features = n_features;
    m.coefs = (double *)calloc(n_features, sizeof(double));
    m.bias  = 0.0;
    return m;
}

void lm_destroy(LinearModel *m) {
    free(m->coefs);
    m->coefs = NULL;
    m->n_features = 0;
}

double lm_predict(const LinearModel *m, const double *x) {
    double y = m->bias;
    for (size_t j = 0; j < m->n_features; ++j)
        y += m->coefs[j] * x[j];
    return y;
}

void lm_predict_batch(const LinearModel *m, const double *X, double *out, size_t n) {
    for (size_t i = 0; i < n; ++i)
        out[i] = lm_predict(m, X + i * m->n_features);
}

/* ── Normal Equation : θ = (XᵀX)⁻¹ Xᵀy ── */

static bool mat_inv_2d(double A[2][2]) {
    double det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
    if (fabs(det) < 1e-12) return false;
    double inv_det = 1.0 / det;
    double tmp00 = A[0][0];
    A[0][0] =  A[1][1] * inv_det;
    A[0][1] = -A[0][1] * inv_det;
    A[1][0] = -A[1][0] * inv_det;
    A[1][1] =  tmp00 * inv_det;
    return true;
}

void lm_fit_normal_eq(LinearModel *m,
                      const double *X, const double *y,
                      size_t n_samples) {
    size_t d = m->n_features;
    /* For simplicity we implement only d==1 so we can demonstrate the idea.
       General case needs (d+1)×(d+1) matrix inversion.          */
    if (d != 1) {
        /* fallback: gradient descent for d > 1 */
        lm_fit_sgd(m, X, y, n_samples, 0.01, 1000, 32);
        return;
    }
    double sum_x = 0.0, sum_y = 0.0, sum_xx = 0.0, sum_xy = 0.0;
    for (size_t i = 0; i < n_samples; ++i) {
        double xi = X[i];
        double yi = y[i];
        sum_x  += xi;
        sum_y  += yi;
        sum_xx += xi * xi;
        sum_xy += xi * yi;
    }
    double A[2][2] = {{(double)n_samples, sum_x}, {sum_x, sum_xx}};
    double rhs[2]   = {sum_y, sum_xy};
    if (mat_inv_2d(A)) {
        m->bias       = A[0][0] * rhs[0] + A[0][1] * rhs[1];
        m->coefs[0]   = A[1][0] * rhs[0] + A[1][1] * rhs[1];
    }
}

/* ── Stochastic Gradient Descent ── */

void lm_fit_sgd(LinearModel *m,
                const double *X, const double *y,
                size_t n_samples,
                double lr, size_t epochs, size_t batch_size) {
    if (batch_size == 0 || batch_size > n_samples) batch_size = n_samples;
    size_t d = m->n_features;
    for (size_t e = 0; e < epochs; ++e) {
        /* shuffle indices */
        size_t *idx = (size_t *)malloc(n_samples * sizeof(size_t));
        for (size_t i = 0; i < n_samples; ++i) idx[i] = i;
        for (size_t i = n_samples - 1; i > 0; --i) {
            size_t j = (size_t)rand() % (i + 1);
            size_t t = idx[i]; idx[i] = idx[j]; idx[j] = t;
        }
        for (size_t b = 0; b < n_samples; b += batch_size) {
            double *g_w = (double *)calloc(d, sizeof(double));
            double  g_b = 0.0;
            size_t bs = (b + batch_size > n_samples) ? n_samples - b : batch_size;
            for (size_t k = 0; k < bs; ++k) {
                size_t i   = idx[b + k];
                double err = lm_predict(m, X + i * d) - y[i];
                for (size_t j = 0; j < d; ++j)
                    g_w[j] += err * X[i * d + j];
                g_b += err;
            }
            for (size_t j = 0; j < d; ++j)
                m->coefs[j] -= lr * g_w[j] / (double)bs;
            m->bias -= lr * g_b / (double)bs;
            free(g_w);
        }
        free(idx);
    }
}

/* ── Momentum ── */

void lm_fit_momentum(LinearModel *m,
                     const double *X, const double *y,
                     size_t n_samples,
                     double lr, double momentum, size_t epochs) {
    size_t d = m->n_features;
    double *v_w = (double *)calloc(d, sizeof(double));
    double  v_b = 0.0;
    for (size_t e = 0; e < epochs; ++e) {
        double *g_w = (double *)calloc(d, sizeof(double));
        double  g_b = 0.0;
        for (size_t i = 0; i < n_samples; ++i) {
            double err = lm_predict(m, X + i * d) - y[i];
            for (size_t j = 0; j < d; ++j)
                g_w[j] += err * X[i * d + j];
            g_b += err;
        }
        for (size_t j = 0; j < d; ++j) {
            v_w[j] = momentum * v_w[j] - lr * g_w[j] / (double)n_samples;
            m->coefs[j] += v_w[j];
        }
        v_b = momentum * v_b - lr * g_b / (double)n_samples;
        m->bias += v_b;
        free(g_w);
    }
    free(v_w);
}

/* ──────────────────────────────────────────────
   LogisticModel
   ────────────────────────────────────────────── */

double logreg_sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}

LogisticModel logreg_create(size_t n_features, int classes) {
    LogisticModel m;
    m.n_features = n_features;
    m.classes    = classes;
    m.weights    = (double *)calloc(n_features, sizeof(double));
    m.bias       = 0.0;
    return m;
}

void logreg_destroy(LogisticModel *m) {
    free(m->weights);
    m->weights = NULL;
}

double logreg_bce_loss(const LogisticModel *m,
                       const double *X, const double *y,
                       size_t n_samples) {
    double loss = 0.0;
    size_t d = m->n_features;
    for (size_t i = 0; i < n_samples; ++i) {
        double z = m->bias;
        for (size_t j = 0; j < d; ++j) z += m->weights[j] * X[i * d + j];
        double p = logreg_sigmoid(z);
        double yi = y[i];
        if (yi > 0.5)
            loss -= log(p + 1e-12);
        else
            loss -= log(1.0 - p + 1e-12);
    }
    return loss / (double)n_samples;
}

void logreg_fit(LogisticModel *m,
                const double *X, const double *y,
                size_t n_samples,
                double lr, size_t epochs) {
    size_t d = m->n_features;
    for (size_t e = 0; e < epochs; ++e) {
        for (size_t i = 0; i < n_samples; ++i) {
            double z = m->bias;
            for (size_t j = 0; j < d; ++j) z += m->weights[j] * X[i * d + j];
            double p = logreg_sigmoid(z);
            double err = p - y[i];
            for (size_t j = 0; j < d; ++j)
                m->weights[j] -= lr * err * X[i * d + j];
            m->bias -= lr * err;
        }
    }
}

int logreg_predict(const LogisticModel *m, const double *x) {
    double z = m->bias;
    for (size_t j = 0; j < m->n_features; ++j) z += m->weights[j] * x[j];
    return logreg_sigmoid(z) >= 0.5 ? 1 : 0;
}

double logreg_decision_boundary(const LogisticModel *m, const double *x) {
    return logreg_predict(m, x) ? 1.0 : 0.0;
}

/* ──────────────────────────────────────────────
   SoftmaxModel
   ────────────────────────────────────────────── */

SoftmaxModel softmax_create(size_t n_features, size_t n_classes) {
    SoftmaxModel m;
    m.n_features = n_features;
    m.n_classes  = n_classes;
    m.W = (double *)calloc(n_features * n_classes, sizeof(double));
    m.b = (double *)calloc(n_classes, sizeof(double));
    return m;
}

void softmax_destroy(SoftmaxModel *m) {
    free(m->W);
    free(m->b);
    m->W = NULL;
    m->b = NULL;
}

void softmax_forward(const SoftmaxModel *m, const double *x, double *probs) {
    size_t d = m->n_features, c = m->n_classes;
    double sum = 0.0;
    for (size_t k = 0; k < c; ++k) {
        double z = m->b[k];
        for (size_t j = 0; j < d; ++j)
            z += m->W[j * c + k] * x[j];
        probs[k] = exp(z);
        sum += probs[k];
    }
    for (size_t k = 0; k < c; ++k)
        probs[k] /= sum;
}

double softmax_cross_entropy(const SoftmaxModel *m,
                             const double *X, const int *y,
                             size_t n_samples) {
    size_t d = m->n_features, c = m->n_classes;
    double loss = 0.0;
    for (size_t i = 0; i < n_samples; ++i) {
        double *probs = (double *)malloc(c * sizeof(double));
        softmax_forward(m, X + i * d, probs);
        loss -= log(probs[y[i]] + 1e-12);
        free(probs);
    }
    return loss / (double)n_samples;
}

void softmax_fit(SoftmaxModel *m,
                 const double *X, const int *y,
                 size_t n_samples,
                 double lr, size_t epochs) {
    size_t d = m->n_features, c = m->n_classes;
    for (size_t e = 0; e < epochs; ++e) {
        for (size_t i = 0; i < n_samples; ++i) {
            double *probs = (double *)malloc(c * sizeof(double));
            softmax_forward(m, X + i * d, probs);
            probs[y[i]] -= 1.0;  /* gradient of cross-entropy */
            for (size_t k = 0; k < c; ++k) {
                for (size_t j = 0; j < d; ++j)
                    m->W[j * c + k] -= lr * probs[k] * X[i * d + j];
                m->b[k] -= lr * probs[k];
            }
            free(probs);
        }
    }
}

int softmax_predict(const SoftmaxModel *m, const double *x) {
    size_t c = m->n_classes;
    double *probs = (double *)malloc(c * sizeof(double));
    softmax_forward(m, x, probs);
    int best = 0;
    for (size_t k = 1; k < c; ++k)
        if (probs[k] > probs[best]) best = (int)k;
    free(probs);
    return best;
}
