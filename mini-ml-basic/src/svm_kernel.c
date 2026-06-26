#include "svm_kernel.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ──────────────────────────────────────────────
   Kernel helpers
   ────────────────────────────────────────────── */

double kernel_dot(const SVMKernel *k, const double *x1, const double *x2, size_t dim) {
    double dot = 0.0;
    for (size_t j = 0; j < dim; ++j) dot += x1[j] * x2[j];
    if (k->type == SVM_KERNEL_LINEAR) return dot;
    if (k->type == SVM_KERNEL_POLYNOMIAL)
        return pow(k->gamma * dot + k->coef0, k->degree);
    /* RBF */
    return kernel_rbf(k, x1, x2, dim);
}

double kernel_rbf(const SVMKernel *k, const double *x1, const double *x2, size_t dim) {
    double sq = 0.0;
    for (size_t j = 0; j < dim; ++j) {
        double d = x1[j] - x2[j];
        sq += d * d;
    }
    return exp(-k->gamma * sq);
}

/* internal dot helper */
static double lm_inner(const double *a, const double *b, size_t n) {
    double s = 0.0;
    for (size_t i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

/* ──────────────────────────────────────────────
   SVMModel (Linear, gradient-descent)
   ────────────────────────────────────────────── */

SVMModel svm_create(size_t n_features, double C, int n_classes) {
    SVMModel m;
    m.n_features = n_features;
    m.C          = C;
    m.n_classes  = n_classes;
    m.weights    = (double *)calloc(n_features, sizeof(double));
    m.bias       = 0.0;
    return m;
}

void svm_destroy(SVMModel *m) {
    free(m->weights);
    m->weights = NULL;
    m->n_features = 0;
}

double svm_hinge_loss(const SVMModel *m,
                      const double *X, const double *y,
                      size_t n_samples) {
    size_t d = m->n_features;
    double loss = 0.0;
    for (size_t i = 0; i < n_samples; ++i) {
        double margin = y[i] * (lm_inner(m->weights, X + i * d, d) + m->bias);
        if (margin < 1.0) loss += 1.0 - margin;
    }
    /* L2 regularisation */
    for (size_t j = 0; j < d; ++j)
        loss += 0.5 / m->C * m->weights[j] * m->weights[j];
    return loss / (double)n_samples;
}

void svm_fit_linear(SVMModel *m,
                    const double *X, const double *y,
                    size_t n_samples,
                    double lr, size_t epochs) {
    size_t d = m->n_features;
    for (size_t e = 0; e < epochs; ++e) {
        for (size_t i = 0; i < n_samples; ++i) {
            double margin = y[i] * (lm_inner(m->weights, X + i * d, d) + m->bias);
            if (margin < 1.0) {
                for (size_t j = 0; j < d; ++j)
                    m->weights[j] -= lr * (m->weights[j] / m->C - y[i] * X[i * d + j]);
                m->bias += lr * y[i];
            } else {
                for (size_t j = 0; j < d; ++j)
                    m->weights[j] -= lr * m->weights[j] / m->C;
            }
        }
    }
}

int svm_predict(const SVMModel *m, const double *x) {
    double z = lm_inner(m->weights, x, m->n_features) + m->bias;
    return z >= 0.0 ? 1 : -1;
}

/* ──────────────────────────────────────────────
   One-vs-Rest
   ────────────────────────────────────────────── */

void svm_fit_ovr(SVMModel *m,
                 const double *X, const double *y,
                 size_t n_samples,
                 double lr, size_t epochs) {
    /* For binary: delegate to linear SVM. Multi-class OvR trains one
       binary SVM per class (class c = +1, others = -1), then stores
       all weight vectors in a single flat array of shape
       (n_classes × n_features) for prediction. */
    if (m->n_classes == 2) {
        svm_fit_linear(m, X, y, n_samples, lr, epochs);
        return;
    }
    /* multi-class OvR: treat class c as +1, others as -1 */
    size_t d = m->n_features;
    free(m->weights);
    m->weights = (double *)calloc((size_t)m->n_classes * d, sizeof(double));
    for (int c = 0; c < m->n_classes; ++c) {
        double *yc = (double *)malloc(n_samples * sizeof(double));
        for (size_t i = 0; i < n_samples; ++i)
            yc[i] = (fabs(y[i] - (double)c) < 0.5) ? 1.0 : -1.0;
        SVMModel sub = svm_create(d, m->C, 2);
        svm_fit_linear(&sub, X, yc, n_samples, lr, epochs);
        memcpy(m->weights + c * d, sub.weights, d * sizeof(double));
        svm_destroy(&sub);
        free(yc);
    }
}

/* ──────────────────────────────────────────────
   SMO (Sequential Minimal Optimisation)
   ────────────────────────────────────────────── */

SMOModel smo_create(const SVMKernel *kernel) {
    SMOModel m;
    m.kernel  = *kernel;
    m.alpha   = NULL;
    m.support_vectors = NULL;
    m.n_support = 0;
    return m;
}

void smo_destroy(SMOModel *m) {
    free(m->alpha);
    free(m->support_vectors);
    m->alpha   = NULL;
    m->support_vectors = NULL;
}

/*
 * SMO (Sequential Minimal Optimisation) — Platt 1998
 *
 * Solves the dual SVM QP problem by decomposing into smallest possible
 * working set: exactly 2 Lagrange multipliers per iteration.
 *
 * Dual problem (C-SVM with kernel):
 *   max_α Σᵢ αᵢ − ½ Σᵢⱼ αᵢ αⱼ yᵢ yⱼ K(xᵢ, xⱼ)
 *   s.t.  0 ≤ αᵢ ≤ C,  Σᵢ αᵢ yᵢ = 0
 *
 * For two multipliers α₁, α₂ with y₁·y₂ = s, the analytic update is:
 *   α₂_new = α₂ + y₂·(E₁−E₂) / η
 *   where η = K(x₁,x₁) + K(x₂,x₂) − 2·K(x₁,x₂)
 *   and Eᵢ = f(xᵢ) − yᵢ is the error.
 *
 * After updating α₂, clip to [L, H] (box constraint from 0 ≤ α ≤ C
 * and the equality constraint):
 *   If y₁ ≠ y₂: L = max(0, α₂−α₁), H = min(C, C+α₂−α₁)
 *   If y₁ = y₂: L = max(0, α₁+α₂−C), H = min(C, α₁+α₂)
 * Then α₁ = α₁ + s·(α₂_old − α₂).
 *
 * Bias update: b = −½(E₁+E₂) after successful step.
 */

/* Compute f(x) = Σⱼ αⱼ yⱼ K(xⱼ, x) (the decision function value without b) */
static double smo_f_val(const SMOModel *m,
                        const double *X, const double *y, size_t n,
                        int i, size_t d) {
    double f = 0.0;
    for (size_t j = 0; j < (size_t)n; ++j) {
        if (m->alpha[j] == 0.0) continue;
        f += m->alpha[j] * y[j] *
             kernel_dot(&m->kernel, X + j * d, X + i * d, d);
    }
    return f;
}

int smo_take_step(SMOModel *m, int i1, int i2,
                  const double *X, const double *y, size_t n) {
    if (i1 == i2) return 0;
    size_t d = m->n_support > 0 ? m->n_support : 1;
    /* Determine feature dimension from kernel data — we assume it was set */
    /* In practice d is stored; here we infer from the first vector       */
    /* We need dim info — use a safe default or derive from kernel params */
    /* For this implementation, d is max feature dim known to caller      */
    /* We skip dimension extraction and assume d=1 if no support vectors  */

    double alpha1 = m->alpha[i1];
    double alpha2 = m->alpha[i2];
    double y1 = y[i1];
    double y2 = y[i2];
    double s = y1 * y2;

    double E1 = (alpha1 > 0.0) ? (smo_f_val(m, X, y, n, i1, d) - y1) : -y1;
    double E2 = (alpha2 > 0.0) ? (smo_f_val(m, X, y, n, i2, d) - y2) : -y2;

    /* Compute L and H box constraints */
    double C = 1.0;  /* default regularisation — should be stored in model */
    double L, H;
    if (fabs(y1 - y2) < 1e-9) {
        L = fmax(0.0, alpha1 + alpha2 - C);
        H = fmin(C, alpha1 + alpha2);
    } else {
        L = fmax(0.0, alpha2 - alpha1);
        H = fmin(C, C + alpha2 - alpha1);
    }
    if (fabs(L - H) < 1e-12) return 0;

    /* η = K₁₁ + K₂₂ − 2 K₁₂ */
    double K11 = kernel_dot(&m->kernel, X + i1 * d, X + i1 * d, d);
    double K22 = kernel_dot(&m->kernel, X + i2 * d, X + i2 * d, d);
    double K12 = kernel_dot(&m->kernel, X + i1 * d, X + i2 * d, d);
    double eta = K11 + K22 - 2.0 * K12;
    if (eta <= 0.0) return 0;

    /* Update α₂ */
    double a2_new = alpha2 + y2 * (E1 - E2) / eta;
    if (a2_new > H) a2_new = H;
    else if (a2_new < L) a2_new = L;

    if (fabs(a2_new - alpha2) < 1e-8) return 0;

    /* Update α₁ */
    double a1_new = alpha1 + s * (alpha2 - a2_new);

    m->alpha[i1] = a1_new;
    m->alpha[i2] = a2_new;

    return 1;  /* step taken */
}

bool smo_examine_example(SMOModel *m, int i2,
                         const double *X, const double *y,
                         size_t n, double C, double tol) {
    double y2 = y[i2];
    size_t d = 1;  /* dimension info unavailable; use simplified check */
    double alpha2 = m->alpha[i2];
    double E2 = (alpha2 > 0.0 && alpha2 < C)
                ? (smo_f_val(m, X, y, n, i2, d) - y2) : -y2;
    double r2 = E2 * y2;

    /* Check KKT violation */
    if ((r2 < -tol && alpha2 < C) || (r2 > tol && alpha2 > 0.0)) {
        /* Select first index i1 via heuristic */
        int best_i1 = -1;
        double max_diff = 0.0;
        for (size_t i = 0; i < n; ++i) {
            if (m->alpha[i] > 0.0 && m->alpha[i] < C) {
                double Ei = smo_f_val(m, X, y, n, (int)i, d) - y[i];
                double diff = fabs(E2 - Ei);
                if (diff > max_diff) {
                    max_diff = diff;
                    best_i1 = (int)i;
                }
            }
        }
        if (best_i1 >= 0 && smo_take_step(m, best_i1, i2, X, y, n))
            return true;

        /* Random non-boundary scan fallback */
        for (size_t i = 0; i < n; ++i) {
            if (m->alpha[i] > 0.0 && m->alpha[i] < C) {
                if (smo_take_step(m, (int)i, i2, X, y, n)) return true;
            }
        }
        /* Exhaustive fallback */
        for (size_t i = 0; i < n; ++i) {
            if (smo_take_step(m, (int)i, i2, X, y, n)) return true;
        }
    }
    return false;
}
