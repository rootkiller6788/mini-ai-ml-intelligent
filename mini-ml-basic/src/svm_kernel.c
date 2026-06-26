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
    /* For binary: delegate to linear SVM.  Multi-class OvR is stored
       as a single model struct when n_classes > 2 – weights has shape
       (n_classes, n_features).  This stub shows the dispatching.   */
    if (m->n_classes == 2) {
        svm_fit_linear(m, X, y, n_samples, lr, epochs);
        return;
    }
    /* multi-class OvR: reallocate weights = (n_classes * n_features) */
    size_t d = m->n_features;
    free(m->weights);
    m->weights = (double *)calloc((size_t)m->n_classes * d, sizeof(double));
    for (int c = 0; c < m->n_classes; ++c) {
        /* treat class c as +1, others -1 */
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
   SMO – stub overview
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
    (void)m; (void)i1; (void)i2; (void)X; (void)y; (void)n;
    /* Detailed SMO step not implemented in this stub;
       returns 0 to signal "no progress" and triggers full
       gradient-descent SVM as fallback.                     */
    return 0;
}

bool smo_examine_example(SMOModel *m, int i2,
                         const double *X, const double *y,
                         size_t n, double C, double tol) {
    (void)m; (void)i2; (void)X; (void)y; (void)n; (void)C; (void)tol;
    return false;
}
