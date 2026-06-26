#ifndef LINEAR_MODELS_H
#define LINEAR_MODELS_H

#include <stdbool.h>
#include <stddef.h>

/* ──────────────────────────────────────────────
   Linear Regression (Normal Equation)
   ────────────────────────────────────────────── */
typedef struct {
    double *coefs;    /* slope / weight vector, shape [n_features]       */
    double  bias;     /* intercept term                                  */
    size_t  n_features;
} LinearModel;

LinearModel  lm_create(size_t n_features);
void         lm_destroy(LinearModel *m);
void         lm_fit_normal_eq(LinearModel *m,
                              const double *X,    /* (n_samples, n_features) row-major */
                              const double *y,    /* (n_samples,)                     */
                              size_t n_samples);
void         lm_fit_sgd(LinearModel *m,
                        const double *X, const double *y,
                        size_t n_samples,
                        double lr, size_t epochs, size_t batch_size);
void         lm_fit_momentum(LinearModel *m,
                             const double *X, const double *y,
                             size_t n_samples,
                             double lr, double momentum, size_t epochs);
double       lm_predict(const LinearModel *m, const double *x);
void         lm_predict_batch(const LinearModel *m, const double *X, double *out, size_t n);

/* ──────────────────────────────────────────────
   Logistic Regression
   ────────────────────────────────────────────── */
typedef struct {
    double *weights;   /* shape [n_features]          */
    double  bias;
    size_t  n_features;
    int     classes;   /* 2 → binary ; >2 → multi     */
} LogisticModel;

LogisticModel  logreg_create(size_t n_features, int classes);
void           logreg_destroy(LogisticModel *m);
double         logreg_sigmoid(double z);
double         logreg_bce_loss(const LogisticModel *m,
                               const double *X, const double *y,
                               size_t n_samples);
void           logreg_fit(LogisticModel *m,
                          const double *X, const double *y,
                          size_t n_samples,
                          double lr, size_t epochs);
int            logreg_predict(const LogisticModel *m, const double *x);
double         logreg_decision_boundary(const LogisticModel *m, const double *x);

/* ──────────────────────────────────────────────
   Softmax (Multi-Class)
   ────────────────────────────────────────────── */
typedef struct {
    double *W;            /* (n_features, n_classes) column-major */
    double *b;            /* (n_classes,)                        */
    size_t  n_features;
    size_t  n_classes;
} SoftmaxModel;

SoftmaxModel  softmax_create(size_t n_features, size_t n_classes);
void          softmax_destroy(SoftmaxModel *m);
void          softmax_forward(const SoftmaxModel *m, const double *x, double *probs);
double        softmax_cross_entropy(const SoftmaxModel *m,
                                    const double *X, const int *y,
                                    size_t n_samples);
void          softmax_fit(SoftmaxModel *m,
                          const double *X, const int *y,
                          size_t n_samples,
                          double lr, size_t epochs);
int           softmax_predict(const SoftmaxModel *m, const double *x);

#endif /* LINEAR_MODELS_H */
