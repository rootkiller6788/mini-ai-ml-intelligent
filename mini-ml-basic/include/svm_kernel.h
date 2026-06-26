#ifndef SVM_KERNEL_H
#define SVM_KERNEL_H

#include <stdbool.h>
#include <stddef.h>

/* ──────────────────────────────────────────────
   Kernel types
   ────────────────────────────────────────────── */
typedef enum {
    SVM_KERNEL_LINEAR = 0,
    SVM_KERNEL_POLYNOMIAL,
    SVM_KERNEL_RBF
} SVMKernelType;

typedef struct {
    SVMKernelType type;
    double        degree;     /* polynomial degree         */
    double        gamma;      /* RBF γ (1/n_features default) */
    double        coef0;      /* polynomial r              */
} SVMKernel;

/* ──────────────────────────────────────────────
   Linear SVM (gradient-descent)
   ────────────────────────────────────────────── */
typedef struct {
    double *weights;          /* (n_features,)             */
    double  bias;
    size_t  n_features;
    double  C;                /* regularisation             */
    int     n_classes;        /* 2 → binary ; >2 → OvR      */
} SVMModel;

SVMModel  svm_create(size_t n_features, double C, int n_classes);
void      svm_destroy(SVMModel *m);
double    svm_hinge_loss(const SVMModel *m,
                         const double *X, const double *y,
                         size_t n_samples);
void      svm_fit_linear(SVMModel *m,
                         const double *X, const double *y,
                         size_t n_samples,
                         double lr, size_t epochs);
int       svm_predict(const SVMModel *m, const double *x);

/* ──────────────────────────────────────────────
   SMO (Sequential Minimal Optimisation) – overview stub
   ────────────────────────────────────────────── */
typedef struct {
    double *alpha;            /* Lagrange multipliers       */
    double *support_vectors;  /* (n_sv, n_features)         */
    size_t  n_support;
    SVMKernel kernel;
} SMOModel;

SMOModel  smo_create(const SVMKernel *kernel);
void      smo_destroy(SMOModel *m);
int       smo_take_step(SMOModel *m, int i1, int i2,
                        const double *X, const double *y, size_t n);
bool      smo_examine_example(SMOModel *m, int i2,
                              const double *X, const double *y,
                              size_t n, double C, double tol);

/* ──────────────────────────────────────────────
   Kernel helpers
   ────────────────────────────────────────────── */
double  kernel_dot(const SVMKernel *k, const double *x1, const double *x2, size_t dim);
double  kernel_rbf(const SVMKernel *k, const double *x1, const double *x2, size_t dim);

/* ──────────────────────────────────────────────
   One-vs-Rest multi-class SVM
   ────────────────────────────────────────────── */
void    svm_fit_ovr(SVMModel *m,
                    const double *X, const double *y,
                    size_t n_samples,
                    double lr, size_t epochs);

#endif /* SVM_KERNEL_H */
