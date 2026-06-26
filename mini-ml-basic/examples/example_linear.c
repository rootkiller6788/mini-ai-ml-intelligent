#include "linear_models.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(void) {
    /* synthetic data: y = 3*x + 2 + noise */
    size_t n = 100;
    double *X = (double *)malloc(n * sizeof(double));
    double *y = (double *)malloc(n * sizeof(double));
    for (size_t i = 0; i < n; ++i) {
        double xi = ((double)i - 50.0) / 10.0;
        X[i] = xi;
        y[i] = 3.0 * xi + 2.0 + ((double)rand() / RAND_MAX - 0.5) * 2.0;
    }

    /* ── Linear Regression: Normal Equation ── */
    LinearModel lm = lm_create(1);
    lm_fit_normal_eq(&lm, X, y, n);
    printf("Normal Eq:  bias=%.4f  coef=%.4f\n", lm.bias, lm.coefs[0]);

    /* ── Linear Regression: SGD ── */
    LinearModel lm_sgd = lm_create(1);
    lm_fit_sgd(&lm_sgd, X, y, n, 0.01, 500, 16);
    printf("SGD:        bias=%.4f  coef=%.4f\n", lm_sgd.bias, lm_sgd.coefs[0]);

    /* ── Linear Regression: Momentum ── */
    LinearModel lm_mom = lm_create(1);
    lm_fit_momentum(&lm_mom, X, y, n, 0.01, 0.9, 500);
    printf("Momentum:   bias=%.4f  coef=%.4f\n", lm_mom.bias, lm_mom.coefs[0]);

    /* ── Logistic Regression ── */
    double *y_bin = (double *)malloc(n * sizeof(double));
    for (size_t i = 0; i < n; ++i) y_bin[i] = (y[i] > 2.0 + 3.0 * X[i]) ? 1.0 : 0.0;

    LogisticModel lr = logreg_create(1, 2);
    logreg_fit(&lr, X, y_bin, n, 0.1, 200);
    printf("LogReg:     bias=%.4f  weight=%.4f\n", lr.bias, lr.weights[0]);

    /* ── Softmax (3 classes) ── */
    double *X2 = (double *)malloc(n * 2 * sizeof(double));
    int    *y3 = (int    *)malloc(n * sizeof(int));
    for (size_t i = 0; i < n; ++i) {
        double x1 = ((double)i - 50.0) / 10.0;
        double x2 = (double)(rand()) / RAND_MAX;
        X2[i * 2]     = x1;
        X2[i * 2 + 1] = x2;
        y3[i] = (int)(((size_t)(x1 * 3.0 + x2 * 5.0)) % 3);
    }
    SoftmaxModel sm = softmax_create(2, 3);
    softmax_fit(&sm, X2, y3, n, 0.01, 300);
    double test[2] = {1.0, 0.5};
    int    cls = softmax_predict(&sm, test);
    printf("Softmax:    pred(% .1f,% .1f)=%d\n", test[0], test[1], cls);

    lm_destroy(&lm);
    lm_destroy(&lm_sgd);
    lm_destroy(&lm_mom);
    logreg_destroy(&lr);
    softmax_destroy(&sm);
    free(X); free(y); free(y_bin); free(X2); free(y3);
    return 0;
}
