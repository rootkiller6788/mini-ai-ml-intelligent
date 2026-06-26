#include "model_selection.h"
#include "linear_models.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* Wrapper functions for cross-validation with LinearModel */
static void *lm_wrap_create(void) {
    LinearModel *m = (LinearModel *)malloc(sizeof(LinearModel));
    *m = lm_create(2);
    return m;
}

static void lm_wrap_fit(void *model, const double *X_train,
                         const double *y_train, size_t n_train) {
    LinearModel *m = (LinearModel *)model;
    lm_fit_sgd(m, X_train, y_train, n_train, 0.01, 200, 16);
}

static double lm_wrap_predict(const void *model, const double *x) {
    const LinearModel *m = (const LinearModel *)model;
    return lm_predict(m, x);
}

static void lm_wrap_destroy(void *model) {
    LinearModel *m = (LinearModel *)model;
    lm_destroy(m);
    free(m);
}

int main(void) {
    size_t n = 120, d = 2;
    double *X = (double *)malloc(n * d * sizeof(double));
    double *y = (double *)malloc(n * sizeof(double));
    int    *y_cls = (int *)malloc(n * sizeof(int));

    /* Generate data: y = 3*x1 + 2*x2 + noise + 5 */
    for (size_t i = 0; i < n; ++i) {
        double x1 = ((double)i - 60.0) / 10.0;
        double x2 = (double)(rand() % 100) / 100.0;
        X[i * d]     = x1;
        X[i * d + 1] = x2;
        y[i] = 3.0 * x1 + 2.0 * x2 + 5.0 + ((double)rand() / RAND_MAX - 0.5);
        y_cls[i] = (y[i] > 5.0) ? 1 : 0;
    }

    /* ── Train/Test Split ── */
    double *X_train, *X_test, *y_train, *y_test;
    size_t n_train, n_test;
    train_test_split_f64(X, y, n, d, 0.2, 42,
                           &X_train, &X_test, &y_train, &y_test,
                           &n_train, &n_test);
    printf("Train/Test split: train=%zu test=%zu\n", n_train, n_test);

    /* ── Regression Metrics ── */
    double y_pred[24];  /* n_test ≈ 24 */
    LinearModel lm = lm_create(d);
    lm_fit_sgd(&lm, X_train, y_train, n_train, 0.01, 300, 16);
    for (size_t i = 0; i < n_test; ++i)
        y_pred[i] = lm_predict(&lm, X_test + i * d);

    printf("MSE:  %.4f\n", metric_mse(y_test, y_pred, n_test));
    printf("RMSE: %.4f\n", metric_rmse(y_test, y_pred, n_test));
    printf("MAE:  %.4f\n", metric_mae(y_test, y_pred, n_test));
    printf("R²:   %.4f\n", metric_r2(y_test, y_pred, n_test));
    lm_destroy(&lm);

    /* ── K-Fold Cross-Validation ── */
    KFoldCV cv = kfold_create(5, true, 42);
    double avg_mse = kfold_regression(&cv,
                                       lm_wrap_create, lm_wrap_fit,
                                       lm_wrap_predict, lm_wrap_destroy,
                                       X, y, n, d, metric_mse);
    printf("5-fold CV MSE: %.4f\n", avg_mse);
    kfold_destroy(&cv);

    /* ── Confusion Matrix ── */
    ConfusionMatrix cm = cm_create();
    LogisticModel lr = logreg_create(d, 2);
    logreg_fit(&lr, X_train, (double*)y_cls, n_train, 0.1, 200);
    for (size_t i = 0; i < n_test; ++i) {
        int pred = logreg_predict(&lr, X_test + i * d);
        cm_record(&cm, y_cls[n_train + i], pred);
    }
    cm_print(&cm);
    logreg_destroy(&lr);

    free(X); free(y); free(y_cls);
    free(X_train); free(X_test); free(y_train); free(y_test);
    return 0;
}