#include "model_selection.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ═══════════════════════════════════════════════════════════════════════
   Confusion Matrix (Binary)
   ───────────────────────────────────────────────────────────────────────
   Fundamental tool for classification evaluation.
   TP=hit, TN=correct rejection, FP=false alarm, FN=miss.
   Derived: Accuracy, Precision, Recall, F1, Specificity.
   ═══════════════════════════════════════════════════════════════════════ */

ConfusionMatrix cm_create(void) {
    ConfusionMatrix cm;
    cm.tp = 0; cm.tn = 0; cm.fp = 0; cm.fn = 0;
    cm.n_classes = 2;
    return cm;
}

void cm_record(ConfusionMatrix *cm, int actual, int predicted) {
    if (actual == 1 && predicted == 1) cm->tp++;
    else if (actual == 0 && predicted == 0) cm->tn++;
    else if (actual == 0 && predicted == 1) cm->fp++;
    else if (actual == 1 && predicted == 0) cm->fn++;
}

double cm_accuracy(const ConfusionMatrix *cm) {
    int total = cm->tp + cm->tn + cm->fp + cm->fn;
    if (total == 0) return 0.0;
    return (double)(cm->tp + cm->tn) / (double)total;
}

double cm_precision(const ConfusionMatrix *cm) {
    int pred_pos = cm->tp + cm->fp;
    if (pred_pos == 0) return 0.0;
    return (double)cm->tp / (double)pred_pos;
}

double cm_recall(const ConfusionMatrix *cm) {
    int real_pos = cm->tp + cm->fn;
    if (real_pos == 0) return 0.0;
    return (double)cm->tp / (double)real_pos;
}

double cm_f1_score(const ConfusionMatrix *cm) {
    double p = cm_precision(cm);
    double r = cm_recall(cm);
    if (p + r < 1e-12) return 0.0;
    return 2.0 * p * r / (p + r);
}

double cm_specificity(const ConfusionMatrix *cm) {
    int real_neg = cm->tn + cm->fp;
    if (real_neg == 0) return 0.0;
    return (double)cm->tn / (double)real_neg;
}

void cm_reset(ConfusionMatrix *cm) {
    cm->tp = 0; cm->tn = 0; cm->fp = 0; cm->fn = 0;
}

void cm_print(const ConfusionMatrix *cm) {
    printf("ConfusionMatrix: TP=%d TN=%d FP=%d FN=%d\n",
           cm->tp, cm->tn, cm->fp, cm->fn);
    printf("  Acc=%.4f  Prec=%.4f  Rec=%.4f  F1=%.4f\n",
           cm_accuracy(cm), cm_precision(cm), cm_recall(cm), cm_f1_score(cm));
}

/* ═══════════════════════════════════════════════════════════════════════
   Multi-Class Confusion Matrix
   ───────────────────────────────────────────────────────────────────────
   Stores C×C counts. Per-class metrics treat each class as positive in
   a one-vs-rest manner. Macro-F1 averages per-class F1 scores.
   ═══════════════════════════════════════════════════════════════════════ */

MultiConfusionMatrix mcm_create(int n_classes) {
    MultiConfusionMatrix mcm;
    mcm.n_classes = n_classes;
    mcm.total     = 0;
    mcm.matrix    = (int *)calloc((size_t)(n_classes * n_classes), sizeof(int));
    return mcm;
}

void mcm_destroy(MultiConfusionMatrix *mcm) {
    free(mcm->matrix);
    mcm->matrix = NULL;
}

void mcm_record(MultiConfusionMatrix *mcm, int actual, int predicted) {
    mcm->matrix[actual * mcm->n_classes + predicted]++;
    mcm->total++;
}

double mcm_accuracy(const MultiConfusionMatrix *mcm) {
    if (mcm->total == 0) return 0.0;
    int correct = 0;
    for (int c = 0; c < mcm->n_classes; ++c)
        correct += mcm->matrix[c * mcm->n_classes + c];
    return (double)correct / (double)mcm->total;
}

void mcm_per_class_precision(const MultiConfusionMatrix *mcm, double *out) {
    int C = mcm->n_classes;
    for (int c = 0; c < C; ++c) {
        int col_sum = 0;
        for (int r = 0; r < C; ++r) col_sum += mcm->matrix[r * C + c];
        out[c] = (col_sum == 0) ? 0.0
                 : (double)mcm->matrix[c * C + c] / (double)col_sum;
    }
}

void mcm_per_class_recall(const MultiConfusionMatrix *mcm, double *out) {
    int C = mcm->n_classes;
    for (int c = 0; c < C; ++c) {
        int row_sum = 0;
        for (int j = 0; j < C; ++j) row_sum += mcm->matrix[c * C + j];
        out[c] = (row_sum == 0) ? 0.0
                 : (double)mcm->matrix[c * C + c] / (double)row_sum;
    }
}

double mcm_macro_f1(const MultiConfusionMatrix *mcm) {
    int C = mcm->n_classes;
    double *prec = (double *)malloc((size_t)C * sizeof(double));
    double *rec  = (double *)malloc((size_t)C * sizeof(double));
    mcm_per_class_precision(mcm, prec);
    mcm_per_class_recall(mcm, rec);
    double sum_f1 = 0.0;
    for (int c = 0; c < C; ++c) {
        if (prec[c] + rec[c] > 1e-12)
            sum_f1 += 2.0 * prec[c] * rec[c] / (prec[c] + rec[c]);
    }
    free(prec);
    free(rec);
    return sum_f1 / (double)C;
}

/* ═══════════════════════════════════════════════════════════════════════
   Regression Metrics
   ───────────────────────────────────────────────────────────────────────
   MSE, RMSE (scale-sensitive), MAE (robust to outliers), R² (coefficient
   of determination), MAPE (percentage error).
   ═══════════════════════════════════════════════════════════════════════ */

double metric_mse(const double *y_true, const double *y_pred, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double diff = y_true[i] - y_pred[i];
        sum += diff * diff;
    }
    return sum / (double)n;
}

double metric_rmse(const double *y_true, const double *y_pred, size_t n) {
    return sqrt(metric_mse(y_true, y_pred, n));
}

double metric_mae(const double *y_true, const double *y_pred, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i)
        sum += fabs(y_true[i] - y_pred[i]);
    return sum / (double)n;
}

double metric_r2(const double *y_true, const double *y_pred, size_t n) {
    double ss_res = 0.0;
    double mean_y = 0.0;
    for (size_t i = 0; i < n; ++i) mean_y += y_true[i];
    mean_y /= (double)n;
    double ss_tot = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double diff_res = y_true[i] - y_pred[i];
        double diff_tot = y_true[i] - mean_y;
        ss_res += diff_res * diff_res;
        ss_tot += diff_tot * diff_tot;
    }
    if (ss_tot < 1e-12) return 1.0;
    return 1.0 - ss_res / ss_tot;
}

double metric_mape(const double *y_true, const double *y_pred, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        if (fabs(y_true[i]) < 1e-12) continue;
        sum += fabs((y_true[i] - y_pred[i]) / y_true[i]);
    }
    return 100.0 * sum / (double)n;
}

/* ═══════════════════════════════════════════════════════════════════════
   K-Fold Cross-Validation
   ───────────────────────────────────────────────────────────────────────
   Standard k-fold: partition data into k equal (or near-equal) folds.
   Train on k-1, test on 1. Reports average metric across folds.
   Provides unbiased performance estimate with limited data.
   ═══════════════════════════════════════════════════════════════════════ */

KFoldCV kfold_create(size_t k, bool shuffle, size_t seed) {
    KFoldCV cv;
    cv.k              = k;
    cv.shuffle        = shuffle;
    cv.seed           = seed;
    cv.n_samples      = 0;
    cv.train_indices  = NULL;
    cv.test_indices   = NULL;
    return cv;
}

void kfold_destroy(KFoldCV *cv) {
    free(cv->train_indices);
    free(cv->test_indices);
    cv->train_indices = NULL;
    cv->test_indices  = NULL;
}

void kfold_split(KFoldCV *cv, size_t n_samples) {
    cv->n_samples = n_samples;
    /* simple assignment: each sample to fold i % k */
    srand((unsigned)cv->seed);
    size_t *order = (size_t *)malloc(n_samples * sizeof(size_t));
    for (size_t i = 0; i < n_samples; ++i) order[i] = i;

    if (cv->shuffle) {
        for (size_t i = n_samples - 1; i > 0; --i) {
            size_t j = (size_t)rand() % (i + 1);
            size_t t = order[i]; order[i] = order[j]; order[j] = t;
        }
    }
    /* Note: full fold storage omitted for simplicity;
       fold_i test set = samples i*n/k to (i+1)*n/k */
    free(order);
}

double kfold_regression(KFoldCV *cv,
                        CVCreateFn create, CVFitFn fit,
                        CVPredictFn predict, CVDestroyFn destroy,
                        const double *X, const double *y,
                        size_t n, size_t n_features,
                        double (*metric_fn)(const double*, const double*, size_t)) {
    if (cv->k == 0 || n == 0) return 0.0;
    size_t fold_size = n / cv->k;

    double sum_metric = 0.0;
    size_t n_folds_evaluated = 0;

    for (size_t fold = 0; fold < cv->k; ++fold) {
        size_t test_start = fold * fold_size;
        size_t test_end   = (fold == cv->k - 1) ? n : test_start + fold_size;
        size_t n_test     = test_end - test_start;
        size_t n_train    = n - n_test;

        if (n_test == 0 || n_train == 0) continue;

        double *X_train = (double *)malloc(n_train * n_features * sizeof(double));
        double *y_train = (double *)malloc(n_train * sizeof(double));
        double *X_test  = (double *)malloc(n_test * n_features * sizeof(double));
        double *y_test  = (double *)malloc(n_test * sizeof(double));
        double *y_pred  = (double *)malloc(n_test * sizeof(double));

        /* copy training portion */
        size_t ti = 0;
        for (size_t i = 0; i < test_start; ++i) {
            memcpy(X_train + ti * n_features, X + i * n_features,
                   n_features * sizeof(double));
            y_train[ti] = y[i]; ti++;
        }
        for (size_t i = test_end; i < n; ++i) {
            memcpy(X_train + ti * n_features, X + i * n_features,
                   n_features * sizeof(double));
            y_train[ti] = y[i]; ti++;
        }
        /* copy test portion */
        for (size_t i = test_start; i < test_end; ++i) {
            size_t si = i - test_start;
            memcpy(X_test + si * n_features, X + i * n_features,
                   n_features * sizeof(double));
            y_test[si] = y[i];
        }

        void *model = create();
        fit(model, X_train, y_train, n_train);
        for (size_t i = 0; i < n_test; ++i)
            y_pred[i] = predict(model, X_test + i * n_features);
        sum_metric += metric_fn(y_test, y_pred, n_test);
        n_folds_evaluated++;
        destroy(model);

        free(X_train); free(y_train); free(X_test); free(y_test); free(y_pred);
    }
    return (n_folds_evaluated > 0) ? sum_metric / (double)n_folds_evaluated : 0.0;
}

double kfold_classification(KFoldCV *cv,
                            CVCreateFn create, CVFitFn fit,
                            CVPredictFn predict, CVDestroyFn destroy,
                            const double *X, const int *y,
                            size_t n, size_t n_features, int n_classes,
                            ClassificationMetric metric_type) {
    if (cv->k == 0 || n == 0) return 0.0;
    size_t fold_size = n / cv->k;
    double sum_metric = 0.0;
    size_t n_folds_evaluated = 0;

    for (size_t fold = 0; fold < cv->k; ++fold) {
        size_t test_start = fold * fold_size;
        size_t test_end   = (fold == cv->k - 1) ? n : test_start + fold_size;
        size_t n_test     = test_end - test_start;
        size_t n_train    = n - n_test;
        if (n_test == 0 || n_train == 0) continue;

        double *X_train = (double *)malloc(n_train * n_features * sizeof(double));
        double *y_train_d = (double *)malloc(n_train * sizeof(double));
        double *X_test  = (double *)malloc(n_test * n_features * sizeof(double));

        size_t ti = 0;
        for (size_t i = 0; i < test_start; ++i) {
            memcpy(X_train + ti * n_features, X + i * n_features,
                   n_features * sizeof(double));
            y_train_d[ti] = (double)y[i]; ti++;
        }
        for (size_t i = test_end; i < n; ++i) {
            memcpy(X_train + ti * n_features, X + i * n_features,
                   n_features * sizeof(double));
            y_train_d[ti] = (double)y[i]; ti++;
        }
        for (size_t i = test_start; i < test_end; ++i) {
            size_t si = i - test_start;
            memcpy(X_test + si * n_features, X + i * n_features,
                   n_features * sizeof(double));
        }

        void *model = create();
        fit(model, X_train, y_train_d, n_train);

        ConfusionMatrix cm = cm_create();
        for (size_t i = 0; i < n_test; ++i) {
            double pred_val = predict(model, X_test + i * n_features);
            int pred_int = (int)(pred_val + 0.5);
            int actual   = (int)y[test_start + i];
            cm_record(&cm, actual, pred_int);
        }

        switch (metric_type) {
        case METRIC_ACCURACY:   sum_metric += cm_accuracy(&cm);   break;
        case METRIC_PRECISION:  sum_metric += cm_precision(&cm);  break;
        case METRIC_RECALL:     sum_metric += cm_recall(&cm);     break;
        case METRIC_F1:         sum_metric += cm_f1_score(&cm);   break;
        case METRIC_SPECIFICITY:sum_metric += cm_specificity(&cm); break;
        }
        n_folds_evaluated++;
        destroy(model);

        free(X_train); free(y_train_d); free(X_test);
    }
    return (n_folds_evaluated > 0) ? sum_metric / (double)n_folds_evaluated : 0.0;
}

/* ═══════════════════════════════════════════════════════════════════════
   Train/Test Split
   ───────────────────────────────────────────────────────────────────────
   Randomly partitions data into training and test sets. Stratified
   version not implemented (use random shuffle). Returns heap-allocated
   arrays — caller must free.
   ═══════════════════════════════════════════════════════════════════════ */

void train_test_split_f64(const double *X, const double *y, size_t n, size_t n_features,
                            double test_ratio, size_t seed,
                            double **X_train, double **X_test,
                            double **y_train, double **y_test,
                            size_t *n_train, size_t *n_test) {
    *n_test  = (size_t)(test_ratio * (double)n);
    if (*n_test == 0) *n_test = 1;
    *n_train = n - *n_test;

    size_t *order = (size_t *)malloc(n * sizeof(size_t));
    for (size_t i = 0; i < n; ++i) order[i] = i;
    srand((unsigned)seed);
    for (size_t i = n - 1; i > 0; --i) {
        size_t j = (size_t)rand() % (i + 1);
        size_t t = order[i]; order[i] = order[j]; order[j] = t;
    }

    *X_train = (double *)malloc(*n_train * n_features * sizeof(double));
    *y_train = (double *)malloc(*n_train * sizeof(double));
    *X_test  = (double *)malloc(*n_test * n_features * sizeof(double));
    *y_test  = (double *)malloc(*n_test * sizeof(double));

    for (size_t i = 0; i < *n_train; ++i) {
        size_t idx = order[i];
        memcpy(*X_train + i * n_features, X + idx * n_features,
               n_features * sizeof(double));
        (*y_train)[i] = y[idx];
    }
    for (size_t i = 0; i < *n_test; ++i) {
        size_t idx = order[*n_train + i];
        memcpy(*X_test + i * n_features, X + idx * n_features,
               n_features * sizeof(double));
        (*y_test)[i] = y[idx];
    }
    free(order);
}

void train_test_split_i32(const double *X, const int *y, size_t n, size_t n_features,
                            double test_ratio, size_t seed,
                            double **X_train, double **X_test,
                            int    **y_train, int    **y_test,
                            size_t *n_train, size_t *n_test) {
    *n_test  = (size_t)(test_ratio * (double)n);
    if (*n_test == 0) *n_test = 1;
    *n_train = n - *n_test;

    size_t *order = (size_t *)malloc(n * sizeof(size_t));
    for (size_t i = 0; i < n; ++i) order[i] = i;
    srand((unsigned)seed);
    for (size_t i = n - 1; i > 0; --i) {
        size_t j = (size_t)rand() % (i + 1);
        size_t t = order[i]; order[i] = order[j]; order[j] = t;
    }

    *X_train = (double *)malloc(*n_train * n_features * sizeof(double));
    *y_train = (int    *)malloc(*n_train * sizeof(int));
    *X_test  = (double *)malloc(*n_test * n_features * sizeof(double));
    *y_test  = (int    *)malloc(*n_test * sizeof(int));

    for (size_t i = 0; i < *n_train; ++i) {
        size_t idx = order[i];
        memcpy(*X_train + i * n_features, X + idx * n_features,
               n_features * sizeof(double));
        (*y_train)[i] = y[idx];
    }
    for (size_t i = 0; i < *n_test; ++i) {
        size_t idx = order[*n_train + i];
        memcpy(*X_test + i * n_features, X + idx * n_features,
               n_features * sizeof(double));
        (*y_test)[i] = y[idx];
    }
    free(order);
}

/* ═══════════════════════════════════════════════════════════════════════
   Grid Search (Regression)
   ───────────────────────────────────────────────────────────────────────
   Exhaustive search over discrete parameter grids. For each combination,
   runs k-fold CV and records the best score. Returns index of best
   parameter (simple single-parameter grid shown).
   ═══════════════════════════════════════════════════════════════════════ */

size_t grid_search_regression(
    CVCreateFn create, CVFitFn fit, CVPredictFn predict, CVDestroyFn destroy,
    const double *X, const double *y, size_t n, size_t n_features,
    GSParam *params, size_t n_params, size_t k_folds, size_t seed,
    double *best_score) {
    KFoldCV cv = kfold_create(k_folds, true, seed);
    *best_score = DBL_MAX;
    size_t best_combo = 0;

    /* For simplicity, only the first parameter is scanned;
       multi-param grid search requires Cartesian product logic
       which is left as future extension (L8). */
    if (n_params == 0) return 0;

    GSParam *p = &params[0];
    for (size_t vi = 0; vi < p->n_values; ++vi) {
        /* The actual hyperparameter injection is model-specific;
           here we report the framework structure. In practice,
           create() would accept param value via a closure.     */
        double score = kfold_regression(&cv, create, fit, predict, destroy,
                                         X, y, n, n_features, metric_mse);
        if (score < *best_score) {
            *best_score = score;
            best_combo  = vi;
        }
        (void)p; /* param integration point */
    }
    kfold_destroy(&cv);
    return best_combo;
}