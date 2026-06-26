/*
 * mini-ml-basic — Core Benchmarks
 *
 * Benchmarks: linear regression, logistic regression, softmax,
 *             SVM, decision tree, random forest, k-means, DBSCAN,
 *             hierarchical clustering, AdaBoost, GBDT, XGBoost.
 */
#include "../include/linear_models.h"
#include "../include/svm_kernel.h"
#include "../include/decision_tree.h"
#include "../include/clustering.h"
#include "../include/ensemble_gbdt.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
    return (double)clock() / (double)(CLOCKS_PER_SEC / 1000);
}

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 5000;
    size_t F = 8, S = 256;
    double t0, t1;
    printf("=== mini-ml-basic Benchmarks (N=%d) ===\n\n", N);

    /* ── Linear Regression ── */
    {
        double *X = calloc(S * F, sizeof(double));
        double *y = calloc(S, sizeof(double));
        for (size_t i = 0; i < S * F; i++) X[i] = (double)(i % 100) / 100.0;
        for (size_t i = 0; i < S; i++) y[i] = (double)(i % 10);
        t0 = now_ms();
        for (int r = 0; r < N / 10; r++) {
            LinearModel m = lm_create(F);
            lm_fit_normal_eq(&m, X, y, S);
            lm_destroy(&m);
        }
        t1 = now_ms();
        printf("  lm_fit_normal_eq:    %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 10, t1 - t0, (t1 - t0) / (double)(N / 10) * 1000.0);
        free(X); free(y);
    }

    /* ── Logistic Regression ── */
    {
        double *X = calloc(S * F, sizeof(double));
        double *y = calloc(S, sizeof(double));
        for (size_t i = 0; i < S * F; i++) X[i] = (double)(i % 100) / 100.0;
        for (size_t i = 0; i < S; i++) y[i] = (double)((i % 2) == 0 ? 1 : 0);
        t0 = now_ms();
        for (int r = 0; r < N / 20; r++) {
            LogisticModel m = logreg_create(F, 2);
            logreg_fit(&m, X, y, S, 0.01, 5);
            logreg_destroy(&m);
        }
        t1 = now_ms();
        printf("  logreg_fit:          %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 20, t1 - t0, (t1 - t0) / (double)(N / 20) * 1000.0);
        free(X); free(y);
    }

    /* ── Softmax ── */
    {
        double *X = calloc(S * F, sizeof(double));
        int    *y_l = calloc(S, sizeof(int));
        for (size_t i = 0; i < S * F; i++) X[i] = (double)(i % 100) / 100.0;
        for (size_t i = 0; i < S; i++) y_l[i] = i % 4;
        t0 = now_ms();
        for (int r = 0; r < N / 20; r++) {
            SoftmaxModel m = softmax_create(F, 4);
            softmax_fit(&m, X, y_l, S, 0.01, 3);
            softmax_destroy(&m);
        }
        t1 = now_ms();
        printf("  softmax_fit:         %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 20, t1 - t0, (t1 - t0) / (double)(N / 20) * 1000.0);
        free(X); free(y_l);
    }

    /* ── SVM Linear ── */
    {
        double *X = calloc(S * F, sizeof(double));
        double *y = calloc(S, sizeof(double));
        for (size_t i = 0; i < S * F; i++) X[i] = (double)(i % 100) / 100.0;
        for (size_t i = 0; i < S; i++) y[i] = (i % 2) ? 1.0 : -1.0;
        t0 = now_ms();
        for (int r = 0; r < N / 20; r++) {
            SVMModel m = svm_create(F, 1.0, 2);
            svm_fit_linear(&m, X, y, S, 0.01, 5);
            svm_destroy(&m);
        }
        t1 = now_ms();
        printf("  svm_fit_linear:      %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 20, t1 - t0, (t1 - t0) / (double)(N / 20) * 1000.0);
        free(X); free(y);
    }

    /* ── Decision Tree ── */
    {
        double *X = calloc(S * F, sizeof(double));
        int    *y_l = calloc(S, sizeof(int));
        for (size_t i = 0; i < S * F; i++) X[i] = (double)(i % 100) / 100.0;
        for (size_t i = 0; i < S; i++) y_l[i] = i % 3;
        t0 = now_ms();
        for (int r = 0; r < N / 20; r++) {
            DecisionTree tree = dt_create(F, 3, 8, 2);
            dt_fit(&tree, X, y_l, S, DT_CRITERION_GINI);
            dt_destroy(&tree);
        }
        t1 = now_ms();
        printf("  dt_fit (Gini):       %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 20, t1 - t0, (t1 - t0) / (double)(N / 20) * 1000.0);
        free(X); free(y_l);
    }

    /* ── Random Forest ── */
    {
        double *X = calloc(S * F, sizeof(double));
        int    *y_l = calloc(S, sizeof(int));
        for (size_t i = 0; i < S * F; i++) X[i] = (double)(i % 100) / 100.0;
        for (size_t i = 0; i < S; i++) y_l[i] = i % 3;
        t0 = now_ms();
        for (int r = 0; r < N / 20; r++) {
            RandomForest rf = rf_create(5, F, 3, 0.8, F / 2, 6, 2);
            rf_fit(&rf, X, y_l, S, DT_CRITERION_GINI);
            rf_destroy(&rf);
        }
        t1 = now_ms();
        printf("  rf_fit (5 trees):    %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 20, t1 - t0, (t1 - t0) / (double)(N / 20) * 1000.0);
        free(X); free(y_l);
    }

    /* ── K-Means ── */
    {
        double *X = calloc(S * F, sizeof(double));
        for (size_t i = 0; i < S * F; i++) X[i] = (double)(i % 100) / 20.0;
        t0 = now_ms();
        for (int r = 0; r < N / 20; r++) {
            KMeans km = kmeans_create(3, F, 10, 1e-4, KMEANS_INIT_RANDOM);
            kmeans_fit(&km, X, S);
            kmeans_destroy(&km);
        }
        t1 = now_ms();
        printf("  kmeans_fit (k=3):    %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 20, t1 - t0, (t1 - t0) / (double)(N / 20) * 1000.0);
        free(X);
    }

    /* ── DBSCAN ── */
    {
        double *X = calloc(S * F, sizeof(double));
        for (size_t i = 0; i < S * F; i++) X[i] = (double)(i % 100) / 20.0;
        t0 = now_ms();
        for (int r = 0; r < N / 20; r++) {
            DBSCAN db = dbscan_create(0.5, 5);
            dbscan_fit(&db, X, S, F);
            dbscan_destroy(&db);
        }
        t1 = now_ms();
        printf("  dbscan_fit:          %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 20, t1 - t0, (t1 - t0) / (double)(N / 20) * 1000.0);
        free(X);
    }

    /* ── Hierarchical Clustering ── */
    {
        double *X = calloc(S * F, sizeof(double));
        for (size_t i = 0; i < S * F; i++) X[i] = (double)(i % 100) / 20.0;
        t0 = now_ms();
        for (int r = 0; r < N / 20; r++) {
            HierarchicalClustering hc = hac_create(3, HAC_LINKAGE_AVERAGE);
            hac_fit(&hc, X, S, F);
            hac_destroy(&hc);
        }
        t1 = now_ms();
        printf("  hac_fit (avg link):  %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 20, t1 - t0, (t1 - t0) / (double)(N / 20) * 1000.0);
        free(X);
    }

    /* ── AdaBoost ── */
    {
        double *X = calloc(S * F, sizeof(double));
        int    *y_l = calloc(S, sizeof(int));
        for (size_t i = 0; i < S * F; i++) X[i] = (double)(i % 100) / 100.0;
        for (size_t i = 0; i < S; i++) y_l[i] = (i % 2) ? 1 : -1;
        t0 = now_ms();
        for (int r = 0; r < N / 20; r++) {
            AdaBoostModel ab = adaboost_create(10, F);
            adaboost_fit(&ab, X, y_l, S, F);
            adaboost_destroy(&ab);
        }
        t1 = now_ms();
        printf("  adaboost_fit (10):   %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 20, t1 - t0, (t1 - t0) / (double)(N / 20) * 1000.0);
        free(X); free(y_l);
    }

    /* ── GBDT ── */
    {
        double *X = calloc(S * F, sizeof(double));
        double *y = calloc(S, sizeof(double));
        for (size_t i = 0; i < S * F; i++) X[i] = (double)(i % 100) / 100.0;
        for (size_t i = 0; i < S; i++) y[i] = (double)(i % 10);
        t0 = now_ms();
        for (int r = 0; r < N / 20; r++) {
            GBDTModel gb = gbdt_create(10, 0.1, 4, F);
            gbdt_fit(&gb, X, y, S, 42);
            gbdt_destroy(&gb);
        }
        t1 = now_ms();
        printf("  gbdt_fit (10 trees): %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 20, t1 - t0, (t1 - t0) / (double)(N / 20) * 1000.0);
        free(X); free(y);
    }

    /* ── XGBoost ── */
    {
        double *X = calloc(S * F, sizeof(double));
        double *y = calloc(S, sizeof(double));
        for (size_t i = 0; i < S * F; i++) X[i] = (double)(i % 100) / 100.0;
        for (size_t i = 0; i < S; i++) y[i] = (double)(i % 10);
        t0 = now_ms();
        for (int r = 0; r < N / 20; r++) {
            XGBoostModel xgb = xgb_create(10, 0.1, 1.0, 0.0, 4, F);
            xgb_fit(&xgb, X, y, S, 42);
            xgb_destroy(&xgb);
        }
        t1 = now_ms();
        printf("  xgb_fit (10 trees):  %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 20, t1 - t0, (t1 - t0) / (double)(N / 20) * 1000.0);
        free(X); free(y);
    }

    printf("\nDone.\n");
    return 0;
}
