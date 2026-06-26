/*
 * mini-ml-basic — Full Demo: Machine Learning Basics
 *
 * Demonstrates: linear regression, logistic regression, softmax,
 *               SVM, decision tree, random forest, k-means, DBSCAN,
 *               hierarchical clustering, AdaBoost, GBDT, XGBoost.
 */
#include "../include/linear_models.h"
#include "../include/svm_kernel.h"
#include "../include/decision_tree.h"
#include "../include/clustering.h"
#include "../include/ensemble_gbdt.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    printf("=== mini-ml-basic: Machine Learning Basics Demo ===\n\n");

    /* Step 1: Linear Regression */
    printf("-- Step 1: Linear Regression --\n");
    size_t n = 100, f = 4;
    double *X = calloc(n * f, sizeof(double));
    double *y = calloc(n, sizeof(double));
    for (size_t i = 0; i < n * f; i++) X[i] = (double)(i % 100) / 50.0;
    for (size_t i = 0; i < n; i++) y[i] = 3.0 * X[i * f] + 2.0 * X[i * f + 1] + 1.0;
    LinearModel lm = lm_create(f);
    lm_fit_normal_eq(&lm, X, y, n);
    printf("Linear model: bias=%.4f, coefs=[", lm.bias);
    for (size_t j = 0; j < f; j++) printf("%.3f%s", lm.coefs[j], j < f - 1 ? ", " : "");
    printf("]\n");
    double pred = lm_predict(&lm, &X[0]);
    printf("  predict sample 0: %.4f (actual=%.4f)\n", pred, y[0]);
    lm_destroy(&lm);

    /* Step 2: Logistic Regression */
    printf("\n-- Step 2: Logistic Regression --\n");
    double *Xb = calloc(n * f, sizeof(double));
    double *yb = calloc(n, sizeof(double));
    for (size_t i = 0; i < n * f; i++) Xb[i] = (double)(i % 100) / 50.0;
    for (size_t i = 0; i < n; i++) yb[i] = (Xb[i * f] > 1.0) ? 1.0 : 0.0;
    LogisticModel logreg = logreg_create(f, 2);
    logreg_fit(&logreg, Xb, yb, n, 0.05, 30);
    int cls = logreg_predict(&logreg, &Xb[0]);
    printf("LogReg: sigmoid at boundary=%.4f, predict[0]=class %d\n",
           logreg_decision_boundary(&logreg, &Xb[0]), cls);
    double loss = logreg_bce_loss(&logreg, Xb, yb, n);
    printf("  BCE loss: %.4f\n", loss);
    logreg_destroy(&logreg);

    /* Step 3: SVM */
    printf("\n-- Step 3: SVM (Linear) --\n");
    double *Xs = calloc(n * f, sizeof(double));
    double *ys = calloc(n, sizeof(double));
    for (size_t i = 0; i < n * f; i++) Xs[i] = (double)(i % 100) / 30.0;
    for (size_t i = 0; i < n; i++) ys[i] = (Xs[i * f] > 1.5) ? 1.0 : -1.0;
    SVMModel svm = svm_create(f, 1.0, 2);
    svm_fit_linear(&svm, Xs, ys, n, 0.01, 50);
    int svm_pred = svm_predict(&svm, &Xs[0]);
    printf("SVM: predict[0]=%d, hinge_loss=%.4f\n", svm_pred,
           svm_hinge_loss(&svm, Xs, ys, n));
    svm_destroy(&svm);

    /* Step 4: Decision Tree & Random Forest */
    printf("\n-- Step 4: Decision Tree & Random Forest --\n");
    int *yc = calloc(n, sizeof(int));
    double *Xc = calloc(n * f, sizeof(double));
    for (size_t i = 0; i < n * f; i++) Xc[i] = (double)(i % 100) / 50.0;
    for (size_t i = 0; i < n; i++) yc[i] = i % 3;
    DecisionTree dt = dt_create(f, 3, 8, 2);
    dt_fit(&dt, Xc, yc, n, DT_CRITERION_GINI);
    printf("Decision Tree: root impurity=%.4f, predict[0]=%d\n",
           dt.root->impurity, dt_predict(&dt, &Xc[0]));
    dt_destroy(&dt);

    RandomForest rf = rf_create(10, f, 3, 0.8, f / 2, 6, 2);
    rf_fit(&rf, Xc, yc, n, DT_CRITERION_ENTROPY);
    int rf_pred = rf_predict(&rf, &Xc[0]);
    printf("Random Forest (10 trees): predict[0]=%d\n", rf_pred);
    rf_destroy(&rf);

    /* Step 5: Clustering */
    printf("\n-- Step 5: Clustering (K-Means, DBSCAN, Hierarchical) --\n");
    double *Xk = calloc(n * f, sizeof(double));
    for (size_t i = 0; i < n * f; i++) Xk[i] = (double)(i % 100) / 20.0;
    KMeans km = kmeans_create(3, f, 15, 1e-4, KMEANS_INIT_PLUSPLUS);
    kmeans_fit(&km, Xk, n);
    double inertia = kmeans_inertia(&km, Xk, n);
    printf("K-Means (k=3): inertia=%.4f, predict[0]=cluster %d\n",
           inertia, kmeans_predict(&km, &Xk[0]));
    kmeans_destroy(&km);

    DBSCAN db = dbscan_create(1.0, 3);
    dbscan_fit(&db, Xk, n, f);
    printf("DBSCAN (eps=1.0, min_pts=3): labels[0]=%d\n", db.labels[0]);
    dbscan_destroy(&db);

    HierarchicalClustering hac = hac_create(3, HAC_LINKAGE_AVERAGE);
    hac_fit(&hac, Xk, n, f);
    printf("HAC (avg linkage, n_clusters=3): labels[0]=%d\n", hac.labels[0]);
    double sil = silhouette_score(Xk, hac.labels, n, f);
    printf("  Silhouette score: %.4f\n", sil);
    hac_destroy(&hac);

    /* Step 6: Ensembles */
    printf("\n-- Step 6: Ensembles (AdaBoost, GBDT, XGBoost) --\n");
    int *ya = calloc(n, sizeof(int));
    double *Xa = calloc(n * f, sizeof(double));
    for (size_t i = 0; i < n * f; i++) Xa[i] = (double)(i % 100) / 50.0;
    for (size_t i = 0; i < n; i++) ya[i] = (i % 2) ? 1 : -1;
    AdaBoostModel ab = adaboost_create(20, f);
    adaboost_fit(&ab, Xa, ya, n, f);
    printf("AdaBoost (20 stumps): predict[0]=%d\n", adaboost_predict(&ab, &Xa[0]));
    adaboost_destroy(&ab);

    GBDTModel gb = gbdt_create(20, 0.1, 4, f);
    gbdt_fit(&gb, Xa, y, n, 42);
    printf("GBDT (20 trees, lr=0.1): predict[0]=%.4f\n", gbdt_predict(&gb, &Xa[0]));
    gbdt_destroy(&gb);

    XGBoostModel xgb = xgb_create(20, 0.1, 1.0, 0.0, 4, f);
    xgb_fit(&xgb, Xa, y, n, 42);
    printf("XGBoost (20 trees, lr=0.1): predict[0]=%.4f\n", xgb_predict(&xgb, &Xa[0]));
    xgb_destroy(&xgb);

    /* Cleanup */
    free(X); free(y); free(Xb); free(yb); free(Xs); free(ys);
    free(Xc); free(yc); free(Xk); free(Xa); free(ya);

    printf("\nML basics demo complete!\n");
    return 0;
}
