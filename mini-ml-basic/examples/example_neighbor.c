#include "neighbor_models.h"
#include "preprocessing.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(void) {
    size_t n = 100, d = 2;
    double *X = (double *)malloc(n * d * sizeof(double));
    int    *y = (int    *)malloc(n * sizeof(int));

    /* Two Gaussian blobs */
    for (size_t i = 0; i < 50; ++i) {
        X[i * 2]     =  0.0 + (double)(rand() % 100) / 100.0;
        X[i * 2 + 1] =  0.0 + (double)(rand() % 100) / 100.0;
        y[i] = 0;
    }
    for (size_t i = 50; i < 100; ++i) {
        X[i * 2]     =  5.0 + (double)(rand() % 100) / 100.0;
        X[i * 2 + 1] =  5.0 + (double)(rand() % 100) / 100.0;
        y[i] = 1;
    }

    /* ── KNN Classifier ── */
    KNNModel knn = knn_create_classifier(5, d, 2, DIST_EUCLIDEAN);
    knn_fit(&knn, X, y, n);

    int correct = 0;
    for (size_t i = 0; i < n; ++i) {
        int pred = knn_predict_class(&knn, X + i * d);
        if (pred == y[i]) correct++;
    }
    printf("KNN Classifier (k=5) accuracy: %.2f%%\n", 100.0 * correct / n);

    /* Test with different metrics */
    KNNModel knn_man = knn_create_classifier(5, d, 2, DIST_MANHATTAN);
    knn_fit(&knn_man, X, y, n);
    correct = 0;
    for (size_t i = 0; i < n; ++i)
        if (knn_predict_class(&knn_man, X + i * d) == y[i]) correct++;
    printf("KNN (Manhattan)  accuracy: %.2f%%\n", 100.0 * correct / n);

    /* ── KD-Tree ── */
    KDTree tree = kdtree_create(X, n, d, 16);
    kdtree_build(&tree);

    double query[2] = {2.5, 2.5};
    size_t indices[5];
    double distances[5];
    kdtree_knn_search(&tree, query, 5, DIST_EUCLIDEAN, indices, distances);
    printf("KD-Tree 5-NN for (2.5,2.5):\n");
    for (size_t i = 0; i < 5 && indices[i] != (size_t)(-1); ++i)
        printf("  idx=%zu dist=%.4f label=%d\n", indices[i], distances[i], y[indices[i]]);

    kdtree_destroy(&tree);

    /* ── Naive Bayes ── */
    NaiveBayesModel nb = nb_create(d, 2);
    nb_fit(&nb, X, y, n);
    correct = 0;
    for (size_t i = 0; i < n; ++i)
        if (nb_predict(&nb, X + i * d) == y[i]) correct++;
    printf("Naive Bayes accuracy: %.2f%%\n", 100.0 * correct / n);
    nb_destroy(&nb);

    /* ── Standard Scaler ── */
    StandardScaler ss = scaler_std_create(d);
    double *X_scaled = (double *)malloc(n * d * sizeof(double));
    scaler_std_fit_transform(&ss, X, X_scaled, n);
    printf("StandardScaler: mean=(%.2f,%.2f) std=(%.2f,%.2f)\n",
           ss.mean[0], ss.mean[1], ss.std[0], ss.std[1]);
    scaler_std_destroy(&ss);

    /* ── Polynomial Features ── */
    double test_pt[2] = {1.0, 2.0};
    PolynomialFeatures pf = polyfeat_create(2, 2, false);
    double *poly_out = (double *)malloc(pf.n_features_out * sizeof(double));
    polyfeat_transform(&pf, test_pt, poly_out, 1);
    printf("PolyFeatures(deg=2): %zu features for (1,2):",
           pf.n_features_out);
    for (size_t j = 0; j < pf.n_features_out; ++j)
        printf(" %.4f", poly_out[j]);
    printf("\n");
    polyfeat_destroy(&pf);
    free(poly_out);

    free(X); free(y); free(X_scaled);
    return 0;
}