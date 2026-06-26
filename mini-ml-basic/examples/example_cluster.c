#include "clustering.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(void) {
    /* 3 Gaussians in 2D */
    size_t n = 150;
    size_t d = 2;
    double *X = (double *)malloc(n * d * sizeof(double));

    for (size_t i = 0; i < 50; ++i) {
        X[i * 2]     =  0.0 + (double)(rand() % 100) / 100.0;
        X[i * 2 + 1] =  0.0 + (double)(rand() % 100) / 100.0;
    }
    for (size_t i = 50; i < 100; ++i) {
        X[i * 2]     =  4.0 + (double)(rand() % 100) / 100.0;
        X[i * 2 + 1] =  4.0 + (double)(rand() % 100) / 100.0;
    }
    for (size_t i = 100; i < 150; ++i) {
        X[i * 2]     =  0.0 + (double)(rand() % 100) / 100.0;
        X[i * 2 + 1] =  4.0 + (double)(rand() % 100) / 100.0;
    }

    /* ── K-Means with k-means++ init ── */
    KMeans km = kmeans_create(3, d, 100, 1e-4, KMEANS_INIT_PLUSPLUS);
    kmeans_fit(&km, X, n);

    size_t counts[3] = {0, 0, 0};
    for (size_t i = 0; i < n; ++i) counts[(size_t)km.labels[i]]++;
    printf("K-Means cluster sizes: %zu / %zu / %zu\n", counts[0], counts[1], counts[2]);
    printf("K-Means inertia:       %.4f\n", kmeans_inertia(&km, X, n));

    /* ── Elbow method ── */
    int max_k = 5;
    double *inertias = (double *)malloc((size_t)max_k * sizeof(double));
    kmeans_elbow(X, n, d, max_k, 100, 1e-4, inertias);
    printf("Elbow inertias: ");
    for (int k = 0; k < max_k; ++k)
        printf("k=%d:%.2f  ", k + 1, inertias[k]);
    printf("\n");

    /* ── Silhouette ── */
    double sil = silhouette_score(X, km.labels, n, d);
    printf("Silhouette score:      %.4f\n", sil);

    /* ── DBSCAN ── */
    DBSCAN db = dbscan_create(0.8, 3);
    dbscan_fit(&db, X, n, d);
    int noise_count = 0;
    for (size_t i = 0; i < n; ++i)
        if (db.labels[i] == -1) noise_count++;
    printf("DBSCAN noise points:   %d / %zu\n", noise_count, n);

    kmeans_destroy(&km);
    dbscan_destroy(&db);
    free(inertias);
    free(X);
    return 0;
}
