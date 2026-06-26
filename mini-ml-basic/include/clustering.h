#ifndef CLUSTERING_H
#define CLUSTERING_H

#include <stdbool.h>
#include <stddef.h>

/* ──────────────────────────────────────────────
   K-Means
   ────────────────────────────────────────────── */
typedef enum {
    KMEANS_INIT_RANDOM = 0,
    KMEANS_INIT_PLUSPLUS
} KMeansInit;

typedef struct {
    double *centroids;        /* (k, n_features) row-major     */
    int    *labels;           /* (n,)                          */
    size_t  k;
    size_t  n_features;
    int     max_iters;
    double  tol;
    KMeansInit init_method;
} KMeans;

KMeans  kmeans_create(size_t k, size_t n_features, int max_iters, double tol,
                      KMeansInit init);
void    kmeans_destroy(KMeans *km);

void    kmeans_fit(KMeans *km, const double *X, size_t n_samples);
int     kmeans_predict(const KMeans *km, const double *x);
void    kmeans_centroids_get(const KMeans *km, double *out);   /* (k, n_features) */
double  kmeans_inertia(const KMeans *km, const double *X, size_t n);

/* Elbow helper – run K-Means for k ∈ [1..max_k], return inertias */
void    kmeans_elbow(const double *X, size_t n, size_t n_features,
                     int max_k, int max_iters, double tol,
                     double *inertias);

/* ──────────────────────────────────────────────
   DBSCAN (overview)
   ────────────────────────────────────────────── */
typedef struct {
    int    *labels;              /* (n,); -1 = noise          */
    size_t  n_samples;
    double  eps;
    int     min_pts;
} DBSCAN;

DBSCAN  dbscan_create(double eps, int min_pts);
void    dbscan_destroy(DBSCAN *db);
void    dbscan_fit(DBSCAN *db, const double *X,
                   size_t n_samples, size_t n_features);

/* ──────────────────────────────────────────────
   Hierarchical (Agglomerative)
   ────────────────────────────────────────────── */
typedef enum {
    HAC_LINKAGE_SINGLE = 0,
    HAC_LINKAGE_COMPLETE,
    HAC_LINKAGE_AVERAGE
} HACLinkage;

typedef struct {
    int       *labels;           /* (n,)                    */
    size_t     n_samples;
    size_t     n_clusters;
    HACLinkage linkage;
    /* internal – dendrogram stored in merge matrix */
    int       *merge;            /* (n-1, 2) merge history */
    double    *merge_height;     /* (n-1,)                 */
} HierarchicalClustering;

HierarchicalClustering hac_create(size_t n_clusters, HACLinkage linkage);
void                   hac_destroy(HierarchicalClustering *hc);
void                   hac_fit(HierarchicalClustering *hc,
                               const double *X, size_t n_samples, size_t dim);

/* ──────────────────────────────────────────────
   Silhouette score
   ────────────────────────────────────────────── */
double  silhouette_score(const double *X, const int *labels,
                         size_t n_samples, size_t n_features);

#endif /* CLUSTERING_H */
