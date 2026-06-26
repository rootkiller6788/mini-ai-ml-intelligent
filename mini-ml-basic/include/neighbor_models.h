#ifndef NEIGHBOR_MODELS_H
#define NEIGHBOR_MODELS_H

#include <stdbool.h>
#include <stddef.h>

/*
 * neighbor_models.h ? Instance-Based Learning & Spatial Indexing
 *
 * L1: KNNModel struct, KDNode/KDTree struct, DistanceMetric enum
 * L2: Instance-based (lazy) learning ? stores training data, defers computation
 * L3: KD-Tree ? spatial partitioning for sub-linear nearest-neighbor search
 * L4: Cover & Hart theorem (1967): 1-NN error ? 2? Bayes optimal error (asymptotic)
 * L5: KNN classification, KD-Tree construction (O(d n log n)), range queries
 */

/* ??????????????????????????????????????????????
   Distance Metrics
   ?????????????????????????????????????????????? */
typedef enum {
    DIST_EUCLIDEAN = 0,
    DIST_MANHATTAN,
    DIST_CHEBYSHEV,
    DIST_MINKOWSKI,
    DIST_COSINE
} DistanceMetric;

/* ??????????????????????????????????????????????
   K-Nearest Neighbors (KNN)
   ?????????????????????????????????????????????? */
typedef struct {
    const double *train_X;       /* (n_samples, n_features) row-major  */
    const double *train_y;       /* (n_samples,) targets (regression)  */
    const int    *train_y_cls;   /* (n_samples,) labels  (classification) */
    size_t  n_samples;
    size_t  n_features;
    size_t  k;                   /* number of neighbours               */
    DistanceMetric metric;
    double  p;                   /* Minkowski power (p=2 ? Euclidean)  */
    bool    classification;      /* true=classify, false=regress       */
    int     n_classes;
} KNNModel;

/* distance between two feature vectors */
double  knn_distance(const double *a, const double *b, size_t dim,
                    DistanceMetric metric, double p);

KNNModel  knn_create_classifier(size_t k, size_t n_features,
                                 int n_classes, DistanceMetric metric);
KNNModel  knn_create_regressor(size_t k, size_t n_features,
                                DistanceMetric metric);

/* Fit: store training data (lazy ? no computation) */
void      knn_fit(KNNModel *knn, const double *X, const void *y,
                  size_t n_samples);

/* Predict */
int       knn_predict_class(const KNNModel *knn, const double *x);
double    knn_predict_reg(const KNNModel *knn, const double *x);

/* ??????????????????????????????????????????????
   KD-Tree ? spatial partitioning for fast NN search
   Implements Bentley (1975) kd-tree with median split
   ?????????????????????????????????????????????? */
typedef struct KDNode {
    size_t    point_idx;         /* index into original points array  */
    size_t    split_dim;         /* axis-aligned splitting dimension  */
    double    split_val;         /* split threshold                   */
    struct KDNode *left;
    struct KDNode *right;
    bool      is_leaf;
} KDNode;

typedef struct {
    KDNode       *root;
    const double *points;        /* (n_points, n_dims) row-major      */
    size_t        n_points;
    size_t        n_dims;
    size_t        leaf_size;     /* stop splitting if ? leaf_size     */
} KDTree;

KDTree   kdtree_create(const double *points, size_t n, size_t dim,
                        size_t leaf_size);
void     kdtree_destroy(KDTree *tree);
void     kdtree_build(KDTree *tree);

/* k-nearest-neighbors query using KD-Tree */
void     kdtree_knn_search(const KDTree *tree, const double *query,
                           size_t k, DistanceMetric metric,
                           size_t *indices, double *distances);

/* radius search: find all points within distance r */
size_t   kdtree_radius_search(const KDTree *tree, const double *query,
                              double radius, DistanceMetric metric,
                              size_t *indices, size_t max_results);

/* ??????????????????????????????????????????????
   Naive Bayes (Gaussian)
   Approximates P(x_j | y=k) ~ N(?_{j,k}, ??_{j,k})
   ?????????????????????????????????????????????? */
typedef struct {
    double *means;               /* (n_classes, n_features) row-major */
    double *vars;                /* (n_classes, n_features) row-major */
    double *priors;              /* (n_classes,)                      */
    size_t  n_features;
    int     n_classes;
} NaiveBayesModel;

NaiveBayesModel  nb_create(size_t n_features, int n_classes);
void             nb_destroy(NaiveBayesModel *model);
void             nb_fit(NaiveBayesModel *model, const double *X,
                        const int *y, size_t n_samples);
int              nb_predict(const NaiveBayesModel *model, const double *x);
void             nb_predict_proba(const NaiveBayesModel *model,
                                  const double *x, double *probs);

#endif /* NEIGHBOR_MODELS_H */
