#include "neighbor_models.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ================================================================
   Distance Functions ? 5 metrics: Euclidean, Manhattan, Chebyshev,
   Minkowski (Lp), Cosine.
   ================================================================ */
double knn_distance(const double *a, const double *b, size_t dim,
                    DistanceMetric metric, double p) {
    switch (metric) {
    case DIST_EUCLIDEAN: {
        double sum = 0.0;
        for (size_t j = 0; j < dim; ++j) {
            double diff = a[j] - b[j];
            sum += diff * diff;
        }
        return sqrt(sum);
    }
    case DIST_MANHATTAN: {
        double sum = 0.0;
        for (size_t j = 0; j < dim; ++j)
            sum += fabs(a[j] - b[j]);
        return sum;
    }
    case DIST_CHEBYSHEV: {
        double max_d = 0.0;
        for (size_t j = 0; j < dim; ++j) {
            double d = fabs(a[j] - b[j]);
            if (d > max_d) max_d = d;
        }
        return max_d;
    }
    case DIST_MINKOWSKI: {
        double sum = 0.0;
        for (size_t j = 0; j < dim; ++j)
            sum += pow(fabs(a[j] - b[j]), p);
        return pow(sum, 1.0 / p);
    }
    case DIST_COSINE: {
        double dot = 0.0, na = 0.0, nb = 0.0;
        for (size_t j = 0; j < dim; ++j) {
            dot += a[j] * b[j];
            na  += a[j] * a[j];
            nb  += b[j] * b[j];
        }
        double denom = sqrt(na) * sqrt(nb);
        if (denom < 1e-12) return 0.0;
        return 1.0 - dot / denom;
    }
    default:
        return 0.0;
    }
}

/* ================================================================
   K-Nearest Neighbors ? Lazy instance-based learning.
   Cover & Hart (1967): asymptotic 1-NN error ? 2? Bayes error.
   ================================================================ */
KNNModel knn_create_classifier(size_t k, size_t n_features,
                                int n_classes, DistanceMetric metric) {
    KNNModel m;
    m.train_X        = NULL;
    m.train_y        = NULL;
    m.train_y_cls    = NULL;
    m.n_samples      = 0;
    m.n_features     = n_features;
    m.k              = k;
    m.metric         = metric;
    m.p              = 2.0;
    m.classification = true;
    m.n_classes      = n_classes;
    return m;
}

KNNModel knn_create_regressor(size_t k, size_t n_features,
                               DistanceMetric metric) {
    KNNModel m;
    m.train_X        = NULL;
    m.train_y        = NULL;
    m.train_y_cls    = NULL;
    m.n_samples      = 0;
    m.n_features     = n_features;
    m.k              = k;
    m.metric         = metric;
    m.p              = 2.0;
    m.classification = false;
    m.n_classes      = 0;
    return m;
}

void knn_fit(KNNModel *knn, const double *X, const void *y, size_t n_samples) {
    knn->train_X   = X;
    knn->n_samples = n_samples;
    if (knn->classification)
        knn->train_y_cls = (const int *)y;
    else
        knn->train_y = (const double *)y;
}

int knn_predict_class(const KNNModel *knn, const double *x) {
    size_t n = knn->n_samples, d = knn->n_features, k = knn->k;
    if (k > n) k = n;
    if (k == 0) return 0;

    size_t *best_idx  = (size_t *)malloc(k * sizeof(size_t));
    double *best_dist = (double *)malloc(k * sizeof(double));
    for (size_t i = 0; i < k; ++i) best_dist[i] = DBL_MAX;

    for (size_t i = 0; i < n; ++i) {
        double dist = knn_distance(x, knn->train_X + i * d, d,
                                    knn->metric, knn->p);
        size_t pos = k;
        while (pos > 0 && best_dist[pos - 1] > dist) --pos;
        if (pos < k) {
            memmove(best_dist + pos + 1, best_dist + pos,
                    (k - pos - 1) * sizeof(double));
            memmove(best_idx + pos + 1, best_idx + pos,
                    (k - pos - 1) * sizeof(size_t));
            best_dist[pos] = dist;
            best_idx[pos]  = i;
        }
    }

    int *votes = (int *)calloc((size_t)knn->n_classes, sizeof(int));
    for (size_t i = 0; i < k; ++i)
        votes[knn->train_y_cls[best_idx[i]]]++;
    int best_class = 0;
    for (int c = 1; c < knn->n_classes; ++c)
        if (votes[c] > votes[best_class]) best_class = c;

    free(best_idx); free(best_dist); free(votes);
    return best_class;
}

double knn_predict_reg(const KNNModel *knn, const double *x) {
    size_t n = knn->n_samples, d = knn->n_features, k = knn->k;
    if (k > n) k = n;
    if (k == 0) return 0.0;

    size_t *best_idx  = (size_t *)malloc(k * sizeof(size_t));
    double *best_dist = (double *)malloc(k * sizeof(double));
    for (size_t i = 0; i < k; ++i) best_dist[i] = DBL_MAX;

    for (size_t i = 0; i < n; ++i) {
        double dist = knn_distance(x, knn->train_X + i * d, d,
                                    knn->metric, knn->p);
        size_t pos = k;
        while (pos > 0 && best_dist[pos - 1] > dist) --pos;
        if (pos < k) {
            memmove(best_dist + pos + 1, best_dist + pos,
                    (k - pos - 1) * sizeof(double));
            memmove(best_idx + pos + 1, best_idx + pos,
                    (k - pos - 1) * sizeof(size_t));
            best_dist[pos] = dist;
            best_idx[pos]  = i;
        }
    }

    double sum = 0.0;
    for (size_t i = 0; i < k; ++i) sum += knn->train_y[best_idx[i]];
    free(best_idx); free(best_dist);
    return sum / (double)k;
}

/* ================================================================
   KD-Tree ? Bentley (1975) spatial partitioning tree.
   Build: O(d n log n). Query: O(log n) expected.
   ================================================================ */
static KDNode *knode_alloc(void) {
    return (KDNode *)calloc(1, sizeof(KDNode));
}

static KDNode *kdtree_build_rec(const double *pts, size_t *indices,
                                 size_t n, size_t dim, size_t depth,
                                 size_t leaf_size) {
    KDNode *node = knode_alloc();
    if (n <= leaf_size) {
        node->is_leaf   = true;
        node->point_idx = indices[0];
        return node;
    }
    size_t split_axis = depth % dim;
    /* selection sort along split axis */
    for (size_t i = 0; i < n; ++i) {
        size_t min_j = i;
        for (size_t j = i + 1; j < n; ++j) {
            double vi = pts[indices[j] * dim + split_axis];
            double vm = pts[indices[min_j] * dim + split_axis];
            if (vi < vm) min_j = j;
        }
        if (min_j != i) {
            size_t t = indices[i];
            indices[i] = indices[min_j];
            indices[min_j] = t;
        }
    }
    size_t mid = n / 2;
    node->split_dim = split_axis;
    node->split_val = pts[indices[mid] * dim + split_axis];
    node->point_idx = indices[mid];
    node->left  = kdtree_build_rec(pts, indices, mid,
                                   dim, depth + 1, leaf_size);
    node->right = kdtree_build_rec(pts, indices + mid + 1,
                                   n - mid - 1, dim, depth + 1, leaf_size);
    return node;
}

static void kdtree_node_free(KDNode *node) {
    if (!node) return;
    kdtree_node_free(node->left);
    kdtree_node_free(node->right);
    free(node);
}

KDTree kdtree_create(const double *points, size_t n, size_t dim,
                      size_t leaf_size) {
    KDTree tree;
    tree.points    = points;
    tree.n_points  = n;
    tree.n_dims    = dim;
    tree.leaf_size = (leaf_size == 0) ? 16 : leaf_size;
    tree.root      = NULL;
    return tree;
}

void kdtree_destroy(KDTree *tree) {
    kdtree_node_free(tree->root);
    tree->root = NULL;
}

void kdtree_build(KDTree *tree) {
    size_t n = tree->n_points;
    if (n == 0) return;
    size_t *indices = (size_t *)malloc(n * sizeof(size_t));
    for (size_t i = 0; i < n; ++i) indices[i] = i;
    tree->root = kdtree_build_rec(tree->points, indices, n,
                                   tree->n_dims, 0, tree->leaf_size);
    free(indices);
}

typedef struct { size_t idx; double dist; } KnnEntry;

static void kdtree_search_rec(const KDNode *node, const double *pts,
                               const double *query, size_t dim,
                               DistanceMetric metric, double p,
                               KnnEntry *best, size_t k, size_t *n_found) {
    if (!node) return;
    if (node->is_leaf) {
        double d = knn_distance(query, pts + node->point_idx * dim,
                                dim, metric, p);
        if (*n_found < k) {
            best[*n_found].idx = node->point_idx;
            best[*n_found].dist = d;
            (*n_found)++;
        } else {
            size_t worst = 0;
            for (size_t i = 1; i < k; ++i)
                if (best[i].dist > best[worst].dist) worst = i;
            if (d < best[worst].dist) {
                best[worst].idx = node->point_idx;
                best[worst].dist = d;
            }
        }
        return;
    }
    double diff = query[node->split_dim] - node->split_val;
    KDNode *near = (diff <= 0.0) ? node->left : node->right;
    KDNode *far  = (diff <= 0.0) ? node->right : node->left;
    kdtree_search_rec(near, pts, query, dim, metric, p, best, k, n_found);
    if (*n_found < k || fabs(diff) < best[k - 1].dist)
        kdtree_search_rec(far, pts, query, dim, metric, p, best, k, n_found);
}

void kdtree_knn_search(const KDTree *tree, const double *query,
                        size_t k, DistanceMetric metric,
                        size_t *indices, double *distances) {
    if (!tree->root || k == 0) return;
    KnnEntry *best = (KnnEntry *)malloc(k * sizeof(KnnEntry));
    size_t n_found = 0;
    for (size_t i = 0; i < k; ++i) best[i].dist = DBL_MAX;
    kdtree_search_rec(tree->root, tree->points, query, tree->n_dims,
                       metric, 2.0, best, k, &n_found);
    for (size_t i = 0; i < n_found && i < k; ++i) {
        indices[i]   = best[i].idx;
        distances[i] = best[i].dist;
    }
    for (size_t i = n_found; i < k; ++i) {
        indices[i]   = (size_t)(-1);
        distances[i] = DBL_MAX;
    }
    free(best);
}

/* Radius search ? find all points within distance r */
static void kdtree_radius_rec(const KDNode *node, const double *pts,
                               const double *query, size_t dim,
                               DistanceMetric metric, double p,
                               double radius, double radius_sq,
                               size_t *indices, size_t *count,
                               size_t max_results) {
    if (!node || *count >= max_results) return;
    if (node->is_leaf) {
        double d = knn_distance(query, pts + node->point_idx * dim,
                                dim, metric, p);
        if (d <= radius && *count < max_results) {
            indices[*count] = node->point_idx;
            (*count)++;
        }
        return;
    }
    double diff = query[node->split_dim] - node->split_val;
    KDNode *near = (diff <= 0.0) ? node->left : node->right;
    KDNode *far  = (diff <= 0.0) ? node->right : node->left;
    kdtree_radius_rec(near, pts, query, dim, metric, p,
                       radius, radius_sq, indices, count, max_results);
    if (fabs(diff) <= radius_sq)
        kdtree_radius_rec(far, pts, query, dim, metric, p,
                           radius, radius_sq, indices, count, max_results);
}

size_t kdtree_radius_search(const KDTree *tree, const double *query,
                             double radius, DistanceMetric metric,
                             size_t *indices, size_t max_results) {
    if (!tree->root) return 0;
    size_t count = 0;
    kdtree_radius_rec(tree->root, tree->points, query, tree->n_dims,
                       metric, 2.0, radius, radius * radius,
                       indices, &count, max_results);
    return count;
}

/* ================================================================
   Naive Bayes (Gaussian) ? P(x_j | y=k) ~ N(mu_jk, sigma^2_jk)
   P(y=k | x) proportional to P(y=k) * prod_j P(x_j | y=k)
   Decision boundary is quadratic in feature space.
   Works well for text classification, medical diagnosis despite
   the naive independence assumption.
   ================================================================ */
NaiveBayesModel nb_create(size_t n_features, int n_classes) {
    NaiveBayesModel m;
    m.n_features = n_features;
    m.n_classes  = n_classes;
    m.means  = (double *)calloc((size_t)n_classes * n_features, sizeof(double));
    m.vars   = (double *)calloc((size_t)n_classes * n_features, sizeof(double));
    m.priors = (double *)calloc((size_t)n_classes, sizeof(double));
    return m;
}

void nb_destroy(NaiveBayesModel *model) {
    free(model->means);  free(model->vars);  free(model->priors);
    model->means = NULL; model->vars = NULL; model->priors = NULL;
}

void nb_fit(NaiveBayesModel *model, const double *X,
             const int *y, size_t n_samples) {
    size_t d = model->n_features;
    int    C = model->n_classes;
    size_t *class_counts = (size_t *)calloc((size_t)C, sizeof(size_t));

    for (size_t i = 0; i < n_samples; ++i) {
        int c = y[i]; class_counts[c]++;
        for (size_t j = 0; j < d; ++j)
            model->means[c * d + j] += X[i * d + j];
    }
    for (int c = 0; c < C; ++c) {
        if (class_counts[c] == 0) continue;
        for (size_t j = 0; j < d; ++j)
            model->means[c * d + j] /= (double)class_counts[c];
    }
    for (size_t i = 0; i < n_samples; ++i) {
        int c = y[i];
        for (size_t j = 0; j < d; ++j) {
            double diff = X[i * d + j] - model->means[c * d + j];
            model->vars[c * d + j] += diff * diff;
        }
    }
    for (int c = 0; c < C; ++c) {
        if (class_counts[c] <= 1) {
            for (size_t j = 0; j < d; ++j)
                model->vars[c * d + j] = 1e-6;
        } else {
            for (size_t j = 0; j < d; ++j)
                model->vars[c * d + j] /= (double)(class_counts[c] - 1);
        }
        model->priors[c] = (double)class_counts[c] / (double)n_samples;
    }
    free(class_counts);
}

static double nb_gauss_log_pdf(double x, double mean, double var) {
    double sigma_sq = var + 1e-9;
    return -0.5 * log(2.0 * 3.14159265358979323846 * sigma_sq)
           - 0.5 * (x - mean) * (x - mean) / sigma_sq;
}

int nb_predict(const NaiveBayesModel *model, const double *x) {
    size_t d = model->n_features;
    int    C = model->n_classes;
    double best_log_prob = -1e308;
    int    best_class    = 0;
    for (int c = 0; c < C; ++c) {
        double log_prob = log(model->priors[c] + 1e-12);
        for (size_t j = 0; j < d; ++j)
            log_prob += nb_gauss_log_pdf(x[j], model->means[c * d + j],
                                          model->vars[c * d + j]);
        if (log_prob > best_log_prob) {
            best_log_prob = log_prob;
            best_class    = c;
        }
    }
    return best_class;
}

void nb_predict_proba(const NaiveBayesModel *model,
                       const double *x, double *probs) {
    size_t d = model->n_features;
    int    C = model->n_classes;
    double log_probs[256];
    double max_log = -1e308;
    for (int c = 0; c < C && c < 256; ++c) {
        log_probs[c] = log(model->priors[c] + 1e-12);
        for (size_t j = 0; j < d; ++j)
            log_probs[c] += nb_gauss_log_pdf(x[j], model->means[c * d + j],
                                              model->vars[c * d + j]);
        if (log_probs[c] > max_log) max_log = log_probs[c];
    }
    double sum = 0.0;
    for (int c = 0; c < C; ++c) {
        probs[c] = exp(log_probs[c] - max_log);
        sum += probs[c];
    }
    for (int c = 0; c < C; ++c)
        probs[c] /= (sum + 1e-12);
}
