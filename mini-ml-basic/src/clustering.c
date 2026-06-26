#include "clustering.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

/* ──────────────────────────────────────────────
   K-Means
   ────────────────────────────────────────────── */

static double euclidean_sq(const double *a, const double *b, size_t d) {
    double s = 0.0;
    for (size_t j = 0; j < d; ++j) {
        double diff = a[j] - b[j];
        s += diff * diff;
    }
    return s;
}

KMeans kmeans_create(size_t k, size_t n_features, int max_iters, double tol,
                     KMeansInit init) {
    KMeans km;
    km.k          = k;
    km.n_features = n_features;
    km.max_iters  = max_iters;
    km.tol        = tol;
    km.init_method = init;
    km.centroids  = (double *)calloc(k * n_features, sizeof(double));
    km.labels     = NULL;
    return km;
}

void kmeans_destroy(KMeans *km) {
    free(km->centroids);
    free(km->labels);
    km->centroids = NULL;
    km->labels    = NULL;
}

static void kmeans_init_random(KMeans *km, const double *X, size_t n) {
    size_t k = km->k, d = km->n_features;
    for (size_t c = 0; c < k; ++c) {
        size_t r = (size_t)rand() % n;
        memcpy(km->centroids + c * d, X + r * d, d * sizeof(double));
    }
}

static void kmeans_init_plusplus(KMeans *km, const double *X, size_t n) {
    size_t k = km->k, d = km->n_features;
    size_t r = (size_t)rand() % n;
    memcpy(km->centroids, X + r * d, d * sizeof(double));

    double *dist = (double *)malloc(n * sizeof(double));
    for (size_t c = 1; c < k; ++c) {
        double sum_d = 0.0;
        for (size_t i = 0; i < n; ++i) {
            double best = DBL_MAX;
            for (size_t p = 0; p < c; ++p) {
                double dsq = euclidean_sq(X + i * d, km->centroids + p * d, d);
                if (dsq < best) best = dsq;
            }
            dist[i] = best;
            sum_d  += best;
        }
        double thresh = ((double)rand() / (double)RAND_MAX) * sum_d;
        double cum = 0.0;
        size_t sel = n - 1;
        for (size_t i = 0; i < n; ++i) {
            cum += dist[i];
            if (cum >= thresh) { sel = i; break; }
        }
        memcpy(km->centroids + c * d, X + sel * d, d * sizeof(double));
    }
    free(dist);
}

void kmeans_fit(KMeans *km, const double *X, size_t n) {
    size_t k = km->k, d = km->n_features;

    if (km->init_method == KMEANS_INIT_PLUSPLUS)
        kmeans_init_plusplus(km, X, n);
    else
        kmeans_init_random(km, X, n);

    km->labels = (int *)malloc(n * sizeof(int));

    for (int iter = 0; iter < km->max_iters; ++iter) {
        /* assign */
        bool changed = false;
        for (size_t i = 0; i < n; ++i) {
            double best_d = DBL_MAX;
            int    best_c = 0;
            for (size_t c = 0; c < k; ++c) {
                double dsq = euclidean_sq(X + i * d, km->centroids + c * d, d);
                if (dsq < best_d) { best_d = dsq; best_c = (int)c; }
            }
            if (km->labels[i] != best_c) changed = true;
            km->labels[i] = best_c;
        }
        /* update */
        double *new_c = (double *)calloc(k * d, sizeof(double));
        size_t *cnt   = (size_t *)calloc(k, sizeof(size_t));
        for (size_t i = 0; i < n; ++i) {
            int c = km->labels[i];
            cnt[c]++;
            for (size_t j = 0; j < d; ++j)
                new_c[c * d + j] += X[i * d + j];
        }
        double max_shift = 0.0;
        for (size_t c = 0; c < k; ++c) {
            if (cnt[c] == 0) continue;
            for (size_t j = 0; j < d; ++j) {
                double val = new_c[c * d + j] / (double)cnt[c];
                double shift = fabs(val - km->centroids[c * d + j]);
                if (shift > max_shift) max_shift = shift;
                km->centroids[c * d + j] = val;
            }
        }
        free(new_c);
        free(cnt);
        if (!changed || max_shift < km->tol) break;
    }
}

int kmeans_predict(const KMeans *km, const double *x) {
    size_t k = km->k, d = km->n_features;
    double best = DBL_MAX;
    int    cls  = 0;
    for (size_t c = 0; c < k; ++c) {
        double dsq = euclidean_sq(x, km->centroids + c * d, d);
        if (dsq < best) { best = dsq; cls = (int)c; }
    }
    return cls;
}

void kmeans_centroids_get(const KMeans *km, double *out) {
    memcpy(out, km->centroids, km->k * km->n_features * sizeof(double));
}

double kmeans_inertia(const KMeans *km, const double *X, size_t n) {
    size_t k = km->k, d = km->n_features;
    double inertia = 0.0;
    for (size_t i = 0; i < n; ++i) {
        int c = km->labels[i];
        inertia += euclidean_sq(X + i * d, km->centroids + c * d, d);
    }
    return inertia;
}

void kmeans_elbow(const double *X, size_t n, size_t n_features,
                  int max_k, int max_iters, double tol,
                  double *inertias) {
    for (int k = 1; k <= max_k; ++k) {
        KMeans km = kmeans_create((size_t)k, n_features, max_iters, tol,
                                  KMEANS_INIT_PLUSPLUS);
        kmeans_fit(&km, X, n);
        inertias[k - 1] = kmeans_inertia(&km, X, n);
        kmeans_destroy(&km);
    }
}

/* ──────────────────────────────────────────────
   DBSCAN
   ────────────────────────────────────────────── */

DBSCAN dbscan_create(double eps, int min_pts) {
    DBSCAN db;
    db.labels    = NULL;
    db.n_samples = 0;
    db.eps       = eps;
    db.min_pts   = min_pts;
    return db;
}

void dbscan_destroy(DBSCAN *db) {
    free(db->labels);
    db->labels = NULL;
}

void dbscan_fit(DBSCAN *db, const double *X,
                size_t n_samples, size_t n_features) {
    db->n_samples = n_samples;
    db->labels = (int *)malloc(n_samples * sizeof(int));
    /* label -2 = undefined, -1 = noise, >= 0 = cluster id */
    for (size_t i = 0; i < n_samples; ++i) db->labels[i] = -2;

    int cluster_id = 0;
    for (size_t i = 0; i < n_samples; ++i) {
        if (db->labels[i] != -2) continue;
        /* find neighbours */
        size_t *neigh = (size_t *)malloc(n_samples * sizeof(size_t));
        size_t  nn = 0;
        for (size_t j = 0; j < n_samples; ++j) {
            if (euclidean_sq(X + i * n_features, X + j * n_features, n_features)
                <= db->eps * db->eps)
                neigh[nn++] = j;
        }
        if (nn < (size_t)db->min_pts) {
            db->labels[i] = -1;  /* noise */
        } else {
            /* expand cluster */
            db->labels[i] = cluster_id;
            for (size_t s = 0; s < nn; ++s) {
                size_t q = neigh[s];
                if (db->labels[q] == -1) db->labels[q] = cluster_id;
                if (db->labels[q] != -2) continue;
                db->labels[q] = cluster_id;
                /* find neighbours of q */
                size_t *nq = (size_t *)malloc(n_samples * sizeof(size_t));
                size_t  nqn = 0;
                for (size_t j = 0; j < n_samples; ++j) {
                    if (euclidean_sq(X + q * n_features, X + j * n_features, n_features)
                        <= db->eps * db->eps)
                        nq[nqn++] = j;
                }
                if (nqn >= (size_t)db->min_pts) {
                    /* merge */
                    for (size_t jj = 0; jj < nqn; ++jj) {
                        bool found = false;
                        for (size_t kk = 0; kk < nn; ++kk)
                            if (neigh[kk] == nq[jj]) { found = true; break; }
                        if (!found && nn < n_samples) neigh[nn++] = nq[jj];
                    }
                }
                free(nq);
            }
            cluster_id++;
        }
        free(neigh);
    }
}

/* ──────────────────────────────────────────────
   Hierarchical (Agglomerative)
   ────────────────────────────────────────────── */

HierarchicalClustering hac_create(size_t n_clusters, HACLinkage linkage) {
    HierarchicalClustering hc;
    hc.n_samples  = 0;
    hc.n_clusters = n_clusters;
    hc.linkage    = linkage;
    hc.labels     = NULL;
    hc.merge      = NULL;
    hc.merge_height = NULL;
    return hc;
}

void hac_destroy(HierarchicalClustering *hc) {
    free(hc->labels);
    free(hc->merge);
    free(hc->merge_height);
    hc->labels = NULL;
    hc->merge  = NULL;
    hc->merge_height = NULL;
}

void hac_fit(HierarchicalClustering *hc,
             const double *X, size_t n_samples, size_t dim) {
    (void)hc; (void)X; (void)n_samples; (void)dim;
    /* Placeholder – the full O(n²)-distance implementation is omitted
       in this lightweight educational library.                       */
}

/* ──────────────────────────────────────────────
   Silhouette score
   ────────────────────────────────────────────── */

double silhouette_score(const double *X, const int *labels,
                        size_t n_samples, size_t n_features) {
    size_t c_max = 0;
    for (size_t i = 0; i < n_samples; ++i)
        if ((size_t)labels[i] > c_max) c_max = (size_t)labels[i];
    c_max++;

    double score = 0.0;
    for (size_t i = 0; i < n_samples; ++i) {
        /* a(i): mean distance to same cluster */
        double a_i = 0.0;
        size_t cnt_a = 0;
        for (size_t j = 0; j < n_samples; ++j) {
            if (i == j || labels[j] != labels[i]) continue;
            a_i += sqrt(euclidean_sq(X + i * n_features, X + j * n_features, n_features));
            cnt_a++;
        }
        if (cnt_a > 0) a_i /= (double)cnt_a;

        /* b(i): min mean distance to other clusters */
        double b_i = DBL_MAX;
        for (size_t c = 0; c < c_max; ++c) {
            if ((int)c == labels[i]) continue;
            double d = 0.0;
            size_t cnt = 0;
            for (size_t j = 0; j < n_samples; ++j) {
                if (labels[j] != (int)c) continue;
                d += sqrt(euclidean_sq(X + i * n_features, X + j * n_features, n_features));
                cnt++;
            }
            if (cnt > 0) {
                d /= (double)cnt;
                if (d < b_i) b_i = d;
            }
        }
        double denom = (a_i > b_i) ? a_i : b_i;
        if (denom > 0)
            score += (b_i - a_i) / denom;
    }
    return score / (double)n_samples;
}
