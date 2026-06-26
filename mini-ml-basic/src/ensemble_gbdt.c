#include "ensemble_gbdt.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ──────────────────────────────────────────────
   Bagging
   ────────────────────────────────────────────── */

BaggingModel bagging_create(size_t n_models,
                            WeakLearnerFit fit, WeakLearnerPredict pred,
                            WeakLearnerDestroy destroy,
                            double (*agg)(const double *, size_t)) {
    BaggingModel bg;
    bg.n_models = n_models;
    bg.fit_fn   = fit;
    bg.pred_fn  = pred;
    bg.destroy_fn = destroy;
    bg.agg_fn   = agg;
    bg.models   = (void **)calloc(n_models, sizeof(void *));
    bg.sample_ratio_num = 1;
    bg.sample_ratio_den = 1;
    return bg;
}

void bagging_destroy(BaggingModel *bg) {
    if (bg->destroy_fn)
        for (size_t i = 0; i < bg->n_models; ++i)
            if (bg->models[i]) bg->destroy_fn(bg->models[i]);
    free(bg->models);
    bg->models = NULL;
}

void bagging_fit(BaggingModel *bg,
                 const double *X, const double *y,
                 size_t n_samples, size_t n_features,
                 size_t seed) {
    srand((unsigned)seed);
    size_t bs = (size_t)((double)bg->sample_ratio_num / bg->sample_ratio_den * n_samples);
    if (bs == 0) bs = n_samples;
    double *Xb = (double *)malloc(bs * n_features * sizeof(double));
    double *yb = (double *)malloc(bs * sizeof(double));
    for (size_t m = 0; m < bg->n_models; ++m) {
        for (size_t i = 0; i < bs; ++i) {
            size_t idx = (size_t)rand() % n_samples;
            memcpy(Xb + i * n_features, X + idx * n_features, n_features * sizeof(double));
            yb[i] = y[idx];
        }
        bg->fit_fn(Xb, yb, bs, n_features, &bg->models[m]);
    }
    free(Xb); free(yb);
}

double bagging_predict(BaggingModel *bg, const double *x, size_t n_features) {
    double *preds = (double *)malloc(bg->n_models * sizeof(double));
    for (size_t i = 0; i < bg->n_models; ++i)
        preds[i] = bg->pred_fn(x, n_features, bg->models[i]);
    double result = bg->agg_fn(preds, bg->n_models);
    free(preds);
    return result;
}

/* ──────────────────────────────────────────────
   AdaBoost
   ────────────────────────────────────────────── */

AdaBoostModel adaboost_create(size_t n_estimators, size_t n_features) {
    AdaBoostModel ab;
    ab.n_stumps   = n_estimators;
    ab.n_features = n_features;
    ab.stumps = (AdaStump *)calloc(n_estimators, sizeof(AdaStump));
    return ab;
}

void adaboost_destroy(AdaBoostModel *ab) {
    free(ab->stumps);
    ab->stumps = NULL;
}

static AdaStump best_stump(const double *X, const int *y,
                           const double *w, size_t n, size_t d) {
    AdaStump best;
    best.feature_idx = -1;
    best.threshold   = 0.0;
    best.polarity    = 1;
    best.alpha       = 0.0;
    double min_err = 1e308;

    for (size_t f = 0; f < d; ++f) {
        for (size_t i = 0; i < n; ++i) {
            double thr = X[i * d + f];
            for (int pol = 1; pol >= -1; pol -= 2) {
                double err = 0.0;
                for (size_t j = 0; j < n; ++j) {
                    int pred = ((X[j * d + f] <= thr) ? pol : -pol);
                    if (pred != y[j]) err += w[j];
                }
                if (err < min_err) {
                    min_err = err;
                    best.feature_idx = (int)f;
                    best.threshold   = thr;
                    best.polarity    = pol;
                }
            }
        }
    }
    return best;
}

void adaboost_fit(AdaBoostModel *ab,
                  const double *X, const int *y,
                  size_t n_samples, size_t n_features) {
    size_t n = n_samples;
    double *w = (double *)malloc(n * sizeof(double));
    for (size_t i = 0; i < n; ++i) w[i] = 1.0 / n;

    for (size_t t = 0; t < ab->n_stumps; ++t) {
        ab->stumps[t] = best_stump(X, y, w, n, n_features);
        AdaStump *s = &ab->stumps[t];
        if (s->feature_idx < 0) break;

        double err = 0.0;
        for (size_t i = 0; i < n; ++i) {
            int pred = ((X[i * n_features + s->feature_idx] <= s->threshold)
                        ? s->polarity : -s->polarity);
            if (pred != y[i]) err += w[i];
        }
        err = fmax(err, 1e-12);
        s->alpha = 0.5 * log((1.0 - err) / err);

        double sum_w = 0.0;
        for (size_t i = 0; i < n; ++i) {
            int pred = ((X[i * n_features + s->feature_idx] <= s->threshold)
                        ? s->polarity : -s->polarity);
            w[i] *= exp(-s->alpha * y[i] * pred);
            sum_w += w[i];
        }
        for (size_t i = 0; i < n; ++i) w[i] /= sum_w;
    }
    free(w);
}

int adaboost_predict(const AdaBoostModel *ab, const double *x) {
    double s = 0.0;
    for (size_t t = 0; t < ab->n_stumps; ++t) {
        AdaStump *st = &ab->stumps[t];
        if (st->feature_idx < 0) continue;
        int pred = ((x[st->feature_idx] <= st->threshold) ? st->polarity : -st->polarity);
        s += st->alpha * pred;
    }
    return s >= 0.0 ? 1 : -1;
}

/* ──────────────────────────────────────────────
   Gradient Boosting (GBDT)
   ────────────────────────────────────────────── */

GBDTModel gbdt_create(size_t n_estimators, double lr,
                      size_t max_depth, size_t n_features) {
    GBDTModel gb;
    gb.n_estimators = n_estimators;
    gb.learning_rate = lr;
    gb.max_depth    = max_depth;
    gb.n_features   = n_features;
    gb.n_samples    = 0;
    gb.pred         = NULL;
    gb.left_child   = NULL;
    gb.right_child  = NULL;
    gb.split_feat   = NULL;
    gb.split_val    = NULL;
    gb.leaf_val     = NULL;
    return gb;
}

void gbdt_destroy(GBDTModel *gb) {
    free(gb->pred);
    free(gb->left_child);
    free(gb->right_child);
    free(gb->split_feat);
    free(gb->split_val);
    free(gb->leaf_val);
    memset(gb, 0, sizeof(GBDTModel));
}

/*
 * Simple regression tree builder for gradient boosting.
 * Fits a single CART tree to residuals using MSE splitting criterion.
 * Tree storage: uses pre-allocated flat arrays indexed by node id.
 * Each tree is stored as left_child, right_child, split_feat, split_val, leaf_val.
 *
 * Returns the root node index.
 */
typedef struct {
    const double *X;
    const double *y;
    size_t n;
    size_t d;
    size_t max_depth;
    /* shared flat tree storage */
    size_t  capacity;
    size_t  node_count;
    int    *left_child;
    int    *right_child;
    int    *split_feat;
    double *split_val;
    double *leaf_val;
} TreeBuilder;

static int tree_builder_new_node(TreeBuilder *tb) {
    int id = (int)tb->node_count;
    tb->node_count++;
    tb->left_child[id] = -1;
    tb->right_child[id] = -1;
    tb->split_feat[id] = -1;
    tb->split_val[id]  = 0.0;
    tb->leaf_val[id]   = 0.0;
    return id;
}

static int tree_builder_build(TreeBuilder *tb, const size_t *indices,
                               size_t n_samples, size_t depth) {
    int node_id = tree_builder_new_node(tb);

    /* Compute mean of residuals for this node */
    double sum_y = 0.0;
    for (size_t i = 0; i < n_samples; ++i)
        sum_y += tb->y[indices[i]];
    double mean_y = sum_y / (double)n_samples;

    if (depth >= tb->max_depth || n_samples <= 1) {
        tb->leaf_val[node_id] = mean_y;
        return node_id;
    }

    /* Find best split */
    double best_gain = DBL_MAX;
    int    best_feat = -1;
    double best_thr  = 0.0;
    double best_left_mean = 0.0;

    for (size_t f = 0; f < tb->d; ++f) {
        for (size_t p = 0; p < n_samples; ++p) {
            double thr = tb->X[indices[p] * tb->d + f];
            double sl = 0.0, sr = 0.0;
            size_t nl = 0, nr = 0;
            for (size_t j = 0; j < n_samples; ++j) {
                size_t idx = indices[j];
                if (tb->X[idx * tb->d + f] <= thr) {
                    sl += tb->y[idx]; nl++;
                } else {
                    sr += tb->y[idx]; nr++;
                }
            }
            if (nl == 0 || nr == 0) continue;
            double ml = sl / (double)nl, mr = sr / (double)nr;
            double gain = 0.0;
            for (size_t j = 0; j < n_samples; ++j) {
                size_t idx = indices[j];
                double r = tb->y[idx];
                double pred = (tb->X[idx * tb->d + f] <= thr) ? ml : mr;
                double err = r - pred;
                gain += err * err;
            }
            if (gain < best_gain) {
                best_gain  = gain;
                best_feat  = (int)f;
                best_thr   = thr;
                best_left_mean  = ml;
            }
        }
    }

    if (best_feat < 0) {
        tb->leaf_val[node_id] = mean_y;
        return node_id;
    }

    /* Split indices */
    size_t *left_idx  = (size_t *)malloc(n_samples * sizeof(size_t));
    size_t *right_idx = (size_t *)malloc(n_samples * sizeof(size_t));
    size_t nl = 0, nr = 0;
    for (size_t i = 0; i < n_samples; ++i) {
        if (tb->X[indices[i] * tb->d + best_feat] <= best_thr)
            left_idx[nl++]  = indices[i];
        else
            right_idx[nr++] = indices[i];
    }

    tb->split_feat[node_id] = best_feat;
    tb->split_val[node_id]  = best_thr;
    tb->leaf_val[node_id]   = best_left_mean;  /* stored but internal nodes use children */

    if (nl > 0)
        tb->left_child[node_id]  = tree_builder_build(tb, left_idx, nl, depth + 1);
    if (nr > 0)
        tb->right_child[node_id] = tree_builder_build(tb, right_idx, nr, depth + 1);

    free(left_idx);
    free(right_idx);
    return node_id;
}

void gbdt_fit(GBDTModel *gb,
              const double *X, const double *y,
              size_t n_samples, size_t seed) {
    (void)seed;
    gb->n_samples = n_samples;
    size_t d = gb->n_features;
    size_t n_est = gb->n_estimators;
    size_t max_nodes = n_est * (size_t)(1 << (gb->max_depth + 1));  /* upper bound per tree */

    /* Allocate flat tree storage */
    gb->pred        = (double *)calloc(n_samples, sizeof(double));
    gb->left_child  = (int    *)calloc(max_nodes, sizeof(int));
    gb->right_child = (int    *)calloc(max_nodes, sizeof(int));
    gb->split_feat  = (int    *)calloc(max_nodes, sizeof(int));
    gb->split_val   = (double *)calloc(max_nodes, sizeof(double));
    gb->leaf_val    = (double *)calloc(max_nodes, sizeof(double));

    /* Initialise with mean of y */
    double mean_y = 0.0;
    for (size_t i = 0; i < n_samples; ++i) mean_y += y[i];
    mean_y /= (double)n_samples;
    for (size_t i = 0; i < n_samples; ++i) gb->pred[i] = mean_y;

    size_t *indices = (size_t *)malloc(n_samples * sizeof(size_t));
    for (size_t i = 0; i < n_samples; ++i) indices[i] = i;

    TreeBuilder tb;
    tb.X = X;
    tb.d = d;
    tb.max_depth = gb->max_depth;
    tb.capacity = max_nodes;
    tb.node_count = 0;

    for (size_t t = 0; t < n_est; ++t) {
        /* Compute residuals */
        double *residuals = (double *)malloc(n_samples * sizeof(double));
        for (size_t i = 0; i < n_samples; ++i)
            residuals[i] = y[i] - gb->pred[i];

        /* Point tree builder at current storage region */
        size_t base = t * (size_t)(1 << (gb->max_depth + 1));
        tb.n          = n_samples;
        tb.y          = residuals;
        tb.left_child  = gb->left_child  + base;
        tb.right_child = gb->right_child + base;
        tb.split_feat  = gb->split_feat  + base;
        tb.split_val   = gb->split_val   + base;
        tb.leaf_val    = gb->leaf_val    + base;

        tree_builder_build(&tb, indices, n_samples, 0);

        /* Update predictions: traverse tree and add lr * leaf_val */
        for (size_t i = 0; i < n_samples; ++i) {
            int node = 0;
            /* traverse until leaf */
            while (true) {
                int feat = gb->split_feat[base + node];
                if (feat < 0) {
                    gb->pred[i] += gb->learning_rate * gb->leaf_val[base + node];
                    break;
                }
                double val = X[i * d + feat];
                if (val <= gb->split_val[base + node])
                    node = gb->left_child[base + node];
                else
                    node = gb->right_child[base + node];
                if (node < 0) {
                    gb->pred[i] += gb->learning_rate * gb->leaf_val[base + 0];
                    break;
                }
            }
        }
        free(residuals);
    }
    free(indices);
}

/* Traverse all trees and sum leaf contributions */
static double gbdt_traverse_tree(const GBDTModel *gb, const double *x,
                                  size_t tree_idx, size_t base_nodes) {
    size_t base = tree_idx * base_nodes;
    int node = 0;
    while (true) {
        int feat = gb->split_feat[base + node];
        if (feat < 0) return gb->leaf_val[base + node];
        double val = x[feat];
        if (val <= gb->split_val[base + node])
            node = gb->left_child[base + node];
        else
            node = gb->right_child[base + node];
        if (node < 0) return gb->leaf_val[base + 0];
    }
}

double gbdt_predict(const GBDTModel *gb, const double *x) {
    double sum = 0.0;
    size_t base_nodes = (size_t)(1 << (gb->max_depth + 1));
    for (size_t t = 0; t < gb->n_estimators; ++t)
        sum += gbdt_traverse_tree(gb, x, t, base_nodes);
    return sum;
}

void gbdt_predict_batch(const GBDTModel *gb, const double *X,
                        double *out, size_t n) {
    for (size_t i = 0; i < n; ++i)
        out[i] = gbdt_predict(gb, X + i * gb->n_features);
}

/* ──────────────────────────────────────────────
   XGBoost-style
   ────────────────────────────────────────────── */

XGBoostModel xgb_create(size_t n_estimators, double eta,
                         double lambda, double gamma,
                         size_t max_depth, size_t n_features) {
    XGBoostModel xgb;
    xgb.n_estimators = n_estimators;
    xgb.eta    = eta;
    xgb.lambda = lambda;
    xgb.gamma  = gamma;
    xgb.base_score  = 0.0;
    xgb.max_depth  = max_depth;
    xgb.n_features = n_features;
    xgb.left_child = NULL;
    xgb.right_child = NULL;
    xgb.split_feat  = NULL;
    xgb.split_val   = NULL;
    xgb.leaf_score  = NULL;
    return xgb;
}

void xgb_destroy(XGBoostModel *xgb) {
    free(xgb->left_child);
    free(xgb->right_child);
    free(xgb->split_feat);
    free(xgb->split_val);
    free(xgb->leaf_score);
    memset(xgb, 0, sizeof(XGBoostModel));
}

/*
 * XGBoost-style fitting using second-order Taylor expansion of loss.
 *
 * For squared-error regression (L2 loss):
 *   gᵢ = ∂L/∂ŷ = ŷᵢ − yᵢ  (negative residual)
 *   hᵢ = ∂²L/∂ŷ² = 1       (constant)
 *
 * The optimal leaf weight for leaf j with sample set Iⱼ:
 *   wⱼ* = − Σ_{i∈Iⱼ} gᵢ / (Σ_{i∈Iⱼ} hᵢ + λ)
 *
 * Split gain:
 *   Gain = ½ [ (G_L²/(H_L+λ) + G_R²/(H_R+λ) − (G_L+G_R)²/(H_L+H_R+λ) ] − γ
 *   where G_L = Σ_{i∈L} gᵢ, H_L = Σ_{i∈L} hᵢ
 *
 * This matches the XGBoost paper (Chen & Guestrin, 2016) for the
 * simple case of regression with L2 loss and L2 regularisation.
 */
void xgb_fit(XGBoostModel *xgb,
             const double *X, const double *y,
             size_t n_samples, size_t seed) {
    (void)seed;
    size_t d = xgb->n_features;
    size_t n_est = xgb->n_estimators;
    size_t max_depth = xgb->max_depth;
    double lambda = xgb->lambda;
    double gamma  = xgb->gamma;
    double eta    = xgb->eta;
    size_t max_nodes = n_est * (size_t)(1 << (max_depth + 1));

    /* Allocate storage */
    xgb->left_child  = (int    *)calloc(max_nodes, sizeof(int));
    xgb->right_child = (int    *)calloc(max_nodes, sizeof(int));
    xgb->split_feat  = (int    *)calloc(max_nodes, sizeof(int));
    xgb->split_val   = (double *)calloc(max_nodes, sizeof(double));
    xgb->leaf_score  = (double *)calloc(max_nodes, sizeof(double));

    /* Initial prediction: mean of y (L2 optimal constant) */
    double *pred = (double *)calloc(n_samples, sizeof(double));
    double mean_y = 0.0;
    for (size_t i = 0; i < n_samples; ++i) mean_y += y[i];
    mean_y /= (double)n_samples;
    xgb->base_score = mean_y;
    for (size_t i = 0; i < n_samples; ++i) pred[i] = mean_y;

    size_t base_nodes = (size_t)(1 << (max_depth + 1));

    for (size_t t = 0; t < n_est; ++t) {
        /* Compute first and second order gradients for L2 loss */
        double *g = (double *)malloc(n_samples * sizeof(double));
        double *h = (double *)malloc(n_samples * sizeof(double));
        for (size_t i = 0; i < n_samples; ++i) {
            g[i] = pred[i] - y[i];   /* gradient of L2 */
            h[i] = 1.0;               /* hessian of L2  */
        }

        /* Build a single regression tree optimising the XGBoost objective */
        /* Simple recursive split-finding at each node */
        size_t *indices = (size_t *)malloc(n_samples * sizeof(size_t));
        for (size_t i = 0; i < n_samples; ++i) indices[i] = i;

        size_t base = t * base_nodes;
        /* Initialize first node as leaf */
        xgb->split_feat[base + 0] = -1;  /* mark as leaf */

        /* Build tree: recursively find best splits using gradient sums */
        /* Stack-based iterative builder */
        /* For simplicity, a single-level split (stump) with exhaustive search */
        {
            double G_all = 0.0, H_all = 0.0;
            for (size_t i = 0; i < n_samples; ++i) { G_all += g[i]; H_all += h[i]; }

            double best_gain = 0.0;
            int    best_feat = -1;
            double best_thr  = 0.0;

            /* Exhaustive search over features and thresholds */
            for (size_t f = 0; f < d; ++f) {
                for (size_t p = 0; p < n_samples && p < 200; ++p) {
                    double thr = X[indices[p] * d + f];
                    double GL = 0.0, HL = 0.0, GR = 0.0, HR = 0.0;
                    size_t nl = 0;
                    for (size_t i = 0; i < n_samples; ++i) {
                        size_t idx = indices[i];
                        if (X[idx * d + f] <= thr) {
                            GL += g[idx]; HL += h[idx]; nl++;
                        } else {
                            GR += g[idx]; HR += h[idx];
                        }
                    }
                    if (nl == 0 || nl == n_samples) continue;
                    double gain = (GL * GL) / (HL + lambda)
                                + (GR * GR) / (HR + lambda)
                                - (G_all * G_all) / (H_all + lambda);
                    gain = 0.5 * gain - gamma;
                    if (gain > best_gain) {
                        best_gain = gain;
                        best_feat = (int)f;
                        best_thr  = thr;
                    }
                }
            }

            if (best_feat >= 0) {
                /* Store split at root */
                xgb->split_feat[base + 0] = best_feat;
                xgb->split_val[base + 0]  = best_thr;

                /* Compute leaf weights for children */
                double GL = 0.0, HL = 0.0, GR = 0.0, HR = 0.0;
                for (size_t i = 0; i < n_samples; ++i) {
                    size_t idx = indices[i];
                    if (X[idx * d + best_feat] <= best_thr) {
                        GL += g[idx]; HL += h[idx];
                    } else {
                        GR += g[idx]; HR += h[idx];
                    }
                }
                /* Left leaf (node 1) */
                xgb->split_feat[base + 1]  = -1;
                xgb->leaf_score[base + 1]  = -GL / (HL + lambda);
                xgb->left_child[base + 0]  = 1;
                /* Right leaf (node 2) */
                xgb->split_feat[base + 2]  = -1;
                xgb->leaf_score[base + 2]  = -GR / (HR + lambda);
                xgb->right_child[base + 0] = 2;
            } else {
                /* No good split — single leaf */
                xgb->split_feat[base + 0] = -1;
                xgb->leaf_score[base + 0] = -G_all / (H_all + lambda);
            }
        }

        /* Update predictions */
        for (size_t i = 0; i < n_samples; ++i) {
            int node = 0;
            while (true) {
                int feat = xgb->split_feat[base + node];
                if (feat < 0) {
                    pred[i] += eta * xgb->leaf_score[base + node];
                    break;
                }
                if (X[i * d + feat] <= xgb->split_val[base + node])
                    node = xgb->left_child[base + node];
                else
                    node = xgb->right_child[base + node];
                if (node < 0) break;
            }
        }

        free(g); free(h); free(indices);
    }
    free(pred);
}

double xgb_predict(const XGBoostModel *xgb, const double *x) {
    double sum = xgb->base_score;
    size_t base_nodes = (size_t)(1 << (xgb->max_depth + 1));
    for (size_t t = 0; t < xgb->n_estimators; ++t) {
        size_t base = t * base_nodes;
        int node = 0;
        while (true) {
            int feat = xgb->split_feat[base + node];
            if (feat < 0) {
                sum += xgb->eta * xgb->leaf_score[base + node];
                break;
            }
            if (x[feat] <= xgb->split_val[base + node])
                node = xgb->left_child[base + node];
            else
                node = xgb->right_child[base + node];
            if (node < 0) break;
        }
    }
    return sum;
}

/* ──────────────────────────────────────────────
   Stacking
   ────────────────────────────────────────────── */

StackingModel stacking_create(size_t n_base, size_t n_features,
                              WeakLearnerFit fit, WeakLearnerPredict pred) {
    StackingModel st;
    st.n_base     = n_base;
    st.n_features = n_features;
    st.fit_fn     = fit;
    st.pred_fn    = pred;
    st.base_models = (void **)calloc(n_base, sizeof(void *));
    st.meta_coefs  = (double *)calloc(n_base, sizeof(double));
    st.meta_bias   = 0.0;
    return st;
}

void stacking_destroy(StackingModel *st) {
    free(st->base_models);
    free(st->meta_coefs);
    st->base_models = NULL;
    st->meta_coefs  = NULL;
}

void stacking_fit(StackingModel *st,
                  const double *X, const double *y,
                  size_t n_samples, size_t seed) {
    (void)seed;
    size_t d = st->n_features;
    for (size_t m = 0; m < st->n_base; ++m)
        st->fit_fn(X, y, n_samples, d, &st->base_models[m]);
    /* fit meta-learner (simple average → equal weights) */
    for (size_t m = 0; m < st->n_base; ++m)
        st->meta_coefs[m] = 1.0 / (double)st->n_base;
    st->meta_bias = 0.0;
}

double stacking_predict(const StackingModel *st, const double *x) {
    double s = st->meta_bias;
    for (size_t m = 0; m < st->n_base; ++m)
        s += st->meta_coefs[m] * st->pred_fn(x, st->n_features, st->base_models[m]);
    return s;
}
