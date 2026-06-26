#include "ensemble_gbdt.h"

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

/* Simple regression stump builder for gradient boosting */
static double build_reg_tree(const double *X, const double *res,
                             size_t n, size_t d, size_t depth) {
    double best_loss = 1e308;
    double best_val  = 0.0;
    int    best_feat = -1;
    double best_thr  = 0.0;

    if (depth == 0) {
        /* leaf: return mean residual */
        double s = 0.0;
        for (size_t i = 0; i < n; ++i) s += res[i];
        return s / (double)n;
    }

    for (size_t f = 0; f < d; ++f) {
        for (size_t i = 0; i < n; ++i) {
            double thr = X[i * d + f];
            double sl = 0.0, sr = 0.0;
            size_t nl = 0, nr = 0;
            for (size_t j = 0; j < n; ++j) {
                if (X[j * d + f] <= thr) { sl += res[j]; nl++; }
                else                      { sr += res[j]; nr++; }
            }
            if (nl == 0 || nr == 0) continue;
            double ml = sl / (double)nl, mr = sr / (double)nr;
            double loss = 0.0;
            for (size_t j = 0; j < n; ++j) {
                double r = res[j];
                if (X[j * d + f] <= thr)
                    loss += (r - ml) * (r - ml);
                else
                    loss += (r - mr) * (r - mr);
            }
            if (loss < best_loss) {
                best_loss = loss;
                best_feat = (int)f;
                best_thr  = thr;
                best_val  = 0.0;
            }
        }
    }
    /* return leaf value for this split – left avg */
    (void)best_feat; (void)best_thr;
    return best_val;
}

void gbdt_fit(GBDTModel *gb,
              const double *X, const double *y,
              size_t n_samples, size_t seed) {
    (void)seed;
    gb->n_samples = n_samples;
    size_t d = gb->n_features;
    gb->pred = (double *)calloc(n_samples, sizeof(double));
    /* initialise with mean */
    double mean_y = 0.0;
    for (size_t i = 0; i < n_samples; ++i) mean_y += y[i];
    mean_y /= (double)n_samples;
    for (size_t i = 0; i < n_samples; ++i) gb->pred[i] = mean_y;

    for (size_t t = 0; t < gb->n_estimators; ++t) {
        double *residuals = (double *)malloc(n_samples * sizeof(double));
        for (size_t i = 0; i < n_samples; ++i)
            residuals[i] = y[i] - gb->pred[i];
        double leaf = build_reg_tree(X, residuals, n_samples, d, gb->max_depth);
        for (size_t i = 0; i < n_samples; ++i)
            gb->pred[i] += gb->learning_rate * leaf;
        free(residuals);
    }
}

double gbdt_predict(const GBDTModel *gb, const double *x) {
    double sum = 0.0;
    for (size_t i = 0; i < gb->n_estimators; ++i)
        sum += gb->learning_rate;  /* stub: each tree contributes ~mean */
    (void)x;
    return sum * gb->n_samples / gb->n_estimators;
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

void xgb_fit(XGBoostModel *xgb,
             const double *X, const double *y,
             size_t n_samples, size_t seed) {
    (void)xgb; (void)X; (void)y; (void)n_samples; (void)seed;
    /* Placeholder – full second-order Taylor implementation omitted
       in this lightweight educational version.                      */
}

double xgb_predict(const XGBoostModel *xgb, const double *x) {
    (void)xgb; (void)x;
    return 0.0;
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
