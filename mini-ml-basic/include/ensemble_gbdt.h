#ifndef ENSEMBLE_GBDT_H
#define ENSEMBLE_GBDT_H

#include <stdbool.h>
#include <stddef.h>

/* ──────────────────────────────────────────────
   Bagging (Bootstrap Aggregating)
   ────────────────────────────────────────────── */
typedef double (*WeakLearnerFit)(const double *X, const double *y,
                                 size_t n_samples, size_t n_features,
                                 void *state);
typedef double (*WeakLearnerPredict)(const double *x, size_t n_features,
                                     void *state);
typedef void   (*WeakLearnerDestroy)(void *state);

typedef struct {
    void       **models;              /* array of per-learner state     */
    size_t       n_models;
    size_t       sample_ratio_num;    /* numerator of bootstrap ratio   */
    size_t       sample_ratio_den;
    WeakLearnerFit     fit_fn;
    WeakLearnerPredict pred_fn;
    WeakLearnerDestroy destroy_fn;
    double     (*agg_fn)(const double *preds, size_t n);  /* aggregation */
} BaggingModel;

BaggingModel  bagging_create(size_t n_models,
                             WeakLearnerFit fit, WeakLearnerPredict pred,
                             WeakLearnerDestroy destroy,
                             double (*agg)(const double *, size_t));
void          bagging_destroy(BaggingModel *bg);
void          bagging_fit(BaggingModel *bg,
                          const double *X, const double *y,
                          size_t n_samples, size_t n_features,
                          size_t seed);
double        bagging_predict(BaggingModel *bg, const double *x, size_t n_features);

/* ──────────────────────────────────────────────
   AdaBoost (binary, +1 / -1 labels)
   ────────────────────────────────────────────── */
typedef struct AdaStump {
    int     feature_idx;
    double  threshold;
    int     polarity;          /* ±1                         */
    double  alpha;             /* weight in strong classifier */
} AdaStump;

typedef struct {
    AdaStump *stumps;
    size_t    n_stumps;
    size_t    n_features;
} AdaBoostModel;

AdaBoostModel  adaboost_create(size_t n_estimators, size_t n_features);
void           adaboost_destroy(AdaBoostModel *ab);
void           adaboost_fit(AdaBoostModel *ab,
                            const double *X, const int *y,
                            size_t n_samples, size_t n_features);
int            adaboost_predict(const AdaBoostModel *ab, const double *x);

/* ──────────────────────────────────────────────
   Gradient Boosting (regression)
   ────────────────────────────────────────────── */
typedef struct {
    double *pred;              /* cumulative prediction (n,)   */
    size_t  n_estimators;
    double  learning_rate;
    size_t  max_depth;
    size_t  n_samples;
    size_t  n_features;
    /* Tree ensemble stored as a flat struct array */
    int    *left_child;
    int    *right_child;
    int    *split_feat;
    double *split_val;
    double *leaf_val;          /* value at each leaf node       */
} GBDTModel;

GBDTModel  gbdt_create(size_t n_estimators, double lr,
                       size_t max_depth, size_t n_features);
void       gbdt_destroy(GBDTModel *gb);
void       gbdt_fit(GBDTModel *gb,
                    const double *X, const double *y,
                    size_t n_samples, size_t seed);
double     gbdt_predict(const GBDTModel *gb, const double *x);
void       gbdt_predict_batch(const GBDTModel *gb, const double *X,
                              double *out, size_t n);

/* ──────────────────────────────────────────────
   XGBoost-style (second-order Taylor, L2 reg)
   ────────────────────────────────────────────── */
typedef struct {
    double  lambda;            /* L2 regularisation             */
    double  gamma;             /* min split loss reduction      */
    double  eta;               /* learning rate                 */
    double  base_score;        /* initial prediction (mean of y)*/
    size_t  n_estimators;
    size_t  max_depth;
    size_t  n_features;
    /* internal tree storage */
    int    *left_child;
    int    *right_child;
    int    *split_feat;
    double *split_val;
    double *leaf_score;
} XGBoostModel;

XGBoostModel  xgb_create(size_t n_estimators, double eta,
                         double lambda, double gamma,
                         size_t max_depth, size_t n_features);
void          xgb_destroy(XGBoostModel *xgb);
void          xgb_fit(XGBoostModel *xgb,
                      const double *X, const double *y,
                      size_t n_samples, size_t seed);
double        xgb_predict(const XGBoostModel *xgb, const double *x);

/* ──────────────────────────────────────────────
   Stacking (meta-learner)
   ────────────────────────────────────────────── */
typedef struct {
    void      **base_models;        /* (n_models,)              */
    size_t      n_base;
    size_t      n_features;
    WeakLearnerFit     fit_fn;
    WeakLearnerPredict pred_fn;
    /* Meta-learner: simple linear regression */
    double     *meta_coefs;
    double      meta_bias;
} StackingModel;

StackingModel  stacking_create(size_t n_base, size_t n_features,
                               WeakLearnerFit fit, WeakLearnerPredict pred);
void           stacking_destroy(StackingModel *st);
void           stacking_fit(StackingModel *st,
                            const double *X, const double *y,
                            size_t n_samples, size_t seed);
double         stacking_predict(const StackingModel *st, const double *x);

#endif /* ENSEMBLE_GBDT_H */
