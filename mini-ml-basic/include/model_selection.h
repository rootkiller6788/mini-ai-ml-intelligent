#ifndef MODEL_SELECTION_H
#define MODEL_SELECTION_H

#include <stdbool.h>
#include <stddef.h>

/*
 * model_selection.h ? Model Evaluation & Hyperparameter Tuning
 *
 * L1: ConfusionMatrix struct, CrossValidator struct, metric types
 * L2: Bias-variance tradeoff ? underfitting vs overfitting detection
 * L3: K-fold stratified cross-validation pipeline
 * L4: Central Limit Theorem ? confidence intervals via resampling
 * L5: Precision/Recall/F1, ROC-AUC computation, GridSearch
 */

/* ??????????????????????????????????????????????
   Confusion Matrix & Classification Metrics
   ?????????????????????????????????????????????? */
typedef struct {
    int tp, tn, fp, fn;          /* binary confusion matrix entries  */
    int n_classes;               /* >2 implies macro/micro averaging  */
} ConfusionMatrix;

typedef enum {
    METRIC_ACCURACY = 0,
    METRIC_PRECISION,
    METRIC_RECALL,
    METRIC_F1,
    METRIC_SPECIFICITY
} ClassificationMetric;

ConfusionMatrix  cm_create(void);
void             cm_record(ConfusionMatrix *cm, int actual, int predicted);
double           cm_accuracy(const ConfusionMatrix *cm);
double           cm_precision(const ConfusionMatrix *cm);
double           cm_recall(const ConfusionMatrix *cm);
double           cm_f1_score(const ConfusionMatrix *cm);
double           cm_specificity(const ConfusionMatrix *cm);
void             cm_reset(ConfusionMatrix *cm);
void             cm_print(const ConfusionMatrix *cm);

/* Multi-class confusion matrix (stored as flat array) */
typedef struct {
    int    *matrix;              /* (n_classes ? n_classes) row-major */
    int     n_classes;
    int     total;
} MultiConfusionMatrix;

MultiConfusionMatrix  mcm_create(int n_classes);
void                  mcm_destroy(MultiConfusionMatrix *mcm);
void                  mcm_record(MultiConfusionMatrix *mcm,
                                  int actual, int predicted);
double                mcm_accuracy(const MultiConfusionMatrix *mcm);
/* per-class precision/recall/F1 */
void                  mcm_per_class_precision(const MultiConfusionMatrix *mcm,
                                              double *out);
void                  mcm_per_class_recall(const MultiConfusionMatrix *mcm,
                                           double *out);
double                mcm_macro_f1(const MultiConfusionMatrix *mcm);

/* ??????????????????????????????????????????????
   Regression Metrics
   ?????????????????????????????????????????????? */
double  metric_mse(const double *y_true, const double *y_pred, size_t n);
double  metric_rmse(const double *y_true, const double *y_pred, size_t n);
double  metric_mae(const double *y_true, const double *y_pred, size_t n);
double  metric_r2(const double *y_true, const double *y_pred, size_t n);
double  metric_mape(const double *y_true, const double *y_pred, size_t n);

/* ??????????????????????????????????????????????
   K-Fold Cross-Validation
   ?????????????????????????????????????????????? */
typedef double (*CVPredictFn)(const void *model, const double *x);
typedef void   (*CVFitFn)(void *model, const double *X_train, const double *y_train,
                          size_t n_train);
typedef void * (*CVCreateFn)(void);
typedef void   (*CVDestroyFn)(void *model);

typedef struct {
    size_t  k;
    bool    shuffle;
    size_t  seed;
    size_t  n_samples;
    size_t *train_indices;       /* per fold: pre-computed indices    */
    size_t *test_indices;
} KFoldCV;

KFoldCV  kfold_create(size_t k, bool shuffle, size_t seed);
void     kfold_destroy(KFoldCV *cv);
void     kfold_split(KFoldCV *cv, size_t n_samples);

/* Regression CV: returns average metric across folds */
double   kfold_regression(KFoldCV *cv,
                          CVCreateFn create, CVFitFn fit,
                          CVPredictFn predict, CVDestroyFn destroy,
                          const double *X, const double *y,
                          size_t n, size_t n_features,
                          double (*metric)(const double*, const double*, size_t));

/* Classification CV: returns average metric */
double   kfold_classification(KFoldCV *cv,
                              CVCreateFn create, CVFitFn fit,
                              CVPredictFn predict, CVDestroyFn destroy,
                              const double *X, const int *y,
                              size_t n, size_t n_features, int n_classes,
                              ClassificationMetric metric);

/* ??????????????????????????????????????????????
   Train/Test Split
   ?????????????????????????????????????????????? */
void    train_test_split_f64(const double *X, const double *y, size_t n, size_t n_features,
                               double test_ratio, size_t seed,
                               double **X_train, double **X_test,
                               double **y_train, double **y_test,
                               size_t *n_train, size_t *n_test);

void    train_test_split_i32(const double *X, const int *y, size_t n, size_t n_features,
                               double test_ratio, size_t seed,
                               double **X_train, double **X_test,
                               int    **y_train, int    **y_test,
                               size_t *n_train, size_t *n_test);

/* ??????????????????????????????????????????????
   Grid Search (simple ? discrete parameter scan)
   ?????????????????????????????????????????????? */
typedef struct {
    double *param_values;
    size_t  n_values;
    const char *name;
} GSParam;

/* Returns best param index; scores[i] filled with scores */
size_t  grid_search_regression(
    CVCreateFn create, CVFitFn fit, CVPredictFn predict, CVDestroyFn destroy,
    const double *X, const double *y, size_t n, size_t n_features,
    GSParam *params, size_t n_params, size_t k_folds, size_t seed,
    double *best_score);

#endif /* MODEL_SELECTION_H */
