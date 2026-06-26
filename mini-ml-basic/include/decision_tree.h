#ifndef DECISION_TREE_H
#define DECISION_TREE_H

#include <stdbool.h>
#include <stddef.h>

/* ──────────────────────────────────────────────
   Decision Tree Node
   ────────────────────────────────────────────── */
typedef struct DTNode {
    int     feature_idx;       /* split feature index      */
    double  threshold;         /* split threshold           */
    int     label;             /* leaf class                */
    bool    is_leaf;
    double  impurity;          /* Gini / entropy at node    */
    size_t  n_samples;
    struct DTNode *left;
    struct DTNode *right;
} DTNode;

typedef struct {
    DTNode *root;
    size_t  n_features;
    int     n_classes;
    int     max_depth;
    int     min_samples_split;
} DecisionTree;

/* splitting-criterion */
typedef enum {
    DT_CRITERION_GINI = 0,
    DT_CRITERION_ENTROPY
} DTCriterion;

DecisionTree  dt_create(size_t n_features, int n_classes,
                        int max_depth, int min_samples_split);
void          dt_destroy(DecisionTree *tree);

double  dt_gini_impurity(const int *labels, size_t n);
double  dt_entropy(const int *labels, size_t n);
double  dt_information_gain(const int *labels,
                            const int *left, size_t nl,
                            const int *right, size_t nr,
                            DTCriterion crit);

void    dt_fit(DecisionTree *tree,
               const double *X, const int *y,
               size_t n_samples, DTCriterion crit);
int     dt_predict(const DecisionTree *tree, const double *x);

/* pruning */
void    dt_pre_prune(DecisionTree *tree);
void    dt_post_prune(DecisionTree *tree,
                      const double *X_val, const int *y_val,
                      size_t n_val);

/* ──────────────────────────────────────────────
   Random Forest
   ────────────────────────────────────────────── */
typedef struct {
    DecisionTree **trees;
    size_t         n_trees;
    size_t         n_features;
    int            n_classes;
    double         sample_ratio;     /* bootstrap ratio        */
    size_t         max_features;     /* feature bagging count  */
} RandomForest;

RandomForest  rf_create(size_t n_trees, size_t n_features, int n_classes,
                        double sample_ratio, size_t max_features,
                        int max_depth, int min_samples_split);
void          rf_destroy(RandomForest *rf);
void          rf_fit(RandomForest *rf,
                     const double *X, const int *y,
                     size_t n_samples, DTCriterion crit);
int           rf_predict(const RandomForest *rf, const double *x);
void          rf_feature_importance(const RandomForest *rf, double *out);

#endif /* DECISION_TREE_H */
