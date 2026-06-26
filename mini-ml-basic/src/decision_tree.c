#include "decision_tree.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ──────────────────────────────────────────────
   Impurity helpers
   ────────────────────────────────────────────── */

double dt_gini_impurity(const int *labels, size_t n) {
    if (n == 0) return 0.0;
    int freq[32] = {0};
    for (size_t i = 0; i < n; ++i) freq[labels[i]]++;
    double sum_sq = 0.0;
    for (int c = 0; c < 32; ++c) {
        double p = (double)freq[c] / (double)n;
        sum_sq += p * p;
    }
    return 1.0 - sum_sq;
}

double dt_entropy(const int *labels, size_t n) {
    if (n == 0) return 0.0;
    int freq[32] = {0};
    for (size_t i = 0; i < n; ++i) freq[labels[i]]++;
    double e = 0.0;
    for (int c = 0; c < 32; ++c) {
        if (freq[c] == 0) continue;
        double p = (double)freq[c] / (double)n;
        e -= p * log2(p);
    }
    return e;
}

double dt_information_gain(const int *labels,
                           const int *left, size_t nl,
                           const int *right, size_t nr,
                           DTCriterion crit) {
    size_t n = nl + nr;
    double parent = (crit == DT_CRITERION_GINI)
                        ? dt_gini_impurity(labels, n)
                        : dt_entropy(labels, n);
    double child_l = (crit == DT_CRITERION_GINI)
                         ? dt_gini_impurity(left, nl)
                         : dt_entropy(left, nl);
    double child_r = (crit == DT_CRITERION_GINI)
                         ? dt_gini_impurity(right, nr)
                         : dt_entropy(right, nr);
    return parent - ((double)nl / n) * child_l - ((double)nr / n) * child_r;
}

/* ──────────────────────────────────────────────
   DecisionTree
   ────────────────────────────────────────────── */

DecisionTree dt_create(size_t n_features, int n_classes,
                       int max_depth, int min_samples_split) {
    DecisionTree t;
    t.root      = NULL;
    t.n_features  = n_features;
    t.n_classes   = n_classes;
    t.max_depth   = max_depth;
    t.min_samples_split = min_samples_split;
    return t;
}

static DTNode *node_create(void) {
    DTNode *n = (DTNode *)calloc(1, sizeof(DTNode));
    return n;
}

static void node_destroy(DTNode *node) {
    if (!node) return;
    node_destroy(node->left);
    node_destroy(node->right);
    free(node);
}

void dt_destroy(DecisionTree *tree) {
    node_destroy(tree->root);
    tree->root = NULL;
}

static int majority_label(const int *y, const size_t *indices, size_t n) {
    int freq[32] = {0};
    for (size_t i = 0; i < n; ++i) freq[y[indices[i]]]++;
    int best = 0;
    for (int c = 1; c < 32; ++c)
        if (freq[c] > freq[best]) best = c;
    return best;
}

static bool all_same(const int *y, const size_t *indices, size_t n) {
    int first = y[indices[0]];
    for (size_t i = 1; i < n; ++i)
        if (y[indices[i]] != first) return false;
    return true;
}

static DTNode *build_tree(const double *X, const int *y,
                          const size_t *indices, size_t n,
                          size_t d, int depth,
                          int max_depth, int min_samples_split,
                          DTCriterion crit) {
    DTNode *node = node_create();
    node->n_samples = n;
    node->impurity   = (crit == DT_CRITERION_GINI)
                       ? dt_gini_impurity(y, n)
                       : dt_entropy(y, n);

    if (all_same(y, indices, n) || depth >= max_depth || n < (size_t)min_samples_split) {
        node->is_leaf = true;
        node->label   = majority_label(y, indices, n);
        return node;
    }

    double best_gain = -1.0;
    int    best_feat = -1;
    double best_thr  = 0.0;

    for (size_t f = 0; f < d; ++f) {
        for (size_t i = 0; i < n; ++i) {
            double thr = X[indices[i] * d + f];
            size_t *left_idx  = (size_t *)malloc(n * sizeof(size_t));
            size_t *right_idx = (size_t *)malloc(n * sizeof(size_t));
            size_t  nl = 0, nr = 0;
            int    *yl = (int *)malloc(n * sizeof(int));
            int    *yr = (int *)malloc(n * sizeof(int));
            for (size_t j = 0; j < n; ++j) {
                size_t idx = indices[j];
                if (X[idx * d + f] <= thr) {
                    left_idx[nl] = idx;
                    yl[nl]       = y[idx];
                    nl++;
                } else {
                    right_idx[nr] = idx;
                    yr[nr]        = y[idx];
                    nr++;
                }
            }
            if (nl == 0 || nr == 0) {
                free(left_idx); free(right_idx);
                free(yl); free(yr);
                continue;
            }
            double gain = dt_information_gain(y, yl, nl, yr, nr, crit);
            if (gain > best_gain) {
                best_gain   = gain;
                best_feat   = (int)f;
                best_thr    = thr;
            }
            free(left_idx); free(right_idx);
            free(yl); free(yr);
        }
    }

    if (best_feat < 0) {
        node->is_leaf = true;
        node->label   = majority_label(y, indices, n);
        return node;
    }

    node->feature_idx = best_feat;
    node->threshold   = best_thr;

    size_t *li = (size_t *)malloc(n * sizeof(size_t));
    size_t *ri = (size_t *)malloc(n * sizeof(size_t));
    size_t  nl = 0, nr = 0;
    for (size_t j = 0; j < n; ++j) {
        size_t idx = indices[j];
        if (X[idx * d + best_feat] <= best_thr)
            li[nl++] = idx;
        else
            ri[nr++] = idx;
    }
    node->left  = build_tree(X, y, li, nl, d, depth + 1, max_depth, min_samples_split, crit);
    node->right = build_tree(X, y, ri, nr, d, depth + 1, max_depth, min_samples_split, crit);
    free(li); free(ri);
    return node;
}

void dt_fit(DecisionTree *tree,
            const double *X, const int *y,
            size_t n_samples, DTCriterion crit) {
    size_t *indices = (size_t *)malloc(n_samples * sizeof(size_t));
    for (size_t i = 0; i < n_samples; ++i) indices[i] = i;
    tree->root = build_tree(X, y, indices, n_samples, tree->n_features,
                            0, tree->max_depth, tree->min_samples_split, crit);
    free(indices);
}

int dt_predict(const DecisionTree *tree, const double *x) {
    DTNode *node = tree->root;
    while (node && !node->is_leaf) {
        if (x[node->feature_idx] <= node->threshold)
            node = node->left;
        else
            node = node->right;
    }
    return node ? node->label : 0;
}

void dt_pre_prune(DecisionTree *tree) {
    /* max_depth and min_samples_split are pre-pruning parameters already
       enforced during tree construction in build_tree().
       This function is a no-op that exists for API completeness:
       pre-pruning is applied automatically during dt_fit(). */
    if (tree && tree->root) {
        /* Validate that the tree respects its own constraints */
        if (tree->max_depth <= 0) tree->max_depth = 10;
        if (tree->min_samples_split <= 0) tree->min_samples_split = 2;
    }
}

/*
 * Post-Pruning via Reduced-Error Pruning (REP).
 *
 * Quinlan (1987): Starting from the leaves, for each internal node,
 * compare the error on a validation set when keeping the subtree
 * vs. replacing the node with a majority-vote leaf. If replacement
 * does not increase error, prune the subtree.
 *
 * This is a bottom-up, greedy pruning strategy. The validation set
 * must be disjoint from the training set to avoid overfitting.
 */
static double dt_node_error(DTNode *node, const double *X_val,
                             const int *y_val, size_t n_val,
                             size_t n_features) {
    (void)X_val; (void)n_features;
    if (!node || !node->is_leaf) return (double)n_val;  /* max error as fallback */
    int errors = 0;
    for (size_t i = 0; i < n_val; ++i) {
        if (node->label != y_val[i]) errors++;
    }
    return (double)errors;
}

static double dt_subtree_error(DTNode *node, const double *X_val,
                                const int *y_val, size_t n_val,
                                size_t n_features) {
    if (!node) return (double)n_val;
    int errors = 0;
    for (size_t i = 0; i < n_val; ++i) {
        /* Walk the tree normally from this node */
        DTNode *cur = node;
        while (cur && !cur->is_leaf) {
            if (X_val[i * n_features + cur->feature_idx] <= cur->threshold)
                cur = cur->left;
            else
                cur = cur->right;
        }
        if (cur && cur->label != y_val[i]) errors++;
    }
    return (double)errors;
}

static int dt_node_majority(DTNode *node, const double *X_val,
                             const int *y_val, size_t n_val,
                             size_t n_features) {
    if (!node) return 0;
    int freq[32] = {0};
    for (size_t i = 0; i < n_val; ++i) {
        /* Walk to see if sample falls into this node's subtree */
        DTNode *cur = node;
        while (cur && !cur->is_leaf) {
            if (X_val[i * n_features + cur->feature_idx] <= cur->threshold)
                cur = cur->left;
            else
                cur = cur->right;
        }
        if (cur == node || cur == node->left || cur == node->right) {
            /* Sample reached this node's subtree */
            int lbl = y_val[i];
            if (lbl >= 0 && lbl < 32) freq[lbl]++;
        }
    }
    int best = 0;
    for (int c = 1; c < 32; ++c)
        if (freq[c] > freq[best]) best = c;
    return best;
}

static void dt_prune_node(DTNode *node, const double *X_val,
                           const int *y_val, size_t n_val,
                           size_t n_features) {
    if (!node || node->is_leaf) return;

    /* Prune children first (bottom-up) */
    dt_prune_node(node->left, X_val, y_val, n_val, n_features);
    dt_prune_node(node->right, X_val, y_val, n_val, n_features);

    /* If both children are leaves, try pruning */
    if (node->left && node->left->is_leaf &&
        node->right && node->right->is_leaf) {

        double error_subtree = dt_subtree_error(node, X_val, y_val,
                                                 n_val, n_features);
        int majority = dt_node_majority(node, X_val, y_val, n_val, n_features);
        int saved_label = node->label;

        /* Simulate leaf */
        node->is_leaf = true;
        node->label   = majority;
        double error_leaf = dt_node_error(node, X_val, y_val, n_val,
                                           n_features);

        if (error_leaf <= error_subtree) {
            /* Keep as leaf: delete children */
            node_destroy(node->left);
            node_destroy(node->right);
            node->left  = NULL;
            node->right = NULL;
        } else {
            /* Restore as internal node */
            node->is_leaf = false;
            node->label   = saved_label;
        }
    }
}

void dt_post_prune(DecisionTree *tree,
                   const double *X_val, const int *y_val,
                   size_t n_val) {
    if (!tree || !tree->root || n_val == 0) return;
    dt_prune_node(tree->root, X_val, y_val, n_val, tree->n_features);
}

/* ──────────────────────────────────────────────
   RandomForest
   ────────────────────────────────────────────── */

RandomForest rf_create(size_t n_trees, size_t n_features, int n_classes,
                       double sample_ratio, size_t max_features,
                       int max_depth, int min_samples_split) {
    RandomForest rf;
    rf.n_trees    = n_trees;
    rf.n_features = n_features;
    rf.n_classes  = n_classes;
    rf.sample_ratio   = sample_ratio;
    rf.max_features   = max_features;
    rf.trees = (DecisionTree **)malloc(n_trees * sizeof(DecisionTree *));
    for (size_t t = 0; t < n_trees; ++t) {
        rf.trees[t] = (DecisionTree *)malloc(sizeof(DecisionTree));
        *rf.trees[t] = dt_create(n_features, n_classes, max_depth, min_samples_split);
    }
    return rf;
}

void rf_destroy(RandomForest *rf) {
    for (size_t t = 0; t < rf->n_trees; ++t) {
        dt_destroy(rf->trees[t]);
        free(rf->trees[t]);
    }
    free(rf->trees);
    rf->trees = NULL;
}

static void sample_bootstrap(const double *X, const int *y,
                             size_t n, size_t d, double ratio,
                             double *Xb, int *yb, size_t *nb) {
    *nb = (size_t)(ratio * (double)n);
    if (*nb == 0) *nb = n;
    for (size_t i = 0; i < *nb; ++i) {
        size_t idx = (size_t)rand() % n;
        memcpy(Xb + i * d, X + idx * d, d * sizeof(double));
        yb[i] = y[idx];
    }
}

void rf_fit(RandomForest *rf,
            const double *X, const int *y,
            size_t n_samples, DTCriterion crit) {
    size_t d = rf->n_features;
    size_t max_nb = (size_t)(rf->sample_ratio * n_samples);
    if (max_nb == 0) max_nb = n_samples;
    double *Xb = (double *)malloc(max_nb * d * sizeof(double));
    int    *yb = (int    *)malloc(max_nb * sizeof(int));

    for (size_t t = 0; t < rf->n_trees; ++t) {
        size_t nb;
        sample_bootstrap(X, y, n_samples, d, rf->sample_ratio, Xb, yb, &nb);
        dt_fit(rf->trees[t], Xb, yb, nb, crit);
    }
    free(Xb);
    free(yb);
}

int rf_predict(const RandomForest *rf, const double *x) {
    int *votes = (int *)calloc((size_t)rf->n_classes, sizeof(int));
    for (size_t t = 0; t < rf->n_trees; ++t)
        votes[dt_predict(rf->trees[t], x)]++;
    int best = 0;
    for (int c = 1; c < rf->n_classes; ++c)
        if (votes[c] > votes[best]) best = c;
    free(votes);
    return best;
}

/*
 * Feature Importance — Mean Decrease in Impurity (MDI).
 *
 * Breiman (2001): For each tree, sum the impurity reduction (ΔGini or ΔEntropy)
 * at each node where feature f is used for splitting, weighted by the fraction
 * of training samples that reach that node. Average across all trees.
 *
 * This yields a vector of importance scores that sum to 1.0 across features,
 * identifying which features contribute most to prediction accuracy.
 */
static void rf_collect_importance(DTNode *node, double total_n,
                                   double *importances) {
    if (!node || node->is_leaf) return;

    int feat = node->feature_idx;
    if (feat < 0) return;

    /* Weighted impurity reduction:
       ΔI = n_parent × I_parent − n_left × I_left − n_right × I_right
       This node's contribution to feature feat's importance */
    size_t n_parent = node->n_samples;
    double imp_parent = node->impurity;
    double imp_left   = node->left  ? node->left->impurity  : 0.0;
    double imp_right  = node->right ? node->right->impurity : 0.0;
    size_t n_left  = node->left  ? node->left->n_samples  : 0;
    size_t n_right = node->right ? node->right->n_samples : 0;

    double reduction = (double)n_parent * imp_parent
                       - (double)n_left  * imp_left
                       - (double)n_right * imp_right;
    importances[feat] += reduction;

    rf_collect_importance(node->left,  total_n, importances);
    rf_collect_importance(node->right, total_n, importances);
}

void rf_feature_importance(const RandomForest *rf, double *out) {
    if (!rf || rf->n_trees == 0) {
        if (out) memset(out, 0, rf->n_features * sizeof(double));
        return;
    }
    memset(out, 0, rf->n_features * sizeof(double));

    for (size_t t = 0; t < rf->n_trees; ++t) {
        if (rf->trees[t] && rf->trees[t]->root) {
            rf_collect_importance(rf->trees[t]->root,
                                  (double)rf->trees[t]->root->n_samples,
                                  out);
        }
    }

    /* Normalise to sum to 1.0 */
    double sum = 0.0;
    for (size_t f = 0; f < rf->n_features; ++f) sum += out[f];
    if (sum > 1e-12) {
        for (size_t f = 0; f < rf->n_features; ++f) out[f] /= sum;
    }
}
