/*
 * mini-ml-basic — Core Tests
 *
 * Unit tests for linear models, SVM, decision tree, clustering, ensembles.
 */
#include "../include/linear_models.h"
#include "../include/svm_kernel.h"
#include "../include/decision_tree.h"
#include "../include/clustering.h"
#include "../include/ensemble_gbdt.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int tests_run = 0, tests_passed = 0;

#define TEST(name) do { tests_run++; printf("  TEST %s ... ", name); } while(0)
#define PASS() do { tests_passed++; printf("PASS\n"); } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); return 1; } while(0)
#define CHECK(cond, msg) if (!(cond)) FAIL(msg)

/* ── Linear Model Tests ── */
static int test_linear_regression(void) {
    TEST("linear_regression_fit_predict");
    double X[] = {1.0, 0.0, 2.0, 0.0, 3.0, 0.0};
    double y[] = {2.0, 4.0, 6.0};
    LinearModel m = lm_create(2);
    lm_fit_normal_eq(&m, X, y, 3);
    double pred = lm_predict(&m, (double[]){4.0, 0.0});
    CHECK(fabs(pred - 8.0) < 0.01, "linear prediction wrong");
    lm_destroy(&m);
    PASS();
    return 0;
}

static int test_logistic_sigmoid(void) {
    TEST("logistic_sigmoid");
    double s = logreg_sigmoid(0.0);
    CHECK(fabs(s - 0.5) < 1e-6, "sigmoid(0) != 0.5");
    CHECK(logreg_sigmoid(10.0) > 0.99, "sigmoid(10) too low");
    CHECK(logreg_sigmoid(-10.0) < 0.01, "sigmoid(-10) too high");
    PASS();
    return 0;
}

static int test_softmax_create_predict(void) {
    TEST("softmax_create_predict");
    SoftmaxModel m = softmax_create(4, 3);
    CHECK(m.n_features == 4, "n_features wrong");
    CHECK(m.n_classes == 3, "n_classes wrong");
    double x[] = {1.0, 0.0, 0.0, 0.0};
    double probs[3];
    softmax_forward(&m, x, probs);
    double sum = probs[0] + probs[1] + probs[2];
    CHECK(fabs(sum - 1.0) < 0.01, "softmax probs don't sum to 1");
    softmax_destroy(&m);
    PASS();
    return 0;
}

/* ── SVM Tests ── */
static int test_svm_create(void) {
    TEST("svm_create");
    SVMModel m = svm_create(8, 1.0, 2);
    CHECK(m.n_features == 8, "n_features wrong");
    CHECK(m.C == 1.0, "C wrong");
    svm_destroy(&m);
    PASS();
    return 0;
}

static int test_kernel_dot(void) {
    TEST("kernel_dot");
    SVMKernel k = {SVM_KERNEL_LINEAR, 3.0, 0.5, 1.0};
    double x1[] = {1.0, 2.0, 3.0};
    double x2[] = {4.0, 5.0, 6.0};
    double dot = kernel_dot(&k, x1, x2, 3);
    CHECK(fabs(dot - 32.0) < 1e-6, "kernel dot wrong"); /* 1*4+2*5+3*6=32 */
    PASS();
    return 0;
}

/* ── Decision Tree Tests ── */
static int test_decision_tree_gini(void) {
    TEST("dt_gini_impurity");
    int labels[] = {0, 0, 1, 1};
    double gini = dt_gini_impurity(labels, 4);
    CHECK(fabs(gini - 0.5) < 0.01, "gini should be 0.5 for balanced");
    int pure[] = {0, 0, 0, 0};
    CHECK(fabs(dt_gini_impurity(pure, 4)) < 1e-6, "gini should be 0 for pure");
    PASS();
    return 0;
}

static int test_decision_tree_entropy(void) {
    TEST("dt_entropy");
    int labels[] = {0, 0, 1, 1};
    double e = dt_entropy(labels, 4);
    CHECK(fabs(e - 1.0) < 0.01, "entropy should be 1.0 for balanced");
    PASS();
    return 0;
}

static int test_decision_tree_fit_predict(void) {
    TEST("dt_fit_predict");
    double X[] = {1.0, 0.5, 2.0, 1.0, 3.0, 2.0, 4.0, 3.0};
    int    y[] = {0, 0, 1, 1};
    DecisionTree tree = dt_create(2, 2, 5, 1);
    dt_fit(&tree, X, y, 4, DT_CRITERION_GINI);
    CHECK(tree.root != NULL, "tree root is NULL");
    int pred = dt_predict(&tree, (double[]){1.0, 0.5});
    CHECK(pred == 0 || pred == 1, "invalid prediction");
    dt_destroy(&tree);
    PASS();
    return 0;
}

/* ── Clustering Tests ── */
static int test_kmeans_create_fit(void) {
    TEST("kmeans_create_fit");
    double X[] = {0.0, 0.0, 1.0, 1.0, 5.0, 5.0, 6.0, 6.0};
    KMeans km = kmeans_create(2, 2, 10, 1e-4, KMEANS_INIT_RANDOM);
    kmeans_fit(&km, X, 4);
    int l0 = kmeans_predict(&km, (double[]){0.0, 0.0});
    int l1 = kmeans_predict(&km, (double[]){6.0, 6.0});
    CHECK(l0 != l1, "far points should be different clusters");
    kmeans_destroy(&km);
    PASS();
    return 0;
}

static int test_silhouette_score(void) {
    TEST("silhouette_score");
    double X[] = {0.0, 0.0, 0.5, 0.0, 5.0, 0.0, 5.5, 0.0};
    int labels[] = {0, 0, 1, 1};
    double s = silhouette_score(X, labels, 4, 2);
    CHECK(s >= -1.0 && s <= 1.0, "silhouette out of range");
    PASS();
    return 0;
}

/* ── Ensemble Tests ── */
static int test_adaboost_fit_predict(void) {
    TEST("adaboost_fit_predict");
    double X[] = {1.0, 0.5, 2.0, 1.0, 3.0, 2.0, 4.0, 3.0};
    int    y[] = {-1, -1, 1, 1};
    AdaBoostModel ab = adaboost_create(10, 2);
    adaboost_fit(&ab, X, y, 4, 2);
    int pred = adaboost_predict(&ab, (double[]){1.0, 0.5});
    CHECK(pred == -1 || pred == 1, "invalid prediction");
    adaboost_destroy(&ab);
    PASS();
    return 0;
}

static int test_xgboost_fit_predict(void) {
    TEST("xgb_fit_predict");
    double X[] = {1.0, 0.5, 2.0, 1.0, 3.0, 2.0, 4.0, 3.0};
    double y[] = {1.0, 2.0, 3.0, 4.0};
    XGBoostModel xgb = xgb_create(5, 0.3, 1.0, 0.0, 3, 2);
    xgb_fit(&xgb, X, y, 4, 42);
    double pred = xgb_predict(&xgb, (double[]){1.0, 0.5});
    CHECK(pred > 0.0, "prediction should be positive");
    xgb_destroy(&xgb);
    PASS();
    return 0;
}

int main(void) {
    printf("=== mini-ml-basic Unit Tests ===\n\n");

    int failed = 0;
    failed += test_linear_regression();
    failed += test_logistic_sigmoid();
    failed += test_softmax_create_predict();
    failed += test_svm_create();
    failed += test_kernel_dot();
    failed += test_decision_tree_gini();
    failed += test_decision_tree_entropy();
    failed += test_decision_tree_fit_predict();
    failed += test_kmeans_create_fit();
    failed += test_silhouette_score();
    failed += test_adaboost_fit_predict();
    failed += test_xgboost_fit_predict();

    printf("\n=== Results: %d/%d passed, %d failed ===\n",
           tests_passed, tests_run, failed);
    return failed > 0 ? 1 : 0;
}
