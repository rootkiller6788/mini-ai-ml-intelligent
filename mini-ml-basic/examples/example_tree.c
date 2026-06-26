#include "decision_tree.h"

#include <stdio.h>
#include <stdlib.h>

int main(void) {
    /* 2D data: blue cluster (0) around (-2,-2), red cluster (1) around (2,2) */
    size_t n = 80;
    double *X = (double *)malloc(n * 2 * sizeof(double));
    int    *y = (int    *)malloc(n * sizeof(int));
    for (size_t i = 0; i < n / 2; ++i) {
        X[i * 2]     = -2.0 + (double)(rand() % 100) / 50.0;
        X[i * 2 + 1] = -2.0 + (double)(rand() % 100) / 50.0;
        y[i] = 0;
    }
    for (size_t i = n / 2; i < n; ++i) {
        X[i * 2]     = 2.0 + (double)(rand() % 100) / 50.0;
        X[i * 2 + 1] = 2.0 + (double)(rand() % 100) / 50.0;
        y[i] = 1;
    }

    /* ── Decision Tree (Gini) ── */
    DecisionTree dt = dt_create(2, 2, 5, 2);
    dt_fit(&dt, X, y, n, DT_CRITERION_GINI);

    int correct = 0;
    for (size_t i = 0; i < n; ++i)
        if (dt_predict(&dt, X + i * 2) == y[i]) correct++;
    printf("DecisionTree train acc: %.2f%%\n", 100.0 * correct / n);

    /* ── Random Forest ── */
    RandomForest rf = rf_create(10, 2, 2, 0.8, 1, 5, 2);
    rf_fit(&rf, X, y, n, DT_CRITERION_GINI);

    correct = 0;
    for (size_t i = 0; i < n; ++i)
        if (rf_predict(&rf, X + i * 2) == y[i]) correct++;
    printf("RandomForest train acc: %.2f%%\n", 100.0 * correct / n);

    dt_destroy(&dt);
    rf_destroy(&rf);
    free(X);
    free(y);
    return 0;
}
