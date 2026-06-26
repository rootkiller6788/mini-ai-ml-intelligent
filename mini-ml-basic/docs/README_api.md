# API 参考文档

## 命名约定

所有公共 API 遵循 `模块名_动词` 的命名模式，使用 snake_case。类型名使用 PascalCase。
宏与常量使用 UPPER_SNAKE_CASE。所有头文件以 `#ifndef` / `#define` / `#endif` 守卫。

---

## 1. linear_models.h

### LinearModel

```c
typedef struct {
    double *coefs;
    double  bias;
    size_t  n_features;
} LinearModel;
```

#### 创建 / 销毁

```c
LinearModel  lm_create(size_t n_features);
void         lm_destroy(LinearModel *m);
```

#### 拟合方法

```c
void  lm_fit_normal_eq(LinearModel *m, const double *X, const double *y, size_t n);
void  lm_fit_sgd(LinearModel *m, const double *X, const double *y,
                 size_t n, double lr, size_t epochs, size_t batch_size);
void  lm_fit_momentum(LinearModel *m, const double *X, const double *y,
                      size_t n, double lr, double momentum, size_t epochs);
```

| 参数       | 类型       | 说明                              |
|----------|----------|---------------------------------|
| `X`      | double*  | (n_samples, n_features) row-major |
| `y`      | double*  | (n_samples,)                     |
| `lr`     | double   | 学习率 (0.001 – 0.1)               |
| `epochs` | size_t   | 迭代轮数                            |
| `batch_size` | size_t | 小批量大小 (SGD)                     |
| `momentum`   | double | 动量系数 (0.8 – 0.99)               |

#### 预测

```c
double  lm_predict(const LinearModel *m, const double *x);
void    lm_predict_batch(const LinearModel *m, const double *X, double *out, size_t n);
```

---

### LogisticModel

```c
typedef struct {
    double *weights;
    double  bias;
    size_t  n_features;
    int     classes;
} LogisticModel;
```

```c
LogisticModel  logreg_create(size_t n_features, int classes);
void           logreg_destroy(LogisticModel *m);

double  logreg_sigmoid(double z);
double  logreg_bce_loss(const LogisticModel *m, const double *X, const double *y, size_t n);

void    logreg_fit(LogisticModel *m, const double *X, const double *y,
                   size_t n, double lr, size_t epochs);
int     logreg_predict(const LogisticModel *m, const double *x);
double  logreg_decision_boundary(const LogisticModel *m, const double *x);
```

---

### SoftmaxModel

```c
typedef struct {
    double *W;        /* (n_features, n_classes) column-major */
    double *b;        /* (n_classes,)                        */
    size_t  n_features;
    size_t  n_classes;
} SoftmaxModel;
```

```c
SoftmaxModel  softmax_create(size_t n_features, size_t n_classes);
void          softmax_destroy(SoftmaxModel *m);

void    softmax_forward(const SoftmaxModel *m, const double *x, double *probs);
double  softmax_cross_entropy(const SoftmaxModel *m, const double *X, const int *y, size_t n);
void    softmax_fit(SoftmaxModel *m, const double *X, const int *y,
                    size_t n, double lr, size_t epochs);
int     softmax_predict(const SoftmaxModel *m, const double *x);
```

---

## 2. svm_kernel.h

### SVMModel

```c
typedef struct {
    double *weights;
    double  bias;
    size_t  n_features;
    double  C;
    int     n_classes;
} SVMModel;
```

```c
SVMModel  svm_create(size_t n_features, double C, int n_classes);
void      svm_destroy(SVMModel *m);

double  svm_hinge_loss(const SVMModel *m, const double *X, const double *y, size_t n);
void    svm_fit_linear(SVMModel *m, const double *X, const double *y,
                       size_t n, double lr, size_t epochs);
int     svm_predict(const SVMModel *m, const double *x);
void    svm_fit_ovr(SVMModel *m, const double *X, const double *y,
                    size_t n, double lr, size_t epochs);
```

### SVMKernel & SMOModel

```c
typedef enum { SVM_KERNEL_LINEAR = 0, SVM_KERNEL_POLYNOMIAL, SVM_KERNEL_RBF } SVMKernelType;

typedef struct {
    SVMKernelType type;
    double  degree;
    double  gamma;
    double  coef0;
} SVMKernel;

double  kernel_dot(const SVMKernel *k, const double *x1, const double *x2, size_t dim);
double  kernel_rbf(const SVMKernel *k, const double *x1, const double *x2, size_t dim);

SMOModel  smo_create(const SVMKernel *kernel);
void      smo_destroy(SMOModel *m);
int       smo_take_step(SMOModel *m, int i1, int i2,
                        const double *X, const double *y, size_t n);
bool      smo_examine_example(SMOModel *m, int i2,
                              const double *X, const double *y,
                              size_t n, double C, double tol);
```

---

## 3. decision_tree.h

### DecisionTree

```c
typedef struct DTNode { int feature_idx; double threshold; int label;
    bool is_leaf; double impurity; size_t n_samples;
    struct DTNode *left, *right; } DTNode;
typedef struct { DTNode *root; size_t n_features; int n_classes;
    int max_depth; int min_samples_split; } DecisionTree;
```

```c
DecisionTree  dt_create(size_t n_features, int n_classes,
                        int max_depth, int min_samples_split);
void          dt_destroy(DecisionTree *tree);

double  dt_gini_impurity(const int *labels, size_t n);
double  dt_entropy(const int *labels, size_t n);
double  dt_information_gain(const int *labels, const int *left, size_t nl,
                            const int *right, size_t nr, DTCriterion crit);

void    dt_fit(DecisionTree *tree, const double *X, const int *y,
               size_t n, DTCriterion crit);
int     dt_predict(const DecisionTree *tree, const double *x);

void    dt_pre_prune(DecisionTree *tree);
void    dt_post_prune(DecisionTree *tree,
                      const double *X_val, const int *y_val, size_t n_val);
```

### RandomForest

```c
typedef struct {
    DecisionTree **trees;
    size_t  n_trees;
    size_t  n_features;
    int     n_classes;
    double  sample_ratio;
    size_t  max_features;
} RandomForest;

RandomForest  rf_create(size_t n_trees, size_t n_features, int n_classes,
                        double sample_ratio, size_t max_features,
                        int max_depth, int min_samples_split);
void          rf_destroy(RandomForest *rf);
void          rf_fit(RandomForest *rf, const double *X, const int *y,
                     size_t n, DTCriterion crit);
int           rf_predict(const RandomForest *rf, const double *x);
void          rf_feature_importance(const RandomForest *rf, double *out);
```

---

## 4. clustering.h

### KMeans

```c
typedef enum { KMEANS_INIT_RANDOM = 0, KMEANS_INIT_PLUSPLUS } KMeansInit;

KMeans  kmeans_create(size_t k, size_t n_features, int max_iters, double tol, KMeansInit init);
void    kmeans_destroy(KMeans *km);
void    kmeans_fit(KMeans *km, const double *X, size_t n);
int     kmeans_predict(const KMeans *km, const double *x);
void    kmeans_centroids_get(const KMeans *km, double *out);
double  kmeans_inertia(const KMeans *km, const double *X, size_t n);
void    kmeans_elbow(const double *X, size_t n, size_t n_features,
                     int max_k, int max_iters, double tol, double *inertias);
```

### DBSCAN

```c
DBSCAN  dbscan_create(double eps, int min_pts);
void    dbscan_destroy(DBSCAN *db);
void    dbscan_fit(DBSCAN *db, const double *X, size_t n, size_t n_features);
```

### HierarchicalClustering

```c
typedef enum { HAC_LINKAGE_SINGLE = 0, HAC_LINKAGE_COMPLETE, HAC_LINKAGE_AVERAGE } HACLinkage;

HierarchicalClustering  hac_create(size_t n_clusters, HACLinkage linkage);
void                    hac_destroy(HierarchicalClustering *hc);
void                    hac_fit(HierarchicalClustering *hc,
                                const double *X, size_t n, size_t dim);
double  silhouette_score(const double *X, const int *labels, size_t n, size_t dim);
```

---

## 5. ensemble_gbdt.h

### Bagging

```c
BaggingModel  bagging_create(size_t n_models,
                             WeakLearnerFit fit, WeakLearnerPredict pred,
                             WeakLearnerDestroy destroy,
                             double (*agg)(const double *, size_t));
void          bagging_destroy(BaggingModel *bg);
void          bagging_fit(BaggingModel *bg, const double *X, const double *y,
                          size_t n, size_t n_features, size_t seed);
double        bagging_predict(BaggingModel *bg, const double *x, size_t n_features);
```

### AdaBoost

```c
AdaBoostModel  adaboost_create(size_t n_estimators, size_t n_features);
void           adaboost_destroy(AdaBoostModel *ab);
void           adaboost_fit(AdaBoostModel *ab, const double *X, const int *y,
                            size_t n, size_t n_features);
int            adaboost_predict(const AdaBoostModel *ab, const double *x);
```

### GBDT

```c
GBDTModel  gbdt_create(size_t n_estimators, double lr,
                       size_t max_depth, size_t n_features);
void       gbdt_destroy(GBDTModel *gb);
void       gbdt_fit(GBDTModel *gb, const double *X, const double *y,
                    size_t n, size_t seed);
double     gbdt_predict(const GBDTModel *gb, const double *x);
void       gbdt_predict_batch(const GBDTModel *gb, const double *X, double *out, size_t n);
```

### XGBoost

```c
XGBoostModel  xgb_create(size_t n_estimators, double eta,
                         double lambda, double gamma,
                         size_t max_depth, size_t n_features);
void          xgb_destroy(XGBoostModel *xgb);
void          xgb_fit(XGBoostModel *xgb, const double *X, const double *y,
                      size_t n, size_t seed);
double        xgb_predict(const XGBoostModel *xgb, const double *x);
```

### Stacking

```c
StackingModel  stacking_create(size_t n_base, size_t n_features,
                               WeakLearnerFit fit, WeakLearnerPredict pred);
void           stacking_destroy(StackingModel *st);
void           stacking_fit(StackingModel *st, const double *X, const double *y,
                            size_t n, size_t seed);
double         stacking_predict(const StackingModel *st, const double *x);
```

---

## 内存管理原则

| 规则                      | 说明                               |
|-------------------------|----------------------------------|
| `create` 后必须 `destroy`  | 防止内存泄漏                           |
| 输入数组由调用方分配              | X, y, outBuffer 由调用方负责生命周期     |
| `_batch` 函数不分配内存         | 调用方提前分配 out 缓冲区                   |
| `free` 后置为 NULL         | destroy 函数将指针置 NULL，防止 use-after-free |

---

## 线程安全性

当前版本**并非线程安全**。所有模型内部状态为单线程设计，多线程
须由调用方加锁保护。

---

*mini-ml-basic API Reference – v0.1.0*
