# Demo 2: 决策树、集成学习与聚类实践指南

## 目录

1. [概述](#概述)
2. [环境准备](#环境准备)
3. [Demo 2-1: 决策树 (CART + Gini/Entropy)](#demo-2-1-决策树-cart--ginientropy)
4. [Demo 2-2: 随机森林 (Bootstrap + Feature Bagging)](#demo-2-2-随机森林-bootstrap--feature-bagging)
5. [Demo 2-3: AdaBoost (指数损失，决策桩)](#demo-2-3-adaboost-指数损失决策桩)
6. [Demo 2-4: GBDT (残差拟合)](#demo-2-4-gbdt-残差拟合)
7. [Demo 2-5: XGBoost 风格 (二阶泰勒近似)](#demo-2-5-xgboost-风格-二阶泰勒近似)
8. [Demo 2-6: K-Means (k-means++ 初始化)](#demo-2-6-k-means-k-means-初始化)
9. [Demo 2-7: DBSCAN & 层次聚类](#demo-2-7-dbscan--层次聚类)
10. [Demo 2-8: 集成学习对比实验](#demo-2-8-集成学习对比实验)
11. [调参速查表](#调参速查表)

---

## 概述

本 Demo 覆盖 `mini-ml-basic` 中的**决策树、集成学习 (Bagging/Boosting/Stacking)**
以及**无监督聚类**算法。所有算法均以 C99 实现，无需外部依赖（除 `libm`）。

### 涉及的核心概念

| 模块           | 关键结构                                      | 算法家族           |
|--------------|-------------------------------------------|----------------|
| DecisionTree | `dt_fit` / `dt_gini_impurity` / `dt_entropy` | CART            |
| RandomForest | `rf_create` / `rf_fit` / `rf_predict`        | Bagging         |
| AdaBoost     | `adaboost_create` / `adaboost_fit`           | Boosting (指数)   |
| GBDT         | `gbdt_create` / `gbdt_fit`                   | Boosting (回归)   |
| XGBoost      | `xgb_create` / `xgb_fit`                     | Boosting (二阶)   |
| Stacking     | `stacking_create` / `stacking_fit`            | Meta-Learning   |
| KMeans       | `kmeans_create` / `kmeans_fit`               | Partitioning    |
| DBSCAN       | `dbscan_create` / `dbscan_fit`               | Density-Based   |
| HAC          | `hac_create` / `hac_fit`                      | Hierarchical    |

---

## 环境准备

```bash
cd mini-ml-basic
make clean && make all
```

编译并运行树模型示例:

```bash
./bin/example_tree
```

编译并运行聚类示例:

```bash
./bin/example_cluster
```

---

## Demo 2-1: 决策树 (CART + Gini/Entropy)

### 分裂准则

**Gini Impurity:**

```
Gini(D) = 1 − Σₖ pₖ²
```

**Entropy (Information Gain):**

```
Ent(D) = − Σₖ pₖ log₂ pₖ
IG(D, A) = Ent(D) − Σᵥ (|Dᵥ|/|D|) Ent(Dᵥ)
```

### 剪枝策略

| 策略     | 参数                      | 说明                         |
|--------|-------------------------|----------------------------|
| 预剪枝    | `max_depth`, `min_samples_split` | 在构建时阻止分裂，防止过拟合               |
| 后剪枝    | `dt_post_prune` (验证集)     | 训练完整树后自底向上剪枝 (本库为占位)          |

### 代码演示

```c
#include "decision_tree.h"

/* 训练 */
DecisionTree dt = dt_create(n_features, n_classes, 8, 5);
dt_fit(&dt, X_train, y_train, n_train, DT_CRITERION_GINI);

/* 评估 */
int correct = 0;
for (size_t i = 0; i < n_test; ++i)
    if (dt_predict(&dt, X_test + i * n_features) == y_test[i])
        correct++;
printf("Test accuracy: %.2f%%\n", 100.0 * (double)correct / n_test);

/* 查看树结构 */
dt_pre_prune(&dt);
dt_destroy(&dt);
```

---

## Demo 2-2: 随机森林 (Bootstrap + Feature Bagging)

### 核心步骤

1. 从原始数据集中**有放回**采样（Bootstrap）得到 Dₜ
2. 在每个分裂节点，从全部 d 个特征中**随机选取** m ≤ d 个候选特征
3. 在候选特征中选择最优分裂
4. 重复 T 次，得到 T 棵决策树
5. 预测时**多数投票**（分类）或**平均**（回归）

### 代码演示

```c
RandomForest rf = rf_create(
    100,            /* n_trees         */
    n_features,     /*                 */
    n_classes,      /*                 */
    0.8,            /* 80% bootstrap   */
    (size_t)sqrt(n_features), /* feature bagging */
    10,             /* max_depth       */
    5               /* min_samples     */
);
rf_fit(&rf, X_train, y_train, n_train, DT_CRITERION_GINI);

int vote = rf_predict(&rf, x_new);

/* 特征重要性 (当前版本为占位) */
double *importance = malloc(n_features * sizeof(double));
rf_feature_importance(&rf, importance);
```

### 超参数建议

| 参数              | 典型值                    | 备注                     |
|-----------------|------------------------|------------------------|
| `n_trees`       | 100 - 1000             | 越多越稳定但越慢                |
| `sample_ratio`  | 0.6 - 1.0              | 0.632 对应经典 Bootstrap   |
| `max_features`  | sqrt(d) 或 log₂(d)      | 分类默认 sqrt，回归默认 d/3     |
| `max_depth`     | 5 - 30                 | 深树捕获交互但可能过拟合           |
| `min_samples_split` | 2 - 20             | 叶子节点最少样本数              |

---

## Demo 2-3: AdaBoost (指数损失，决策桩)

### 算法流程

```
初始化权重 wᵢ⁽¹⁾ = 1/n
for t = 1 to T:
    在加权数据集上训练弱分类器 hₜ(x)
    计算加权误差 εₜ = Σᵢ wᵢ⁽ᵗ⁾ · 1[hₜ(xᵢ) ≠ yᵢ]
    计算分类器权重 αₜ = ½ ln((1−εₜ)/εₜ)
    更新样本权重 wᵢ⁽ᵗ⁺¹⁾ ∝ wᵢ⁽ᵗ⁾ exp(−αₜ yᵢ hₜ(xᵢ))
输出: H(x) = sign(Σₜ αₜ hₜ(x))
```

### 代码演示

```c
#include "ensemble_gbdt.h"

/* 弱学习器使用决策桩 (单层决策树) */
AdaBoostModel ab = adaboost_create(50, n_features);
adaboost_fit(&ab, X, y_labels, n_samples, n_features);

int pred = adaboost_predict(&ab, test_sample);
```

### 注意事项

- 标签必须为 ±1（本库内部转换）
- 对噪声和异常值敏感，因为它们会被不断加权
- 弱学习器误差 εₜ 必须 < 0.5，否则停止迭代

---

## Demo 2-4: GBDT (残差拟合)

### 数学推导

第 t 轮拟合的目标是上一轮结果的**残差**:

```
rᵢ⁽ᵗ⁾ = − [∂L(yᵢ, F⁽ᵗ⁻¹⁾(xᵢ)) / ∂F]    (负梯度)

F⁽ᵗ⁾(x) = F⁽ᵗ⁻¹⁾(x) + ν * hₜ(x)
```

其中 ν 为学习率 (shrinkage)，通常设为 0.01–0.1。

### 代码演示

```c
GBDTModel gb = gbdt_create(100, 0.1, 6, n_features);
gbdt_fit(&gb, X_train, y_train, n_train, 42);

double y_pred = gbdt_predict(&gb, x_new);
double *batch_preds = malloc(n_test * sizeof(double));
gbdt_predict_batch(&gb, X_test, batch_preds, n_test);
```

---

## Demo 2-5: XGBoost 风格 (二阶泰勒近似)

### 核心创新

XGBoost 使用**二阶泰勒展开**逼近目标函数:

```
Obj⁽ᵗ⁾ ≈ Σᵢ [ l(yᵢ, ŷᵢ⁽ᵗ⁻¹⁾) + gᵢ fₜ(xᵢ) + ½ hᵢ fₜ²(xᵢ) ] + Ω(fₜ)

其中 gᵢ = ∂l/∂ŷ,   hᵢ = ∂²l/∂ŷ²   (一阶和二阶梯度)
Ω(fₜ) = γ·T + ½·λ Σⱼ wⱼ²         (正则化项)
```

### 代码演示

```c
XGBoostModel xgb = xgb_create(
    100,    /* n_estimators   */
    0.1,    /* eta (lr)       */
    1.0,    /* lambda (L2)    */
    0.0,    /* gamma          */
    6,      /* max_depth      */
    n_features
);
xgb_fit(&xgb, X, y, n, 12345);
double pred = xgb_predict(&xgb, x_new);
xgb_destroy(&xgb);
```

> **注意**：当前版本为标准 API 占位，完整二阶实现待后续版本加入。
> 详见 [docs/README_math.md](../docs/README_math.md) 中的公式推导。

---

## Demo 2-6: K-Means (k-means++ 初始化)

### Lloyd's 迭代

```
Assign:   c⁽ⁱ⁾ = argminⱼ ‖x⁽ⁱ⁾ − μⱼ‖²
Update:   μⱼ = (1/|Cⱼ|) Σ_{i∈Cⱼ} x⁽ⁱ⁾
```

收敛条件: 标签不再变化 或 质心位移 < tol。

### k-means++ 初始化

```
c₁ ← 从数据中均匀随机选取
for k = 2..K:
    计算每个点到最近已有中心的距离 D(x)
    以概率 ∝ D(x)² 选取下一个中心
```

### 代码演示

```c
KMeans km = kmeans_create(3, 2, 300, 1e-5, KMEANS_INIT_PLUSPLUS);
kmeans_fit(&km, X, n_samples);

/* Elbow 方法确定最佳 K */
int max_k = 10;
double *inertias = malloc(max_k * sizeof(double));
kmeans_elbow(X, n_samples, 2, max_k, 300, 1e-5, inertias);
for (int k = 0; k < max_k; ++k)
    printf("k=%d  inertia=%.2f\n", k+1, inertias[k]);
/* 观察 elbow 点 → 选择 K */
```

---

## Demo 2-7: DBSCAN & 层次聚类

### DBSCAN 参数

- `eps`: 邻域半径，定义了 "Density-Reachable" 的距离阈值
- `min_pts`: 核心点的最小邻居数

**核心概念:**

- **Core point**: 邻域内至少有 min_pts 个点（含自身）
- **Border point**: 不是核心点，但在某核心点的邻域内
- **Noise point**: 既不是核心点也不是边界点

### 层次聚类 (Agglomerative)

| Linkage          | 距离定义                                    |
|-----------------|-----------------------------------------|
| Single (MIN)    | d(C₁, C₂) = min_{a∈C₁, b∈C₂} d(a,b)     |
| Complete (MAX)  | d(C₁, C₂) = max_{a∈C₁, b∈C₂} d(a,b)     |
| Average         | d(C₁, C₂) = avg_{a∈C₁, b∈C₂} d(a,b)     |

### 代码演示

```c
/* DBSCAN */
DBSCAN db = dbscan_create(0.5, 5);
dbscan_fit(&db, X, n_samples, n_features);
int cluster_id = db.labels[i];  /* ≥0: cluster,  −1: noise */

/* HAC */
HierarchicalClustering hc = hac_create(3, HAC_LINKAGE_COMPLETE);
hac_fit(&hc, X, n_samples, n_features);
/* hac.labels[i] = assigned cluster */

/* Silhouette */
double sil = silhouette_score(X, labels, n, d);
printf("Silhouette = %.4f  (range [−1, 1], higher is better)\n", sil);
```

---

## Demo 2-8: 集成学习对比实验

### 实验设置

在同一数据集上对比以下模型的性能:

```c
/* 1. 单一决策树 */
DecisionTree dt = dt_create(d, nc, 10, 5);
dt_fit(&dt, X_train, y_train, n_train, DT_CRITERION_GINI);

/* 2. 随机森林 */
RandomForest rf = rf_create(50, d, nc, 0.8, (size_t)sqrt(d), 10, 5);
rf_fit(&rf, X_train, y_train, n_train, DT_CRITERION_GINI);

/* 3. AdaBoost */
AdaBoostModel ab = adaboost_create(50, d);
adaboost_fit(&ab, X, y_bin, n_samples, d);

/* 4. GBDT */
GBDTModel gb = gbdt_create(100, 0.05, 6, d);
gbdt_fit(&gb, X_train, y_train, n_train, 2024);

/* 5. Stacking (meta-learner on top) */
/* ... */
```

### 结果分析模板

| 模型         | Train Acc | Test Acc | 训练时间  | 可解释性  |
|------------|-----------|---------|-------|-------|
| DecisionTree | 98%       | 85%     | 快     | ★★★★★  |
| RandomForest | 99%       | 91%     | 中     | ★★★☆☆  |
| AdaBoost   | 95%       | 87%     | 中     | ★★★☆☆  |
| GBDT       | 97%       | 92%     | 较慢    | ★★☆☆☆  |
| Stacking   | 96%       | 89%     | 慢     | ★★☆☆☆  |

---

## 调参速查表

### 决策树

| 参数                   | 作用              | 搜索建议       |
|----------------------|-----------------|------------|
| `max_depth`          | 限制树深            | [3, 5, 10, 20, None] |
| `min_samples_split`  | 节点分裂最小样本数       | [2, 5, 10, 20] |
| `criterion`          | 分裂准则            | Gini / Entropy (经验上差异不大) |

### 随机森林

| 参数                | 作用             | 搜索建议              |
|-------------------|----------------|-------------------|
| `n_trees`         | 集成规模           | [50, 100, 200, 500] |
| `max_features`    | 特征子集大小         | [sqrt, log2, all]  |
| `sample_ratio`    | Bootstrap 比例    | [0.6, 0.8, 1.0]   |

### Boosting

| 参数               | 作用                | 搜索建议               |
|------------------|-------------------|--------------------|
| `n_estimators`   | 基学习器数量            | [50, 100, 300, 500] |
| `learning_rate`  | 学习率 / 收缩          | [0.001, 0.01, 0.05, 0.1, 0.3] |
| `max_depth`      | 单棵树深度             | [3, 4, 5, 6, 8]   |

### K-Means

| 参数            | 作用            | 搜索建议          |
|---------------|---------------|---------------|
| `k`           | 聚类数           | Elbow method  |
| `max_iters`   | 最大迭代          | [100, 300, 500] |
| `tol`         | 收敛阈值          | 1e-4 to 1e-6  |
| `init_method` | 初始化方式         | 推荐 k-means++   |

---

## 调试清单

- [ ] 类别特征是否已进行 One-Hot 编码？
- [ ] 树模型的特征值是否被异常值主导？
- [ ] K-Means 的数据是否标准化（各维度方差一致）？
- [ ] DBSCAN 的 eps 是否通过 k-distance 图确定？
- [ ] Boosting 是否因学习率过大而发散？
- [ ] 验证集是否与训练集独立同分布？

---

## 下一步

- 查看 [Demo 1: 线性模型与 SVM](../demo/README_demo_1.md)
- 阅读 [API 参考文档](../docs/README_api.md)
- 阅读 [数学原理文档](../docs/README_math.md)

---

*mini-ml-basic – 每一个分割、每一票都具有重量。*
