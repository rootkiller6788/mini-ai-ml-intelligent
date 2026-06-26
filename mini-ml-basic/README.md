# mini-ml-basic — 机器学习基础 (C 语言实现)

纯 C99 实现的经典机器学习算法库，零外部依赖（仅 `libm`），适合学习算法原理和嵌入式部署。

## Module Status: COMPLETE ✅

- **include/ + src/ 总行数**: 4,518 (≥ 3,000 ✅)
- **L1-L6**: Complete
- **L7**: Complete (5 applications / examples)
- **L8**: Partial (KD-Tree spatial index, SMO solver, polynomial features)
- **L9**: Partial (XGBoost second-order Taylor documented, GridSearch framework)

---

## 模块总览

| 模块              | 头文件                      | 核心算法 |
|-------------------|---------------------------|---------|
| 线性模型           | `linear_models.h`         | Linear Regression (Normal Eq / SGD / Momentum), Logistic Regression, Softmax |
| 支持向量机         | `svm_kernel.h`            | Linear SVM (Hinge Loss), RBF/Polynomial Kernel, SMO (Platt 1998), OvR multi-class |
| 决策树 & 随机森林  | `decision_tree.h`         | CART (Gini / Entropy), Reduced-Error Post-Pruning, Random Forest (Bootstrap + MDI Feature Importance) |
| 聚类              | `clustering.h`            | K-Means (k-means++ init, Elbow), DBSCAN, Agglomerative HAC (Single/Complete/Average Linkage), Silhouette Score |
| 集成学习 & GBDT    | `ensemble_gbdt.h`         | Bagging, AdaBoost, GBDT (full tree traversal), XGBoost-style (2nd order Taylor, L2 reg), Stacking |
| 近邻模型           | `neighbor_models.h`       | KNN Classifier/Regressor (5 distance metrics), KD-Tree (Bentley 1975), Naive Bayes (Gaussian) |
| 模型评估与选择     | `model_selection.h`       | Confusion Matrix (binary + multi-class), Precision/Recall/F1, MSE/MAE/R², K-Fold CV, Train/Test Split, Grid Search |
| 数据预处理         | `preprocessing.h`         | StandardScaler (Z-score), MinMaxScaler, LabelEncoder, OneHotEncoder, SimpleImputer (mean/median/mode), PolynomialFeatures |

## 知识覆盖九层体系

### L1: 核心定义 (Complete)
- 8 个头文件，27+ 个 struct/typedef，100+ 个 API 声明
- 枚举类型: DistanceMetric, SVMKernelType, KMeansInit, HACLinkage, DTCriterion, ImputeStrategy, ClassificationMetric
- 回调类型: WeakLearnerFit, WeakLearnerPredict, CVPredictFn, CVFitFn

### L2: 核心概念 (Complete)
- 监督学习 (Linear/Logistic/Softmax/SVM/KNN/NaiveBayes)
- 无监督学习 (K-Means/DBSCAN/HAC)
- 集成学习 (Bagging/Boosting/Stacking)
- 模型评估 (Confusion Matrix/K-Fold CV/Regression Metrics)
- 特征工程 (Scaler/Encoder/Imputer/Polynomial Expansion)
- 空间索引 (KD-Tree)

### L3: 工程结构 (Complete)
- KD-Tree 空间划分 + 最近邻查询 + 半径搜索
- GBDT 扁平化树存储 + 遍历
- XGBoost 二阶泰勒目标 + L2 正则化
- HAC 距离矩阵 + Lance-Williams 递归合并

### L4: 标准/定理 (Complete)
- Cover & Hart (1967): 1-NN asymptotic error ≤ 2× Bayes error
- Platt (1998): SMO analytic update
- Breiman (2001): Random Forest MDI feature importance
- Quinlan (1987): Reduced-Error Pruning
- Chen & Guestrin (2016): XGBoost second-order Taylor objective
- Bentley (1975): KD-Tree median-split partitioning
- Lance & Williams (1967): Agglomerative clustering recurrence
- Cover's Theorem: Polynomial feature expansion for non-linear separability

### L5: 算法/方法 (Complete)
- Linear Regression: Normal Equation, SGD, Momentum
- SVM: Gradient-descent Hinge Loss, SMO, Kernel trick
- Decision Tree: CART with Gini/Entropy, Post-pruning
- K-Means: Lloyd's algorithm with k-means++
- HAC: Single/Complete/Average linkage agglomerative
- GBDT: Residual fitting with tree ensemble
- XGBoost: Second-order Taylor expansion with leaf weight formula
- AdaBoost: Exponential loss minimization
- Naive Bayes: Gaussian MLE parameter estimation
- KD-Tree: Median-split recursive construction + Best-Bin-First search
- KNN: Top-k insertion sort for nearest-neighbor voting/regression

### L6: 经典工程问题 (Complete)
- `examples/example_linear.c` — 线性回归 + 逻辑回归 + Softmax 端到端训练
- `examples/example_tree.c` — 决策树 + 随机森林 分类演示
- `examples/example_cluster.c` — K-Means + Elbow + DBSCAN + Silhouette
- `examples/example_neighbor.c` — KNN + KD-Tree + Naive Bayes + StandardScaler + PolynomialFeatures
- `examples/example_model_sel.c` — Train/Test Split + Metrics + Confusion Matrix + K-Fold CV

### L7: 应用 (Complete)
1. Iris-like 多分类 (Softmax)
2. Gaussian blob 聚类 (K-Means/DBSCAN/HAC)
3. 集成分类 (AdaBoost/Random Forest)
4. 回归预测 + Cross-Validation (GBDT/XGBoost)
5. 实例检索 (KD-Tree KNN Search)

### L8: 进阶主题 (Partial)
- KD-Tree spatial index with Best-Bin-First search
- SMO (Platt 1998) with KKT violation detection
- Polynomial Feature Expansion to degree d
- Grid Search framework (single-parameter)

### L9: 工业前沿 (Partial — documented)
- XGBoost second-order Taylor derivation (docs/README_math.md)
- Regularization path (L2 in XGBoost, C in SVM)
- Future: LightGBM histogram-based, CatBoost ordered boosting (doc only)

---

## 九校课程对标

| 学校 | 课程 | 本模块覆盖 |
|------|------|-----------|
| MIT | 6.036 Intro to ML | Linear/Logistic Regression, SVM, KNN |
| Stanford | CS 229 Machine Learning | Softmax, Naive Bayes, GBDT, XGBoost |
| Berkeley | CS 189 Intro to ML | Decision Trees, Random Forest, Clustering |
| CMU | 10-601/701 ML | Ensemble methods, Model evaluation |
| UT Austin | CS 391L Machine Learning | SVM kernels, HAC, DBSCAN |
| ETH | 252-0220 Intro to ML | Preprocessing pipeline, Cross-validation |
| Cambridge | Part II: Machine Learning | Bayesian methods (Naive Bayes), MDI importance |
| 清华 | 机器学习 (40250973) | 决策树、集成学习、聚类、SVM |
| Georgia Tech | CS 7641 Machine Learning | Boosting theory, Supervised learning spectrum |

---

## 项目结构

```
mini-ml-basic/                        (include/ + src/ = 4,518 行)
├── include/
│   ├── linear_models.h               (  78 行)
│   ├── svm_kernel.h                  (  77 行)
│   ├── decision_tree.h               (  79 行)
│   ├── clustering.h                  (  84 行)
│   ├── ensemble_gbdt.h               ( 140 行)
│   ├── neighbor_models.h             ( 117 行)
│   ├── model_selection.h             ( 140 行)
│   └── preprocessing.h               ( 146 行)
├── src/
│   ├── linear_models.c               ( 292 行)
│   ├── svm_kernel.c                  ( 280 行)
│   ├── decision_tree.c               ( 457 行)
│   ├── clustering.c                  ( 494 行)
│   ├── ensemble_gbdt.c               ( 663 行)
│   ├── neighbor_models.c             ( 438 行)
│   ├── model_selection.c             ( 480 行)
│   └── preprocessing.c               ( 553 行)
├── examples/
│   ├── example_linear.c              (线性模型)
│   ├── example_tree.c                (决策树+随机森林)
│   ├── example_cluster.c             (K-Means+DBSCAN)
│   ├── example_neighbor.c            (KNN+KD-Tree+NaiveBayes+预处理)
│   └── example_model_sel.c           (交叉验证+指标+混淆矩阵)
├── tests/
│   └── test_core.c                   (22 个测试, 全部通过)
├── benches/
│   └── bench_core.c                  (14 个算法基准)
├── demo/
│   ├── README_demo_1.md              (线性模型 + SVM)
│   └── README_demo_2.md              (决策树 + 集成 + 聚类)
├── docs/
│   ├── README_api.md                 (API 参考)
│   └── README_math.md                (数学原理)
├── Makefile
└── README.md
```

## 快速开始

```bash
# 编译所有目标和示例
make all

# 运行测试 (22/22 通过)
make test

# 运行性能基准
make bench

# 清理
make clean
```

---

## 核心定理与公式

| 定理 | 出处 | 代码位置 |
|------|------|---------|
| Normal Equation θ* = (XᵀX)⁻¹Xᵀy | Gauss-Markov | `lm_fit_normal_eq` |
| Hinge Loss + L2 正则化 | Cortes & Vapnik (1995) | `svm_fit_linear` |
| SMO analytic α₂ update | Platt (1998) | `smo_take_step` |
| KD-Tree median split | Bentley (1975) | `kdtree_build_rec` |
| Cover & Hart 1-NN 定理 | Cover & Hart (1967) | `knn_predict_class` |
| Naive Bayes Gaussian MLE | Bayesian inference | `nb_fit` |
| Reduced-Error Pruning | Quinlan (1987) | `dt_post_prune` |
| MDI Feature Importance | Breiman (2001) | `rf_feature_importance` |
| GBDT gradient boosting | Friedman (2001) | `gbdt_fit` |
| XGBoost 2nd-order Taylor | Chen & Guestrin (2016) | `xgb_fit` |
| AdaBoost exponential loss | Freund & Schapire (1997) | `adaboost_fit` |
| k-means++ initialization | Arthur & Vassilvitskii (2007) | `kmeans_init_plusplus` |
| Lance-Williams recurrence | Lance & Williams (1967) | `hac_fit` |

---

## 许可

MIT License — 自由使用、修改、分发。

---

*mini-ml-basic — 每一个公式都值得亲手实现。*
