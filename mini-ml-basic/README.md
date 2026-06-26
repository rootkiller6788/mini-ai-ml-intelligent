# mini-ml-basic — 机器学习基础 (C 语言实现)

纯 C99 实现的经典机器学习算法库，零外部依赖（仅 `libm`），适合学习算法原理和嵌入式部署。

## 模块总览

| 模块            | 头文件                   | 核心算法                                                                 |
|---------------|-----------------------|----------------------------------------------------------------------|
| 线性模型         | `linear_models.h`     | Linear Regression (Normal Eq / SGD / Momentum), Logistic Regression, Softmax |
| 支持向量机       | `svm_kernel.h`        | Linear SVM (Hinge Loss), RBF/Polynomial Kernel, SMO overview, OvR multi-class |
| 决策树 & 随机森林  | `decision_tree.h`     | CART (Gini / Entropy), Pre/Post Pruning, Random Forest (Bootstrap + Feature Bagging) |
| 聚类            | `clustering.h`        | K-Means (k-means++ init, Elbow), DBSCAN, Agglomerative HAC, Silhouette Score |
| 集成学习 & GBDT  | `ensemble_gbdt.h`     | Bagging, AdaBoost, Gradient Boosting, XGBoost-style, Stacking     |

## 项目结构

```
mini-ml-basic/
├── include/
│   ├── linear_models.h
│   ├── svm_kernel.h
│   ├── decision_tree.h
│   ├── clustering.h
│   └── ensemble_gbdt.h
├── src/
│   ├── linear_models.c
│   ├── svm_kernel.c
│   ├── decision_tree.c
│   ├── clustering.c
│   └── ensemble_gbdt.c
├── examples/
│   ├── example_linear.c
│   ├── example_tree.c
│   └── example_cluster.c
├── demo/
│   ├── README_demo_1.md      (线性模型 + SVM)
│   └── README_demo_2.md      (决策树 + 集成 + 聚类)
├── docs/
│   ├── README_api.md         (API 参考)
│   └── README_math.md        (数学原理)
├── Makefile
└── README.md
```

## 快速开始

```bash
# 编译所有目标
make all

# 运行示例
./bin/example_linear
./bin/example_tree
./bin/example_cluster

# 清理
make clean
```

## 使用示例

### 线性回归

```c
#include "linear_models.h"

LinearModel m = lm_create(2);                     /* 2 维特征     */
double X[100][2] = {{...}};
double y[100]    = {...};

lm_fit_sgd(&m, (double*)X, y, 100, 0.01, 500, 32);
double y_hat = lm_predict(&m, (double[]){1.5, -0.8});

lm_destroy(&m);
```

### 决策树

```c
#include "decision_tree.h"

DecisionTree t = dt_create(4, 3, 10, 5);         /* d=4, K=3   */
dt_fit(&t, X, y, n, DT_CRITERION_GINI);
int cls = dt_predict(&t, x_new);
dt_destroy(&t);
```

### K-Means

```c
#include "clustering.h"

KMeans km = kmeans_create(3, 2, 300, 1e-5, KMEANS_INIT_PLUSPLUS);
kmeans_fit(&km, X, 500);
int label = kmeans_predict(&km, test_point);
kmeans_destroy(&km);
```

## 代码风格

- C99 标准，`#ifndef` 头文件守卫
- 类型: PascalCase (`LinearModel`, `RandomForest`)
- 函数: snake_case (`lm_fit_sgd`, `softmax_forward`)
- 宏: UPPER_SNAKE_CASE (`KMEANS_INIT_PLUSPLUS`)
- 依赖 `<stdbool.h>`, `<stddef.h>`

## 构建选项

```makefile
CC      = gcc
CFLAGS  = -Wall -Wextra -O2 -I include
LDFLAGS = -lm
OUTDIR  = bin
```

可通过环境变量覆盖:

```bash
make CC=clang CFLAGS="-O3 -march=native" all
```

## 文档

- [Demo 1: 线性模型与 SVM](demo/README_demo_1.md) — 手把手从线性回归到 RBF 核 SVM
- [Demo 2: 决策树、集成学习与聚类](demo/README_demo_2.md) — CART、随机森林、Boosting、K-Means 实践
- [API 参考](docs/README_api.md) — 完整函数签名与参数说明
- [数学原理](docs/README_math.md) — 梯度推导、损失函数、核方法、Boosting 理论

## 许可

MIT License — 自由使用、修改、分发。

---

*mini-ml-basic — 每一个公式都值得亲手实现。*
