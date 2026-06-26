# Demo 1: 线性模型与 SVM 实践指南

## 目录

1. [概述](#概述)
2. [环境准备](#环境准备)
3. [Demo 1-1: 线性回归 (Normal Equation vs SGD)](#demo-1-1-线性回归-normal-equation-vs-sgd)
4. [Demo 1-2: 逻辑回归与决策边界](#demo-1-2-逻辑回归与决策边界)
5. [Demo 1-3: Softmax 多分类](#demo-1-3-softmax-多分类)
6. [Demo 1-4: SVM (线性 + RBF Kernel)](#demo-1-4-svm-线性--rbf-kernel)
7. [Demo 1-5: One-vs-Rest 多类 SVM](#demo-1-5-one-vs-rest-多类-svm)
8. [参数调优建议](#参数调优建议)

---

## 概述

`mini-ml-basic` 提供了 C99 实现的机器学习基础算法，本 Demo 聚焦于**线性模型**
（线性回归、逻辑回归、Softmax）以及**支持向量机 (SVM)**，包括核技巧。

### 涉及的核心概念

| 模块            | 关键方法                                        | 数学原理                                |
|---------------|---------------------------------------------|-------------------------------------|
| 线性回归         | `lm_fit_normal_eq` / `lm_fit_sgd` / `lm_fit_momentum` | 最小二乘、梯度下降、动量                     |
| 逻辑回归         | `logreg_fit` / `logreg_bce_loss`            | Sigmoid 激活、交叉熵损失、随机梯度下降            |
| Softmax       | `softmax_fit` / `softmax_cross_entropy`     | 多类交叉熵、Softmax 归一化指数族                |
| SVM (Linear)  | `svm_fit_linear` / `svm_hinge_loss`         | Hinge loss、最大间隔、L2 正则化              |
| SVM (Kernel)  | `kernel_dot` / `kernel_rbf`                 | 核技巧、多项式核、高斯 RBF                    |
| One-vs-Rest   | `svm_fit_ovr`                               | 将 K 类问题分解为 K 个二分类 SVM               |

---

## 环境准备

```bash
cd mini-ml-basic
make all
```

预期输出:

```
gcc -Wall -Wextra -O2 -I include -c src/linear_models.c -o bin/linear_models.o
gcc -Wall -Wextra -O2 -I include -c src/svm_kernel.c -o bin/svm_kernel.o
...
build complete – all objects in bin/
```

运行线性模型示例:

```bash
./bin/example_linear
```

---

## Demo 1-1: 线性回归 (Normal Equation vs SGD)

### 背景

给定特征矩阵 **X** ∈ ℝ^(n×d) 和目标向量 **y** ∈ ℝ^n，目标是找到权重 **θ**
使得 ‖**Xθ** − **y**‖^2 最小。

**解析解** (Normal Equation):

```
θ = (XᵀX)⁻¹ Xᵀ y
```

**迭代解** (SGD with mini-batch):

```
θⱼ := θⱼ − η * (1/|B|) Σᵢ (h_θ(x⁽ⁱ⁾) − y⁽ⁱ⁾) * xⱼ⁽ⁱ⁾
```

### 代码演示

```c
#include "linear_models.h"

/* 生成 y = 3x + 2 + noise 的合成数据 */
size_t n = 500;
double X[n], y[n];
for (size_t i = 0; i < n; ++i) {
    X[i] = ((double)i - 250.0) / 50.0;
    y[i] = 3.0 * X[i] + 2.0 + gauss_noise(0.0, 1.0);
}

/* 方法 A: Normal Equation */
LinearModel m1 = lm_create(1);
lm_fit_normal_eq(&m1, X, y, n);
printf("NormalEq:  w=%.6f, b=%.6f\n", m1.coefs[0], m1.bias);

/* 方法 B: 小批量 SGD (lr=0.01, epochs=1000, batch=32) */
LinearModel m2 = lm_create(1);
lm_fit_sgd(&m2, X, y, n, 0.01, 1000, 32);
printf("SGD:       w=%.6f, b=%.6f\n", m2.coefs[0], m2.bias);

/* 方法 C: Momentum (lr=0.01, β=0.9) */
LinearModel m3 = lm_create(1);
lm_fit_momentum(&m3, X, y, n, 0.01, 0.9, 1000);
printf("Momentum:  w=%.6f, b=%.6f\n", m3.coefs[0], m3.bias);
```

### 预期输出

```
NormalEq:  w=3.001234, b=1.998765
SGD:       w=3.001567, b=1.998123
Momentum:  w=3.000892, b=1.999345
```

三条曲线均收敛至真值 w≈3, b≈2。

### 调参建议

| 参数          | 建议值                | 影响                           |
|-------------|--------------------|------------------------------|
| `lr`        | 0.01 - 0.1          | 过大震荡不收敛；过小收敛太慢               |
| `momentum`  | 0.8 - 0.95          | 加速穿越峡谷，阻尼震荡                   |
| `batch_size`| 32 / 64 / 全量        | 小批量 = 更随机但更 noisy → 隐式正则化      |
| `epochs`    | 500 - 5000          | 视学习率而定；观察损失曲线 plateau         |

---

## Demo 1-2: 逻辑回归与决策边界

### Sigmoid 函数

```
σ(z) = 1 / (1 + e⁻ᶻ)
```

### 二分类交叉熵 (BCE)

```
L = − 1/n Σᵢ [ yᵢ log(σ(zᵢ)) + (1−yᵢ) log(1−σ(zᵢ)) ]
```

梯度:

```
∂L / ∂wⱼ = 1/n Σᵢ (σ(zᵢ) − yᵢ) xⱼ⁽ⁱ⁾
```

### 代码演示

```c
#include "linear_models.h"

double X[n][2];      /* 2D 特征                          */
double y[n];         /* 0/1 标签                         */
/* ... 数据生成 ... */

LogisticModel lr = logreg_create(2, 2);
logreg_fit(&lr, (double *)X, y, n, 0.05, 500);

/* 预测 & 决策边界 */
double x0[2] = {1.5, -0.3};
int pred = logreg_predict(&lr, x0);       /* 0 or 1         */
double z  = logreg_decision_boundary(&lr, x0);  /* 0.0 or 1.0    */
printf("pred=%d,  boundary=%f\n", pred, z);
```

---

## Demo 1-3: Softmax 多分类

### 数学定义

```
P(y=k | x) = exp(zₖ) / Σⱼ exp(zⱼ) ,   zₖ = wₖᵀ x + bₖ
```

交叉熵损失:

```
L = − 1/n Σᵢ log P(y⁽ⁱ⁾ | x⁽ⁱ⁾)
```

### 代码演示

```c
SoftmaxModel sm = softmax_create(4, 10);  /* 4 特征, 10 类  */
softmax_fit(&sm, X_train, y_train, n_train, 0.01, 1000);

int cls = softmax_predict(&sm, test_sample);
printf("Predicted class: %d\n", cls);
```

### 数值稳定性

本库对 `softmax_forward` 不显式减去 max(z)，调用方需自行注意
输入量级。生产环境中建议改为 `z -= max(z)` 后再 exp。

---

## Demo 1-4: SVM (线性 + RBF Kernel)

### Hinge Loss

```
L = λ/2 ‖w‖² + 1/n Σᵢ max(0, 1 − yᵢ (wᵀ xᵢ + b))
```

### RBF 核 (高斯核)

```
K(x, x′) = exp(−γ ‖x − x′‖²)
```

### 代码演示

```c
#include "svm_kernel.h"

/* 二分类线性 SVM */
SVMModel svm = svm_create(d, 1.0, 2);
svm_fit_linear(&svm, X, y, n, 0.001, 1000);

int pred = svm_predict(&svm, test_x);  /* +1 / −1 */

/* 核函数演示 */
SVMKernel rbf = {.type = SVM_KERNEL_RBF, .gamma = 0.5};
double k_val = kernel_dot(&rbf, x1, x2, d);
printf("RBF(x1, x2) = %f\n", k_val);
```

---

## Demo 1-5: One-vs-Rest 多类 SVM

对 K 个类别，训练 K 个二分类 SVM，每个 SVM 将当前类视为 +1、
其他类视为 −1。预测时选取 `wₖᵀ x + bₖ` 最大的 k。

```c
SVMModel svm_mc = svm_create(d, 1.0, 5);   /* 5-class    */
svm_fit_ovr(&svm_mc, X, y, n, 0.001, 500);
/* svm_mc.weights 内部存储 K×d 的矩阵                 */
```

---

## 参数调优建议

| 超参数     | 搜索范围          | 说明                         |
|-----------|----------------|----------------------------|
| 学习率    | 10⁻⁴ → 10⁻¹    | 对数均匀采样                    |
| L2 系数 C | 10⁻³ → 10³     | C 越大正则化越弱 → 可能过拟合        |
| RBF γ     | 1/(2σ²) 范围    | 控制单样本影响半径                 |
| 动量 β    | 0.8 → 0.99     | RMSProp/Adam 用户可以忽略        |
| epochs    | 100 → 10k      | early-stopping 看验证集          |

### 调试清单

- [ ] 特征是否已标准化（归一化到 [0,1] 或标准化到 μ=0, σ=1）？
- [ ] 标签是否使用 ±1（SVM）或 0/1（逻辑回归）？
- [ ] 学习率是否过大 → loss 持续上升？
- [ ] 学习率是否过小 → loss 几乎不下降？
- [ ] 类别是否平衡？不平衡数据考虑 class_weight 或重采样。
- [ ] 是否忘记 shuffle 数据（SGD 依赖 i.i.d. 样本）？

---

## 下一步

- 查看 [Demo 2: 树模型与集成学习](../demo/README_demo_2.md)
- 阅读 [API 参考文档](../docs/README_api.md)
- 阅读 [数学原理文档](../docs/README_math.md)

---

*mini-ml-basic – 从零开始，理解每一个梯度。*
