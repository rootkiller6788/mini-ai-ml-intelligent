# ARCHITECTURE.md — mini-dl-framework 架构文档

## 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                   User Code / Examples                   │
├─────────────────────────────────────────────────────────┤
│  loss_funcs.h  │  optimizers.h  │  nn_layers.h          │
├─────────────────────────────────────────────────────────┤
│           autograd.h  │  tensor_ops.h                    │
├─────────────────────────────────────────────────────────┤
│                   C99 Standard Library                   │
└─────────────────────────────────────────────────────────┘
```

## 模块说明

### 1. `autograd.h` / `autograd.c` — 计算图与自动微分

**核心数据结构**: `Node` (计算图节点)
- `value`: 前向计算结果
- `grad`: 反向传播累积的梯度
- `op`: 操作类型 (ADD/MUL/RELU/SIGMOID/TANH/MATMUL/...)
- `inputs[]`: 直接子节点指针数组
- `cache`: 反向传播所需缓存 (中间值)
- `ref_count`: 引用计数,用于 DAG 内存管理
- `requires_grad`: 是否需要计算梯度

**前向传播**: `forward(node)` — 递归计算所有节点的 value,缓存中间结果

**反向传播**: `backward(node)` — BFS 拓扑排序,从输出开始反向应用链式法则:
1. 拓扑排序 (逆向 BFS)
2. 输出节点 gra=1.0
3. 逆序遍历,对每个节点调用其 op 对应的 grad_fn
4. 将梯度累加到子节点的 grad 中

**关键操作梯度**:
```
OP_ADD:    ∂(a+b)/∂a=1, ∂(a+b)/∂b=1
OP_MUL:    ∂(a*b)/∂a=b, ∂(a*b)/∂b=a
OP_RELU:   ∂max(0,x)/∂x = 1 if x>0 else 0
OP_SIGMOID: ∂σ(x)/∂x = σ(x)*(1-σ(x))
OP_MSE:    ∂½(y-ŷ)²/∂y = (y-ŷ)
OP_BCE:    ∂BCE/∂logit = σ(logit)-target
```

### 2. `tensor_ops.h` / `tensor_ops.c` — 张量运算

**核心数据结构**: `Tensor` (多维数组,行主序)
- `dims[]`: 各维度大小
- `strides[]`: 步长 (用于索引计算)
- `data`: `float*` 数据缓冲区
- `ndim`: 维度数
- `size`: 总元素数
- `owns_data`: 是否拥有数据所有权

**内存布局**: 行主序 (row-major), C 风格连续存储

**广播机制**: 对齐形状,从尾部维度开始:
1. 如果维度数不同,将长度为1的维度在左侧补齐
2. 如果对应维度大小不同,允许一方为1 (复制扩展)
3. 否则抛出错误

**关键操作**:
- Element-wise: `add/sub/mul/div` (支持广播)
- `matmul`: 矩阵乘法 (Naive O(n³) 实现 + 批处理扩展)
- `transpose`: 交换 axis
- `reshape`: 改变形状 (保持 total size)
- `slice`: 子区域提取
- `concatenate`: 沿 axis 拼接
- `softmax`: 数值稳定的 Softmax (减最大值)
- `sum/reduce`: 沿 axis 缩减
- 激活函数: `relu/sigmoid/tanh`
- In-place 操作: `add_/sub_/mul_/div_/fill_/scale_/clip_`

**GEMM**: `C = α*A*B + β*C`, 支持 transpose 标志

### 3. `nn_layers.h` / `nn_layers.c` — 神经网络层

**Linear 层**: `y = xW^T + b`
- 前向: matmul(input, weight^T) + bias
- 反向: 
  - `dW = grad^T * input`
  - `db = sum(grad)`
  - `dx = grad * W`

**Conv2d 层**: im2col → GEMM 实现
- `im2col`: 将输入图像 patch 展平为矩阵列
- 前向: `col = im2col(input) → out = reshape(col * W^T) + bias`
- 反向: `grad_col = grad * W → col2im(grad_col) → dx`

**BatchNorm1d**:
- 训练: `mean = E[x]`, `var = Var[x]`, `x̂ = (x-mean)/√(var+ε)`, `y = γ*x̂+β`
- 推理: 使用 running mean/var
- 反向: 通过链式法则计算 dx/dγ/dβ

**LayerNorm**: 沿特征维度归一化
- `mean = mean(x, dim=-1)`, `std = std(x, dim=-1)`
- `y = γ*(x-mean)/std + β`

**Dropout**: 随机丢弃神经元
- 训练: 以概率 p 置零输出,剩余元素乘以 1/(1-p)
- 推理: 恒等映射
- Mask 缓存用于反向传播

**MaxPool2d**:
- 前向: 窗口内取最大值,记录索引
- 反向: 梯度仅传递给最大值位置

**参数收集**: `LayerParam` 链表,配合优化器使用

### 4. `optimizers.h` / `optimizers.c` — 优化器

**SGD**:
```
v = momentum * v + lr * grad
if weight_decay: grad += weight_decay * param
if nesterov: param -= lr * (momentum * v + grad)
else: param -= v
```

**Adam**:
```
t++
m = beta1*m + (1-beta1)*grad
v = beta2*v + (1-beta2)*grad²
m̂ = m / (1 - beta1^t)
v̂ = v / (1 - beta2^t)
param -= lr * m̂ / (√v̂ + ε)
```

**AdamW**: 解耦 weight decay
```
param -= lr * m̂ / (√v̂ + ε) + lr * weight_decay * param
```

**LR Scheduler**:
- Step: 每 step_size 步乘以 gamma
- Cosine: 余弦退火 `lr = η_min + 0.5*(η_0 - η_min)*(1 + cos(π*t/T))`
- LinearWarmup: 线性预热到 initial_lr

**Gradient Clipping**: `||grad||_2 > max_norm` 时缩放梯度

### 5. `loss_funcs.h` / `loss_funcs.c` — 损失函数

**MSE**: `½ * mean((pred - target)²)` (缩放 ½ 便于求导)
```
∂L/∂pred = (pred - target) / N  (mean reduction)
```

**BCEWithLogits**: 在 logits 上计算,通过 sigmoid 稳定数值
```
L = -[t*log(σ(x)) + (1-t)*log(1-σ(x))]
∂L/∂x = (σ(x) - t) / N
```

**CrossEntropyLoss**: `log_softmax → NLL`
```
log_softmax(x) = x - log(sum(exp(x)))
L = -mean(log_softmax(x)[target])
∂L/∂x = (softmax(x) - one_hot(target)) / N
```

**Label Smoothing**: 用平滑标签替换 one-hot
```
smooth_target = (1 - α) * one_hot + α / C
```

**Focal Loss**: 处理类别不平衡
```
FL = -α * (1-p_t)^γ * log(p_t)
∂FL/∂x = α * (1-p_t)^(γ-1) * [γ*p_t*log(p_t) + p_t - 1] * p_t * (1-p_t)
```

**L1 Loss**: `mean(|pred - target|)`, subgradient → sign

**Huber Loss**: 平滑 L1
```
if |diff| ≤ δ: ½*diff²
else: δ*(|diff| - ½δ)
```

## 数据流

```
Input Data → Tensor.create()
    → Layer.forward()      [计算预测]
    → Loss.forward()       [计算损失]
    → Loss.backward()      [损失梯度]
    → Layer.backward()     [逐层反向传播]
    → Optimizer.step()     [更新参数]
    → Optimizer.zero_grad()[清零梯度]
    → (repeat)
```

## 设计原则

1. **纯 C99**: 无第三方依赖,仅依赖标准库 math.h/stdio.h/stdlib.h
2. **显式内存管理**: 所有分配需显式释放,无 GC
3. **模块化**: 各模块职责清晰,依赖层次化
4. **教育优先**: 实现优先可读性,采用朴素算法
5. **可扩展**: 通过操作类型枚举 + 工厂模式可轻松添加新操作
