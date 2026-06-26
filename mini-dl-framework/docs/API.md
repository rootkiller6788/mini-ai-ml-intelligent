# API.md — mini-dl-framework API 参考

## Tensor 操作 (`tensor_ops.h`)

### 创建与销毁

```c
Tensor* tensor_create(int* dims, int ndim);                                   // 创建零张量
Tensor* tensor_create_zeros(int* dims, int ndim);                             // 创建零张量
Tensor* tensor_create_ones(int* dims, int ndim);                              // 创建全1张量
Tensor* tensor_create_randomn(int* dims, int ndim);                           // 标准正态分布
Tensor* tensor_create_from_data(float* data, int* dims, int ndim, bool copy); // 从数据创建
void tensor_free(Tensor* t);                                                   // 释放张量
```

### 基本运算

```c
Tensor* tensor_add(Tensor* a, Tensor* b);       // 逐元素加法
Tensor* tensor_sub(Tensor* a, Tensor* b);       // 逐元素减法
Tensor* tensor_mul(Tensor* a, Tensor* b);       // 逐元素乘法
Tensor* tensor_div(Tensor* a, Tensor* b);       // 逐元素除法

Tensor* tensor_add_broadcast(Tensor* a, Tensor* b);  // 广播加法
Tensor* tensor_sub_broadcast(Tensor* a, Tensor* b);  // 广播减法
Tensor* tensor_mul_broadcast(Tensor* a, Tensor* b);  // 广播乘法
Tensor* tensor_div_broadcast(Tensor* a, Tensor* b);  // 广播除法
```

### 矩阵运算

```c
Tensor* tensor_matmul(Tensor* a, Tensor* b);          // 矩阵乘法 (支持batch)
Tensor* tensor_matmul_gemm(Tensor* a, Tensor* b, Tensor* c); // C = A*B + C
void gemm_naive(bool trans_a, bool trans_b,            // 底层 GEMM
                int M, int N, int K,
                float alpha, float* A, int lda,
                float* B, int ldb,
                float beta, float* C, int ldc);
```

### 形状操作

```c
Tensor* tensor_transpose(Tensor* a, int dim0, int dim1);           // 转置
Tensor* tensor_reshape(Tensor* a, int* new_dims, int new_ndim);     // 重塑
Tensor* tensor_slice(Tensor* a, int* starts, int* ends);            // 切片
Tensor* tensor_concatenate(Tensor** tensors, int num, int axis);    // 拼接
```

### 归约操作

```c
Tensor* tensor_softmax(Tensor* a, int axis);                      // Softmax
Tensor* tensor_sum(Tensor* a, int axis, bool keepdim);            // 求和
Tensor* tensor_mean(Tensor* a, int axis, bool keepdim);           // 均值
Tensor* tensor_max(Tensor* a, int axis, bool keepdim);            // 最大值
float tensor_sum_all(Tensor* a);                                   // 全元素求和
```

### 就地操作 (In-place)

```c
void tensor_add_(Tensor* a, Tensor* b);
void tensor_sub_(Tensor* a, Tensor* b);
void tensor_mul_(Tensor* a, Tensor* b);
void tensor_div_(Tensor* a, Tensor* b);
void tensor_fill_(Tensor* a, float val);
void tensor_scale_(Tensor* a, float val);
void tensor_clip_(Tensor* a, float min_val, float max_val);
```

### 激活函数

```c
Tensor* tensor_relu(Tensor* a);
Tensor* tensor_sigmoid(Tensor* a);
Tensor* tensor_tanh(Tensor* a);
```

### 工具函数

```c
Tensor* tensor_copy(Tensor* a);
float tensor_get(Tensor* t, int* indices);
void tensor_set(Tensor* t, int* indices, float val);
int tensor_offset(Tensor* t, int* indices);
void tensor_print(Tensor* t, const char* name);
bool broadcastable(int* dims_a, int ndim_a, int* dims_b, int ndim_b);
```

---

## 计算图 (`autograd.h`)

```c
Node* node_create(float value, bool requires_grad);   // 创建叶子节点
Node* node_create_op(OpType op, Node** inputs, int num_inputs); // 创建操作节点
void node_free(Node* node);                            // 递归释放
void forward(Node* node);                              // 前向传播
void backward(Node* node);                             // 反向传播
void zero_grad(Node* node);                            // 清零所有梯度

// 便捷节点工厂
Node* node_add(Node* a, Node* b);
Node* node_mul(Node* a, Node* b);
Node* node_relu(Node* a);
Node* node_sigmoid(Node* a);
Node* node_tanh(Node* a);
Node* node_matmul(Node* a, Node* b);
Node* node_sum(Node* a, int axis);
```

---

## 神经网络层 (`nn_layers.h`)

### Linear

```c
Linear* linear_create(int in_features, int out_features, bool use_bias);
Tensor* linear_forward(Linear* layer, Tensor* input);
Tensor* linear_backward(Linear* layer, Tensor* grad_output);
void linear_free(Linear* layer);
```

### Conv2d

```c
Conv2d* conv2d_create(int in_channels, int out_channels,
                      int kernel_h, int kernel_w,
                      int stride, int padding, bool use_bias);
Tensor* conv2d_forward(Conv2d* layer, Tensor* input);
Tensor* conv2d_backward(Conv2d* layer, Tensor* grad_output);
void conv2d_free(Conv2d* layer);

Tensor* im2col(Tensor* input, int kernel_h, int kernel_w,
               int stride_h, int stride_w, int pad_h, int pad_w);
Tensor* col2im(Tensor* col, int h_out, int w_out, int channels,
               int kernel_h, int kernel_w, int stride_h, int stride_w,
               int pad_h, int pad_w, int h_in, int w_in);
```

### 归一化层

```c
// BatchNorm1d
BatchNorm1d* batchnorm1d_create(int num_features, float eps, float momentum);
Tensor* batchnorm1d_forward(BatchNorm1d* bn, Tensor* input);
Tensor* batchnorm1d_backward(BatchNorm1d* bn, Tensor* grad_output);
void batchnorm1d_free(BatchNorm1d* bn);

// LayerNorm
LayerNorm* layernorm_create(int num_features, float eps);
Tensor* layernorm_forward(LayerNorm* ln, Tensor* input);
Tensor* layernorm_backward(LayerNorm* ln, Tensor* grad_output);
void layernorm_free(LayerNorm* ln);
```

### 正则化

```c
Dropout* dropout_create(float p);
Tensor* dropout_forward(Dropout* dp, Tensor* input);
Tensor* dropout_backward(Dropout* dp, Tensor* grad_output);
void dropout_free(Dropout* dp);
```

### 池化

```c
typedef struct { int pool_h, pool_w; int stride_h, stride_w;
                 int pad_h, pad_w; PoolMode mode; int* max_indices; } MaxPool2d;

Tensor* maxpool2d_forward(Tensor* input, MaxPool2d* pool, int h_in, int w_in,
                          int channels, Tensor** out, int* out_h, int* out_w);
Tensor* maxpool2d_backward(Tensor* grad_output, MaxPool2d* pool);
```

---

## 优化器 (`optimizers.h`)

### SGD

```c
SGD* sgd_create(float lr, float momentum, float weight_decay, bool nesterov);
void sgd_set_params(SGD* opt, Tensor** params, int num_params);
void sgd_step(SGD* opt);
void sgd_zero_grad(SGD* opt);
void sgd_free(SGD* opt);
```

### Adam

```c
Adam* adam_create(float lr, float beta1, float beta2, float eps, float weight_decay);
void adam_set_params(Adam* opt, Tensor** params, int num_params);
void adam_step(Adam* opt);
void adam_zero_grad(Adam* opt);
void adam_free(Adam* opt);
```

### AdamW

```c
AdamW* adamw_create(float lr, float beta1, float beta2, float eps, float weight_decay);
void adamw_set_params(AdamW* opt, Tensor** params, int num_params);
void adamw_step(AdamW* opt);
void adamw_free(AdamW* opt);
```

### LR Scheduler

```c
LRScheduler* lr_scheduler_create(LRSchedulerType type, float initial_lr,
                                 int step_size, float gamma,
                                 float T_max, float eta_min, int warmup_steps);
float lr_scheduler_get_lr(LRScheduler* sched);
void lr_scheduler_step(LRScheduler* sched);
void lr_scheduler_free(LRScheduler* sched);

// SchedulerType: LR_STEP | LR_COSINE | LR_LINEAR_WARMUP
```

### 工具

```c
void gradient_clip_norm(Tensor** params, int num_params, float max_norm);

// 优化器包装器
OptimizerWrapper* opt_wrapper_create(void* optimizer, LRScheduler* scheduler,
                                     void (*step)(void*), void (*zero_grad)(void*));
void opt_wrapper_step(OptimizerWrapper* w);
void opt_wrapper_free(OptimizerWrapper* w);
```

---

## 损失函数 (`loss_funcs.h`)

```c
// 损失值 (标量浮点)
float mse_loss_value(Tensor* pred, Tensor* target, LossReduction reduction);
float bce_with_logits_value(Tensor* logits, Tensor* target, LossReduction reduction);
float cross_entropy_loss_value(Tensor* logits, Tensor* target,
                               LossReduction reduction, float label_smoothing);
float focal_loss_value(Tensor* logits, Tensor* target,
                       float alpha, float gamma, LossReduction reduction);
float l1_loss_value(Tensor* pred, Tensor* target, LossReduction reduction);
float huber_loss_value(Tensor* pred, Tensor* target, float delta,
                      LossReduction reduction);

// 前向 (标量Tensor)
Tensor* mse_loss_forward(Tensor* pred, Tensor* target, LossReduction reduction);
Tensor* bce_with_logits_forward(Tensor* logits, Tensor* target, LossReduction reduction);
Tensor* cross_entropy_loss_forward(Tensor* logits, Tensor* target,
                                   LossReduction reduction, float label_smoothing);
Tensor* focal_loss_forward(Tensor* logits, Tensor* target,
                           float alpha, float gamma, LossReduction reduction);
Tensor* l1_loss_forward(Tensor* pred, Tensor* target, LossReduction reduction);
Tensor* huber_loss_forward(Tensor* pred, Tensor* target, float delta,
                           LossReduction reduction);

// 反向 (梯度Tensor)
Tensor* mse_loss_backward(Tensor* pred, Tensor* target, LossReduction reduction);
Tensor* bce_with_logits_backward(Tensor* logits, Tensor* target, LossReduction reduction);
Tensor* cross_entropy_loss_backward(Tensor* logits, Tensor* target,
                                    LossReduction reduction, float label_smoothing);
Tensor* focal_loss_backward(Tensor* logits, Tensor* target,
                            float alpha, float gamma, LossReduction reduction);
Tensor* l1_loss_backward(Tensor* pred, Tensor* target, LossReduction reduction);
Tensor* huber_loss_backward(Tensor* pred, Tensor* target, float delta,
                            LossReduction reduction);
```

### LossReduction 枚举

```c
REDUCE_MEAN  // 除以样本数
REDUCE_SUM   // 直接求和
```

---

## 使用示例

### 简单 MLP 训练

```c
Linear* fc = linear_create(784, 10, true);

Tensor* x = tensor_create_randomn((int[]){32, 784}, 2);
Tensor* y = tensor_create_from_data(labels, (int[]){32}, 1, true);

Tensor* logits = linear_forward(fc, x);
float loss = cross_entropy_loss_value(logits, y, REDUCE_MEAN, 0.0f);

Tensor* d_logits = cross_entropy_loss_backward(logits, y, REDUCE_MEAN, 0.0f);
Tensor* d_input = linear_backward(fc, d_logits);

// 手动 SGD 更新
for (int i = 0; i < fc->weight->size; i++)
    fc->weight->data[i] -= 0.01f * fc->weight_grad->data[i];
tensor_fill_(fc->weight_grad, 0);

tensor_free(x); tensor_free(y); tensor_free(logits);
tensor_free(d_logits); tensor_free(d_input);
linear_free(fc);
```
