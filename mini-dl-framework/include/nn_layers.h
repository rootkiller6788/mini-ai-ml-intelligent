#ifndef NN_LAYERS_H
#define NN_LAYERS_H

#include "tensor_ops.h"
#include <stdbool.h>

typedef struct {
    Tensor* weight;
    Tensor* bias;
    Tensor* weight_grad;
    Tensor* bias_grad;
    Tensor* input_cache;
    int in_features;
    int out_features;
} Linear;

Linear* linear_create(int in_features, int out_features, bool use_bias);
Tensor* linear_forward(Linear* layer, Tensor* input);
Tensor* linear_backward(Linear* layer, Tensor* grad_output);
void linear_free(Linear* layer);

typedef struct {
    int in_channels;
    int out_channels;
    int kernel_h, kernel_w;
    int stride_h, stride_w;
    int pad_h, pad_w;
    Tensor* weight;
    Tensor* bias;
    Tensor* weight_grad;
    Tensor* bias_grad;
    Tensor* im2col_cache;
    Tensor* input_cache;
    int input_h, input_w;
} Conv2d;

Conv2d* conv2d_create(int in_channels, int out_channels,
                      int kernel_h, int kernel_w,
                      int stride, int padding, bool use_bias);
Tensor* conv2d_forward(Conv2d* layer, Tensor* input);
Tensor* conv2d_backward(Conv2d* layer, Tensor* grad_output);
Tensor* im2col(Tensor* input, int kernel_h, int kernel_w,
               int stride_h, int stride_w, int pad_h, int pad_w);
Tensor* col2im(Tensor* col, int h_out, int w_out, int channels,
               int kernel_h, int kernel_w, int stride_h, int stride_w,
               int pad_h, int pad_w, int h_in, int w_in);
void conv2d_free(Conv2d* layer);

typedef enum {
    POOL_MAX,
    POOL_AVG
} PoolMode;

typedef struct {
    int pool_h, pool_w;
    int stride_h, stride_w;
    int pad_h, pad_w;
    PoolMode mode;
    int* max_indices;
} MaxPool2d;

Tensor* maxpool2d_forward(Tensor* input, MaxPool2d* pool, int h_in, int w_in,
                          int channels, Tensor** out, int* out_h, int* out_w);
Tensor* maxpool2d_backward(Tensor* grad_output, MaxPool2d* pool);

typedef struct {
    int num_features;
    float eps;
    float momentum;
    Tensor* gamma;
    Tensor* beta;
    Tensor* gamma_grad;
    Tensor* beta_grad;
    Tensor* running_mean;
    Tensor* running_var;
    Tensor* x_hat_cache;
    Tensor* var_cache;
    Tensor* x_centered_cache;
    bool is_training;
} BatchNorm1d;

BatchNorm1d* batchnorm1d_create(int num_features, float eps, float momentum);
Tensor* batchnorm1d_forward(BatchNorm1d* bn, Tensor* input);
Tensor* batchnorm1d_backward(BatchNorm1d* bn, Tensor* grad_output);
void batchnorm1d_free(BatchNorm1d* bn);

typedef struct {
    int num_features;
    float eps;
    Tensor* gamma;
    Tensor* beta;
    Tensor* gamma_grad;
    Tensor* beta_grad;
    Tensor* mean_cache;
    Tensor* inv_std_cache;
    Tensor* x_centered_cache;
} LayerNorm;

LayerNorm* layernorm_create(int num_features, float eps);
Tensor* layernorm_forward(LayerNorm* ln, Tensor* input);
Tensor* layernorm_backward(LayerNorm* ln, Tensor* grad_output);
void layernorm_free(LayerNorm* ln);

typedef struct {
    float p;
    bool is_training;
    Tensor* mask;
} Dropout;

Dropout* dropout_create(float p);
Tensor* dropout_forward(Dropout* dp, Tensor* input);
Tensor* dropout_backward(Dropout* dp, Tensor* grad_output);
void dropout_free(Dropout* dp);

typedef struct LayerParam {
    Tensor* param;
    Tensor* grad;
    struct LayerParam* next;
} LayerParam;

LayerParam* param_list_create(void);
void param_list_add(LayerParam** head, Tensor* param, Tensor* grad);
void param_list_free(LayerParam* head);

#endif
