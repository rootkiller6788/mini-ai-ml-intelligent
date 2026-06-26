#include "nn_layers.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

Linear* linear_create(int in_features, int out_features, bool use_bias) {
    Linear* l = (Linear*)malloc(sizeof(Linear));
    l->in_features = in_features;
    l->out_features = out_features;
    l->weight = tensor_create_randomn((int[]){out_features, in_features}, 2);
    l->weight_grad = tensor_create_zeros((int[]){out_features, in_features}, 2);
    if (use_bias) {
        l->bias = tensor_create_zeros((int[]){out_features}, 1);
        l->bias_grad = tensor_create_zeros((int[]){out_features}, 1);
    } else {
        l->bias = NULL;
        l->bias_grad = NULL;
    }
    l->input_cache = NULL;
    return l;
}

Tensor* linear_forward(Linear* layer, Tensor* input) {
    if (layer->input_cache) tensor_free(layer->input_cache);
    layer->input_cache = tensor_copy(input);

    Tensor* out = tensor_matmul(input, tensor_transpose(layer->weight, 0, 1));
    if (layer->bias) {
        Tensor* out_b = tensor_add_broadcast(out, layer->bias);
        tensor_free(out);
        return out_b;
    }
    return out;
}

Tensor* linear_backward(Linear* layer, Tensor* grad_output) {
    if (!layer->input_cache) return NULL;

    Tensor* weight_t = tensor_transpose(layer->weight, 0, 1);

    int batch = grad_output->ndim > 1 ? grad_output->dims[0] : 1;
    int M = batch;
    int N = layer->in_features;
    int K = layer->out_features;

    Tensor* d_input = tensor_matmul(grad_output, layer->weight);

    Tensor* grad_reshape = grad_output->ndim == 1
        ? tensor_reshape(grad_output, (int[]){1, layer->out_features}, 2)
        : tensor_copy(grad_output);
    Tensor* input_reshape = layer->input_cache->ndim == 1
        ? tensor_reshape(layer->input_cache, (int[]){1, layer->in_features}, 2)
        : tensor_copy(layer->input_cache);

    Tensor* dw = tensor_matmul(
        tensor_transpose(grad_reshape, 0, 1), input_reshape);
    tensor_add_(layer->weight_grad, dw);

    if (layer->bias && layer->bias_grad) {
        Tensor* db = tensor_sum(grad_reshape, 0, false);
        tensor_add_(layer->bias_grad, db);
        tensor_free(db);
    }

    tensor_free(weight_t);
    tensor_free(grad_reshape);
    tensor_free(input_reshape);
    tensor_free(dw);

    return d_input;
}

void linear_free(Linear* layer) {
    if (!layer) return;
    tensor_free(layer->weight);
    tensor_free(layer->weight_grad);
    if (layer->bias) tensor_free(layer->bias);
    if (layer->bias_grad) tensor_free(layer->bias_grad);
    if (layer->input_cache) tensor_free(layer->input_cache);
    free(layer);
}

Conv2d* conv2d_create(int in_channels, int out_channels,
                      int kernel_h, int kernel_w,
                      int stride, int padding, bool use_bias) {
    Conv2d* l = (Conv2d*)malloc(sizeof(Conv2d));
    l->in_channels = in_channels;
    l->out_channels = out_channels;
    l->kernel_h = kernel_h;
    l->kernel_w = kernel_w;
    l->stride_h = stride;
    l->stride_w = stride;
    l->pad_h = padding;
    l->pad_w = padding;
    int k_size = out_channels * in_channels * kernel_h * kernel_w;
    l->weight = tensor_create_randomn(
        (int[]){out_channels, in_channels, kernel_h, kernel_w}, 4);
    l->weight_grad = tensor_create_zeros(
        (int[]){out_channels, in_channels, kernel_h, kernel_w}, 4);
    if (use_bias) {
        l->bias = tensor_create_zeros((int[]){out_channels}, 1);
        l->bias_grad = tensor_create_zeros((int[]){out_channels}, 1);
    } else {
        l->bias = NULL;
        l->bias_grad = NULL;
    }
    l->im2col_cache = NULL;
    l->input_cache = NULL;
    l->input_h = 0;
    l->input_w = 0;
    return l;
}

Tensor* im2col(Tensor* input, int kernel_h, int kernel_w,
               int stride_h, int stride_w, int pad_h, int pad_w) {
    int N = input->dims[0];
    int C = input->dims[1];
    int H = input->dims[2];
    int W = input->dims[3];

    int H_out = (H + 2 * pad_h - kernel_h) / stride_h + 1;
    int W_out = (W + 2 * pad_w - kernel_w) / stride_w + 1;

    int* col_dims = (int[]){N, H_out * W_out, C * kernel_h * kernel_w};
    Tensor* col = tensor_create(col_dims, 3);

    for (int n = 0; n < N; n++) {
        for (int h = 0; h < H_out; h++) {
            for (int w = 0; w < W_out; w++) {
                int col_row = n * H_out * W_out + h * W_out + w;
                for (int c = 0; c < C; c++) {
                    for (int kh = 0; kh < kernel_h; kh++) {
                        for (int kw = 0; kw < kernel_w; kw++) {
                            int h_in = h * stride_h + kh - pad_h;
                            int w_in = w * stride_w + kw - pad_w;
                            int col_col = c * kernel_h * kernel_w + kh * kernel_w + kw;
                            float val = 0;
                            if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                                val = input->data[n * (C * H * W) + c * (H * W) + h_in * W + w_in];
                            }
                            col->data[col_row * (C * kernel_h * kernel_w) + col_col] = val;
                        }
                    }
                }
            }
        }
    }
    return col;
}

Tensor* col2im(Tensor* col, int h_out, int w_out, int channels,
               int kernel_h, int kernel_w, int stride_h, int stride_w,
               int pad_h, int pad_w, int h_in, int w_in) {
    int N = col->dims[0];
    int* out_dims = (int[]){N, channels, h_in, w_in};
    Tensor* img = tensor_create_zeros(out_dims, 4);

    for (int n = 0; n < N; n++) {
        for (int h = 0; h < h_out; h++) {
            for (int w = 0; w < w_out; w++) {
                int col_row = n * h_out * w_out + h * w_out + w;
                for (int c = 0; c < channels; c++) {
                    for (int kh = 0; kh < kernel_h; kh++) {
                        for (int kw = 0; kw < kernel_w; kw++) {
                            int hi = h * stride_h + kh - pad_h;
                            int wi = w * stride_w + kw - pad_w;
                            if (hi >= 0 && hi < h_in && wi >= 0 && wi < w_in) {
                                int col_col = c * kernel_h * kernel_w + kh * kernel_w + kw;
                                int img_idx = n * channels * h_in * w_in + c * h_in * w_in + hi * w_in + wi;
                                img->data[img_idx] += col->data[col_row * (channels * kernel_h * kernel_w) + col_col];
                            }
                        }
                    }
                }
            }
        }
    }
    return img;
}

Tensor* conv2d_forward(Conv2d* layer, Tensor* input) {
    int N = input->dims[0];
    int C = input->dims[1];
    int H = input->dims[2];
    int W = input->dims[3];

    layer->input_h = H;
    layer->input_w = W;
    if (layer->input_cache) tensor_free(layer->input_cache);
    layer->input_cache = tensor_copy(input);

    int H_out = (H + 2 * layer->pad_h - layer->kernel_h) / layer->stride_h + 1;
    int W_out = (W + 2 * layer->pad_w - layer->kernel_w) / layer->stride_w + 1;

    Tensor* col = im2col(input, layer->kernel_h, layer->kernel_w,
                         layer->stride_h, layer->stride_w,
                         layer->pad_h, layer->pad_w);
    if (layer->im2col_cache) tensor_free(layer->im2col_cache);
    layer->im2col_cache = col;

    Tensor* w_reshaped = tensor_reshape(layer->weight,
        (int[]){layer->out_channels, layer->in_channels * layer->kernel_h * layer->kernel_w}, 2);

    Tensor* out = tensor_matmul(col, tensor_transpose(w_reshaped, 1, 0));

    int* final_dims = (int[]){N, H_out, W_out, layer->out_channels};
    Tensor* reshaped_out = tensor_reshape(out, final_dims, 4);

    if (layer->bias) {
        Tensor* bias_reshaped = tensor_reshape(layer->bias, (int[]){1, 1, 1, layer->out_channels}, 4);
        Tensor* out_biased = tensor_add_broadcast(reshaped_out, bias_reshaped);
        tensor_free(reshaped_out);
        tensor_free(bias_reshaped);
        tensor_free(w_reshaped);
        return out_biased;
    }

    tensor_free(out);
    tensor_free(w_reshaped);
    return reshaped_out;
}

Tensor* conv2d_backward(Conv2d* layer, Tensor* grad_output) {
    int N = grad_output->dims[0];
    int H_out = grad_output->dims[1];
    int W_out = grad_output->dims[2];
    int OC = grad_output->dims[3];

    Tensor* grad_reshaped = tensor_reshape(grad_output,
        (int[]){N * H_out * W_out, OC}, 2);

    Tensor* w_reshaped = tensor_reshape(layer->weight,
        (int[]){OC, layer->in_channels * layer->kernel_h * layer->kernel_w}, 2);

    Tensor* grad_col = tensor_matmul(grad_reshaped, w_reshaped);

    Tensor* col_t = tensor_transpose(layer->im2col_cache, 1, 0);
    Tensor* dw_reshaped = tensor_matmul(
        tensor_transpose(grad_reshaped, 0, 1), layer->im2col_cache);
    Tensor* dw_reshaped2 = tensor_reshape(dw_reshaped,
        (int[]){OC, layer->in_channels, layer->kernel_h, layer->kernel_w}, 4);
    tensor_add_(layer->weight_grad, dw_reshaped2);

    if (layer->bias_grad) {
        Tensor* db = tensor_sum(tensor_sum(tensor_sum(grad_output, 0, false), 0, false), 0, false);
        tensor_add_(layer->bias_grad, db);
        tensor_free(db);
    }

    Tensor* d_input = col2im(grad_col, H_out, W_out, layer->in_channels,
                             layer->kernel_h, layer->kernel_w,
                             layer->stride_h, layer->stride_w,
                             layer->pad_h, layer->pad_w,
                             layer->input_h, layer->input_w);

    tensor_free(grad_reshaped);
    tensor_free(w_reshaped);
    tensor_free(grad_col);
    tensor_free(col_t);
    tensor_free(dw_reshaped);
    tensor_free(dw_reshaped2);

    return d_input;
}

void conv2d_free(Conv2d* layer) {
    if (!layer) return;
    tensor_free(layer->weight);
    tensor_free(layer->weight_grad);
    if (layer->bias) tensor_free(layer->bias);
    if (layer->bias_grad) tensor_free(layer->bias_grad);
    if (layer->im2col_cache) tensor_free(layer->im2col_cache);
    if (layer->input_cache) tensor_free(layer->input_cache);
    free(layer);
}

Tensor* maxpool2d_forward(Tensor* input, MaxPool2d* pool, int h_in, int w_in,
                          int channels, Tensor** out, int* out_h, int* out_w) {
    int N = input->dims[0];
    *out_h = (h_in + 2 * pool->pad_h - pool->pool_h) / pool->stride_h + 1;
    *out_w = (w_in + 2 * pool->pad_w - pool->pool_w) / pool->stride_w + 1;

    int* od = (int[]){N, channels, *out_h, *out_w};
    *out = tensor_create_zeros(od, 4);
    pool->max_indices = (int*)malloc(sizeof(int) * (*out)->size);

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < *out_h; h++) {
                for (int w = 0; w < *out_w; w++) {
                    float max_val = -1e30f;
                    int max_idx = -1;
                    for (int ph = 0; ph < pool->pool_h; ph++) {
                        for (int pw = 0; pw < pool->pool_w; pw++) {
                            int hi = h * pool->stride_h + ph - pool->pad_h;
                            int wi = w * pool->stride_w + pw - pool->pad_w;
                            if (hi >= 0 && hi < h_in && wi >= 0 && wi < w_in) {
                                int idx = n * channels * h_in * w_in + c * h_in * w_in + hi * w_in + wi;
                                if (input->data[idx] > max_val) {
                                    max_val = input->data[idx];
                                    max_idx = idx;
                                }
                            }
                        }
                    }
                    int out_idx = n * channels * (*out_h) * (*out_w) + c * (*out_h) * (*out_w) + h * (*out_w) + w;
                    (*out)->data[out_idx] = max_val;
                    pool->max_indices[out_idx] = max_idx;
                }
            }
        }
    }
    return *out;
}

Tensor* maxpool2d_backward(Tensor* grad_output, MaxPool2d* pool) {
    Tensor* d_input = tensor_create_zeros(
        (int[]){grad_output->dims[0], grad_output->dims[1],
                grad_output->dims[2] * pool->stride_h + pool->pool_h,
                grad_output->dims[3] * pool->stride_w + pool->pool_w}, 4);

    for (int i = 0; i < grad_output->size; i++) {
        if (pool->max_indices[i] >= 0) {
            d_input->data[pool->max_indices[i]] += grad_output->data[i];
        }
    }
    return d_input;
}

BatchNorm1d* batchnorm1d_create(int num_features, float eps, float momentum) {
    BatchNorm1d* bn = (BatchNorm1d*)malloc(sizeof(BatchNorm1d));
    bn->num_features = num_features;
    bn->eps = eps;
    bn->momentum = momentum;
    bn->gamma = tensor_create_ones((int[]){num_features}, 1);
    bn->beta = tensor_create_zeros((int[]){num_features}, 1);
    bn->gamma_grad = tensor_create_zeros((int[]){num_features}, 1);
    bn->beta_grad = tensor_create_zeros((int[]){num_features}, 1);
    bn->running_mean = tensor_create_zeros((int[]){num_features}, 1);
    bn->running_var = tensor_create_ones((int[]){num_features}, 1);
    bn->x_hat_cache = NULL;
    bn->var_cache = NULL;
    bn->x_centered_cache = NULL;
    bn->is_training = true;
    return bn;
}

Tensor* batchnorm1d_forward(BatchNorm1d* bn, Tensor* input) {
    int N = input->dims[0];
    int D = input->dims[1];
    float eps = bn->eps;

    if (bn->x_hat_cache) tensor_free(bn->x_hat_cache);
    if (bn->var_cache) tensor_free(bn->var_cache);
    if (bn->x_centered_cache) tensor_free(bn->x_centered_cache);

    Tensor* mean = tensor_sum(input, 0, true);
    for (int i = 0; i < D; i++) mean->data[i] /= (float)N;

    Tensor* x_centered = tensor_sub_broadcast(input,
        tensor_reshape(mean, (int[]){D}, 1));
    bn->x_centered_cache = tensor_copy(x_centered);

    Tensor* var = tensor_create_zeros((int[]){D}, 1);
    for (int n = 0; n < N; n++) {
        for (int d = 0; d < D; d++) {
            float diff = x_centered->data[n * D + d];
            var->data[d] += diff * diff;
        }
    }
    for (int d = 0; d < D; d++) var->data[d] /= (float)N;
    bn->var_cache = tensor_copy(var);

    Tensor* std = tensor_create(var->dims, var->ndim);
    for (int i = 0; i < std->size; i++)
        std->data[i] = sqrtf(var->data[i] + eps);

    Tensor* x_hat = tensor_copy(x_centered);
    for (int n = 0; n < N; n++) {
        for (int d = 0; d < D; d++) {
            x_hat->data[n * D + d] /= std->data[d];
        }
    }
    bn->x_hat_cache = tensor_copy(x_hat);

    Tensor* out = tensor_copy(x_hat);
    for (int n = 0; n < N; n++) {
        for (int d = 0; d < D; d++) {
            out->data[n * D + d] = out->data[n * D + d] * bn->gamma->data[d] + bn->beta->data[d];
        }
    }

    if (bn->is_training) {
        for (int d = 0; d < D; d++) {
            bn->running_mean->data[d] = bn->momentum * bn->running_mean->data[d] + (1 - bn->momentum) * mean->data[d];
            bn->running_var->data[d] = bn->momentum * bn->running_var->data[d] + (1 - bn->momentum) * var->data[d];
        }
    } else {
        for (int n = 0; n < N; n++) {
            for (int d = 0; d < D; d++) {
                out->data[n * D + d] = bn->gamma->data[d] * ((input->data[n * D + d] - bn->running_mean->data[d]) / sqrtf(bn->running_var->data[d] + eps)) + bn->beta->data[d];
            }
        }
    }

    tensor_free(mean);
    tensor_free(std);
    tensor_free(x_centered);
    tensor_free(x_hat);
    tensor_free(var);

    return out;
}

Tensor* batchnorm1d_backward(BatchNorm1d* bn, Tensor* grad_output) {
    int N = grad_output->dims[0];
    int D = grad_output->dims[1];

    Tensor* dgamma = tensor_create_zeros((int[]){D}, 1);
    Tensor* dbeta = tensor_create_zeros((int[]){D}, 1);
    for (int n = 0; n < N; n++) {
        for (int d = 0; d < D; d++) {
            dgamma->data[d] += grad_output->data[n * D + d] * bn->x_hat_cache->data[n * D + d];
            dbeta->data[d] += grad_output->data[n * D + d];
        }
    }
    tensor_add_(bn->gamma_grad, dgamma);
    tensor_add_(bn->beta_grad, dbeta);

    Tensor* dx_hat = tensor_copy(grad_output);
    for (int n = 0; n < N; n++) {
        for (int d = 0; d < D; d++) {
            dx_hat->data[n * D + d] *= bn->gamma->data[d];
        }
    }

    float inv_N = 1.0f / (float)N;
    Tensor* dx = tensor_create_zeros((int[]){N, D}, 2);
    for (int d = 0; d < D; d++) {
        float sum_dx = 0, sum_dx_x = 0;
        for (int n = 0; n < N; n++) {
            sum_dx += dx_hat->data[n * D + d];
            sum_dx_x += dx_hat->data[n * D + d] * bn->x_hat_cache->data[n * D + d];
        }
        float std_inv = 1.0f / sqrtf(bn->var_cache->data[d] + bn->eps);
        for (int n = 0; n < N; n++) {
            dx->data[n * D + d] = std_inv * inv_N * (N * dx_hat->data[n * D + d] - sum_dx - bn->x_hat_cache->data[n * D + d] * sum_dx_x);
        }
    }

    tensor_free(dgamma);
    tensor_free(dbeta);
    tensor_free(dx_hat);
    return dx;
}

void batchnorm1d_free(BatchNorm1d* bn) {
    if (!bn) return;
    tensor_free(bn->gamma);
    tensor_free(bn->beta);
    tensor_free(bn->gamma_grad);
    tensor_free(bn->beta_grad);
    tensor_free(bn->running_mean);
    tensor_free(bn->running_var);
    if (bn->x_hat_cache) tensor_free(bn->x_hat_cache);
    if (bn->var_cache) tensor_free(bn->var_cache);
    if (bn->x_centered_cache) tensor_free(bn->x_centered_cache);
    free(bn);
}

LayerNorm* layernorm_create(int num_features, float eps) {
    LayerNorm* ln = (LayerNorm*)malloc(sizeof(LayerNorm));
    ln->num_features = num_features;
    ln->eps = eps;
    ln->gamma = tensor_create_ones((int[]){num_features}, 1);
    ln->beta = tensor_create_zeros((int[]){num_features}, 1);
    ln->gamma_grad = tensor_create_zeros((int[]){num_features}, 1);
    ln->beta_grad = tensor_create_zeros((int[]){num_features}, 1);
    ln->mean_cache = NULL;
    ln->inv_std_cache = NULL;
    ln->x_centered_cache = NULL;
    return ln;
}

Tensor* layernorm_forward(LayerNorm* ln, Tensor* input) {
    int N = input->dims[0];
    int D = input->dims[1];
    float eps = ln->eps;

    if (ln->mean_cache) tensor_free(ln->mean_cache);
    if (ln->inv_std_cache) tensor_free(ln->inv_std_cache);
    if (ln->x_centered_cache) tensor_free(ln->x_centered_cache);

    Tensor* mean = tensor_create_zeros((int[]){N, 1}, 2);
    Tensor* var = tensor_create_zeros((int[]){N, 1}, 2);
    for (int n = 0; n < N; n++) {
        float sum = 0, sum_sq = 0;
        for (int d = 0; d < D; d++) {
            sum += input->data[n * D + d];
        }
        mean->data[n] = sum / D;
        for (int d = 0; d < D; d++) {
            float diff = input->data[n * D + d] - mean->data[n];
            sum_sq += diff * diff;
        }
        var->data[n] = sum_sq / D;
    }
    ln->mean_cache = mean;
    ln->inv_std_cache = tensor_create((int[]){N, 1}, 2);
    for (int n = 0; n < N; n++) {
        ln->inv_std_cache->data[n] = 1.0f / sqrtf(var->data[n] + eps);
    }
    tensor_free(var);

    Tensor* out = tensor_create((int[]){N, D}, 2);
    for (int n = 0; n < N; n++) {
        for (int d = 0; d < D; d++) {
            float x_hat = (input->data[n * D + d] - mean->data[n]) * ln->inv_std_cache->data[n];
            out->data[n * D + d] = x_hat * ln->gamma->data[d] + ln->beta->data[d];
        }
    }
    return out;
}

Tensor* layernorm_backward(LayerNorm* ln, Tensor* grad_output) {
    int N = grad_output->dims[0];
    int D = grad_output->dims[1];

    Tensor* dgamma = tensor_create_zeros((int[]){D}, 1);
    Tensor* dbeta = tensor_create_zeros((int[]){D}, 1);
    for (int n = 0; n < N; n++) {
        for (int d = 0; d < D; d++) {
            float x_centered = (input_cache_placeholder(n, d, ln));
            dgamma->data[d] += grad_output->data[n * D + d] * x_centered * ln->inv_std_cache->data[n];
            dbeta->data[d] += grad_output->data[n * D + d];
        }
    }
    tensor_add_(ln->gamma_grad, dgamma);
    tensor_add_(ln->beta_grad, dbeta);

    Tensor* dx = tensor_create_zeros((int[]){N, D}, 2);
    for (int n = 0; n < N; n++) {
        float sum_dx = 0, sum_dx_x = 0;
        for (int d = 0; d < D; d++) {
            float x_centered = (input_cache_placeholder(n, d, ln));
            float dout = grad_output->data[n * D + d] * ln->gamma->data[d];
            sum_dx += dout;
            sum_dx_x += dout * x_centered * ln->inv_std_cache->data[n];
        }
        for (int d = 0; d < D; d++) {
            float x_centered = (input_cache_placeholder(n, d, ln));
            dx->data[n * D + d] = (1.0f / D) * ln->inv_std_cache->data[n] * (D * grad_output->data[n * D + d] * ln->gamma->data[d] - sum_dx - x_centered * ln->inv_std_cache->data[n] * sum_dx_x);
        }
    }

    tensor_free(dgamma);
    tensor_free(dbeta);
    return dx;
}

void layernorm_free(LayerNorm* ln) {
    if (!ln) return;
    tensor_free(ln->gamma);
    tensor_free(ln->beta);
    tensor_free(ln->gamma_grad);
    tensor_free(ln->beta_grad);
    if (ln->mean_cache) tensor_free(ln->mean_cache);
    if (ln->inv_std_cache) tensor_free(ln->inv_std_cache);
    if (ln->x_centered_cache) tensor_free(ln->x_centered_cache);
    free(ln);
}

Dropout* dropout_create(float p) {
    Dropout* dp = (Dropout*)malloc(sizeof(Dropout));
    dp->p = p;
    dp->is_training = true;
    dp->mask = NULL;
    return dp;
}

Tensor* dropout_forward(Dropout* dp, Tensor* input) {
    if (!dp->is_training) return tensor_copy(input);

    if (dp->mask) tensor_free(dp->mask);
    dp->mask = tensor_create(input->dims, input->ndim);

    Tensor* out = tensor_copy(input);
    float scale = 1.0f / (1.0f - dp->p);
    for (int i = 0; i < out->size; i++) {
        float r = (float)rand() / (float)RAND_MAX;
        if (r < dp->p) {
            out->data[i] = 0;
            dp->mask->data[i] = 0;
        } else {
            out->data[i] *= scale;
            dp->mask->data[i] = 1;
        }
    }
    return out;
}

Tensor* dropout_backward(Dropout* dp, Tensor* grad_output) {
    Tensor* dx = tensor_copy(grad_output);
    float scale = 1.0f / (1.0f - dp->p);
    for (int i = 0; i < dx->size; i++) {
        dx->data[i] *= dp->mask->data[i] * scale;
    }
    return dx;
}

void dropout_free(Dropout* dp) {
    if (!dp) return;
    if (dp->mask) tensor_free(dp->mask);
    free(dp);
}

LayerParam* param_list_create(void) {
    LayerParam* head = (LayerParam*)malloc(sizeof(LayerParam));
    head->param = NULL;
    head->grad = NULL;
    head->next = NULL;
    return head;
}

void param_list_add(LayerParam** head, Tensor* param, Tensor* grad) {
    LayerParam* node = (LayerParam*)malloc(sizeof(LayerParam));
    node->param = param;
    node->grad = grad;
    node->next = *head;
    *head = node;
}

void param_list_free(LayerParam* head) {
    while (head) {
        LayerParam* next = head->next;
        free(head);
        head = next;
    }
}

static float input_cache_placeholder(int n, int d, LayerNorm* ln) {
    (void)n;
    (void)d;
    (void)ln;
    return 1.0f;
}
