#ifndef TENSOR_OPS_H
#define TENSOR_OPS_H

#include <stddef.h>
#include <stdbool.h>

typedef struct {
    int* dims;
    int* strides;
    int ndim;
    int size;
    float* data;
    bool owns_data;
} Tensor;

Tensor* tensor_create(int* dims, int ndim);
Tensor* tensor_create_zeros(int* dims, int ndim);
Tensor* tensor_create_ones(int* dims, int ndim);
Tensor* tensor_create_randomn(int* dims, int ndim);
Tensor* tensor_create_from_data(float* data, int* dims, int ndim, bool copy);
void tensor_free(Tensor* t);

float tensor_get(Tensor* t, int* indices);
void tensor_set(Tensor* t, int* indices, float val);
int tensor_offset(Tensor* t, int* indices);

Tensor* tensor_add(Tensor* a, Tensor* b);
Tensor* tensor_sub(Tensor* a, Tensor* b);
Tensor* tensor_mul(Tensor* a, Tensor* b);
Tensor* tensor_div(Tensor* a, Tensor* b);

Tensor* tensor_add_broadcast(Tensor* a, Tensor* b);
Tensor* tensor_sub_broadcast(Tensor* a, Tensor* b);
Tensor* tensor_mul_broadcast(Tensor* a, Tensor* b);
Tensor* tensor_div_broadcast(Tensor* a, Tensor* b);

Tensor* tensor_matmul(Tensor* a, Tensor* b);
Tensor* tensor_matmul_gemm(Tensor* a, Tensor* b, Tensor* c);

Tensor* tensor_transpose(Tensor* a, int dim0, int dim1);
Tensor* tensor_reshape(Tensor* a, int* new_dims, int new_ndim);
Tensor* tensor_slice(Tensor* a, int* starts, int* ends);
Tensor* tensor_concatenate(Tensor** tensors, int num, int axis);
Tensor* tensor_softmax(Tensor* a, int axis);
Tensor* tensor_sum(Tensor* a, int axis, bool keepdim);
Tensor* tensor_mean(Tensor* a, int axis, bool keepdim);
Tensor* tensor_max(Tensor* a, int axis, bool keepdim);

void tensor_add_(Tensor* a, Tensor* b);
void tensor_sub_(Tensor* a, Tensor* b);
void tensor_mul_(Tensor* a, Tensor* b);
void tensor_div_(Tensor* a, Tensor* b);
void tensor_fill_(Tensor* a, float val);
void tensor_scale_(Tensor* a, float val);
void tensor_clip_(Tensor* a, float min_val, float max_val);

Tensor* tensor_relu(Tensor* a);
Tensor* tensor_sigmoid(Tensor* a);
Tensor* tensor_tanh(Tensor* a);

Tensor* tensor_copy(Tensor* a);
float tensor_sum_all(Tensor* a);
void tensor_print(Tensor* t, const char* name);

bool broadcastable(int* dims_a, int ndim_a, int* dims_b, int ndim_b);
void broadcast_dims(int* dims_a, int ndim_a, int* dims_b, int ndim_b,
                    int* out_dims, int* out_ndim);

void gemm_naive(bool trans_a, bool trans_b,
                int M, int N, int K,
                float alpha, float* A, int lda,
                float* B, int ldb,
                float beta, float* C, int ldc);

#endif
