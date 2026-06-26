#include "tensor_ops.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

Tensor* tensor_create(int* dims, int ndim) {
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    t->ndim = ndim;
    t->dims = (int*)malloc(sizeof(int) * ndim);
    t->strides = (int*)malloc(sizeof(int) * ndim);
    t->size = 1;
    for (int i = 0; i < ndim; i++) {
        t->dims[i] = dims[i];
        t->size *= dims[i];
    }
    t->strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
        t->strides[i] = t->strides[i + 1] * t->dims[i + 1];
    }
    t->data = (float*)calloc(t->size, sizeof(float));
    t->owns_data = true;
    return t;
}

Tensor* tensor_create_zeros(int* dims, int ndim) {
    return tensor_create(dims, ndim);
}

Tensor* tensor_create_ones(int* dims, int ndim) {
    Tensor* t = tensor_create(dims, ndim);
    for (int i = 0; i < t->size; i++) {
        t->data[i] = 1.0f;
    }
    return t;
}

Tensor* tensor_create_randomn(int* dims, int ndim) {
    Tensor* t = tensor_create(dims, ndim);
    for (int i = 0; i < t->size; i++) {
        float u1 = (float)rand() / (float)RAND_MAX;
        float u2 = (float)rand() / (float)RAND_MAX;
        float r = sqrtf(-2.0f * logf(u1 + 1e-12f));
        float theta = 2.0f * 3.14159265f * u2;
        t->data[i] = r * cosf(theta) * 0.02f;
    }
    return t;
}

Tensor* tensor_create_from_data(float* data, int* dims, int ndim, bool copy) {
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    t->ndim = ndim;
    t->dims = (int*)malloc(sizeof(int) * ndim);
    t->strides = (int*)malloc(sizeof(int) * ndim);
    t->size = 1;
    for (int i = 0; i < ndim; i++) {
        t->dims[i] = dims[i];
        t->size *= dims[i];
    }
    t->strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
        t->strides[i] = t->strides[i + 1] * t->dims[i + 1];
    }
    if (copy) {
        t->data = (float*)malloc(sizeof(float) * t->size);
        memcpy(t->data, data, sizeof(float) * t->size);
        t->owns_data = true;
    } else {
        t->data = data;
        t->owns_data = false;
    }
    return t;
}

void tensor_free(Tensor* t) {
    if (!t) return;
    if (t->dims) free(t->dims);
    if (t->strides) free(t->strides);
    if (t->data && t->owns_data) free(t->data);
    free(t);
}

float tensor_get(Tensor* t, int* indices) {
    int offset = tensor_offset(t, indices);
    return t->data[offset];
}

void tensor_set(Tensor* t, int* indices, float val) {
    int offset = tensor_offset(t, indices);
    t->data[offset] = val;
}

int tensor_offset(Tensor* t, int* indices) {
    int offset = 0;
    for (int i = 0; i < t->ndim; i++) {
        offset += indices[i] * t->strides[i];
    }
    return offset;
}

Tensor* tensor_add(Tensor* a, Tensor* b) {
    Tensor* out = tensor_create(a->dims, a->ndim);
    for (int i = 0; i < out->size; i++) {
        out->data[i] = a->data[i] + b->data[i];
    }
    return out;
}

Tensor* tensor_sub(Tensor* a, Tensor* b) {
    Tensor* out = tensor_create(a->dims, a->ndim);
    for (int i = 0; i < out->size; i++) {
        out->data[i] = a->data[i] - b->data[i];
    }
    return out;
}

Tensor* tensor_mul(Tensor* a, Tensor* b) {
    Tensor* out = tensor_create(a->dims, a->ndim);
    for (int i = 0; i < out->size; i++) {
        out->data[i] = a->data[i] * b->data[i];
    }
    return out;
}

Tensor* tensor_div(Tensor* a, Tensor* b) {
    Tensor* out = tensor_create(a->dims, a->ndim);
    for (int i = 0; i < out->size; i++) {
        out->data[i] = a->data[i] / (b->data[i] + 1e-8f);
    }
    return out;
}

static void compute_broadcast_dims(Tensor* a, Tensor* b,
                                   int* out_dims, int* out_ndim) {
    int max_ndim = a->ndim > b->ndim ? a->ndim : b->ndim;
    *out_ndim = max_ndim;
    int* ad = (int*)calloc(max_ndim, sizeof(int));
    int* bd = (int*)calloc(max_ndim, sizeof(int));
    for (int i = 0; i < a->ndim; i++) ad[max_ndim - a->ndim + i] = a->dims[i];
    for (int i = 0; i < b->ndim; i++) bd[max_ndim - b->ndim + i] = b->dims[i];
    for (int i = 0; i < max_ndim; i++) {
        if (ad[i] == bd[i]) out_dims[i] = ad[i];
        else if (ad[i] == 1) out_dims[i] = bd[i];
        else if (bd[i] == 1) out_dims[i] = ad[i];
        else out_dims[i] = ad[i] > bd[i] ? ad[i] : bd[i];
    }
    free(ad);
    free(bd);
}

Tensor* tensor_add_broadcast(Tensor* a, Tensor* b) {
    int out_dims[8], out_ndim;
    compute_broadcast_dims(a, b, out_dims, &out_ndim);
    Tensor* out = tensor_create(out_dims, out_ndim);
    int strides_a[8] = {0}, strides_b[8] = {0};
    for (int i = 0; i < out_ndim; i++) {
        int ai = i - (out_ndim - a->ndim);
        int bi = i - (out_ndim - b->ndim);
        strides_a[i] = (ai >= 0 && a->dims[ai] == out_dims[i]) ? a->strides[ai] : 0;
        strides_b[i] = (bi >= 0 && b->dims[bi] == out_dims[i]) ? b->strides[bi] : 0;
    }
    for (int idx = 0; idx < out->size; idx++) {
        int temp = idx;
        int offset_a = 0, offset_b = 0;
        for (int d = out_ndim - 1; d >= 0; d--) {
            int coord = temp % out_dims[d];
            temp /= out_dims[d];
            if (strides_a[d] != 0) {
                int ad = d - (out_ndim - a->ndim);
                offset_a += (coord % a->dims[ad]) * strides_a[d];
            }
            if (strides_b[d] != 0) {
                int bd = d - (out_ndim - b->ndim);
                offset_b += (coord % b->dims[bd]) * strides_b[d];
            }
        }
        out->data[idx] = a->data[offset_a] + b->data[offset_b];
    }
    return out;
}

Tensor* tensor_sub_broadcast(Tensor* a, Tensor* b) {
    int out_dims[8], out_ndim;
    compute_broadcast_dims(a, b, out_dims, &out_ndim);
    Tensor* out = tensor_create(out_dims, out_ndim);
    int strides_a[8] = {0}, strides_b[8] = {0};
    for (int i = 0; i < out_ndim; i++) {
        int ai = i - (out_ndim - a->ndim);
        strides_a[i] = (ai >= 0) ? a->strides[ai] : 0;
        int bi = i - (out_ndim - b->ndim);
        strides_b[i] = (bi >= 0) ? b->strides[bi] : 0;
    }
    for (int idx = 0; idx < out->size; idx++) {
        int temp = idx;
        int offset_a = 0, offset_b = 0;
        for (int d = out_ndim - 1; d >= 0; d--) {
            int coord = temp % out_dims[d];
            temp /= out_dims[d];
            if (strides_a[d] != 0) {
                int ad = d - (out_ndim - a->ndim);
                offset_a += (coord % a->dims[ad]) * strides_a[d];
            }
            if (strides_b[d] != 0) {
                int bd = d - (out_ndim - b->ndim);
                offset_b += (coord % b->dims[bd]) * strides_b[d];
            }
        }
        out->data[idx] = a->data[offset_a] - b->data[offset_b];
    }
    return out;
}

Tensor* tensor_mul_broadcast(Tensor* a, Tensor* b) {
    int out_dims[8], out_ndim;
    compute_broadcast_dims(a, b, out_dims, &out_ndim);
    Tensor* out = tensor_create(out_dims, out_ndim);
    int strides_a[8] = {0}, strides_b[8] = {0};
    for (int i = 0; i < out_ndim; i++) {
        int ai = i - (out_ndim - a->ndim);
        strides_a[i] = (ai >= 0) ? a->strides[ai] : 0;
        int bi = i - (out_ndim - b->ndim);
        strides_b[i] = (bi >= 0) ? b->strides[bi] : 0;
    }
    for (int idx = 0; idx < out->size; idx++) {
        int temp = idx;
        int offset_a = 0, offset_b = 0;
        for (int d = out_ndim - 1; d >= 0; d--) {
            int coord = temp % out_dims[d];
            temp /= out_dims[d];
            if (strides_a[d] != 0) {
                int ad = d - (out_ndim - a->ndim);
                offset_a += (coord % a->dims[ad]) * strides_a[d];
            }
            if (strides_b[d] != 0) {
                int bd = d - (out_ndim - b->ndim);
                offset_b += (coord % b->dims[bd]) * strides_b[d];
            }
        }
        out->data[idx] = a->data[offset_a] * b->data[offset_b];
    }
    return out;
}

Tensor* tensor_div_broadcast(Tensor* a, Tensor* b) {
    int out_dims[8], out_ndim;
    compute_broadcast_dims(a, b, out_dims, &out_ndim);
    Tensor* out = tensor_create(out_dims, out_ndim);
    int strides_a[8] = {0}, strides_b[8] = {0};
    for (int i = 0; i < out_ndim; i++) {
        int ai = i - (out_ndim - a->ndim);
        strides_a[i] = (ai >= 0) ? a->strides[ai] : 0;
        int bi = i - (out_ndim - b->ndim);
        strides_b[i] = (bi >= 0) ? b->strides[bi] : 0;
    }
    for (int idx = 0; idx < out->size; idx++) {
        int temp = idx;
        int offset_a = 0, offset_b = 0;
        for (int d = out_ndim - 1; d >= 0; d--) {
            int coord = temp % out_dims[d];
            temp /= out_dims[d];
            if (strides_a[d] != 0) {
                int ad = d - (out_ndim - a->ndim);
                offset_a += (coord % a->dims[ad]) * strides_a[d];
            }
            if (strides_b[d] != 0) {
                int bd = d - (out_ndim - b->ndim);
                offset_b += (coord % b->dims[bd]) * strides_b[d];
            }
        }
        out->data[idx] = a->data[offset_a] / (b->data[offset_b] + 1e-8f);
    }
    return out;
}

Tensor* tensor_matmul(Tensor* a, Tensor* b) {
    int M, K_a, K_b, N;
    int ndim = a->ndim > b->ndim ? a->ndim : b->ndim;

    if (a->ndim == 1 && b->ndim == 1) {
        float dot = 0;
        for (int i = 0; i < a->size; i++) dot += a->data[i] * b->data[i];
        int od[] = {1};
        Tensor* out = tensor_create(od, 1);
        out->data[0] = dot;
        return out;
    }

    M = (a->ndim == 1) ? 1 : a->dims[0];
    K_a = (a->ndim == 1) ? a->dims[0] : a->dims[a->ndim - 1];
    K_b = (b->ndim == 1) ? b->dims[0] : b->dims[b->ndim - 2];
    N = (b->ndim == 1) ? 1 : b->dims[b->ndim - 1];

    if (K_a != K_b) {
        printf("matmul shape mismatch: last dim of A (%d) != first dim of B (%d)\n", K_a, K_b);
        return NULL;
    }
    int K = K_a;

    if (ndim > 2) {
        int batch_size = 1;
        int* batch_dims = (int*)malloc(sizeof(int) * (ndim - 2));
        for (int i = 0; i < ndim - 2; i++) {
            int ad = (a->ndim > 2 && i < a->ndim - 2) ? a->dims[i] : 1;
            int bd = (b->ndim > 2 && i < b->ndim - 2) ? b->dims[i] : 1;
            batch_dims[i] = ad > bd ? ad : bd;
            batch_size *= batch_dims[i];
        }

        int out_dims_tmp[8];
        for (int i = 0; i < ndim - 2; i++) out_dims_tmp[i] = batch_dims[i];
        if (a->ndim == 1) {
            out_dims_tmp[ndim - 2] = N;
        } else {
            out_dims_tmp[ndim - 2] = M;
            out_dims_tmp[ndim - 1] = N;
        }
        int out_ndim = (a->ndim == 1) ? ndim - 1 : ndim;

        Tensor* out = tensor_create(out_dims_tmp, out_ndim);
        int a_mat_stride = M * K;
        int b_mat_stride = K * N;
        int c_mat_stride = M * N;

        for (int b_idx = 0; b_idx < batch_size; b_idx++) {
            int a_idx = b_idx * a_mat_stride;
            int b_idx2 = b_idx * b_mat_stride;
            int c_idx2 = b_idx * c_mat_stride;

            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    float sum = 0;
                    for (int k = 0; k < K; k++) {
                        sum += a->data[a_idx + i * K + k] * b->data[b_idx2 + k * N + j];
                    }
                    out->data[c_idx2 + i * N + j] = sum;
                }
            }
        }
        free(batch_dims);
        return out;
    }

    int out_dims[] = {M, N};
    Tensor* out = tensor_create(out_dims, 2);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int k = 0; k < K; k++) {
                sum += a->data[i * K + k] * b->data[k * N + j];
            }
            out->data[i * N + j] = sum;
        }
    }
    return out;
}

Tensor* tensor_matmul_gemm(Tensor* a, Tensor* b, Tensor* c) {
    Tensor* ab = tensor_matmul(a, b);
    if (!c) return ab;
    for (int i = 0; i < ab->size; i++) {
        ab->data[i] += c->data[i];
    }
    return ab;
}

Tensor* tensor_transpose(Tensor* a, int dim0, int dim1) {
    Tensor* out = tensor_create(a->dims, a->ndim);
    int tmp = out->dims[dim0];
    out->dims[dim0] = out->dims[dim1];
    out->dims[dim1] = tmp;
    out->strides[out->ndim - 1] = 1;
    for (int i = out->ndim - 2; i >= 0; i--) {
        out->strides[i] = out->strides[i + 1] * out->dims[i + 1];
    }
    for (int idx = 0; idx < a->size; idx++) {
        int temp = idx;
        int indices[8];
        for (int d = a->ndim - 1; d >= 0; d--) {
            indices[d] = temp % a->dims[d];
            temp /= a->dims[d];
        }
        int tmp2 = indices[dim0];
        indices[dim0] = indices[dim1];
        indices[dim1] = tmp2;
        int src_offset = tensor_offset(a, indices);
        out->data[tensor_offset(out, indices)] = a->data[src_offset];
    }
    return out;
}

Tensor* tensor_reshape(Tensor* a, int* new_dims, int new_ndim) {
    int new_size = 1;
    for (int i = 0; i < new_ndim; i++) new_size *= new_dims[i];
    if (new_size != a->size) {
        printf("reshape error: total size mismatch (%d != %d)\n", new_size, a->size);
        return NULL;
    }
    Tensor* out = (Tensor*)malloc(sizeof(Tensor));
    out->ndim = new_ndim;
    out->dims = (int*)malloc(sizeof(int) * new_ndim);
    out->strides = (int*)malloc(sizeof(int) * new_ndim);
    for (int i = 0; i < new_ndim; i++) out->dims[i] = new_dims[i];
    out->strides[new_ndim - 1] = 1;
    for (int i = new_ndim - 2; i >= 0; i--) {
        out->strides[i] = out->strides[i + 1] * out->dims[i + 1];
    }
    out->size = new_size;
    out->data = (float*)malloc(sizeof(float) * new_size);
    memcpy(out->data, a->data, sizeof(float) * new_size);
    out->owns_data = true;
    return out;
}

Tensor* tensor_slice(Tensor* a, int* starts, int* ends) {
    int* slice_dims = (int*)malloc(sizeof(int) * a->ndim);
    int total = 1;
    for (int i = 0; i < a->ndim; i++) {
        slice_dims[i] = ends[i] - starts[i];
        total *= slice_dims[i];
    }
    Tensor* out = tensor_create(slice_dims, a->ndim);
    free(slice_dims);

    int* indices = (int*)calloc(a->ndim, sizeof(int));
    int* max_idx = (int*)malloc(sizeof(int) * a->ndim);
    for (int i = 0; i < a->ndim; i++) max_idx[i] = ends[i];

    int out_idx = 0;
    for (int i = 0; i < a->ndim; i++) indices[i] = starts[i];
    while (out_idx < total) {
        out->data[out_idx++] = a->data[tensor_offset(a, indices)];
        indices[a->ndim - 1]++;
        for (int d = a->ndim - 1; d >= 0; d--) {
            if (indices[d] >= max_idx[d]) {
                if (d > 0) {
                    indices[d] = starts[d];
                    indices[d - 1]++;
                }
            }
        }
        if (indices[0] >= max_idx[0]) break;
    }
    free(indices);
    free(max_idx);
    return out;
}

Tensor* tensor_concatenate(Tensor** tensors, int num, int axis) {
    int ndim = tensors[0]->ndim;
    int total_axis = 0;
    for (int i = 0; i < num; i++) total_axis += tensors[i]->dims[axis];

    int* out_dims = (int*)malloc(sizeof(int) * ndim);
    for (int i = 0; i < ndim; i++) {
        out_dims[i] = (i == axis) ? total_axis : tensors[0]->dims[i];
    }
    Tensor* out = tensor_create(out_dims, ndim);
    free(out_dims);

    int offset = 0;
    for (int t = 0; t < num; t++) {
        Tensor* ct = tensors[t];
        int block_size = 1;
        for (int i = axis + 1; i < ndim; i++) block_size *= ct->dims[i];
        int outer_size = 1;
        for (int i = 0; i < axis; i++) outer_size *= ct->dims[i];
        int axis_size = ct->dims[axis];

        for (int o = 0; o < outer_size; o++) {
            int src_base = o * axis_size * block_size;
            int dst_base = o * total_axis * block_size + offset * block_size;
            memcpy(out->data + dst_base, ct->data + src_base,
                   sizeof(float) * axis_size * block_size);
        }
        offset += axis_size;
    }
    return out;
}

Tensor* tensor_softmax(Tensor* a, int axis) {
    Tensor* out = tensor_create(a->dims, a->ndim);
    memcpy(out->data, a->data, sizeof(float) * a->size);

    int outer_size = 1;
    for (int i = 0; i < axis; i++) outer_size *= a->dims[i];
    int axis_size = a->dims[axis];
    int inner_size = 1;
    for (int i = axis + 1; i < a->ndim; i++) inner_size *= a->dims[i];
    int stride = a->strides[axis];

    for (int o = 0; o < outer_size; o++) {
        for (int k = 0; k < inner_size; k++) {
            int base = o * axis_size * stride + k;
            float max_val = out->data[base];
            for (int i = 1; i < axis_size; i++) {
                if (out->data[base + i * stride] > max_val)
                    max_val = out->data[base + i * stride];
            }
            float sum = 0;
            for (int i = 0; i < axis_size; i++) {
                out->data[base + i * stride] = expf(out->data[base + i * stride] - max_val);
                sum += out->data[base + i * stride];
            }
            for (int i = 0; i < axis_size; i++) {
                out->data[base + i * stride] /= (sum + 1e-12f);
            }
        }
    }
    return out;
}

Tensor* tensor_sum(Tensor* a, int axis, bool keepdim) {
    if (axis < 0) {
        Tensor* out = tensor_create((int[]){1}, 1);
        out->data[0] = 0;
        for (int i = 0; i < a->size; i++) out->data[0] += a->data[i];
        return out;
    }
    int out_ndim = keepdim ? a->ndim : a->ndim - 1;
    int* out_dims = (int*)malloc(sizeof(int) * out_ndim);
    int di = 0;
    for (int i = 0; i < a->ndim; i++) {
        if (i != axis || keepdim) {
            out_dims[di++] = keepdim && i == axis ? 1 : a->dims[i];
        }
    }
    Tensor* out = tensor_create_zeros(out_dims, out_ndim);
    free(out_dims);

    int outer = 1;
    for (int i = 0; i < axis; i++) outer *= a->dims[i];
    int inner = 1;
    for (int i = axis + 1; i < a->ndim; i++) inner *= a->dims[i];
    int stride = a->strides[axis];

    for (int o = 0; o < outer; o++) {
        for (int k = 0; k < inner; k++) {
            int base_out = o * inner + k;
            float s = 0;
            for (int i = 0; i < a->dims[axis]; i++) {
                s += a->data[o * a->dims[axis] * stride + i * stride + k];
            }
            out->data[base_out] = s;
        }
    }
    return out;
}

Tensor* tensor_mean(Tensor* a, int axis, bool keepdim) {
    Tensor* s = tensor_sum(a, axis, keepdim);
    if (axis < 0) {
        s->data[0] /= (float)a->size;
    } else {
        float div = (float)a->dims[axis];
        for (int i = 0; i < s->size; i++) s->data[i] /= div;
    }
    return s;
}

Tensor* tensor_max(Tensor* a, int axis, bool keepdim) {
    if (axis < 0) {
        Tensor* out = tensor_create((int[]){1}, 1);
        out->data[0] = a->data[0];
        for (int i = 1; i < a->size; i++) {
            if (a->data[i] > out->data[0]) out->data[0] = a->data[i];
        }
        return out;
    }
    int out_ndim = keepdim ? a->ndim : a->ndim - 1;
    int* out_dims = (int*)malloc(sizeof(int) * out_ndim);
    int di = 0;
    for (int i = 0; i < a->ndim; i++) {
        if (i != axis || keepdim) out_dims[di++] = keepdim && i == axis ? 1 : a->dims[i];
    }
    Tensor* out = tensor_create(out_dims, out_ndim);
    free(out_dims);

    int outer = 1;
    for (int i = 0; i < axis; i++) outer *= a->dims[i];
    int inner = 1;
    for (int i = axis + 1; i < a->ndim; i++) inner *= a->dims[i];
    int stride = a->strides[axis];

    for (int o = 0; o < outer; o++) {
        for (int k = 0; k < inner; k++) {
            int base_out = o * inner + k;
            float m = a->data[o * a->dims[axis] * stride + k];
            for (int i = 1; i < a->dims[axis]; i++) {
                float v = a->data[o * a->dims[axis] * stride + i * stride + k];
                if (v > m) m = v;
            }
            out->data[base_out] = m;
        }
    }
    return out;
}

void tensor_add_(Tensor* a, Tensor* b) {
    for (int i = 0; i < a->size; i++) a->data[i] += b->data[i];
}

void tensor_sub_(Tensor* a, Tensor* b) {
    for (int i = 0; i < a->size; i++) a->data[i] -= b->data[i];
}

void tensor_mul_(Tensor* a, Tensor* b) {
    for (int i = 0; i < a->size; i++) a->data[i] *= b->data[i];
}

void tensor_div_(Tensor* a, Tensor* b) {
    for (int i = 0; i < a->size; i++) a->data[i] /= (b->data[i] + 1e-8f);
}

void tensor_fill_(Tensor* a, float val) {
    for (int i = 0; i < a->size; i++) a->data[i] = val;
}

void tensor_scale_(Tensor* a, float val) {
    for (int i = 0; i < a->size; i++) a->data[i] *= val;
}

void tensor_clip_(Tensor* a, float min_val, float max_val) {
    for (int i = 0; i < a->size; i++) {
        if (a->data[i] < min_val) a->data[i] = min_val;
        if (a->data[i] > max_val) a->data[i] = max_val;
    }
}

Tensor* tensor_relu(Tensor* a) {
    Tensor* out = tensor_create(a->dims, a->ndim);
    for (int i = 0; i < out->size; i++) {
        out->data[i] = a->data[i] > 0 ? a->data[i] : 0;
    }
    return out;
}

Tensor* tensor_sigmoid(Tensor* a) {
    Tensor* out = tensor_create(a->dims, a->ndim);
    for (int i = 0; i < out->size; i++) {
        out->data[i] = 1.0f / (1.0f + expf(-a->data[i]));
    }
    return out;
}

Tensor* tensor_tanh(Tensor* a) {
    Tensor* out = tensor_create(a->dims, a->ndim);
    for (int i = 0; i < out->size; i++) {
        out->data[i] = tanhf(a->data[i]);
    }
    return out;
}

Tensor* tensor_copy(Tensor* a) {
    Tensor* out = tensor_create(a->dims, a->ndim);
    memcpy(out->data, a->data, sizeof(float) * a->size);
    return out;
}

float tensor_sum_all(Tensor* a) {
    float s = 0;
    for (int i = 0; i < a->size; i++) s += a->data[i];
    return s;
}

void tensor_print(Tensor* t, const char* name) {
    printf("Tensor %s: shape=(", name);
    for (int i = 0; i < t->ndim; i++) {
        printf("%d%s", t->dims[i], i < t->ndim - 1 ? ", " : "");
    }
    printf(") data=[");
    int print_n = t->size < 20 ? t->size : 10;
    for (int i = 0; i < print_n; i++) {
        printf("%.4f%s", t->data[i], i < print_n - 1 ? ", " : "");
    }
    if (t->size > 20) printf(", ... (%d more)", t->size - 20);
    printf("]\n");
}

bool broadcastable(int* dims_a, int ndim_a, int* dims_b, int ndim_b) {
    int max_ndim = ndim_a > ndim_b ? ndim_a : ndim_b;
    for (int i = 0; i < max_ndim; i++) {
        int da = 1, db = 1;
        if (i < ndim_a) da = dims_a[ndim_a - 1 - i];
        if (i < ndim_b) db = dims_b[ndim_b - 1 - i];
        if (da != db && da != 1 && db != 1) return false;
    }
    return true;
}

void broadcast_dims(int* dims_a, int ndim_a, int* dims_b, int ndim_b,
                    int* out_dims, int* out_ndim) {
    *out_ndim = ndim_a > ndim_b ? ndim_a : ndim_b;
    for (int i = 0; i < *out_ndim; i++) {
        int da = 1, db = 1;
        if (i < ndim_a) da = dims_a[ndim_a - 1 - i];
        if (i < ndim_b) db = dims_b[ndim_b - 1 - i];
        out_dims[*out_ndim - 1 - i] = da > db ? da : db;
    }
}

void gemm_naive(bool trans_a, bool trans_b,
                int M, int N, int K,
                float alpha, float* A, int lda,
                float* B, int ldb,
                float beta, float* C, int ldc) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int k = 0; k < K; k++) {
                float a_val = trans_a ? A[k * lda + i] : A[i * lda + k];
                float b_val = trans_b ? B[j * ldb + k] : B[k * ldb + j];
                sum += a_val * b_val;
            }
            C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
        }
    }
}
