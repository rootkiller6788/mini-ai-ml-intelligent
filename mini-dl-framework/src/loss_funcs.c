#include "loss_funcs.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>

float mse_loss_value(Tensor* pred, Tensor* target, LossReduction reduction) {
    float total = 0;
    int n = pred->size;
    for (int i = 0; i < n; i++) {
        float diff = pred->data[i] - target->data[i];
        total += diff * diff;
    }
    if (reduction == REDUCE_MEAN) total /= (float)n;
    return total;
}

Tensor* mse_loss_forward(Tensor* pred, Tensor* target, LossReduction reduction) {
    float loss_val = mse_loss_value(pred, target, reduction);
    Tensor* out = tensor_create((int[]){1}, 1);
    out->data[0] = loss_val;
    return out;
}

Tensor* mse_loss_backward(Tensor* pred, Tensor* target,
                          LossReduction reduction) {
    Tensor* grad = tensor_create(pred->dims, pred->ndim);
    int n = pred->size;
    float scale = (reduction == REDUCE_MEAN) ? 2.0f / (float)n : 2.0f;
    for (int i = 0; i < n; i++) {
        grad->data[i] = scale * (pred->data[i] - target->data[i]);
    }
    return grad;
}

float bce_with_logits_value(Tensor* logits, Tensor* target,
                            LossReduction reduction) {
    float total = 0;
    int n = logits->size;
    for (int i = 0; i < n; i++) {
        float x = logits->data[i];
        float t = target->data[i];
        float eps = 1e-12f;
        float p = 1.0f / (1.0f + expf(-x));
        p = fmaxf(fminf(p, 1.0f - eps), eps);
        total += -(t * logf(p) + (1.0f - t) * logf(1.0f - p));
    }
    if (reduction == REDUCE_MEAN) total /= (float)n;
    return total;
}

Tensor* bce_with_logits_forward(Tensor* logits, Tensor* target,
                                LossReduction reduction) {
    float loss_val = bce_with_logits_value(logits, target, reduction);
    Tensor* out = tensor_create((int[]){1}, 1);
    out->data[0] = loss_val;
    return out;
}

Tensor* bce_with_logits_backward(Tensor* logits, Tensor* target,
                                 LossReduction reduction) {
    Tensor* grad = tensor_create(logits->dims, logits->ndim);
    int n = logits->size;
    float scale = (reduction == REDUCE_MEAN) ? 1.0f / (float)n : 1.0f;
    for (int i = 0; i < n; i++) {
        float p = 1.0f / (1.0f + expf(-logits->data[i]));
        grad->data[i] = scale * (p - target->data[i]);
    }
    return grad;
}

Tensor* log_softmax(Tensor* logits, int axis) {
    Tensor* sm = tensor_softmax(logits, axis);
    Tensor* out = tensor_create(sm->dims, sm->ndim);
    for (int i = 0; i < sm->size; i++) {
        out->data[i] = logf(sm->data[i] + 1e-12f);
    }
    tensor_free(sm);
    return out;
}

float cross_entropy_loss_value(Tensor* logits, Tensor* target,
                               LossReduction reduction,
                               float label_smoothing) {
    int N = logits->dims[0];
    int C = logits->dims[1];
    float total = 0;

    Tensor* sm = tensor_softmax(logits, 1);
    for (int n = 0; n < N; n++) {
        float loss = 0;
        for (int c = 0; c < C; c++) {
            float smooth_target = label_smoothing / (float)C;
            int label = 0;
            if (target->ndim == 1) {
                label = (int)target->data[n];
                smooth_target += (c == label) ? (1.0f - label_smoothing) : 0;
            } else {
                smooth_target += target->data[n * C + c] * (1.0f - label_smoothing);
            }
            loss -= smooth_target * logf(sm->data[n * C + c] + 1e-12f);
        }
        total += loss;
    }
    tensor_free(sm);
    if (reduction == REDUCE_MEAN) total /= (float)N;
    return total;
}

Tensor* cross_entropy_loss_forward(Tensor* logits, Tensor* target,
                                   LossReduction reduction,
                                   float label_smoothing) {
    float loss_val = cross_entropy_loss_value(logits, target, reduction,
                                              label_smoothing);
    Tensor* out = tensor_create((int[]){1}, 1);
    out->data[0] = loss_val;
    return out;
}

Tensor* cross_entropy_loss_backward(Tensor* logits, Tensor* target,
                                    LossReduction reduction,
                                    float label_smoothing) {
    int N = logits->dims[0];
    int C = logits->dims[1];
    Tensor* grad = tensor_create(logits->dims, logits->ndim);
    Tensor* sm = tensor_softmax(logits, 1);

    float scale = (reduction == REDUCE_MEAN) ? 1.0f / (float)N : 1.0f;
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            float smooth_target = label_smoothing / (float)C;
            if (target->ndim == 1) {
                int label = (int)target->data[n];
                smooth_target += (c == label) ? (1.0f - label_smoothing) : 0;
            } else {
                smooth_target += target->data[n * C + c] * (1.0f - label_smoothing);
            }
            grad->data[n * C + c] = scale * (sm->data[n * C + c] - smooth_target);
        }
    }
    tensor_free(sm);
    return grad;
}

float focal_loss_value(Tensor* logits, Tensor* target,
                       float alpha, float gamma, LossReduction reduction) {
    int N = logits->dims[0];
    int C = logits->dims[1];
    float total = 0;
    Tensor* sm = tensor_softmax(logits, 1);

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            float p = sm->data[n * C + c];
            float p_t = p > 0.999f ? 0.999f : (p < 1e-7f ? 1e-7f : p);
            float t = 0;
            if (target->ndim == 1) {
                t = ((int)target->data[n] == c) ? 1.0f : 0.0f;
            } else {
                t = target->data[n * C + c];
            }
            float a_t = (t > 0.5f) ? alpha : (1.0f - alpha);
            float ce = -(t * logf(p_t) + (1.0f - t) * logf(1.0f - p_t));
            total += a_t * powf(1.0f - p_t, gamma) * ce;
        }
    }
    tensor_free(sm);
    if (reduction == REDUCE_MEAN) total /= (float)N;
    return total;
}

Tensor* focal_loss_forward(Tensor* logits, Tensor* target,
                           float alpha, float gamma, LossReduction reduction) {
    float loss_val = focal_loss_value(logits, target, alpha, gamma, reduction);
    Tensor* out = tensor_create((int[]){1}, 1);
    out->data[0] = loss_val;
    return out;
}

Tensor* focal_loss_backward(Tensor* logits, Tensor* target,
                            float alpha, float gamma, LossReduction reduction) {
    int N = logits->dims[0];
    int C = logits->dims[1];
    Tensor* grad = tensor_create(logits->dims, logits->ndim);
    Tensor* sm = tensor_softmax(logits, 1);

    float scale = (reduction == REDUCE_MEAN) ? 1.0f / (float)N : 1.0f;
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            float p = sm->data[n * C + c];
            float p_clip = p > 0.999f ? 0.999f : (p < 1e-7f ? 1e-7f : p);
            float t = (target->ndim == 1) ? (((int)target->data[n] == c) ? 1.0f : 0.0f) : target->data[n * C + c];
            float a_t = (t > 0.5f) ? alpha : (1.0f - alpha);
            float modulating = powf(1.0f - p_clip, gamma);
            float grad_modulating = gamma * powf(1.0f - p_clip, gamma - 1.0f) * (-1.0f);
            float ce_grad = p_clip - t;
            float total_grad = a_t * (modulating * ce_grad + grad_modulating * (t * logf(p_clip + 1e-12f) + (1.0f - t) * logf(1.0f - p_clip + 1e-12f)) * p_clip * (1.0f - p_clip));
            grad->data[n * C + c] = scale * total_grad;
        }
    }
    tensor_free(sm);
    return grad;
}

float l1_loss_value(Tensor* pred, Tensor* target, LossReduction reduction) {
    float total = 0;
    int n = pred->size;
    for (int i = 0; i < n; i++) {
        total += fabsf(pred->data[i] - target->data[i]);
    }
    if (reduction == REDUCE_MEAN) total /= (float)n;
    return total;
}

Tensor* l1_loss_forward(Tensor* pred, Tensor* target, LossReduction reduction) {
    float loss_val = l1_loss_value(pred, target, reduction);
    Tensor* out = tensor_create((int[]){1}, 1);
    out->data[0] = loss_val;
    return out;
}

Tensor* l1_loss_backward(Tensor* pred, Tensor* target, LossReduction reduction) {
    Tensor* grad = tensor_create(pred->dims, pred->ndim);
    int n = pred->size;
    float scale = (reduction == REDUCE_MEAN) ? 1.0f / (float)n : 1.0f;
    for (int i = 0; i < n; i++) {
        float diff = pred->data[i] - target->data[i];
        grad->data[i] = scale * (diff > 0 ? 1.0f : (diff < 0 ? -1.0f : 0.0f));
    }
    return grad;
}

float huber_loss_value(Tensor* pred, Tensor* target, float delta,
                      LossReduction reduction) {
    float total = 0;
    int n = pred->size;
    for (int i = 0; i < n; i++) {
        float diff = fabsf(pred->data[i] - target->data[i]);
        if (diff <= delta) {
            total += 0.5f * diff * diff;
        } else {
            total += delta * (diff - 0.5f * delta);
        }
    }
    if (reduction == REDUCE_MEAN) total /= (float)n;
    return total;
}

Tensor* huber_loss_forward(Tensor* pred, Tensor* target, float delta,
                           LossReduction reduction) {
    float loss_val = huber_loss_value(pred, target, delta, reduction);
    Tensor* out = tensor_create((int[]){1}, 1);
    out->data[0] = loss_val;
    return out;
}

Tensor* huber_loss_backward(Tensor* pred, Tensor* target, float delta,
                            LossReduction reduction) {
    Tensor* grad = tensor_create(pred->dims, pred->ndim);
    int n = pred->size;
    float scale = (reduction == REDUCE_MEAN) ? 1.0f / (float)n : 1.0f;
    for (int i = 0; i < n; i++) {
        float diff = pred->data[i] - target->data[i];
        float abs_diff = fabsf(diff);
        if (abs_diff <= delta) {
            grad->data[i] = scale * diff;
        } else {
            grad->data[i] = scale * delta * (diff > 0 ? 1.0f : -1.0f);
        }
    }
    return grad;
}
