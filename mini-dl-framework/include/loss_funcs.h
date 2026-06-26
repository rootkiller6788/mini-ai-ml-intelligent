#ifndef LOSS_FUNCS_H
#define LOSS_FUNCS_H

#include "tensor_ops.h"
#include <stdbool.h>

typedef enum {
    REDUCE_MEAN,
    REDUCE_SUM
} LossReduction;

float mse_loss_value(Tensor* pred, Tensor* target, LossReduction reduction);
Tensor* mse_loss_forward(Tensor* pred, Tensor* target, LossReduction reduction);
Tensor* mse_loss_backward(Tensor* pred, Tensor* target,
                          LossReduction reduction);

float bce_with_logits_value(Tensor* logits, Tensor* target,
                            LossReduction reduction);
Tensor* bce_with_logits_forward(Tensor* logits, Tensor* target,
                                LossReduction reduction);
Tensor* bce_with_logits_backward(Tensor* logits, Tensor* target,
                                 LossReduction reduction);

Tensor* log_softmax(Tensor* logits, int axis);
float cross_entropy_loss_value(Tensor* logits, Tensor* target,
                               LossReduction reduction,
                               float label_smoothing);
Tensor* cross_entropy_loss_forward(Tensor* logits, Tensor* target,
                                   LossReduction reduction,
                                   float label_smoothing);
Tensor* cross_entropy_loss_backward(Tensor* logits, Tensor* target,
                                    LossReduction reduction,
                                    float label_smoothing);

float focal_loss_value(Tensor* logits, Tensor* target,
                       float alpha, float gamma, LossReduction reduction);
Tensor* focal_loss_forward(Tensor* logits, Tensor* target,
                           float alpha, float gamma, LossReduction reduction);
Tensor* focal_loss_backward(Tensor* logits, Tensor* target,
                            float alpha, float gamma, LossReduction reduction);

float l1_loss_value(Tensor* pred, Tensor* target, LossReduction reduction);
Tensor* l1_loss_forward(Tensor* pred, Tensor* target, LossReduction reduction);
Tensor* l1_loss_backward(Tensor* pred, Tensor* target, LossReduction reduction);

float huber_loss_value(Tensor* pred, Tensor* target, float delta,
                      LossReduction reduction);
Tensor* huber_loss_forward(Tensor* pred, Tensor* target, float delta,
                           LossReduction reduction);
Tensor* huber_loss_backward(Tensor* pred, Tensor* target, float delta,
                            LossReduction reduction);

#endif
