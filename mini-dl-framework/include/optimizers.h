#ifndef OPTIMIZERS_H
#define OPTIMIZERS_H

#include "nn_layers.h"
#include <stdbool.h>

typedef struct {
    float lr;
    float momentum;
    float weight_decay;
    bool nesterov;
    int t;
    Tensor** velocity;
    int num_params;
} SGD;

SGD* sgd_create(float lr, float momentum, float weight_decay, bool nesterov);
void sgd_set_params(SGD* opt, Tensor** params, int num_params);
void sgd_step(SGD* opt);
void sgd_zero_grad(SGD* opt);
void sgd_free(SGD* opt);

typedef struct {
    float lr;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
    int t;
    Tensor** m;
    Tensor** v;
    int num_params;
} Adam;

Adam* adam_create(float lr, float beta1, float beta2, float eps,
                  float weight_decay);
void adam_set_params(Adam* opt, Tensor** params, int num_params);
void adam_step(Adam* opt);
void adam_zero_grad(Adam* opt);
void adam_free(Adam* opt);

typedef struct {
    float lr;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
    int t;
    Tensor** m;
    Tensor** v;
    int num_params;
} AdamW;

AdamW* adamw_create(float lr, float beta1, float beta2, float eps,
                    float weight_decay);
void adamw_set_params(AdamW* opt, Tensor** params, int num_params);
void adamw_step(AdamW* opt);
void adamw_zero_grad(AdamW* opt);
void adamw_free(AdamW* opt);

typedef enum {
    LR_STEP,
    LR_COSINE,
    LR_LINEAR_WARMUP
} LRSchedulerType;

typedef struct {
    LRSchedulerType type;
    float initial_lr;
    float current_lr;
    int step_size;
    float gamma;
    float T_max;
    float eta_min;
    int warmup_steps;
    int current_step;
} LRScheduler;

LRScheduler* lr_scheduler_create(LRSchedulerType type, float initial_lr,
                                 int step_size, float gamma,
                                 float T_max, float eta_min,
                                 int warmup_steps);
float lr_scheduler_get_lr(LRScheduler* sched);
void lr_scheduler_step(LRScheduler* sched);
void lr_scheduler_free(LRScheduler* sched);

void gradient_clip_norm(Tensor** params, int num_params, float max_norm);

typedef struct {
    void* optimizer;
    LRScheduler* scheduler;
    void (*step_fn)(void*);
    void (*zero_grad_fn)(void*);
    float current_lr;
} OptimizerWrapper;

OptimizerWrapper* opt_wrapper_create(void* optimizer, LRScheduler* scheduler,
                                     void (*step)(void*),
                                     void (*zero_grad)(void*));
void opt_wrapper_step(OptimizerWrapper* w);
void opt_wrapper_free(OptimizerWrapper* w);

#endif
