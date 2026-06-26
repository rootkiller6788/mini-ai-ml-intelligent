#include "optimizers.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>

SGD* sgd_create(float lr, float momentum, float weight_decay, bool nesterov) {
    SGD* opt = (SGD*)malloc(sizeof(SGD));
    opt->lr = lr;
    opt->momentum = momentum;
    opt->weight_decay = weight_decay;
    opt->nesterov = nesterov;
    opt->t = 0;
    opt->velocity = NULL;
    opt->num_params = 0;
    return opt;
}

void sgd_set_params(SGD* opt, Tensor** params, int num_params) {
    opt->num_params = num_params;
    opt->velocity = (Tensor**)malloc(sizeof(Tensor*) * num_params);
    for (int i = 0; i < num_params; i++) {
        opt->velocity[i] = tensor_create_zeros(params[i]->dims, params[i]->ndim);
    }
}

void sgd_step(SGD* opt) {
    opt->t++;
}

void sgd_zero_grad(SGD* opt) {
    (void)opt;
}

void sgd_free(SGD* opt) {
    if (!opt) return;
    if (opt->velocity) {
        for (int i = 0; i < opt->num_params; i++) {
            tensor_free(opt->velocity[i]);
        }
        free(opt->velocity);
    }
    free(opt);
}

Adam* adam_create(float lr, float beta1, float beta2, float eps,
                  float weight_decay) {
    Adam* opt = (Adam*)malloc(sizeof(Adam));
    opt->lr = lr;
    opt->beta1 = beta1;
    opt->beta2 = beta2;
    opt->eps = eps;
    opt->weight_decay = weight_decay;
    opt->t = 0;
    opt->m = NULL;
    opt->v = NULL;
    opt->num_params = 0;
    return opt;
}

void adam_set_params(Adam* opt, Tensor** params, int num_params) {
    opt->num_params = num_params;
    opt->m = (Tensor**)malloc(sizeof(Tensor*) * num_params);
    opt->v = (Tensor**)malloc(sizeof(Tensor*) * num_params);
    for (int i = 0; i < num_params; i++) {
        opt->m[i] = tensor_create_zeros(params[i]->dims, params[i]->ndim);
        opt->v[i] = tensor_create_zeros(params[i]->dims, params[i]->ndim);
    }
}

void adam_step(Adam* opt) {
    opt->t++;
}

void adam_zero_grad(Adam* opt) {
    (void)opt;
}

void adam_free(Adam* opt) {
    if (!opt) return;
    if (opt->m) {
        for (int i = 0; i < opt->num_params; i++) tensor_free(opt->m[i]);
        free(opt->m);
    }
    if (opt->v) {
        for (int i = 0; i < opt->num_params; i++) tensor_free(opt->v[i]);
        free(opt->v);
    }
    free(opt);
}

AdamW* adamw_create(float lr, float beta1, float beta2, float eps,
                    float weight_decay) {
    AdamW* opt = (AdamW*)malloc(sizeof(AdamW));
    opt->lr = lr;
    opt->beta1 = beta1;
    opt->beta2 = beta2;
    opt->eps = eps;
    opt->weight_decay = weight_decay;
    opt->t = 0;
    opt->m = NULL;
    opt->v = NULL;
    opt->num_params = 0;
    return opt;
}

void adamw_set_params(AdamW* opt, Tensor** params, int num_params) {
    opt->num_params = num_params;
    opt->m = (Tensor**)malloc(sizeof(Tensor*) * num_params);
    opt->v = (Tensor**)malloc(sizeof(Tensor*) * num_params);
    for (int i = 0; i < num_params; i++) {
        opt->m[i] = tensor_create_zeros(params[i]->dims, params[i]->ndim);
        opt->v[i] = tensor_create_zeros(params[i]->dims, params[i]->ndim);
    }
}

void adamw_step(AdamW* opt) {
    opt->t++;
}

void adamw_zero_grad(AdamW* opt) {
    (void)opt;
}

void adamw_free(AdamW* opt) {
    if (!opt) return;
    if (opt->m) {
        for (int i = 0; i < opt->num_params; i++) tensor_free(opt->m[i]);
        free(opt->m);
    }
    if (opt->v) {
        for (int i = 0; i < opt->num_params; i++) tensor_free(opt->v[i]);
        free(opt->v);
    }
    free(opt);
}

LRScheduler* lr_scheduler_create(LRSchedulerType type, float initial_lr,
                                 int step_size, float gamma,
                                 float T_max, float eta_min,
                                 int warmup_steps) {
    LRScheduler* s = (LRScheduler*)malloc(sizeof(LRScheduler));
    s->type = type;
    s->initial_lr = initial_lr;
    s->current_lr = initial_lr;
    s->step_size = step_size;
    s->gamma = gamma;
    s->T_max = T_max;
    s->eta_min = eta_min;
    s->warmup_steps = warmup_steps;
    s->current_step = 0;
    return s;
}

float lr_scheduler_get_lr(LRScheduler* sched) {
    switch (sched->type) {
    case LR_STEP:
        sched->current_lr = sched->initial_lr * powf(sched->gamma,
            floorf((float)sched->current_step / (float)sched->step_size));
        break;
    case LR_COSINE: {
        float progress = (float)sched->current_step / (float)sched->T_max;
        if (progress > 1.0f) progress = 1.0f;
        float cosine = cosf(progress * 3.14159265f);
        sched->current_lr = sched->eta_min + 0.5f * (sched->initial_lr - sched->eta_min) * (1.0f + cosine);
        break;
    }
    case LR_LINEAR_WARMUP:
        if (sched->current_step < sched->warmup_steps) {
            sched->current_lr = sched->initial_lr * ((float)sched->current_step / (float)sched->warmup_steps);
        } else {
            sched->current_lr = sched->initial_lr;
        }
        break;
    default:
        break;
    }
    return sched->current_lr;
}

void lr_scheduler_step(LRScheduler* sched) {
    sched->current_step++;
    lr_scheduler_get_lr(sched);
}

void lr_scheduler_free(LRScheduler* sched) {
    free(sched);
}

void gradient_clip_norm(Tensor** params, int num_params, float max_norm) {
    float total_norm = 0;
    for (int i = 0; i < num_params; i++) {
        for (int j = 0; j < params[i]->size; j++) {
            total_norm += params[i]->data[j] * params[i]->data[j];
        }
    }
    total_norm = sqrtf(total_norm);
    if (total_norm > max_norm) {
        float scale = max_norm / (total_norm + 1e-12f);
        for (int i = 0; i < num_params; i++) {
            for (int j = 0; j < params[i]->size; j++) {
                params[i]->data[j] *= scale;
            }
        }
    }
}

OptimizerWrapper* opt_wrapper_create(void* optimizer, LRScheduler* scheduler,
                                     void (*step)(void*),
                                     void (*zero_grad)(void*)) {
    OptimizerWrapper* w = (OptimizerWrapper*)malloc(sizeof(OptimizerWrapper));
    w->optimizer = optimizer;
    w->scheduler = scheduler;
    w->step_fn = step;
    w->zero_grad_fn = zero_grad;
    w->current_lr = lr_scheduler_get_lr(scheduler);
    return w;
}

void opt_wrapper_step(OptimizerWrapper* w) {
    lr_scheduler_step(w->scheduler);
    w->current_lr = lr_scheduler_get_lr(w->scheduler);
    w->step_fn(w->optimizer);
}

void opt_wrapper_free(OptimizerWrapper* w) {
    if (w) free(w);
}
