#ifndef OPTIMIZERS_H
#define OPTIMIZERS_H

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ── L1: Core Definitions ──
 * Optimization algorithms for neural network training.
 * Reference: Sebastian Ruder, "An overview of gradient descent optimization algorithms" (2016)
 */

typedef enum {
    OPT_SGD = 0,
    OPT_MOMENTUM,
    OPT_NESTEROV,
    OPT_ADAM,
    OPT_ADAMW,
    OPT_RMSPROP
} OptimizerType;

/* L1: SGD parameter struct — Kingma & Ba (Adam, 2014) */
typedef struct {
    OptimizerType type;
    float lr;                /* learning rate */
    float beta1;             /* momentum / first-moment decay (Adam) */
    float beta2;             /* second-moment decay (RMSProp/Adam) */
    float epsilon;           /* numerical stability */
    float weight_decay;      /* L2 regularization (AdamW) */
    float *m;                /* first moment buffer */
    float *v;                /* second moment buffer */
    int    param_count;
    int    t;                /* timestep counter */
} Optimizer;

/* ── L1: LR Schedule types ── */
typedef enum {
    LR_CONSTANT = 0,
    LR_STEP,
    LR_EXPONENTIAL,
    LR_COSINE,
    LR_COSINE_RESTART,
    LR_WARMUP_COSINE
} LRScheduleType;

typedef struct {
    LRScheduleType type;
    float base_lr;
    float min_lr;
    int   warmup_steps;
    int   total_steps;
    int   step_size;         /* for step decay */
    float gamma;             /* decay factor */
    int   T_0;               /* cosine restart period */
    float T_mult;            /* restart period multiplier */
} LRSchedule;

/* ── L2: Core API ── */
Optimizer *optimizer_create(int param_count, OptimizerType type, float lr,
                            float beta1, float beta2, float eps, float wd);
void       optimizer_free(Optimizer *o);
void       optimizer_step(Optimizer *o, float *params, const float *grads);

/* ── L5: Learning rate scheduling ── */
LRSchedule *lr_schedule_create(LRScheduleType type, float base_lr, int total_steps);
void        lr_schedule_free(LRSchedule *s);
float       lr_schedule_get(const LRSchedule *s, int step);

/* ── L5: Gradient clipping (Pascanu et al., 2013) ── */
void  grad_clip_by_norm(float *grads, int n, float max_norm);
void  grad_clip_by_value(float *grads, int n, float min_val, float max_val);
float grad_l2_norm(const float *grads, int n);

/* ── L8: LAMB optimizer (You et al., 2019) — layer-wise adaptive large batch ── */
typedef struct {
    float *m, *v;
    float beta1, beta2, epsilon;
    int    param_count;
    int    t;
} LAMBState;

LAMBState *lamb_create(int param_count, float beta1, float beta2, float eps);
void       lamb_free(LAMBState *l);
void       lamb_step(LAMBState *l, float *params, const float *grads, float lr, float wd);

/* ── L8: Learning rate warmup ── */
float warmup_cosine_lr(int step, int warmup, int total, float base_lr, float min_lr);

#endif
