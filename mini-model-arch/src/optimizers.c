#include "optimizers.h"

/*
 * L2: Adam Optimizer (Kingma & Ba, ICLR 2015)
 * L4: O(1/sqrt(T)) regret bound in convex setting
 * L8: AdamW decouples weight decay (Loshchilov & Hutter, 2017)
 */

Optimizer *optimizer_create(int param_count, OptimizerType type, float lr,
                            float beta1, float beta2, float eps, float wd) {
    Optimizer *o = (Optimizer *)malloc(sizeof(Optimizer));
    o->type = type; o->lr = lr; o->epsilon = eps;
    o->beta1 = beta1; o->beta2 = beta2;
    o->weight_decay = wd; o->param_count = param_count; o->t = 0;
    if (type == OPT_SGD) { o->m = NULL; o->v = NULL; }
    else if (type == OPT_MOMENTUM || type == OPT_NESTEROV) {
        o->m = (float *)calloc(param_count, sizeof(float)); o->v = NULL;
    } else {
        o->m = (float *)calloc(param_count, sizeof(float));
        o->v = (float *)calloc(param_count, sizeof(float));
    }
    return o;
}

void optimizer_free(Optimizer *o) {
    if (o->m) free(o->m); if (o->v) free(o->v); free(o);
}

/* L5: SGD with momentum and Nesterov */
static void sgd_step(Optimizer *o, float *params, const float *grads) {
    float lr = o->lr;
    if (o->type == OPT_SGD) {
        for (int i = 0; i < o->param_count; i++)
            params[i] -= lr * grads[i];
    } else if (o->type == OPT_MOMENTUM) {
        float mu = o->beta1;
        for (int i = 0; i < o->param_count; i++) {
            o->m[i] = mu * o->m[i] + lr * grads[i];
            params[i] -= o->m[i];
        }
    } else if (o->type == OPT_NESTEROV) {
        float mu = o->beta1;
        for (int i = 0; i < o->param_count; i++) {
            float prev_v = o->m[i];
            o->m[i] = mu * prev_v + lr * grads[i];
            params[i] -= mu * prev_v + (1.0f + mu) * (o->m[i] - mu * prev_v);
        }
    }
}

/* L5: RMSProp (Tieleman & Hinton, 2012) */
static void rmsprop_step(Optimizer *o, float *params, const float *grads) {
    float lr = o->lr, beta2 = o->beta2, eps = o->epsilon;
    for (int i = 0; i < o->param_count; i++) {
        o->v[i] = beta2 * o->v[i] + (1.0f - beta2) * grads[i] * grads[i];
        params[i] -= lr * grads[i] / (sqrtf(o->v[i]) + eps);
    }
}

/* L5: Adam step with bias correction */
static void adam_step(Optimizer *o, float *params, const float *grads) {
    float lr = o->lr, b1 = o->beta1, b2 = o->beta2, eps = o->epsilon;
    float wd = o->weight_decay;
    o->t++;
    if (o->type == OPT_ADAMW && wd > 0.0f) {
        for (int i = 0; i < o->param_count; i++)
            params[i] -= lr * wd * params[i];
    }
    float bc1 = 1.0f - powf(b1, (float)o->t);
    float bc2 = 1.0f - powf(b2, (float)o->t);
    for (int i = 0; i < o->param_count; i++) {
        o->m[i] = b1 * o->m[i] + (1.0f - b1) * grads[i];
        o->v[i] = b2 * o->v[i] + (1.0f - b2) * grads[i] * grads[i];
        float m_hat = o->m[i] / bc1;
        float v_hat = o->v[i] / bc2;
        params[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
    }
}

void optimizer_step(Optimizer *o, float *params, const float *grads) {
    switch (o->type) {
        case OPT_SGD: case OPT_MOMENTUM: case OPT_NESTEROV:
            sgd_step(o, params, grads); break;
        case OPT_RMSPROP: rmsprop_step(o, params, grads); break;
        case OPT_ADAM: case OPT_ADAMW: adam_step(o, params, grads); break;
        default: break;
    }
}

/* L5: Gradient Clipping (Pascanu et al., ICML 2013) */
float grad_l2_norm(const float *grads, int n) {
    float norm = 0.0f;
    for (int i = 0; i < n; i++) norm += grads[i] * grads[i];
    return sqrtf(norm);
}

void grad_clip_by_norm(float *grads, int n, float max_norm) {
    float norm = grad_l2_norm(grads, n);
    if (norm > max_norm && norm > 1e-8f) {
        float scale = max_norm / norm;
        for (int i = 0; i < n; i++) grads[i] *= scale;
    }
}

void grad_clip_by_value(float *grads, int n, float min_val, float max_val) {
    for (int i = 0; i < n; i++) {
        if (grads[i] < min_val) grads[i] = min_val;
        else if (grads[i] > max_val) grads[i] = max_val;
    }
}

/* L5: Learning Rate Schedules
 * L2: Cosine decay = min_lr + 0.5*(base-min)*(1+cos(pi*t/T))
 */
LRSchedule *lr_schedule_create(LRScheduleType type, float base_lr, int total_steps) {
    LRSchedule *s = (LRSchedule *)malloc(sizeof(LRSchedule));
    s->type = type; s->base_lr = base_lr; s->total_steps = total_steps;
    s->min_lr = base_lr * 0.01f;
    s->warmup_steps = total_steps / 10;
    if (s->warmup_steps < 1) s->warmup_steps = 1;
    s->step_size = total_steps / 3;
    if (s->step_size < 1) s->step_size = 1;
    s->gamma = 0.1f;
    s->T_0 = total_steps / 2;
    if (s->T_0 < 1) s->T_0 = 1;
    s->T_mult = 2.0f;
    return s;
}

void lr_schedule_free(LRSchedule *s) { free(s); }

float lr_schedule_get(const LRSchedule *s, int step) {
    if (step >= s->total_steps) return s->min_lr;
    switch (s->type) {
        case LR_CONSTANT: return s->base_lr;
        case LR_STEP: {
            int periods = step / s->step_size;
            float lr = s->base_lr;
            for (int i = 0; i < periods; i++) lr *= s->gamma;
            return fmaxf(lr, s->min_lr);
        }
        case LR_EXPONENTIAL: {
            float lr = s->base_lr * powf(s->gamma, (float)step);
            return fmaxf(lr, s->min_lr);
        }
        case LR_COSINE: {
            float progress = (float)step / (float)s->total_steps;
            return s->min_lr + 0.5f * (s->base_lr - s->min_lr) *
                   (1.0f + cosf(3.14159265f * progress));
        }
        case LR_COSINE_RESTART: {
            int T_cur = step, T_i = s->T_0;
            float mult = s->T_mult;
            while (T_cur >= T_i) {
                T_cur -= T_i;
                T_i = (int)((float)T_i * mult);
                if (T_i < 1) T_i = 1;
            }
            float progress = (float)T_cur / (float)T_i;
            return s->min_lr + 0.5f * (s->base_lr - s->min_lr) *
                   (1.0f + cosf(3.14159265f * progress));
        }
        case LR_WARMUP_COSINE:
            return warmup_cosine_lr(step, s->warmup_steps, s->total_steps,
                                    s->base_lr, s->min_lr);
        default: return s->base_lr;
    }
}

float warmup_cosine_lr(int step, int warmup, int total, float base_lr, float min_lr) {
    if (step < warmup)
        return base_lr * (float)step / (float)warmup;
    int decay_steps = total - warmup;
    int t = step - warmup;
    float progress = (float)t / (float)decay_steps;
    return min_lr + 0.5f * (base_lr - min_lr) *
           (1.0f + cosf(3.14159265f * progress));
}

/* L8: LAMB (You et al., 2019) - layer-wise adaptive large-batch */
LAMBState *lamb_create(int param_count, float beta1, float beta2, float eps) {
    LAMBState *l = (LAMBState *)malloc(sizeof(LAMBState));
    l->m = (float *)calloc(param_count, sizeof(float));
    l->v = (float *)calloc(param_count, sizeof(float));
    l->beta1 = beta1; l->beta2 = beta2; l->epsilon = eps;
    l->param_count = param_count; l->t = 0;
    return l;
}

void lamb_free(LAMBState *l) { free(l->m); free(l->v); free(l); }

void lamb_step(LAMBState *l, float *params, const float *grads, float lr, float wd) {
    l->t++;
    int n = l->param_count;
    float bc1 = 1.0f - powf(l->beta1, (float)l->t);
    float bc2 = 1.0f - powf(l->beta2, (float)l->t);
    float eps = l->epsilon;
    float *r = (float *)malloc(n * sizeof(float));
    float param_norm = 0.0f, r_norm = 0.0f;
    for (int i = 0; i < n; i++) {
        l->m[i] = l->beta1 * l->m[i] + (1.0f - l->beta1) * grads[i];
        l->v[i] = l->beta2 * l->v[i] + (1.0f - l->beta2) * grads[i] * grads[i];
        float m_hat = l->m[i] / bc1;
        float v_hat = l->v[i] / bc2;
        r[i] = m_hat / (sqrtf(v_hat) + eps) + wd * params[i];
        param_norm += params[i] * params[i];
        r_norm += r[i] * r[i];
    }
    param_norm = sqrtf(param_norm);
    r_norm = sqrtf(r_norm);
    float trust_ratio = (param_norm > 0.0f && r_norm > 0.0f)
                        ? param_norm / r_norm : 1.0f;
    if (trust_ratio > 10.0f) trust_ratio = 10.0f;
    for (int i = 0; i < n; i++)
        params[i] -= lr * trust_ratio * r[i];
    free(r);
}
