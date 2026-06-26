#include "hyperparam_tune.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <float.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static uint64_t hpt_splitmix64(uint64_t* state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static float hpt_randf(uint64_t* state) {
    return (float)(hpt_splitmix64(state) >> 40) / (float)(1ULL << 24);
}

static int hpt_rand_int(uint64_t* state, int lo, int hi) {
    if (hi <= lo) return lo;
    return lo + (int)(hpt_randf(state) * (float)(hi - lo + 1));
}

static float hpt_rand_range(uint64_t* state, float lo, float hi) {
    return lo + hpt_randf(state) * (hi - lo);
}

void hpt_init(hpt_context_t* ctx, const hpt_config_t* cfg) {
    if (!ctx) return;
    memset(ctx, 0, sizeof(hpt_context_t));
    if (cfg) {
        ctx->config = *cfg;
    } else {
        ctx->config.method = HPT_SEARCH_RANDOM;
        ctx->config.mode = HPT_MODE_MINIMIZE;
        ctx->config.num_trials = 100;
        ctx->config.random_seed = 42;
        ctx->config.early_stop_patience = 10;
        ctx->config.early_stop_min_delta = 1e-4f;
        ctx->config.n_initial_points = 5;
        ctx->config.exploration_ratio = 0.1f;
        ctx->config.grace_period = 1;
        ctx->config.reduction_factor = 3.0f;
        ctx->config.min_epochs = 1;
        ctx->config.max_epochs = 81;
    }
    ctx->trials = (hpt_trial_t*)calloc((size_t)ctx->config.num_trials, sizeof(hpt_trial_t));
    ctx->best_value = (ctx->config.mode == HPT_MODE_MINIMIZE) ? FLT_MAX : -FLT_MAX;
    ctx->best_trial_idx = -1;

    if (ctx->config.method == HPT_SEARCH_BAYESIAN) {
        ctx->gp = (gp_model_t*)malloc(sizeof(gp_model_t));
        if (ctx->gp) gp_model_init(ctx->gp, 1, 1e-4f);
    }
    if (ctx->config.method == HPT_SEARCH_HYPERBAND) {
        hyperband_init(&ctx->hyperband, ctx->config.max_epochs,
                        ctx->config.reduction_factor);
    }
    ctx->total_budget = ctx->config.num_trials;
    ctx->remaining_budget = ctx->config.num_trials;
}

void hpt_free(hpt_context_t* ctx) {
    if (!ctx) return;
    if (ctx->trials) {
        for (int i = 0; i < ctx->config.num_trials; i++) {
            hpt_param_t* params = ctx->trials[i].params;
            for (int j = 0; j < ctx->trials[i].num_params; j++) {
                if (params[j].type == HPT_TYPE_CATEGORICAL && params[j].domain.categorical.values) {
                }
            }
        }
        free(ctx->trials);
    }
    if (ctx->gp) { gp_model_free(ctx->gp); free(ctx->gp); }
    if (ctx->hyperband.bracket_milestones) free(ctx->hyperband.bracket_milestones);
    if (ctx->hyperband.bracket_num_configs) free(ctx->hyperband.bracket_num_configs);
    if (ctx->hyperband.bracket_budgets) free(ctx->hyperband.bracket_budgets);
}

void hpt_add_int(hpt_context_t* ctx, const char* name,
                  int min_v, int max_v, int step) {
    if (!ctx || ctx->config.num_params >= HPT_MAX_PARAMS) return;
    hpt_param_t* p = &ctx->trials ? &ctx->trials[0].params[ctx->config.num_params] : NULL;
    if (!p) return;
    strncpy(p->name, name, HPT_MAX_NAME - 1);
    p->type = HPT_TYPE_INT;
    p->domain.int_range.min_v = min_v;
    p->domain.int_range.max_v = max_v;
    p->domain.int_range.step = step > 0 ? step : 1;
}

void hpt_add_float(hpt_context_t* ctx, const char* name,
                    float min_v, float max_v, float step) {
    if (!ctx || ctx->config.num_params >= HPT_MAX_PARAMS) return;
}

void hpt_add_log_float(hpt_context_t* ctx, const char* name,
                        float min_v, float max_v) {
    if (!ctx || ctx->config.num_params >= HPT_MAX_PARAMS) return;
}

void hpt_add_log_int(hpt_context_t* ctx, const char* name,
                      int min_v, int max_v) {
    if (!ctx || ctx->config.num_params >= HPT_MAX_PARAMS) return;
}

void hpt_add_categorical(hpt_context_t* ctx, const char* name,
                          const char* values[], int count) {
    if (!ctx || ctx->config.num_params >= HPT_MAX_PARAMS) return;
}

static void hpt_generate_trial(const hpt_context_t* ctx, hpt_trial_t* trial,
                                uint64_t* rng) {
    if (!ctx || !trial) return;
    trial->num_params = ctx->config.num_params;
    for (int i = 0; i < ctx->config.num_params; i++) {
        switch (trial->params[i].type) {
        case HPT_TYPE_INT:
            trial->values[i].ival = hpt_rand_int(rng,
                trial->params[i].domain.int_range.min_v,
                trial->params[i].domain.int_range.max_v);
            break;
        case HPT_TYPE_FLOAT:
            trial->values[i].fval = hpt_rand_range(rng,
                trial->params[i].domain.float_range.min_v,
                trial->params[i].domain.float_range.max_v);
            break;
        case HPT_TYPE_LOG_FLOAT:
            trial->values[i].fval = expf(hpt_rand_range(rng,
                logf(trial->params[i].domain.log_float.min_v),
                logf(trial->params[i].domain.log_float.max_v)));
            break;
        case HPT_TYPE_LOG_INT: {
            float l = logf((float)trial->params[i].domain.log_int.min_v);
            float h = logf((float)trial->params[i].domain.log_int.max_v);
            trial->values[i].ival = (int)expf(hpt_rand_range(rng, l, h));
            break;
        }
        case HPT_TYPE_CATEGORICAL:
            trial->values[i].cat_idx = hpt_rand_int(rng, 0,
                trial->params[i].domain.categorical.count - 1);
            break;
        }
    }
}

hpt_trial_t hpt_suggest(hpt_context_t* ctx) {
    hpt_trial_t trial; memset(&trial, 0, sizeof(trial));
    if (!ctx || !ctx->trials) return trial;
    if (ctx->num_trials_completed >= ctx->config.num_trials) return trial;

    uint64_t rng = (uint64_t)(ctx->config.random_seed + ctx->num_trials_completed);

    if (ctx->gp && ctx->num_trials_completed >= (int)(size_t)ctx->config.n_initial_points) {
        hpt_generate_trial(ctx, &trial, &rng);
    } else {
        hpt_generate_trial(ctx, &trial, &rng);
    }

    trial.trial_id = ctx->num_trials_completed;
    memcpy(&ctx->trials[ctx->num_trials_completed], &trial, sizeof(trial));
    return trial;
}

int hpt_get_int(const hpt_trial_t* trial, int param_idx) {
    if (!trial || param_idx < 0 || param_idx >= trial->num_params) return 0;
    return trial->values[param_idx].ival;
}

float hpt_get_float(const hpt_trial_t* trial, int param_idx) {
    if (!trial || param_idx < 0 || param_idx >= trial->num_params) return 0.0f;
    return trial->values[param_idx].fval;
}

int hpt_get_categorical(const hpt_trial_t* trial, int param_idx) {
    if (!trial || param_idx < 0 || param_idx >= trial->num_params) return 0;
    return trial->values[param_idx].cat_idx;
}

int hpt_get_param_index(const hpt_context_t* ctx, const char* name) {
    if (!ctx || !ctx->trials) return -1;
    for (int i = 0; i < HPT_MAX_PARAMS && ctx->trials[0].params[i].name[0]; i++) {
        if (strcmp(ctx->trials[0].params[i].name, name) == 0) return i;
    }
    return -1;
}

void hpt_report(hpt_context_t* ctx, int trial_id, float value) {
    if (!ctx || !ctx->trials || trial_id < 0 || trial_id >= ctx->config.num_trials) return;
    ctx->trials[trial_id].objective_value = value;
    bool better = (ctx->config.mode == HPT_MODE_MINIMIZE) ?
        (value < ctx->best_value) : (value > ctx->best_value);
    if (better) {
        ctx->best_value = value;
        ctx->best_trial_idx = trial_id;
        ctx->patience_counter = 0;
    } else {
        ctx->patience_counter++;
    }
    ctx->num_trials_completed = (ctx->num_trials_completed > trial_id + 1) ?
        ctx->num_trials_completed : trial_id + 1;

    if (ctx->gp && ctx->num_trials_completed >= ctx->config.n_initial_points) {
    }
}

void hpt_report_intermediate(hpt_context_t* ctx, int trial_id,
                              int step, float value) {
    if (!ctx) return;
    (void)trial_id; (void)step; (void)value;
}

bool hpt_should_prune(hpt_context_t* ctx, int trial_id, int step,
                       float value) {
    if (!ctx || !ctx->config.pruner_enabled) return false;
    if (step < ctx->config.pruner_warmup_steps) return false;
    if (ctx->best_trial_idx >= 0 && ctx->num_trials_completed > 1) {
        float best = ctx->best_value;
        float threshold = best * 1.5f;
        if (ctx->config.mode == HPT_MODE_MINIMIZE && value > threshold) {
            ctx->trials[trial_id].pruned = true;
            return true;
        }
    }
    (void)trial_id;
    return false;
}

void gp_model_init(gp_model_t* gp, int n_dims, float noise) {
    if (!gp) return;
    memset(gp, 0, sizeof(gp_model_t));
    gp->n_dims = n_dims;
    gp->noise = noise;
    gp->length_scale = 1.0f;
    gp->signal_var = 1.0f;
}

void gp_model_free(gp_model_t* gp) {
    if (!gp) return;
    free(gp->X); free(gp->y); free(gp->kernel_mat);
    free(gp->L); free(gp->alpha);
    memset(gp, 0, sizeof(gp_model_t));
}

static float gp_kernel_rbf(const gp_model_t* gp, const float* x1, const float* x2) {
    float sq = 0.0f;
    for (int i = 0; i < gp->n_dims; i++) {
        float d = x1[i] - x2[i];
        sq += d * d;
    }
    return gp->signal_var * expf(-0.5f * sq / (gp->length_scale * gp->length_scale));
}

void gp_model_fit(gp_model_t* gp, const float* X, const float* y, int n) {
    if (!gp || !X || !y || n <= 0) return;
    gp->n_observed = n;
    gp->X = (float*)realloc(gp->X, (size_t)(n * gp->n_dims) * sizeof(float));
    gp->y = (float*)realloc(gp->y, (size_t)n * sizeof(float));
    if (!gp->X || !gp->y) return;
    memcpy(gp->X, X, (size_t)(n * gp->n_dims) * sizeof(float));
    memcpy(gp->y, y, (size_t)n * sizeof(float));

    size_t mat_sz = (size_t)(n * n);
    gp->kernel_mat = (float*)realloc(gp->kernel_mat, mat_sz * sizeof(float));
    gp->L = (float*)realloc(gp->L, mat_sz * sizeof(float));
    gp->alpha = (float*)realloc(gp->alpha, (size_t)n * sizeof(float));
    if (!gp->kernel_mat || !gp->L || !gp->alpha) return;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            gp->kernel_mat[i * n + j] = gp_kernel_rbf(gp,
                &X[i * gp->n_dims], &X[j * gp->n_dims]);
            if (i == j) gp->kernel_mat[i * n + j] += gp->noise;
        }
    }
    memcpy(gp->L, gp->kernel_mat, mat_sz * sizeof(float));
    for (int i = 0; i < n; i++) {
        float d = gp->L[i * n + i];
        for (int k = 0; k < i; k++) d -= gp->L[i * n + k] * gp->L[i * n + k];
        if (d <= 0) d = 1e-6f;
        gp->L[i * n + i] = sqrtf(d);
        float inv = 1.0f / gp->L[i * n + i];
        for (int j = i + 1; j < n; j++) {
            float s = 0.0f;
            for (int k = 0; k < i; k++) s += gp->L[i * n + k] * gp->L[j * n + k];
            gp->L[j * n + i] = (gp->L[j * n + i] - s) * inv;
        }
    }
    for (int i = 0; i < n; i++) {
        float s = gp->y[i];
        for (int j = 0; j < i; j++) s -= gp->L[i * n + j] * gp->alpha[j];
        gp->alpha[i] = s / gp->L[i * n + i];
    }
    for (int i = n - 1; i >= 0; i--) {
        float s = gp->alpha[i];
        for (int j = i + 1; j < n; j++) s -= gp->L[j * n + i] * gp->alpha[j];
        gp->alpha[i] = s / gp->L[i * n + i];
    }
}

void gp_model_predict(const gp_model_t* gp, const float* x,
                       float* mean, float* std) {
    if (!gp || !x || !mean || !std || gp->n_observed <= 0) {
        if (mean) *mean = 0.0f;
        if (std) *std = 1.0f;
        return;
    }
    int n = gp->n_observed;
    float* kstar = (float*)malloc((size_t)n * sizeof(float));
    if (!kstar) return;
    for (int i = 0; i < n; i++)
        kstar[i] = gp_kernel_rbf(gp, x, &gp->X[i * gp->n_dims]);

    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += kstar[i] * gp->alpha[i];
    *mean = sum;

    float var = gp->signal_var + gp->noise;
    for (int i = 0; i < n; i++) {
        float s = 0.0f;
        for (int j = 0; j < n; j++) s += gp->L[j * n + i] * kstar[j];
        var -= s * s;
    }
    if (var < 1e-6f) var = 1e-6f;
    *std = sqrtf(var);
    free(kstar);
}

float gp_model_acq_ei(const gp_model_t* gp, const float* x, float y_best) {
    float mean, std;
    gp_model_predict(gp, x, &mean, &std);
    float diff = y_best - mean;
    if (std < 1e-9f) return diff > 0 ? diff : 0.0f;
    float z = diff / std;
    float pdf = expf(-0.5f * z * z) / sqrtf(2.0f * (float)M_PI);
    float cdf = 0.5f * (1.0f + erff(z / sqrtf(2.0f)));
    return diff * cdf + std * pdf;
}

float tpe_sample(const float* observed, int n_observed,
                  float gamma, int n_dims, int64_t seed) {
    (void)observed; (void)n_observed; (void)gamma; (void)n_dims; (void)seed;
    return 0.0f;
}

float tpe_log_ratio(const float* x, const float* good, int n_good,
                     const float* bad, int n_bad) {
    (void)x; (void)good; (void)n_good; (void)bad; (void)n_bad;
    return 0.0f;
}

void hyperband_init(hyperband_state_t* state, int max_epochs,
                     float reduction_factor) {
    if (!state) return;
    memset(state, 0, sizeof(hyperband_state_t));
    int max_total = max_epochs;
    float rf = reduction_factor > 1.0f ? reduction_factor : 3.0f;
    state->num_brackets = (int)(logf((float)max_total) / logf(rf)) + 1;
    if (state->num_brackets > HPT_MAX_BRACKETS) state->num_brackets = HPT_MAX_BRACKETS;

    state->bracket_milestones = (int*)calloc((size_t)state->num_brackets, sizeof(int));
    state->bracket_num_configs = (int*)calloc((size_t)state->num_brackets, sizeof(int));
    state->bracket_budgets = (float*)calloc((size_t)state->num_brackets, sizeof(float));

    for (int s = state->num_brackets - 1; s >= 0; s--) {
        int n = (int)((float)max_total * powf(rf, (float)s) / (float)(s + 1));
        if (n < 1) n = 1;
        state->bracket_num_configs[s] = n;
        state->bracket_milestones[s] = max_total / (int)powf(rf, (float)s);
    }
}

int hyperband_get_num_configs(const hyperband_state_t* state,
                               int bracket, int total_budget) {
    if (!state || bracket < 0 || bracket >= state->num_brackets) return 0;
    (void)total_budget;
    return state->bracket_num_configs[bracket];
}

int hyperband_get_budget(const hyperband_state_t* state,
                          int bracket, int rung) {
    if (!state || bracket < 0 || bracket >= state->num_brackets) return 0;
    return state->bracket_milestones ? state->bracket_milestones[bracket] : 0;
    (void)rung;
}

int hyperband_select_top(const float* values, int n, int top_k,
                          int* indices) {
    if (!values || !indices || n <= 0 || top_k <= 0) return 0;
    if (top_k > n) top_k = n;
    for (int i = 0; i < n; i++) indices[i] = i;
    for (int i = 0; i < top_k; i++) {
        int best = i;
        for (int j = i + 1; j < n; j++) {
            if (values[indices[j]] < values[indices[best]]) best = j;
        }
        int tmp = indices[i]; indices[i] = indices[best]; indices[best] = tmp;
    }
    return top_k;
}

bool hpt_early_stopping_check(float current_best, int patience_counter,
                               int patience, float min_delta) {
    (void)current_best; (void)min_delta;
    return patience_counter >= patience;
}

hpt_trial_t hpt_best_trial(const hpt_context_t* ctx) {
    hpt_trial_t trial; memset(&trial, 0, sizeof(trial));
    if (!ctx || ctx->best_trial_idx < 0) return trial;
    return ctx->trials[ctx->best_trial_idx];
}

void hpt_plot_optimization_history(const hpt_context_t* ctx,
                                    const char* filename) {
    if (!ctx || !filename) return;
    FILE* f = fopen(filename, "w");
    if (!f) return;
    fprintf(f, "# Optimization History\n# step, value\n");
    for (int i = 0; i < ctx->num_trials_completed; i++) {
        fprintf(f, "%d %f\n", i, ctx->trials[i].objective_value);
    }
    fclose(f);
}

void hpt_plot_param_importance(const hpt_context_t* ctx,
                                const char* filename) {
    if (!ctx || !filename) return;
    FILE* f = fopen(filename, "w");
    if (!f) return;
    fprintf(f, "# Parameter Importance\n# param_idx, importance\n");
    fclose(f);
}
