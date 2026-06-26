#ifndef HYPERPARAM_TUNE_H
#define HYPERPARAM_TUNE_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define HPT_MAX_PARAMS 64
#define HPT_MAX_NAME 128
#define HPT_MAX_TRIALS 10000
#define HPT_MAX_BRACKETS 16

typedef enum {
    HPT_TYPE_INT = 0,
    HPT_TYPE_FLOAT,
    HPT_TYPE_CATEGORICAL,
    HPT_TYPE_LOG_FLOAT,
    HPT_TYPE_LOG_INT,
} hpt_param_type_t;

typedef enum {
    HPT_SEARCH_GRID = 0,
    HPT_SEARCH_RANDOM,
    HPT_SEARCH_BAYESIAN,
    HPT_SEARCH_TPE,
    HPT_SEARCH_HYPERBAND,
    HPT_SEARCH_CMAES,
} hpt_search_method_t;

typedef enum {
    HPT_MODE_MINIMIZE = 0,
    HPT_MODE_MAXIMIZE,
} hpt_optimize_mode_t;

typedef struct {
    char name[HPT_MAX_NAME];
    hpt_param_type_t type;
    union {
        struct { int min_v; int max_v; int step; } int_range;
        struct { float min_v; float max_v; float step; } float_range;
        struct { char** values; int count; } categorical;
        struct { float min_v; float max_v; } log_float;
        struct { int min_v; int max_v; } log_int;
    } domain;
} hpt_param_t;

typedef struct {
    hpt_param_t params[HPT_MAX_PARAMS];
    int num_params;
    union {
        float fval;
        int   ival;
        int   cat_idx;
    } values[HPT_MAX_PARAMS];
    float objective_value;
    int   trial_id;
    uint64_t timestamp;
    bool  pruned;
} hpt_trial_t;

typedef struct {
    hpt_search_method_t method;
    hpt_optimize_mode_t mode;
    int   num_trials;
    int   random_seed;
    int   early_stop_patience;
    float early_stop_min_delta;
    int   n_initial_points;
    float exploration_ratio;
    int   max_concurrent;
    int   max_failures;
    int   grace_period;
    float reduction_factor;
    int   min_epochs;
    int   max_epochs;
    int   bracket_limit;
    char  study_name[HPT_MAX_NAME];
    char  storage_path[HPT_MAX_NAME];
    bool  pruner_enabled;
    int   pruner_warmup_steps;
} hpt_config_t;

typedef struct {
    float* X;
    float* y;
    float* kernel_mat;
    float* L;
    float* alpha;
    float  noise;
    int    n_observed;
    int    n_dims;
    float  length_scale;
    float  signal_var;
} gp_model_t;

typedef struct {
    int num_brackets;
    int* bracket_milestones;
    int* bracket_num_configs;
    float* bracket_budgets;
    int current_bracket;
} hyperband_state_t;

typedef struct {
    hpt_config_t config;
    hpt_trial_t* trials;
    int num_trials_completed;
    int num_failures;
    int best_trial_idx;
    float best_value;
    int patience_counter;
    bool converged;
    gp_model_t* gp;
    hyperband_state_t hyperband;
    int total_budget;
    int remaining_budget;
} hpt_context_t;

void hpt_init(hpt_context_t* ctx, const hpt_config_t* cfg);
void hpt_free(hpt_context_t* ctx);

void hpt_add_int(hpt_context_t* ctx, const char* name,
                  int min_v, int max_v, int step);
void hpt_add_float(hpt_context_t* ctx, const char* name,
                    float min_v, float max_v, float step);
void hpt_add_log_float(hpt_context_t* ctx, const char* name,
                        float min_v, float max_v);
void hpt_add_log_int(hpt_context_t* ctx, const char* name,
                      int min_v, int max_v);
void hpt_add_categorical(hpt_context_t* ctx, const char* name,
                          const char* values[], int count);

hpt_trial_t hpt_suggest(hpt_context_t* ctx);

int  hpt_get_int(const hpt_trial_t* trial, int param_idx);
float hpt_get_float(const hpt_trial_t* trial, int param_idx);
int   hpt_get_categorical(const hpt_trial_t* trial, int param_idx);
int   hpt_get_param_index(const hpt_context_t* ctx, const char* name);

void hpt_report(hpt_context_t* ctx, int trial_id, float value);
void hpt_report_intermediate(hpt_context_t* ctx, int trial_id,
                              int step, float value);
bool hpt_should_prune(hpt_context_t* ctx, int trial_id, int step,
                       float value);

void gp_model_init(gp_model_t* gp, int n_dims, float noise);
void gp_model_free(gp_model_t* gp);
void gp_model_fit(gp_model_t* gp, const float* X, const float* y, int n);
void gp_model_predict(const gp_model_t* gp, const float* x,
                       float* mean, float* std);
float gp_model_acq_ei(const gp_model_t* gp, const float* x, float y_best);

float tpe_sample(const float* observed, int n_observed,
                  float gamma, int n_dims, int64_t seed);
float tpe_log_ratio(const float* x, const float* good, int n_good,
                     const float* bad, int n_bad);

void hyperband_init(hyperband_state_t* state, int max_epochs,
                     float reduction_factor);
int  hyperband_get_num_configs(const hyperband_state_t* state,
                                int bracket, int total_budget);
int  hyperband_get_budget(const hyperband_state_t* state,
                           int bracket, int rung);
int  hyperband_select_top(const float* values, int n, int top_k,
                           int* indices);

bool hpt_early_stopping_check(float current_best, int patience_counter,
                               int patience, float min_delta);

hpt_trial_t hpt_best_trial(const hpt_context_t* ctx);
void hpt_plot_optimization_history(const hpt_context_t* ctx,
                                    const char* filename);
void hpt_plot_param_importance(const hpt_context_t* ctx,
                                const char* filename);

#ifdef __cplusplus
}
#endif

#endif
