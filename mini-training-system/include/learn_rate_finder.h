#ifndef LEARN_RATE_FINDER_H
#define LEARN_RATE_FINDER_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Learning Rate Range Test (Smith, 2015)
 *
 * Theorem: There exists an optimal learning rate that yields the steepest
 *   descent in loss. The LR Range Test identifies this by training with
 *   exponentially increasing LR and finding the point of minimum loss.
 *
 * Reference: Leslie N. Smith, "Cyclical Learning Rates for Training Neural
 *   Networks," WACV 2017. Also: Smith & Topin, "Super-Convergence," 2018.
 */

#define LRF_MAX_STEPS 2048
#define LRF_MAX_NAME 256

typedef enum {
    LRF_SCHEDULE_EXPONENTIAL = 0,
    LRF_SCHEDULE_LINEAR,
} lrf_schedule_t;

typedef struct {
    float  suggested_lr;
    float  suggested_lr_min;
    float  suggested_lr_max;
    float  min_loss;
    int    min_loss_step;
    float* lr_history;
    float* loss_history;
    int    num_steps;
    bool   diverged;
    int    divergence_step;
} lrf_result_t;

typedef struct {
    float  start_lr;
    float  end_lr;
    int    num_steps;
    lrf_schedule_t schedule;
    float  smooth_factor;
    float  divergence_threshold;
    bool   stop_on_diverge;
    int    min_steps_before_stop;
} lrf_config_t;

typedef struct {
    lrf_config_t config;
    lrf_result_t result;
    float  current_lr;
    float  smoothed_loss;
    int    current_step;
    bool   finished;
    bool   diverged;
    float  best_loss;
    int    best_step;
} lrf_context_t;

void lrf_init(lrf_context_t* ctx, const lrf_config_t* cfg);
void lrf_free(lrf_context_t* ctx);
float lrf_get_lr(const lrf_context_t* ctx, int step);
bool lrf_report_loss(lrf_context_t* ctx, float loss);
float lrf_suggest_lr(lrf_context_t* ctx);
float lrf_suggest_lr_steepest(lrf_context_t* ctx);
void lrf_plot(lrf_context_t* ctx, const char* filename);
float lrf_binary_search_lr(float start_lr, float end_lr, float tol,
                            float (*eval_fn)(float lr, void* userdata),
                            void* userdata);

#ifdef __cplusplus
}
#endif

#endif
