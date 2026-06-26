#include "learn_rate_finder.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

void lrf_init(lrf_context_t* ctx, const lrf_config_t* cfg) {
    if (!ctx) return;
    memset(ctx, 0, sizeof(lrf_context_t));
    if (cfg) {
        ctx->config = *cfg;
    } else {
        ctx->config.start_lr = 1e-7f;
        ctx->config.end_lr = 10.0f;
        ctx->config.num_steps = 200;
        ctx->config.schedule = LRF_SCHEDULE_EXPONENTIAL;
        ctx->config.smooth_factor = 0.98f;
        ctx->config.divergence_threshold = 4.0f;
        ctx->config.stop_on_diverge = true;
        ctx->config.min_steps_before_stop = 10;
    }
    if (ctx->config.num_steps <= 0) ctx->config.num_steps = 100;
    if (ctx->config.num_steps > LRF_MAX_STEPS) ctx->config.num_steps = LRF_MAX_STEPS;
    ctx->current_lr = ctx->config.start_lr;
    ctx->best_loss = 1e38f;
    ctx->best_step = 0;
    ctx->result.lr_history = (float*)calloc((size_t)ctx->config.num_steps, sizeof(float));
    ctx->result.loss_history = (float*)calloc((size_t)ctx->config.num_steps, sizeof(float));
    ctx->result.num_steps = 0;
}

void lrf_free(lrf_context_t* ctx) {
    if (!ctx) return;
    free(ctx->result.lr_history);
    free(ctx->result.loss_history);
    memset(ctx, 0, sizeof(lrf_context_t));
}

float lrf_get_lr(const lrf_context_t* ctx, int step) {
    if (!ctx || step < 0) return 1e-7f;
    int n = ctx->config.num_steps;
    if (n <= 0) return ctx->config.start_lr;

    float start = ctx->config.start_lr;
    float end   = ctx->config.end_lr;

    if (ctx->config.schedule == LRF_SCHEDULE_EXPONENTIAL) {
        /* lr = start * (end/start)^(step/n)
         * This yields equal multiplicative steps, which better explores
         * the log-scale LR space as recommended by Smith (2017). */
        float ratio = end / start;
        float progress = (float)step / (float)n;
        if (progress > 1.0f) progress = 1.0f;
        return start * powf(ratio, progress);
    } else {
        /* Linear schedule */
        float progress = (float)step / (float)n;
        if (progress > 1.0f) progress = 1.0f;
        return start + (end - start) * progress;
    }
}

bool lrf_report_loss(lrf_context_t* ctx, float loss) {
    if (!ctx || ctx->finished) return false;
    int step = ctx->current_step;
    if (step >= ctx->config.num_steps) {
        ctx->finished = true;
        return false;
    }

    float lr = lrf_get_lr(ctx, step);
    ctx->result.lr_history[step] = lr;

    /* Exponentially smoothed loss (like momentum for loss values) */
    if (step == 0) {
        ctx->smoothed_loss = loss;
    } else {
        ctx->smoothed_loss = ctx->config.smooth_factor * ctx->smoothed_loss
                           + (1.0f - ctx->config.smooth_factor) * loss;
    }
    ctx->result.loss_history[step] = ctx->smoothed_loss;
    ctx->result.num_steps = step + 1;

    /* Track minimum loss point */
    if (ctx->smoothed_loss < ctx->best_loss) {
        ctx->best_loss = ctx->smoothed_loss;
        ctx->best_step = step;
    }

    /* Divergence detection: loss > threshold * min_loss */
    if (step > ctx->config.min_steps_before_stop
        && ctx->config.stop_on_diverge
        && ctx->smoothed_loss > ctx->config.divergence_threshold * ctx->best_loss) {
        ctx->diverged = true;
        ctx->result.diverged = true;
        ctx->result.divergence_step = step;
        ctx->finished = true;
        return false;
    }

    ctx->current_step++;
    if (step + 1 >= ctx->config.num_steps) ctx->finished = true;
    return !ctx->finished;
}

float lrf_suggest_lr(lrf_context_t* ctx) {
    if (!ctx || ctx->result.num_steps <= 0) return 1e-3f;

    /* Smith (2017): suggested_lr = lr_at_min_loss / 10
     * This conservative estimate avoids divergence. */
    float lr_at_min = ctx->result.lr_history[ctx->best_step];
    float suggested = lr_at_min / 10.0f;

    if (suggested < 1e-7f) suggested = 1e-7f;
    if (suggested > 1.0f) suggested = 1.0f;

    ctx->result.suggested_lr = suggested;
    ctx->result.min_loss = ctx->best_loss;
    ctx->result.min_loss_step = ctx->best_step;
    ctx->result.suggested_lr_max = lr_at_min;
    ctx->result.suggested_lr_min = suggested / 10.0f;

    return suggested;
}

float lrf_suggest_lr_steepest(lrf_context_t* ctx) {
    if (!ctx || ctx->result.num_steps < 3) return 1e-3f;

    /* Find steepest descent: max -dloss/dlr via central differences */
    float max_neg_deriv = 0.0f;
    int best_idx = 0;

    for (int i = 1; i < ctx->result.num_steps - 1; i++) {
        float dlr = ctx->result.lr_history[i + 1] - ctx->result.lr_history[i - 1];
        if (fabsf(dlr) < 1e-12f) continue;
        float dloss = ctx->result.loss_history[i + 1] - ctx->result.loss_history[i - 1];
        float deriv = -dloss / dlr;
        if (deriv > max_neg_deriv) {
            max_neg_deriv = deriv;
            best_idx = i;
        }
    }
    float steepest_lr = ctx->result.lr_history[best_idx];
    if (steepest_lr < 1e-7f) steepest_lr = 1e-7f;
    return steepest_lr;
}

void lrf_plot(lrf_context_t* ctx, const char* filename) {
    if (!ctx || !filename) return;
    FILE* f = fopen(filename, "w");
    if (!f) return;
    fprintf(f, "# LR Range Test Plot\n");
    fprintf(f, "# step, learning_rate, loss\n");
    for (int i = 0; i < ctx->result.num_steps; i++) {
        fprintf(f, "%d %.10f %.6f\n", i,
                ctx->result.lr_history[i],
                ctx->result.loss_history[i]);
    }
    float suggested = lrf_suggest_lr(ctx);
    fprintf(f, "# Suggested LR: %.10f (at step %d, loss=%.6f)\n",
            suggested, ctx->best_step, ctx->best_loss);
    fclose(f);
}

float lrf_binary_search_lr(float start_lr, float end_lr, float tol,
                            float (*eval_fn)(float lr, void* userdata),
                            void* userdata) {
    if (!eval_fn) return (start_lr + end_lr) * 0.5f;
    if (tol <= 0.0f) tol = 1e-6f;

    float lo = start_lr, hi = end_lr;
    float best_lr = lo, best_loss = eval_fn(lo, userdata);

    /* Evaluate endpoints */
    float loss_hi = eval_fn(hi, userdata);
    if (loss_hi < best_loss) { best_loss = loss_hi; best_lr = hi; }

    /* Ternary search in log-space for the minimum */
    for (int iter = 0; iter < 50; iter++) {
        if (hi - lo < tol) break;
        float m1 = lo + (hi - lo) / 3.0f;
        float m2 = hi - (hi - lo) / 3.0f;
        float f1 = eval_fn(m1, userdata);
        float f2 = eval_fn(m2, userdata);

        if (f1 < best_loss) { best_loss = f1; best_lr = m1; }
        if (f2 < best_loss) { best_loss = f2; best_lr = m2; }

        if (f1 < f2) hi = m2;
        else         lo = m1;
    }
    return best_lr;
}
