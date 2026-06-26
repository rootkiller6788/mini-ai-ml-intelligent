#include "model_averaging.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

void ma_init(ma_context_t* ctx, const ma_config_t* cfg, size_t num_params) {
    if (!ctx || num_params == 0) return;
    memset(ctx, 0, sizeof(ma_context_t));
    ctx->num_params = num_params;
    ctx->ema_decay = 0.999f;
    ctx->started = false;

    ctx->ema_weights = (float*)calloc(num_params, sizeof(float));
    ctx->swa_weights = (float*)calloc(num_params, sizeof(float));
    ctx->swa_accum   = (float*)calloc(num_params, sizeof(float));

    if (cfg) {
        ctx->ema_decay = cfg->ema_decay > 0.0f ? cfg->ema_decay : 0.999f;
    }
}

void ma_free(ma_context_t* ctx) {
    if (!ctx) return;
    free(ctx->ema_weights);
    free(ctx->swa_weights);
    free(ctx->swa_accum);
    memset(ctx, 0, sizeof(ma_context_t));
}

void ma_reset(ma_context_t* ctx, const float* current_weights) {
    if (!ctx || !current_weights) return;
    memcpy(ctx->ema_weights, current_weights, ctx->num_params * sizeof(float));
    ctx->swa_count = 0;
    memset(ctx->swa_accum, 0, ctx->num_params * sizeof(float));
    ctx->started = false;
}

void ma_update_ema(ma_context_t* ctx, const float* weights, int step) {
    if (!ctx || !weights || ctx->num_params == 0) return;
    if (!ctx->ema_weights) return;

    float beta = ctx->ema_decay;

    if (!ctx->started) {
        /* Cold start: copy weights directly */
        memcpy(ctx->ema_weights, weights, ctx->num_params * sizeof(float));
        ctx->started = true;
        ctx->correction_factor = 1.0f;
        return;
    }

    for (size_t i = 0; i < ctx->num_params; i++) {
        ctx->ema_weights[i] = beta * ctx->ema_weights[i] + (1.0f - beta) * weights[i];
    }

    /* Bias correction: EMA is biased toward zero initially.
     * corrected_ema = ema / (1 - beta^step)
     * Applied lazily via correction_factor stored in context. */
    if (step > 0) {
        ctx->correction_factor = 1.0f - powf(beta, (float)step);
        if (ctx->correction_factor < 1e-8f) ctx->correction_factor = 1e-8f;
    }
    ctx->global_step = step;
}

void ma_collect_swa(ma_context_t* ctx, const float* weights, int epoch, int step) {
    if (!ctx || !weights || ctx->num_params == 0) return;
    if (!ctx->swa_accum) return;

    /* Accumulate weights for SWA */
    for (size_t i = 0; i < ctx->num_params; i++) {
        ctx->swa_accum[i] += weights[i];
    }
    ctx->swa_count++;
    ctx->current_epoch = epoch;
    ctx->global_step = step;
}

void ma_get_averaged_weights(const ma_context_t* ctx, float* out_weights) {
    if (!ctx || !out_weights || ctx->num_params == 0) return;

    /* Return EMA with bias correction */
    if (ctx->ema_weights) {
        float inv_correction = 1.0f / ctx->correction_factor;
        for (size_t i = 0; i < ctx->num_params; i++) {
            out_weights[i] = ctx->ema_weights[i] * inv_correction;
        }
    } else {
        memset(out_weights, 0, ctx->num_params * sizeof(float));
    }
}

void ma_swap_to_ema(ma_context_t* ctx, float* model_weights) {
    if (!ctx || !model_weights) return;
    float inv_correction = 1.0f / ctx->correction_factor;
    for (size_t i = 0; i < ctx->num_params; i++) {
        model_weights[i] = ctx->ema_weights[i] * inv_correction;
    }
}

void ma_restore_original(ma_context_t* ctx, float* model_weights) {
    (void)ctx;
    (void)model_weights;
}

void ma_update_bn_stats(ma_context_t* ctx, int num_steps) {
    (void)ctx;
    (void)num_steps;
    /* BN stat update requires forward pass infrastructure not available here.
     * This is a placeholder for integration with the training loop. */
}
