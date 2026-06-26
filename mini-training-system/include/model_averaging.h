#ifndef MODEL_AVERAGING_H
#define MODEL_AVERAGING_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Model Averaging — EMA and Stochastic Weight Averaging (SWA)
 *
 * Theorem: Polyak-Ruppert Averaging (1992):
 *   For convex optimization, averaging the iterates w̄_T = (1/T) Σ w_t
 *   achieves optimal O(1/√T) convergence rate.
 *
 * EMA (Exponential Moving Average):
 *   θ_ema = β·θ_ema + (1-β)·θ_current
 *   Typically β = 0.999 for large models (common in SSL and diffusion).
 *
 * SWA (Stochastic Weight Averaging) — Izmailov et al. (2018):
 *   Averaging weights at the end of training finds wider minima with
 *   better generalization. Unlike EMA, SWA uses uniform averaging.
 *
 * Reference:
 * - Polyak & Juditsky (1992) "Acceleration of Stochastic Approximation"
 * - Izmailov et al. (2018) "Averaging Weights Leads to Wider Optima"
 * - Athiwaratkun et al. (2019) "There Are Many Consistent Explanations"
 */

#define MA_MAX_PREFIX 256

typedef enum {
    MA_MODE_EMA = 0,     /**< Exponential Moving Average */
    MA_MODE_SWA,         /**< Stochastic Weight Averaging (uniform) */
    MA_MODE_LAWA,        /**< Latest Weight Averaging (window-based) */
} ma_mode_t;

typedef struct {
    ma_mode_t mode;
    float  ema_decay;          /**< Decay factor β for EMA (default: 0.999) */
    int    swa_start_epoch;    /**< Epoch to start SWA collection */
    int    swa_freq_steps;     /**< Collect weights every N steps */
    int    lawa_window_size;   /**< Window size for LAWA */
    bool   use_bias_correction;/**< Apply bias correction for EMA */
    bool   update_bn_stats;    /**< Update BN running stats after averaging */
    int    bn_update_steps;    /**< Steps for BN statistics update */
} ma_config_t;

typedef struct {
    float* ema_weights;
    float* swa_weights;
    float* swa_accum;
    int    swa_count;
    size_t num_params;
    float  ema_decay;
    float  correction_factor;
    bool   started;
    int    current_epoch;
    int    global_step;
} ma_context_t;

/**
 * Initialize model averaging context.
 * Allocates internal buffers to hold EMA and SWA copies of weights.
 */
void ma_init(ma_context_t* ctx, const ma_config_t* cfg, size_t num_params);
void ma_free(ma_context_t* ctx);

/**
 * Reset EMA/SWA to the current model weights (cold start).
 */
void ma_reset(ma_context_t* ctx, const float* current_weights);

/**
 * Update EMA: θ_ema = β·θ_ema + (1-β)·θ_current
 * With bias correction: θ_ema_corrected = θ_ema / (1 - β^t)
 *
 * @param ctx          Averaging context
 * @param weights      Current model weights (size = num_params)
 * @param step         Current global step (for bias correction)
 */
void ma_update_ema(ma_context_t* ctx, const float* weights, int step);

/**
 * Collect weights for SWA on schedule.
 * Should be called at the configured frequency after swa_start_epoch.
 */
void ma_collect_swa(ma_context_t* ctx, const float* weights, int epoch, int step);

/**
 * Compute the averaged weights into `out_weights`.
 * For EMA: simply copies the EMA buffer.
 * For SWA: computes swa_accum / swa_count.
 */
void ma_get_averaged_weights(const ma_context_t* ctx, float* out_weights);

/**
 * Copy EMA weights back into the model (for inference).
 */
void ma_swap_to_ema(ma_context_t* ctx, float* model_weights);

/**
 * Restore original (pre-EMA) weights.
 * Only meaningful if swap counts are tracked; here we don't store originals.
 */
void ma_restore_original(ma_context_t* ctx, float* model_weights);

/**
 * SWA batch normalization statistics update.
 * After SWA model is built, run a few forward passes with BN in eval mode
 * to update running means/variances.
 */
void ma_update_bn_stats(ma_context_t* ctx, int num_steps);

#ifdef __cplusplus
}
#endif

#endif
