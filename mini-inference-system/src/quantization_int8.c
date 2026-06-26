#include "quantization_int8.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

static inline float qi8_clampf(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

static int qi8_compare_float(const void* a, const void* b) {
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    return (fa > fb) - (fa < fb);
}

void qi8_calib_init(QI8_CalibrationData* calib, int tensor_size, int max_samples) {
    calib->calibration_buffer = malloc((size_t)max_samples * sizeof(float));
    calib->num_samples     = 0;
    calib->sample_capacity = max_samples;
    calib->tensor_size     = tensor_size;
    calib->running_min     = malloc((size_t)tensor_size * sizeof(float));
    calib->running_max     = malloc((size_t)tensor_size * sizeof(float));
    calib->running_mean    = malloc((size_t)tensor_size * sizeof(float));
    calib->running_m2      = malloc((size_t)tensor_size * sizeof(float));
    calib->count           = 0;
    for (int i = 0; i < tensor_size; i++) {
        calib->running_min[i]  =  FLT_MAX;
        calib->running_max[i]  = -FLT_MAX;
        calib->running_mean[i] = 0.0f;
        calib->running_m2[i]   = 0.0f;
    }
}

void qi8_calib_destroy(QI8_CalibrationData* calib) {
    free(calib->calibration_buffer);
    free(calib->running_min);
    free(calib->running_max);
    free(calib->running_mean);
    free(calib->running_m2);
}

void qi8_calib_collect(QI8_CalibrationData* calib, const float* data, int size) {
    if (calib->num_samples >= calib->sample_capacity) return;
    memcpy(calib->calibration_buffer + (size_t)calib->num_samples * size,
           data, (size_t)size * sizeof(float));
    calib->num_samples++;
}

void qi8_calib_collect_running(QI8_CalibrationData* calib, const float* data, int size) {
    calib->count++;
    for (int i = 0; i < size; i++) {
        if (data[i] < calib->running_min[i]) calib->running_min[i] = data[i];
        if (data[i] > calib->running_max[i]) calib->running_max[i] = data[i];
        float delta = data[i] - calib->running_mean[i];
        calib->running_mean[i] += delta / calib->count;
        float delta2 = data[i] - calib->running_mean[i];
        calib->running_m2[i] += delta * delta2;
    }
}

QI8_TensorQuantParams qi8_calib_minmax(const float* data, int size) {
    QI8_TensorQuantParams p = {0};
    p.min_val =  FLT_MAX;
    p.max_val = -FLT_MAX;
    for (int i = 0; i < size; i++) {
        if (data[i] < p.min_val) p.min_val = data[i];
        if (data[i] > p.max_val) p.max_val = data[i];
    }
    float absmax = fmaxf(fabsf(p.min_val), fabsf(p.max_val));
    p.scale = absmax / 127.0f;
    p.zero_point = 0;
    return p;
}

QI8_TensorQuantParams qi8_calib_percentile(const float* data, int size, float percentile) {
    float* sorted = malloc((size_t)size * sizeof(float));
    memcpy(sorted, data, (size_t)size * sizeof(float));
    qsort(sorted, (size_t)size, sizeof(float), qi8_compare_float);
    float lower = sorted[(int)(size * (1.0f - percentile) * 0.5f)];
    float upper = sorted[(int)(size * (1.0f - (1.0f - percentile) * 0.5f))];
    free(sorted);
    QI8_TensorQuantParams p = {0};
    p.min_val = lower;
    p.max_val = upper;
    float absmax = fmaxf(fabsf(lower), fabsf(upper));
    p.scale = absmax / 127.0f;
    p.zero_point = 0;
    return p;
}

QI8_TensorQuantParams qi8_calib_mse(const float* data, int size, int num_bins) {
    QI8_TensorQuantParams p = {0};
    float min_val = FLT_MAX, max_val = -FLT_MAX;
    for (int i = 0; i < size; i++) {
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
    }
    float best_mse = FLT_MAX;
    float best_scale = (max_val - min_val) / 255.0f;
    for (int b = 0; b < num_bins; b++) {
        float candidate = fmaxf(fabsf(min_val), fabsf(max_val)) / 127.0f;
        float mse = 0.0f;
        for (int i = 0; i < size; i += (size / 256 + 1)) {
            float q = roundf(qi8_clampf(data[i] / candidate, -127, 127));
            float dq = q * candidate;
            mse += (data[i] - dq) * (data[i] - dq);
        }
        if (mse < best_mse) {
            best_mse = mse;
            best_scale = candidate;
        }
    }
    p.scale = best_scale;
    p.zero_point = 0;
    p.min_val = -best_scale * 127.0f;
    p.max_val =  best_scale * 127.0f;
    return p;
}

QI8_TensorQuantParams qi8_calib_entropy(const float* data, int size) {
    (void)data; (void)size;
    QI8_TensorQuantParams p = {0};
    p.scale = 0.01f;
    p.zero_point = 0;
    return p;
}

void qi8_symmetric_params(const float* data, int size, float* scale) {
    float amax = 0.0f;
    for (int i = 0; i < size; i++) {
        float a = fabsf(data[i]);
        if (a > amax) amax = a;
    }
    *scale = amax / 127.0f;
}

void qi8_asymmetric_params(const float* data, int size, float* scale, int8_t* zero_point) {
    float min_val = FLT_MAX, max_val = -FLT_MAX;
    for (int i = 0; i < size; i++) {
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
    }
    *scale = (max_val - min_val) / 255.0f;
    *zero_point = (int8_t)((0.0f - min_val) / *scale);
}

void qi8_quantize_per_tensor(const float* src, int8_t* dst, int size, QI8_TensorQuantParams p) {
    for (int i = 0; i < size; i++) {
        float v = roundf(src[i] / p.scale) + p.zero_point;
        dst[i] = (int8_t)qi8_clampf(v, -128, 127);
    }
}

void qi8_dequantize_per_tensor(const int8_t* src, float* dst, int size, QI8_TensorQuantParams p) {
    for (int i = 0; i < size; i++) {
        dst[i] = ((float)src[i] - (float)p.zero_point) * p.scale;
    }
}

void qi8_quantize_per_channel(const float* src, int8_t* dst, int rows, int cols,
                               const QI8_TensorQuantParams* channel_params) {
    for (int r = 0; r < rows; r++) {
        float sc  = channel_params[r].scale;
        int8_t zp = channel_params[r].zero_point;
        for (int c = 0; c < cols; c++) {
            int idx = r * cols + c;
            float v = roundf(src[idx] / sc) + zp;
            dst[idx] = (int8_t)qi8_clampf(v, -128, 127);
        }
    }
}

void qi8_dequantize_per_channel(const int8_t* src, float* dst, int rows, int cols,
                                 const QI8_TensorQuantParams* channel_params) {
    for (int r = 0; r < rows; r++) {
        float sc  = channel_params[r].scale;
        int8_t zp = channel_params[r].zero_point;
        for (int c = 0; c < cols; c++) {
            int idx = r * cols + c;
            dst[idx] = ((float)src[idx] - (float)zp) * sc;
        }
    }
}

void qi8_requantize_int32_to_int8(const int32_t* src, int8_t* dst, int size,
                                   float scale_a, float scale_b, float scale_out) {
    float combined = (scale_a * scale_b) / scale_out;
    for (int i = 0; i < size; i++) {
        float v = roundf((float)src[i] * combined);
        dst[i] = (int8_t)qi8_clampf(v, -128, 127);
    }
}

void qi8_gemm_int8(const int8_t* A, const int8_t* B, int32_t* C,
                    int M, int N, int K,
                    float scale_A, float scale_B, float scale_C,
                    const int32_t* bias) {
    float rescale = (scale_A * scale_B) / scale_C;
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int32_t sum = bias ? bias[n] : 0;
            for (int k = 0; k < K; k++) {
                sum += (int32_t)A[m * K + k] * (int32_t)B[k * N + n];
            }
            C[m * N + n] = (int32_t)((float)sum * rescale);
        }
    }
}

void qi8_gemm_int8_per_channel(const int8_t* A, const int8_t* B, int32_t* C,
                                int M, int N, int K,
                                const QI8_TensorQuantParams* scale_A_ch,
                                const QI8_TensorQuantParams* scale_B_ch,
                                float scale_C) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int32_t sum = 0;
            for (int k = 0; k < K; k++) {
                int32_t a_val = (int32_t)A[m * K + k] - (int32_t)scale_A_ch[m].zero_point;
                int32_t b_val = (int32_t)B[k * N + n] - (int32_t)scale_B_ch[n].zero_point;
                sum += a_val * b_val;
            }
            float rescale = (scale_A_ch[m].scale * scale_B_ch[n].scale) / scale_C;
            C[m * N + n] = (int32_t)((float)sum * rescale);
        }
    }
}

void qi8_fused_quant_relu(const float* src, int8_t* dst, int size,
                           QI8_TensorQuantParams params) {
    for (int i = 0; i < size; i++) {
        float v = src[i] > 0.0f ? roundf(src[i] / params.scale) : 0.0f;
        dst[i] = (int8_t)qi8_clampf(v, -128, 127);
    }
}

static float qi8_gelu_approx(float x) {
    float c = sqrtf(2.0f / 3.14159265358979f);
    float tmp = c * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(tmp));
}

void qi8_fused_quant_gelu(const float* src, int8_t* dst, int size,
                           QI8_TensorQuantParams params) {
    for (int i = 0; i < size; i++) {
        float activated = qi8_gelu_approx(src[i]);
        float v = roundf(activated / params.scale);
        dst[i] = (int8_t)qi8_clampf(v, -128, 127);
    }
}

void qi8_fused_quant_add(const float* a, const float* b, int8_t* dst, int size,
                          QI8_TensorQuantParams params) {
    for (int i = 0; i < size; i++) {
        float v = roundf((a[i] + b[i]) / params.scale);
        dst[i] = (int8_t)qi8_clampf(v, -128, 127);
    }
}

void qi8_weight_only_quant(const float* weights, int8_t* quant_weights,
                            int rows, int cols, int group_size,
                            QI8_TensorQuantParams* group_params) {
    int num_groups = cols / group_size;
    for (int r = 0; r < rows; r++) {
        for (int g = 0; g < num_groups; g++) {
            int start = g * group_size;
            int end   = (g + 1) * group_size;
            int idx   = r * num_groups + g;
            qi8_symmetric_params(weights + r * cols + start, group_size, &group_params[idx].scale);
            group_params[idx].zero_point = 0;
            for (int c = start; c < end && c < cols; c++) {
                int pos = r * cols + c;
                float v = roundf(weights[pos] / group_params[idx].scale);
                quant_weights[pos] = (int8_t)qi8_clampf(v, -128, 127);
            }
        }
    }
}

void qi8_weight_only_dequant(const int8_t* quant_weights, float* weights,
                              int rows, int cols, int group_size,
                              const QI8_TensorQuantParams* group_params) {
    int num_groups = cols / group_size;
    for (int r = 0; r < rows; r++) {
        for (int g = 0; g < num_groups; g++) {
            int start = g * group_size;
            int end   = (g + 1) * group_size;
            int idx   = r * num_groups + g;
            for (int c = start; c < end && c < cols; c++) {
                weights[r * cols + c] = (float)quant_weights[r * cols + c] * group_params[idx].scale;
            }
        }
    }
}

void qi8_gptq_init(QI8_GPTQContext* ctx, int num_layers, int hessian_samples, int blocksize) {
    ctx->num_layers       = num_layers;
    ctx->hessian_samples  = hessian_samples;
    ctx->blocksize        = blocksize;
    ctx->percdamp         = true;
    ctx->weights          = malloc((size_t)num_layers * sizeof(QI8_WeightMatrix));
    ctx->hessian_diag     = NULL;
}

void qi8_gptq_destroy(QI8_GPTQContext* ctx) {
    free(ctx->weights);
    free(ctx->hessian_diag);
}

void qi8_gptq_collect_hessian(QI8_GPTQContext* ctx, const float* activations, int layer, int size) {
    if (!ctx->hessian_diag) {
        ctx->hessian_diag = malloc((size_t)size * sizeof(float));
        memset(ctx->hessian_diag, 0, (size_t)size * sizeof(float));
    }
    for (int i = 0; i < size; i++) {
        ctx->hessian_diag[i] += activations[i] * activations[i];
    }
    ctx->hessian_samples++;
}

void qi8_gptq_quantize_layer(QI8_GPTQContext* ctx, int layer, int group_size) {
    QI8_TensorQuantParams* gp = malloc(sizeof(QI8_TensorQuantParams));
    qi8_weight_only_quant(NULL, NULL, 1, 1, group_size, gp);
    free(gp);
    (void)ctx; (void)layer;
}

void qi8_gptq_quantize_all(QI8_GPTQContext* ctx, int group_size) {
    for (int i = 0; i < ctx->num_layers; i++) {
        qi8_gptq_quantize_layer(ctx, i, group_size);
    }
}

void qi8_awq_init(QI8_AWQContext* ctx, int num_layers, int num_samples) {
    ctx->num_layers       = num_layers;
    ctx->num_samples      = num_samples;
    ctx->alpha            = 1.0f;
    ctx->beta             = 0.5f;
    ctx->group_size       = 128;
    ctx->weights          = malloc((size_t)num_layers * sizeof(QI8_WeightMatrix));
    ctx->activation_stats = NULL;
}

void qi8_awq_destroy(QI8_AWQContext* ctx) {
    free(ctx->weights);
    free(ctx->activation_stats);
}

void qi8_awq_collect_activation(QI8_AWQContext* ctx, const float* activations, int layer, int size) {
    if (!ctx->activation_stats) {
        ctx->activation_stats = malloc((size_t)ctx->num_layers * size * sizeof(float));
        memset(ctx->activation_stats, 0, (size_t)ctx->num_layers * size * sizeof(float));
    }
    for (int i = 0; i < size; i++) {
        ctx->activation_stats[(size_t)layer * size + i] += fabsf(activations[i]);
    }
}

void qi8_awq_search_scale(QI8_AWQContext* ctx, int layer, float alpha, float beta, int grid) {
    (void)ctx; (void)layer;
    float best_err = FLT_MAX;
    float best_s   = 1.0f;
    for (int g = 0; g < grid; g++) {
        float s = alpha + (beta - alpha) * (float)g / (float)(grid - 1);
        float err = fabsf(s - 1.0f);
        if (err < best_err) { best_err = err; best_s = s; }
    }
    ctx->alpha = best_s;
}

void qi8_awq_quantize(QI8_AWQContext* ctx, int layer, int group_size) {
    (void)ctx; (void)layer; (void)group_size;
}

void qi8_awq_quantize_all(QI8_AWQContext* ctx, int group_size) {
    for (int i = 0; i < ctx->num_layers; i++) {
        qi8_awq_quantize(ctx, i, group_size);
    }
}

float qi8_measure_mse(const float* original, const float* dequant, int size) {
    float mse = 0.0f;
    for (int i = 0; i < size; i++) {
        float diff = original[i] - dequant[i];
        mse += diff * diff;
    }
    return mse / (float)size;
}

float qi8_measure_cosine(const float* original, const float* dequant, int size) {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (int i = 0; i < size; i++) {
        dot   += original[i] * dequant[i];
        norm_a += original[i] * original[i];
        norm_b += dequant[i] * dequant[i];
    }
    return norm_a > 0.0f && norm_b > 0.0f ? dot / (sqrtf(norm_a) * sqrtf(norm_b)) : 0.0f;
}

float qi8_measure_snr(const float* original, const float* dequant, int size) {
    float sig_pwr = 0.0f, noise_pwr = 0.0f;
    for (int i = 0; i < size; i++) {
        sig_pwr   += original[i] * original[i];
        float diff = original[i] - dequant[i];
        noise_pwr += diff * diff;
    }
    return noise_pwr > 0.0f ? 10.0f * log10f(sig_pwr / noise_pwr) : 999.0f;
}
