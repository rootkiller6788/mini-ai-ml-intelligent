#include "mixed_precision.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

static const uint32_t FP16_EXP_MASK  = 0x7C00u;
static const uint32_t FP16_SIGN_MASK = 0x8000u;
static const uint32_t FP16_MANT_MASK = 0x03FFu;

uint16_t fp32_to_fp16_scalar(float x) {
    uint32_t bits; memcpy(&bits, &x, 4);
    uint32_t sign = (bits >> 16) & FP16_SIGN_MASK;
    int32_t  exp  = (int32_t)((bits >> 23) & 0xFF) - 127;
    uint32_t mant = bits & 0x7FFFFF;

    if (exp >= 16) return (uint16_t)(sign | FP16_EXP_MASK);
    if (exp <= -15) return (uint16_t)sign;

    if (exp <= -1) {
        mant = (mant | 0x800000) >> (1 - exp);
        return (uint16_t)(sign | (mant >> 13));
    }

    return (uint16_t)(sign | ((uint32_t)(exp + 15) << 10) | (mant >> 13));
}

float fp16_to_fp32_scalar(uint16_t x) {
    uint32_t sign = (x & FP16_SIGN_MASK) << 16;
    uint32_t exp  = (x & FP16_EXP_MASK) >> 10;
    uint32_t mant = (x & FP16_MANT_MASK);

    if (exp == 0) {
        if (mant == 0) { uint32_t r = sign; float f; memcpy(&f, &r, 4); return f; }
        uint32_t e2 = 0;
        while ((mant & 0x400) == 0) { mant <<= 1; e2++; }
        mant &= 0x3FF;
        uint32_t r = sign | ((127 - 14 - e2) << 23) | (mant << 13);
        float f; memcpy(&f, &r, 4); return f;
    }
    if (exp == 31) {
        uint32_t r = sign | 0x7F800000 | (mant << 13);
        float f; memcpy(&f, &r, 4); return f;
    }
    uint32_t r = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    float f; memcpy(&f, &r, 4); return f;
}

uint16_t fp32_to_bf16_scalar(float x) {
    uint32_t bits; memcpy(&bits, &x, 4);
    return (uint16_t)(bits >> 16);
}

float bf16_to_fp32_scalar(uint16_t x) {
    uint32_t r = ((uint32_t)x) << 16;
    float f; memcpy(&f, &r, 4); return f;
}

void fp32_to_fp16(const float* src, uint16_t* dst, size_t n) {
    for (size_t i = 0; i < n; i++) dst[i] = fp32_to_fp16_scalar(src[i]);
}

void fp16_to_fp32(const uint16_t* src, float* dst, size_t n) {
    for (size_t i = 0; i < n; i++) dst[i] = fp16_to_fp32_scalar(src[i]);
}

void fp32_to_bf16(const float* src, uint16_t* dst, size_t n) {
    for (size_t i = 0; i < n; i++) dst[i] = fp32_to_bf16_scalar(src[i]);
}

void bf16_to_fp32(const uint16_t* src, float* dst, size_t n) {
    for (size_t i = 0; i < n; i++) dst[i] = bf16_to_fp32_scalar(src[i]);
}

void mp_init(mp_context_t* ctx, const mp_config_t* cfg) {
    if (!ctx) return;
    memset(ctx, 0, sizeof(mp_context_t));
    if (cfg) {
        ctx->config = *cfg;
    } else {
        ctx->config.mode = MP_MODE_FP16;
        ctx->config.init_loss_scale = 65536.0f;
        ctx->config.min_loss_scale = 1.0f;
        ctx->config.max_loss_scale = 16777216.0f;
        ctx->config.scale_window = 2000;
        ctx->config.scale_factor = 2.0f;
        ctx->config.backoff_factor = 0.5f;
        ctx->config.min_scale_window = 100;
        ctx->config.scale_mode = MP_LOSS_SCALE_DYNAMIC;
        ctx->config.master_weights_in_fp32 = true;
    }
    ctx->loss_scale = ctx->config.init_loss_scale;
    ctx->good_steps = 0;
}

void mp_reset(mp_context_t* ctx) {
    if (!ctx) return;
    ctx->loss_scale = ctx->config.init_loss_scale;
    ctx->good_steps = 0;
    ctx->overflow_count = 0;
    ctx->found_inf = false;
    ctx->found_nan = false;
}

bool mp_is_safe_fp16_op(mp_safe_op_t op) {
    switch (op) {
    case MP_OP_MATMUL:    case MP_OP_CONV2D:
    case MP_OP_LAYER_NORM: case MP_OP_SOFTMAX:
    case MP_OP_ATTENTION:
        return true;
    default:
        return false;
    }
}

bool mp_check_fp16_overflow(const uint16_t* data, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if ((data[i] & FP16_EXP_MASK) == FP16_EXP_MASK) return true;
    }
    return false;
}

void mp_scale_loss(float* loss, float scale) {
    if (loss) *loss *= scale;
}

void mp_unscale_gradients(float* grads, size_t numel, float scale,
                           bool* overflow) {
    if (!grads || !overflow) return;
    *overflow = false;
    float inv = 1.0f / scale;
    for (size_t i = 0; i < numel; i++) {
        grads[i] *= inv;
        if (!isfinite(grads[i])) { *overflow = true; return; }
    }
}

float mp_update_loss_scale(mp_context_t* ctx, bool had_overflow) {
    if (!ctx) return 1.0f;
    if (had_overflow) {
        ctx->good_steps = 0;
        ctx->overflow_count++;
        ctx->loss_scale *= ctx->config.backoff_factor;
        if (ctx->loss_scale < ctx->config.min_loss_scale)
            ctx->loss_scale = ctx->config.min_loss_scale;
    } else {
        ctx->good_steps++;
        if (ctx->good_steps >= ctx->config.scale_window) {
            ctx->loss_scale *= ctx->config.scale_factor;
            if (ctx->loss_scale > ctx->config.max_loss_scale)
                ctx->loss_scale = ctx->config.max_loss_scale;
            ctx->good_steps = 0;
        }
    }
    ctx->total_steps++;
    return ctx->loss_scale;
}

void mp_master_to_model_fp16(const float* master, uint16_t* model,
                              size_t numel) {
    fp32_to_fp16(master, model, numel);
}

void mp_model_to_master_fp16(const uint16_t* model, float* master,
                              size_t numel) {
    fp16_to_fp32(model, master, numel);
}

void mp_fp16_matmul(const uint16_t* A, const uint16_t* B, float* C,
                    int M, int N, int K, bool use_tensorcore) {
    if (use_tensorcore && mp_tensorcore_available()) {
        mp_fp16_matmul_tensorcore(A, B, C, M, N, K);
        return;
    }
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += fp16_to_fp32_scalar(A[i * K + k]) * fp16_to_fp32_scalar(B[k * N + j]);
            }
            C[i * N + j] = sum;
        }
    }
}

void mp_fp16_matmul_tensorcore(const uint16_t* A, const uint16_t* B,
                                 float* C, int M, int N, int K) {
    mp_fp16_matmul(A, B, C, M, N, K, false);
}

bool mp_tensorcore_available(void) { return false; }

void mp_grad_scale_and_check(float* grads, size_t numel, float scale,
                              bool* has_inf_nan) {
    if (!grads || !has_inf_nan) return;
    *has_inf_nan = false;
    for (size_t i = 0; i < numel; i++) {
        grads[i] *= scale;
        if (!isfinite(grads[i])) { *has_inf_nan = true; grads[i] = 0.0f; }
    }
}

void fp32_accum_to_fp16(const float* accum, uint16_t* out, size_t n) {
    fp32_to_fp16(accum, out, n);
}

void fp16_gemm_accum(uint16_t* result, const uint16_t* A, const uint16_t* B,
                      int M, int N, int K) {
    float* temp = (float*)calloc((size_t)(M * N), sizeof(float));
    if (!temp) return;
    mp_fp16_matmul(A, B, temp, M, N, K, false);
    fp32_to_fp16(temp, result, (size_t)(M * N));
    free(temp);
}

float mp_half_precision_error(const float* ref, const uint16_t* half,
                               size_t n) {
    float max_err = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float diff = fabsf(ref[i] - fp16_to_fp32_scalar(half[i]));
        if (diff > max_err) max_err = diff;
    }
    return max_err;
}

void mp_autocast_forward(const float* input, uint16_t* output, size_t n,
                          mp_mode_t mode) {
    switch (mode) {
    case MP_MODE_FP16: case MP_MODE_PURE_FP16: fp32_to_fp16(input, output, n); break;
    case MP_MODE_BF16: fp32_to_bf16(input, output, n); break;
    default: for (size_t i = 0; i < n; i++) output[i] = fp32_to_fp16_scalar(input[i]); break;
    }
}

void mp_autocast_backward(const uint16_t* grads, float* grads_fp32,
                           size_t n, mp_mode_t mode) {
    switch (mode) {
    case MP_MODE_FP16: case MP_MODE_PURE_FP16: fp16_to_fp32(grads, grads_fp32, n); break;
    case MP_MODE_BF16: bf16_to_fp32(grads, grads_fp32, n); break;
    default: for (size_t i = 0; i < n; i++) grads_fp32[i] = fp16_to_fp32_scalar(grads[i]); break;
    }
}
