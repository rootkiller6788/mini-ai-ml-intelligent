#ifndef MIXED_PRECISION_H
#define MIXED_PRECISION_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MP_FP16_MIN_NORMAL (6.103515625e-05f)
#define MP_FP16_MAX (65504.0f)
#define MP_BF16_EPSILON (0.0078125f)

typedef enum {
    MP_MODE_FP32 = 0,
    MP_MODE_FP16,
    MP_MODE_BF16,
    MP_MODE_PURE_FP16,
} mp_mode_t;

typedef enum {
    MP_LOSS_SCALE_DYNAMIC = 0,
    MP_LOSS_SCALE_CONSTANT,
} mp_loss_scale_mode_t;

typedef struct {
    mp_mode_t mode;
    float init_loss_scale;
    float min_loss_scale;
    float max_loss_scale;
    int   scale_window;
    float scale_factor;
    float backoff_factor;
    int   min_scale_window;
    mp_loss_scale_mode_t scale_mode;
    bool  use_tensor_cores;
    bool  master_weights_in_fp32;
    bool  verbose;
} mp_config_t;

typedef struct {
    mp_config_t config;
    float loss_scale;
    int   good_steps;
    int   overflow_count;
    bool  found_inf;
    bool  found_nan;
    int   total_steps;
} mp_context_t;

typedef enum {
    MP_OP_MATMUL = 0,
    MP_OP_CONV2D,
    MP_OP_LAYER_NORM,
    MP_OP_SOFTMAX,
    MP_OP_ATTENTION,
    MP_OP_SAFE_OPS_COUNT,
} mp_safe_op_t;

void mp_init(mp_context_t* ctx, const mp_config_t* cfg);
void mp_reset(mp_context_t* ctx);

void fp32_to_fp16(const float* src, uint16_t* dst, size_t n);
void fp16_to_fp32(const uint16_t* src, float* dst, size_t n);

void fp32_to_bf16(const float* src, uint16_t* dst, size_t n);
void bf16_to_fp32(const uint16_t* src, float* dst, size_t n);

uint16_t fp32_to_fp16_scalar(float x);
float    fp16_to_fp32_scalar(uint16_t x);
uint16_t fp32_to_bf16_scalar(float x);
float    bf16_to_fp32_scalar(uint16_t x);

bool mp_is_safe_fp16_op(mp_safe_op_t op);
bool mp_check_fp16_overflow(const uint16_t* data, size_t n);

void  mp_scale_loss(float* loss, float scale);
void  mp_unscale_gradients(float* grads, size_t numel, float scale,
                            bool* overflow);
float mp_update_loss_scale(mp_context_t* ctx, bool had_overflow);

void mp_master_to_model_fp16(const float* master, uint16_t* model,
                              size_t numel);
void mp_model_to_master_fp16(const uint16_t* model, float* master,
                              size_t numel);

void mp_fp16_matmul(const uint16_t* A, const uint16_t* B, float* C,
                    int M, int N, int K, bool use_tensorcore);

void mp_fp16_matmul_tensorcore(const uint16_t* A, const uint16_t* B,
                                 float* C, int M, int N, int K);
bool mp_tensorcore_available(void);

void mp_grad_scale_and_check(float* grads, size_t numel, float scale,
                              bool* has_inf_nan);

void fp32_accum_to_fp16(const float* accum, uint16_t* out, size_t n);
void fp16_gemm_accum(uint16_t* result, const uint16_t* A, const uint16_t* B,
                      int M, int N, int K);

float mp_half_precision_error(const float* ref, const uint16_t* half,
                               size_t n);

void mp_autocast_forward(const float* input, uint16_t* output, size_t n,
                          mp_mode_t mode);
void mp_autocast_backward(const uint16_t* grads, float* grads_fp32,
                           size_t n, mp_mode_t mode);

#ifdef __cplusplus
}
#endif

#endif
