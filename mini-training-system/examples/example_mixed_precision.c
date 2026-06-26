#include "mixed_precision.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define TENSOR_SIZE 1024
#define DEMO_MATRIX_M 128
#define DEMO_MATRIX_N 128
#define DEMO_MATRIX_K 256

static void print_fp16(const char* label, uint16_t v) {
    float f = fp16_to_fp32_scalar(v);
    printf("  %s: 0x%04X = %.6f\n", label, (unsigned)v, f);
}

static void demo_conversion(void) {
    printf("\n--- FP32 <-> FP16 Conversion ---\n");

    float test_values[] = {1.0f, -1.0f, 0.5f, 0.0f, 3.14159f,
                           65504.0f, 6.1035e-5f, 1e-8f, 1e8f, 1e-4f};
    int n = (int)(sizeof(test_values) / sizeof(test_values[0]));

    printf("  %-14s %8s %8s\n", "FP32", "FP16-hex", "FP32-back");
    printf("  %-14s %8s %8s\n", "----", "--------", "--------");
    for (int i = 0; i < n; i++) {
        uint16_t h = fp32_to_fp16_scalar(test_values[i]);
        float back = fp16_to_fp32_scalar(h);
        printf("  %-14.6f 0x%04X   %-14.6f\n", test_values[i], (unsigned)h, back);
    }
}

static void demo_bf16_conversion(void) {
    printf("\n--- FP32 <-> BF16 Conversion ---\n");

    float test_values[] = {1.0f, 3.14159f, 65536.0f, 1e-3f, 1e8f};
    int n = (int)(sizeof(test_values) / sizeof(test_values[0]));

    for (int i = 0; i < n; i++) {
        uint16_t b = fp32_to_bf16_scalar(test_values[i]);
        float back = bf16_to_fp32_scalar(b);
        printf("  FP32: %-12.6f -> BF16: 0x%04X -> FP32: %-12.6f (err=%.2e)\n",
               test_values[i], (unsigned)b, back,
               (double)fabsf(test_values[i] - back));
    }
}

static void demo_loss_scaling(void) {
    printf("\n--- Dynamic Loss Scaling ---\n");

    mp_config_t cfg = {0};
    cfg.mode = MP_MODE_FP16;
    cfg.init_loss_scale = 1024.0f;
    cfg.scale_window = 3;
    cfg.scale_factor = 2.0f;
    cfg.backoff_factor = 0.5f;
    cfg.scale_mode = MP_LOSS_SCALE_DYNAMIC;

    mp_context_t ctx;
    mp_init(&ctx, &cfg);
    printf("  Init loss_scale = %.0f\n", ctx.loss_scale);

    int steps = 20;
    int p = 7;
    for (int s = 0; s < steps; s++) {
        bool overflow = (s % p == 0);
        float new_scale = mp_update_loss_scale(&ctx, overflow);
        printf("  Step %3d | scale=%10.0f | good=%2d | overflow=%s\n",
               s, new_scale, ctx.good_steps, overflow ? "YES" : "no");
        if (overflow) p = (p == 7) ? 13 : 7;
    }
}

static void demo_fp16_matmul(void) {
    printf("\n--- FP16 Matrix Multiply (M=%d, N=%d, K=%d) ---\n",
           DEMO_MATRIX_M, DEMO_MATRIX_N, DEMO_MATRIX_K);

    uint16_t* A = (uint16_t*)malloc((size_t)(DEMO_MATRIX_M * DEMO_MATRIX_K) * sizeof(uint16_t));
    uint16_t* B = (uint16_t*)malloc((size_t)(DEMO_MATRIX_K * DEMO_MATRIX_N) * sizeof(uint16_t));
    float* C_fp16 = (float*)calloc((size_t)(DEMO_MATRIX_M * DEMO_MATRIX_N), sizeof(float));
    float* C_fp32 = (float*)calloc((size_t)(DEMO_MATRIX_M * DEMO_MATRIX_N), sizeof(float));

    for (int i = 0; i < DEMO_MATRIX_M * DEMO_MATRIX_K; i++)
        A[i] = fp32_to_fp16_scalar(((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f);
    for (int i = 0; i < DEMO_MATRIX_K * DEMO_MATRIX_N; i++)
        B[i] = fp32_to_fp16_scalar(((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f);

    clock_t t0 = clock();
    mp_fp16_matmul(A, B, C_fp16, DEMO_MATRIX_M, DEMO_MATRIX_N, DEMO_MATRIX_K, false);
    clock_t tfp16 = clock() - t0;

    float* Af32 = (float*)malloc((size_t)(DEMO_MATRIX_M * DEMO_MATRIX_K) * sizeof(float));
    float* Bf32 = (float*)malloc((size_t)(DEMO_MATRIX_K * DEMO_MATRIX_N) * sizeof(float));
    fp16_to_fp32(A, Af32, (size_t)(DEMO_MATRIX_M * DEMO_MATRIX_K));
    fp16_to_fp32(B, Bf32, (size_t)(DEMO_MATRIX_K * DEMO_MATRIX_N));

    t0 = clock();
    for (int i = 0; i < DEMO_MATRIX_M; i++)
        for (int j = 0; j < DEMO_MATRIX_N; j++)
            for (int k = 0; k < DEMO_MATRIX_K; k++)
                C_fp32[i * DEMO_MATRIX_N + j] += Af32[i * DEMO_MATRIX_K + k]
                                                * Bf32[k * DEMO_MATRIX_N + j];
    clock_t tfp32 = clock() - t0;

    float max_err = 0.0f;
    int err_count = 0;
    for (int i = 0; i < DEMO_MATRIX_M * DEMO_MATRIX_N; i++) {
        float e = fabsf(C_fp16[i] - C_fp32[i]);
        if (e > max_err) max_err = e;
        if (e > 0.01f) err_count++;
    }

    printf("  FP16 time: %.2f ms\n", 1000.0 * (double)tfp16 / (double)CLOCKS_PER_SEC);
    printf("  FP32 time: %.2f ms\n", 1000.0 * (double)tfp32 / (double)CLOCKS_PER_SEC);
    printf("  Max error: %.6f (errors > 0.01: %d / %d)\n",
           max_err, err_count, DEMO_MATRIX_M * DEMO_MATRIX_N);

    free(A); free(B); free(C_fp16); free(C_fp32); free(Af32); free(Bf32);
}

static void demo_gradient_scaling(void) {
    printf("\n--- Gradient Scaling & Overflow Check ---\n");

    size_t n = 100;
    float* grads = (float*)malloc(n * sizeof(float));
    float scale = 65536.0f;

    for (int iter = 0; iter < 10; iter++) {
        for (size_t i = 0; i < n; i++)
            grads[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 1e-6f;

        if (iter == 5) grads[50] = 1e20f;

        bool has_inf_nan = false;
        mp_grad_scale_and_check(grads, n, scale, &has_inf_nan);

        float norm = 0.0f;
        for (size_t i = 0; i < n; i++) norm += grads[i] * grads[i];
        norm = sqrtf(norm);

        printf("  Iter %d | scale=%.0f | norm=%.4e | overflow=%s\n",
               iter, scale, norm, has_inf_nan ? "YES" : "no");

        if (has_inf_nan) scale *= 0.5f;
        else scale *= 2.0f;
        if (scale > 1e7f) scale = 1e7f;
        if (scale < 1.0f) scale = 1.0f;
    }
    free(grads);
}

static void demo_autocast_pipeline(void) {
    printf("\n--- Automatic Mixed Precision Pipeline ---\n");

    size_t n = 64;
    float* input = (float*)malloc(n * sizeof(float));
    float* output = (float*)malloc(n * sizeof(float));
    uint16_t* half_buf = (uint16_t*)malloc(n * sizeof(uint16_t));

    for (size_t i = 0; i < n; i++) input[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;

    printf("  Forward (float -> half) -> Backward (half -> float):\n");

    mp_autocast_forward(input, half_buf, n, MP_MODE_FP16);
    mp_autocast_backward(half_buf, output, n, MP_MODE_FP16);

    float max_err = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float e = fabsf(input[i] - output[i]);
        if (e > max_err) max_err = e;
    }
    printf("    Max round-trip error (FP16): %.6e\n", max_err);

    mp_autocast_forward(input, half_buf, n, MP_MODE_BF16);
    mp_autocast_backward(half_buf, output, n, MP_MODE_BF16);

    max_err = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float e = fabsf(input[i] - output[i]);
        if (e > max_err) max_err = e;
    }
    printf("    Max round-trip error (BF16): %.6e\n", max_err);

    free(input); free(output); free(half_buf);
}

static void demo_precision_error(void) {
    printf("\n--- Half Precision Error Analysis ---\n");

    size_t n = 1024;
    float* ref = (float*)malloc(n * sizeof(float));
    uint16_t* half = (uint16_t*)malloc(n * sizeof(uint16_t));

    for (size_t i = 0; i < n; i++) ref[i] = ((float)rand() / (float)RAND_MAX) * 1000.0f - 500.0f;
    fp32_to_fp16(ref, half, n);

    float max_err = mp_half_precision_error(ref, half, n);
    printf("  Max FP16 quantization error over %zu values: %.6e\n", n, max_err);

    free(ref); free(half);
}

static void demo_safe_ops(void) {
    printf("\n--- FP16 Safe Operations Check ---\n");

    const char* op_names[] = {
        "matmul", "conv2d", "layer_norm", "softmax", "attention"
    };
    for (int i = 0; i < MP_OP_SAFE_OPS_COUNT; i++) {
        mp_safe_op_t op = (mp_safe_op_t)i;
        bool is_safe = mp_is_safe_fp16_op(op);
        printf("  %-12s -> %s\n", op_names[i], is_safe ? "FP16-safe" : "needs FP32");
    }
}

int main(void) {
    printf("============================================================\n");
    printf("  Mixed Precision Training Example\n");
    printf("============================================================\n");

    srand(42);

    demo_conversion();
    demo_bf16_conversion();
    demo_loss_scaling();
    demo_fp16_matmul();
    demo_gradient_scaling();
    demo_autocast_pipeline();
    demo_precision_error();
    demo_safe_ops();

    printf("\nTensorCore available: %s\n", mp_tensorcore_available() ? "YES" : "NO");
    printf("All demos complete.\n");
    return 0;
}
