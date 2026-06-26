#include "quantization_int8.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TEST_SIZE   1024
#define NUM_SAMPLES 64

int main(void) {
    float* data = malloc(TEST_SIZE * sizeof(float));
    for (int i = 0; i < TEST_SIZE; i++) {
        data[i] = sinf((float)i * 0.01f) * 5.0f + (float)(i % 17) * 0.1f;
    }

    QI8_TensorQuantParams params_minmax = qi8_calib_minmax(data, TEST_SIZE);
    printf("MinMax calibration: scale=%.6f, min=%.3f, max=%.3f\n",
           params_minmax.scale, params_minmax.min_val, params_minmax.max_val);

    QI8_TensorQuantParams params_mse = qi8_calib_mse(data, TEST_SIZE, 64);
    printf("MSE calibration: scale=%.6f\n", params_mse.scale);

    int8_t* quant = malloc(TEST_SIZE * sizeof(int8_t));
    float*  dequant = malloc(TEST_SIZE * sizeof(float));

    qi8_quantize_per_tensor(data, quant, TEST_SIZE, params_minmax);
    qi8_dequantize_per_tensor(quant, dequant, TEST_SIZE, params_minmax);

    float mse = qi8_measure_mse(data, dequant, TEST_SIZE);
    float cos_sim = qi8_measure_cosine(data, dequant, TEST_SIZE);
    float snr = qi8_measure_snr(data, dequant, TEST_SIZE);

    printf("INT8 quality: MSE=%.6f, Cosine=%.6f, SNR=%.2f dB\n", mse, cos_sim, snr);

    QI8_CalibrationData calib;
    qi8_calib_init(&calib, TEST_SIZE, NUM_SAMPLES);
    for (int s = 0; s < NUM_SAMPLES; s++) {
        float* sample = malloc(TEST_SIZE * sizeof(float));
        for (int i = 0; i < TEST_SIZE; i++) {
            sample[i] = data[i] * (0.8f + 0.4f * ((float)s / NUM_SAMPLES));
        }
        qi8_calib_collect(&calib, sample, TEST_SIZE);
        free(sample);
    }
    printf("Collected %d calibration samples\n", calib.num_samples);
    qi8_calib_destroy(&calib);

    int8_t A[64] = {0}, B[64] = {0};
    int32_t C[64] = {0};
    qi8_gemm_int8(A, B, C, 4, 4, 4, 0.02f, 0.02f, 0.001f, NULL);
    printf("INT8 GEMM (4x4x4) complete: C[0]=%d\n", C[0]);

    float weight[128] = {0};
    int8_t wq[128] = {0};
    QI8_TensorQuantParams wp[1];
    qi8_weight_only_quant(weight, wq, 1, 128, 32, wp);
    printf("Weight-only quant (128 cols, group=32): scale=%.6f\n", wp[0].scale);

    printf("Quantization example complete.\n");
    free(data);
    free(quant);
    free(dequant);
    return 0;
}
