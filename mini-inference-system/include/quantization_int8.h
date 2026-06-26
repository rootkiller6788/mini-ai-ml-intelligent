#ifndef QUANTIZATION_INT8_H
#define QUANTIZATION_INT8_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#define QI8_MAX_CALIB_SAMPLES   1024
#define QI8_MAX_CHANNELS        4096
#define QI8_MAX_GPTQ_COLUMNS    8192
#define QI8_MAX_AWQ_SAMPLES     128

typedef enum {
    QI8_PER_TENSOR   = 0,
    QI8_PER_CHANNEL  = 1,
    QI8_PER_GROUP    = 2,
} QI8_Granularity;

typedef enum {
    QI8_SYMMETRIC     = 0,
    QI8_ASYMMETRIC    = 1,
} QI8_QuantMode;

typedef struct {
    float    scale;
    int8_t   zero_point;
    float    min_val;
    float    max_val;
} QI8_TensorQuantParams;

typedef struct {
    QI8_TensorQuantParams* channel_params;
    int      num_channels;
    QI8_Granularity granularity;
} QI8_QuantParams;

typedef struct {
    float*   data;
    int      rows;
    int      cols;
    int      group_size;
    bool     is_quantized;
} QI8_WeightMatrix;

typedef struct {
    float*   calibration_buffer;
    int      num_samples;
    int      sample_capacity;
    int      tensor_size;
    float*   running_min;
    float*   running_max;
    float*   running_mean;
    float*   running_m2;
    int      count;
} QI8_CalibrationData;

typedef struct {
    QI8_WeightMatrix* weights;
    int      num_layers;
    float*   hessian_diag;
    int      hessian_samples;
    int      blocksize;
    bool     percdamp;
} QI8_GPTQContext;

typedef struct {
    QI8_WeightMatrix* weights;
    int      num_layers;
    float*   activation_stats;
    int      num_samples;
    float    alpha;
    float    beta;
    int      group_size;
} QI8_AWQContext;

void  qi8_calib_init(QI8_CalibrationData* calib, int tensor_size, int max_samples);
void  qi8_calib_destroy(QI8_CalibrationData* calib);
void  qi8_calib_collect(QI8_CalibrationData* calib, const float* data, int size);
void  qi8_calib_collect_running(QI8_CalibrationData* calib, const float* data, int size);

QI8_TensorQuantParams qi8_calib_minmax(const float* data, int size);
QI8_TensorQuantParams qi8_calib_mse(const float* data, int size, int num_bins);
QI8_TensorQuantParams qi8_calib_percentile(const float* data, int size, float percentile);
QI8_TensorQuantParams qi8_calib_entropy(const float* data, int size);

void  qi8_symmetric_params(const float* data, int size, float* scale);
void  qi8_asymmetric_params(const float* data, int size, float* scale, int8_t* zero_point);

void  qi8_quantize_per_tensor(const float* src, int8_t* dst, int size, QI8_TensorQuantParams params);
void  qi8_dequantize_per_tensor(const int8_t* src, float* dst, int size, QI8_TensorQuantParams params);

void  qi8_quantize_per_channel(const float* src, int8_t* dst, int rows, int cols,
                                const QI8_TensorQuantParams* channel_params);
void  qi8_dequantize_per_channel(const int8_t* src, float* dst, int rows, int cols,
                                  const QI8_TensorQuantParams* channel_params);

void  qi8_requantize_int32_to_int8(const int32_t* src, int8_t* dst, int size,
                                    float scale_a, float scale_b, float scale_out);

void  qi8_gemm_int8(const int8_t* A, const int8_t* B, int32_t* C,
                     int M, int N, int K,
                     float scale_A, float scale_B, float scale_C,
                     const int32_t* bias);

void  qi8_gemm_int8_per_channel(const int8_t* A, const int8_t* B, int32_t* C,
                                 int M, int N, int K,
                                 const QI8_TensorQuantParams* scale_A_ch,
                                 const QI8_TensorQuantParams* scale_B_ch,
                                 float scale_C);

void  qi8_fused_quant_relu(const float* src, int8_t* dst, int size,
                            QI8_TensorQuantParams params);
void  qi8_fused_quant_gelu(const float* src, int8_t* dst, int size,
                            QI8_TensorQuantParams params);
void  qi8_fused_quant_add(const float* a, const float* b, int8_t* dst, int size,
                           QI8_TensorQuantParams params);

void  qi8_weight_only_quant(const float* weights, int8_t* quant_weights,
                             int rows, int cols, int group_size,
                             QI8_TensorQuantParams* group_params);
void  qi8_weight_only_dequant(const int8_t* quant_weights, float* weights,
                               int rows, int cols, int group_size,
                               const QI8_TensorQuantParams* group_params);

void  qi8_gptq_init(QI8_GPTQContext* ctx, int num_layers, int hessian_samples, int blocksize);
void  qi8_gptq_destroy(QI8_GPTQContext* ctx);
void  qi8_gptq_collect_hessian(QI8_GPTQContext* ctx, const float* activations, int layer, int size);
void  qi8_gptq_quantize_layer(QI8_GPTQContext* ctx, int layer, int group_size);
void  qi8_gptq_quantize_all(QI8_GPTQContext* ctx, int group_size);

void  qi8_awq_init(QI8_AWQContext* ctx, int num_layers, int num_samples);
void  qi8_awq_destroy(QI8_AWQContext* ctx);
void  qi8_awq_collect_activation(QI8_AWQContext* ctx, const float* activations, int layer, int size);
void  qi8_awq_search_scale(QI8_AWQContext* ctx, int layer, float alpha, float beta, int grid);
void  qi8_awq_quantize(QI8_AWQContext* ctx, int layer, int group_size);
void  qi8_awq_quantize_all(QI8_AWQContext* ctx, int group_size);

float qi8_measure_mse(const float* original, const float* dequant, int size);
float qi8_measure_cosine(const float* original, const float* dequant, int size);
float qi8_measure_snr(const float* original, const float* dequant, int size);

#endif
