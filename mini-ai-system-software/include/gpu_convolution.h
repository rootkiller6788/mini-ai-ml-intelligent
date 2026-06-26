#ifndef GPU_CONVOLUTION_H
#define GPU_CONVOLUTION_H

#include <stdint.h>
#include <stddef.h>

/* ───────────────────────────────────────────────────────────────
 * GPU Convolution — im2col, GEMM, Winograd
 *
 * L5 — Algorithms/Methods:
 *   im2col: image-to-column transformation for GEMM-based conv
 *   Winograd F(2×2, 3×3) minimal filtering algorithm
 *   Direct convolution with tiling in shared memory
 *
 * L6 — Canonical Problems:
 *   cuDNN-style convolution pipeline
 *
 * L4 — Operational intensity analysis (roofline model)
 * ─────────────────────────────────────────────────────────────── */

/* --- Convolution descriptor (2D) --- */
typedef struct {
    uint32_t batchSize;       /* N */
    uint32_t inputChannels;   /* C */
    uint32_t inputHeight;     /* H */
    uint32_t inputWidth;      /* W */
    uint32_t outputChannels;  /* K */
    uint32_t kernelHeight;    /* R */
    uint32_t kernelWidth;     /* S */
    uint32_t padH, padW;      /* padding */
    uint32_t strideH, strideW; /* stride */
    uint32_t dilationH, dilationW; /* dilation */
    uint32_t outputHeight;    /* P = (H + 2*pad - R) / stride + 1 */
    uint32_t outputWidth;     /* Q */
} Conv2DDesc;

/* --- Convolution algorithm type --- */
typedef enum {
    CONV_ALGO_DIRECT    = 0,  /* direct nested-loop convolution */
    CONV_ALGO_IM2COL    = 1,  /* im2col + GEMM */
    CONV_ALGO_WINOGRAD  = 2,  /* Winograd F(m×m, r×r) */
    CONV_ALGO_FFT       = 3   /* FFT-based (large filters) */
} ConvAlgorithm;

/* --- Performance estimate --- */
typedef struct {
    ConvAlgorithm algo;
    double totalFlops;        /* total floating-point operations */
    double memoryBytes;       /* data moved (reads + writes) */
    double operationalIntensity; /* FLOP/byte = totalFlops / memoryBytes */
    double estimatedTimeMs;   /* estimated execution time */
    double tflops;            /* effective TFLOPS */
} ConvPerfEstimate;

/* ─── Descriptor API ─── */

/* Compute output dimensions and validate convolution descriptor.
 * Returns 1 if valid, 0 otherwise. */
int gpu_conv2d_desc_init(Conv2DDesc *desc,
                          uint32_t N, uint32_t C, uint32_t H, uint32_t W,
                          uint32_t K, uint32_t R, uint32_t S,
                          uint32_t padH, uint32_t padW,
                          uint32_t strideH, uint32_t strideW,
                          uint32_t dilationH, uint32_t dilationW);

/* ─── im2col ─── */

/* Image-to-column transformation.
 * Extracts patches from input (N×C×H×W) into a matrix (CRS × NPQ).
 * Each column = one filter position, unrolled into 1D.
 * im2colMat: output, size = (C*R*S) × (N*P*Q), column-major. */
void gpu_im2col(const double *input, const Conv2DDesc *desc,
                 double *im2colMat);

/* Reverse: col2im — used for gradient w.r.t. input in backward pass.
 * Accumulates columns back into image tensor. */
void gpu_col2im(const double *im2colMat, const Conv2DDesc *desc,
                 double *inputGrad);

/* ─── GEMM-based Convolution ─── */

/* Convolution via im2col + GEMM:
 *   filters: K × (C*R*S) matrix, row-major
 *   im2col(input): (C*R*S) × (N*P*Q) matrix, col-major
 *   output = filters × im2col → K × (N*P*Q), reshaped to (N×K×P×Q) */
void gpu_conv2d_im2col_gemm(const double *input, const double *filters,
                              const Conv2DDesc *desc, double *output);

/* ─── Direct Convolution (Tiled) ─── */

/* Direct 2D convolution with output-stationary tiling.
 * Each thread block computes one output tile.
 * Filter loaded into shared memory, input tile loaded cooperatively. */
void gpu_conv2d_direct_tiled(const double *input, const double *filters,
                               const Conv2DDesc *desc,
                               uint32_t tileH, uint32_t tileW,
                               double *output);

/* ─── Winograd Minimal Filtering ─── */

/* Winograd F(2×2, 3×3) forward transform.
 * Aᵀ, G, Bᵀ are the Winograd transform matrices.
 * Image transform: U = G·g·Gᵀ for each filter.
 * Input transform: V = Bᵀ·d·B for each input tile.
 * Output: Y = Aᵀ·[U⊙V]·A (element-wise multiply + inverse transform).
 *
 * Reference: Lavin & Gray, "Fast Algorithms for Convolutional Neural
 * Networks", CVPR 2016. */
void gpu_winograd_f2x2_3x3(const double *input, const double *filters,
                             const Conv2DDesc *desc, double *output);

/* ─── Performance Modeling ─── */

/* Estimate FLOPs, memory traffic, and operational intensity.
 * Direct: 2·K·C·R·S·P·Q FLOPs.
 * im2col+GEMM: 2·K·(C·R·S)·(N·P·Q) FLOPs.
 * Winograd F(2×2,3×3): 2·K·C·(tile_count)·(4×4) ops. */
ConvPerfEstimate gpu_conv2d_perf_estimate(const Conv2DDesc *desc,
                                            ConvAlgorithm algo,
                                            double peakTflops,
                                            double memBandwidthGBps);

/* Compare all algorithms and recommend best for given dimensions */
ConvAlgorithm gpu_conv2d_recommend_algo(const Conv2DDesc *desc,
                                          double peakTflops,
                                          double memBandwidthGBps);

/* ─── Utility ─── */

/* Allocate workspace for im2col matrix.
 * Returns size in doubles. */
size_t gpu_im2col_workspace_size(const Conv2DDesc *desc);

/* Number of Winograd tiles: each tile covers 4×4 input, 2×2 output */
uint32_t gpu_winograd_num_tiles(const Conv2DDesc *desc);

#endif /* GPU_CONVOLUTION_H */
