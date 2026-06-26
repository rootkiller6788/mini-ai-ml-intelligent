/*
 * gpu_convolution.c — GPU Convolution: im2col, GEMM, Winograd
 *
 * Knowledge points (L4-L5-L6):
 *   L5: im2col: image-to-column transformation for GEMM-based conv
 *   L5: Direct tiled convolution with shared memory
 *   L5: Winograd F(2×2, 3×3) minimal filtering algorithm
 *   L6: cuDNN-style convolution pipeline with algo selection
 *   L4: Operational intensity & roofline model for conv algorithms
 *
 * References:
 *   Chetlur et al., "cuDNN: Efficient Primitives for Deep Learning" (2014)
 *   Lavin & Gray, "Fast Algorithms for CNNs" (CVPR 2016)
 *   Winograd, "Arithmetic Complexity of Computations" (1980)
 */

#include "gpu_convolution.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* ─── Conv2D Descriptor ─── */

int gpu_conv2d_desc_init(Conv2DDesc *desc,
                          uint32_t N, uint32_t C, uint32_t H, uint32_t W,
                          uint32_t K, uint32_t R, uint32_t S,
                          uint32_t padH, uint32_t padW,
                          uint32_t strideH, uint32_t strideW,
                          uint32_t dilationH, uint32_t dilationW) {
    if (!desc) return 0;
    memset(desc, 0, sizeof(*desc));
    desc->batchSize     = N;
    desc->inputChannels = C;
    desc->inputHeight   = H;
    desc->inputWidth    = W;
    desc->outputChannels = K;
    desc->kernelHeight  = R;
    desc->kernelWidth   = S;
    desc->padH = padH;  desc->padW = padW;
    desc->strideH = strideH;  desc->strideW = strideW;
    desc->dilationH = dilationH ? dilationH : 1;
    desc->dilationW = dilationW ? dilationW : 1;

    /* Output size: P = (H + 2*pad - dil*(R-1) - 1) / stride + 1 */
    int64_t oh = ((int64_t)H + 2 * (int64_t)padH
                 - (int64_t)desc->dilationH * ((int64_t)R - 1) - 1)
                 / (int64_t)strideH + 1;
    int64_t ow = ((int64_t)W + 2 * (int64_t)padW
                 - (int64_t)desc->dilationW * ((int64_t)S - 1) - 1)
                 / (int64_t)strideW + 1;

    if (oh <= 0 || ow <= 0) {
        printf("[CONV2D] ERROR: invalid output size (%lld,%lld)\n", oh, ow);
        return 0;
    }
    desc->outputHeight = (uint32_t)oh;
    desc->outputWidth  = (uint32_t)ow;

    printf("[CONV2D] %u×%u×%u×%u → %u×%u×%u×%u, kernel=%u×%u, "
           "pad=%u/%u, stride=%u/%u, dilation=%u/%u\n",
           N, C, H, W, N, K, desc->outputHeight, desc->outputWidth,
           R, S, padH, padW, strideH, strideW, dilationH, dilationW);
    return 1;
}

/* ─── im2col Transformation ─── */

size_t gpu_im2col_workspace_size(const Conv2DDesc *desc) {
    size_t CRS = (size_t)desc->inputChannels
               * desc->kernelHeight * desc->kernelWidth;
    size_t NPQ = (size_t)desc->batchSize
               * desc->outputHeight * desc->outputWidth;
    return CRS * NPQ;  /* doubles */
}

void gpu_im2col(const double *input, const Conv2DDesc *desc,
                 double *im2colMat) {
    if (!input || !desc || !im2colMat) return;

    uint32_t N = desc->batchSize;
    uint32_t C = desc->inputChannels;
    uint32_t H = desc->inputHeight;
    uint32_t W = desc->inputWidth;
    uint32_t R = desc->kernelHeight;
    uint32_t S = desc->kernelWidth;
    uint32_t P = desc->outputHeight;
    uint32_t Q = desc->outputWidth;
    uint32_t strH = desc->strideH, strW = desc->strideW;
    uint32_t padH = desc->padH, padW = desc->padW;
    uint32_t dilH = desc->dilationH, dilW = desc->dilationW;

    size_t CRS = (size_t)C * R * S;
    size_t NPQ = (size_t)N * P * Q;

    printf("[IM2COL] %u×%u×%u×%u → (%zu × %zu) matrix\n",
           N, C, H, W, CRS, NPQ);

    /* im2col: for each (n, p, q) position, extract the patch
     * of size C×R×S and flatten as a column. Column-major layout:
     * im2colMat[col * CRS + row] = input element. */
    for (uint32_t n = 0; n < N; n++) {
        for (uint32_t p = 0; p < P; p++) {
            for (uint32_t q = 0; q < Q; q++) {
                size_t col = (size_t)n * P * Q + (size_t)p * Q + q;
                for (uint32_t c = 0; c < C; c++) {
                    for (uint32_t r = 0; r < R; r++) {
                        for (uint32_t s_ = 0; s_ < S; s_++) {
                            int hIn = (int)p * (int)strH
                                     + (int)r * (int)dilH - (int)padH;
                            int wIn = (int)q * (int)strW
                                     + (int)s_ * (int)dilW - (int)padW;
                            size_t row = (size_t)c * R * S
                                        + (size_t)r * S + s_;
                            double val = 0.0;
                            if (hIn >= 0 && hIn < (int)H
                                && wIn >= 0 && wIn < (int)W) {
                                size_t inIdx = ((size_t)n * C + c) * H * W
                                             + (size_t)hIn * W + wIn;
                                val = input[inIdx];
                            }
                            im2colMat[col * CRS + row] = val;
                        }
                    }
                }
            }
        }
    }
    printf("[IM2COL] Done: %zu elements written\n", CRS * NPQ);
}

/* ─── col2im (reverse, for gradient) ─── */

void gpu_col2im(const double *im2colMat, const Conv2DDesc *desc,
                 double *inputGrad) {
    if (!im2colMat || !desc || !inputGrad) return;

    uint32_t N = desc->batchSize, C = desc->inputChannels;
    uint32_t H = desc->inputHeight, W = desc->inputWidth;
    uint32_t R = desc->kernelHeight, S = desc->kernelWidth;
    uint32_t P = desc->outputHeight, Q = desc->outputWidth;
    uint32_t strH = desc->strideH, strW = desc->strideW;
    uint32_t padH = desc->padH, padW = desc->padW;
    uint32_t dilH = desc->dilationH, dilW = desc->dilationW;

    size_t CRS = (size_t)C * R * S;

    /* Accumulate columns back to image (used in backward pass for dE/dX) */
    for (uint32_t n = 0; n < N; n++) {
        for (uint32_t p = 0; p < P; p++) {
            for (uint32_t q = 0; q < Q; q++) {
                size_t col = (size_t)n * P * Q + (size_t)p * Q + q;
                for (uint32_t c = 0; c < C; c++) {
                    for (uint32_t r = 0; r < R; r++) {
                        for (uint32_t s_ = 0; s_ < S; s_++) {
                            int hIn = (int)p * (int)strH
                                     + (int)r * (int)dilH - (int)padH;
                            int wIn = (int)q * (int)strW
                                     + (int)s_ * (int)dilW - (int)padW;
                            if (hIn >= 0 && hIn < (int)H
                                && wIn >= 0 && wIn < (int)W) {
                                size_t row = (size_t)c * R * S
                                            + (size_t)r * S + s_;
                                size_t inIdx = ((size_t)n * C + c) * H * W
                                             + (size_t)hIn * W + wIn;
                                inputGrad[inIdx] += im2colMat[col * CRS + row];
                            }
                        }
                    }
                }
            }
        }
    }
    printf("[COL2IM] Done\n");
}

/* ─── GEMM-based Convolution ─── */

/* Naive matrix multiply for the GEMM step */
static void gemm_naive(const double *A, const double *B, double *C,
                       uint32_t M, uint32_t N_geom, uint32_t K_inner) {
    /* C = A × B  where A: M×K, B: K×N, C: M×N */
    for (uint32_t i = 0; i < M; i++) {
        for (uint32_t j = 0; j < N_geom; j++) {
            double sum = 0.0;
            for (uint32_t k = 0; k < K_inner; k++)
                sum += A[i * K_inner + k] * B[k * N_geom + j];
            C[i * N_geom + j] = sum;
        }
    }
}

void gpu_conv2d_im2col_gemm(const double *input, const double *filters,
                              const Conv2DDesc *desc, double *output) {
    if (!input || !filters || !desc || !output) return;

    uint32_t K = desc->outputChannels;
    uint32_t C = desc->inputChannels;
    uint32_t R = desc->kernelHeight, S = desc->kernelWidth;
    uint32_t N = desc->batchSize, P = desc->outputHeight;
    uint32_t Q = desc->outputWidth;

    size_t CRS = (size_t)C * R * S;
    size_t NPQ = (size_t)N * P * Q;

    printf("[CONV-GEMM] im2col + GEMM: K=%u, CRS=%zu, NPQ=%zu\n",
           K, CRS, NPQ);

    /* Step 1: im2col */
    double *im2colMat = (double *)malloc(CRS * NPQ * sizeof(double));
    gpu_im2col(input, desc, im2colMat);

    /* Step 2: GEMM: filters (K × CRS) × im2col (CRS × NPQ) = output (K × NPQ) */
    double *gemmOut = (double *)calloc(K * NPQ, sizeof(double));
    gemm_naive(filters, im2colMat, gemmOut, K, (uint32_t)NPQ, (uint32_t)CRS);

    /* Step 3: reshape from K×NPQ to N×K×P×Q */
    for (uint32_t n = 0; n < N; n++) {
        for (uint32_t k = 0; k < K; k++) {
            for (uint32_t p = 0; p < P; p++) {
                for (uint32_t q = 0; q < Q; q++) {
                    size_t col = (size_t)n * P * Q + (size_t)p * Q + q;
                    size_t outIdx = ((size_t)n * K + k) * P * Q
                                   + (size_t)p * Q + q;
                    output[outIdx] = gemmOut[(size_t)k * NPQ + col];
                }
            }
        }
    }

    printf("[CONV-GEMM] Done: %.2f GFLOPs\n",
           2.0 * (double)K * (double)CRS * (double)NPQ / 1e9);

    free(im2colMat);
    free(gemmOut);
}

/* ─── Direct Tiled Convolution ─── */

void gpu_conv2d_direct_tiled(const double *input, const double *filters,
                               const Conv2DDesc *desc,
                               uint32_t tileH, uint32_t tileW,
                               double *output) {
    if (!input || !filters || !desc || !output) return;

    uint32_t N = desc->batchSize, C = desc->inputChannels;
    uint32_t H = desc->inputHeight, W = desc->inputWidth;
    uint32_t K = desc->outputChannels;
    uint32_t R = desc->kernelHeight, S = desc->kernelWidth;
    uint32_t P = desc->outputHeight, Q = desc->outputWidth;
    uint32_t strH = desc->strideH, strW = desc->strideW;
    uint32_t padH = desc->padH, padW = desc->padW;
    uint32_t dilH = desc->dilationH, dilW = desc->dilationW;

    printf("[CONV-TILED] Direct convolution with tile %u×%u\n", tileH, tileW);

    /* Output-stationary tiling: each tile maps to output[p..p+tileH-1][q..q+tileW-1].
     * Filter loaded to shared memory once per tile.
     * Input window: (p*stride-pad)..(p*stride-pad+(tileH-1)*stride+dil*(R-1)) */
    for (uint32_t n = 0; n < N; n++) {
        for (uint32_t pt = 0; pt < P; pt += tileH) {
            uint32_t pEnd = (pt + tileH < P) ? pt + tileH : P;
            for (uint32_t qt = 0; qt < Q; qt += tileW) {
                uint32_t qEnd = (qt + tileW < Q) ? qt + tileW : Q;

                for (uint32_t p = pt; p < pEnd; p++) {
                    for (uint32_t q = qt; q < qEnd; q++) {
                        for (uint32_t k = 0; k < K; k++) {
                            double sum = 0.0;
                            for (uint32_t c = 0; c < C; c++) {
                                for (uint32_t r = 0; r < R; r++) {
                                    for (uint32_t s_ = 0; s_ < S; s_++) {
                                        int hIn = (int)p * (int)strH
                                                 + (int)r * (int)dilH - (int)padH;
                                        int wIn = (int)q * (int)strW
                                                 + (int)s_ * (int)dilW - (int)padW;
                                        if (hIn < 0 || hIn >= (int)H
                                            || wIn < 0 || wIn >= (int)W)
                                            continue;
                                        size_t inIdx = ((size_t)n * C + c)
                                                      * H * W
                                                      + (size_t)hIn * W + wIn;
                                        size_t fIdx = (((size_t)k * C + c)
                                                       * R + r) * S + s_;
                                        sum += input[inIdx] * filters[fIdx];
                                    }
                                }
                            }
                            size_t outIdx = ((size_t)n * K + k) * P * Q
                                           + (size_t)p * Q + q;
                            output[outIdx] = sum;
                        }
                    }
                }
            }
        }
    }
    printf("[CONV-TILED] Done\n");
}

/* ─── Winograd F(2×2, 3×3) Minimal Filtering ─── */

/* Winograd transform matrices for F(2×2, 3×3):
 *   Bᵀ = [[1, 0,-1, 0],
 *         [0, 1, 1, 0],
 *         [0,-1, 1, 0],
 *         [0, 1, 0,-1]]      → 4×4 for input tiles
 *   G   = [[ 1,   0,   0  ],
 *         [ 1/2, 1/2, 1/2 ],
 *         [ 1/2,-1/2, 1/2 ],
 *         [ 0,   0,   1  ]]  → 4×3 for filter
 *   Aᵀ = [[1, 1, 1, 0],
 *         [0, 1,-1,-1]]      → 2×4 for output inverse transform
 */
static const double Bt[4][4] = {
    { 1.0,  0.0, -1.0,  0.0},
    { 0.0,  1.0,  1.0,  0.0},
    { 0.0, -1.0,  1.0,  0.0},
    { 0.0,  1.0,  0.0, -1.0}
};
static const double G_mat[4][3] = {
    { 1.0,  0.0,  0.0},
    { 0.5,  0.5,  0.5},
    { 0.5, -0.5,  0.5},
    { 0.0,  0.0,  1.0}
};
static const double At[2][4] = {
    {1.0,  1.0,  1.0,  0.0},
    {0.0,  1.0, -1.0, -1.0}
};

/* Transform a 4×4 input tile: V = Bᵀ · d · B */
static void winograd_transform_input(const double tile[4][4],
                                      double V[4][4]) {
    /* temp = tile · B */
    double temp[4][4];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            temp[i][j] = 0.0;
            for (int k = 0; k < 4; k++)
                temp[i][j] += tile[i][k] * Bt[j][k]; /* Bᵀ[k][j] → Bt[j][k]=Bᵀ[j,k] */
        }
    }
    /* V = Bᵀ · temp */
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            V[i][j] = 0.0;
            for (int k = 0; k < 4; k++)
                V[i][j] += Bt[i][k] * temp[k][j];
        }
    }
}

/* Transform a 3×3 filter: U = G · g · Gᵀ */
static void winograd_transform_filter(const double filt[3][3],
                                       double U[4][4]) {
    /* temp = g · Gᵀ */
    double temp[3][4];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            temp[i][j] = 0.0;
            for (int k = 0; k < 3; k++)
                temp[i][j] += filt[i][k] * G_mat[j][k];
        }
    }
    /* U = G · temp */
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            U[i][j] = 0.0;
            for (int k = 0; k < 3; k++)
                U[i][j] += G_mat[i][k] * temp[k][j];
        }
    }
}

/* Inverse transform for output: Y = Aᵀ · M · A  where M = Hadamard(U, V) */
static void winograd_inverse_output(const double M[4][4],
                                     double Y[2][2]) {
    /* temp = M · A */
    double temp[4][2];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 2; j++) {
            temp[i][j] = 0.0;
            for (int k = 0; k < 4; k++)
                temp[i][j] += M[i][k] * At[j][k];
        }
    }
    /* Y = Aᵀ · temp */
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            Y[i][j] = 0.0;
            for (int k = 0; k < 4; k++)
                Y[i][j] += At[i][k] * temp[k][j];
        }
    }
}

uint32_t gpu_winograd_num_tiles(const Conv2DDesc *desc) {
    /* F(2×2, 3×3): each tile produces 2×2 output, consumes 4×4 input */
    uint32_t tilesH = desc->outputHeight / 2;
    uint32_t tilesW = desc->outputWidth / 2;
    return tilesH * tilesW;
}

void gpu_winograd_f2x2_3x3(const double *input, const double *filters,
                             const Conv2DDesc *desc, double *output) {
    if (!input || !filters || !desc || !output) return;
    if (desc->kernelHeight != 3 || desc->kernelWidth != 3) {
        printf("[WINOGRAD] ERROR: F(2×2,3×3) requires 3×3 kernel, got %u×%u\n",
               desc->kernelHeight, desc->kernelWidth);
        return;
    }

    uint32_t N = desc->batchSize, C = desc->inputChannels, K = desc->outputChannels;
    uint32_t H = desc->inputHeight, W = desc->inputWidth;
    uint32_t P = desc->outputHeight, Q = desc->outputWidth;
    uint32_t padH = desc->padH, padW = desc->padW;

    /* Number of 4×4 tiles = ceil(P/2), ceil(Q/2).
     * For simplicity, handle only sizes that are multiples of 2. */
    uint32_t tilesH = P / 2;
    uint32_t tilesW = Q / 2;
    if (tilesH == 0 || tilesW == 0) {
        printf("[WINOGRAD] ERROR: output too small for F(2×2,3×3)\n");
        return;
    }

    printf("[WINOGRAD] F(2×2,3×3): %u×%u input, %u×%u output, "
           "%u×%u tiles\n", H, W, P, Q, tilesH, tilesW);

    /* For each filter k and each input channel c: transform filter */
    /* U[c][k]: 4×4 transformed filter for channel c, output k */
    double (*U)[4][4] = (double(*)[4][4])malloc(
        (size_t)C * K * 4 * 4 * sizeof(double));

    for (uint32_t k = 0; k < K; k++) {
        for (uint32_t c = 0; c < C; c++) {
            double filt3x3[3][3];
            for (int r = 0; r < 3; r++)
                for (int s_ = 0; s_ < 3; s_++)
                    filt3x3[r][s_] = filters[(((size_t)k * C + c) * 3 + r) * 3 + s_];
            winograd_transform_filter(filt3x3,
                                      U[(size_t)c * K + k]);
        }
    }

    /* For each batch, for each tile: transform input, multiply, inverse */
    for (uint32_t n = 0; n < N; n++) {
        for (uint32_t th = 0; th < tilesH; th++) {
            for (uint32_t tw = 0; tw < tilesW; tw++) {
                /* Input tile: 4×4 patch starting at (th*2 - pad, tw*2 - pad) */
                int hStart = (int)(th * 2) - (int)padH;
                int wStart = (int)(tw * 2) - (int)padW;

                for (uint32_t k = 0; k < K; k++) {
                    /* Accumulate over channels */
                    double M[4][4] = {{0}};

                    for (uint32_t c = 0; c < C; c++) {
                        /* Extract 4×4 input tile */
                        double tile4x4[4][4];
                        for (int r = 0; r < 4; r++) {
                            for (int s_ = 0; s_ < 4; s_++) {
                                int hIn = hStart + r;
                                int wIn = wStart + s_;
                                if (hIn >= 0 && hIn < (int)H
                                    && wIn >= 0 && wIn < (int)W) {
                                    size_t inIdx = ((size_t)n * C + c)
                                                  * H * W
                                                  + (size_t)hIn * W + wIn;
                                    tile4x4[r][s_] = input[inIdx];
                                } else {
                                    tile4x4[r][s_] = 0.0;
                                }
                            }
                        }

                        /* V = Bᵀ · tile · B */
                        double V[4][4];
                        winograd_transform_input(tile4x4, V);

                        /* Hadamard product: M += U[c][k] ⊙ V */
                        double (*Uk)[4] = U[(size_t)c * K + k];
                        for (int r = 0; r < 4; r++)
                            for (int s_ = 0; s_ < 4; s_++)
                                M[r][s_] += Uk[r][s_] * V[r][s_];
                    }

                    /* Inverse transform: Y = Aᵀ · M · A */
                    double Y[2][2];
                    winograd_inverse_output(M, Y);

                    /* Store to output[2×2] at position (th*2, tw*2) */
                    for (int r = 0; r < 2; r++) {
                        for (int s_ = 0; s_ < 2; s_++) {
                            uint32_t pOut = th * 2 + (uint32_t)r;
                            uint32_t qOut = tw * 2 + (uint32_t)s_;
                            if (pOut < P && qOut < Q) {
                                size_t outIdx = ((size_t)n * K + k) * P * Q
                                               + (size_t)pOut * Q + qOut;
                                output[outIdx] = Y[r][s_];
                            }
                        }
                    }
                }
            }
        }
    }

    printf("[WINOGRAD] Done. Arithmetic complexity reduction: "
           "F(2×2,3×3) uses 4×4=16 multiplies vs 2×2×3×3=36 naively "
           "(×%.1f reduction)\n", 36.0 / 16.0);

    free(U);
}

/* ─── Performance Estimation ─── */

ConvPerfEstimate gpu_conv2d_perf_estimate(const Conv2DDesc *desc,
                                            ConvAlgorithm algo,
                                            double peakTflops,
                                            double memBandwidthGBps) {
    ConvPerfEstimate est;
    memset(&est, 0, sizeof(est));
    est.algo = algo;

    if (!desc) return est;

    uint32_t K = desc->outputChannels, C = desc->inputChannels;
    uint32_t R = desc->kernelHeight, S = desc->kernelWidth;
    uint32_t N = desc->batchSize, P = desc->outputHeight, Q = desc->outputWidth;

    switch (algo) {
    case CONV_ALGO_DIRECT:
        est.totalFlops = 2.0 * (double)N * (double)K * (double)C
                        * (double)R * (double)S * (double)P * (double)Q;
        est.memoryBytes = (double)N * ((double)C * (double)desc->inputHeight
                         * desc->inputWidth * sizeof(double)
                         + (double)K * P * Q * sizeof(double));
        break;
    case CONV_ALGO_IM2COL:
        est.totalFlops = 2.0 * (double)K * (double)C * (double)R * (double)S
                        * (double)N * (double)P * (double)Q;
        est.memoryBytes = (double)C * R * S * N * P * Q * sizeof(double)
                         + (double)K * C * R * S * sizeof(double);
        break;
    case CONV_ALGO_WINOGRAD:
        /* F(2×2,3×3): each of ceil(P/2)·ceil(Q/2) tiles: 4×4 Hadamard per channel */
        est.totalFlops = 2.0 * (double)N * (double)K * (double)C
                        * 4.0 * 4.0 * (double)gpu_winograd_num_tiles(desc);
        est.memoryBytes = (double)N * (double)C * desc->inputHeight
                         * desc->inputWidth * sizeof(double) * 0.5;
        break;
    default:
        break;
    }

    est.operationalIntensity = (est.memoryBytes > 0.0)
                                ? est.totalFlops / est.memoryBytes : 0.0;

    /* Execution time = max(computeTime, memoryTime) on GPU */
    double computeTimeMs = (peakTflops > 0.0)
                           ? est.totalFlops / (peakTflops * 1e12) * 1000.0
                           : 1e9;
    double memoryTimeMs = (memBandwidthGBps > 0.0)
                          ? est.memoryBytes / (memBandwidthGBps * 1e9) * 1000.0
                          : 1e9;
    est.estimatedTimeMs = (computeTimeMs > memoryTimeMs)
                           ? computeTimeMs : memoryTimeMs;

    est.tflops = est.totalFlops / (est.estimatedTimeMs / 1000.0) / 1e12;

    const char *algoNames[] = {"DIRECT", "IM2COL", "WINOGRAD", "FFT"};
    printf("[PERF] %s: %.1f GFLOPs, %.1f MB, OI=%.2f FLOP/byte, "
           "t=%.3f ms, %.2f TFLOPS\n",
           algoNames[algo], est.totalFlops / 1e9,
           est.memoryBytes / 1e6, est.operationalIntensity,
           est.estimatedTimeMs, est.tflops);
    return est;
}

/* ─── Algorithm Recommendation ─── */

ConvAlgorithm gpu_conv2d_recommend_algo(const Conv2DDesc *desc,
                                          double peakTflops,
                                          double memBandwidthGBps) {
    if (!desc) return CONV_ALGO_DIRECT;

    uint32_t R = desc->kernelHeight, S = desc->kernelWidth;
    uint32_t P = desc->outputHeight;

    /* Heuristic selection:
     *   - Small kernels (3×3): Winograd F(2×2,3×3) is most efficient
     *   - Medium kernels (5×5, 7×7): im2col+GEMM (cuBLAS is optimized)
     *   - Large kernels (>7×7): im2col+GEMM or FFT
     *   - 1×1 kernels: direct (converts to matmul naturally) */
    ConvAlgorithm best;

    if (R == 3 && S == 3 && P >= 2
        && desc->strideH == 1 && desc->strideW == 1
        && desc->dilationH == 1 && desc->dilationW == 1) {
        best = CONV_ALGO_WINOGRAD;
    } else if (R * S >= 25) {
        best = CONV_ALGO_IM2COL;
    } else {
        /* For small maps, direct can beat im2col (no extra memory) */
        if (desc->outputHeight * desc->outputWidth <= 64)
            best = CONV_ALGO_DIRECT;
        else
            best = CONV_ALGO_IM2COL;
    }

    /* Estimate all three and pick the fastest */
    ConvPerfEstimate estBest = gpu_conv2d_perf_estimate(
        desc, best, peakTflops, memBandwidthGBps);

    static const ConvAlgorithm allAlgos[] = {
        CONV_ALGO_DIRECT, CONV_ALGO_IM2COL, CONV_ALGO_WINOGRAD
    };
    for (int i = 0; i < 3; i++) {
        if (allAlgos[i] == best) continue;
        ConvPerfEstimate est = gpu_conv2d_perf_estimate(
            desc, allAlgos[i], peakTflops, memBandwidthGBps);
        if (est.estimatedTimeMs < estBest.estimatedTimeMs) {
            best = allAlgos[i];
            estBest = est;
        }
    }

    const char *algoNames[] = {"DIRECT", "IM2COL", "WINOGRAD", "FFT"};
    printf("[RECOMMEND] Best algorithm for %u×%u kernel: %s (%.3f ms)\n",
           R, S, algoNames[best], estBest.estimatedTimeMs);
    return best;
}
