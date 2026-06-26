#ifndef NORMALIZATION_H
#define NORMALIZATION_H

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ── L1: Core Definitions ──
 * Normalization layers for stable neural network training.
 * References:
 *   - Ioffe & Szegedy, "Batch Normalization" (2015)
 *   - Ba, Kiros & Hinton, "Layer Normalization" (2016)
 *   - Zhang & Sennrich, "RMSNorm" (2019) — used in LLaMA
 *   - Wu & He, "Group Normalization" (2018)
 */

/* ── L1: BatchNorm1D ── */
typedef struct {
    int    num_features;
    float  eps;
    float  momentum;         /* running mean/var momentum */
    float *gamma;
    float *beta;
    float *running_mean;
    float *running_var;
} BatchNorm1D;

/* ── L1: BatchNorm2D (for CNN feature maps: NCHW) ── */
typedef struct {
    int    num_features;     /* = channels */
    float  eps;
    float  momentum;
    float *gamma;
    float *beta;
    float *running_mean;
    float *running_var;
} BatchNorm2D;

/* ── L8: RMSNorm — used in LLaMA, Gemma, Mistral ── */
typedef struct {
    int    dim;
    float  eps;
    float *scale;            /* single learnable vector (no bias) */
} RMSNorm;

/* ── L8: GroupNorm — used in Stable Diffusion ── */
typedef struct {
    int    num_groups;
    int    num_channels;
    float  eps;
    float *gamma;
    float *beta;
} GroupNorm;

/* ── L2: Core API ── */

/* BatchNorm1D */
BatchNorm1D *bn1d_create(int features, float momentum);
void         bn1d_free(BatchNorm1D *b);
void         bn1d_forward_train(BatchNorm1D *b, float *x, int batch_size);
void         bn1d_forward_infer(const BatchNorm1D *b, float *x, int batch_size);

/* BatchNorm2D */
BatchNorm2D *bn2d_create(int channels, float momentum);
void         bn2d_free(BatchNorm2D *b);
void         bn2d_forward_train(BatchNorm2D *b, float *x, int N, int H, int W);
void         bn2d_forward_infer(const BatchNorm2D *b, float *x, int N, int H, int W);

/* RMSNorm */
RMSNorm *rmsnorm_create(int dim);
void     rmsnorm_free(RMSNorm *r);
void     rmsnorm_forward(RMSNorm *r, float *x, int batch, int dim);

/* GroupNorm */
GroupNorm *gn_create(int num_groups, int num_channels);
void       gn_free(GroupNorm *g);
void       gn_forward(GroupNorm *g, float *x, int N, int H, int W);

/* ── L5: Input/Layer statistics utilities ── */
float feature_mean(const float *x, int n);
float feature_var(const float *x, int n, float mean);

#endif
