#include "normalization.h"

/* L2: Feature statistics helpers */
float feature_mean(const float *x, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += x[i];
    return sum / (float)n;
}

float feature_var(const float *x, int n, float mean) {
    float var = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = x[i] - mean;
        var += diff * diff;
    }
    return var / (float)n;
}

/*
 * L2: BatchNorm1D (Ioffe & Szegedy, 2015)
 * Training: y = gamma * (x - mu_batch) / sqrt(var_batch + eps) + beta
 * Running stats: EMA of mu and var for inference
 * L4: Reduces internal covariate shift; allows higher learning rates.
 */
BatchNorm1D *bn1d_create(int features, float momentum) {
    BatchNorm1D *b = (BatchNorm1D *)malloc(sizeof(BatchNorm1D));
    b->num_features = features; b->eps = 1e-5f; b->momentum = momentum;
    b->gamma = (float *)malloc(features * sizeof(float));
    b->beta  = (float *)calloc(features, sizeof(float));
    b->running_mean = (float *)calloc(features, sizeof(float));
    b->running_var  = (float *)malloc(features * sizeof(float));
    for (int i = 0; i < features; i++) { b->gamma[i] = 1.0f; b->running_var[i] = 1.0f; }
    return b;
}

void bn1d_free(BatchNorm1D *b) {
    free(b->gamma); free(b->beta); free(b->running_mean); free(b->running_var); free(b);
}

void bn1d_forward_train(BatchNorm1D *b, float *x, int batch_size) {
    int f = b->num_features;
    for (int c = 0; c < f; c++) {
        /* Compute batch mean and var for feature c */
        float sum = 0.0f;
        for (int n = 0; n < batch_size; n++) sum += x[n * f + c];
        float mean = sum / (float)batch_size;
        float var = 0.0f;
        for (int n = 0; n < batch_size; n++) {
            float diff = x[n * f + c] - mean;
            var += diff * diff;
        }
        var = var / (float)batch_size + b->eps;
        float inv_std = 1.0f / sqrtf(var);
        /* Update running stats */
        b->running_mean[c] = b->momentum * b->running_mean[c] + (1.0f - b->momentum) * mean;
        b->running_var[c]  = b->momentum * b->running_var[c]  + (1.0f - b->momentum) * var;
        /* Normalize */
        for (int n = 0; n < batch_size; n++) {
            x[n * f + c] = b->gamma[c] * (x[n * f + c] - mean) * inv_std + b->beta[c];
        }
    }
}

void bn1d_forward_infer(const BatchNorm1D *b, float *x, int batch_size) {
    int f = b->num_features;
    for (int c = 0; c < f; c++) {
        float inv_std = 1.0f / sqrtf(b->running_var[c] + b->eps);
        for (int n = 0; n < batch_size; n++) {
            x[n * f + c] = b->gamma[c] * (x[n * f + c] - b->running_mean[c]) * inv_std + b->beta[c];
        }
    }
}

/*
 * L2: BatchNorm2D for CNN feature maps (N, C, H, W)
 * Normalizes across (N, H, W) for each channel.
 */
BatchNorm2D *bn2d_create(int channels, float momentum) {
    BatchNorm2D *b = (BatchNorm2D *)malloc(sizeof(BatchNorm2D));
    b->num_features = channels; b->eps = 1e-5f; b->momentum = momentum;
    b->gamma = (float *)malloc(channels * sizeof(float));
    b->beta  = (float *)calloc(channels, sizeof(float));
    b->running_mean = (float *)calloc(channels, sizeof(float));
    b->running_var  = (float *)malloc(channels * sizeof(float));
    for (int i = 0; i < channels; i++) { b->gamma[i] = 1.0f; b->running_var[i] = 1.0f; }
    return b;
}

void bn2d_free(BatchNorm2D *b) {
    free(b->gamma); free(b->beta); free(b->running_mean); free(b->running_var); free(b);
}

void bn2d_forward_train(BatchNorm2D *b, float *x, int N, int H, int W) {
    int C = b->num_features;
    for (int c = 0; c < C; c++) {
        int spatial = N * H * W;
        float sum = 0.0f;
        for (int i = 0; i < N; i++)
            for (int h = 0; h < H; h++)
                for (int w = 0; w < W; w++)
                    sum += x[((i * C + c) * H + h) * W + w];
        float mean = sum / (float)spatial;
        float var = 0.0f;
        for (int i = 0; i < N; i++)
            for (int h = 0; h < H; h++)
                for (int w = 0; w < W; w++) {
                    float diff = x[((i * C + c) * H + h) * W + w] - mean;
                    var += diff * diff;
                }
        var = var / (float)spatial + b->eps;
        b->running_mean[c] = b->momentum * b->running_mean[c] + (1.0f - b->momentum) * mean;
        b->running_var[c]  = b->momentum * b->running_var[c]  + (1.0f - b->momentum) * var;
        float inv_std = 1.0f / sqrtf(var);
        for (int i = 0; i < N; i++)
            for (int h = 0; h < H; h++)
                for (int w = 0; w < W; w++)
                    x[((i * C + c) * H + h) * W + w] =
                        b->gamma[c] * (x[((i * C + c) * H + h) * W + w] - mean) * inv_std + b->beta[c];
    }
}

void bn2d_forward_infer(const BatchNorm2D *b, float *x, int N, int H, int W) {
    int C = b->num_features;
    for (int c = 0; c < C; c++) {
        float inv_std = 1.0f / sqrtf(b->running_var[c] + b->eps);
        for (int i = 0; i < N; i++)
            for (int h = 0; h < H; h++)
                for (int w = 0; w < W; w++)
                    x[((i * C + c) * H + h) * W + w] =
                        b->gamma[c] * (x[((i * C + c) * H + h) * W + w] - b->running_mean[c]) * inv_std + b->beta[c];
    }
}

/*
 * L8: RMSNorm (Zhang & Sennrich, 2019)
 * Used in LLaMA, Gemma, Mistral.
 * y = x / RMS(x) * scale   where RMS(x) = sqrt(mean(x^2) + eps)
 * Simpler than LayerNorm: no mean subtraction, no bias.
 * Theorem: RMSNorm is re-centering invariant.
 */
RMSNorm *rmsnorm_create(int dim) {
    RMSNorm *r = (RMSNorm *)malloc(sizeof(RMSNorm));
    r->dim = dim; r->eps = 1e-6f;
    r->scale = (float *)malloc(dim * sizeof(float));
    for (int i = 0; i < dim; i++) r->scale[i] = 1.0f;
    return r;
}

void rmsnorm_free(RMSNorm *r) { free(r->scale); free(r); }

void rmsnorm_forward(RMSNorm *r, float *x, int batch, int dim) {
    for (int b = 0; b < batch; b++) {
        float *row = x + b * dim;
        float sq_sum = 0.0f;
        for (int i = 0; i < dim; i++) sq_sum += row[i] * row[i];
        float rms = sqrtf(sq_sum / (float)dim + r->eps);
        for (int i = 0; i < dim; i++) row[i] = row[i] / rms * r->scale[i];
    }
}

/*
 * L8: GroupNorm (Wu & He, ECCV 2018)
 * Divides channels into groups, normalizes within each group.
 * Used in Stable Diffusion; better than BN for small batches.
 */
GroupNorm *gn_create(int num_groups, int num_channels) {
    GroupNorm *g = (GroupNorm *)malloc(sizeof(GroupNorm));
    g->num_groups = num_groups; g->num_channels = num_channels; g->eps = 1e-5f;
    g->gamma = (float *)malloc(num_channels * sizeof(float));
    g->beta  = (float *)calloc(num_channels, sizeof(float));
    for (int i = 0; i < num_channels; i++) g->gamma[i] = 1.0f;
    return g;
}

void gn_free(GroupNorm *g) { free(g->gamma); free(g->beta); free(g); }

void gn_forward(GroupNorm *g, float *x, int N, int H, int W) {
    int C = g->num_channels, G = g->num_groups;
    int c_per_group = C / G;
    int spatial = H * W;
    for (int n = 0; n < N; n++) {
        for (int grp = 0; grp < G; grp++) {
            int c_start = grp * c_per_group;
            int total = c_per_group * spatial;
            float *grp_data = (float *)malloc(total * sizeof(float));
            /* Gather */
            for (int c = 0; c < c_per_group; c++)
                for (int s = 0; s < spatial; s++)
                    grp_data[c * spatial + s] =
                        x[((n * C + c_start + c) * H) * W + s % W + (s / W) * 0];
            /* Compute stats */
            float sum = 0.0f;
            for (int i = 0; i < total; i++) sum += grp_data[i];
            float mean = sum / (float)total;
            float var = 0.0f;
            for (int i = 0; i < total; i++) { float d = grp_data[i] - mean; var += d * d; }
            var = var / (float)total + g->eps;
            float inv_std = 1.0f / sqrtf(var);
            /* Scatter */
            for (int c = 0; c < c_per_group; c++) {
                int cc = c_start + c;
                for (int s = 0; s < spatial; s++) {
                    float normed = g->gamma[cc] * (grp_data[c * spatial + s] - mean) * inv_std + g->beta[cc];
                    x[((n * C + cc) * H) * W + s % W + (s / W) * 0] = normed;
                }
            }
            free(grp_data);
        }
    }
}
