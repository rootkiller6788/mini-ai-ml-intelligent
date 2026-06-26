#include "activations.h"

/*
 * L2: GELU ！ Gaussian Error Linear Unit (Hendrycks & Gimpel, 2016)
 * gelu(x) = x * Phi(x) where Phi is standard normal CDF.
 * Approximate: 0.5 * x * (1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
 * Used in BERT, GPT-2, ViT.
 */
float gelu(float x) {
    float c = sqrtf(2.0f / 3.14159265f);
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x3)));
}

void gelu_forward(const float *x, float *out, int n) {
    float c = sqrtf(2.0f / 3.14159265f);
    for (int i = 0; i < n; i++) {
        float xi = x[i];
        out[i] = 0.5f * xi * (1.0f + tanhf(c * (xi + 0.044715f * xi * xi * xi)));
    }
}

/*
 * L2: SiLU / Swish (Ramachandran et al., 2017; Elfwing et al., 2018)
 * silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
 * Self-gated; smooth, non-monotonic. Used in LLaMA, PaLM.
 */
float silu(float x) {
    return x / (1.0f + expf(-x));
}

void silu_forward(const float *x, float *out, int n) {
    for (int i = 0; i < n; i++)
        out[i] = x[i] / (1.0f + expf(-x[i]));
}

/*
 * L2: LeakyReLU ！ prevents dying ReLU neurons
 * leaky_relu(x, alpha) = x if x > 0 else alpha*x
 * Typical alpha = 0.01
 */
float leaky_relu(float x, float alpha) {
    return (x > 0.0f) ? x : alpha * x;
}

void leaky_relu_forward(const float *x, float *out, int n, float alpha) {
    for (int i = 0; i < n; i++)
        out[i] = (x[i] > 0.0f) ? x[i] : alpha * x[i];
}

/*
 * L2: ELU ！ Exponential Linear Unit (Clevert et al., 2015)
 * elu(x, alpha) = x                 if x > 0
 *                 alpha*(exp(x)-1)  if x <= 0
 * Negative saturation for noise-robustness.
 */
float elu(float x, float alpha) {
    return (x > 0.0f) ? x : alpha * (expf(x) - 1.0f);
}

void elu_forward(const float *x, float *out, int n, float alpha) {
    for (int i = 0; i < n; i++)
        out[i] = (x[i] > 0.0f) ? x[i] : alpha * (expf(x[i]) - 1.0f);
}

/*
 * L8: SwiGLU ！ Swish-Gated Linear Unit (Shazeer, 2020)
 * FFN_SwiGLU(x) = (silu(xW1) * (xW2)) W3
 * Replaces standard FFN in LLaMA, PaLM, Gemini.
 * Params: 3 weight matrices instead of 2, but better quality.
 */
void swiglu_forward(const float *x, const float *W1, const float *W2, const float *W3,
                    float *out, int batch, int d_model, int d_ff) {
    for (int b = 0; b < batch; b++) {
        float *gate = (float *)calloc(d_ff, sizeof(float));
        float *up   = (float *)calloc(d_ff, sizeof(float));
        /* gate = x @ W1, then silu */
        for (int j = 0; j < d_ff; j++)
            for (int i = 0; i < d_model; i++)
                gate[j] += x[b * d_model + i] * W1[i * d_ff + j];
        for (int j = 0; j < d_ff; j++) gate[j] = gate[j] / (1.0f + expf(-gate[j]));
        /* up = x @ W2 (linear) */
        for (int j = 0; j < d_ff; j++)
            for (int i = 0; i < d_model; i++)
                up[j] += x[b * d_model + i] * W2[i * d_ff + j];
        /* element-wise multiply: gate * up */
        for (int j = 0; j < d_ff; j++) gate[j] *= up[j];
        /* out = (gate*up) @ W3 */
        for (int j = 0; j < d_model; j++) {
            out[b * d_model + j] = 0.0f;
            for (int k = 0; k < d_ff; k++)
                out[b * d_model + j] += gate[k] * W3[k * d_model + j];
        }
        free(gate); free(up);
    }
}

/*
 * L8: GEGLU ！ GELU-Gated Linear Unit
 * Same structure as SwiGLU but gates with GELU instead of SiLU.
 */
void geglu_forward(const float *x, const float *W1, const float *W2, const float *W3,
                   float *out, int batch, int d_model, int d_ff) {
    float c = sqrtf(2.0f / 3.14159265f);
    for (int b = 0; b < batch; b++) {
        float *gate = (float *)calloc(d_ff, sizeof(float));
        float *up   = (float *)calloc(d_ff, sizeof(float));
        for (int j = 0; j < d_ff; j++)
            for (int i = 0; i < d_model; i++)
                gate[j] += x[b * d_model + i] * W1[i * d_ff + j];
        for (int j = 0; j < d_ff; j++) {
            float xi = gate[j];
            gate[j] = 0.5f * xi * (1.0f + tanhf(c * (xi + 0.044715f * xi * xi * xi)));
        }
        for (int j = 0; j < d_ff; j++)
            for (int i = 0; i < d_model; i++)
                up[j] += x[b * d_model + i] * W2[i * d_ff + j];
        for (int j = 0; j < d_ff; j++) gate[j] *= up[j];
        for (int j = 0; j < d_model; j++) {
            out[b * d_model + j] = 0.0f;
            for (int k = 0; k < d_ff; k++)
                out[b * d_model + j] += gate[k] * W3[k * d_model + j];
        }
        free(gate); free(up);
    }
}

/*
 * L5: Mish (Misra, 2019) ！ self-regularized non-monotonic activation
 * mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
 */
float mish(float x) {
    return x * tanhf(logf(1.0f + expf(x)));
}

void mish_forward(const float *x, float *out, int n) {
    for (int i = 0; i < n; i++)
        out[i] = x[i] * tanhf(logf(1.0f + expf(x[i])));
}

/*
 * L7: Softplus ！ smooth approximation of ReLU
 * softplus(x) = ln(1 + exp(x))
 * Used in VAEs for positive variance parameter.
 */
float softplus(float x) {
    /* Numerically stable: for large x, softplus(x) ~= x */
    if (x > 20.0f) return x;
    return logf(1.0f + expf(x));
}

void softplus_forward(const float *x, float *out, int n) {
    for (int i = 0; i < n; i++)
        out[i] = softplus(x[i]);
}
