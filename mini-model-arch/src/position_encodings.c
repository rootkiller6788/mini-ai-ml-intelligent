#include "position_encodings.h"

/*
 * L8: Rotary Position Embedding (RoPE) ¡ª Su et al., 2021
 *
 * L4: Theorem ¡ª RoPE encodes relative position via absolute rotation.
 * For positions m,n and vectors q,k in R^d (split into d/2 complex pairs):
 *   q_m = R_theta(m) * q    where R_theta(m) rotates pair i by m * theta_i
 *   k_n = R_theta(n) * k
 * Then: (R_m q)^T (R_n k) = q^T R_{n-m} k
 * The dot product depends only on relative distance (n-m).
 *
 * theta_i = base^{-2i/d} for i = 0, 1, ..., d/2-1
 * Used in LLaMA, Mistral, Gemma, Qwen, DeepSeek.
 */
RoPE *rope_create(int dim, int max_seq_len, float theta_base) {
    RoPE *r = (RoPE *)malloc(sizeof(RoPE));
    r->dim = dim;
    r->max_seq_len = max_seq_len;
    r->theta_base = theta_base;
    r->cos_cached = (float *)malloc(max_seq_len * dim * sizeof(float));
    r->sin_cached = (float *)malloc(max_seq_len * dim * sizeof(float));

    int half_dim = dim / 2;
    for (int pos = 0; pos < max_seq_len; pos++) {
        for (int i = 0; i < half_dim; i++) {
            float theta = (float)pos / powf(theta_base, (2.0f * i) / (float)dim);
            r->cos_cached[pos * dim + i * 2]     = cosf(theta);
            r->cos_cached[pos * dim + i * 2 + 1] = cosf(theta);
            r->sin_cached[pos * dim + i * 2]     = sinf(theta);
            r->sin_cached[pos * dim + i * 2 + 1] = sinf(theta);
        }
    }
    return r;
}

void rope_free(RoPE *r) {
    free(r->cos_cached); free(r->sin_cached); free(r);
}

/*
 * Apply RoPE to query and key vectors.
 * q,k: [seq_len, dim] row-major layout.
 * For each pair (2i, 2i+1) in the dimension:
 *   [x_2i, x_2i+1] * [cos, sin] => rotation in 2D plane
 *   out_2i   = x_2i * cos - x_2i+1 * sin
 *   out_2i+1 = x_2i * sin + x_2i+1 * cos
 */
void rope_apply(RoPE *r, float *q, float *k, int seq_len, int start_pos) {
    int dim = r->dim;
    int half_dim = dim / 2;

    for (int s = 0; s < seq_len; s++) {
        int pos = start_pos + s;
        float *cos_ptr = r->cos_cached + pos * dim;
        float *sin_ptr = r->sin_cached + pos * dim;

        float *q_row = q + s * dim;
        float *k_row = k + s * dim;

        for (int i = 0; i < half_dim; i++) {
            int i2 = i * 2;
            float q0 = q_row[i2], q1 = q_row[i2 + 1];
            float k0 = k_row[i2], k1 = k_row[i2 + 1];
            float c = cos_ptr[i2], sval = sin_ptr[i2];

            q_row[i2]     = q0 * c - q1 * sval;
            q_row[i2 + 1] = q0 * sval + q1 * c;
            k_row[i2]     = k0 * c - k1 * sval;
            k_row[i2 + 1] = k0 * sval + k1 * c;
        }
    }
}

/*
 * L8: ALiBi ¡ª Attention with Linear Biases (Press et al., ICLR 2022)
 *
 * Adds a static bias to attention scores:
 *   bias(i,j) = -m_h * |i - j|
 * where m_h is a head-specific slope.
 *
 * L4: ALiBi enables length extrapolation: model trained on L tokens
 *     can perform well on 2L+ tokens without fine-tuning.
 *     Slopes form a geometric sequence.
 */
ALiBi *alibi_create(int n_heads) {
    ALiBi *a = (ALiBi *)malloc(sizeof(ALiBi));
    a->n_heads = n_heads;
    a->slopes = (float *)malloc(n_heads * sizeof(float));

    /* Geometric slopes: start at 2^{-8/n_heads} and multiply */
    float base = powf(2.0f, -8.0f / (float)n_heads);
    for (int h = 0; h < n_heads; h++)
        a->slopes[h] = powf(base, (float)(h + 1));
    return a;
}

void alibi_free(ALiBi *a) { free(a->slopes); free(a); }

void alibi_apply(const ALiBi *a, float *attn_scores, int seq_len) {
    for (int h = 0; h < a->n_heads; h++) {
        float m = a->slopes[h];
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < seq_len; j++) {
                attn_scores[h * seq_len * seq_len + i * seq_len + j] -= m * (float)abs(i - j);
            }
        }
    }
}

/*
 * L7: Sinusoidal Position Encoding (Vaswani et al., 2017)
 * PE(pos, 2i)   = sin(pos / 10000^{2i/d_model})
 * PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})
 * Allows model to attend to relative positions via linear combinations.
 */
void sinusoidal_pe(float *pe, int max_len, int d_model) {
    for (int pos = 0; pos < max_len; pos++) {
        for (int i = 0; i < d_model; i++) {
            float angle = (float)pos / powf(10000.0f, (2.0f * (i / 2)) / (float)d_model);
            pe[pos * d_model + i] = (i % 2 == 0) ? sinf(angle) : cosf(angle);
        }
    }
}

/*
 * L7: Learnable Position Embedding
 * Initialized from N(0, 0.02^2) as in GPT-2.
 */
void learnable_pe_init(float *pe, int max_len, int d_model) {
    for (int i = 0; i < max_len * d_model; i++)
        pe[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 0.02f * 2.0f;
}

void learnable_pe_apply(const float *pe, float *x, int seq_len, int d_model) {
    for (int i = 0; i < seq_len * d_model; i++)
        x[i] += pe[i];
}
