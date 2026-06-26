#include "transformer_arch.h"
#include <stdio.h>

Matrix2D *matrix2d_create(int dim, int seq) {
    Matrix2D *m = (Matrix2D *)malloc(sizeof(Matrix2D));
    m->dim = dim; m->seq_len = seq;
    m->data = (float *)calloc(dim * seq, sizeof(float));
    return m;
}
void matrix2d_free(Matrix2D *m) { free(m->data); free(m); }

void matmul(const float *A, const float *B, float *C, int m, int k, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0;
            for (int p = 0; p < k; p++)
                sum += A[i * k + p] * B[p * n + j];
            C[i * n + j] = sum;
        }
    }
}
void matmul_transpose_b(const float *A, const float *B, float *C, int m, int k, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0;
            for (int p = 0; p < k; p++)
                sum += A[i * k + p] * B[j * k + p];
            C[i * n + j] = sum;
        }
    }
}

void softmax_rows(float *x, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        float maxv = x[r * cols];
        for (int c = 1; c < cols; c++)
            if (x[r * cols + c] > maxv) maxv = x[r * cols + c];
        float sum = 0;
        for (int c = 0; c < cols; c++) {
            x[r * cols + c] = expf(x[r * cols + c] - maxv);
            sum += x[r * cols + c];
        }
        for (int c = 0; c < cols; c++)
            x[r * cols + c] /= sum;
    }
}

MultiHeadAttn *mha_create(int d_model, int n_heads, int causal) {
    MultiHeadAttn *m = (MultiHeadAttn *)malloc(sizeof(MultiHeadAttn));
    m->d_model = d_model; m->n_heads = n_heads;
    m->d_k = d_model / n_heads; m->d_v = d_model / n_heads;
    m->causal_mask = causal;
    int sz = d_model * d_model;
    m->W_Q = (float *)malloc(sz * sizeof(float));
    m->W_K = (float *)malloc(sz * sizeof(float));
    m->W_V = (float *)malloc(sz * sizeof(float));
    m->W_O = (float *)malloc(sz * sizeof(float));
    float scale = sqrtf(2.0f / d_model);
    for (int i = 0; i < sz; i++) {
        m->W_Q[i] = ((float)rand() / RAND_MAX - 0.5f) * scale;
        m->W_K[i] = ((float)rand() / RAND_MAX - 0.5f) * scale;
        m->W_V[i] = ((float)rand() / RAND_MAX - 0.5f) * scale;
        m->W_O[i] = ((float)rand() / RAND_MAX - 0.5f) * scale;
    }
    return m;
}
void mha_free(MultiHeadAttn *m) {
    free(m->W_Q); free(m->W_K); free(m->W_V); free(m->W_O); free(m);
}

Matrix2D *mha_forward(const MultiHeadAttn *m, const Matrix2D *Q,
                       const Matrix2D *K, const Matrix2D *V) {
    int d = m->d_model, h = m->n_heads, dk = m->d_k, dv = m->d_v;
    int sq = Q->seq_len, sk = K->seq_len;

    float *q_proj = (float *)malloc(sq * d * sizeof(float));
    float *k_proj = (float *)malloc(sk * d * sizeof(float));
    float *v_proj = (float *)malloc(sk * d * sizeof(float));
    matmul(Q->data, m->W_Q, q_proj, sq, d, d);
    matmul(K->data, m->W_K, k_proj, sk, d, d);
    matmul(V->data, m->W_V, v_proj, sk, d, d);

    float *attn_out = (float *)calloc(sq * d, sizeof(float));
    float *scores   = (float *)malloc(sq * sk * sizeof(float));
    float scale_fac = 1.0f / sqrtf((float)dk);

    for (int head = 0; head < h; head++) {
        int q_off = head * dk, k_off = head * dk, v_off = head * dv;

        for (int i = 0; i < sq; i++) {
            for (int j = 0; j < sk; j++) {
                float dot = 0;
                for (int p = 0; p < dk; p++)
                    dot += q_proj[i * d + q_off + p] * k_proj[j * d + k_off + p];
                scores[i * sk + j] = dot * scale_fac;
            }
        }

        if (m->causal_mask) {
            for (int i = 0; i < sq; i++)
                for (int j = i + 1; j < sk; j++)
                    scores[i * sk + j] = -1e9f;
        }

        softmax_rows(scores, sq, sk);

        for (int i = 0; i < sq; i++) {
            for (int p = 0; p < dv; p++) {
                float wsum = 0;
                for (int j = 0; j < sk; j++)
                    wsum += scores[i * sk + j] * v_proj[j * d + v_off + p];
                attn_out[i * d + head * dv + p] += wsum;
            }
        }
    }

    float *final = (float *)malloc(sq * d * sizeof(float));
    matmul(attn_out, m->W_O, final, sq, d, d);

    Matrix2D *result = matrix2d_create(d, sq);
    memcpy(result->data, final, sq * d * sizeof(float));

    free(q_proj); free(k_proj); free(v_proj);
    free(attn_out); free(scores); free(final);
    return result;
}

PositionEncoding *pe_create(int d_model, int max_len, int learnable) {
    PositionEncoding *p = (PositionEncoding *)malloc(sizeof(PositionEncoding));
    p->d_model = d_model; p->max_len = max_len; p->learnable = learnable;
    p->pe = (float *)malloc(max_len * d_model * sizeof(float));
    for (int pos = 0; pos < max_len; pos++) {
        for (int i = 0; i < d_model; i++) {
            float angle = (float)pos / powf(10000.0f, (2.0f * i) / d_model);
            p->pe[pos * d_model + i] = (i % 2 == 0) ? sinf(angle) : cosf(angle);
        }
    }
    return p;
}
void pe_free(PositionEncoding *p) { free(p->pe); free(p); }
void pe_apply(PositionEncoding *p, Matrix2D *x) {
    int n = x->dim * (x->seq_len < p->max_len ? x->seq_len : p->max_len);
    for (int i = 0; i < n; i++) x->data[i] += p->pe[i];
}

FeedForward *ffn_create(int d_model, int d_ff) {
    FeedForward *f = (FeedForward *)malloc(sizeof(FeedForward));
    f->d_model = d_model; f->d_ff = d_ff;
    f->W1 = (float *)malloc(d_model * d_ff * sizeof(float));
    f->b1 = (float *)calloc(d_ff, sizeof(float));
    f->W2 = (float *)malloc(d_ff * d_model * sizeof(float));
    f->b2 = (float *)calloc(d_model, sizeof(float));
    float s1 = sqrtf(2.0f / d_model), s2 = sqrtf(2.0f / d_ff);
    for (int i = 0; i < d_model * d_ff; i++)
        f->W1[i] = ((float)rand() / RAND_MAX - 0.5f) * s1;
    for (int i = 0; i < d_ff * d_model; i++)
        f->W2[i] = ((float)rand() / RAND_MAX - 0.5f) * s2;
    return f;
}
void ffn_free(FeedForward *f) { free(f->W1); free(f->b1); free(f->W2); free(f->b2); free(f); }

Matrix2D *ffn_forward(const FeedForward *f, const Matrix2D *x) {
    int s = x->seq_len, d = f->d_model, dff = f->d_ff;
    float *hid = (float *)malloc(s * dff * sizeof(float));
    matmul(x->data, f->W1, hid, s, d, dff);
    for (int i = 0; i < s * dff; i++) { hid[i] += f->b1[i % dff]; if (hid[i] < 0) hid[i] = 0; }
    float *out = (float *)malloc(s * d * sizeof(float));
    matmul(hid, f->W2, out, s, dff, d);
    for (int i = 0; i < s * d; i++) out[i] += f->b2[i % d];
    Matrix2D *r = matrix2d_create(d, s);
    memcpy(r->data, out, s * d * sizeof(float));
    free(hid); free(out);
    return r;
}

LayerNorm *ln_create(int d_model) {
    LayerNorm *l = (LayerNorm *)malloc(sizeof(LayerNorm));
    l->d_model = d_model; l->eps = 1e-5f;
    l->gamma = (float *)malloc(d_model * sizeof(float));
    l->beta  = (float *)calloc(d_model, sizeof(float));
    for (int i = 0; i < d_model; i++) l->gamma[i] = 1.0f;
    return l;
}
void ln_free(LayerNorm *l) { free(l->gamma); free(l->beta); free(l); }
void ln_forward(const LayerNorm *l, Matrix2D *x) {
    int s = x->seq_len, d = l->d_model;
    for (int i = 0; i < s; i++) {
        float mean = 0, var = 0;
        float *row = x->data + i * d;
        for (int j = 0; j < d; j++) mean += row[j];
        mean /= d;
        for (int j = 0; j < d; j++) { float diff = row[j] - mean; var += diff * diff; }
        var = var / d + l->eps;
        float inv_std = 1.0f / sqrtf(var);
        for (int j = 0; j < d; j++)
            row[j] = l->gamma[j] * (row[j] - mean) * inv_std + l->beta[j];
    }
}

EncoderBlock *enc_block_create(int d_model, int n_heads, int d_ff, int pre_ln) {
    EncoderBlock *e = (EncoderBlock *)malloc(sizeof(EncoderBlock));
    e->d_model = d_model; e->pre_ln = pre_ln;
    e->self_attn = mha_create(d_model, n_heads, 0);
    e->ffn = ffn_create(d_model, d_ff);
    e->norm1 = ln_create(d_model);
    e->norm2 = ln_create(d_model);
    return e;
}
void enc_block_free(EncoderBlock *e) {
    mha_free(e->self_attn); ffn_free(e->ffn);
    ln_free(e->norm1); ln_free(e->norm2); free(e);
}
Matrix2D *enc_block_forward(const EncoderBlock *e, const Matrix2D *x) {
    if (e->pre_ln) {
        Matrix2D *x_norm = matrix2d_create(e->d_model, x->seq_len);
        memcpy(x_norm->data, x->data, x->dim * x->seq_len * sizeof(float));
        ln_forward(e->norm1, x_norm);
        Matrix2D *attn = mha_forward(e->self_attn, x_norm, x_norm, x_norm);
        matrix2d_free(x_norm);
        for (int i = 0; i < x->dim * x->seq_len; i++) x->data[i] += attn->data[i];
        Matrix2D *x_n2 = matrix2d_create(e->d_model, x->seq_len);
        memcpy(x_n2->data, x->data, x->dim * x->seq_len * sizeof(float));
        ln_forward(e->norm2, x_n2);
        Matrix2D *ff_out = ffn_forward(e->ffn, x_n2);
        for (int i = 0; i < x->dim * x->seq_len; i++) x->data[i] += ff_out->data[i];
        matrix2d_free(x_n2); matrix2d_free(ff_out);
        Matrix2D *result = matrix2d_create(e->d_model, x->seq_len);
        memcpy(result->data, x->data, x->dim * x->seq_len * sizeof(float));
        matrix2d_free(attn);
        return result;
    } else {
        Matrix2D *attn = mha_forward(e->self_attn, x, x, x);
        for (int i = 0; i < x->dim * x->seq_len; i++) attn->data[i] += x->data[i];
        ln_forward(e->norm1, attn);
        Matrix2D *ff_out = ffn_forward(e->ffn, attn);
        for (int i = 0; i < x->dim * x->seq_len; i++) ff_out->data[i] += attn->data[i];
        ln_forward(e->norm2, ff_out);
        matrix2d_free(attn);
        return ff_out;
    }
}

DecoderBlock *dec_block_create(int d_model, int n_heads, int d_ff, int pre_ln) {
    DecoderBlock *d = (DecoderBlock *)malloc(sizeof(DecoderBlock));
    d->d_model = d_model; d->pre_ln = pre_ln;
    d->self_attn  = mha_create(d_model, n_heads, 1);
    d->cross_attn = mha_create(d_model, n_heads, 0);
    d->ffn = ffn_create(d_model, d_ff);
    d->norm1 = ln_create(d_model);
    d->norm2 = ln_create(d_model);
    d->norm3 = ln_create(d_model);
    return d;
}
void dec_block_free(DecoderBlock *d) {
    mha_free(d->self_attn); mha_free(d->cross_attn);
    ffn_free(d->ffn); ln_free(d->norm1); ln_free(d->norm2); ln_free(d->norm3); free(d);
}
Matrix2D *dec_block_forward(const DecoderBlock *d, const Matrix2D *x, const Matrix2D *enc_out) {
    Matrix2D *x_copy = matrix2d_create(d->d_model, x->seq_len);
    memcpy(x_copy->data, x->data, x->dim * x->seq_len * sizeof(float));
    ln_forward(d->norm1, x_copy);
    Matrix2D *self_a = mha_forward(d->self_attn, x_copy, x_copy, x_copy);
    matrix2d_free(x_copy);
    for (int i = 0; i < x->dim * x->seq_len; i++) self_a->data[i] += x->data[i];
    Matrix2D *x_copy2 = matrix2d_create(d->d_model, self_a->seq_len);
    memcpy(x_copy2->data, self_a->data, d->d_model * self_a->seq_len * sizeof(float));
    ln_forward(d->norm2, x_copy2);
    Matrix2D *cross_a = mha_forward(d->cross_attn, x_copy2, enc_out, enc_out);
    matrix2d_free(x_copy2);
    for (int i = 0; i < d->d_model * self_a->seq_len; i++) cross_a->data[i] += self_a->data[i];
    Matrix2D *x_copy3 = matrix2d_create(d->d_model, cross_a->seq_len);
    memcpy(x_copy3->data, cross_a->data, d->d_model * cross_a->seq_len * sizeof(float));
    ln_forward(d->norm3, x_copy3);
    Matrix2D *ff_out = ffn_forward(d->ffn, x_copy3);
    matrix2d_free(x_copy3);
    for (int i = 0; i < d->d_model * cross_a->seq_len; i++) ff_out->data[i] += cross_a->data[i];
    matrix2d_free(self_a); matrix2d_free(cross_a);
    return ff_out;
}

TransformerEncoder *trans_enc_create(int n_layers, int d_model, int n_heads, int d_ff, int pre_ln) {
    TransformerEncoder *e = (TransformerEncoder *)malloc(sizeof(TransformerEncoder));
    e->num_layers = n_layers; e->d_model = d_model; e->pre_ln = pre_ln;
    e->layers = (EncoderBlock **)malloc(n_layers * sizeof(EncoderBlock *));
    for (int i = 0; i < n_layers; i++)
        e->layers[i] = enc_block_create(d_model, n_heads, d_ff, pre_ln);
    e->pe = pe_create(d_model, 512, 0);
    return e;
}
void trans_enc_free(TransformerEncoder *e) {
    for (int i = 0; i < e->num_layers; i++) enc_block_free(e->layers[i]);
    free(e->layers); pe_free(e->pe); free(e);
}
Matrix2D *trans_enc_forward(const TransformerEncoder *e, const Matrix2D *x) {
    pe_apply(e->pe, x);
    Matrix2D *cur = x;
    for (int i = 0; i < e->num_layers; i++) {
        Matrix2D *next = enc_block_forward(e->layers[i], cur);
        if (i > 0) matrix2d_free(cur);
        cur = next;
    }
    return cur;
}

TransformerDecoder *trans_dec_create(int n_layers, int d_model, int n_heads, int d_ff, int pre_ln) {
    TransformerDecoder *d = (TransformerDecoder *)malloc(sizeof(TransformerDecoder));
    d->num_layers = n_layers; d->d_model = d_model; d->pre_ln = pre_ln;
    d->layers = (DecoderBlock **)malloc(n_layers * sizeof(DecoderBlock *));
    for (int i = 0; i < n_layers; i++)
        d->layers[i] = dec_block_create(d_model, n_heads, d_ff, pre_ln);
    d->pe = pe_create(d_model, 512, 0);
    return d;
}
void trans_dec_free(TransformerDecoder *d) {
    for (int i = 0; i < d->num_layers; i++) dec_block_free(d->layers[i]);
    free(d->layers); pe_free(d->pe); free(d);
}
Matrix2D *trans_dec_forward(const TransformerDecoder *d, const Matrix2D *x, const Matrix2D *enc_out) {
    pe_apply(d->pe, x);
    Matrix2D *cur = x;
    for (int i = 0; i < d->num_layers; i++) {
        Matrix2D *next = dec_block_forward(d->layers[i], cur, enc_out);
        if (i > 0) matrix2d_free(cur);
        cur = next;
    }
    return cur;
}

Matrix2D *gpt_forward(const Matrix2D *tokens, int n_layers, int d_model, int n_heads, int d_ff) {
    TransformerDecoder *dec = trans_dec_create(n_layers, d_model, n_heads, d_ff, 1);
    Matrix2D *x = matrix2d_create(d_model, tokens->seq_len);
    memcpy(x->data, tokens->data, d_model * tokens->seq_len * sizeof(float));
    pe_apply(dec->pe, x);
    Matrix2D *cur = x;
    for (int i = 0; i < n_layers; i++) {
        Matrix2D *next = dec_block_forward(dec->layers[i], cur, cur);
        if (i > 0) matrix2d_free(cur);
        cur = next;
    }
    trans_dec_free(dec);
    return cur;
}

Matrix2D *bert_forward(const Matrix2D *tokens, int n_layers, int d_model, int n_heads, int d_ff) {
    TransformerEncoder *enc = trans_enc_create(n_layers, d_model, n_heads, d_ff, 1);
    Matrix2D *result = trans_enc_forward(enc, tokens);
    trans_enc_free(enc);
    return result;
}
