#ifndef TRANSFORMER_ARCH_H
#define TRANSFORMER_ARCH_H

#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int    dim, seq_len;
    float *data;
} Matrix2D;

typedef struct {
    int    d_model, n_heads, d_k, d_v;
    float *W_Q, *W_K, *W_V, *W_O;
    int    causal_mask;
} MultiHeadAttn;

typedef struct {
    int    d_model, max_len;
    float *pe;
    int    learnable;
} PositionEncoding;

typedef struct {
    int    d_model, d_ff;
    float *W1, *b1, *W2, *b2;
} FeedForward;

typedef struct {
    int    d_model;
    float *gamma, *beta;
    float  eps;
} LayerNorm;

typedef struct {
    MultiHeadAttn    *self_attn;
    FeedForward      *ffn;
    LayerNorm        *norm1, *norm2;
    int               d_model;
    int               pre_ln;
} EncoderBlock;

typedef struct {
    MultiHeadAttn    *self_attn;
    MultiHeadAttn    *cross_attn;
    FeedForward      *ffn;
    LayerNorm        *norm1, *norm2, *norm3;
    int               d_model;
    int               pre_ln;
} DecoderBlock;

typedef struct {
    EncoderBlock **layers;
    int            num_layers, d_model;
    PositionEncoding *pe;
    int               pre_ln;
} TransformerEncoder;

typedef struct {
    DecoderBlock **layers;
    int            num_layers, d_model;
    PositionEncoding *pe;
    int               pre_ln;
} TransformerDecoder;

Matrix2D *matrix2d_create(int dim, int seq);
void      matrix2d_free(Matrix2D *m);

MultiHeadAttn *mha_create(int d_model, int n_heads, int causal);
void           mha_free(MultiHeadAttn *m);
Matrix2D      *mha_forward(const MultiHeadAttn *m, const Matrix2D *Q,
                           const Matrix2D *K, const Matrix2D *V);

PositionEncoding *pe_create(int d_model, int max_len, int learnable);
void              pe_free(PositionEncoding *p);
void              pe_apply(PositionEncoding *p, Matrix2D *x);

FeedForward *ffn_create(int d_model, int d_ff);
void         ffn_free(FeedForward *f);
Matrix2D    *ffn_forward(const FeedForward *f, const Matrix2D *x);

LayerNorm *ln_create(int d_model);
void       ln_free(LayerNorm *l);
void       ln_forward(const LayerNorm *l, Matrix2D *x);

EncoderBlock *enc_block_create(int d_model, int n_heads, int d_ff, int pre_ln);
void          enc_block_free(EncoderBlock *e);
Matrix2D     *enc_block_forward(const EncoderBlock *e, const Matrix2D *x);

DecoderBlock *dec_block_create(int d_model, int n_heads, int d_ff, int pre_ln);
void          dec_block_free(DecoderBlock *d);
Matrix2D     *dec_block_forward(const DecoderBlock *d, const Matrix2D *x,
                                const Matrix2D *enc_out);

TransformerEncoder *trans_enc_create(int n_layers, int d_model, int n_heads, int d_ff, int pre_ln);
void                trans_enc_free(TransformerEncoder *e);
Matrix2D           *trans_enc_forward(const TransformerEncoder *e, const Matrix2D *x);

TransformerDecoder *trans_dec_create(int n_layers, int d_model, int n_heads, int d_ff, int pre_ln);
void                trans_dec_free(TransformerDecoder *d);
Matrix2D           *trans_dec_forward(const TransformerDecoder *d, const Matrix2D *x,
                                      const Matrix2D *enc_out);

Matrix2D *gpt_forward(const Matrix2D *tokens, int n_layers, int d_model, int n_heads, int d_ff);
Matrix2D *bert_forward(const Matrix2D *tokens, int n_layers, int d_model, int n_heads, int d_ff);

void softmax_rows(float *x, int rows, int cols);
void matmul(const float *A, const float *B, float *C, int m, int k, int n);
void matmul_transpose_b(const float *A, const float *B, float *C, int m, int k, int n);

#endif
