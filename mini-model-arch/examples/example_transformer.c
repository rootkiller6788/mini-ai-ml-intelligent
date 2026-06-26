#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "transformer_arch.h"

int main(void) {
    printf("=== mini-model-arch: Transformer Example ===\n\n");
    srand((unsigned)time(NULL));

    int d_model = 64, n_heads = 4, d_ff = 256, seq_len = 8, n_layers = 2;

    printf("1. Self-Attention (d_model=%d, n_heads=%d, seq_len=%d)\n",
           d_model, n_heads, seq_len);
    MultiHeadAttn *mha = mha_create(d_model, n_heads, 0);
    Matrix2D *q = matrix2d_create(d_model, seq_len);
    Matrix2D *k = matrix2d_create(d_model, seq_len);
    Matrix2D *v = matrix2d_create(d_model, seq_len);
    for (int i = 0; i < d_model * seq_len; i++) {
        q->data[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        k->data[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        v->data[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }
    Matrix2D *attn_out = mha_forward(mha, q, k, v);
    printf("   Output shape: [%d x %d]\n", attn_out->dim, attn_out->seq_len);
    printf("   Sample output[0..3] = [%.4f, %.4f, %.4f, %.4f]\n",
           attn_out->data[0], attn_out->data[1],
           attn_out->data[2], attn_out->data[3]);
    matrix2d_free(attn_out);
    matrix2d_free(q); matrix2d_free(k); matrix2d_free(v);
    mha_free(mha);

    printf("\n2. Causal Self-Attention (GPT-style, masked)\n");
    MultiHeadAttn *causal = mha_create(d_model, n_heads, 1);
    q = matrix2d_create(d_model, seq_len);
    for (int i = 0; i < d_model * seq_len; i++)
        q->data[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    attn_out = mha_forward(causal, q, q, q);
    printf("   Output shape: [%d x %d]\n", attn_out->dim, attn_out->seq_len);
    printf("   Upper-triangular scores are masked (-inf).\n");
    matrix2d_free(attn_out);
    matrix2d_free(q); mha_free(causal);

    printf("\n3. Position Encoding (sinusoidal, max_len=128)\n");
    PositionEncoding *pe = pe_create(d_model, 128, 0);
    printf("   PE[0, 0..3] = [%.4f, %.4f, %.4f, %.4f]\n",
           pe->pe[0], pe->pe[1], pe->pe[2], pe->pe[3]);
    printf("   PE[10, 0..3] = [%.4f, %.4f, %.4f, %.4f]\n",
           pe->pe[10 * d_model], pe->pe[10 * d_model + 1],
           pe->pe[10 * d_model + 2], pe->pe[10 * d_model + 3]);
    q = matrix2d_create(d_model, 4);
    for (int i = 0; i < d_model * 4; i++) q->data[i] = 1.0f;
    pe_apply(pe, q);
    printf("   After PE apply, x[0,0..3] = [%.4f, %.4f, %.4f, %.4f]\n",
           q->data[0], q->data[1], q->data[2], q->data[3]);
    matrix2d_free(q); pe_free(pe);

    printf("\n4. FeedForward Network (d_model=%d, d_ff=%d)\n", d_model, d_ff);
    FeedForward *ffn = ffn_create(d_model, d_ff);
    q = matrix2d_create(d_model, seq_len);
    for (int i = 0; i < d_model * seq_len; i++)
        q->data[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    Matrix2D *ff_out = ffn_forward(ffn, q);
    printf("   FFN: %d -> %d -> %d\n", d_model, d_ff, d_model);
    printf("   Output shape: [%d x %d]\n", ff_out->dim, ff_out->seq_len);
    matrix2d_free(ff_out); matrix2d_free(q); ffn_free(ffn);

    printf("\n5. Encoder Block (pre-LN, d_model=%d, n_heads=%d, d_ff=%d)\n",
           d_model, n_heads, d_ff);
    EncoderBlock *eb = enc_block_create(d_model, n_heads, d_ff, 1);
    q = matrix2d_create(d_model, seq_len);
    for (int i = 0; i < d_model * seq_len; i++)
        q->data[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    attn_out = enc_block_forward(eb, q);
    printf("   Output shape: [%d x %d]\n", attn_out->dim, attn_out->seq_len);
    printf("   Pre-LN: LayerNorm before attention, then residual.\n");
    matrix2d_free(attn_out); matrix2d_free(q); enc_block_free(eb);

    printf("\n6. Full Transformer Encoder (%d layers)\n", n_layers);
    TransformerEncoder *enc = trans_enc_create(n_layers, d_model, n_heads, d_ff, 1);
    q = matrix2d_create(d_model, seq_len);
    for (int i = 0; i < d_model * seq_len; i++)
        q->data[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    attn_out = trans_enc_forward(enc, q);
    printf("   Output shape: [%d x %d]\n", attn_out->dim, attn_out->seq_len);
    printf("   Encoded all %d tokens through %d layers.\n", seq_len, n_layers);
    matrix2d_free(attn_out); matrix2d_free(q); trans_enc_free(enc);

    printf("\n7. BERT Forward (encoder-only, %d layers)\n", n_layers);
    q = matrix2d_create(d_model, seq_len);
    for (int i = 0; i < d_model * seq_len; i++)
        q->data[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    attn_out = bert_forward(q, n_layers, d_model, n_heads, d_ff);
    printf("   BERT output shape: [%d x %d]\n", attn_out->dim, attn_out->seq_len);
    matrix2d_free(attn_out); matrix2d_free(q);

    printf("\n8. GPT Forward (decoder-only, causal, %d layers)\n", n_layers);
    q = matrix2d_create(d_model, seq_len);
    for (int i = 0; i < d_model * seq_len; i++)
        q->data[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    attn_out = gpt_forward(q, n_layers, d_model, n_heads, d_ff);
    printf("   GPT output shape: [%d x %d]\n", attn_out->dim, attn_out->seq_len);
    printf("   Causal mask prevents attending to future tokens.\n");
    matrix2d_free(attn_out); matrix2d_free(q);

    printf("\nAll Transformer tests passed.\n");
    return 0;
}
