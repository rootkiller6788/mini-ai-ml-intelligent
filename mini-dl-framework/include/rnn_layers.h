#ifndef RNN_LAYERS_H
#define RNN_LAYERS_H

#include "tensor_ops.h"
#include <stdbool.h>

/* ── L1 Definitions ──────────────────────────────────────────── */

/* Embedding layer: maps discrete token indices to dense vectors.
 * Theorem: embed_dim ≥ log₂(V) for unique representation (Shannon),
 *           practical: d_model ≈ V^{1/4} (Mikolov et al. 2013). */
typedef struct {
    int vocab_size;
    int embed_dim;
    Tensor* weight;       /* [vocab_size, embed_dim] */
    Tensor* weight_grad;
    float padding_idx;
} Embedding;

Embedding* embedding_create(int vocab_size, int embed_dim, float padding_idx);
Tensor* embedding_forward(Embedding* emb, Tensor* indices);
Tensor* embedding_backward(Embedding* emb, Tensor* grad_output,
                          Tensor* indices);
void embedding_free(Embedding* emb);

/* ── RNNCell: Simple Elman Recurrent Network Cell ───────────────
 * h_t = tanh(W_hh·h_{t-1} + W_ih·x_t + b)
 * BPTT gradient: dL/dW = Σ_t dL/dh_t · dh_t/dW (Werbos 1990) */
typedef struct {
    int input_size;
    int hidden_size;
    Tensor* W_ih;          /* [hidden_size, input_size] */
    Tensor* W_hh;          /* [hidden_size, hidden_size] */
    Tensor* b_ih;          /* [hidden_size] */
    Tensor* b_hh;          /* [hidden_size] */
    Tensor* W_ih_grad;
    Tensor* W_hh_grad;
    Tensor* b_ih_grad;
    Tensor* b_hh_grad;
} RNNCell;

RNNCell* rnn_cell_create(int input_size, int hidden_size);
Tensor* rnn_cell_forward(RNNCell* cell, Tensor* input, Tensor* hidden);
void rnn_cell_backward(RNNCell* cell, Tensor* grad_output,
                       Tensor* input, Tensor* hidden,
                       Tensor** d_input, Tensor** d_hidden);
void rnn_cell_free(RNNCell* cell);

/* ── LSTMCell: Hochreiter & Schmidhuber 1997 ───────────────────
 * f_t = σ(W_f·[h_{t-1}, x_t] + b_f)  — forget gate
 * i_t = σ(W_i·[h_{t-1}, x_t] + b_i)  — input gate
 * g_t = tanh(W_g·[h_{t-1}, x_t] + b_g) — candidate
 * o_t = σ(W_o·[h_{t-1}, x_t] + b_o)  — output gate
 * c_t = f_t·c_{t-1} + i_t·g_t
 * h_t = o_t·tanh(c_t) */

typedef struct {
    int input_size;
    int hidden_size;

    Tensor* W_f;  Tensor* W_i;   Tensor* W_g;   Tensor* W_o;
    Tensor* U_f;  Tensor* U_i;   Tensor* U_g;   Tensor* U_o;
    Tensor* b_f;  Tensor* b_i;   Tensor* b_g;   Tensor* b_o;

    Tensor* W_f_grad; Tensor* W_i_grad; Tensor* W_g_grad; Tensor* W_o_grad;
    Tensor* U_f_grad; Tensor* U_i_grad; Tensor* U_g_grad; Tensor* U_o_grad;
    Tensor* b_f_grad; Tensor* b_i_grad; Tensor* b_g_grad; Tensor* b_o_grad;

    /* Caches for backward pass */
    Tensor* input_cache;
    Tensor* hidden_cache;
    Tensor* cell_cache;
    Tensor* f_gate_cache;
    Tensor* i_gate_cache;
    Tensor* g_gate_cache;
    Tensor* o_gate_cache;
} LSTMCell;

LSTMCell* lstm_cell_create(int input_size, int hidden_size);
void lstm_cell_forward(LSTMCell* cell, Tensor* input,
                       Tensor* hidden, Tensor* cell_state,
                       Tensor** out_hidden, Tensor** out_cell);
void lstm_cell_backward(LSTMCell* cell, Tensor* grad_h, Tensor* grad_c,
                         Tensor** d_input, Tensor** d_hidden,
                         Tensor** d_cell);
void lstm_cell_free(LSTMCell* cell);

/* ── GRUCell: Cho et al. 2014 ──────────────────────────────────
 * r_t = σ(W_r·x_t + U_r·h_{t-1} + b_r) — reset gate
 * z_t = σ(W_z·x_t + U_z·h_{t-1} + b_z) — update gate
 * n_t = tanh(W_n·x_t + r_t⊙(U_n·h_{t-1}) + b_n) — new gate
 * h_t = (1-z_t)⊙n_t + z_t⊙h_{t-1} */

typedef struct {
    int input_size;
    int hidden_size;

    Tensor* W_r;  Tensor* W_z;   Tensor* W_n;
    Tensor* U_r;  Tensor* U_z;   Tensor* U_n;
    Tensor* b_r;  Tensor* b_z;   Tensor* b_n;

    Tensor* W_r_grad; Tensor* W_z_grad; Tensor* W_n_grad;
    Tensor* U_r_grad; Tensor* U_z_grad; Tensor* U_n_grad;
    Tensor* b_r_grad; Tensor* b_z_grad; Tensor* b_n_grad;

    Tensor* input_cache;
    Tensor* hidden_cache;
    Tensor* r_gate_cache;
    Tensor* z_gate_cache;
    Tensor* n_gate_cache;
} GRUCell;

GRUCell* gru_cell_create(int input_size, int hidden_size);
void gru_cell_forward(GRUCell* cell, Tensor* input, Tensor* hidden,
                      Tensor** out_hidden);
void gru_cell_backward(GRUCell* cell, Tensor* grad_output,
                       Tensor** d_input, Tensor** d_hidden);
void gru_cell_free(GRUCell* cell);

/* ── Multi-Head Scaled Dot-Product Attention (Vaswani et al. 2017)
 * Attention(Q,K,V) = softmax(QK^T/√d_k)·V
 * MultiHead(Q,K,V) = Concat(head_1,...,head_h)·W_O
 * where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V) */

typedef struct {
    int embed_dim;
    int num_heads;
    int head_dim;
    float dropout_p;

    Tensor* W_Q;  Tensor* W_K;  Tensor* W_V;  Tensor* W_O;
    Tensor* W_Q_grad; Tensor* W_K_grad; Tensor* W_V_grad; Tensor* W_O_grad;

    /* Caches for backward */
    Tensor* Q_cache;   Tensor* K_cache;   Tensor* V_cache;
    Tensor* attn_cache; /* attention weights before dropout */
    Tensor* mask_cache;
    bool is_training;
} MultiHeadAttention;

MultiHeadAttention* mha_create(int embed_dim, int num_heads,
                                float dropout_p);
Tensor* mha_forward(MultiHeadAttention* mha, Tensor* query,
                    Tensor* key, Tensor* value, Tensor* mask);
Tensor* mha_backward(MultiHeadAttention* mha, Tensor* grad_output);
void mha_free(MultiHeadAttention* mha);

/* ── Positional Encoding (sinusoidal, Vaswani et al. 2017) ────
 * PE(pos, 2i)   = sin(pos / 10000^{2i/d_model})
 * PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model}) */
Tensor* sinusoidal_positional_encoding(int seq_len, int d_model);

/* ── Weight Initialization (L4: Glorot & Bengio 2010, He et al. 2015)
 * Xavier uniform: W ~ U[-√(6/(fan_in+fan_out)), √(6/(fan_in+fan_out))]
 * Kaiming normal: W ~ N(0, √(2/fan_in)) */

typedef enum {
    INIT_XAVIER_UNIFORM,
    INIT_XAVIER_NORMAL,
    INIT_KAIMING_UNIFORM,
    INIT_KAIMING_NORMAL,
    INIT_UNIFORM,
    INIT_NORMAL
} InitMethod;

void tensor_init(Tensor* t, InitMethod method, int fan_in, int fan_out);

#endif /* RNN_LAYERS_H */
