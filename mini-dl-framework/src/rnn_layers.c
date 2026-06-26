#include "rnn_layers.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

/* ═══════════════════════════════════════════════════════════════════════
 * Embedding Layer (L2): Discrete->continuous mapping.
 * L4: Mikolov et al. 2013 - linear regularities: king-man+woman ~ queen
 * ═══════════════════════════════════════════════════════════════════════ */

Embedding* embedding_create(int vocab_size, int embed_dim, float padding_idx) {
    Embedding* emb = (Embedding*)malloc(sizeof(Embedding));
    emb->vocab_size = vocab_size;
    emb->embed_dim = embed_dim;
    emb->padding_idx = padding_idx;
    emb->weight = tensor_create_randomn((int[]){vocab_size, embed_dim}, 2);
    emb->weight_grad = tensor_create_zeros((int[]){vocab_size, embed_dim}, 2);
    if (padding_idx >= 0 && (int)padding_idx < vocab_size) {
        int pidx = (int)padding_idx;
        for (int j = 0; j < embed_dim; j++)
            emb->weight->data[pidx * embed_dim + j] = 0.0f;
    }
    return emb;
}

Tensor* embedding_forward(Embedding* emb, Tensor* indices) {
    int batch = indices->dims[0];
    int seq_len = (indices->ndim > 1) ? indices->dims[1] : 1;
    int embed_dim = emb->embed_dim;
    int total = batch * seq_len;
    Tensor* out;
    if (indices->ndim > 1)
        out = tensor_create((int[]){batch, seq_len, embed_dim}, 3);
    else
        out = tensor_create((int[]){batch, embed_dim}, 2);
    for (int i = 0; i < total; i++) {
        int idx = (int)(indices->data[i] + 0.5f);
        if (idx < 0 || idx >= emb->vocab_size) idx = 0;
        float* src = emb->weight->data + idx * embed_dim;
        float* dst = out->data + i * embed_dim;
        memcpy(dst, src, sizeof(float) * embed_dim);
    }
    return out;
}

Tensor* embedding_backward(Embedding* emb, Tensor* grad_output,
                          Tensor* indices) {
    int batch = indices->dims[0];
    int seq_len = (indices->ndim > 1) ? indices->dims[1] : 1;
    int total = batch * seq_len;
    int embed_dim = emb->embed_dim;
    for (int i = 0; i < total; i++) {
        int idx = (int)(indices->data[i] + 0.5f);
        if (idx < 0 || idx >= emb->vocab_size) idx = 0;
        float* src_grad = grad_output->data + i * embed_dim;
        float* dst_grad = emb->weight_grad->data + idx * embed_dim;
        for (int j = 0; j < embed_dim; j++)
            dst_grad[j] += src_grad[j];
    }
    Tensor* d_input = tensor_create_zeros(grad_output->dims, grad_output->ndim);
    return d_input;
}

void embedding_free(Embedding* emb) {
    if (!emb) return;
    tensor_free(emb->weight);
    tensor_free(emb->weight_grad);
    free(emb);
}

/* ═══════════════════════════════════════════════════════════════════════
 * RNN Cell — Elman Network (L5: BPTT)
 * h_t = tanh(W_hh*h_{t-1} + W_ih*x_t + b)
 * d(tanh(x))/dx = 1 - tanh^2(x)
 * ═══════════════════════════════════════════════════════════════════════ */

RNNCell* rnn_cell_create(int input_size, int hidden_size) {
    RNNCell* cell = (RNNCell*)malloc(sizeof(RNNCell));
    cell->input_size = input_size;
    cell->hidden_size = hidden_size;
    cell->W_ih = tensor_create_randomn((int[]){hidden_size, input_size}, 2);
    cell->W_hh = tensor_create_randomn((int[]){hidden_size, hidden_size}, 2);
    cell->b_ih = tensor_create_zeros((int[]){hidden_size}, 1);
    cell->b_hh = tensor_create_zeros((int[]){hidden_size}, 1);
    cell->W_ih_grad = tensor_create_zeros((int[]){hidden_size, input_size}, 2);
    cell->W_hh_grad = tensor_create_zeros((int[]){hidden_size, hidden_size}, 2);
    cell->b_ih_grad = tensor_create_zeros((int[]){hidden_size}, 1);
    cell->b_hh_grad = tensor_create_zeros((int[]){hidden_size}, 1);
    return cell;
}

Tensor* rnn_cell_forward(RNNCell* cell, Tensor* input, Tensor* hidden) {
    int isize = cell->input_size;
    int hsize = cell->hidden_size;
    Tensor* out = tensor_create((int[]){hsize}, 1);
    for (int h = 0; h < hsize; h++) {
        float sum = 0;
        for (int i = 0; i < isize; i++)
            sum += cell->W_ih->data[h * isize + i] * input->data[i];
        for (int i = 0; i < hsize; i++)
            sum += cell->W_hh->data[h * hsize + i] * hidden->data[i];
        sum += cell->b_ih->data[h] + cell->b_hh->data[h];
        out->data[h] = tanhf(sum);
    }
    return out;
}

void rnn_cell_backward(RNNCell* cell, Tensor* grad_output,
                       Tensor* input, Tensor* hidden,
                       Tensor** d_input, Tensor** d_hidden) {
    int isize = cell->input_size;
    int hsize = cell->hidden_size;
    *d_input  = tensor_create_zeros((int[]){isize}, 1);
    *d_hidden = tensor_create_zeros((int[]){hsize}, 1);
    for (int h = 0; h < hsize; h++) {
        float pre_act = 0;
        for (int i = 0; i < isize; i++)
            pre_act += cell->W_ih->data[h * isize + i] * input->data[i];
        for (int i = 0; i < hsize; i++)
            pre_act += cell->W_hh->data[h * hsize + i] * hidden->data[i];
        pre_act += cell->b_ih->data[h] + cell->b_hh->data[h];
        float h_val = tanhf(pre_act);
        float delta = grad_output->data[h] * (1.0f - h_val * h_val);
        for (int i = 0; i < isize; i++) {
            cell->W_ih_grad->data[h * isize + i] += delta * input->data[i];
            (*d_input)->data[i] += delta * cell->W_ih->data[h * isize + i];
        }
        for (int i = 0; i < hsize; i++) {
            cell->W_hh_grad->data[h * hsize + i] += delta * hidden->data[i];
            (*d_hidden)->data[i] += delta * cell->W_hh->data[h * hsize + i];
        }
        cell->b_ih_grad->data[h] += delta;
        cell->b_hh_grad->data[h] += delta;
    }
}

void rnn_cell_free(RNNCell* cell) {
    if (!cell) return;
    tensor_free(cell->W_ih);  tensor_free(cell->W_ih_grad);
    tensor_free(cell->W_hh);  tensor_free(cell->W_hh_grad);
    tensor_free(cell->b_ih);  tensor_free(cell->b_ih_grad);
    tensor_free(cell->b_hh);  tensor_free(cell->b_hh_grad);
    free(cell);
}

/* ═══════════════════════════════════════════════════════════════════════
 * LSTM Cell — Hochreiter & Schmidhuber 1997
 * L4: CEC ensures gradient flow via c_t = f_t*c_{t-1} + i_t*g_t
 * L5: Four-gate structure with sigmoid/tanh activations
 * ═══════════════════════════════════════════════════════════════════════ */

LSTMCell* lstm_cell_create(int input_size, int hidden_size) {
    LSTMCell* c = (LSTMCell*)malloc(sizeof(LSTMCell));
    c->input_size = input_size;
    c->hidden_size = hidden_size;
    int h = hidden_size, inp = input_size;
    c->W_f = tensor_create_randomn((int[]){h, inp}, 2);
    c->W_f_grad = tensor_create_zeros((int[]){h, inp}, 2);
    c->W_i = tensor_create_randomn((int[]){h, inp}, 2);
    c->W_i_grad = tensor_create_zeros((int[]){h, inp}, 2);
    c->W_g = tensor_create_randomn((int[]){h, inp}, 2);
    c->W_g_grad = tensor_create_zeros((int[]){h, inp}, 2);
    c->W_o = tensor_create_randomn((int[]){h, inp}, 2);
    c->W_o_grad = tensor_create_zeros((int[]){h, inp}, 2);
    c->U_f = tensor_create_randomn((int[]){h, h}, 2);
    c->U_f_grad = tensor_create_zeros((int[]){h, h}, 2);
    c->U_i = tensor_create_randomn((int[]){h, h}, 2);
    c->U_i_grad = tensor_create_zeros((int[]){h, h}, 2);
    c->U_g = tensor_create_randomn((int[]){h, h}, 2);
    c->U_g_grad = tensor_create_zeros((int[]){h, h}, 2);
    c->U_o = tensor_create_randomn((int[]){h, h}, 2);
    c->U_o_grad = tensor_create_zeros((int[]){h, h}, 2);
    c->b_f = tensor_create_zeros((int[]){h}, 1);
    c->b_f_grad = tensor_create_zeros((int[]){h}, 1);
    c->b_i = tensor_create_zeros((int[]){h}, 1);
    c->b_i_grad = tensor_create_zeros((int[]){h}, 1);
    c->b_g = tensor_create_zeros((int[]){h}, 1);
    c->b_g_grad = tensor_create_zeros((int[]){h}, 1);
    c->b_o = tensor_create_zeros((int[]){h}, 1);
    c->b_o_grad = tensor_create_zeros((int[]){h}, 1);
    c->input_cache = NULL;  c->hidden_cache = NULL;
    c->cell_cache = NULL;   c->f_gate_cache = NULL;
    c->i_gate_cache = NULL; c->g_gate_cache = NULL;
    c->o_gate_cache = NULL;
    return c;
}

void lstm_cell_forward(LSTMCell* cell, Tensor* input,
                       Tensor* hidden, Tensor* cell_state,
                       Tensor** out_hidden, Tensor** out_cell) {
    int isize = cell->input_size;
    int hsize = cell->hidden_size;
    int ds = hsize;
    if (cell->input_cache)  tensor_free(cell->input_cache);
    if (cell->hidden_cache) tensor_free(cell->hidden_cache);
    if (cell->cell_cache)   tensor_free(cell->cell_cache);
    if (cell->f_gate_cache) tensor_free(cell->f_gate_cache);
    if (cell->i_gate_cache) tensor_free(cell->i_gate_cache);
    if (cell->g_gate_cache) tensor_free(cell->g_gate_cache);
    if (cell->o_gate_cache) tensor_free(cell->o_gate_cache);
    cell->input_cache  = tensor_copy(input);
    cell->hidden_cache = tensor_copy(hidden);
    cell->cell_cache   = tensor_copy(cell_state);
    *out_hidden = tensor_create((int[]){ds}, 1);
    *out_cell   = tensor_create((int[]){ds}, 1);
    cell->f_gate_cache = tensor_create((int[]){ds}, 1);
    cell->i_gate_cache = tensor_create((int[]){ds}, 1);
    cell->g_gate_cache = tensor_create((int[]){ds}, 1);
    cell->o_gate_cache = tensor_create((int[]){ds}, 1);
    for (int d = 0; d < ds; d++) {
        float f_pre = 0, i_pre = 0, g_pre = 0, o_pre = 0;
        for (int j = 0; j < isize; j++) {
            f_pre += cell->W_f->data[d * isize + j] * input->data[j];
            i_pre += cell->W_i->data[d * isize + j] * input->data[j];
            g_pre += cell->W_g->data[d * isize + j] * input->data[j];
            o_pre += cell->W_o->data[d * isize + j] * input->data[j];
        }
        for (int j = 0; j < hsize; j++) {
            f_pre += cell->U_f->data[d * hsize + j] * hidden->data[j];
            i_pre += cell->U_i->data[d * hsize + j] * hidden->data[j];
            g_pre += cell->U_g->data[d * hsize + j] * hidden->data[j];
            o_pre += cell->U_o->data[d * hsize + j] * hidden->data[j];
        }
        f_pre += cell->b_f->data[d]; i_pre += cell->b_i->data[d];
        g_pre += cell->b_g->data[d]; o_pre += cell->b_o->data[d];
        float f_t = 1.0f / (1.0f + expf(-f_pre));
        float i_t = 1.0f / (1.0f + expf(-i_pre));
        float g_t = tanhf(g_pre);
        float o_t = 1.0f / (1.0f + expf(-o_pre));
        cell->f_gate_cache->data[d] = f_t;
        cell->i_gate_cache->data[d] = i_t;
        cell->g_gate_cache->data[d] = g_t;
        cell->o_gate_cache->data[d] = o_t;
        (*out_cell)->data[d] = f_t * cell_state->data[d] + i_t * g_t;
        (*out_hidden)->data[d] = o_t * tanhf((*out_cell)->data[d]);
    }
}

void lstm_cell_backward(LSTMCell* cell, Tensor* grad_h, Tensor* grad_c,
                         Tensor** d_input, Tensor** d_hidden,
                         Tensor** d_cell) {
    int isize = cell->input_size;
    int hsize = cell->hidden_size;
    int ds = hsize;
    Tensor* di_out = tensor_create_zeros((int[]){isize}, 1);
    Tensor* dh_out = tensor_create_zeros((int[]){hsize}, 1);
    Tensor* dc_out = tensor_create_zeros((int[]){hsize}, 1);
    for (int d = 0; d < ds; d++) {
        float dc = grad_c ? grad_c->data[d] : 0.0f;
        float dh = grad_h ? grad_h->data[d] : 0.0f;
        float c_t = cell->cell_cache->data[d];
        float o_t = cell->o_gate_cache->data[d];
        float tanh_c = tanhf(c_t);
        float dtanh_c = 1.0f - tanh_c * tanh_c;
        float d_o = dh * tanh_c;
        float d_c = dh * o_t * dtanh_c + dc;
        float f_t = cell->f_gate_cache->data[d];
        float i_t = cell->i_gate_cache->data[d];
        float g_t = cell->g_gate_cache->data[d];
        float d_f = d_c * cell->cell_cache->data[d];
        float d_i = d_c * g_t;
        float d_g = d_c * i_t;
        float d_f_pre = d_f * f_t * (1.0f - f_t);
        float d_i_pre = d_i * i_t * (1.0f - i_t);
        float d_g_pre = d_g * (1.0f - g_t * g_t);
        float d_o_pre = d_o * o_t * (1.0f - o_t);
        for (int j = 0; j < isize; j++) {
            float x = cell->input_cache->data[j];
            cell->W_f_grad->data[d * isize + j] += d_f_pre * x;
            cell->W_i_grad->data[d * isize + j] += d_i_pre * x;
            cell->W_g_grad->data[d * isize + j] += d_g_pre * x;
            cell->W_o_grad->data[d * isize + j] += d_o_pre * x;
            di_out->data[j] += d_f_pre * cell->W_f->data[d * isize + j]
                             + d_i_pre * cell->W_i->data[d * isize + j]
                             + d_g_pre * cell->W_g->data[d * isize + j]
                             + d_o_pre * cell->W_o->data[d * isize + j];
        }
        for (int j = 0; j < hsize; j++) {
            float hp = cell->hidden_cache->data[j];
            cell->U_f_grad->data[d * hsize + j] += d_f_pre * hp;
            cell->U_i_grad->data[d * hsize + j] += d_i_pre * hp;
            cell->U_g_grad->data[d * hsize + j] += d_g_pre * hp;
            cell->U_o_grad->data[d * hsize + j] += d_o_pre * hp;
            dh_out->data[j] += d_f_pre * cell->U_f->data[d * hsize + j]
                             + d_i_pre * cell->U_i->data[d * hsize + j]
                             + d_g_pre * cell->U_g->data[d * hsize + j]
                             + d_o_pre * cell->U_o->data[d * hsize + j];
        }
        cell->b_f_grad->data[d] += d_f_pre;
        cell->b_i_grad->data[d] += d_i_pre;
        cell->b_g_grad->data[d] += d_g_pre;
        cell->b_o_grad->data[d] += d_o_pre;
        dc_out->data[d] = d_c * f_t;
    }
    *d_input = di_out; *d_hidden = dh_out; *d_cell = dc_out;
}

void lstm_cell_free(LSTMCell* cell) {
    if (!cell) return;
    tensor_free(cell->W_f); tensor_free(cell->W_f_grad);
    tensor_free(cell->W_i); tensor_free(cell->W_i_grad);
    tensor_free(cell->W_g); tensor_free(cell->W_g_grad);
    tensor_free(cell->W_o); tensor_free(cell->W_o_grad);
    tensor_free(cell->U_f); tensor_free(cell->U_f_grad);
    tensor_free(cell->U_i); tensor_free(cell->U_i_grad);
    tensor_free(cell->U_g); tensor_free(cell->U_g_grad);
    tensor_free(cell->U_o); tensor_free(cell->U_o_grad);
    tensor_free(cell->b_f); tensor_free(cell->b_f_grad);
    tensor_free(cell->b_i); tensor_free(cell->b_i_grad);
    tensor_free(cell->b_g); tensor_free(cell->b_g_grad);
    tensor_free(cell->b_o); tensor_free(cell->b_o_grad);
    if (cell->input_cache)  tensor_free(cell->input_cache);
    if (cell->hidden_cache) tensor_free(cell->hidden_cache);
    if (cell->cell_cache)   tensor_free(cell->cell_cache);
    if (cell->f_gate_cache) tensor_free(cell->f_gate_cache);
    if (cell->i_gate_cache) tensor_free(cell->i_gate_cache);
    if (cell->g_gate_cache) tensor_free(cell->g_gate_cache);
    if (cell->o_gate_cache) tensor_free(cell->o_gate_cache);
    free(cell);
}

/* ═══════════════════════════════════════════════════════════════════════
 * GRU Cell — Cho et al. 2014
 * L4: Simpler gating (reset + update) than LSTM, comparable performance.
 * L5: Update gate controls retention; reset gate controls candidate.
 * ═══════════════════════════════════════════════════════════════════════ */

GRUCell* gru_cell_create(int input_size, int hidden_size) {
    GRUCell* c = (GRUCell*)malloc(sizeof(GRUCell));
    c->input_size = input_size;
    c->hidden_size = hidden_size;
    int h = hidden_size, inp = input_size;
    c->W_r = tensor_create_randomn((int[]){h, inp}, 2);
    c->W_r_grad = tensor_create_zeros((int[]){h, inp}, 2);
    c->W_z = tensor_create_randomn((int[]){h, inp}, 2);
    c->W_z_grad = tensor_create_zeros((int[]){h, inp}, 2);
    c->W_n = tensor_create_randomn((int[]){h, inp}, 2);
    c->W_n_grad = tensor_create_zeros((int[]){h, inp}, 2);
    c->U_r = tensor_create_randomn((int[]){h, h}, 2);
    c->U_r_grad = tensor_create_zeros((int[]){h, h}, 2);
    c->U_z = tensor_create_randomn((int[]){h, h}, 2);
    c->U_z_grad = tensor_create_zeros((int[]){h, h}, 2);
    c->U_n = tensor_create_randomn((int[]){h, h}, 2);
    c->U_n_grad = tensor_create_zeros((int[]){h, h}, 2);
    c->b_r = tensor_create_zeros((int[]){h}, 1);
    c->b_r_grad = tensor_create_zeros((int[]){h}, 1);
    c->b_z = tensor_create_zeros((int[]){h}, 1);
    c->b_z_grad = tensor_create_zeros((int[]){h}, 1);
    c->b_n = tensor_create_zeros((int[]){h}, 1);
    c->b_n_grad = tensor_create_zeros((int[]){h}, 1);
    c->input_cache = NULL;  c->hidden_cache = NULL;
    c->r_gate_cache = NULL; c->z_gate_cache = NULL;
    c->n_gate_cache = NULL;
    return c;
}

void gru_cell_forward(GRUCell* cell, Tensor* input, Tensor* hidden,
                      Tensor** out_hidden) {
    int isize = cell->input_size;
    int hsize = cell->hidden_size;
    int ds = hsize;
    if (cell->input_cache)  tensor_free(cell->input_cache);
    if (cell->hidden_cache) tensor_free(cell->hidden_cache);
    if (cell->r_gate_cache) tensor_free(cell->r_gate_cache);
    if (cell->z_gate_cache) tensor_free(cell->z_gate_cache);
    if (cell->n_gate_cache) tensor_free(cell->n_gate_cache);
    cell->input_cache  = tensor_copy(input);
    cell->hidden_cache = tensor_copy(hidden);
    *out_hidden = tensor_create((int[]){ds}, 1);
    cell->r_gate_cache = tensor_create((int[]){ds}, 1);
    cell->z_gate_cache = tensor_create((int[]){ds}, 1);
    cell->n_gate_cache = tensor_create((int[]){ds}, 1);
    for (int d = 0; d < ds; d++) {
        float r_pre = 0;
        for (int j = 0; j < isize; j++)
            r_pre += cell->W_r->data[d * isize + j] * input->data[j];
        for (int j = 0; j < hsize; j++)
            r_pre += cell->U_r->data[d * hsize + j] * hidden->data[j];
        r_pre += cell->b_r->data[d];
        float r_t = 1.0f / (1.0f + expf(-r_pre));
        float z_pre = 0;
        for (int j = 0; j < isize; j++)
            z_pre += cell->W_z->data[d * isize + j] * input->data[j];
        for (int j = 0; j < hsize; j++)
            z_pre += cell->U_z->data[d * hsize + j] * hidden->data[j];
        z_pre += cell->b_z->data[d];
        float z_t = 1.0f / (1.0f + expf(-z_pre));
        float n_pre = 0;
        for (int j = 0; j < isize; j++)
            n_pre += cell->W_n->data[d * isize + j] * input->data[j];
        for (int j = 0; j < hsize; j++)
            n_pre += r_t * cell->U_n->data[d * hsize + j] * hidden->data[j];
        n_pre += cell->b_n->data[d];
        float n_t = tanhf(n_pre);
        cell->r_gate_cache->data[d] = r_t;
        cell->z_gate_cache->data[d] = z_t;
        cell->n_gate_cache->data[d] = n_t;
        (*out_hidden)->data[d] = (1.0f - z_t) * n_t + z_t * hidden->data[d];
    }
}

void gru_cell_backward(GRUCell* cell, Tensor* grad_output,
                       Tensor** d_input, Tensor** d_hidden) {
    int isize = cell->input_size;
    int hsize = cell->hidden_size;
    int ds = hsize;
    Tensor* di_out = tensor_create_zeros((int[]){isize}, 1);
    Tensor* dh_out = tensor_create_zeros((int[]){hsize}, 1);
    for (int d = 0; d < ds; d++) {
        float dh = grad_output->data[d];
        float z_t = cell->z_gate_cache->data[d];
        float r_t = cell->r_gate_cache->data[d];
        float n_t = cell->n_gate_cache->data[d];
        float d_z = dh * (cell->hidden_cache->data[d] - n_t);
        float d_n = dh * (1.0f - z_t);
        float d_z_pre = d_z * z_t * (1.0f - z_t);
        float d_n_pre = d_n * (1.0f - n_t * n_t);
        float r_part = 0;
        for (int j = 0; j < hsize; j++)
            r_part += cell->U_n->data[d * hsize + j] * cell->hidden_cache->data[j];
        float d_r = d_n_pre * r_part;
        float d_r_pre = d_r * r_t * (1.0f - r_t);
        for (int j = 0; j < isize; j++) {
            float x = cell->input_cache->data[j];
            cell->W_r_grad->data[d * isize + j] += d_r_pre * x;
            cell->W_z_grad->data[d * isize + j] += d_z_pre * x;
            cell->W_n_grad->data[d * isize + j] += d_n_pre * x;
            di_out->data[j] += d_r_pre * cell->W_r->data[d * isize + j]
                             + d_z_pre * cell->W_z->data[d * isize + j]
                             + d_n_pre * cell->W_n->data[d * isize + j];
        }
        for (int j = 0; j < hsize; j++) {
            float hp = cell->hidden_cache->data[j];
            cell->U_r_grad->data[d * hsize + j] += d_r_pre * hp;
            cell->U_z_grad->data[d * hsize + j] += d_z_pre * hp;
            cell->U_n_grad->data[d * hsize + j] += d_n_pre * r_t * hp;
            dh_out->data[j] += d_r_pre * cell->U_r->data[d * hsize + j]
                             + d_z_pre * cell->U_z->data[d * hsize + j]
                             + d_n_pre * r_t * cell->U_n->data[d * hsize + j];
        }
        cell->b_r_grad->data[d] += d_r_pre;
        cell->b_z_grad->data[d] += d_z_pre;
        cell->b_n_grad->data[d] += d_n_pre;
        dh_out->data[d] += dh * z_t;
    }
    *d_input = di_out; *d_hidden = dh_out;
}

void gru_cell_free(GRUCell* cell) {
    if (!cell) return;
    tensor_free(cell->W_r); tensor_free(cell->W_r_grad);
    tensor_free(cell->W_z); tensor_free(cell->W_z_grad);
    tensor_free(cell->W_n); tensor_free(cell->W_n_grad);
    tensor_free(cell->U_r); tensor_free(cell->U_r_grad);
    tensor_free(cell->U_z); tensor_free(cell->U_z_grad);
    tensor_free(cell->U_n); tensor_free(cell->U_n_grad);
    tensor_free(cell->b_r); tensor_free(cell->b_r_grad);
    tensor_free(cell->b_z); tensor_free(cell->b_z_grad);
    tensor_free(cell->b_n); tensor_free(cell->b_n_grad);
    if (cell->input_cache)  tensor_free(cell->input_cache);
    if (cell->hidden_cache) tensor_free(cell->hidden_cache);
    if (cell->r_gate_cache) tensor_free(cell->r_gate_cache);
    if (cell->z_gate_cache) tensor_free(cell->z_gate_cache);
    if (cell->n_gate_cache) tensor_free(cell->n_gate_cache);
    free(cell);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Multi-Head Self-Attention (Vaswani et al. 2017)
 * L4: Scaled dot-product: softmax(QK^T/sqrt(d_k))*V
 * L8: Parallelizable O(n^2*d). Core of Transformer architecture.
 * ═══════════════════════════════════════════════════════════════════════ */

MultiHeadAttention* mha_create(int embed_dim, int num_heads, float dropout_p) {
    MultiHeadAttention* m = (MultiHeadAttention*)malloc(sizeof(MultiHeadAttention));
    m->embed_dim = embed_dim;
    m->num_heads = num_heads;
    m->head_dim = embed_dim / num_heads;
    m->dropout_p = dropout_p;
    m->is_training = true;
    int d = embed_dim;
    m->W_Q = tensor_create_randomn((int[]){d, d}, 2);
    m->W_K = tensor_create_randomn((int[]){d, d}, 2);
    m->W_V = tensor_create_randomn((int[]){d, d}, 2);
    m->W_O = tensor_create_randomn((int[]){d, d}, 2);
    m->W_Q_grad = tensor_create_zeros((int[]){d, d}, 2);
    m->W_K_grad = tensor_create_zeros((int[]){d, d}, 2);
    m->W_V_grad = tensor_create_zeros((int[]){d, d}, 2);
    m->W_O_grad = tensor_create_zeros((int[]){d, d}, 2);
    m->Q_cache = NULL; m->K_cache = NULL; m->V_cache = NULL;
    m->attn_cache = NULL; m->mask_cache = NULL;
    return m;
}

Tensor* mha_forward(MultiHeadAttention* mha, Tensor* query,
                    Tensor* key, Tensor* value, Tensor* mask) {
    if (mha->Q_cache) tensor_free(mha->Q_cache);
    if (mha->K_cache) tensor_free(mha->K_cache);
    if (mha->V_cache) tensor_free(mha->V_cache);
    if (mha->attn_cache) tensor_free(mha->attn_cache);
    mha->Q_cache = tensor_copy(query);
    mha->K_cache = tensor_copy(key);
    mha->V_cache = tensor_copy(value);
    if (mask) {
        if (mha->mask_cache) tensor_free(mha->mask_cache);
        mha->mask_cache = tensor_copy(mask);
    }
    Tensor* Q_proj = tensor_matmul(query, mha->W_Q);
    Tensor* K_proj = tensor_matmul(key, mha->W_K);
    Tensor* V_proj = tensor_matmul(value, mha->W_V);
    int seq_len = query->dims[0];
    int d = mha->embed_dim;
    int h = mha->num_heads;
    int dk = mha->head_dim;
    float scale = 1.0f / sqrtf((float)dk);
    /* Split into heads: [seq, d] -> [h, seq, dk] */
    Tensor* Q_h = tensor_create((int[]){h, seq_len, dk}, 3);
    Tensor* K_h = tensor_create((int[]){h, seq_len, dk}, 3);
    Tensor* V_h = tensor_create((int[]){h, seq_len, dk}, 3);
    for (int hi = 0; hi < h; hi++) {
        for (int s = 0; s < seq_len; s++) {
            for (int dki = 0; dki < dk; dki++) {
                int src = s * d + hi * dk + dki;
                int dst = hi * seq_len * dk + s * dk + dki;
                Q_h->data[dst] = Q_proj->data[src];
                K_h->data[dst] = K_proj->data[src];
                V_h->data[dst] = V_proj->data[src];
            }
        }
    }
    /* Scores: Q*K^T / sqrt(dk) -> [h, seq, seq] */
    Tensor* scores = tensor_create((int[]){h, seq_len, seq_len}, 3);
    for (int hi = 0; hi < h; hi++) {
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < seq_len; j++) {
                float dot = 0;
                for (int ki = 0; ki < dk; ki++)
                    dot += Q_h->data[hi * seq_len * dk + i * dk + ki]
                         * K_h->data[hi * seq_len * dk + j * dk + ki];
                float s = dot * scale;
                if (mask && mask->data[i * seq_len + j] < 0.5f) s = -1e9f;
                scores->data[hi * seq_len * seq_len + i * seq_len + j] = s;
            }
        }
    }
    /* Softmax per head per query */
    Tensor* attn_w = tensor_create((int[]){h, seq_len, seq_len}, 3);
    for (int hi = 0; hi < h; hi++) {
        for (int i = 0; i < seq_len; i++) {
            float max_v = -1e9f;
            for (int j = 0; j < seq_len; j++) {
                float s = scores->data[hi * seq_len * seq_len + i * seq_len + j];
                if (s > max_v) max_v = s;
            }
            float sum = 0;
            for (int j = 0; j < seq_len; j++) {
                int idx = hi * seq_len * seq_len + i * seq_len + j;
                attn_w->data[idx] = expf(scores->data[idx] - max_v);
                sum += attn_w->data[idx];
            }
            for (int j = 0; j < seq_len; j++) {
                int idx = hi * seq_len * seq_len + i * seq_len + j;
                attn_w->data[idx] /= (sum + 1e-12f);
            }
        }
    }
    mha->attn_cache = attn_w;
    /* Weighted sum: attn * V_h -> [h, seq, dk] */
    Tensor* a_out = tensor_create((int[]){h, seq_len, dk}, 3);
    for (int hi = 0; hi < h; hi++) {
        for (int i = 0; i < seq_len; i++) {
            for (int ki = 0; ki < dk; ki++) {
                float sum = 0;
                for (int j = 0; j < seq_len; j++)
                    sum += attn_w->data[hi * seq_len * seq_len + i * seq_len + j]
                         * V_h->data[hi * seq_len * dk + j * dk + ki];
                a_out->data[hi * seq_len * dk + i * dk + ki] = sum;
            }
        }
    }
    /* Concat heads: [h, seq, dk] -> [seq, d] */
    Tensor* concat = tensor_create((int[]){seq_len, d}, 2);
    for (int s = 0; s < seq_len; s++)
        for (int hi = 0; hi < h; hi++)
            for (int dki = 0; dki < dk; dki++)
                concat->data[s * d + hi * dk + dki] =
                    a_out->data[hi * seq_len * dk + s * dk + dki];
    Tensor* out = tensor_matmul(concat, mha->W_O);
    tensor_free(Q_proj); tensor_free(K_proj); tensor_free(V_proj);
    tensor_free(Q_h); tensor_free(K_h); tensor_free(V_h);
    tensor_free(scores); tensor_free(a_out); tensor_free(concat);
    return out;
}

Tensor* mha_backward(MultiHeadAttention* mha, Tensor* grad_output) {
    int seq_len = grad_output->dims[0];
    int d = mha->embed_dim;
    int dk = mha->head_dim;
    /* Accumulate W_O gradient: use cached projections */
    Tensor* W_O_t = tensor_transpose(mha->W_O, 0, 1);
    Tensor* d_concat = tensor_matmul(grad_output, W_O_t);
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
            float g = 0;
            for (int s = 0; s < seq_len; s++) {
                float concat_val = 0;
                int hi = i / dk;
                if (hi < mha->num_heads) {
                    for (int ki = 0; ki < dk; ki++)
                        concat_val += mha->V_cache->data[s * d + hi * dk + ki];
                    concat_val /= (float)dk;
                }
                g += concat_val * grad_output->data[s * d + j];
            }
            mha->W_O_grad->data[i * d + j] += g;
        }
    }
    /* Backprop through Q projection (simplified) */
    Tensor* d_query = tensor_create_zeros((int[]){seq_len, d}, 2);
    for (int s = 0; s < seq_len; s++)
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                d_query->data[s * d + i] += d_concat->data[s * d + j]
                    * mha->W_Q->data[j * d + i] * 0.01f;
    tensor_free(W_O_t);
    tensor_free(d_concat);
    return d_query;
}

void mha_free(MultiHeadAttention* mha) {
    if (!mha) return;
    tensor_free(mha->W_Q); tensor_free(mha->W_Q_grad);
    tensor_free(mha->W_K); tensor_free(mha->W_K_grad);
    tensor_free(mha->W_V); tensor_free(mha->W_V_grad);
    tensor_free(mha->W_O); tensor_free(mha->W_O_grad);
    if (mha->Q_cache)    tensor_free(mha->Q_cache);
    if (mha->K_cache)    tensor_free(mha->K_cache);
    if (mha->V_cache)    tensor_free(mha->V_cache);
    if (mha->attn_cache) tensor_free(mha->attn_cache);
    if (mha->mask_cache) tensor_free(mha->mask_cache);
    free(mha);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Positional Encoding (L8: Transformer building block)
 * PE(pos,2i)=sin(pos/10000^(2i/d)), PE(pos,2i+1)=cos(pos/10000^(2i/d))
 * ═══════════════════════════════════════════════════════════════════════ */

Tensor* sinusoidal_positional_encoding(int seq_len, int d_model) {
    Tensor* pe = tensor_create((int[]){seq_len, d_model}, 2);
    for (int pos = 0; pos < seq_len; pos++) {
        for (int i = 0; i < d_model; i++) {
            float angle = (float)pos / powf(10000.0f,
                (float)((i / 2) * 2) / (float)d_model);
            pe->data[pos * d_model + i] = (i % 2 == 0) ? sinf(angle) : cosf(angle);
        }
    }
    return pe;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Weight Initialization (L4: Glorot & Bengio 2010, He et al. 2015)
 * Xavier: Var(W) = 2/(fan_in+fan_out), Kaiming: Var(W) = 2/fan_in
 * Both derived from forward/backward signal variance preservation.
 * ═══════════════════════════════════════════════════════════════════════ */

void tensor_init(Tensor* t, InitMethod method, int fan_in, int fan_out) {
    float limit, std;
    switch (method) {
    case INIT_XAVIER_UNIFORM:
        limit = sqrtf(6.0f / (float)(fan_in + fan_out));
        for (int i = 0; i < t->size; i++) {
            float r = (float)rand() / (float)RAND_MAX;
            t->data[i] = -limit + 2.0f * limit * r;
        }
        break;
    case INIT_XAVIER_NORMAL:
        std = sqrtf(2.0f / (float)(fan_in + fan_out));
        for (int i = 0; i < t->size; i++) {
            float u1 = (float)rand() / (float)RAND_MAX;
            float u2 = (float)rand() / (float)RAND_MAX;
            t->data[i] = std * sqrtf(-2.0f * logf(u1 + 1e-12f))
                       * cosf(6.2831853f * u2);
        }
        break;
    case INIT_KAIMING_UNIFORM:
        limit = sqrtf(6.0f / (float)fan_in);
        for (int i = 0; i < t->size; i++) {
            float r = (float)rand() / (float)RAND_MAX;
            t->data[i] = -limit + 2.0f * limit * r;
        }
        break;
    case INIT_KAIMING_NORMAL:
        std = sqrtf(2.0f / (float)fan_in);
        for (int i = 0; i < t->size; i++) {
            float u1 = (float)rand() / (float)RAND_MAX;
            float u2 = (float)rand() / (float)RAND_MAX;
            t->data[i] = std * sqrtf(-2.0f * logf(u1 + 1e-12f))
                       * cosf(6.2831853f * u2);
        }
        break;
    case INIT_UNIFORM:
        limit = 0.1f;
        for (int i = 0; i < t->size; i++) {
            float r = (float)rand() / (float)RAND_MAX;
            t->data[i] = -limit + 2.0f * limit * r;
        }
        break;
    case INIT_NORMAL:
        std = 0.02f;
        for (int i = 0; i < t->size; i++) {
            float u1 = (float)rand() / (float)RAND_MAX;
            float u2 = (float)rand() / (float)RAND_MAX;
            t->data[i] = std * sqrtf(-2.0f * logf(u1 + 1e-12f))
                       * cosf(6.2831853f * u2);
        }
        break;
    }
}
