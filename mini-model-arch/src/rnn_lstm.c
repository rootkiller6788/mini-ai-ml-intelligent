#include "rnn_lstm.h"

float sigmoid_act(float x) { return 1.0f / (1.0f + expf(-x)); }
float tanh_act(float x) { return tanhf(x); }

float *linear_rnn(const float *w, const float *x, const float *b, int rows, int cols) {
    float *y = (float *)calloc(rows, sizeof(float));
    for (int i = 0; i < rows; i++) {
        y[i] = b ? b[i] : 0.0f;
        for (int j = 0; j < cols; j++)
            y[i] += w[i * cols + j] * x[j];
    }
    return y;
}

RNNCell *rnn_cell_create(int in_sz, int hid_sz) {
    RNNCell *c = (RNNCell *)malloc(sizeof(RNNCell));
    c->input_size = in_sz; c->hidden_size = hid_sz;
    int wh_sz = hid_sz * hid_sz, wx_sz = hid_sz * in_sz;
    c->W_xh = (float *)malloc(wx_sz * sizeof(float));
    c->W_hh = (float *)malloc(wh_sz * sizeof(float));
    c->b_h  = (float *)calloc(hid_sz, sizeof(float));
    for (int i = 0; i < wx_sz; i++)
        c->W_xh[i] = ((float)rand() / RAND_MAX - 0.5f) * sqrtf(2.0f / in_sz);
    for (int i = 0; i < wh_sz; i++)
        c->W_hh[i] = ((float)rand() / RAND_MAX - 0.5f) * sqrtf(2.0f / hid_sz);
    return c;
}
void rnn_cell_free(RNNCell *c) { free(c->W_xh); free(c->W_hh); free(c->b_h); free(c); }

float *rnn_cell_forward(const RNNCell *c, const float *x, float *h, int steps) {
    float *out = (float *)malloc(steps * c->hidden_size * sizeof(float));
    for (int t = 0; t < steps; t++) {
        float *wx = linear_rnn(c->W_xh, x + t * c->input_size, NULL, c->hidden_size, c->input_size);
        float *wh = linear_rnn(c->W_hh, h, NULL, c->hidden_size, c->hidden_size);
        for (int i = 0; i < c->hidden_size; i++)
            h[i] = tanh_act(wx[i] + wh[i] + c->b_h[i]);
        memcpy(out + t * c->hidden_size, h, c->hidden_size * sizeof(float));
        free(wx); free(wh);
    }
    return out;
}

LSTMCell *lstm_cell_create(int in_sz, int hid_sz) {
    LSTMCell *c = (LSTMCell *)malloc(sizeof(LSTMCell));
    c->input_size = in_sz; c->hidden_size = hid_sz;
    int ih = hid_sz * in_sz, hh = hid_sz * hid_sz;
    c->W_f = (float *)malloc(ih * sizeof(float)); c->U_f = (float *)malloc(hh * sizeof(float));
    c->W_i = (float *)malloc(ih * sizeof(float)); c->U_i = (float *)malloc(hh * sizeof(float));
    c->W_c = (float *)malloc(ih * sizeof(float)); c->U_c = (float *)malloc(hh * sizeof(float));
    c->W_o = (float *)malloc(ih * sizeof(float)); c->U_o = (float *)malloc(hh * sizeof(float));
    c->b_f = (float *)calloc(hid_sz, sizeof(float));
    c->b_i = (float *)calloc(hid_sz, sizeof(float));
    c->b_c = (float *)calloc(hid_sz, sizeof(float));
    c->b_o = (float *)calloc(hid_sz, sizeof(float));
    float scale = sqrtf(2.0f / (in_sz + hid_sz));
    for (int i = 0; i < ih; i++) {
        c->W_f[i] = ((float)rand() / RAND_MAX - 0.5f) * scale;
        c->W_i[i] = ((float)rand() / RAND_MAX - 0.5f) * scale;
        c->W_c[i] = ((float)rand() / RAND_MAX - 0.5f) * scale;
        c->W_o[i] = ((float)rand() / RAND_MAX - 0.5f) * scale;
    }
    for (int i = 0; i < hh; i++) {
        c->U_f[i] = ((float)rand() / RAND_MAX - 0.5f) * scale;
        c->U_i[i] = ((float)rand() / RAND_MAX - 0.5f) * scale;
        c->U_c[i] = ((float)rand() / RAND_MAX - 0.5f) * scale;
        c->U_o[i] = ((float)rand() / RAND_MAX - 0.5f) * scale;
    }
    return c;
}
void lstm_cell_free(LSTMCell *c) {
    free(c->W_f); free(c->U_f); free(c->b_f);
    free(c->W_i); free(c->U_i); free(c->b_i);
    free(c->W_c); free(c->U_c); free(c->b_c);
    free(c->W_o); free(c->U_o); free(c->b_o);
    free(c);
}

void lstm_cell_forward(const LSTMCell *c, const float *x, float *h, float *cell, int steps) {
    int hs = c->hidden_size, is = c->input_size;
    for (int t = 0; t < steps; t++) {
        float *xf = linear_rnn(c->W_f, x + t * is, c->b_f, hs, is);
        float *xi = linear_rnn(c->W_i, x + t * is, c->b_i, hs, is);
        float *xc = linear_rnn(c->W_c, x + t * is, c->b_c, hs, is);
        float *xo = linear_rnn(c->W_o, x + t * is, c->b_o, hs, is);
        for (int i = 0; i < hs; i++) {
            float f = sigmoid_act(xf[i] + c->U_f[i * hs + 0] * h[i]);
            float ii = sigmoid_act(xi[i] + c->U_i[i * hs + 0] * h[i]);
            float cand = tanh_act(xc[i] + c->U_c[i * hs + 0] * h[i]);
            float o = sigmoid_act(xo[i] + c->U_o[i * hs + 0] * h[i]);
            cell[i] = f * cell[i] + ii * cand;
            h[i] = o * tanh_act(cell[i]);
        }
        free(xf); free(xi); free(xc); free(xo);
    }
}

GRUCell *gru_cell_create(int in_sz, int hid_sz) {
    GRUCell *c = (GRUCell *)malloc(sizeof(GRUCell));
    c->input_size = in_sz; c->hidden_size = hid_sz;
    int ih = hid_sz * in_sz, hh = hid_sz * hid_sz;
    c->W_r = (float *)malloc(ih * sizeof(float)); c->U_r = (float *)malloc(hh * sizeof(float));
    c->W_z = (float *)malloc(ih * sizeof(float)); c->U_z = (float *)malloc(hh * sizeof(float));
    c->W_h = (float *)malloc(ih * sizeof(float)); c->U_h = (float *)malloc(hh * sizeof(float));
    c->b_r = (float *)calloc(hid_sz, sizeof(float));
    c->b_z = (float *)calloc(hid_sz, sizeof(float));
    c->b_h = (float *)calloc(hid_sz, sizeof(float));
    float scale = sqrtf(2.0f / (in_sz + hid_sz));
    for (int i = 0; i < ih; i++) {
        c->W_r[i] = ((float)rand() / RAND_MAX - 0.5f) * scale;
        c->W_z[i] = ((float)rand() / RAND_MAX - 0.5f) * scale;
        c->W_h[i] = ((float)rand() / RAND_MAX - 0.5f) * scale;
    }
    for (int i = 0; i < hh; i++) {
        c->U_r[i] = ((float)rand() / RAND_MAX - 0.5f) * scale;
        c->U_z[i] = ((float)rand() / RAND_MAX - 0.5f) * scale;
        c->U_h[i] = ((float)rand() / RAND_MAX - 0.5f) * scale;
    }
    return c;
}
void gru_cell_free(GRUCell *c) {
    free(c->W_r); free(c->U_r); free(c->b_r);
    free(c->W_z); free(c->U_z); free(c->b_z);
    free(c->W_h); free(c->U_h); free(c->b_h);
    free(c);
}
void gru_cell_forward(const GRUCell *c, const float *x, float *h, int steps) {
    int hs = c->hidden_size, is = c->input_size;
    for (int t = 0; t < steps; t++) {
        float *xr = linear_rnn(c->W_r, x + t * is, c->b_r, hs, is);
        float *xz = linear_rnn(c->W_z, x + t * is, c->b_z, hs, is);
        for (int i = 0; i < hs; i++) {
            float r = sigmoid_act(xr[i] + c->U_r[i * hs + 0] * h[i]);
            float z = sigmoid_act(xz[i] + c->U_z[i * hs + 0] * h[i]);
            float xh_at_t = 0;
            for (int j = 0; j < is; j++) xh_at_t += c->W_h[i * is + j] * x[t * is + j];
            float cand = tanh_act(xh_at_t + c->b_h[i] + c->U_h[i * hs + 0] * (r * h[i]));
            h[i] = (1 - z) * h[i] + z * cand;
        }
        free(xr); free(xz);
    }
}

BiRNN *birnn_create(int in_sz, int hid_sz) {
    BiRNN *b = (BiRNN *)malloc(sizeof(BiRNN));
    b->input_size = in_sz; b->hidden_size = hid_sz;
    b->forward  = rnn_cell_create(in_sz, hid_sz);
    b->backward = rnn_cell_create(in_sz, hid_sz);
    return b;
}
void birnn_free(BiRNN *b) { rnn_cell_free(b->forward); rnn_cell_free(b->backward); free(b); }

float *birnn_forward(const BiRNN *b, const float *x, float *h_fw, float *h_bw, int steps) {
    int hs = b->hidden_size, is = b->input_size;
    float *out = (float *)malloc(steps * 2 * hs * sizeof(float));
    for (int t = 0; t < steps; t++) {
        float *wx = linear_rnn(b->forward->W_xh, x + t * is, NULL, hs, is);
        float *wh = linear_rnn(b->forward->W_hh, h_fw, NULL, hs, hs);
        for (int i = 0; i < hs; i++)
            h_fw[i] = tanh_act(wx[i] + wh[i] + b->forward->b_h[i]);
        memcpy(out + t * 2 * hs, h_fw, hs * sizeof(float));
        free(wx); free(wh);
    }
    for (int t = steps - 1; t >= 0; t--) {
        float *wx = linear_rnn(b->backward->W_xh, x + t * is, NULL, hs, is);
        float *wh = linear_rnn(b->backward->W_hh, h_bw, NULL, hs, hs);
        for (int i = 0; i < hs; i++)
            h_bw[i] = tanh_act(wx[i] + wh[i] + b->backward->b_h[i]);
        memcpy(out + t * 2 * hs + hs, h_bw, hs * sizeof(float));
        free(wx); free(wh);
    }
    return out;
}

/*
 * L3: Stacked RNN — multiple RNN/LSTM/GRU layers stacked sequentially.
 * Each layer output feeds into the next layer as input.
 * Deep RNNs capture hierarchical temporal features (Graves, 2013).
 */
StackedRNN *stacked_rnn_create(int in_sz, int hid_sz, int layers, int cell_type) {
    StackedRNN *s = (StackedRNN *)malloc(sizeof(StackedRNN));
    s->num_layers = layers;
    s->cell_type = cell_type;
    s->cells = (void **)malloc(layers * sizeof(void *));

    int cur_in = in_sz;
    for (int l = 0; l < layers; l++) {
        if (cell_type == 0) {
            s->cells[l] = (void *)rnn_cell_create(cur_in, hid_sz);
        } else if (cell_type == 1) {
            s->cells[l] = (void *)lstm_cell_create(cur_in, hid_sz);
        } else {
            s->cells[l] = (void *)gru_cell_create(cur_in, hid_sz);
        }
        cur_in = hid_sz;
    }
    return s;
}

void stacked_rnn_free(StackedRNN *s) {
    for (int l = 0; l < s->num_layers; l++) {
        if (s->cell_type == 0) rnn_cell_free((RNNCell *)s->cells[l]);
        else if (s->cell_type == 1) lstm_cell_free((LSTMCell *)s->cells[l]);
        else gru_cell_free((GRUCell *)s->cells[l]);
    }
    free(s->cells);
    free(s);
}

void stacked_rnn_forward(const StackedRNN *s, const float *x, float *h, int steps) {
    int hid_sz = 0, in_sz = 0;
    if (s->cell_type == 0) {
        hid_sz = ((RNNCell *)s->cells[0])->hidden_size;
        in_sz  = ((RNNCell *)s->cells[0])->input_size;
    } else if (s->cell_type == 1) {
        hid_sz = ((LSTMCell *)s->cells[0])->hidden_size;
        in_sz  = ((LSTMCell *)s->cells[0])->input_size;
    } else {
        hid_sz = ((GRUCell *)s->cells[0])->hidden_size;
        in_sz  = ((GRUCell *)s->cells[0])->input_size;
    }

    /* Allocate enough space for the largest layer's input */
    int max_input_sz = (in_sz > hid_sz) ? in_sz : hid_sz;
    float *layer_input = (float *)malloc(steps * max_input_sz * sizeof(float));
    memcpy(layer_input, x, steps * in_sz * sizeof(float));
    float *layer_hidden = (float *)calloc(hid_sz, sizeof(float));
    float *layer_cell   = (float *)calloc(hid_sz, sizeof(float));

    for (int l = 0; l < s->num_layers; l++) {
        float *layer_output = NULL;
        int cur_input_sz = (l == 0) ? in_sz : hid_sz;

        memset(layer_hidden, 0, hid_sz * sizeof(float));
        memset(layer_cell, 0, hid_sz * sizeof(float));

        if (s->cell_type == 0) {
            layer_output = rnn_cell_forward(
                (RNNCell *)s->cells[l], layer_input, layer_hidden, steps);
        } else if (s->cell_type == 1) {
            lstm_cell_forward(
                (LSTMCell *)s->cells[l], layer_input, layer_hidden, layer_cell, steps);
            layer_output = (float *)malloc(steps * hid_sz * sizeof(float));
            for (int t = 0; t < steps; t++)
                memcpy(layer_output + t * hid_sz, layer_hidden, hid_sz * sizeof(float));
        } else {
            gru_cell_forward(
                (GRUCell *)s->cells[l], layer_input, layer_hidden, steps);
            layer_output = (float *)malloc(steps * hid_sz * sizeof(float));
            for (int t = 0; t < steps; t++)
                memcpy(layer_output + t * hid_sz, layer_hidden, hid_sz * sizeof(float));
        }

        if (l < s->num_layers - 1) {
            memcpy(layer_input, layer_output, steps * hid_sz * sizeof(float));
            free(layer_output);
        } else {
            memcpy(h, layer_hidden, hid_sz * sizeof(float));
            free(layer_output);
        }
    }
    free(layer_input);
    free(layer_hidden);
    free(layer_cell);
}
