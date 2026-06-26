#ifndef RNN_LSTM_H
#define RNN_LSTM_H

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define MAX_SEQ_LEN 512

typedef struct {
    int input_size, hidden_size;
    float *W_xh, *W_hh, *b_h;
} RNNCell;

typedef struct {
    int input_size, hidden_size;
    float *W_f, *U_f, *b_f;
    float *W_i, *U_i, *b_i;
    float *W_c, *U_c, *b_c;
    float *W_o, *U_o, *b_o;
} LSTMCell;

typedef struct {
    int input_size, hidden_size;
    float *W_r, *U_r, *b_r;
    float *W_z, *U_z, *b_z;
    float *W_h, *U_h, *b_h;
} GRUCell;

RNNCell  *rnn_cell_create(int in_sz, int hid_sz);
void      rnn_cell_free(RNNCell *c);
float    *rnn_cell_forward(const RNNCell *c, const float *x, float *h, int steps);

LSTMCell *lstm_cell_create(int in_sz, int hid_sz);
void      lstm_cell_free(LSTMCell *c);
void      lstm_cell_forward(const LSTMCell *c, const float *x, float *h, float *cell, int steps);

GRUCell  *gru_cell_create(int in_sz, int hid_sz);
void      gru_cell_free(GRUCell *c);
void      gru_cell_forward(const GRUCell *c, const float *x, float *h, int steps);

typedef struct {
    RNNCell *forward;
    RNNCell *backward;
    int      input_size, hidden_size;
} BiRNN;

BiRNN *birnn_create(int in_sz, int hid_sz);
void   birnn_free(BiRNN *b);
float *birnn_forward(const BiRNN *b, const float *x, float *h_fw, float *h_bw, int steps);

typedef struct {
    void   **cells;
    int      num_layers;
    int      cell_type;
} StackedRNN;

StackedRNN *stacked_rnn_create(int in_sz, int hid_sz, int layers, int cell_type);
void        stacked_rnn_free(StackedRNN *s);
void        stacked_rnn_forward(const StackedRNN *s, const float *x, float *h, int steps);

float sigmoid_act(float x);
float tanh_act(float x);
float *linear_rnn(const float *w, const float *x, const float *b, int rows, int cols);

#endif
