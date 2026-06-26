/*
 * mini-model-arch — Full Demo: Neural Network Architectures
 *
 * Demonstrates: CNN (LeNet5, ResBlock, Inception), RNN/LSTM/GRU,
 *               Transformer, GAN, Diffusion model.
 */
#include "../include/cnn_models.h"
#include "../include/rnn_lstm.h"
#include "../include/transformer_arch.h"
#include "../include/gan_model.h"
#include "../include/diffusion_model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(void) {
    printf("=== mini-model-arch: NN Architectures Demo ===\n\n");

    /* Step 1: CNN — LeNet5 */
    printf("-- Step 1: LeNet5 on MNIST-like input --\n");
    Tensor4D in;
    tensor4d_init(&in, 1, 1, 28, 28);
    for (int i = 0; i < 28 * 28; i++) in.data[i] = (float)(i % 256) / 255.0f;
    LeNet5 *lenet = lenet5_create();
    Tensor4D *out = lenet5_forward(lenet, &in);
    printf("LeNet5: input 1x28x28 -> output %dx%dx%dx%d\n",
           out->batch, out->channels, out->height, out->width);
    lenet5_free(lenet);
    tensor4d_free(&in);
    tensor4d_free(out); free(out);

    /* Step 2: ResBlock */
    printf("\n-- Step 2: ResBlock (skip connection) --\n");
    tensor4d_init(&in, 1, 64, 16, 16);
    for (int i = 0; i < 64 * 16 * 16; i++) in.data[i] = 1.0f;
    ResBlock *rb = resblock_create(64, 64, 1, 0);
    Tensor4D *rb_out = resblock_forward(rb, &in);
    printf("ResBlock: in=%dx%dx%dx%d out=%dx%dx%dx%d%s\n",
           in.batch, in.channels, in.height, in.width,
           rb_out->batch, rb_out->channels, rb_out->height, rb_out->width,
           rb->use_projection ? " (projection)" : "");
    resblock_free(rb);
    tensor4d_free(&in);
    tensor4d_free(rb_out); free(rb_out);

    /* Step 3: Inception Module */
    printf("\n-- Step 3: Inception Module --\n");
    tensor4d_init(&in, 1, 192, 28, 28);
    InceptionModule *im = inception_create(192, 64, 96, 128, 16, 32, 32);
    Tensor4D *im_out = inception_forward(im, &in);
    printf("Inception: in=%d channels -> out=%d channels\n", 192, im_out->channels);
    inception_free(im);
    tensor4d_free(&in);
    tensor4d_free(im_out); free(im_out);

    /* Step 4: RNN / LSTM / GRU */
    printf("\n-- Step 4: RNN, LSTM, GRU sequence processing --\n");
    float x[16] = {0}, h[32] = {0}, cell[32] = {0};
    for (int i = 0; i < 16; i++) x[i] = (float)i / 16.0f;

    RNNCell *rnn = rnn_cell_create(16, 32);
    float *rnn_out = rnn_cell_forward(rnn, x, h, 4);
    printf("RNN:  in=16, hid=32, steps=4, out[0]=%.4f\n", rnn_out[0]);
    rnn_cell_free(rnn); free(rnn_out);

    LSTMCell *lstm = lstm_cell_create(16, 32);
    lstm_cell_forward(lstm, x, h, cell, 4);
    printf("LSTM: in=16, hid=32, steps=4, h[0]=%.4f\n", h[0]);
    lstm_cell_free(lstm);

    GRUCell *gru = gru_cell_create(16, 32);
    memset(h, 0, sizeof(h));
    gru_cell_forward(gru, x, h, 4);
    printf("GRU:  in=16, hid=32, steps=4, h[0]=%.4f\n", h[0]);
    gru_cell_free(gru);

    /* Step 5: Transformer */
    printf("\n-- Step 5: Transformer (MHA + Encoder/Decoder) --\n");
    MultiHeadAttn *mha = mha_create(64, 8, 0);
    Matrix2D *Q = matrix2d_create(64, 10);
    Matrix2D *K = matrix2d_create(64, 10);
    Matrix2D *V = matrix2d_create(64, 10);
    Matrix2D *attn_out = mha_forward(mha, Q, K, V);
    printf("MHA: d_model=64, heads=8, seq=10 -> dim=%d, seq=%d\n",
           attn_out->dim, attn_out->seq_len);
    mha_free(mha);
    matrix2d_free(Q); free(Q);
    matrix2d_free(K); free(K);
    matrix2d_free(V); free(V);
    matrix2d_free(attn_out); free(attn_out);

    TransformerEncoder *enc = trans_enc_create(2, 64, 8, 256, 0);
    Matrix2D *x_enc = matrix2d_create(64, 10);
    Matrix2D *enc_out = trans_enc_forward(enc, x_enc);
    printf("TransformerEncoder: 2 layers, d=64 -> out dim=%d seq=%d\n",
           enc_out->dim, enc_out->seq_len);
    trans_enc_free(enc);
    matrix2d_free(x_enc); free(x_enc);
    matrix2d_free(enc_out); free(enc_out);

    /* Step 6: GAN */
    printf("\n-- Step 6: GAN Training Step --\n");
    int gen_hid[] = {128, 256};
    int disc_hid[] = {256, 128};
    GAN *gan = gan_create(100, 784, gen_hid, 2, disc_hid, 2, 0);
    float real_batch[4 * 784];
    for (int i = 0; i < 4 * 784; i++) real_batch[i] = (float)(rand() % 1000) / 1000.0f;
    gan_train_step_d(gan, real_batch, 4, 0.0002f);
    gan_train_step_g(gan, 4, 0.0002f);
    printf("GAN: noise_dim=100, data_dim=784, 1 D-step + 1 G-step done\n");
    gan_free(gan);

    /* Step 7: Diffusion */
    printf("\n-- Step 7: Diffusion Model (DDPM) --\n");
    DiffusionModel *dm = diff_model_create(3, 32, 100, 0, 1.0f);
    float img[3 * 32 * 32];
    for (int i = 0; i < 3 * 32 * 32; i++) img[i] = (float)(rand() % 256) / 255.0f;
    float *loss_out = diff_train_step(dm, img, 2, 0.0001f);
    printf("Diffusion: img=3x32x32, steps=100, loss=%.4f\n", *loss_out);
    free(loss_out);
    diff_model_free(dm);

    printf("\nNN architectures demo complete!\n");
    return 0;
}
