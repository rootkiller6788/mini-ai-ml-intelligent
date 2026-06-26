/*
 * mini-model-arch — Core Benchmarks
 *
 * Benchmarks: CNN (LeNet5, ResBlock, Inception), RNN/LSTM/GRU,
 *             Transformer (MHA, Encoder, Decoder), GAN, Diffusion.
 */
#include "../include/cnn_models.h"
#include "../include/rnn_lstm.h"
#include "../include/transformer_arch.h"
#include "../include/gan_model.h"
#include "../include/diffusion_model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
    return (double)clock() / (double)(CLOCKS_PER_SEC / 1000);
}

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 5000;
    double t0, t1;
    printf("=== mini-model-arch Benchmarks (N=%d) ===\n\n", N);

    /* ── Conv2D ── */
    {
        t0 = now_ms();
        for (int r = 0; r < N / 10; r++) {
            Tensor4D in;
            tensor4d_init(&in, 1, 3, 32, 32);
            Conv2D *c = conv2d_create(3, 16, 3, 1, 1);
            Tensor4D *out = conv2d_forward(c, &in);
            conv2d_free(c);
            tensor4d_free(&in);
            tensor4d_free(out); free(out);
        }
        t1 = now_ms();
        printf("  conv2d_fwd:        %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 10, t1 - t0, (t1 - t0) / (double)(N / 10) * 1000.0);
    }

    /* ── LeNet5 ── */
    {
        t0 = now_ms();
        for (int r = 0; r < N / 20; r++) {
            Tensor4D in;
            tensor4d_init(&in, 1, 1, 28, 28);
            LeNet5 *l = lenet5_create();
            Tensor4D *out = lenet5_forward(l, &in);
            lenet5_free(l);
            tensor4d_free(&in);
            tensor4d_free(out); free(out);
        }
        t1 = now_ms();
        printf("  lenet5_fwd:        %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 20, t1 - t0, (t1 - t0) / (double)(N / 20) * 1000.0);
    }

    /* ── ResBlock ── */
    {
        t0 = now_ms();
        for (int r = 0; r < N / 20; r++) {
            Tensor4D in;
            tensor4d_init(&in, 1, 64, 16, 16);
            ResBlock *rb = resblock_create(64, 64, 1, 0);
            Tensor4D *out = resblock_forward(rb, &in);
            resblock_free(rb);
            tensor4d_free(&in);
            tensor4d_free(out); free(out);
        }
        t1 = now_ms();
        printf("  resblock_fwd:      %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 20, t1 - t0, (t1 - t0) / (double)(N / 20) * 1000.0);
    }

    /* ── Inception ── */
    {
        t0 = now_ms();
        for (int r = 0; r < N / 20; r++) {
            Tensor4D in;
            tensor4d_init(&in, 1, 192, 28, 28);
            InceptionModule *im = inception_create(192, 64, 96, 128, 16, 32, 32);
            Tensor4D *out = inception_forward(im, &in);
            inception_free(im);
            tensor4d_free(&in);
            tensor4d_free(out); free(out);
        }
        t1 = now_ms();
        printf("  inception_fwd:     %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 20, t1 - t0, (t1 - t0) / (double)(N / 20) * 1000.0);
    }

    /* ── RNN ── */
    {
        t0 = now_ms();
        for (int r = 0; r < N / 20; r++) {
            RNNCell *c = rnn_cell_create(16, 32);
            float x[16] = {0}, h[32] = {0};
            float *out = rnn_cell_forward(c, x, h, 4);
            rnn_cell_free(c);
            free(out);
        }
        t1 = now_ms();
        printf("  rnn_fwd:           %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 20, t1 - t0, (t1 - t0) / (double)(N / 20) * 1000.0);
    }

    /* ── LSTM ── */
    {
        t0 = now_ms();
        for (int r = 0; r < N / 20; r++) {
            LSTMCell *c = lstm_cell_create(16, 32);
            float x[16] = {0}, h[32] = {0}, cell[32] = {0};
            lstm_cell_forward(c, x, h, cell, 4);
            lstm_cell_free(c);
        }
        t1 = now_ms();
        printf("  lstm_fwd:          %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 20, t1 - t0, (t1 - t0) / (double)(N / 20) * 1000.0);
    }

    /* ── Multi-Head Attention ── */
    {
        t0 = now_ms();
        for (int r = 0; r < N / 20; r++) {
            MultiHeadAttn *m = mha_create(64, 8, 0);
            Matrix2D *Q = matrix2d_create(64, 10);
            Matrix2D *K = matrix2d_create(64, 10);
            Matrix2D *V = matrix2d_create(64, 10);
            Matrix2D *out = mha_forward(m, Q, K, V);
            mha_free(m);
            matrix2d_free(Q); free(Q);
            matrix2d_free(K); free(K);
            matrix2d_free(V); free(V);
            matrix2d_free(out); free(out);
        }
        t1 = now_ms();
        printf("  mha_fwd:           %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 20, t1 - t0, (t1 - t0) / (double)(N / 20) * 1000.0);
    }

    /* ── Transformer Encoder ── */
    {
        t0 = now_ms();
        for (int r = 0; r < N / 20; r++) {
            TransformerEncoder *e = trans_enc_create(2, 64, 8, 256, 0);
            Matrix2D *x = matrix2d_create(64, 10);
            Matrix2D *out = trans_enc_forward(e, x);
            trans_enc_free(e);
            matrix2d_free(x); free(x);
            matrix2d_free(out); free(out);
        }
        t1 = now_ms();
        printf("  trans_enc_fwd:     %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 20, t1 - t0, (t1 - t0) / (double)(N / 20) * 1000.0);
    }

    /* ── GAN ── */
    {
        int gh[] = {64}, dh[] = {64};
        t0 = now_ms();
        for (int r = 0; r < N / 20; r++) {
            GAN *gan = gan_create(10, 16, gh, 1, dh, 1, 0);
            float real[32] = {0};
            gan_train_step_d(gan, real, 2, 0.01f);
            gan_train_step_g(gan, 2, 0.01f);
            gan_free(gan);
        }
        t1 = now_ms();
        printf("  gan_train_step:    %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 20, t1 - t0, (t1 - t0) / (double)(N / 20) * 1000.0);
    }

    /* ── UNet ── */
    {
        t0 = now_ms();
        for (int r = 0; r < N / 20; r++) {
            UNet *u = unet_create(3, 32, 3, 64, 1);
            float x[3 * 16 * 16];
            memset(x, 0, sizeof(x));
            float *out = unet_forward(u, x, 10, 16, 16);
            unet_free(u);
            free(out);
        }
        t1 = now_ms();
        printf("  unet_fwd:          %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 20, t1 - t0, (t1 - t0) / (double)(N / 20) * 1000.0);
    }

    /* ── Diffusion Schedule ── */
    {
        t0 = now_ms();
        for (int r = 0; r < N / 10; r++) {
            DiffusionSchedule *s = diff_schedule_create(100, 1e-4f, 0.02f, 0);
            diff_schedule_free(s);
        }
        t1 = now_ms();
        printf("  diff_sched_create: %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 10, t1 - t0, (t1 - t0) / (double)(N / 10) * 1000.0);
    }

    printf("\nDone.\n");
    return 0;
}
