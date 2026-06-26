/*
 * mini-model-arch — Core Tests
 *
 * Unit tests for CNN models, RNN/LSTM, Transformer, GAN, Diffusion.
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

static int tests_run = 0, tests_passed = 0;

#define TEST(name) do { tests_run++; printf("  TEST %s ... ", name); } while(0)
#define PASS() do { tests_passed++; printf("PASS\n"); } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); return 1; } while(0)
#define CHECK(cond, msg) if (!(cond)) FAIL(msg)

/* ── CNN Tests ── */
static int test_tensor4d_create(void) {
    TEST("tensor4d_create");
    Tensor4D t;
    tensor4d_init(&t, 2, 3, 32, 32);
    CHECK(t.batch == 2, "batch wrong");
    CHECK(t.channels == 3, "channels wrong");
    CHECK(t.height == 32, "height wrong");
    CHECK(t.width == 32, "width wrong");
    tensor4d_free(&t);
    PASS();
    return 0;
}

static int test_conv2d_create_forward(void) {
    TEST("conv2d_create_forward");
    Conv2D *c = conv2d_create(3, 16, 3, 1, 1);
    CHECK(c != NULL, "conv2d create failed");
    CHECK(c->out_channels == 16, "out_channels wrong");
    Tensor4D in;
    tensor4d_init(&in, 1, 3, 8, 8);
    Tensor4D *out = conv2d_forward(c, &in);
    CHECK(out != NULL, "conv2d forward failed");
    conv2d_free(c);
    tensor4d_free(&in);
    tensor4d_free(out);
    free(out);
    PASS();
    return 0;
}

static int test_lenet5_create_forward(void) {
    TEST("lenet5_create_forward");
    LeNet5 *l = lenet5_create();
    CHECK(l != NULL, "lenet5 create failed");
    Tensor4D in;
    tensor4d_init(&in, 1, 1, 28, 28);
    Tensor4D *out = lenet5_forward(l, &in);
    CHECK(out != NULL, "lenet5 forward failed");
    lenet5_free(l);
    tensor4d_free(&in);
    tensor4d_free(out);
    free(out);
    PASS();
    return 0;
}

static int test_resblock_create_forward(void) {
    TEST("resblock_create_forward");
    ResBlock *r = resblock_create(64, 64, 1, 0);
    CHECK(r != NULL, "resblock create failed");
    Tensor4D in;
    tensor4d_init(&in, 1, 64, 16, 16);
    Tensor4D *out = resblock_forward(r, &in);
    CHECK(out != NULL, "resblock forward failed");
    resblock_free(r);
    tensor4d_free(&in);
    tensor4d_free(out);
    free(out);
    PASS();
    return 0;
}

static int test_inception_create_forward(void) {
    TEST("inception_create_forward");
    InceptionModule *im = inception_create(192, 64, 96, 128, 16, 32, 32);
    CHECK(im != NULL, "inception create failed");
    Tensor4D in;
    tensor4d_init(&in, 1, 192, 28, 28);
    Tensor4D *out = inception_forward(im, &in);
    CHECK(out != NULL, "inception forward failed");
    inception_free(im);
    tensor4d_free(&in);
    tensor4d_free(out);
    free(out);
    PASS();
    return 0;
}

/* ── RNN/LSTM Tests ── */
static int test_rnn_cell_create_forward(void) {
    TEST("rnn_cell_create_forward");
    RNNCell *c = rnn_cell_create(16, 32);
    CHECK(c != NULL, "rnn cell create failed");
    float x[16], h[32];
    memset(x, 0, sizeof(x));
    memset(h, 0, sizeof(h));
    float *out = rnn_cell_forward(c, x, h, 4);
    CHECK(out != NULL, "rnn forward failed");
    rnn_cell_free(c);
    free(out);
    PASS();
    return 0;
}

static int test_lstm_cell_create_forward(void) {
    TEST("lstm_cell_create_forward");
    LSTMCell *c = lstm_cell_create(16, 32);
    CHECK(c != NULL, "lstm cell create failed");
    float x[16], h[32], cell[32];
    memset(x, 0, sizeof(x));
    memset(h, 0, sizeof(h));
    memset(cell, 0, sizeof(cell));
    lstm_cell_forward(c, x, h, cell, 4);
    lstm_cell_free(c);
    PASS();
    return 0;
}

static int test_gru_cell_create_forward(void) {
    TEST("gru_cell_create_forward");
    GRUCell *c = gru_cell_create(16, 32);
    CHECK(c != NULL, "gru cell create failed");
    float x[16], h[32];
    memset(x, 0, sizeof(x));
    memset(h, 0, sizeof(h));
    gru_cell_forward(c, x, h, 4);
    gru_cell_free(c);
    PASS();
    return 0;
}

static int test_birnn_create_forward(void) {
    TEST("birnn_create_forward");
    BiRNN *b = birnn_create(16, 32);
    CHECK(b != NULL, "birnn create failed");
    float x[16], h_fw[32], h_bw[32];
    memset(x, 0, sizeof(x));
    memset(h_fw, 0, sizeof(h_fw));
    memset(h_bw, 0, sizeof(h_bw));
    float *out = birnn_forward(b, x, h_fw, h_bw, 4);
    CHECK(out != NULL, "birnn forward failed");
    birnn_free(b);
    free(out);
    PASS();
    return 0;
}

/* ── Transformer Tests ── */
static int test_mha_create_forward(void) {
    TEST("mha_create_forward");
    MultiHeadAttn *m = mha_create(64, 8, 0);
    CHECK(m != NULL, "mha create failed");
    Matrix2D *Q = matrix2d_create(64, 10);
    Matrix2D *K = matrix2d_create(64, 10);
    Matrix2D *V = matrix2d_create(64, 10);
    Matrix2D *out = mha_forward(m, Q, K, V);
    CHECK(out != NULL, "mha forward failed");
    mha_free(m);
    matrix2d_free(Q); free(Q);
    matrix2d_free(K); free(K);
    matrix2d_free(V); free(V);
    matrix2d_free(out); free(out);
    PASS();
    return 0;
}

static int test_transformer_encoder(void) {
    TEST("transformer_encoder");
    TransformerEncoder *e = trans_enc_create(2, 64, 8, 256, 0);
    CHECK(e != NULL, "trans enc create failed");
    CHECK(e->num_layers == 2, "num_layers wrong");
    Matrix2D *x = matrix2d_create(64, 10);
    Matrix2D *out = trans_enc_forward(e, x);
    CHECK(out != NULL, "trans enc forward failed");
    trans_enc_free(e);
    matrix2d_free(x); free(x);
    matrix2d_free(out); free(out);
    PASS();
    return 0;
}

static int test_transformer_decoder(void) {
    TEST("transformer_decoder");
    TransformerDecoder *d = trans_dec_create(2, 64, 8, 256, 0);
    CHECK(d != NULL, "trans dec create failed");
    Matrix2D *x = matrix2d_create(64, 10);
    Matrix2D *enc = matrix2d_create(64, 10);
    Matrix2D *out = trans_dec_forward(d, x, enc);
    CHECK(out != NULL, "trans dec forward failed");
    trans_dec_free(d);
    matrix2d_free(x); free(x);
    matrix2d_free(enc); free(enc);
    matrix2d_free(out); free(out);
    PASS();
    return 0;
}

/* ── GAN Tests ── */
static int test_gan_create(void) {
    TEST("gan_create");
    int gen_hid[] = {128, 256};
    int disc_hid[] = {256, 128};
    GAN *gan = gan_create(100, 784, gen_hid, 2, disc_hid, 2, 0);
    CHECK(gan != NULL, "gan create failed");
    CHECK(gan->noise_dim == 100, "noise_dim wrong");
    gan_free(gan);
    PASS();
    return 0;
}

static int test_gan_train_step(void) {
    TEST("gan_train_step");
    int gen_hid[] = {64};
    int disc_hid[] = {64};
    GAN *gan = gan_create(10, 16, gen_hid, 1, disc_hid, 1, 0);
    float real[32]; memset(real, 0, sizeof(real));
    gan_train_step_d(gan, real, 2, 0.01f);
    gan_train_step_g(gan, 2, 0.01f);
    gan_free(gan);
    PASS();
    return 0;
}

/* ── Diffusion Tests ── */
static int test_diff_schedule_create(void) {
    TEST("diff_schedule_create");
    DiffusionSchedule *s = diff_schedule_create(100, 1e-4f, 0.02f, 0);
    CHECK(s != NULL, "diff schedule create failed");
    CHECK(s->num_timesteps == 100, "num_timesteps wrong");
    diff_schedule_free(s);
    PASS();
    return 0;
}

static int test_unet_create_forward(void) {
    TEST("unet_create_forward");
    UNet *u = unet_create(3, 64, 3, 128, 2);
    CHECK(u != NULL, "unet create failed");
    float x[3 * 32 * 32]; memset(x, 0, sizeof(x));
    float *out = unet_forward(u, x, 50, 32, 32);
    CHECK(out != NULL, "unet forward failed");
    unet_free(u);
    free(out);
    PASS();
    return 0;
}

static int test_diff_model_create(void) {
    TEST("diff_model_create");
    DiffusionModel *m = diff_model_create(3, 32, 100, 0, 1.0f);
    CHECK(m != NULL, "diff model create failed");
    CHECK(m->image_channels == 3, "image_channels wrong");
    diff_model_free(m);
    PASS();
    return 0;
}

int main(void) {
    printf("=== mini-model-arch Unit Tests ===\n\n");

    int failed = 0;
    failed += test_tensor4d_create();
    failed += test_conv2d_create_forward();
    failed += test_lenet5_create_forward();
    failed += test_resblock_create_forward();
    failed += test_inception_create_forward();
    failed += test_rnn_cell_create_forward();
    failed += test_lstm_cell_create_forward();
    failed += test_gru_cell_create_forward();
    failed += test_birnn_create_forward();
    failed += test_mha_create_forward();
    failed += test_transformer_encoder();
    failed += test_transformer_decoder();
    failed += test_gan_create();
    failed += test_gan_train_step();
    failed += test_diff_schedule_create();
    failed += test_unet_create_forward();
    failed += test_diff_model_create();

    printf("\n=== Results: %d/%d passed, %d failed ===\n",
           tests_passed, tests_run, failed);
    return failed > 0 ? 1 : 0;
}
