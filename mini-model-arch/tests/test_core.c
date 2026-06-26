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
#include "../include/optimizers.h"
#include "../include/loss_functions.h"
#include "../include/normalization.h"
#include "../include/activations.h"
#include "../include/regularization.h"
#include "../include/position_encodings.h"
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
    matrix2d_free(Q);
    matrix2d_free(K);
    matrix2d_free(V);
    matrix2d_free(out);
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
    matrix2d_free(x);
    matrix2d_free(out);
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
    matrix2d_free(x);
    matrix2d_free(enc);
    matrix2d_free(out);
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

/* ── StackedRNN Tests ── */
static int test_stacked_rnn_create(void) {
    TEST("stacked_rnn_create");
    StackedRNN *s = stacked_rnn_create(16, 32, 3, 0);
    CHECK(s != NULL, "stacked rnn create failed");
    CHECK(s->num_layers == 3, "num_layers wrong");
    stacked_rnn_free(s);
    PASS();
    return 0;
}

static int test_stacked_rnn_forward(void) {
    TEST("stacked_rnn_forward");
    StackedRNN *s = stacked_rnn_create(8, 16, 2, 0);
    CHECK(s != NULL, "create failed");
    float x[16]; float h[16];
    memset(x, 0, sizeof(x)); memset(h, 0, sizeof(h));
    stacked_rnn_forward(s, x, h, 2);
    stacked_rnn_free(s);
    PASS();
    return 0;
}

/* ── Optimizer Tests ── */
static int test_optimizer_adam(void) {
    TEST("optimizer_adam");
    Optimizer *o = optimizer_create(10, OPT_ADAM, 0.001f, 0.9f, 0.999f, 1e-8f, 0.0f);
    CHECK(o != NULL, "adam create failed");
    float params[10], grads[10];
    memset(params, 0, sizeof(params));
    memset(grads, 0, sizeof(grads));
    optimizer_step(o, params, grads);
    optimizer_free(o);
    PASS();
    return 0;
}

static int test_optimizer_momentum(void) {
    TEST("optimizer_momentum");
    Optimizer *o = optimizer_create(5, OPT_MOMENTUM, 0.01f, 0.9f, 0.0f, 0.0f, 0.0f);
    CHECK(o != NULL, "momentum create failed");
    float p[5] = {1,2,3,4,5}, g[5] = {0.1f,0.2f,0.3f,0.4f,0.5f};
    optimizer_step(o, p, g);
    CHECK(p[0] < 1.0f, "param should decrease");
    optimizer_free(o);
    PASS();
    return 0;
}

static int test_lr_schedule_cosine(void) {
    TEST("lr_schedule_cosine");
    LRSchedule *s = lr_schedule_create(LR_COSINE, 0.01f, 100);
    float lr0 = lr_schedule_get(s, 0);
    float lr50 = lr_schedule_get(s, 50);
    float lr99 = lr_schedule_get(s, 99);
    CHECK(lr0 > lr99, "cosine lr should decay");
    lr_schedule_free(s);
    PASS();
    return 0;
}

/* ── Loss Function Tests ── */
static int test_mse_loss(void) {
    TEST("mse_loss");
    float pred[4] = {1,2,3,4}, tgt[4] = {1,2,3,4};
    float l = mse_loss(pred, tgt, 4);
    CHECK(l < 1e-5f, "perfect prediction mse should be 0");
    float grad[4];
    mse_loss_grad(pred, tgt, grad, 4);
    PASS();
    return 0;
}

static int test_cross_entropy(void) {
    TEST("cross_entropy");
    float logits[6] = {1.0f,0, 1.0f,0, 1.0f,0};
    int targets[3] = {0, 0, 0};
    float l = cross_entropy_loss(logits, targets, 3, 2);
    CHECK(l > 0.0f, "cross entropy should be positive");
    float grad[6];
    cross_entropy_grad(logits, targets, grad, 3, 2);
    PASS();
    return 0;
}

static int test_focal_loss(void) {
    TEST("focal_loss");
    float logits[4] = {2.0f, -2.0f, -2.0f, 2.0f};
    int targets[2] = {0, 1};
    float l = focal_loss(logits, targets, 2, 2, 2.0f, 0.25f);
    CHECK(l > 0.0f, "focal loss should be positive");
    float grad[4];
    focal_loss_grad(logits, targets, grad, 2, 2, 2.0f, 0.25f);
    PASS();
    return 0;
}

static int test_huber_loss(void) {
    TEST("huber_loss");
    float pred[4] = {0,1,2,3}, tgt[4] = {0,1.5f,2,3};
    float l1 = huber_loss(pred, tgt, 4, 1.0f);
    CHECK(l1 >= 0.0f, "huber loss should be non-negative");
    float grad[4];
    huber_loss_grad(pred, tgt, grad, 4, 1.0f);
    PASS();
    return 0;
}

/* ── Normalization Tests ── */
static int test_batchnorm1d(void) {
    TEST("batchnorm1d");
    BatchNorm1D *b = bn1d_create(4, 0.1f);
    CHECK(b != NULL, "bn1d create failed");
    float x[8] = {1,2,3,4, 5,6,7,8};
    bn1d_forward_train(b, x, 2);
    CHECK(fabsf(x[0]) < 5.0f, "batchnorm output should be normalized");
    bn1d_free(b);
    PASS();
    return 0;
}

static int test_rmsnorm(void) {
    TEST("rmsnorm");
    RMSNorm *r = rmsnorm_create(8);
    CHECK(r != NULL, "rmsnorm create failed");
    float x[16];
    for (int i = 0; i < 16; i++) x[i] = (float)(i + 1);
    rmsnorm_forward(r, x, 2, 8);
    rmsnorm_free(r);
    PASS();
    return 0;
}

static int test_groupnorm(void) {
    TEST("groupnorm");
    GroupNorm *g = gn_create(2, 8);
    CHECK(g != NULL, "groupnorm create failed");
    float x[4 * 8 * 2 * 2];
    memset(x, 0, sizeof(x));
    for (int i = 0; i < 4 * 8 * 4; i++) x[i] = (float)(i % 10);
    gn_forward(g, x, 4, 2, 2);
    gn_free(g);
    PASS();
    return 0;
}

/* ── Activation Tests ── */
static int test_gelu_silu(void) {
    TEST("gelu_silu");
    float x[4] = {-1.0f, 0.0f, 1.0f, 2.0f};
    float out[4];
    gelu_forward(x, out, 4);
    CHECK(out[1] < 1e-4f, "GELU(0) should be ~0");

    silu_forward(x, out, 4);
    CHECK(out[1] < 1e-4f, "SiLU(0) should be ~0");
    PASS();
    return 0;
}

static int test_swiglu(void) {
    TEST("swiglu");
    int b=1, d=8, ff=16;
    float x[8] = {0.1f,0.2f,0.3f,0.4f,0.5f,0.6f,0.7f,0.8f};
    float *W1=(float*)calloc(d*ff,sizeof(float));
    float *W2=(float*)calloc(d*ff,sizeof(float));
    float *W3=(float*)calloc(ff*d,sizeof(float));
    for(int i=0;i<d*ff;i++) { W1[i]=0.01f; W2[i]=0.01f; }
    for(int i=0;i<ff*d;i++) W3[i]=0.01f;
    float *out=(float*)calloc(d,sizeof(float));
    swiglu_forward(x,W1,W2,W3,out,b,d,ff);
    free(W1); free(W2); free(W3); free(out);
    PASS();
    return 0;
}

/* ── Regularization Tests ── */
static int test_dropout(void) {
    TEST("dropout");
    Dropout *d = dropout_create(0.8f, 1, 256);
    CHECK(d != NULL, "dropout create failed");
    float x[256];
    for (int i = 0; i < 256; i++) x[i] = 1.0f;
    dropout_forward(d, x, 256, 1);
    int dropped = 0;
    for (int i = 0; i < 256; i++) if (x[i] == 0.0f) dropped++;
    CHECK(dropped < 200, "too many dropped with keep=0.8");
    dropout_free(d);
    PASS();
    return 0;
}

static int test_label_smoothing(void) {
    TEST("label_smoothing");
    int targets[2] = {0, 2};
    float smooth[2 * 3];
    label_smoothing(targets, smooth, 2, 3, 0.1f);
    CHECK(fabsf(smooth[0] - (0.9f + 0.1f/3.0f)) < 0.01f, "smoothed label wrong");
    PASS();
    return 0;
}

static int test_mixup(void) {
    TEST("mixup");
    float x1[4]={1,2,3,4}, x2[4]={4,3,2,1}, y1[4]={1,0,0,0}, y2[4]={0,1,0,0};
    float xo[4], yo[4];
    mixup_augment(x1,x2,y1,y2,xo,yo,4,0.2f);
    CHECK(xo[0] >= 0.0f && xo[0] <= 4.0f, "mixup x out of range");
    CHECK(yo[0] >= 0.0f && yo[0] <= 1.0f, "mixup y out of range");
    PASS();
    return 0;
}

/* ── Position Encoding Tests ── */
static int test_rope(void) {
    TEST("rope");
    RoPE *r = rope_create(64, 128, 10000.0f);
    CHECK(r != NULL, "rope create failed");
    float q[128], k[128];
    memset(q, 0, sizeof(q)); memset(k, 0, sizeof(k));
    q[0] = 1.0f; k[0] = 1.0f;
    rope_apply(r, q, k, 2, 0);
    rope_free(r);
    PASS();
    return 0;
}

static int test_alibi(void) {
    TEST("alibi");
    ALiBi *a = alibi_create(8);
    CHECK(a != NULL, "alibi create failed");
    float scores[8 * 16 * 16];
    memset(scores, 0, sizeof(scores));
    alibi_apply(a, scores, 16);
    CHECK(scores[0] <= 0.0f, "alibi bias should be non-positive");
    alibi_free(a);
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
    failed += test_stacked_rnn_create();
    failed += test_stacked_rnn_forward();
    failed += test_optimizer_adam();
    failed += test_optimizer_momentum();
    failed += test_lr_schedule_cosine();
    failed += test_mse_loss();
    failed += test_cross_entropy();
    failed += test_focal_loss();
    failed += test_huber_loss();
    failed += test_batchnorm1d();
    failed += test_rmsnorm();
    failed += test_groupnorm();
    failed += test_gelu_silu();
    failed += test_swiglu();
    failed += test_dropout();
    failed += test_label_smoothing();
    failed += test_mixup();
    failed += test_rope();
    failed += test_alibi();

    printf("\n=== Results: %d/%d passed, %d failed ===\n",
           tests_passed, tests_run, failed);
    return failed > 0 ? 1 : 0;
}
