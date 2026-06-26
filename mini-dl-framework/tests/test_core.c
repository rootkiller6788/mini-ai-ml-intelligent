/*
 * mini-dl-framework — Core Tests
 *
 * Unit tests for autograd, tensor ops, NN layers, optimizers, loss functions.
 */
#include "../include/autograd.h"
#include "../include/tensor_ops.h"
#include "../include/nn_layers.h"
#include "../include/optimizers.h"
#include "../include/loss_funcs.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int tests_run = 0, tests_passed = 0;

#define TEST(name) do { tests_run++; printf("  TEST %s ... ", name); } while(0)
#define PASS() do { tests_passed++; printf("PASS\n"); } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); return 1; } while(0)
#define CHECK(cond, msg) if (!(cond)) FAIL(msg)

/* ── Autograd Tests ── */
static int test_node_create(void) {
    TEST("node_create");
    Node *n = node_create(3.14f, true);
    CHECK(n != NULL, "node create failed");
    CHECK(fabsf(n->value - 3.14f) < 0.001f, "value wrong");
    CHECK(n->requires_grad, "requires_grad wrong");
    node_free(n);
    PASS();
    return 0;
}

static int test_node_add_mul(void) {
    TEST("node_add_mul");
    Node *a = node_create(2.0f, true);
    Node *b = node_create(3.0f, true);
    Node *sum = node_add(a, b);
    CHECK(sum != NULL, "add failed");
    CHECK(fabsf(sum->value - 5.0f) < 0.01f, "add value wrong");
    Node *prod = node_mul(a, b);
    CHECK(prod != NULL, "mul failed");
    CHECK(fabsf(prod->value - 6.0f) < 0.01f, "mul value wrong");
    node_free(a); node_free(b);
    node_free(sum); node_free(prod);
    PASS();
    return 0;
}

static int test_node_relu(void) {
    TEST("node_relu");
    Node *a = node_create(-1.0f, true);
    Node *r = node_relu(a);
    CHECK(fabsf(r->value - 0.0f) < 0.01f, "relu(-1) != 0");
    node_free(a); node_free(r);
    a = node_create(2.0f, true);
    r = node_relu(a);
    CHECK(fabsf(r->value - 2.0f) < 0.01f, "relu(2) != 2");
    node_free(a); node_free(r);
    PASS();
    return 0;
}

/* ── Tensor Tests ── */
static int test_tensor_create(void) {
    TEST("tensor_create");
    int dims[] = {2, 3};
    Tensor *t = tensor_create(dims, 2);
    CHECK(t != NULL, "tensor create failed");
    CHECK(t->ndim == 2, "ndim wrong");
    CHECK(t->size == 6, "size wrong");
    tensor_free(t);
    PASS();
    return 0;
}

static int test_tensor_add(void) {
    TEST("tensor_add");
    int dims[] = {2, 2};
    Tensor *a = tensor_create_ones(dims, 2);
    Tensor *b = tensor_create_ones(dims, 2);
    Tensor *c = tensor_add(a, b);
    CHECK(c != NULL, "add failed");
    int idx[] = {0, 0};
    CHECK(fabsf(tensor_get(c, idx) - 2.0f) < 0.01f, "add value wrong");
    tensor_free(a); tensor_free(b); tensor_free(c);
    PASS();
    return 0;
}

static int test_tensor_matmul(void) {
    TEST("tensor_matmul");
    int da[] = {2, 3}, db[] = {3, 2};
    Tensor *a = tensor_create_ones(da, 2);
    Tensor *b = tensor_create_ones(db, 2);
    Tensor *c = tensor_matmul(a, b);
    CHECK(c != NULL, "matmul failed");
    int idx[] = {0, 0};
    CHECK(fabsf(tensor_get(c, idx) - 3.0f) < 0.01f, "matmul value wrong");
    tensor_free(a); tensor_free(b); tensor_free(c);
    PASS();
    return 0;
}

/* ── NN Layers Tests ── */
static int test_linear_create_forward(void) {
    TEST("linear_create_forward");
    Linear *l = linear_create(16, 8, true);
    CHECK(l != NULL, "linear create failed");
    CHECK(l->in_features == 16, "in_features wrong");
    int dims[] = {4, 16};
    Tensor *in = tensor_create_ones(dims, 2);
    Tensor *out = linear_forward(l, in);
    CHECK(out != NULL, "linear forward failed");
    linear_free(l);
    tensor_free(in); tensor_free(out);
    PASS();
    return 0;
}

static int test_conv2d_create_forward(void) {
    TEST("conv2d_create_forward");
    Conv2d *c = conv2d_create(3, 16, 3, 3, 1, 1, true);
    CHECK(c != NULL, "conv2d create failed");
    int dims[] = {1, 3, 32, 32};
    Tensor *in = tensor_create_ones(dims, 4);
    Tensor *out = conv2d_forward(c, in);
    CHECK(out != NULL, "conv2d forward failed");
    conv2d_free(c);
    tensor_free(in); tensor_free(out);
    PASS();
    return 0;
}

static int test_batchnorm1d(void) {
    TEST("batchnorm1d_create_forward");
    BatchNorm1d *bn = batchnorm1d_create(64, 1e-5f, 0.1f);
    CHECK(bn != NULL, "bn create failed");
    int dims[] = {32, 64};
    Tensor *in = tensor_create_randomn(dims, 2);
    Tensor *out = batchnorm1d_forward(bn, in);
    CHECK(out != NULL, "bn forward failed");
    batchnorm1d_free(bn);
    tensor_free(in); tensor_free(out);
    PASS();
    return 0;
}

static int test_layernorm(void) {
    TEST("layernorm_create_forward");
    LayerNorm *ln = layernorm_create(64, 1e-5f);
    CHECK(ln != NULL, "ln create failed");
    int dims[] = {8, 64};
    Tensor *in = tensor_create_randomn(dims, 2);
    Tensor *out = layernorm_forward(ln, in);
    CHECK(out != NULL, "ln forward failed");
    layernorm_free(ln);
    tensor_free(in); tensor_free(out);
    PASS();
    return 0;
}

/* ── Optimizer Tests ── */
static int test_sgd_create(void) {
    TEST("sgd_create_step");
    SGD *sgd = sgd_create(0.01f, 0.9f, 0.0001f, false);
    CHECK(sgd != NULL, "sgd create failed");
    int dims[] = {4, 8};
    Tensor *params[2];
    params[0] = tensor_create_randomn(dims, 2);
    params[1] = tensor_create_randomn(dims, 2);
    sgd_set_params(sgd, params, 2);
    sgd_step(sgd);
    sgd_zero_grad(sgd);
    sgd_free(sgd);
    tensor_free(params[0]); tensor_free(params[1]);
    PASS();
    return 0;
}

static int test_adam_create(void) {
    TEST("adam_create_step");
    Adam *adam = adam_create(0.001f, 0.9f, 0.999f, 1e-8f, 0.0f);
    CHECK(adam != NULL, "adam create failed");
    int dims[] = {4, 8};
    Tensor *params[2];
    params[0] = tensor_create_randomn(dims, 2);
    params[1] = tensor_create_randomn(dims, 2);
    adam_set_params(adam, params, 2);
    adam_step(adam);
    adam_zero_grad(adam);
    adam_free(adam);
    tensor_free(params[0]); tensor_free(params[1]);
    PASS();
    return 0;
}

static int test_adamw_create(void) {
    TEST("adamw_create_step");
    AdamW *adamw = adamw_create(0.001f, 0.9f, 0.999f, 1e-8f, 0.01f);
    CHECK(adamw != NULL, "adamw create failed");
    int dims[] = {4, 8};
    Tensor *params[2];
    params[0] = tensor_create_randomn(dims, 2);
    params[1] = tensor_create_randomn(dims, 2);
    adamw_set_params(adamw, params, 2);
    adamw_step(adamw);
    adamw_zero_grad(adamw);
    adamw_free(adamw);
    tensor_free(params[0]); tensor_free(params[1]);
    PASS();
    return 0;
}

/* ── Loss Tests ── */
static int test_mse_loss(void) {
    TEST("mse_loss");
    int dims[] = {4, 1};
    Tensor *pred = tensor_create_ones(dims, 2);
    Tensor *targ = tensor_create_zeros(dims, 2);
    float loss = mse_loss_value(pred, targ, REDUCE_MEAN);
    CHECK(fabsf(loss - 1.0f) < 0.01f, "mse should be 1.0");
    tensor_free(pred); tensor_free(targ);
    PASS();
    return 0;
}

static int test_cross_entropy_loss(void) {
    TEST("cross_entropy_loss");
    int dims[] = {2, 4};
    Tensor *logits = tensor_create_zeros(dims, 2);
    int tdims[] = {2};
    Tensor *target = tensor_create_zeros(tdims, 1);
    float loss = cross_entropy_loss_value(logits, target, REDUCE_MEAN, 0.0f);
    CHECK(isfinite(loss), "ce loss should be finite");
    tensor_free(logits); tensor_free(target);
    PASS();
    return 0;
}

int main(void) {
    printf("=== mini-dl-framework Unit Tests ===\n\n");

    int failed = 0;
    failed += test_node_create();
    failed += test_node_add_mul();
    failed += test_node_relu();
    failed += test_tensor_create();
    failed += test_tensor_add();
    failed += test_tensor_matmul();
    failed += test_linear_create_forward();
    failed += test_conv2d_create_forward();
    failed += test_batchnorm1d();
    failed += test_layernorm();
    failed += test_sgd_create();
    failed += test_adam_create();
    failed += test_adamw_create();
    failed += test_mse_loss();
    failed += test_cross_entropy_loss();

    printf("\n=== Results: %d/%d passed, %d failed ===\n",
           tests_passed, tests_run, failed);
    return failed > 0 ? 1 : 0;
}
