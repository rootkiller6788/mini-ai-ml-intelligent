/*
 * mini-dl-framework — Full Demo: Deep Learning Framework Basics
 *
 * Demonstrates: autograd, tensor ops, NN layers, optimizers, loss functions.
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

int main(void) {
    printf("=== mini-dl-framework: DL Framework Basics Demo ===\n\n");

    /* Step 1: Autograd — compute graph */
    printf("-- Step 1: Autograd (compute graph) --\n");
    Node *a = node_create(2.0f, true);
    Node *b = node_create(3.0f, true);
    Node *sum = node_add(a, b);
    Node *prod = node_mul(sum, b);
    forward(prod);
    backward(prod);
    printf("a=2.0, b=3.0: a+b=%.1f, (a+b)*b=%.1f\n", sum->value, prod->value);
    printf("  grad: d/db((a+b)*b) = a+2b = 2+6 = %.1f (actual=%.1f)\n", 8.0f, b->grad);
    node_free(a); node_free(b);
    node_free(sum); node_free(prod);

    /* Step 2: Tensor ops */
    printf("\n-- Step 2: Tensor Operations --\n");
    int dims[] = {2, 3};
    Tensor *t1 = tensor_create_ones(dims, 2);
    Tensor *t2 = tensor_create_zeros(dims, 2);
    tensor_fill_(t2, 2.0f);
    Tensor *t_add = tensor_add(t1, t2);
    int idx[] = {0, 0};
    printf("ones + twos: [0,0]=%.1f\n", tensor_get(t_add, idx));
    tensor_free(t1); tensor_free(t2); tensor_free(t_add);

    int da[] = {2, 3}, db[] = {3, 2};
    Tensor *A = tensor_create_ones(da, 2);
    Tensor *B = tensor_create_ones(db, 2);
    Tensor *C = tensor_matmul(A, B);
    printf("matmul(2x3 * 3x2): [0,0]=%.1f\n", tensor_get(C, idx));
    tensor_free(A); tensor_free(B); tensor_free(C);

    /* Step 3: NN Layers — Linear + Conv2d */
    printf("\n-- Step 3: NN Layers --\n");
    int ld[] = {4, 16};
    Tensor *x = tensor_create_randomn(ld, 2);
    Linear *fc = linear_create(16, 8, true);
    Tensor *fc_out = linear_forward(fc, x);
    printf("Linear: 16->8, in=(4,16) -> out=(%d,%d)\n",
           fc_out->dims[0], fc_out->dims[1]);
    linear_free(fc);
    tensor_free(x); tensor_free(fc_out);

    int cd[] = {1, 3, 32, 32};
    Tensor *img = tensor_create_randomn(cd, 4);
    Conv2d *conv = conv2d_create(3, 16, 3, 3, 1, 1, true);
    Tensor *conv_out = conv2d_forward(conv, img);
    printf("Conv2d: 3->16, k=3 -> out=(%d,%d,%d,%d)\n",
           conv_out->dims[0], conv_out->dims[1],
           conv_out->dims[2], conv_out->dims[3]);
    conv2d_free(conv);
    tensor_free(img); tensor_free(conv_out);

    /* Step 4: Optimizers — SGD, Adam, AdamW */
    printf("\n-- Step 4: Optimizers --\n");
    int pd[] = {4, 8};
    Tensor *w = tensor_create_randomn(pd, 2);
    Tensor *b = tensor_create_randomn(pd, 2);

    SGD *sgd = sgd_create(0.01f, 0.9f, 0.0001f, false);
    Tensor *sgd_params[] = {w, b};
    sgd_set_params(sgd, sgd_params, 2);
    sgd_step(sgd);
    printf("SGD: lr=0.01, momentum=0.9 — 1 step done\n");
    sgd_free(sgd);

    Adam *adam = adam_create(0.001f, 0.9f, 0.999f, 1e-8f, 0.0f);
    AdamW *adamw = adamw_create(0.001f, 0.9f, 0.999f, 1e-8f, 0.01f);
    printf("Adam + AdamW created (lr=0.001)\n");
    adam_free(adam);
    adamw_free(adamw);
    tensor_free(w); tensor_free(b);

    /* Step 5: Loss Functions */
    printf("\n-- Step 5: Loss Functions --\n");
    int ldim[] = {4, 1};
    Tensor *pred = tensor_create_ones(ldim, 2);
    Tensor *targ = tensor_create_zeros(ldim, 2);
    float mse = mse_loss_value(pred, targ, REDUCE_MEAN);
    printf("MSE(all ones vs zeros): %.4f\n", mse);
    tensor_free(pred); tensor_free(targ);

    int cdim[] = {3, 5};
    Tensor *logits = tensor_create_randomn(cdim, 2);
    int tdim[] = {3};
    Tensor *labels = tensor_create_zeros(tdim, 1);
    float ce = cross_entropy_loss_value(logits, labels, REDUCE_MEAN, 0.0f);
    printf("CrossEntropy(3x5): %.4f\n", ce);
    tensor_free(logits); tensor_free(labels);

    /* Step 6: LR Scheduler */
    printf("\n-- Step 6: LR Scheduler --\n");
    LRScheduler *sched = lr_scheduler_create(LR_COSINE, 0.1f, 50, 0.5f, 100, 0.001f, 10);
    printf("Cosine LR: base=0.1, T_max=100, warmup=10\n");
    for (int i = 0; i < 5; i++) { lr_scheduler_step(sched); }
    printf("  after 5 steps: lr=%.6f\n", lr_scheduler_get_lr(sched));
    lr_scheduler_free(sched);

    printf("\nDL framework basics demo complete!\n");
    return 0;
}
