/*
 * mini-dl-framework — Core Benchmarks
 *
 * Benchmarks: autograd, tensor ops, NN layers, optimizers, loss functions.
 */
#include "../include/autograd.h"
#include "../include/tensor_ops.h"
#include "../include/nn_layers.h"
#include "../include/optimizers.h"
#include "../include/loss_funcs.h"
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
    printf("=== mini-dl-framework Benchmarks (N=%d) ===\n\n", N);

    /* ── Autograd ── */
    {
        t0 = now_ms();
        for (int r = 0; r < N / 10; r++) {
            Node *a = node_create(2.0f, true);
            Node *b = node_create(3.0f, true);
            Node *s = node_add(a, b);
            Node *m = node_mul(s, b);
            forward(m);
            backward(m);
            node_free(a); node_free(b);
            node_free(s); node_free(m);
        }
        t1 = now_ms();
        printf("  autograd_fwd_bwd:   %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 10, t1 - t0, (t1 - t0) / (double)(N / 10) * 1000.0);
    }

    /* ── Tensor add ── */
    {
        int dims[] = {128, 64};
        Tensor *a = tensor_create_randomn(dims, 2);
        Tensor *b = tensor_create_randomn(dims, 2);
        t0 = now_ms();
        for (int r = 0; r < N / 10; r++) {
            Tensor *c = tensor_add(a, b);
            tensor_free(c);
        }
        t1 = now_ms();
        printf("  tensor_add:         %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 10, t1 - t0, (t1 - t0) / (double)(N / 10) * 1000.0);
        tensor_free(a); tensor_free(b);
    }

    /* ── Tensor matmul ── */
    {
        int da[] = {64, 128}, db[] = {128, 64};
        Tensor *a = tensor_create_randomn(da, 2);
        Tensor *b = tensor_create_randomn(db, 2);
        t0 = now_ms();
        for (int r = 0; r < N / 20; r++) {
            Tensor *c = tensor_matmul(a, b);
            tensor_free(c);
        }
        t1 = now_ms();
        printf("  tensor_matmul:      %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 20, t1 - t0, (t1 - t0) / (double)(N / 20) * 1000.0);
        tensor_free(a); tensor_free(b);
    }

    /* ── Linear ── */
    {
        int dims[] = {32, 128};
        t0 = now_ms();
        for (int r = 0; r < N / 20; r++) {
            Linear *l = linear_create(128, 64, true);
            Tensor *in = tensor_create_randomn(dims, 2);
            Tensor *out = linear_forward(l, in);
            linear_free(l);
            tensor_free(in); tensor_free(out);
        }
        t1 = now_ms();
        printf("  linear_fwd:         %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 20, t1 - t0, (t1 - t0) / (double)(N / 20) * 1000.0);
    }

    /* ── Conv2d ── */
    {
        int dims[] = {4, 3, 32, 32};
        t0 = now_ms();
        for (int r = 0; r < N / 20; r++) {
            Conv2d *c = conv2d_create(3, 16, 3, 3, 1, 1, true);
            Tensor *in = tensor_create_randomn(dims, 4);
            Tensor *out = conv2d_forward(c, in);
            conv2d_free(c);
            tensor_free(in); tensor_free(out);
        }
        t1 = now_ms();
        printf("  conv2d_fwd:         %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 20, t1 - t0, (t1 - t0) / (double)(N / 20) * 1000.0);
    }

    /* ── BatchNorm1d ── */
    {
        int dims[] = {32, 64};
        t0 = now_ms();
        for (int r = 0; r < N / 20; r++) {
            BatchNorm1d *bn = batchnorm1d_create(64, 1e-5f, 0.1f);
            Tensor *in = tensor_create_randomn(dims, 2);
            Tensor *out = batchnorm1d_forward(bn, in);
            batchnorm1d_free(bn);
            tensor_free(in); tensor_free(out);
        }
        t1 = now_ms();
        printf("  batchnorm1d_fwd:    %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 20, t1 - t0, (t1 - t0) / (double)(N / 20) * 1000.0);
    }

    /* ── SGD ── */
    {
        int dims[] = {128, 64};
        t0 = now_ms();
        for (int r = 0; r < N / 20; r++) {
            SGD *sgd = sgd_create(0.01f, 0.9f, 0.0f, false);
            Tensor *params[2];
            params[0] = tensor_create_randomn(dims, 2);
            params[1] = tensor_create_randomn(dims, 2);
            sgd_set_params(sgd, params, 2);
            sgd_step(sgd);
            sgd_free(sgd);
            tensor_free(params[0]); tensor_free(params[1]);
        }
        t1 = now_ms();
        printf("  sgd_step:           %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 20, t1 - t0, (t1 - t0) / (double)(N / 20) * 1000.0);
    }

    /* ── Adam ── */
    {
        int dims[] = {128, 64};
        t0 = now_ms();
        for (int r = 0; r < N / 20; r++) {
            Adam *adam = adam_create(0.001f, 0.9f, 0.999f, 1e-8f, 0.0f);
            Tensor *params[2];
            params[0] = tensor_create_randomn(dims, 2);
            params[1] = tensor_create_randomn(dims, 2);
            adam_set_params(adam, params, 2);
            adam_step(adam);
            adam_free(adam);
            tensor_free(params[0]); tensor_free(params[1]);
        }
        t1 = now_ms();
        printf("  adam_step:          %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 20, t1 - t0, (t1 - t0) / (double)(N / 20) * 1000.0);
    }

    /* ── MSE Loss ── */
    {
        int dims[] = {64, 1};
        Tensor *pred = tensor_create_randomn(dims, 2);
        Tensor *targ = tensor_create_randomn(dims, 2);
        t0 = now_ms();
        for (int r = 0; r < N / 10; r++) {
            mse_loss_value(pred, targ, REDUCE_MEAN);
        }
        t1 = now_ms();
        printf("  mse_loss:           %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 10, t1 - t0, (t1 - t0) / (double)(N / 10) * 1000.0);
        tensor_free(pred); tensor_free(targ);
    }

    /* ── CrossEntropy ── */
    {
        int ld[] = {32, 10}, td[] = {32};
        Tensor *logits = tensor_create_randomn(ld, 2);
        Tensor *target = tensor_create_zeros(td, 1);
        t0 = now_ms();
        for (int r = 0; r < N / 10; r++) {
            cross_entropy_loss_value(logits, target, REDUCE_MEAN, 0.0f);
        }
        t1 = now_ms();
        printf("  ce_loss:            %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 10, t1 - t0, (t1 - t0) / (double)(N / 10) * 1000.0);
        tensor_free(logits); tensor_free(target);
    }

    printf("\nDone.\n");
    return 0;
}
