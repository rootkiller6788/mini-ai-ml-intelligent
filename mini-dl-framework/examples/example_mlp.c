#include "nn_layers.h"
#include "optimizers.h"
#include "loss_funcs.h"
#include <stdio.h>
#include <time.h>

int main(void) {
    srand((unsigned)time(NULL));
    printf("=== mini-dl-framework: MLP on XOR ===\n\n");

    Linear* fc1 = linear_create(2, 16, true);
    Linear* fc2 = linear_create(16, 8, true);
    Linear* fc3 = linear_create(8, 1, true);
    printf("Model: Linear(2,16)->ReLU->Linear(16,8)->ReLU->Linear(8,1)->Sigmoid\n\n");

    float xor_x[4][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
    float xor_y[4] = {0, 1, 1, 0};

    SGD* sgd = sgd_create(0.5f, 0.0f, 0.0f, false);
    Tensor* params[6];
    params[0] = fc1->weight; params[1] = fc1->bias;
    params[2] = fc2->weight; params[3] = fc2->bias;
    params[4] = fc3->weight; params[5] = fc3->bias;
    sgd_set_params(sgd, params, 6);

    for (int epoch = 0; epoch < 5000; epoch++) {
        float total_loss = 0;

        for (int b = 0; b < 4; b++) {
            Tensor* x = tensor_create_from_data(xor_x[b], (int[]){1, 2}, 2, true);
            Tensor* y = tensor_create_from_data(&xor_y[b], (int[]){1, 1}, 2, true);

            Tensor* h1 = linear_forward(fc1, x);
            Tensor* a1 = tensor_relu(h1);
            Tensor* h2 = linear_forward(fc2, a1);
            Tensor* a2 = tensor_relu(h2);
            Tensor* h3 = linear_forward(fc3, a2);
            Tensor* pred = tensor_sigmoid(h3);

            float loss = bce_with_logits_value(h3, y, REDUCE_MEAN);
            total_loss += loss;

            Tensor* d_loss = bce_with_logits_backward(h3, y, REDUCE_MEAN);
            Tensor* d_a2 = linear_backward(fc3, d_loss);
            Tensor* d_h2 = tensor_copy(d_a2);
            for (int i = 0; i < d_h2->size; i++)
                d_h2->data[i] *= (a2->data[i] > 0 ? 1.0f : 0.0f);
            Tensor* d_a1 = linear_backward(fc2, d_h2);
            Tensor* d_h1 = tensor_copy(d_a1);
            for (int i = 0; i < d_h1->size; i++)
                d_h1->data[i] *= (a1->data[i] > 0 ? 1.0f : 0.0f);
            Tensor* ignore = linear_backward(fc1, d_h1);

            float lr = 0.5f;
            for (int i = 0; i < fc1->weight->size; i++)
                fc1->weight->data[i] -= lr * fc1->weight_grad->data[i];
            for (int i = 0; i < fc1->bias->size; i++)
                fc1->bias->data[i] -= lr * fc1->bias_grad->data[i];
            for (int i = 0; i < fc2->weight->size; i++)
                fc2->weight->data[i] -= lr * fc2->weight_grad->data[i];
            for (int i = 0; i < fc2->bias->size; i++)
                fc2->bias->data[i] -= lr * fc2->bias_grad->data[i];
            for (int i = 0; i < fc3->weight->size; i++)
                fc3->weight->data[i] -= lr * fc3->weight_grad->data[i];
            for (int i = 0; i < fc3->bias->size; i++)
                fc3->bias->data[i] -= lr * fc3->bias_grad->data[i];

            tensor_fill_(fc1->weight_grad, 0);
            tensor_fill_(fc1->bias_grad, 0);
            tensor_fill_(fc2->weight_grad, 0);
            tensor_fill_(fc2->bias_grad, 0);
            tensor_fill_(fc3->weight_grad, 0);
            tensor_fill_(fc3->bias_grad, 0);

            tensor_free(x); tensor_free(y); tensor_free(h1); tensor_free(a1);
            tensor_free(h2); tensor_free(a2); tensor_free(h3); tensor_free(pred);
            tensor_free(d_loss); tensor_free(d_a2); tensor_free(d_h2);
            tensor_free(d_a1); tensor_free(d_h1); tensor_free(ignore);
        }

        if (epoch % 1000 == 0)
            printf("Epoch %4d: loss=%.6f\n", epoch, total_loss / 4.0f);
    }

    printf("\n=== Final predictions ===\n");
    for (int b = 0; b < 4; b++) {
        Tensor* x = tensor_create_from_data(xor_x[b], (int[]){1, 2}, 2, true);
        Tensor* h1 = linear_forward(fc1, x);
        Tensor* a1 = tensor_relu(h1);
        Tensor* h2 = linear_forward(fc2, a1);
        Tensor* a2 = tensor_relu(h2);
        Tensor* h3 = linear_forward(fc3, a2);
        Tensor* pred = tensor_sigmoid(h3);
        printf("XOR(%.0f, %.0f) = %.6f (target: %.0f)\n",
               xor_x[b][0], xor_x[b][1], pred->data[0], xor_y[b]);

        tensor_free(x); tensor_free(h1); tensor_free(a1); tensor_free(h2);
        tensor_free(a2); tensor_free(h3); tensor_free(pred);
    }

    sgd_free(sgd);
    linear_free(fc1); linear_free(fc2); linear_free(fc3);
    printf("\nDone.\n");
    return 0;
}
