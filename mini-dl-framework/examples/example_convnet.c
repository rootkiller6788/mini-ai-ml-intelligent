#include "nn_layers.h"
#include "loss_funcs.h"
#include <stdio.h>
#include <time.h>

int main(void) {
    srand((unsigned)time(NULL));
    printf("=== mini-dl-framework: Conv2d on MNIST-like data ===\n\n");

    Conv2d* conv1 = conv2d_create(1, 8, 3, 3, 1, 0, true);
    Conv2d* conv2 = conv2d_create(8, 16, 3, 3, 1, 0, true);
    printf("Model: Conv2d(1,8,3)->ReLU->Conv2d(8,16,3)->ReLU->Flatten->Linear(?,10)\n\n");

    int B = 2, C = 1, H = 8, W = 8, K = 10;
    Tensor* x = tensor_create_randomn((int[]){B, C, H, W}, 4);
    float label_data[4] = {3, 7};
    Tensor* labels = tensor_create_from_data(label_data, (int[]){B}, 1, true);

    Tensor* out1 = conv2d_forward(conv1, x);
    printf("Conv1 output: "); tensor_print(out1, "conv1");

    Tensor* a1 = tensor_relu(out1);
    Tensor* out2 = conv2d_forward(conv2, a1);
    printf("Conv2 output: "); tensor_print(out2, "conv2");

    Tensor* a2 = tensor_relu(out2);

    int feat_size = a2->size / B;
    Tensor* flat = tensor_reshape(a2, (int[]){B, feat_size}, 2);
    printf("Flattened: "); tensor_print(flat, "flat");

    printf("\nDone. (Convolutional feature extraction demo)\n");

    tensor_free(x); tensor_free(out1); tensor_free(a1); tensor_free(out2);
    tensor_free(a2); tensor_free(flat); tensor_free(labels);
    conv2d_free(conv1); conv2d_free(conv2);
    return 0;
}
