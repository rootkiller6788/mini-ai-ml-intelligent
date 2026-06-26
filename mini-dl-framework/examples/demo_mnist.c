#include "nn_layers.h"
#include "optimizers.h"
#include "loss_funcs.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define TRAIN_SAMPLES 600
#define TEST_SAMPLES 100
#define IMAGE_SIZE 28
#define NUM_CLASSES 10
#define BATCH_SIZE 32
#define EPOCHS 5

static float train_images[TRAIN_SAMPLES * IMAGE_SIZE * IMAGE_SIZE];
static float train_labels[TRAIN_SAMPLES];

static float test_images[TEST_SAMPLES * IMAGE_SIZE * IMAGE_SIZE];
static float test_labels[TEST_SAMPLES];

static void generate_synthetic_mnist(void) {
    printf("Generating synthetic MNIST-like data...\n");
    for (int n = 0; n < TRAIN_SAMPLES; n++) {
        int digit = n % 10;
        train_labels[n] = (float)digit;
        float cx = 14.0f + 4.0f * sinf((float)n * 0.3f);
        float cy = 14.0f + 4.0f * cosf((float)n * 0.7f);
        for (int r = 0; r < IMAGE_SIZE; r++) {
            for (int c = 0; c < IMAGE_SIZE; c++) {
                float dist = sqrtf((r - cx) * (r - cx) + (c - cy) * (c - cy));
                float val = expf(-dist * dist / (15.0f + digit * 1.5f));
                float noise = 0.02f * ((float)rand() / (float)RAND_MAX);
                train_images[n * IMAGE_SIZE * IMAGE_SIZE + r * IMAGE_SIZE + c] = val + noise;
            }
        }
    }
    for (int n = 0; n < TEST_SAMPLES; n++) {
        int digit = (n + 3) % 10;
        test_labels[n] = (float)digit;
        float cx = 14.0f + 3.0f * sinf((float)n * 0.5f);
        float cy = 14.0f + 3.0f * cosf((float)n * 0.6f);
        for (int r = 0; r < IMAGE_SIZE; r++) {
            for (int c = 0; c < IMAGE_SIZE; c++) {
                float dist = sqrtf((r - cx) * (r - cx) + (c - cy) * (c - cy));
                float val = expf(-dist * dist / (15.0f + digit * 1.5f));
                float noise = 0.02f * ((float)rand() / (float)RAND_MAX);
                test_images[n * IMAGE_SIZE * IMAGE_SIZE + r * IMAGE_SIZE + c] = val + noise;
            }
        }
    }
}

static void shuffle_data(float* images, float* labels, int N) {
    for (int i = N - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        for (int k = 0; k < IMAGE_SIZE * IMAGE_SIZE; k++) {
            float tmp = images[i * IMAGE_SIZE * IMAGE_SIZE + k];
            images[i * IMAGE_SIZE * IMAGE_SIZE + k] = images[j * IMAGE_SIZE * IMAGE_SIZE + k];
            images[j * IMAGE_SIZE * IMAGE_SIZE + k] = tmp;
        }
        float tmp_l = labels[i];
        labels[i] = labels[j];
        labels[j] = tmp_l;
    }
}

static float compute_accuracy(Linear* fc1, Linear* fc2,
                              float* images, float* labels, int N) {
    int correct = 0;
    for (int n = 0; n < N; n++) {
        Tensor* x = tensor_create_from_data(
            images + n * IMAGE_SIZE * IMAGE_SIZE,
            (int[]){1, IMAGE_SIZE * IMAGE_SIZE}, 2, true);
        Tensor* h1 = linear_forward(fc1, x);
        Tensor* a1 = tensor_relu(h1);
        Tensor* h2 = linear_forward(fc2, a1);
        int pred = 0;
        float max_val = h2->data[0];
        for (int c = 1; c < NUM_CLASSES; c++) {
            if (h2->data[c] > max_val) { max_val = h2->data[c]; pred = c; }
        }
        if (pred == (int)labels[n]) correct++;
        tensor_free(x); tensor_free(h1); tensor_free(a1); tensor_free(h2);
    }
    return 100.0f * (float)correct / (float)N;
}

int main(void) {
    srand((unsigned)time(NULL));
    printf("=== mini-dl-framework: MNIST Demo ===\n\n");

    generate_synthetic_mnist();
    printf("Train samples: %d, Test samples: %d\n\n", TRAIN_SAMPLES, TEST_SAMPLES);

    Linear* fc1 = linear_create(IMAGE_SIZE * IMAGE_SIZE, 128, true);
    Linear* fc2 = linear_create(128, NUM_CLASSES, true);
    printf("Model: Linear(784,128)->ReLU->Linear(128,10)\n\n");

    float lr = 0.01f;
    int steps_per_epoch = TRAIN_SAMPLES / BATCH_SIZE;

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        shuffle_data(train_images, train_labels, TRAIN_SAMPLES);
        float epoch_loss = 0;

        for (int step = 0; step < steps_per_epoch; step++) {
            int start = step * BATCH_SIZE;

            Tensor* x = tensor_create_from_data(
                train_images + start * IMAGE_SIZE * IMAGE_SIZE,
                (int[]){BATCH_SIZE, IMAGE_SIZE * IMAGE_SIZE}, 2, true);
            Tensor* y = tensor_create_from_data(
                train_labels + start,
                (int[]){BATCH_SIZE}, 1, true);

            Tensor* h1 = linear_forward(fc1, x);
            Tensor* a1 = tensor_relu(h1);
            Tensor* logits = linear_forward(fc2, a1);

            float loss = cross_entropy_loss_value(logits, y, REDUCE_MEAN, 0.0f);
            epoch_loss += loss;

            Tensor* d_logits = cross_entropy_loss_backward(logits, y, REDUCE_MEAN, 0.0f);
            Tensor* d_a1 = linear_backward(fc2, d_logits);
            for (int i = 0; i < d_a1->size; i++)
                d_a1->data[i] *= (a1->data[i] > 0 ? 1.0f : 0.0f);
            Tensor* d_x = linear_backward(fc1, d_a1);

            for (int i = 0; i < fc1->weight->size; i++)
                fc1->weight->data[i] -= lr * fc1->weight_grad->data[i];
            for (int i = 0; i < fc1->bias->size; i++)
                fc1->bias->data[i] -= lr * fc1->bias_grad->data[i];
            for (int i = 0; i < fc2->weight->size; i++)
                fc2->weight->data[i] -= lr * fc2->weight_grad->data[i];
            for (int i = 0; i < fc2->bias->size; i++)
                fc2->bias->data[i] -= lr * fc2->bias_grad->data[i];

            tensor_fill_(fc1->weight_grad, 0);
            tensor_fill_(fc1->bias_grad, 0);
            tensor_fill_(fc2->weight_grad, 0);
            tensor_fill_(fc2->bias_grad, 0);

            tensor_free(x); tensor_free(y); tensor_free(h1); tensor_free(a1);
            tensor_free(logits); tensor_free(d_logits); tensor_free(d_a1); tensor_free(d_x);
        }

        float train_acc = compute_accuracy(fc1, fc2, train_images, train_labels, TRAIN_SAMPLES);
        float test_acc = compute_accuracy(fc1, fc2, test_images, test_labels, TEST_SAMPLES);
        printf("Epoch %d: loss=%.4f, train_acc=%.1f%%, test_acc=%.1f%%\n",
               epoch + 1, epoch_loss / steps_per_epoch, train_acc, test_acc);
    }

    printf("\n=== Final evaluation ===\n");
    float final_acc = compute_accuracy(fc1, fc2, test_images, test_labels, TEST_SAMPLES);
    printf("Test accuracy: %.2f%%\n", final_acc);

    linear_free(fc1);
    linear_free(fc2);
    printf("\nDone.\n");
    return 0;
}
