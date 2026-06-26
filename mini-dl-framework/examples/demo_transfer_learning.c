#include "nn_layers.h"
#include "optimizers.h"
#include "loss_funcs.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define INPUT_DIM 64
#define HIDDEN_DIM 128
#define PRETRAIN_CLASSES 5
#define FINETUNE_CLASSES 3
#define SAMPLES 200
#define EPOCHS 10

static float data[SAMPLES * INPUT_DIM];
static float pretrain_labels[SAMPLES];
static float finetune_labels[SAMPLES];

static void generate_data(void) {
    printf("Generating synthetic data...\n");
    for (int n = 0; n < SAMPLES; n++) {
        float mean = 2.0f * sinf((float)n * 0.1f);
        for (int d = 0; d < INPUT_DIM; d++) {
            float u1 = (float)rand() / (float)RAND_MAX;
            float u2 = (float)rand() / (float)RAND_MAX;
            data[n * INPUT_DIM + d] = mean + 0.5f * sqrtf(-2.0f * logf(u1 + 1e-12f)) * cosf(2.0f * 3.14159265f * u2);
        }
        pretrain_labels[n] = (float)(n % PRETRAIN_CLASSES);
        finetune_labels[n] = (float)(n % PRETRAIN_CLASSES + 1);
        if (finetune_labels[n] >= PRETRAIN_CLASSES) finetune_labels[n] = (float)(n % FINETUNE_CLASSES);
    }
}

static float compute_accuracy_linear(Linear* fc1, Linear* fc2,
                                     float* x_data, float* y_data,
                                     int N, int num_classes) {
    int correct = 0;
    for (int n = 0; n < N; n++) {
        Tensor* x = tensor_create_from_data(
            x_data + n * INPUT_DIM, (int[]){1, INPUT_DIM}, 2, true);
        Tensor* h1 = linear_forward(fc1, x);
        Tensor* a1 = tensor_relu(h1);
        Tensor* logits = linear_forward(fc2, a1);
        int pred = 0;
        float max_val = logits->data[0];
        for (int c = 1; c < num_classes; c++) {
            if (logits->data[c] > max_val) { max_val = logits->data[c]; pred = c; }
        }
        if (pred == (int)y_data[n]) correct++;
        tensor_free(x); tensor_free(h1); tensor_free(a1); tensor_free(logits);
    }
    return 100.0f * (float)correct / (float)N;
}

int main(void) {
    srand((unsigned)time(NULL));
    printf("=== mini-dl-framework: Transfer Learning Demo ===\n\n");

    generate_data();

    printf("Phase 1: Pretraining encoder on %d-class task\n", PRETRAIN_CLASSES);
    Linear* encoder = linear_create(INPUT_DIM, HIDDEN_DIM, true);
    Linear* pretrain_head = linear_create(HIDDEN_DIM, PRETRAIN_CLASSES, true);

    float lr = 0.01f;
    int batch_size = 16;
    int steps = SAMPLES / batch_size;

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float epoch_loss = 0;
        for (int step = 0; step < steps; step++) {
            int start = step * batch_size;
            Tensor* x = tensor_create_from_data(
                data + start * INPUT_DIM, (int[]){batch_size, INPUT_DIM}, 2, true);
            Tensor* y = tensor_create_from_data(
                pretrain_labels + start, (int[]){batch_size}, 1, true);
            Tensor* h = linear_forward(encoder, x);
            Tensor* a = tensor_relu(h);
            Tensor* logits = linear_forward(pretrain_head, a);
            float loss = cross_entropy_loss_value(logits, y, REDUCE_MEAN, 0.0f);
            epoch_loss += loss;

            Tensor* d_logits = cross_entropy_loss_backward(logits, y, REDUCE_MEAN, 0.0f);
            Tensor* d_a = linear_backward(pretrain_head, d_logits);
            for (int i = 0; i < d_a->size; i++)
                d_a->data[i] *= (a->data[i] > 0 ? 1.0f : 0.0f);
            Tensor* ignore = linear_backward(encoder, d_a);

            for (int i = 0; i < encoder->weight->size; i++)
                encoder->weight->data[i] -= lr * encoder->weight_grad->data[i];
            for (int i = 0; i < encoder->bias->size; i++)
                encoder->bias->data[i] -= lr * encoder->bias_grad->data[i];
            for (int i = 0; i < pretrain_head->weight->size; i++)
                pretrain_head->weight->data[i] -= lr * pretrain_head->weight_grad->data[i];
            for (int i = 0; i < pretrain_head->bias->size; i++)
                pretrain_head->bias->data[i] -= lr * pretrain_head->bias_grad->data[i];

            tensor_fill_(encoder->weight_grad, 0); tensor_fill_(encoder->bias_grad, 0);
            tensor_fill_(pretrain_head->weight_grad, 0); tensor_fill_(pretrain_head->bias_grad, 0);

            tensor_free(x); tensor_free(y); tensor_free(h); tensor_free(a);
            tensor_free(logits); tensor_free(d_logits); tensor_free(d_a); tensor_free(ignore);
        }
        float acc = compute_accuracy_linear(encoder, pretrain_head,
            data, pretrain_labels, SAMPLES, PRETRAIN_CLASSES);
        printf("Pretrain epoch %2d: loss=%.4f, acc=%.1f%%\n",
               epoch + 1, epoch_loss / steps, acc);
    }

    printf("\nPhase 2: Fine-tuning on %d-class task (frozen encoder)\n", FINETUNE_CLASSES);
    Linear* finetune_head = linear_create(HIDDEN_DIM, FINETUNE_CLASSES, true);

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float epoch_loss = 0;
        for (int step = 0; step < steps; step++) {
            int start = step * batch_size;
            Tensor* x = tensor_create_from_data(
                data + start * INPUT_DIM, (int[]){batch_size, INPUT_DIM}, 2, true);
            Tensor* y = tensor_create_from_data(
                finetune_labels + start, (int[]){batch_size}, 1, true);
            Tensor* h = linear_forward(encoder, x);
            Tensor* a = tensor_relu(h);
            Tensor* logits = linear_forward(finetune_head, a);
            float loss = cross_entropy_loss_value(logits, y, REDUCE_MEAN, 0.0f);
            epoch_loss += loss;

            Tensor* d_logits = cross_entropy_loss_backward(logits, y, REDUCE_MEAN, 0.0f);
            Tensor* d_a = linear_backward(finetune_head, d_logits);
            for (int i = 0; i < d_a->size; i++)
                d_a->data[i] *= (a->data[i] > 0 ? 1.0f : 0.0f);

            for (int i = 0; i < finetune_head->weight->size; i++)
                finetune_head->weight->data[i] -= lr * finetune_head->weight_grad->data[i];
            for (int i = 0; i < finetune_head->bias->size; i++)
                finetune_head->bias->data[i] -= lr * finetune_head->bias_grad->data[i];
            tensor_fill_(finetune_head->weight_grad, 0);
            tensor_fill_(finetune_head->bias_grad, 0);

            tensor_free(x); tensor_free(y); tensor_free(h); tensor_free(a);
            tensor_free(logits); tensor_free(d_logits); tensor_free(d_a);
        }
        float acc = compute_accuracy_linear(encoder, finetune_head,
            data, finetune_labels, SAMPLES, FINETUNE_CLASSES);
        printf("Finetune epoch %2d: loss=%.4f, acc=%.1f%%\n",
               epoch + 1, epoch_loss / steps, acc);
    }

    printf("\n=== Summary ===\n");
    printf("Pretrain (encoder + %d-class head): complete\n", PRETRAIN_CLASSES);
    printf("Finetune (encoder frozen + %d-class head): complete\n", FINETUNE_CLASSES);
    printf("Transfer learning successfully demonstrated.\n");

    linear_free(encoder);
    linear_free(pretrain_head);
    linear_free(finetune_head);
    printf("\nDone.\n");
    return 0;
}
