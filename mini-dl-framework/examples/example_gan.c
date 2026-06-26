#include "nn_layers.h"
#include "optimizers.h"
#include "loss_funcs.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

static float rand_norm(void) {
    float u1 = (float)rand() / (float)RAND_MAX;
    float u2 = (float)rand() / (float)RAND_MAX;
    return sqrtf(-2.0f * logf(u1 + 1e-12f)) * cosf(2.0f * 3.14159265f * u2);
}

static float* generate_sine_data(int N, int seq_len) {
    float* data = (float*)malloc(sizeof(float) * N * seq_len);
    for (int i = 0; i < N; i++) {
        float freq = 0.5f + 1.5f * ((float)rand() / (float)RAND_MAX);
        float phase = 2.0f * 3.14159265f * ((float)rand() / (float)RAND_MAX);
        for (int t = 0; t < seq_len; t++) {
            data[i * seq_len + t] = sinf(freq * 0.1f * t + phase) + 0.1f * rand_norm();
        }
    }
    return data;
}

int main(void) {
    srand((unsigned)time(NULL));
    printf("=== mini-dl-framework: GAN-like time-series ===\n\n");

    int N = 128, seq_len = 32, latent_dim = 16;
    printf("Generator: Linear(%d,64)->ReLU->Linear(64,%d)\n", latent_dim, seq_len);
    printf("Discriminator: Linear(%d,32)->ReLU->Linear(32,1)->Sigmoid\n\n", seq_len);

    float* real_data = generate_sine_data(N, seq_len);

    Linear* gen_fc1 = linear_create(latent_dim, 64, true);
    Linear* gen_fc2 = linear_create(64, seq_len, true);
    Linear* disc_fc1 = linear_create(seq_len, 32, true);
    Linear* disc_fc2 = linear_create(32, 1, true);

    float disc_loss_accum = 0, gen_loss_accum = 0;
    int batch_size = 16;

    for (int epoch = 0; epoch < 1000; epoch++) {
        for (int step = 0; step < N / batch_size; step++) {
            int idx = (step * batch_size) % N;

            Tensor* z = tensor_create_randomn((int[]){batch_size, latent_dim}, 2);
            Tensor* h1 = linear_forward(gen_fc1, z);
            Tensor* a1 = tensor_relu(h1);
            Tensor* fake = linear_forward(gen_fc2, a1);

            Tensor* real_batch = tensor_create_from_data(
                real_data + idx * seq_len * sizeof(float) / sizeof(float),
                (int[]){batch_size, seq_len}, 2, true);

            Tensor* dr1 = linear_forward(disc_fc1, real_batch);
            Tensor* da1 = tensor_relu(dr1);
            Tensor* dr_out = linear_forward(disc_fc2, da1);
            Tensor* real_pred = tensor_sigmoid(dr_out);

            Tensor* df1 = linear_forward(disc_fc1, fake);
            Tensor* da2 = tensor_relu(df1);
            Tensor* df_out = linear_forward(disc_fc2, da2);
            Tensor* fake_pred = tensor_sigmoid(df_out);

            Tensor* ones = tensor_create_ones((int[]){batch_size, 1}, 2);
            Tensor* zeros = tensor_create_zeros((int[]){batch_size, 1}, 2);

            float d_loss_real = bce_with_logits_value(dr_out, ones, REDUCE_MEAN);
            float d_loss_fake = bce_with_logits_value(df_out, zeros, REDUCE_MEAN);
            disc_loss_accum = 0.9f * disc_loss_accum + 0.1f * (d_loss_real + d_loss_fake);

            float g_loss = bce_with_logits_value(df_out, ones, REDUCE_MEAN);
            gen_loss_accum = 0.9f * gen_loss_accum + 0.1f * g_loss;

            tensor_free(z); tensor_free(h1); tensor_free(a1); tensor_free(fake);
            tensor_free(real_batch); tensor_free(dr1); tensor_free(da1);
            tensor_free(dr_out); tensor_free(real_pred); tensor_free(df1);
            tensor_free(da2); tensor_free(df_out); tensor_free(fake_pred);
            tensor_free(ones); tensor_free(zeros);
        }

        if (epoch % 100 == 0) {
            printf("Epoch %4d: D_loss=%.4f G_loss=%.4f\n",
                   epoch, disc_loss_accum, gen_loss_accum);
        }
    }

    printf("\n=== Generate samples ===\n");
    Tensor* z = tensor_create_randomn((int[]){4, latent_dim}, 2);
    Tensor* h1 = linear_forward(gen_fc1, z);
    Tensor* a1 = tensor_relu(h1);
    Tensor* gen_out = linear_forward(gen_fc2, a1);

    for (int i = 0; i < 4; i++) {
        printf("Sample %d: [", i);
        for (int t = 0; t < 8; t++) {
            printf("%.3f%s", gen_out->data[i * seq_len + t], t < 7 ? ", " : "");
        }
        printf(", ...]\n");
    }

    free(real_data);
    tensor_free(z); tensor_free(h1); tensor_free(a1); tensor_free(gen_out);
    linear_free(gen_fc1); linear_free(gen_fc2);
    linear_free(disc_fc1); linear_free(disc_fc2);
    printf("\nDone.\n");
    return 0;
}
