#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "gan_model.h"

int main(void) {
    printf("=== mini-model-arch: GAN Demo ===\n\n");
    srand((unsigned)time(NULL));

    int noise_dim = 100, data_dim = 784;
    int gen_hid[] = {256, 512, 1024};
    int disc_hid[] = {512, 256};
    int batch_size = 32, epochs = 5;

    printf("1. Creating GAN (noise=%d, data=%d)\n", noise_dim, data_dim);
    printf("   Generator: %d -> 256 -> 512 -> 1024 -> %d\n", noise_dim, data_dim);
    printf("   Discriminator: %d -> 512 -> 256 -> 1\n", data_dim);
    GAN *gan = gan_create(noise_dim, data_dim, gen_hid, 3, disc_hid, 2, 0);
    printf("   GAN created successfully.\n");

    printf("\n2. Generator Forward Pass\n");
    float noise[100];
    for (int i = 0; i < 100; i++)
        noise[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    float *fake_img = gen_forward(gan->gen, noise);
    printf("   Generated sample: [%.4f, %.4f, %.4f, %.4f, ...]\n",
           fake_img[0], fake_img[1], fake_img[2], fake_img[3]);
    printf("   Range: [%.4f, %.4f] (should be ~[-1, 1])\n",
           fake_img[0], fake_img[784 - 1]);
    free(fake_img);

    printf("\n3. Discriminator Forward Pass\n");
    float *real_img = (float *)malloc(data_dim * sizeof(float));
    for (int i = 0; i < data_dim; i++)
        real_img[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    float real_score = disc_forward(gan->disc, real_img);
    printf("   Real score: %.4f (higher = more real)\n", real_score);

    float *fake_img2 = gen_forward(gan->gen, noise);
    float fake_score = disc_forward(gan->disc, fake_img2);
    printf("   Fake score: %.4f (should be lower than real)\n", fake_score);
    free(fake_img2); free(real_img);

    printf("\n4. Training Loop (mini-batch=%d, epochs=%d)\n", batch_size, epochs);
    printf("   Loss types:\n");
    printf("     Minimax:   L_D = -E[log D(x)] - E[log(1-D(G(z)))]\n");
    printf("     Non-sat:   L_G = -E[log D(G(z))]  (stronger gradients)\n");
    printf("   Training...\n");
    for (int e = 0; e < epochs; e++) {
        float *batch = (float *)malloc(batch_size * data_dim * sizeof(float));
        for (int i = 0; i < batch_size * data_dim; i++)
            batch[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
        gan_train_step_d(gan, batch, batch_size, 0.0002f);
        gan_train_step_g(gan, batch_size, 0.0002f);
        printf("   Epoch %d: D step, G step complete.\n", e + 1);
        free(batch);
    }

    printf("\n5. WGAN-GP Gradient Penalty\n");
    float *real_b = (float *)malloc(batch_size * data_dim * sizeof(float));
    float *fake_b = (float *)malloc(batch_size * data_dim * sizeof(float));
    for (int i = 0; i < batch_size * data_dim; i++) {
        real_b[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
        fake_b[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    }
    float gp = wgan_gp_penalty(gan->disc, real_b, fake_b, batch_size, 10.0f);
    printf("   WGAN-GP penalty: %.6f (aim for ~0 with gradient penalty)\n", gp);
    printf("   WGAN uses Wasserstein distance, not JS divergence.\n");
    printf("   Gradient penalty enforces 1-Lipschitz constraint.\n");
    free(real_b); free(fake_b);

    printf("\n6. Mode Collapse Detection\n");
    int n_gen = 100;
    float *samples = (float *)malloc(n_gen * data_dim * sizeof(float));
    for (int i = 0; i < n_gen; i++) {
        for (int j = 0; j < noise_dim; j++)
            noise[j] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
        float *s = gen_forward(gan->gen, noise);
        memcpy(samples + i * data_dim, s, data_dim * sizeof(float));
        free(s);
    }
    float diversity = 0;
    mode_collapse_detect(samples, n_gen, data_dim, &diversity);
    printf("   Diversity score: %.4f (higher = more diverse samples)\n", diversity);
    printf("   Low diversity may indicate mode collapse.\n");
    free(samples);

    printf("\n7. DCGAN-style Architecture Note\n");
    printf("   DCGAN uses ConvTranspose2d (generator) and Conv2d (discriminator).\n");
    printf("   BatchNorm in generator, LeakyReLU in discriminator.\n");
    printf("   No pooling layers -- strided convolutions instead.\n");

    gan_free(gan);
    printf("\nDemo complete.\n");
    return 0;
}
