#include "image_generation.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#define SD_IMG_SIZE   256
#define SD_STEPS      20
#define SD_GUIDANCE   7.5f

static void print_image_stats(const float* image, int h, int w, int c, const char* label) {
    int total = h * w * c;
    float min_v = 1e9f, max_v = -1e9f, mean = 0.0f;
    for (int i = 0; i < total; i++) {
        if (image[i] < min_v) min_v = image[i];
        if (image[i] > max_v) max_v = image[i];
        mean += image[i];
    }
    mean /= (float)total;
    printf("%s: min=%.3f, max=%.3f, mean=%.3f\n", label, min_v, max_v, mean);
}

int main(void) {
    srand((unsigned)time(NULL));

    printf("=== Stable Diffusion Image Generation Demo ===\n\n");

    printf("--- VAE (Variational Autoencoder) ---\n");
    int vae_blocks = 2;
    mm_vae_t vae;
    mm_vae_init(&vae, 4, 64, vae_blocks);
    printf("VAE initialized: latent_dim=%d, base_channels=%d, res_blocks=%d\n\n", 4, 64, vae_blocks);

    /* VAE with N res_blocks does (N-1) 2x downsamplings */
    int vae_scale = 1 << (vae_blocks - 1);  /* 2^(N-1) */
    int img_h = 128, img_w = 128, img_c = 3;
    int latent_h = img_h / vae_scale, latent_w = img_w / vae_scale;
    int img_n = img_h * img_w * img_c;
    int latent_n = latent_h * latent_w * 4;

    float* test_image = (float*)malloc((size_t)img_n * sizeof(float));
    for (int i = 0; i < img_n; i++) test_image[i] = (float)rand() / (float)RAND_MAX;

    float* mu = (float*)calloc((size_t)latent_n, sizeof(float));
    float* logvar = (float*)calloc((size_t)latent_n, sizeof(float));
    mm_vae_encode(&vae, test_image, img_h, img_w, img_c, mu, logvar);
    printf("Image encoded to latent space (shape: %d x %d x 4 = %d)\n", latent_h, latent_w, latent_n);

    float* sampled = (float*)malloc((size_t)latent_n * sizeof(float));
    mm_vae_sample(mu, logvar, latent_n, sampled);

    float* reconstructed = (float*)malloc((size_t)img_n * sizeof(float));
    mm_vae_decode(&vae, sampled, reconstructed, img_h, img_w, img_c);
    print_image_stats(reconstructed, img_h, img_w, img_c, "Decoded image");

    printf("\n--- Sinusoidal Time Embedding ---\n");
    float t_emb[256] = {0};
    mm_sinusoidal_embedding(500, 256, t_emb);
    printf("t=500: [");
    for (int i = 0; i < 8; i++) printf("%.3f%s", t_emb[i], i < 7 ? ", " : "");
    printf(" ...]\n\n");

    printf("--- UNet Denoiser ---\n");
    mm_unet_t unet;
    mm_unet_init(&unet, 64, 2);
    printf("UNet initialized: base_channels=%d, down=%d, up=%d\n\n", 64, unet.num_down, unet.num_up);

    int ls = 32;
    float* test_latent = (float*)malloc((size_t)ls * ls * 4 * sizeof(float));
    for (int i = 0; i < ls * ls * 4; i++) test_latent[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 2.0f;

    float t_arr[1] = {0.5f};
    float* unet_out = (float*)malloc((size_t)ls * ls * 4 * sizeof(float));
    float context[768] = {0};
    for (int i = 0; i < 768; i++) context[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 0.02f;

    mm_unet_forward(&unet, test_latent, t_arr, context, ls, ls, 768 / 768, unet_out);
    print_image_stats(unet_out, ls, ls, 4, "UNet output (latent)");

    printf("\n--- Beta Schedule ---\n");
    mm_schedule_t betas, alphas, alpha_cp;
    mm_sd_beta_schedule(&betas, 1000, 0.00085f, 0.012f, MM_SD_SCHEDULE_COSINE);
    mm_sd_alphas_from_betas(&betas, &alphas, &alpha_cp);

    printf("Beta schedule (1000 steps, cosine):\n");
    printf("  beta[0]=%.6f, beta[500]=%.6f, beta[999]=%.6f\n", betas.data[0], betas.data[500], betas.data[999]);
    printf("  alpha_cumprod[0]=%.3f, alpha_cumprod[500]=%.3f, alpha_cumprod[999]=%.6f\n\n",
           alpha_cp.data[0], alpha_cp.data[500], alpha_cp.data[999]);

    printf("--- DDIM Sampling Step ---\n");
    float* x_t = (float*)malloc((size_t)ls * ls * 4 * sizeof(float));
    float* noise_pred = (float*)malloc((size_t)ls * ls * 4 * sizeof(float));
    float* x_t_prev = (float*)malloc((size_t)ls * ls * 4 * sizeof(float));
    for (int i = 0; i < ls * ls * 4; i++) {
        x_t[i] = ((float)rand() / (float)RAND_MAX - 0.5f);
        noise_pred[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 0.1f;
    }
    mm_sd_ddim_step(x_t, noise_pred, 500, 480, &alpha_cp, ls * ls * 4, 0.0f, x_t_prev);
    print_image_stats(x_t_prev, ls, ls, 4, "DDIM step output");

    printf("\n--- DPM-Solver++ 2M Step ---\n");
    float* noise_prev = (float*)malloc((size_t)ls * ls * 4 * sizeof(float));
    for (int i = 0; i < ls * ls * 4; i++) noise_prev[i] = noise_pred[i] * 0.9f;
    mm_sd_dpm_pp_2m_step(x_t, noise_pred, noise_prev, 500, 480, 470, &alpha_cp, ls * ls * 4, x_t_prev);
    print_image_stats(x_t_prev, ls, ls, 4, "DPM++ 2M step output");

    printf("\n--- CFG (Classifier-Free Guidance) ---\n");
    float* noise_cond = (float*)malloc((size_t)ls * ls * 4 * sizeof(float));
    float* noise_uncond = (float*)malloc((size_t)ls * ls * 4 * sizeof(float));
    for (int i = 0; i < ls * ls * 4; i++) {
        noise_cond[i] = noise_pred[i];
        noise_uncond[i] = noise_pred[i] * 0.5f;
    }
    mm_sd_cfg_guidance(noise_cond, noise_uncond, SD_GUIDANCE, ls * ls * 4);
    print_image_stats(noise_cond, ls, ls, 4, "CFG result (scale=7.5)");

    printf("\n--- Full Stable Diffusion Pipeline ---\n");
    int sd_blocks = 4;  /* 4 blocks → 3 downsamples → 8x reduction */
    mm_stable_diffusion_t sd;
    mm_sd_init(&sd, SD_IMG_SIZE, 4, 32, sd_blocks);
    printf("SD initialized: image=%d, latent=%d, timesteps=%d\n\n", sd.image_size, sd.latent_size, sd.num_timesteps);

    float* generated = (float*)malloc((size_t)SD_IMG_SIZE * SD_IMG_SIZE * 3 * sizeof(float));
    float* placeholder_ctx = (float*)calloc((size_t)768, sizeof(float));
    for (int i = 0; i < 768; i++) placeholder_ctx[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 0.02f;

    mm_sd_generate(&sd, placeholder_ctx, 1, SD_STEPS, MM_SD_SAMPLER_DDIM,
                   SD_IMG_SIZE, SD_IMG_SIZE, 3, generated);
    print_image_stats(generated, SD_IMG_SIZE, SD_IMG_SIZE, 3, "Generated image");

    printf("\n--- Inpainting ---\n");
    float* mask = (float*)malloc((size_t)SD_IMG_SIZE * SD_IMG_SIZE * 3 * sizeof(float));
    for (int i = 0; i < SD_IMG_SIZE * SD_IMG_SIZE * 3; i++) {
        mask[i] = ((i / 3) % (SD_IMG_SIZE * SD_IMG_SIZE) < SD_IMG_SIZE * SD_IMG_SIZE / 4) ? 1.0f : 0.0f;
    }

    float* inpainted = (float*)malloc((size_t)SD_IMG_SIZE * SD_IMG_SIZE * 3 * sizeof(float));
    mm_sd_inpaint(&sd, generated, mask, placeholder_ctx, 1,
                  SD_IMG_SIZE, SD_IMG_SIZE, 3, SD_STEPS / 2, inpainted);
    print_image_stats(inpainted, SD_IMG_SIZE, SD_IMG_SIZE, 3, "Inpainted image");

    printf("\n=== Demo Complete ===\n");

    free(test_image); free(mu); free(logvar); free(sampled); free(reconstructed);
    free(test_latent); free(unet_out); free(x_t); free(noise_pred); free(x_t_prev);
    free(noise_prev); free(noise_cond); free(noise_uncond);
    free(generated); free(placeholder_ctx); free(mask); free(inpainted);

    mm_vae_free(&vae);
    mm_unet_free(&unet);
    mm_sd_free(&sd);
    mm_schedule_free(&betas);
    mm_schedule_free(&alphas);
    mm_schedule_free(&alpha_cp);

    return 0;
}
