#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "diffusion_model.h"

int main(void) {
    printf("=== mini-model-arch: Diffusion Model Demo ===\n\n");
    srand((unsigned)time(NULL));

    int image_c = 1, image_sz = 28, timesteps = 1000;
    float guidance = 3.0f;

    printf("1. Beta Schedules\n");
    float *beta_linear = (float *)malloc(timesteps * sizeof(float));
    float *beta_cosine = (float *)malloc(timesteps * sizeof(float));
    beta_schedule_linear(beta_linear, timesteps, 0.0001f, 0.02f);
    beta_schedule_cosine(beta_cosine, timesteps, 0.008f);
    printf("   Linear:  beta_1 = %.6f, beta_%d = %.6f\n",
           beta_linear[0], timesteps, beta_linear[timesteps - 1]);
    printf("   Cosine:  beta_1 = %.6f, beta_%d = %.6f\n",
           beta_cosine[0], timesteps, beta_cosine[timesteps - 1]);

    printf("\n2. Diffusion Schedule (alpha bars)\n");
    DiffusionSchedule *sched = diff_schedule_create(timesteps, 0.0001f, 0.02f, 0);
    printf("   alpha_bar[0] = %.6f (t=0: mostly original)\n", sched->alpha_bars[0]);
    printf("   alpha_bar[500] = %.6f\n", sched->alpha_bars[500]);
    printf("   alpha_bar[999] = %.6f (t=T: mostly noise)\n", sched->alpha_bars[999]);
    printf("   sqrt(alpha_bar[999]) = %.4f, sqrt(1-alpha_bar[999]) = %.4f\n",
           sched->sqrt_alpha_bars[999], sched->sqrt_one_minus_alpha_bars[999]);

    printf("\n3. Forward Diffusion Process\n");
    int dim = image_c * image_sz * image_sz;
    float *x0 = (float *)malloc(dim * sizeof(float));
    for (int i = 0; i < dim; i++)
        x0[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    float *x_t = (float *)malloc(dim * sizeof(float));
    float *eps  = (float *)malloc(dim * sizeof(float));

    printf("   x0: [%.4f, %.4f, %.4f, %.4f, ...]\n",
           x0[0], x0[1], x0[2], x0[3]);

    forward_diffuse(sched, x0, x_t, eps, 0, dim);
    printf("   t=0:   x_t ~ x0 (alpha_bar=%.4f)\n", sched->alpha_bars[0]);
    printf("     x_t: [%.4f, %.4f, %.4f, %.4f, ...]\n",
           x_t[0], x_t[1], x_t[2], x_t[3]);

    forward_diffuse(sched, x0, x_t, eps, 500, dim);
    printf("   t=500: partially noisy\n");
    printf("     x_t: [%.4f, %.4f, %.4f, %.4f, ...]\n",
           x_t[0], x_t[1], x_t[2], x_t[3]);

    forward_diffuse(sched, x0, x_t, eps, 999, dim);
    printf("   t=999: x_t ~ N(0,1) (alpha_bar ~ %.4f)\n", sched->alpha_bars[999]);
    printf("     x_t: [%.4f, %.4f, %.4f, %.4f, ...]\n",
           x_t[0], x_t[1], x_t[2], x_t[3]);

    printf("\n4. U-Net Denoising Model\n");
    UNet *unet = unet_create(image_c, 64, image_c, 128, 2);
    printf("   In: %d channels, Hidden: 64, Out: %d channels\n", image_c, image_c);
    printf("   Time embedding dim: 128, Res blocks: 2\n");
    float *denoised = unet_forward(unet, x_t, 500, image_sz, image_sz);
    printf("   U-Net output (t=500): [%.4f, %.4f, %.4f, %.4f, ...]\n",
           denoised[0], denoised[1], denoised[2], denoised[3]);
    free(denoised);

    printf("\n5. Training Step\n");
    DiffusionModel *model = diff_model_create(image_c, image_sz, timesteps, 0, guidance);
    printf("   Loss = MSE(eps_predicted - eps_true)\n");
    printf("   Random timestep t ~ Uniform(0, T-1)\n");
    float *grad = diff_train_step(model, x0, 4, 1e-4f);
    printf("   Gradient norm (sample): [%.6f, %.6f, %.6f, %.6f, ...]\n",
           grad[0], grad[1], grad[2], grad[3]);
    free(grad);

    printf("\n6. DDPM Sampling (reverse process)\n");
    printf("   Start from pure noise x_T, iterate T steps.\n");
    printf("   Formula: x_{t-1} = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1-alpha_bar_t) * eps_t) + sigma_t * z\n");
    int small_steps = 10;
    DiffusionModel *small_model = diff_model_create(image_c, image_sz, small_steps, 0, guidance);
    float *ddpm_out = ddpm_sample(small_model, 1, 42);
    printf("   DDPM sample (T=%d): [%.4f, %.4f, %.4f, %.4f, ...]\n",
           small_steps, ddpm_out[0], ddpm_out[1], ddpm_out[2], ddpm_out[3]);
    free(ddpm_out);
    diff_model_free(small_model);

    printf("\n7. DDIM Sampling (deterministic, fewer steps)\n");
    int ddim_steps = 50;
    DiffusionModel *ddim_model = diff_model_create(image_c, image_sz, timesteps, 0, guidance);
    printf("   DDIM uses %d steps instead of %d (eta=0 = deterministic).\n",
           ddim_steps, timesteps);
    printf("   Subsampling: stride = T/ddim_steps = %d\n", timesteps / ddim_steps);
    float *ddim_out = ddim_sample(ddim_model, ddim_steps, 0.0f, 42);
    printf("   DDIM sample: [%.4f, %.4f, %.4f, %.4f, ...]\n",
           ddim_out[0], ddim_out[1], ddim_out[2], ddim_out[3]);
    free(ddim_out);
    diff_model_free(ddim_model);

    printf("\n8. Classifier-Free Guidance\n");
    float *eps_u = (float *)malloc(dim * sizeof(float));
    float *eps_c = (float *)malloc(dim * sizeof(float));
    float *cfg_out = (float *)malloc(dim * sizeof(float));
    for (int i = 0; i < dim; i++) {
        eps_u[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        eps_c[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f + 0.1f;
    }
    classifier_free_guidance(eps_u, eps_c, cfg_out, 3.0f, dim);
    printf("   CFG(w=3.0): eps = eps_uncond + 3.0 * (eps_cond - eps_uncond)\n");
    printf("   Sample: [%.4f, %.4f, %.4f, %.4f]\n",
           cfg_out[0], cfg_out[1], cfg_out[2], cfg_out[3]);
    free(eps_u); free(eps_c); free(cfg_out);

    printf("\n9. Schedule Comparison Summary\n");
    printf("   Type     | beta_1     | beta_T     | Noise at t=T\n");
    printf("   Linear   | 0.000100    | 0.020000   | ~pure noise\n");
    printf("   Cosine   | %.6f | %.6f | ~pure noise\n", beta_cosine[0], beta_cosine[999]);
    printf("   Cosine schedule preserves information longer (more gradual).\n");

    free(beta_linear); free(beta_cosine);
    free(x0); free(x_t); free(eps);
    diff_schedule_free(sched);
    unet_free(unet);
    diff_model_free(model);

    printf("\nDemo complete.\n");
    return 0;
}
