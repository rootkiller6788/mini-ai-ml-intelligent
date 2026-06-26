#ifndef DIFFUSION_MODEL_H
#define DIFFUSION_MODEL_H

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define MAX_TIMESTEPS 1000

typedef struct {
    int    num_timesteps;
    float  beta_start, beta_end;
    float *betas;
    float *alphas;
    float *alpha_bars;
    float *sqrt_alpha_bars;
    float *sqrt_one_minus_alpha_bars;
    int    schedule_type;
} DiffusionSchedule;

typedef struct {
    int    in_channels, hidden_channels, out_channels;
    int    time_emb_dim;
    int    num_res_blocks;
    float *time_emb_w1, *time_emb_b1, *time_emb_w2, *time_emb_b2;
    float *conv_in_w, *conv_in_b;
} UNet;

typedef struct {
    DiffusionSchedule *schedule;
    UNet              *model;
    int                image_channels, image_size;
    float              guidance_scale;
} DiffusionModel;

DiffusionSchedule *diff_schedule_create(int steps, float b_start, float b_end, int sched_type);
void               diff_schedule_free(DiffusionSchedule *s);

void forward_diffuse(const DiffusionSchedule *s, const float *x0,
                     float *x_t, float *noise, int t, int dim);

float *ddpm_sample(const DiffusionModel *model, int batch_size, int seed);
float *ddim_sample(const DiffusionModel *model, int steps, float eta, int seed);

UNet *unet_create(int in_c, int hid_c, int out_c, int t_emb_dim, int n_res);
void  unet_free(UNet *u);
float *unet_forward(const UNet *u, const float *x, int t_emb, int h, int w);

DiffusionModel *diff_model_create(int img_c, int img_sz, int steps, int sched, float guidance);
void            diff_model_free(DiffusionModel *m);

float *diff_train_step(const DiffusionModel *model, const float *x0, int bs, float lr);

void classifier_free_guidance(float *eps_uncond, float *eps_cond, float *out,
                              float w, int dim);

void beta_schedule_linear(float *betas, int T, float start, float end);
void beta_schedule_cosine(float *betas, int T, float s);

#endif
