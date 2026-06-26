#ifndef IMAGE_GENERATION_H
#define IMAGE_GENERATION_H

#include <stddef.h>
#include <math.h>
#include "clip_contrastive.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MM_SD_IMAGE_SIZE      512
#define MM_SD_LATENT_SIZE     64
#define MM_SD_LATENT_CHANNELS 4
#define MM_SD_MAX_TEXT_LEN    77
#define MM_SD_CROSS_ATTN_DIM  768
#define MM_SD_NUM_TIMESTEPS   1000
#define MM_SD_BLOCK_CHANNELS  320
#define MM_UNET_CH_MULT       {1, 2, 4, 4}

typedef struct {
    float* weight;
    float* bias;
    int    in_ch;
    int    out_ch;
    int    kernel_size;
    int    stride;
    int    padding;
    int    groups;
    int    use_bias;
} mm_conv2d_t;

typedef struct {
    float* weight;
    float* bias;
    int    num_groups;
    int    num_ch;
    float  eps;
} mm_groupnorm_t;

typedef struct {
    mm_conv2d_t   conv1, conv2;
    mm_groupnorm_t norm1, norm2;
    float*        time_embed_weight;
    float*        time_embed_bias;
    int           in_ch, out_ch;
    int           use_time_embed;
} mm_resblock_t;

typedef struct {
    mm_linear_t  q_proj, k_proj, v_proj, out_proj;
    int          num_heads;
    int          head_dim;
    float        scale;
} mm_cross_attn_t;

typedef struct {
    mm_groupnorm_t  norm;
    mm_conv2d_t     proj_in;
    mm_cross_attn_t attn;
    mm_conv2d_t     proj_out;
    int             dim;
    int             context_dim;
} mm_spatial_transformer_t;

typedef struct {
    mm_resblock_t*           res_blocks;
    mm_spatial_transformer_t* attn_blocks;
    mm_conv2d_t              downsampler;
    int                      num_res_blocks;
    int                      num_attn_blocks;
    int                      ch_in, ch_out;
    int                      has_downsampler;
} mm_unet_down_block_t;

typedef struct {
    mm_resblock_t*           res_blocks;
    mm_spatial_transformer_t* attn_blocks;
    int                      num_res_blocks;
    int                      num_attn_blocks;
    int                      ch_in, ch_out;
} mm_unet_mid_block_t;

typedef struct {
    mm_resblock_t*           res_blocks;
    mm_spatial_transformer_t* attn_blocks;
    mm_conv2d_t              upsampler;
    int                      num_res_blocks;
    int                      num_attn_blocks;
    int                      ch_in, ch_out;
    int                      has_upsampler;
} mm_unet_up_block_t;

typedef struct {
    mm_conv2d_t             conv_in;
    mm_unet_down_block_t*   down_blocks;
    mm_unet_mid_block_t     mid_block;
    mm_unet_up_block_t*     up_blocks;
    mm_groupnorm_t          norm_out;
    mm_conv2d_t             conv_out;
    int                     num_down;
    int                     num_up;
    int                     base_channels;
    int                     num_res_blocks;
    float*                  time_embed[2];
    int                     time_embed_dim;
} mm_unet_t;

typedef struct {
    mm_conv2d_t    encoder_conv_in;
    mm_resblock_t* encoder_res_blocks;
    mm_conv2d_t    encoder_conv_out_mu;
    mm_conv2d_t    encoder_conv_out_logvar;
    mm_conv2d_t    decoder_conv_in;
    mm_resblock_t* decoder_res_blocks;
    mm_conv2d_t    decoder_conv_out;
    int            latent_dim;
    int            num_res_blocks;
    int            base_channels;
} mm_vae_t;

typedef struct {
    float* data;
    int    n;
    int    allocated;
} mm_schedule_t;

typedef struct {
    mm_vae_t   vae;
    mm_unet_t  unet;
    int        image_size;
    int        latent_size;
    int        latent_channels;
    int        num_timesteps;
    float      guidance_scale;
    float      beta_start;
    float      beta_end;
    int        schedule_type;
} mm_stable_diffusion_t;

typedef enum {
    MM_SD_SCHEDULE_LINEAR = 0,
    MM_SD_SCHEDULE_COSINE = 1,
    MM_SD_SCHEDULE_SQRT   = 2,
} mm_schedule_type_t;

typedef enum {
    MM_SD_SAMPLER_DDIM      = 0,
    MM_SD_SAMPLER_DPM_PP_2M = 1,
    MM_SD_SAMPLER_EULER_A   = 2,
} mm_sd_sampler_t;

void mm_conv2d_init(mm_conv2d_t* c, int in_ch, int out_ch, int k, int s, int p, int g);
void mm_conv2d_free(mm_conv2d_t* c);
void mm_conv2d_forward(const mm_conv2d_t* c, const float* x,
                       int h, int w, float* y);

void mm_groupnorm_init(mm_groupnorm_t* gn, int num_groups, int num_ch, float eps);
void mm_groupnorm_free(mm_groupnorm_t* gn);
void mm_groupnorm_forward(const mm_groupnorm_t* gn, const float* x,
                          int h, int w, float* y);

void mm_resblock_init(mm_resblock_t* rb, int in_ch, int out_ch, int time_embed_dim);
void mm_resblock_free(mm_resblock_t* rb);
void mm_resblock_forward(const mm_resblock_t* rb, const float* x,
                         const float* temb, int h, int w, float* y);

void mm_cross_attn_init(mm_cross_attn_t* attn, int query_dim, int context_dim, int num_heads);
void mm_cross_attn_free(mm_cross_attn_t* attn);
void mm_cross_attn_forward(const mm_cross_attn_t* attn, const float* x,
                           const float* context, int seq_len, int ctx_len, float* out);

void mm_spatial_transformer_init(mm_spatial_transformer_t* st, int dim, int context_dim, int num_heads);
void mm_spatial_transformer_free(mm_spatial_transformer_t* st);
void mm_spatial_transformer_forward(const mm_spatial_transformer_t* st, const float* x,
                                    const float* context, int h, int w, int ctx_len, float* y);

void mm_unet_init(mm_unet_t* unet, int base_channels, int num_res_blocks);
void mm_unet_free(mm_unet_t* unet);
void mm_unet_forward(const mm_unet_t* unet, const float* x, const float* timestep,
                     const float* context, int h, int w, int ctx_len, float* y);

void mm_vae_init(mm_vae_t* vae, int latent_dim, int base_channels, int num_res_blocks);
void mm_vae_free(mm_vae_t* vae);
void mm_vae_encode(const mm_vae_t* vae, const float* image, int h, int w, int c,
                   float* mean, float* logvar);
void mm_vae_decode(const mm_vae_t* vae, const float* latent, float* image,
                   int h, int w, int c);
void mm_vae_sample(const float* mean, const float* logvar, int n, float* latent);

void mm_sd_init(mm_stable_diffusion_t* sd, int image_size, int latent_dim,
                int base_channels, int num_res_blocks);
void mm_sd_free(mm_stable_diffusion_t* sd);

void mm_sd_beta_schedule(mm_schedule_t* schedule, int num_timesteps,
                         float beta_start, float beta_end, mm_schedule_type_t type);
void mm_sd_alphas_from_betas(const mm_schedule_t* betas, mm_schedule_t* alphas,
                             mm_schedule_t* alpha_cumprod);

void mm_sd_add_noise(const float* x, const float* noise, float sqrt_alpha_cumprod,
                     float sqrt_one_minus_alpha_cumprod, int n, float* out);

void mm_sd_predict_noise(const mm_stable_diffusion_t* sd, const float* latent,
                         int timestep, const float* text_embedding, int ctx_len,
                         float* noise_pred_uncond, float* noise_pred_text);

void mm_sd_ddim_step(const float* x_t, const float* noise_pred,
                     int t, int t_prev, const mm_schedule_t* alpha_cumprod,
                     int n, float eta, float* x_t_prev);

void mm_sd_dpm_pp_2m_step(const float* x_t, const float* noise_pred,
                          const float* noise_pred_prev,
                          int t, int t_prev, int t_prev2,
                          const mm_schedule_t* alpha_cumprod,
                          int n, float* x_t_prev);

void mm_sd_generate(const mm_stable_diffusion_t* sd, const float* text_embedding,
                    int ctx_len, int num_steps, mm_sd_sampler_t sampler,
                    int h, int w, int c, float* image);

void mm_sd_inpaint(const mm_stable_diffusion_t* sd, const float* image,
                   const float* mask, const float* text_embedding,
                   int ctx_len, int h, int w, int c, int num_steps, float* out);

void mm_sd_pipe_encode(const mm_stable_diffusion_t* sd, const float* image,
                       int h, int w, int c, float* latent);
void mm_sd_pipe_decode(const mm_stable_diffusion_t* sd, const float* latent,
                       int latent_h, int latent_w, int h, int w, int c, float* image);

void mm_sd_cfg_guidance(float* noise_pred_cond, const float* noise_pred_uncond,
                        float guidance_scale, int n);

void mm_schedule_init(mm_schedule_t* s, int n);
void mm_schedule_free(mm_schedule_t* s);

void mm_sinusoidal_embedding(int t, int dim, float* emb);

#ifdef __cplusplus
}
#endif

#endif
