#ifndef GAN_MODEL_H
#define GAN_MODEL_H

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    int    input_dim, output_dim;
    int    *hidden_dims;
    int    num_layers;
    float **weights;
    float **biases;
    int     use_batch_norm;
} Generator;

typedef struct {
    int    input_dim;
    int    *hidden_dims;
    int    num_layers;
    float **weights;
    float **biases;
} Discriminator;

typedef struct {
    Generator     *gen;
    Discriminator *disc;
    int            noise_dim, data_dim;
    int            loss_type;
    float          grad_penalty_lambda;
} GAN;

Generator *gen_create(int noise_dim, int *hidden, int n_hid, int out_dim, int bn);
void       gen_free(Generator *g);
float     *gen_forward(const Generator *g, const float *noise);

Discriminator *disc_create(int in_dim, int *hidden, int n_hid);
void           disc_free(Discriminator *d);
float          disc_forward(const Discriminator *d, const float *x);

GAN *gan_create(int noise_dim, int data_dim, int *gen_hid, int n_gen,
                int *disc_hid, int n_disc, int loss_type);
void gan_free(GAN *gan);

void gan_train_step_d(GAN *gan, const float *real_batch, int bs, float lr);
void gan_train_step_g(GAN *gan, int bs, float lr);

typedef struct {
    float *gen_avg, *gen_std;
    float *disc_avg, *disc_std;
} DCGANNormalizer;

float *dcgan_gen_forward(const float *noise, int nz, int nc, int ngf);
float *dcgan_disc_forward(const float *x, int nc, int ndf);

float wgan_gp_penalty(const Discriminator *d, const float *real,
                       const float *fake, int batch_size, float lambda);

void mode_collapse_detect(const float *samples, int n_samples, int dim, float *diversity);

#endif
