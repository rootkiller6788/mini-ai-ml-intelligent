#include "gan_model.h"

Generator *gen_create(int noise_dim, int *hidden, int n_hid, int out_dim, int bn) {
    Generator *g = (Generator *)malloc(sizeof(Generator));
    g->input_dim = noise_dim; g->output_dim = out_dim;
    g->num_layers = n_hid + 1; g->use_batch_norm = bn;
    g->weights = (float **)malloc(g->num_layers * sizeof(float *));
    g->biases  = (float **)malloc(g->num_layers * sizeof(float *));
    int prev = noise_dim;
    for (int i = 0; i < n_hid; i++) {
        int cur = hidden[i];
        g->weights[i] = (float *)malloc(prev * cur * sizeof(float));
        g->biases[i]  = (float *)calloc(cur, sizeof(float));
        float scale = sqrtf(2.0f / prev);
        for (int j = 0; j < prev * cur; j++)
            g->weights[i][j] = ((float)rand() / RAND_MAX - 0.5f) * scale;
        prev = cur;
    }
    g->weights[n_hid] = (float *)malloc(prev * out_dim * sizeof(float));
    g->biases[n_hid]  = (float *)calloc(out_dim, sizeof(float));
    float scale = sqrtf(2.0f / prev);
    for (int j = 0; j < prev * out_dim; j++)
        g->weights[n_hid][j] = ((float)rand() / RAND_MAX - 0.5f) * scale;
    return g;
}
void gen_free(Generator *g) {
    for (int i = 0; i < g->num_layers; i++) { free(g->weights[i]); free(g->biases[i]); }
    free(g->weights); free(g->biases); free(g);
}

float *gen_forward(const Generator *g, const float *noise) {
    int prev_dim = g->input_dim;
    float *cur = (float *)malloc(prev_dim * sizeof(float));
    memcpy(cur, noise, prev_dim * sizeof(float));

    for (int l = 0; l < g->num_layers - 1; l++) {
        int cur_dim = (l < g->num_layers - 2) ?
            g->biases[l + 1] ? 0 : g->output_dim : g->output_dim;
        int next_dim = 0;
        float *next = NULL;
        if (l == g->num_layers - 2) {
            next_dim = g->output_dim;
            next = (float *)calloc(next_dim, sizeof(float));
            for (int i = 0; i < next_dim; i++) {
                for (int j = 0; j < prev_dim; j++)
                    next[i] += g->weights[l][i * prev_dim + j] * cur[j];
                next[i] += g->biases[l][i];
                next[i] = tanhf(next[i]);
            }
        } else {
            float *w = g->weights[l];
            float *b = g->biases[l];
            int nd = (int)(g->biases[l] ? (size_t)1 : 0);
            next_dim = prev_dim;
            next = (float *)calloc(next_dim, sizeof(float));
            for (int i = 0; i < next_dim; i++) {
                for (int j = 0; j < prev_dim; j++)
                    next[i] += w[i * prev_dim + j] * cur[j];
                if (b) next[i] += b[i];
                if (next[i] < 0) next[i] = 0.01f * next[i];
            }
        }
        free(cur);
        cur = next;
        prev_dim = next_dim;
    }
    int last = g->num_layers - 1;
    float *out = (float *)calloc(g->output_dim, sizeof(float));
    for (int i = 0; i < g->output_dim; i++) {
        for (int j = 0; j < prev_dim; j++)
            out[i] += g->weights[last][i * prev_dim + j] * cur[j];
        out[i] += g->biases[last][i];
        out[i] = tanhf(out[i]);
    }
    free(cur);
    return out;
}

Discriminator *disc_create(int in_dim, int *hidden, int n_hid) {
    Discriminator *d = (Discriminator *)malloc(sizeof(Discriminator));
    d->input_dim = in_dim; d->num_layers = n_hid + 1;
    d->weights = (float **)malloc(d->num_layers * sizeof(float *));
    d->biases  = (float **)malloc(d->num_layers * sizeof(float *));
    int prev = in_dim;
    for (int i = 0; i < n_hid; i++) {
        int cur = hidden[i];
        d->weights[i] = (float *)malloc(prev * cur * sizeof(float));
        d->biases[i]  = (float *)calloc(cur, sizeof(float));
        float scale = sqrtf(2.0f / prev);
        for (int j = 0; j < prev * cur; j++)
            d->weights[i][j] = ((float)rand() / RAND_MAX - 0.5f) * scale;
        prev = cur;
    }
    d->weights[n_hid] = (float *)malloc(prev * 1 * sizeof(float));
    d->biases[n_hid]  = (float *)calloc(1, sizeof(float));
    for (int j = 0; j < prev; j++)
        d->weights[n_hid][j] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f;
    return d;
}
void disc_free(Discriminator *d) {
    for (int i = 0; i < d->num_layers; i++) { free(d->weights[i]); free(d->biases[i]); }
    free(d->weights); free(d->biases); free(d);
}
float disc_forward(const Discriminator *d, const float *x) {
    int prev = d->input_dim;
    float *cur = (float *)malloc(prev * sizeof(float));
    memcpy(cur, x, prev * sizeof(float));

    for (int l = 0; l < d->num_layers - 1; l++) {
        int next_dim = (int)(d->weights[l + 1] ? (size_t)d->input_dim : 1);
        float *next = NULL;
        if (l == 0) next_dim = (int)(d->biases[0] ? d->input_dim / 2 : prev);
        next = (float *)calloc(prev, sizeof(float));
        float *next_out = (float *)calloc(prev, sizeof(float));
        for (int i = 0; i < prev; i++)
            for (int j = 0; j < prev; j++)
                next_out[i] += d->weights[l][i * prev + j] * cur[j];
        for (int i = 0; i < prev; i++) {
            next_out[i] += d->biases[l][i];
            if (next_out[i] < 0) next_out[i] = 0.2f * next_out[i];
        }
        memcpy(next, next_out, prev * sizeof(float));
        free(next_out); free(cur);
        cur = next;
    }
    int last = d->num_layers - 1;
    float score = d->biases[last][0];
    for (int j = 0; j < prev; j++)
        score += d->weights[last][j] * cur[j];
    free(cur);
    return score;
}

GAN *gan_create(int noise_dim, int data_dim, int *gen_hid, int n_gen,
                int *disc_hid, int n_disc, int loss_type) {
    GAN *g = (GAN *)malloc(sizeof(GAN));
    g->noise_dim = noise_dim; g->data_dim = data_dim; g->loss_type = loss_type;
    g->grad_penalty_lambda = 10.0f;
    g->gen  = gen_create(noise_dim, gen_hid, n_gen, data_dim, 1);
    g->disc = disc_create(data_dim, disc_hid, n_disc);
    return g;
}
void gan_free(GAN *gan) { gen_free(gan->gen); disc_free(gan->disc); free(gan); }

void gan_train_step_d(GAN *gan, const float *real_batch, int bs, float lr) {
    float *noise = (float *)malloc(bs * gan->noise_dim * sizeof(float));
    for (int i = 0; i < bs * gan->noise_dim; i++)
        noise[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    float *fake = (float *)malloc(bs * gan->data_dim * sizeof(float));
    for (int i = 0; i < bs; i++) {
        float *sample = gen_forward(gan->gen, noise + i * gan->noise_dim);
        memcpy(fake + i * gan->data_dim, sample, gan->data_dim * sizeof(float));
        free(sample);
    }
    float d_loss = 0;
    for (int i = 0; i < bs; i++) {
        float real_score = disc_forward(gan->disc, real_batch + i * gan->data_dim);
        float fake_score = disc_forward(gan->disc, fake + i * gan->data_dim);
        d_loss += -logf(1.0f / (1.0f + expf(-real_score))) - logf(1.0f - 1.0f / (1.0f + expf(-fake_score)));
    }
    d_loss /= bs;
    free(noise); free(fake);
}

void gan_train_step_g(GAN *gan, int bs, float lr) {
    float *noise = (float *)malloc(bs * gan->noise_dim * sizeof(float));
    for (int i = 0; i < bs * gan->noise_dim; i++)
        noise[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    float g_loss = 0;
    for (int i = 0; i < bs; i++) {
        float *sample = gen_forward(gan->gen, noise + i * gan->noise_dim);
        float score = disc_forward(gan->disc, sample);
        g_loss += -logf(1.0f / (1.0f + expf(-score)));
        free(sample);
    }
    g_loss /= bs;
    free(noise);
}

float wgan_gp_penalty(const Discriminator *d, const float *real,
                       const float *fake, int batch_size, float lambda) {
    float penalty = 0;
    for (int i = 0; i < batch_size; i++) {
        float real_s = disc_forward(d, real + i * d->input_dim);
        float fake_s = disc_forward(d, fake + i * d->input_dim);
        float diff = real_s - fake_s;
        penalty += (diff * diff - 1.0f) * (diff * diff - 1.0f);
    }
    return lambda * penalty / batch_size;
}

void mode_collapse_detect(const float *samples, int n_samples, int dim, float *diversity) {
    float *mean = (float *)calloc(dim, sizeof(float));
    for (int i = 0; i < n_samples; i++)
        for (int j = 0; j < dim; j++)
            mean[j] += samples[i * dim + j];
    for (int j = 0; j < dim; j++) mean[j] /= n_samples;

    *diversity = 0;
    for (int i = 0; i < n_samples; i++) {
        float dist = 0;
        for (int j = 0; j < dim; j++) {
            float d = samples[i * dim + j] - mean[j];
            dist += d * d;
        }
        *diversity += sqrtf(dist);
    }
    *diversity /= n_samples;
    free(mean);
}

float *dcgan_gen_forward(const float *noise, int nz, int nc, int ngf) {
    (void)nc; (void)ngf;
    float *out = (float *)malloc(nz * sizeof(float));
    for (int i = 0; i < nz; i++)
        out[i] = tanhf(noise[i]);
    return out;
}

float *dcgan_disc_forward(const float *x, int nc, int ndf) {
    (void)nc; (void)ndf;
    return NULL;
}
