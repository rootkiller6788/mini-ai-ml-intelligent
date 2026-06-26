#include "diffusion_model.h"

void beta_schedule_linear(float *betas, int T, float start, float end) {
    for (int t = 0; t < T; t++)
        betas[t] = start + (end - start) * t / (T - 1);
}

void beta_schedule_cosine(float *betas, int T, float s) {
    for (int t = 0; t < T; t++) {
        float angle = (float)(t + 1) / T * 1.57079632679f;
        float ft = cosf(angle) / cosf(1.57079632679f * (float)t / T);
        betas[t] = 1.0f - ft * ft;
        if (betas[t] > 0.999f) betas[t] = 0.999f;
        betas[t] *= s;
    }
}

DiffusionSchedule *diff_schedule_create(int steps, float b_start, float b_end, int sched_type) {
    DiffusionSchedule *s = (DiffusionSchedule *)malloc(sizeof(DiffusionSchedule));
    s->num_timesteps = steps; s->beta_start = b_start; s->beta_end = b_end;
    s->schedule_type = sched_type;
    int T = steps;
    s->betas                   = (float *)malloc(T * sizeof(float));
    s->alphas                  = (float *)malloc(T * sizeof(float));
    s->alpha_bars              = (float *)malloc(T * sizeof(float));
    s->sqrt_alpha_bars         = (float *)malloc(T * sizeof(float));
    s->sqrt_one_minus_alpha_bars = (float *)malloc(T * sizeof(float));

    if (sched_type == 0) beta_schedule_linear(s->betas, T, b_start, b_end);
    else                 beta_schedule_cosine(s->betas, T, 0.008f);

    for (int t = 0; t < T; t++) {
        s->alphas[t] = 1.0f - s->betas[t];
        s->alpha_bars[t] = 1.0f;
        for (int i = 0; i <= t; i++)
            s->alpha_bars[t] *= s->alphas[i];
        s->sqrt_alpha_bars[t] = sqrtf(s->alpha_bars[t]);
        s->sqrt_one_minus_alpha_bars[t] = sqrtf(1.0f - s->alpha_bars[t]);
    }
    return s;
}

void diff_schedule_free(DiffusionSchedule *s) {
    free(s->betas); free(s->alphas); free(s->alpha_bars);
    free(s->sqrt_alpha_bars); free(s->sqrt_one_minus_alpha_bars);
    free(s);
}

void forward_diffuse(const DiffusionSchedule *s, const float *x0,
                     float *x_t, float *noise, int t, int dim) {
    float a_bar = s->alpha_bars[t];
    float sqrt_a_bar = sqrtf(a_bar);
    float sqrt_one_minus_a_bar = sqrtf(1.0f - a_bar);

    for (int i = 0; i < dim; i++) {
        noise[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f; /* approx N(0,1) scaled */
        noise[i] = noise[i] / 1.732f; /* normalize uniform to ~N(0,1) std */
        x_t[i] = sqrt_a_bar * x0[i] + sqrt_one_minus_a_bar * noise[i];
    }
}

UNet *unet_create(int in_c, int hid_c, int out_c, int t_emb_dim, int n_res) {
    UNet *u = (UNet *)malloc(sizeof(UNet));
    u->in_channels = in_c; u->hidden_channels = hid_c;
    u->out_channels = out_c; u->time_emb_dim = t_emb_dim;
    u->num_res_blocks = n_res;

    u->time_emb_w1 = (float *)malloc(t_emb_dim * hid_c * 4 * sizeof(float));
    u->time_emb_b1 = (float *)calloc(hid_c * 4, sizeof(float));
    u->time_emb_w2 = (float *)malloc(hid_c * 4 * hid_c * sizeof(float));
    u->time_emb_b2 = (float *)calloc(hid_c, sizeof(float));

    float scale = sqrtf(2.0f / t_emb_dim);
    for (int i = 0; i < t_emb_dim * hid_c * 4; i++)
        u->time_emb_w1[i] = ((float)rand() / RAND_MAX - 0.5f) * scale;
    for (int i = 0; i < hid_c * 4 * hid_c; i++)
        u->time_emb_w2[i] = ((float)rand() / RAND_MAX - 0.5f) * scale;

    u->conv_in_w = (float *)malloc(in_c * hid_c * 3 * 3 * sizeof(float));
    u->conv_in_b = (float *)calloc(hid_c, sizeof(float));
    for (int i = 0; i < in_c * hid_c * 9; i++)
        u->conv_in_w[i] = ((float)rand() / RAND_MAX - 0.5f) * sqrtf(2.0f / (in_c * 9));

    return u;
}

void unet_free(UNet *u) {
    free(u->time_emb_w1); free(u->time_emb_b1);
    free(u->time_emb_w2); free(u->time_emb_b2);
    free(u->conv_in_w); free(u->conv_in_b);
    free(u);
}

static float *time_embedding(int t, int dim) {
    float *emb = (float *)calloc(dim, sizeof(float));
    for (int i = 0; i < dim / 2; i++) {
        float angle = (float)t / powf(10000.0f, (2.0f * i) / dim);
        emb[2 * i] = sinf(angle);
        emb[2 * i + 1] = cosf(angle);
    }
    return emb;
}

float *unet_forward(const UNet *u, const float *x, int t_emb, int h, int w) {
    int c_out = u->out_channels;
    int total = c_out * h * w;
    float *out = (float *)calloc(total, sizeof(float));

    float *t_e = time_embedding(t_emb, u->time_emb_dim);
    float *t_mid = (float *)calloc(u->hidden_channels * 4, sizeof(float));
    for (int i = 0; i < u->hidden_channels * 4; i++)
        for (int j = 0; j < u->time_emb_dim; j++)
            t_mid[i] += u->time_emb_w1[i * u->time_emb_dim + j] * t_e[j];
    for (int i = 0; i < u->hidden_channels * 4; i++) {
        t_mid[i] += u->time_emb_b1[i];
        if (t_mid[i] < 0) t_mid[i] = 0;
    }

    float *t_out = (float *)calloc(u->hidden_channels, sizeof(float));
    for (int i = 0; i < u->hidden_channels; i++)
        for (int j = 0; j < u->hidden_channels * 4; j++)
            t_out[i] += u->time_emb_w2[i * u->hidden_channels * 4 + j] * t_mid[j];
    for (int i = 0; i < u->hidden_channels; i++)
        t_out[i] += u->time_emb_b2[i];

    for (int i = 0; i < c_out; i++)
        for (int y = 0; y < h; y++)
            for (int x_pos = 0; x_pos < w; x_pos++)
                out[i * h * w + y * w + x_pos] = x[i * h * w + y * w + x_pos] * 0.5f;

    free(t_e); free(t_mid); free(t_out);
    return out;
}

DiffusionModel *diff_model_create(int img_c, int img_sz, int steps, int sched, float guidance) {
    DiffusionModel *m = (DiffusionModel *)malloc(sizeof(DiffusionModel));
    m->image_channels = img_c; m->image_size = img_sz;
    m->guidance_scale = guidance;
    m->schedule = diff_schedule_create(steps, 0.0001f, 0.02f, sched);
    m->model = unet_create(img_c, 64, img_c, 128, 2);
    return m;
}

void diff_model_free(DiffusionModel *m) {
    diff_schedule_free(m->schedule);
    unet_free(m->model);
    free(m);
}

float *diff_train_step(const DiffusionModel *model, const float *x0, int bs, float lr) {
    int dim = model->image_channels * model->image_size * model->image_size;
    int total = bs * dim;
    float *x_t = (float *)malloc(total * sizeof(float));
    float *noise = (float *)malloc(total * sizeof(float));

    for (int i = 0; i < bs; i++) {
        int t = rand() % model->schedule->num_timesteps;
        forward_diffuse(model->schedule, x0 + i * dim, x_t + i * dim,
                        noise + i * dim, t, dim);
    }

    float *pred = (float *)calloc(total, sizeof(float));
    for (int i = 0; i < bs; i++) {
        float *p = unet_forward(model->model, x_t + i * dim, 0,
                                 model->image_size, model->image_size);
        memcpy(pred + i * dim, p, dim * sizeof(float));
        free(p);
    }

    float loss = 0;
    for (int i = 0; i < total; i++) {
        float diff = pred[i] - noise[i];
        loss += diff * diff;
    }
    loss /= total;

    float *grad = (float *)malloc(total * sizeof(float));
    for (int i = 0; i < total; i++)
        grad[i] = 2.0f * (pred[i] - noise[i]) / total;

    (void)lr; (void)loss;
    free(x_t); free(noise); free(pred);
    return grad;
}

float *ddpm_sample(const DiffusionModel *model, int batch_size, int seed) {
    int dim = model->image_channels * model->image_size * model->image_size;
    int total = batch_size * dim;
    float *x = (float *)malloc(total * sizeof(float));
    srand(seed);

    for (int i = 0; i < total; i++)
        x[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f / 1.732f;

    for (int t = model->schedule->num_timesteps - 1; t >= 0; t--) {
        float alpha = model->schedule->alphas[t];
        float alpha_bar = model->schedule->alpha_bars[t];
        float beta = model->schedule->betas[t];

        for (int b = 0; b < batch_size; b++) {
            float *eps = unet_forward(model->model, x + b * dim, t,
                                       model->image_size, model->image_size);
            for (int i = 0; i < dim; i++) {
                float noise_term = 0;
                if (t > 0) {
                    noise_term = sqrtf(beta) * ((float)rand() / RAND_MAX - 0.5f) * 2.0f / 1.732f;
                }
                x[b * dim + i] = (1.0f / sqrtf(alpha)) *
                    (x[b * dim + i] - (beta / sqrtf(1.0f - alpha_bar)) * eps[i]) +
                    noise_term;
            }
            free(eps);
        }
    }
    return x;
}

float *ddim_sample(const DiffusionModel *model, int steps, float eta, int seed) {
    int dim = model->image_channels * model->image_size * model->image_size;
    float *x = (float *)malloc(dim * sizeof(float));
    srand(seed);

    for (int i = 0; i < dim; i++)
        x[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f / 1.732f;

    int T = model->schedule->num_timesteps;
    int stride = T / steps;
    if (stride < 1) stride = 1;

    for (int s = steps - 1; s >= 0; s--) {
        int t = s * stride;
        int t_prev = (s > 0) ? (s - 1) * stride : 0;

        float alpha_bar = model->schedule->alpha_bars[t];
        float alpha_bar_prev = model->schedule->alpha_bars[t_prev];

        float *eps = unet_forward(model->model, x, t, model->image_size, model->image_size);

        float sigma = eta * sqrtf((1.0f - alpha_bar_prev) / (1.0f - alpha_bar)) *
                      sqrtf(1.0f - alpha_bar / alpha_bar_prev);

        for (int i = 0; i < dim; i++) {
            float pred_x0 = (x[i] - sqrtf(1.0f - alpha_bar) * eps[i]) / sqrtf(alpha_bar);
            float dir_xt = sqrtf(1.0f - alpha_bar_prev - sigma * sigma) * eps[i];
            float noise = 0;
            if (eta > 0 && s > 0)
                noise = sigma * ((float)rand() / RAND_MAX - 0.5f) * 2.0f / 1.732f;
            x[i] = sqrtf(alpha_bar_prev) * pred_x0 + dir_xt + noise;
        }
        free(eps);
    }
    return x;
}

void classifier_free_guidance(float *eps_uncond, float *eps_cond, float *out,
                               float w, int dim) {
    for (int i = 0; i < dim; i++)
        out[i] = eps_uncond[i] + w * (eps_cond[i] - eps_uncond[i]);
}
