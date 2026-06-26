#include "image_generation.h"
#include "clip_contrastive.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

void mm_conv2d_init(mm_conv2d_t* c, int in_ch, int out_ch, int k, int s, int p, int g) {
    c->in_ch = in_ch;
    c->out_ch = out_ch;
    c->kernel_size = k;
    c->stride = s;
    c->padding = p;
    c->groups = g;
    c->use_bias = 1;
    int kw_size = out_ch * in_ch / g * k * k;
    c->weight = (float*)calloc((size_t)kw_size, sizeof(float));
    c->bias = (float*)calloc((size_t)out_ch, sizeof(float));
    float scale = sqrtf(2.0f / (float)(in_ch * k * k / g));
    for (int i = 0; i < kw_size; i++) {
        c->weight[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale;
    }
}

void mm_conv2d_free(mm_conv2d_t* c) { free(c->weight); free(c->bias); }

void mm_conv2d_forward(const mm_conv2d_t* c, const float* x, int h, int w, float* y) {
    int oh = (h + 2 * c->padding - c->kernel_size) / c->stride + 1;
    int ow = (w + 2 * c->padding - c->kernel_size) / c->stride + 1;
    int k = c->kernel_size;
    int in_ch = c->in_ch;
    int out_ch = c->out_ch;
    int g = c->groups;
    int ich_per_g = in_ch / g;
    int och_per_g = out_ch / g;

    for (int oy = 0; oy < oh; oy++) {
        for (int ox = 0; ox < ow; ox++) {
            for (int grp = 0; grp < g; grp++) {
                for (int oc = 0; oc < och_per_g; oc++) {
                    int occ = grp * och_per_g + oc;
                    float sum = c->bias[occ];
                    for (int ic = 0; ic < ich_per_g; ic++) {
                        int icc = grp * ich_per_g + ic;
                        for (int ky = 0; ky < k; ky++) {
                            for (int kx = 0; kx < k; kx++) {
                                int iy = oy * c->stride + ky - c->padding;
                                int ix = ox * c->stride + kx - c->padding;
                                if (iy >= 0 && iy < h && ix >= 0 && ix < w) {
                                    int w_idx = ((occ * ich_per_g + ic) * k + ky) * k + kx;
                                    sum += x[(iy * w + ix) * in_ch + icc] * c->weight[w_idx];
                                }
                            }
                        }
                    }
                    y[(oy * ow + ox) * out_ch + occ] = sum;
                }
            }
        }
    }
}

void mm_groupnorm_init(mm_groupnorm_t* gn, int num_groups, int num_ch, float eps) {
    gn->num_groups = num_groups;
    gn->num_ch = num_ch;
    gn->eps = eps;
    gn->weight = (float*)calloc((size_t)num_ch, sizeof(float));
    gn->bias = (float*)calloc((size_t)num_ch, sizeof(float));
    for (int i = 0; i < num_ch; i++) gn->weight[i] = 1.0f;
}

void mm_groupnorm_free(mm_groupnorm_t* gn) { free(gn->weight); free(gn->bias); }

void mm_groupnorm_forward(const mm_groupnorm_t* gn, const float* x, int h, int w, float* y) {
    int n = h * w;
    int c = gn->num_ch;
    int g = gn->num_groups;
    int cpg = c / g;
    float eps = gn->eps;

    for (int gi = 0; gi < g; gi++) {
        int start_c = gi * cpg;
        int end_c = start_c + cpg;
        int num_el = n * cpg;

        float mean = 0.0f, var = 0.0f;
        for (int i = 0; i < n; i++) {
            for (int ci = start_c; ci < end_c; ci++) {
                mean += x[i * c + ci];
            }
        }
        mean /= (float)num_el;

        for (int i = 0; i < n; i++) {
            for (int ci = start_c; ci < end_c; ci++) {
                float diff = x[i * c + ci] - mean;
                var += diff * diff;
            }
        }
        var = var / (float)num_el + eps;
        float inv_std = 1.0f / sqrtf(var);

        for (int i = 0; i < n; i++) {
            for (int ci = start_c; ci < end_c; ci++) {
                y[i * c + ci] = (x[i * c + ci] - mean) * inv_std * gn->weight[ci] + gn->bias[ci];
            }
        }
    }
}

void mm_resblock_init(mm_resblock_t* rb, int in_ch, int out_ch, int time_embed_dim) {
    rb->in_ch = in_ch;
    rb->out_ch = out_ch;
    rb->use_time_embed = (time_embed_dim > 0);
    mm_conv2d_init(&rb->conv1, in_ch, out_ch, 3, 1, 1, 1);
    mm_conv2d_init(&rb->conv2, out_ch, out_ch, 3, 1, 1, 1);
    mm_groupnorm_init(&rb->norm1, 32, in_ch, 1e-5f);
    mm_groupnorm_init(&rb->norm2, 32, out_ch, 1e-5f);

    if (rb->use_time_embed) {
        rb->time_embed_weight = (float*)malloc((size_t)time_embed_dim * out_ch * sizeof(float));
        rb->time_embed_bias = (float*)calloc((size_t)out_ch, sizeof(float));
        float scale = sqrtf(1.0f / (float)time_embed_dim);
        for (int i = 0; i < time_embed_dim * out_ch; i++) {
            rb->time_embed_weight[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale;
        }
    } else {
        rb->time_embed_weight = NULL;
        rb->time_embed_bias = NULL;
    }
}

void mm_resblock_free(mm_resblock_t* rb) {
    mm_conv2d_free(&rb->conv1);
    mm_conv2d_free(&rb->conv2);
    mm_groupnorm_free(&rb->norm1);
    mm_groupnorm_free(&rb->norm2);
    free(rb->time_embed_weight);
    free(rb->time_embed_bias);
}

void mm_resblock_forward(const mm_resblock_t* rb, const float* x,
                         const float* temb, int h, int w, float* y) {
    int n = h * w * rb->out_ch;
    int n_in = h * w * rb->in_ch;
    float* h1 = (float*)malloc((size_t)n_in * sizeof(float));
    float* gn1 = (float*)malloc((size_t)n_in * sizeof(float));
    float* c1 = (float*)malloc((size_t)n * sizeof(float));
    float* gn2 = (float*)malloc((size_t)n * sizeof(float));

    mm_groupnorm_forward(&rb->norm1, x, h, w, gn1);
    mm_gelu_forward(gn1, n_in, h1);
    mm_conv2d_forward(&rb->conv1, h1, h, w, c1);

    if (rb->use_time_embed && temb) {
        int ch = rb->out_ch;
        int temb_dim = rb->conv2.in_ch;
        for (int i = 0; i < n; i++) {
            int ci = i % ch;
            float te_sum = rb->time_embed_bias[ci];
            for (int d = 0; d < temb_dim; d++) {
                te_sum += temb[d] * rb->time_embed_weight[d * ch + ci];
            }
            c1[i] += te_sum;
        }
    }

    mm_groupnorm_forward(&rb->norm2, c1, h, w, gn2);
    mm_gelu_forward(gn2, n, c1);

    float* c2 = (float*)malloc((size_t)n * sizeof(float));
    mm_conv2d_forward(&rb->conv2, c1, h, w, c2);

    for (int i = 0; i < n; i++) {
        y[i] = c2[i] + (rb->in_ch == rb->out_ch ? x[i] : 0.0f);
    }

    free(h1); free(gn1); free(c1); free(gn2); free(c2);
}

void mm_cross_attn_init(mm_cross_attn_t* attn, int query_dim, int context_dim, int num_heads) {
    attn->num_heads = num_heads;
    attn->head_dim = query_dim / num_heads;
    attn->scale = 1.0f / sqrtf((float)attn->head_dim);
    mm_linear_init(&attn->q_proj, query_dim, query_dim);
    mm_linear_init(&attn->k_proj, context_dim, query_dim);
    mm_linear_init(&attn->v_proj, context_dim, query_dim);
    mm_linear_init(&attn->out_proj, query_dim, query_dim);
}

void mm_cross_attn_free(mm_cross_attn_t* attn) {
    mm_linear_free(&attn->q_proj);
    mm_linear_free(&attn->k_proj);
    mm_linear_free(&attn->v_proj);
    mm_linear_free(&attn->out_proj);
}

void mm_cross_attn_forward(const mm_cross_attn_t* attn, const float* x,
                           const float* context, int seq_len, int ctx_len, float* out) {
    int dim = attn->num_heads * attn->head_dim;
    int nh = attn->num_heads;
    int hd = attn->head_dim;

    float* q = (float*)malloc((size_t)seq_len * dim * sizeof(float));
    float* k = (float*)malloc((size_t)ctx_len * dim * sizeof(float));
    float* v = (float*)malloc((size_t)ctx_len * dim * sizeof(float));

    for (int s = 0; s < seq_len; s++) {
        mm_linear_forward(&attn->q_proj, x + s * dim, q + s * dim);
    }
    for (int s = 0; s < ctx_len; s++) {
        mm_linear_forward(&attn->k_proj, context + s * dim, k + s * dim);
        mm_linear_forward(&attn->v_proj, context + s * dim, v + s * dim);
    }

    for (int s = 0; s < seq_len; s++) {
        float* attn_out = (float*)calloc((size_t)dim, sizeof(float));
        for (int h = 0; h < nh; h++) {
            float max_score = -1e9f;
            float* scores = (float*)malloc((size_t)ctx_len * sizeof(float));
            for (int j = 0; j < ctx_len; j++) {
                float dot = 0.0f;
                for (int d = 0; d < hd; d++) {
                    dot += q[s * dim + h * hd + d] * k[j * dim + h * hd + d];
                }
                scores[j] = dot * attn->scale;
                if (scores[j] > max_score) max_score = scores[j];
            }
            float sum = 0.0f;
            for (int j = 0; j < ctx_len; j++) {
                scores[j] = expf(scores[j] - max_score);
                sum += scores[j];
            }
            for (int j = 0; j < ctx_len; j++) {
                scores[j] /= sum;
                for (int d = 0; d < hd; d++) {
                    attn_out[h * hd + d] += scores[j] * v[j * dim + h * hd + d];
                }
            }
            free(scores);
        }

        float final_out[768] = {0};
        mm_linear_forward(&attn->out_proj, attn_out, final_out);
        float* out_ptr = out + s * dim;
        for (int d = 0; d < dim && d < 768; d++) {
            out_ptr[d] = final_out[d];
        }
        free(attn_out);
    }

    free(q); free(k); free(v);
}

void mm_spatial_transformer_init(mm_spatial_transformer_t* st, int dim, int context_dim, int num_heads) {
    st->dim = dim;
    st->context_dim = context_dim;
    mm_groupnorm_init(&st->norm, 32, dim, 1e-5f);
    mm_conv2d_init(&st->proj_in, dim, dim, 1, 1, 0, 1);
    mm_cross_attn_init(&st->attn, dim, context_dim, num_heads);
    mm_conv2d_init(&st->proj_out, dim, dim, 1, 1, 0, 1);
}

void mm_spatial_transformer_free(mm_spatial_transformer_t* st) {
    mm_groupnorm_free(&st->norm);
    mm_conv2d_free(&st->proj_in);
    mm_cross_attn_free(&st->attn);
    mm_conv2d_free(&st->proj_out);
}

void mm_spatial_transformer_forward(const mm_spatial_transformer_t* st, const float* x,
                                    const float* context, int h, int w, int ctx_len, float* y) {
    int dim = st->dim;
    int n = h * w;
    int total = n * dim;

    float* h_gn = (float*)malloc((size_t)total * sizeof(float));
    float* h_conv = (float*)malloc((size_t)total * sizeof(float));
    float* tokens = (float*)malloc((size_t)n * dim * sizeof(float));
    float* attn_out = (float*)malloc((size_t)n * dim * sizeof(float));

    mm_groupnorm_forward(&st->norm, x, h, w, h_gn);
    mm_conv2d_forward(&st->proj_in, h_gn, h, w, h_conv);
    memcpy(tokens, h_conv, (size_t)total * sizeof(float));

    mm_cross_attn_forward(&st->attn, tokens, context, n, ctx_len, attn_out);

    float* h_out = (float*)malloc((size_t)total * sizeof(float));
    mm_conv2d_forward(&st->proj_out, attn_out, h, w, h_out);
    for (int i = 0; i < total; i++) y[i] = x[i] + h_out[i];

    free(h_gn); free(h_conv); free(tokens); free(attn_out); free(h_out);
}

void mm_unet_init(mm_unet_t* unet, int base_channels, int num_res_blocks) {
    unet->base_channels = base_channels;
    unet->num_res_blocks = num_res_blocks;
    unet->time_embed_dim = base_channels * 4;
    unet->num_down = 4;
    unet->num_up = 4;

    mm_conv2d_init(&unet->conv_in, 4, base_channels, 3, 1, 1, 1);

    int ch_mult[4] = {1, 2, 4, 4};
    unet->down_blocks = (mm_unet_down_block_t*)malloc((size_t)unet->num_down * sizeof(mm_unet_down_block_t));
    for (int i = 0; i < unet->num_down; i++) {
        mm_unet_down_block_t* db = &unet->down_blocks[i];
        db->ch_in = (i == 0) ? base_channels : base_channels * ch_mult[i - 1];
        db->ch_out = base_channels * ch_mult[i];
        db->num_res_blocks = num_res_blocks;
        db->num_attn_blocks = (i >= 2) ? 1 : 0;
        db->has_downsampler = (i < unet->num_down - 1);

        db->res_blocks = (mm_resblock_t*)malloc((size_t)num_res_blocks * sizeof(mm_resblock_t));
        mm_resblock_init(&db->res_blocks[0], db->ch_in, db->ch_out, unet->time_embed_dim);
        for (int r = 1; r < num_res_blocks; r++) {
            mm_resblock_init(&db->res_blocks[r], db->ch_out, db->ch_out, unet->time_embed_dim);
        }

        if (db->num_attn_blocks > 0) {
            db->attn_blocks = (mm_spatial_transformer_t*)malloc(sizeof(mm_spatial_transformer_t));
            mm_spatial_transformer_init(db->attn_blocks, db->ch_out, MM_SD_CROSS_ATTN_DIM, 8);
        } else {
            db->attn_blocks = NULL;
        }

        if (db->has_downsampler) {
            mm_conv2d_init(&db->downsampler, db->ch_out, db->ch_out, 3, 2, 1, 1);
        }
    }

    unet->mid_block.ch_in = base_channels * ch_mult[3];
    unet->mid_block.ch_out = base_channels * ch_mult[3];
    unet->mid_block.num_res_blocks = 1;
    unet->mid_block.num_attn_blocks = 1;
    unet->mid_block.res_blocks = (mm_resblock_t*)malloc(sizeof(mm_resblock_t));
    mm_resblock_init(&unet->mid_block.res_blocks[0], unet->mid_block.ch_in,
                     unet->mid_block.ch_out, unet->time_embed_dim);
    unet->mid_block.attn_blocks = (mm_spatial_transformer_t*)malloc(sizeof(mm_spatial_transformer_t));
    mm_spatial_transformer_init(unet->mid_block.attn_blocks, unet->mid_block.ch_out,
                                MM_SD_CROSS_ATTN_DIM, 8);

    unet->up_blocks = (mm_unet_up_block_t*)malloc((size_t)unet->num_up * sizeof(mm_unet_up_block_t));
    for (int i = 0; i < unet->num_up; i++) {
        mm_unet_up_block_t* ub = &unet->up_blocks[i];
        int idx = unet->num_up - i - 1;
        ub->ch_in = base_channels * ch_mult[idx] + (idx > 0 ? base_channels * ch_mult[idx] : 0);
        ub->ch_out = (idx > 0) ? base_channels * ch_mult[idx - 1] : base_channels;
        ub->num_res_blocks = num_res_blocks;
        ub->num_attn_blocks = (idx >= 2) ? 1 : 0;
        ub->has_upsampler = (i < unet->num_up - 1);

        ub->res_blocks = (mm_resblock_t*)malloc((size_t)num_res_blocks * sizeof(mm_resblock_t));
        mm_resblock_init(&ub->res_blocks[0], ub->ch_in, ub->ch_out, unet->time_embed_dim);
        for (int r = 1; r < num_res_blocks; r++) {
            mm_resblock_init(&ub->res_blocks[r], ub->ch_out, ub->ch_out, unet->time_embed_dim);
        }

        if (ub->num_attn_blocks > 0) {
            ub->attn_blocks = (mm_spatial_transformer_t*)malloc(sizeof(mm_spatial_transformer_t));
            mm_spatial_transformer_init(ub->attn_blocks, ub->ch_out, MM_SD_CROSS_ATTN_DIM, 8);
        } else {
            ub->attn_blocks = NULL;
        }

        if (ub->has_upsampler) {
            mm_conv2d_init(&ub->upsampler, ub->ch_out, ub->ch_out, 3, 1, 1, 1);
        }
    }

    mm_groupnorm_init(&unet->norm_out, 32, base_channels, 1e-5f);
    mm_conv2d_init(&unet->conv_out, base_channels, 4, 3, 1, 1, 1);

    for (int i = 0; i < 2; i++) {
        unet->time_embed[i] = (float*)malloc((size_t)unet->time_embed_dim * unet->time_embed_dim * sizeof(float));
        float scale = sqrtf(1.0f / (float)unet->time_embed_dim);
        for (int j = 0; j < unet->time_embed_dim * unet->time_embed_dim; j++) {
            unet->time_embed[i][j] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale;
        }
    }
}

void mm_unet_free(mm_unet_t* unet) {
    mm_conv2d_free(&unet->conv_in);
    for (int i = 0; i < unet->num_down; i++) {
        for (int r = 0; r < unet->down_blocks[i].num_res_blocks; r++)
            mm_resblock_free(&unet->down_blocks[i].res_blocks[r]);
        free(unet->down_blocks[i].res_blocks);
        if (unet->down_blocks[i].attn_blocks) {
            mm_spatial_transformer_free(unet->down_blocks[i].attn_blocks);
            free(unet->down_blocks[i].attn_blocks);
        }
        if (unet->down_blocks[i].has_downsampler)
            mm_conv2d_free(&unet->down_blocks[i].downsampler);
    }
    free(unet->down_blocks);

    mm_resblock_free(&unet->mid_block.res_blocks[0]);
    free(unet->mid_block.res_blocks);
    mm_spatial_transformer_free(unet->mid_block.attn_blocks);
    free(unet->mid_block.attn_blocks);

    for (int i = 0; i < unet->num_up; i++) {
        for (int r = 0; r < unet->up_blocks[i].num_res_blocks; r++)
            mm_resblock_free(&unet->up_blocks[i].res_blocks[r]);
        free(unet->up_blocks[i].res_blocks);
        if (unet->up_blocks[i].attn_blocks) {
            mm_spatial_transformer_free(unet->up_blocks[i].attn_blocks);
            free(unet->up_blocks[i].attn_blocks);
        }
        if (unet->up_blocks[i].has_upsampler)
            mm_conv2d_free(&unet->up_blocks[i].upsampler);
    }
    free(unet->up_blocks);

    mm_groupnorm_free(&unet->norm_out);
    mm_conv2d_free(&unet->conv_out);
    free(unet->time_embed[0]);
    free(unet->time_embed[1]);
}

void mm_unet_forward(const mm_unet_t* unet, const float* x, const float* timestep,
                     const float* context, int h, int w, int ctx_len, float* y) {
    int bc = unet->base_channels;
    int ted = unet->time_embed_dim;

    float* t_emb = (float*)malloc((size_t)ted * sizeof(float));
    float* temb_hidden = (float*)calloc(256, sizeof(float));
    mm_sinusoidal_embedding((int)(timestep[0] * 1000), ted, t_emb);
    for (int o = 0; o < ted && o < 256; o++)
        for (int i = 0; i < ted; i++)
            temb_hidden[o] += t_emb[i] * unet->time_embed[0][i * ted + o];
    for (int o = 0; o < ted && o < 256; o++) {
        float val = temb_hidden[o]; temb_hidden[o] = 0.0f;
        for (int i = 0; i < ted; i++)
            temb_hidden[o] += val * unet->time_embed[1][i * ted + o];
    }

    int unet_max_ch = bc;
    for (int i = 0; i < unet->num_down; i++) {
        int ch = unet->down_blocks[i].ch_out;
        if (ch > unet_max_ch) unet_max_ch = ch;
    }
    if (unet->mid_block.ch_out > unet_max_ch) unet_max_ch = unet->mid_block.ch_out;
    for (int i = 0; i < unet->num_up; i++) {
        int ch = unet->up_blocks[i].ch_out;
        if (ch > unet_max_ch) unet_max_ch = ch;
    }

    int unet_nmax = h * w * unet_max_ch;
    float* cur_feat = (float*)calloc((size_t)unet_nmax, sizeof(float));
    mm_conv2d_forward(&unet->conv_in, x, h, w, cur_feat);

    typedef struct { float* feat; int h; int w; int c; } saved_t;
    saved_t saved[4];
    int cfeat_h = h, cfeat_w = w, cfeat_c = bc;

    /* ── Down blocks ── */
    for (int i = 0; i < unet->num_down; i++) {
        mm_unet_down_block_t* db = &unet->down_blocks[i];
        for (int r = 0; r < db->num_res_blocks; r++)
            mm_resblock_forward(&db->res_blocks[r], cur_feat, temb_hidden, cfeat_h, cfeat_w, cur_feat);
        if (db->attn_blocks) {
            int n_el = cfeat_h * cfeat_w * db->ch_out;
            float* temp = (float*)malloc((size_t)n_el * sizeof(float));
            mm_spatial_transformer_forward(db->attn_blocks, cur_feat, context, cfeat_h, cfeat_w, ctx_len, temp);
            memcpy(cur_feat, temp, (size_t)n_el * sizeof(float));
            free(temp);
        }
        saved[i].feat = (float*)malloc((size_t)(cfeat_h * cfeat_w * db->ch_out) * sizeof(float));
        memcpy(saved[i].feat, cur_feat, (size_t)(cfeat_h * cfeat_w * db->ch_out) * sizeof(float));
        saved[i].h = cfeat_h; saved[i].w = cfeat_w; saved[i].c = db->ch_out;
        if (db->has_downsampler) {
            int nh = (cfeat_h + 1) / 2, nw = (cfeat_w + 1) / 2;
            float* fn = (float*)calloc((size_t)nh * nw * unet_max_ch, sizeof(float));
            mm_conv2d_forward(&db->downsampler, cur_feat, cfeat_h, cfeat_w, fn);
            free(cur_feat);
            cur_feat = fn; cfeat_h = nh; cfeat_w = nw; cfeat_c = db->ch_out;
        }
    }

    /* ── Mid block ── */
    for (int r = 0; r < unet->mid_block.num_res_blocks; r++)
        mm_resblock_forward(&unet->mid_block.res_blocks[r], cur_feat, temb_hidden, cfeat_h, cfeat_w, cur_feat);
    {
        int n_el = cfeat_h * cfeat_w * unet->mid_block.ch_out;
        float* temp = (float*)malloc((size_t)n_el * sizeof(float));
        mm_spatial_transformer_forward(unet->mid_block.attn_blocks, cur_feat, context, cfeat_h, cfeat_w, ctx_len, temp);
        memcpy(cur_feat, temp, (size_t)n_el * sizeof(float));
        free(temp);
    }

    /* ── Up blocks ── */
    for (int i = 0; i < unet->num_up; i++) {
        mm_unet_up_block_t* ub = &unet->up_blocks[i];
        int si = unet->num_up - i - 1;
        saved_t* sv = &saved[si];
        int concat_ch = cfeat_c + sv->c;
        float* concat = (float*)malloc((size_t)cfeat_h * cfeat_w * concat_ch * sizeof(float));
        for (int p = 0; p < cfeat_h * cfeat_w; p++) {
            int pc = p * cfeat_c, ps = p * sv->c, pt = p * concat_ch;
            for (int d = 0; d < cfeat_c; d++) concat[pt + d] = cur_feat[pc + d];
            for (int d = 0; d < sv->c; d++) concat[pt + cfeat_c + d] = sv->feat[ps + d];
        }
        free(cur_feat);
        cur_feat = concat; cfeat_c = concat_ch;
        for (int r = 0; r < ub->num_res_blocks; r++) {
            mm_resblock_forward(&ub->res_blocks[r], cur_feat, temb_hidden, cfeat_h, cfeat_w, cur_feat);
            cfeat_c = ub->ch_out;
        }
        if (ub->attn_blocks) {
            int n_el = cfeat_h * cfeat_w * ub->ch_out;
            float* temp = (float*)malloc((size_t)n_el * sizeof(float));
            mm_spatial_transformer_forward(ub->attn_blocks, cur_feat, context, cfeat_h, cfeat_w, ctx_len, temp);
            memcpy(cur_feat, temp, (size_t)n_el * sizeof(float));
            free(temp);
        }
        if (ub->has_upsampler) {
            int nh = cfeat_h * 2, nw = cfeat_w * 2;
            float* up = (float*)calloc((size_t)nh * nw * unet_max_ch, sizeof(float));
            for (int yy = 0; yy < cfeat_h; yy++) {
                for (int xx = 0; xx < cfeat_w; xx++) {
                    int src = (yy * cfeat_w + xx) * cfeat_c;
                    int d00 = ((yy*2)*nw+(xx*2))*cfeat_c;
                    int d01 = ((yy*2)*nw+(xx*2+1))*cfeat_c;
                    int d10 = ((yy*2+1)*nw+(xx*2))*cfeat_c;
                    int d11 = ((yy*2+1)*nw+(xx*2+1))*cfeat_c;
                    for (int k = 0; k < cfeat_c; k++) {
                        float v = cur_feat[src+k];
                        up[d00+k]=v; up[d01+k]=v; up[d10+k]=v; up[d11+k]=v;
                    }
                }
            }
            free(cur_feat); cur_feat = up; cfeat_h = nh; cfeat_w = nw;
        }
    }

    /* ── Final ── */
    float* h_norm = (float*)malloc((size_t)(cfeat_h * cfeat_w * bc) * sizeof(float));
    mm_groupnorm_forward(&unet->norm_out, cur_feat, cfeat_h, cfeat_w, h_norm);
    mm_gelu_forward(h_norm, cfeat_h * cfeat_w * bc, h_norm);
    mm_conv2d_forward(&unet->conv_out, h_norm, cfeat_h, cfeat_w, y);

    for (int i = 0; i < unet->num_down; i++) free(saved[i].feat);
    free(cur_feat); free(h_norm); free(t_emb); free(temb_hidden);
}

void mm_sinusoidal_embedding(int t, int dim, float* emb) {
    float half = (float)dim / 2.0f;
    for (int i = 0; i < dim; i++) {
        float freq = (i < dim / 2) ?
            (float)t / powf(10000.0f, (float)i / half) :
            (float)t / powf(10000.0f, (float)(i - dim / 2) / half);
        emb[i] = (i % 2 == 0) ? sinf(freq) : cosf(freq);
    }
}

void mm_vae_init(mm_vae_t* vae, int latent_dim, int base_channels, int num_res_blocks) {
    vae->latent_dim = latent_dim;
    vae->base_channels = base_channels;
    vae->num_res_blocks = num_res_blocks;

    mm_conv2d_init(&vae->encoder_conv_in, 3, base_channels, 3, 1, 1, 1);

    vae->encoder_res_blocks = (mm_resblock_t*)malloc((size_t)num_res_blocks * sizeof(mm_resblock_t));
    int ch = base_channels;
    for (int i = 0; i < num_res_blocks; i++) {
        int out_ch = (i < num_res_blocks - 1) ? ch * 2 : ch;
        mm_resblock_init(&vae->encoder_res_blocks[i], ch, out_ch, 0);
        ch = out_ch;
    }

    mm_conv2d_init(&vae->encoder_conv_out_mu, ch, latent_dim, 3, 1, 1, 1);
    mm_conv2d_init(&vae->encoder_conv_out_logvar, ch, latent_dim, 3, 1, 1, 1);

    mm_conv2d_init(&vae->decoder_conv_in, latent_dim, ch, 3, 1, 1, 1);

    vae->decoder_res_blocks = (mm_resblock_t*)malloc((size_t)num_res_blocks * sizeof(mm_resblock_t));
    for (int i = 0; i < num_res_blocks; i++) {
        int out_ch = (i < num_res_blocks - 1) ? ch / 2 : ch;
        if (out_ch < base_channels) out_ch = base_channels;
        mm_resblock_init(&vae->decoder_res_blocks[i], ch, out_ch, 0);
        ch = out_ch;
    }

    mm_conv2d_init(&vae->decoder_conv_out, ch, 3, 3, 1, 1, 1);
}

void mm_vae_free(mm_vae_t* vae) {
    mm_conv2d_free(&vae->encoder_conv_in);
    for (int i = 0; i < vae->num_res_blocks; i++) mm_resblock_free(&vae->encoder_res_blocks[i]);
    free(vae->encoder_res_blocks);
    mm_conv2d_free(&vae->encoder_conv_out_mu);
    mm_conv2d_free(&vae->encoder_conv_out_logvar);

    mm_conv2d_free(&vae->decoder_conv_in);
    for (int i = 0; i < vae->num_res_blocks; i++) mm_resblock_free(&vae->decoder_res_blocks[i]);
    free(vae->decoder_res_blocks);
    mm_conv2d_free(&vae->decoder_conv_out);
}

void mm_vae_encode(const mm_vae_t* vae, const float* image, int h, int w, int c,
                   float* mean, float* logvar) {
    (void)c;
    int bc = vae->base_channels;
    int lh = h, lw = w;
    /* Compute max channels: resblocks may double channels each except the last */
    int max_ch = bc;
    for (int i = 0; i < vae->num_res_blocks - 1; i++) max_ch *= 2;
    int n_in = h * w * max_ch;
    float* h_enc = (float*)calloc((size_t)n_in, sizeof(float));
    mm_conv2d_forward(&vae->encoder_conv_in, image, h, w, h_enc);

    int ch = bc;
    for (int i = 0; i < vae->num_res_blocks; i++) {
        mm_resblock_forward(&vae->encoder_res_blocks[i], h_enc, NULL, lh, lw, h_enc);
        if (i < vae->num_res_blocks - 1) {
            ch *= 2;
            int nlh = (lh + 1) / 2;
            int nlw = (lw + 1) / 2;
            float* down = (float*)calloc((size_t)nlh * nlw * max_ch, sizeof(float));
            float* temp_gelu = (float*)malloc((size_t)lh * lw * ch * sizeof(float));
            mm_gelu_forward(h_enc, lh * lw * ch, temp_gelu);
            for (int y = 0; y < nlh; y++) {
                for (int x = 0; x < nlw; x++) {
                    for (int ci = 0; ci < ch; ci++) {
                        down[(y * nlw + x) * ch + ci] = temp_gelu[((y * 2) * lw + (x * 2)) * ch + ci];
                    }
                }
            }
            free(temp_gelu);
            free(h_enc);
            h_enc = down;
            lh = nlh; lw = nlw;
        }
    }

    mm_conv2d_forward(&vae->encoder_conv_out_mu, h_enc, lh, lw, mean);
    mm_conv2d_forward(&vae->encoder_conv_out_logvar, h_enc, lh, lw, logvar);
    free(h_enc);
}

void mm_vae_decode(const mm_vae_t* vae, const float* latent, float* image,
                   int h, int w, int c) {
    (void)c;
    (void)vae->latent_dim;  /* stored for inspection, shape derived from h/w */

    /* Compute upsampling factor: decoder does (num_res_blocks-1) 2x upsamples */
    int upsample_scale = 1 << (vae->num_res_blocks - 1);
    int lh = h / upsample_scale, lw = w / upsample_scale;

    /* Compute max channels for decoder (conv_in may have more than base_channels) */
    int max_ch = vae->base_channels;
    for (int i = 0; i < vae->num_res_blocks - 1; i++) max_ch *= 2;
    /* decoder_conv_in out_ch = final encoder ch = max_ch */
    if (max_ch < vae->base_channels * 2) max_ch = vae->base_channels * 2;
    int n_dec = lh * lw * max_ch;
    float* h_dec = (float*)calloc((size_t)n_dec, sizeof(float));
    mm_conv2d_forward(&vae->decoder_conv_in, latent, lh, lw, h_dec);

    int ch = vae->base_channels;  /* starting channel count for resblock input */
    for (int i = 0; i < vae->num_res_blocks; i++) {
        mm_resblock_forward(&vae->decoder_res_blocks[i], h_dec, NULL, lh, lw, h_dec);
        if (i < vae->num_res_blocks - 1) {
            int nlh = lh * 2;
            int nlw = lw * 2;
            float* up = (float*)calloc((size_t)nlh * nlw * max_ch, sizeof(float));
            float* temp_gelu = (float*)malloc((size_t)lh * lw * ch * sizeof(float));
            mm_gelu_forward(h_dec, lh * lw * ch, temp_gelu);
            for (int y = 0; y < lh; y++) {
                for (int x = 0; x < lw; x++) {
                    for (int ci = 0; ci < ch; ci++) {
                        float v = temp_gelu[(y * lw + x) * ch + ci];
                        up[((y * 2) * nlw + (x * 2)) * ch + ci] = v;
                        up[((y * 2) * nlw + (x * 2 + 1)) * ch + ci] = v;
                        up[((y * 2 + 1) * nlw + (x * 2)) * ch + ci] = v;
                        up[((y * 2 + 1) * nlw + (x * 2 + 1)) * ch + ci] = v;
                    }
                }
            }
            free(temp_gelu);
            free(h_dec);
            h_dec = up;
            lh = nlh; lw = nlw;
        }
    }

    mm_conv2d_forward(&vae->decoder_conv_out, h_dec, lh, lw, image);
    free(h_dec);
}

void mm_vae_sample(const float* mean, const float* logvar, int n, float* latent) {
    for (int i = 0; i < n; i++) {
        float std = expf(0.5f * logvar[i]);
        float eps = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f);
        latent[i] = mean[i] + std * eps;
    }
}

void mm_sd_init(mm_stable_diffusion_t* sd, int image_size, int latent_dim,
                int base_channels, int num_res_blocks) {
    sd->image_size = image_size;
    /* VAE downsampling: (num_res_blocks-1) 2x reductions */
    sd->latent_size = image_size / (1 << (num_res_blocks - 1));
    sd->latent_channels = latent_dim;
    sd->num_timesteps = MM_SD_NUM_TIMESTEPS;
    sd->guidance_scale = 7.5f;
    sd->beta_start = 0.00085f;
    sd->beta_end = 0.012f;
    sd->schedule_type = MM_SD_SCHEDULE_LINEAR;

    mm_sd_beta_schedule(NULL, sd->num_timesteps, sd->beta_start, sd->beta_end, sd->schedule_type);
    mm_vae_init(&sd->vae, latent_dim, base_channels, num_res_blocks);
    mm_unet_init(&sd->unet, base_channels, num_res_blocks);
}

void mm_sd_free(mm_stable_diffusion_t* sd) {
    mm_vae_free(&sd->vae);
    mm_unet_free(&sd->unet);
}

void mm_sd_beta_schedule(mm_schedule_t* schedule, int num_timesteps,
                         float beta_start, float beta_end, mm_schedule_type_t type) {
    if (!schedule) return;
    mm_schedule_init(schedule, num_timesteps);
    for (int i = 0; i < num_timesteps; i++) {
        float frac = (float)i / (float)(num_timesteps - 1);
        switch (type) {
            case MM_SD_SCHEDULE_COSINE:
                schedule->data[i] = beta_start + (beta_end - beta_start) * 0.5f * (1.0f - cosf(3.14159265f * frac));
                break;
            case MM_SD_SCHEDULE_SQRT:
                schedule->data[i] = beta_start + (beta_end - beta_start) * sqrtf(frac);
                break;
            default:
                schedule->data[i] = beta_start + (beta_end - beta_start) * frac;
                break;
        }
    }
}

void mm_sd_alphas_from_betas(const mm_schedule_t* betas, mm_schedule_t* alphas,
                             mm_schedule_t* alpha_cumprod) {
    mm_schedule_init(alphas, betas->n);
    mm_schedule_init(alpha_cumprod, betas->n);
    alpha_cumprod->data[0] = 1.0f - betas->data[0];
    alphas->data[0] = 1.0f - betas->data[0];
    for (int i = 1; i < betas->n; i++) {
        alphas->data[i] = 1.0f - betas->data[i];
        alpha_cumprod->data[i] = alpha_cumprod->data[i - 1] * alphas->data[i];
    }
}

void mm_sd_add_noise(const float* x, const float* noise, float sqrt_alpha_cumprod,
                     float sqrt_one_minus_alpha_cumprod, int n, float* out) {
    for (int i = 0; i < n; i++) {
        out[i] = sqrt_alpha_cumprod * x[i] + sqrt_one_minus_alpha_cumprod * noise[i];
    }
}

void mm_sd_predict_noise(const mm_stable_diffusion_t* sd, const float* latent,
                         int timestep, const float* text_embedding, int ctx_len,
                         float* noise_pred_uncond, float* noise_pred_text) {
    float t_arr[1] = {(float)timestep / (float)sd->num_timesteps};
    mm_unet_forward(&sd->unet, latent, t_arr, text_embedding,
                    sd->latent_size, sd->latent_size, ctx_len, noise_pred_text);
    mm_unet_forward(&sd->unet, latent, t_arr, NULL,
                    sd->latent_size, sd->latent_size, 0, noise_pred_uncond);
}

void mm_sd_ddim_step(const float* x_t, const float* noise_pred, int t, int t_prev,
                     const mm_schedule_t* alpha_cumprod, int n, float eta, float* x_t_prev) {
    float alpha_t = alpha_cumprod->data[t];
    float alpha_tp = (t_prev >= 0) ? alpha_cumprod->data[t_prev] : 1.0f;
    float sigma_t = eta * sqrtf((1.0f - alpha_tp) / (1.0f - alpha_t) * (1.0f - alpha_t / alpha_tp));

    float pred_x0_sqrt = sqrtf(alpha_tp);
    float noise_sqrt = sqrtf(1.0f - alpha_tp - sigma_t * sigma_t);

    for (int i = 0; i < n; i++) {
        float pred_orig = (x_t[i] - sqrtf(1.0f - alpha_t) * noise_pred[i]) / sqrtf(alpha_t);
        float dir_xt = sigma_t * (x_t[i] - pred_orig * sqrtf(alpha_t)) / sqrtf(1.0f - alpha_t);
        x_t_prev[i] = pred_x0_sqrt * pred_orig + noise_sqrt * noise_pred[i] + dir_xt;
    }
}

void mm_sd_dpm_pp_2m_step(const float* x_t, const float* noise_pred,
                          const float* noise_pred_prev,
                          int t, int t_prev, int t_prev2,
                          const mm_schedule_t* alpha_cumprod,
                          int n, float* x_t_prev) {
    (void)t_prev2;
    float alpha_t = alpha_cumprod->data[t];
    float alpha_s = alpha_cumprod->data[t_prev];
    float h = logf(alpha_s) - logf(alpha_t);

    for (int i = 0; i < n; i++) {
        float pred_orig = (x_t[i] - sqrtf(1.0f - alpha_t) * noise_pred[i]) / sqrtf(alpha_t);
        float noise_comb = noise_pred[i] + (noise_pred[i] - noise_pred_prev[i]) * 0.5f;
        float sigma_s = sqrtf(1.0f - alpha_s);
        x_t_prev[i] = sqrtf(alpha_s) * pred_orig + sigma_s * noise_comb;
    }
    (void)h;
}

void mm_sd_cfg_guidance(float* noise_pred_cond, const float* noise_pred_uncond,
                        float guidance_scale, int n) {
    for (int i = 0; i < n; i++) {
        noise_pred_cond[i] = noise_pred_uncond[i] + guidance_scale * (noise_pred_cond[i] - noise_pred_uncond[i]);
    }
}

void mm_sd_generate(const mm_stable_diffusion_t* sd, const float* text_embedding,
                    int ctx_len, int num_steps, mm_sd_sampler_t sampler,
                    int h, int w, int c, float* image) {
    int ls = sd->latent_size;
    int lc = sd->latent_channels;
    int n_latent = ls * ls * lc;

    float* latent = (float*)malloc((size_t)n_latent * sizeof(float));
    for (int i = 0; i < n_latent; i++) {
        latent[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f);
    }

    mm_schedule_t betas, alphas, alpha_cp;
    mm_sd_beta_schedule(&betas, sd->num_timesteps, sd->beta_start, sd->beta_end, sd->schedule_type);
    mm_sd_alphas_from_betas(&betas, &alphas, &alpha_cp);

    int* ts = (int*)malloc((size_t)num_steps * sizeof(int));
    for (int i = 0; i < num_steps; i++) {
        ts[i] = (int)((float)sd->num_timesteps * (1.0f - (float)i / (float)num_steps));
        if (ts[i] >= sd->num_timesteps) ts[i] = sd->num_timesteps - 1;
    }

    float* noise_pred_prev = (float*)calloc((size_t)n_latent, sizeof(float));

    for (int i = 0; i < num_steps - 1; i++) {
        int t = ts[i], t_prev = ts[i + 1];

        float* noise_cond = (float*)malloc((size_t)n_latent * sizeof(float));
        float* noise_uncond = (float*)malloc((size_t)n_latent * sizeof(float));
        mm_sd_predict_noise(sd, latent, t, text_embedding, ctx_len, noise_uncond, noise_cond);
        mm_sd_cfg_guidance(noise_cond, noise_uncond, sd->guidance_scale, n_latent);

        float* latent_new = (float*)malloc((size_t)n_latent * sizeof(float));

        if (sampler == MM_SD_SAMPLER_DDIM) {
            mm_sd_ddim_step(latent, noise_cond, t, t_prev, &alpha_cp, n_latent, 0.0f, latent_new);
        } else {
            int t_prev2 = (i > 0) ? ts[i - 1] : t;
            mm_sd_dpm_pp_2m_step(latent, noise_cond, noise_pred_prev, t, t_prev, t_prev2, &alpha_cp, n_latent, latent_new);
        }

        memcpy(noise_pred_prev, noise_cond, (size_t)n_latent * sizeof(float));
        free(latent);
        free(noise_cond);
        free(noise_uncond);
        latent = latent_new;
    }

    float* decoded = (float*)malloc((size_t)h * w * c * sizeof(float));
    mm_vae_decode(&sd->vae, latent, decoded, h, w, c);

    float mean = 0.0f, std = 0.0f;
    int total = h * w * c;
    for (int i = 0; i < total; i++) mean += decoded[i];
    mean /= (float)total;
    for (int i = 0; i < total; i++) {
        float diff = decoded[i] - mean;
        std += diff * diff;
    }
    std = sqrtf(std / (float)total) + 1e-6f;
    for (int i = 0; i < total; i++) {
        image[i] = ((decoded[i] - mean) / std) * 0.5f + 0.5f;
        if (image[i] < 0.0f) image[i] = 0.0f;
        if (image[i] > 1.0f) image[i] = 1.0f;
    }

    free(decoded);
    free(latent);
    free(noise_pred_prev);
    free(ts);
    mm_schedule_free(&betas);
    mm_schedule_free(&alphas);
    mm_schedule_free(&alpha_cp);
}

void mm_sd_inpaint(const mm_stable_diffusion_t* sd, const float* image,
                   const float* mask, const float* text_embedding,
                   int ctx_len, int h, int w, int c, int num_steps, float* out) {
    int ls = sd->latent_size;
    int lc = sd->latent_channels;
    int n_latent = ls * ls * lc;

    float* latent_orig = (float*)malloc((size_t)n_latent * sizeof(float));
    mm_vae_encode(&sd->vae, image, h, w, c, latent_orig, latent_orig);

    float* latent = (float*)malloc((size_t)n_latent * sizeof(float));
    for (int i = 0; i < n_latent; i++) {
        latent[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f);
    }

    mm_schedule_t betas, alphas, alpha_cp;
    mm_sd_beta_schedule(&betas, sd->num_timesteps, sd->beta_start, sd->beta_end, sd->schedule_type);
    mm_sd_alphas_from_betas(&betas, &alphas, &alpha_cp);

    int* ts = (int*)malloc((size_t)num_steps * sizeof(int));
    for (int i = 0; i < num_steps; i++) {
        ts[i] = (int)((float)sd->num_timesteps * (1.0f - (float)i / (float)num_steps));
        if (ts[i] >= sd->num_timesteps) ts[i] = sd->num_timesteps - 1;
    }

    for (int i = 0; i < num_steps - 1; i++) {
        int t = ts[i], t_prev = ts[i + 1];

        float* noise_cond = (float*)malloc((size_t)n_latent * sizeof(float));
        float* noise_uncond = (float*)malloc((size_t)n_latent * sizeof(float));
        mm_sd_predict_noise(sd, latent, t, text_embedding, ctx_len, noise_uncond, noise_cond);
        mm_sd_cfg_guidance(noise_cond, noise_uncond, sd->guidance_scale, n_latent);

        float* latent_new = (float*)malloc((size_t)n_latent * sizeof(float));
        mm_sd_ddim_step(latent, noise_cond, t, t_prev, &alpha_cp, n_latent, 0.0f, latent_new);

        for (int j = 0; j < n_latent; j++) {
            int py = j / (ls * lc);
            int off = j % (ls * lc);
            int px = off / lc;
            float m = mask[(py * 8 * w + px * 8) * c];
            if (m > 0.5f) {
                float sqrt_a = sqrtf(alpha_cp.data[t_prev]);
                latent_new[j] = sqrt_a * latent_orig[j] + sqrtf(1.0f - alpha_cp.data[t_prev]) * ((float)rand() / (float)RAND_MAX - 0.5f);
            }
        }

        free(latent);
        free(noise_cond);
        free(noise_uncond);
        latent = latent_new;
    }

    mm_vae_decode(&sd->vae, latent, out, h, w, c);

    free(latent_orig);
    free(latent);
    free(ts);
    mm_schedule_free(&betas);
    mm_schedule_free(&alphas);
    mm_schedule_free(&alpha_cp);
}

void mm_sd_pipe_encode(const mm_stable_diffusion_t* sd, const float* image,
                       int h, int w, int c, float* latent) {
    int scale = 1 << (sd->vae.num_res_blocks - 1);
    float* logvar = (float*)malloc((size_t)(h / scale) * (w / scale) * sd->latent_channels * sizeof(float));
    mm_vae_encode(&sd->vae, image, h, w, c, latent, logvar);
    free(logvar);
}

void mm_sd_pipe_decode(const mm_stable_diffusion_t* sd, const float* latent,
                       int latent_h, int latent_w, int h, int w, int c, float* image) {
    (void)latent_h; (void)latent_w;
    mm_vae_decode(&sd->vae, latent, image, h, w, c);
}

void mm_schedule_init(mm_schedule_t* s, int n) {
    s->n = n;
    s->allocated = 1;
    s->data = (float*)calloc((size_t)n, sizeof(float));
}

void mm_schedule_free(mm_schedule_t* s) {
    if (s->allocated) {
        free(s->data);
        s->allocated = 0;
    }
}
