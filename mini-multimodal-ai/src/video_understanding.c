#include "video_understanding.h"
#include "clip_contrastive.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

void mm_video_clip_init(mm_video_clip_t* clip, int num_frames, int h, int w, int c) {
    clip->num_frames = num_frames;
    clip->height = h;
    clip->width = w;
    clip->channels = c;
    clip->data = (float*)calloc((size_t)num_frames * h * w * c, sizeof(float));
}

void mm_video_clip_free(mm_video_clip_t* clip) { free(clip->data); }

void mm_video_sample_frames(const float* video, int total_frames,
                            int h, int w, int c, int num_sample,
                            mm_video_clip_t* clip) {
    int* indices = (int*)malloc((size_t)num_sample * sizeof(int));
    mm_video_uniform_sample(video, total_frames, h, w, c, num_sample, indices);
    int frame_size = h * w * c;
    for (int i = 0; i < num_sample && i < clip->num_frames; i++) {
        int src_idx = indices[i];
        if (src_idx >= 0 && src_idx < total_frames) {
            memcpy(clip->data + i * frame_size,
                   video + src_idx * frame_size,
                   (size_t)frame_size * sizeof(float));
        }
    }
    free(indices);
}

void mm_video_uniform_sample(const float* video, int total_frames,
                             int h, int w, int c, int num_sample,
                             int* frame_indices) {
    (void)video; (void)h; (void)w; (void)c;
    for (int i = 0; i < num_sample; i++) {
        frame_indices[i] = i * total_frames / num_sample;
        if (frame_indices[i] >= total_frames) frame_indices[i] = total_frames - 1;
    }
}

void mm_conv3d_init(mm_conv3d_t* c3d, int in_ch, int out_ch,
                    int kt, int kh, int kw, int st, int sh, int sw,
                    int pt, int ph, int pw) {
    c3d->in_ch = in_ch;
    c3d->out_ch = out_ch;
    c3d->kt = kt; c3d->kh = kh; c3d->kw = kw;
    c3d->st = st; c3d->sh = sh; c3d->sw = sw;
    c3d->pt = pt; c3d->ph = ph; c3d->pw = pw;

    int nw = out_ch * in_ch * kt * kh * kw;
    c3d->weight = (float*)calloc((size_t)nw, sizeof(float));
    c3d->bias = (float*)calloc((size_t)out_ch, sizeof(float));
    float scale = sqrtf(2.0f / (float)(in_ch * kt * kh * kw));
    for (int i = 0; i < nw; i++) {
        c3d->weight[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale;
    }
}

void mm_conv3d_free(mm_conv3d_t* c3d) { free(c3d->weight); free(c3d->bias); }

void mm_conv3d_forward(const mm_conv3d_t* c3d, const float* x,
                       int t, int h, int w, float* y, int* ot, int* oh, int* ow) {
    *ot = (t + 2 * c3d->pt - c3d->kt) / c3d->st + 1;
    *oh = (h + 2 * c3d->ph - c3d->kh) / c3d->sh + 1;
    *ow = (w + 2 * c3d->pw - c3d->kw) / c3d->sw + 1;
    int dt = *ot, dh = *oh, dw = *ow;

    int ic = c3d->in_ch, oc = c3d->out_ch;
    int kt = c3d->kt, kh = c3d->kh, kw = c3d->kw;
    int st = c3d->st, sh = c3d->sh, sw = c3d->sw;
    int pt = c3d->pt, ph = c3d->ph, pw = c3d->pw;

    for (int o_t = 0; o_t < dt; o_t++) {
        for (int o_h = 0; o_h < dh; o_h++) {
            for (int o_w = 0; o_w < dw; o_w++) {
                for (int o_c = 0; o_c < oc; o_c++) {
                    float sum = c3d->bias[o_c];
                    for (int i_c = 0; i_c < ic; i_c++) {
                        for (int k_t = 0; k_t < kt; k_t++) {
                            for (int k_h = 0; k_h < kh; k_h++) {
                                for (int k_w = 0; k_w < kw; k_w++) {
                                    int i_t = o_t * st + k_t - pt;
                                    int i_h = o_h * sh + k_h - ph;
                                    int i_w = o_w * sw + k_w - pw;
                                    if (i_t >= 0 && i_t < t && i_h >= 0 && i_h < h && i_w >= 0 && i_w < w) {
                                        int w_idx = (((o_c * ic + i_c) * kt + k_t) * kh + k_h) * kw + k_w;
                                        int x_idx = ((i_t * h + i_h) * w + i_w) * ic + i_c;
                                        sum += x[x_idx] * c3d->weight[w_idx];
                                    }
                                }
                            }
                        }
                    }
                    y[((o_t * dh + o_h) * dw + o_w) * oc + o_c] = sum;
                }
            }
        }
    }
}

void mm_batchnorm3d_init(mm_batchnorm3d_t* bn, int nf) {
    bn->num_features = nf;
    bn->weight = (float*)calloc((size_t)nf, sizeof(float));
    bn->bias = (float*)calloc((size_t)nf, sizeof(float));
    for (int i = 0; i < nf; i++) bn->weight[i] = 1.0f;
}

void mm_batchnorm3d_free(mm_batchnorm3d_t* bn) { free(bn->weight); free(bn->bias); }

void mm_batchnorm3d_forward(const mm_batchnorm3d_t* bn, const float* x,
                            int t, int h, int w, float* y) {
    int nf = bn->num_features;
    int sp = h * w;
    int total = t * sp * nf;

    for (int c = 0; c < nf; c++) {
        float mean = 0.0f, var = 0.0f;
        for (int i = 0; i < t; i++) {
            for (int s = 0; s < sp; s++) {
                mean += x[(i * sp + s) * nf + c];
            }
        }
        mean /= (float)(t * sp);
        for (int i = 0; i < t; i++) {
            for (int s = 0; s < sp; s++) {
                float diff = x[(i * sp + s) * nf + c] - mean;
                var += diff * diff;
            }
        }
        var = var / (float)(t * sp) + 1e-5f;
        float inv_std = 1.0f / sqrtf(var);

        for (int i = 0; i < t; i++) {
            for (int s = 0; s < sp; s++) {
                int idx = (i * sp + s) * nf + c;
                y[idx] = (x[idx] - mean) * inv_std * bn->weight[c] + bn->bias[c];
            }
        }
    }
    (void)total;
}

void mm_c3d_block_init(mm_c3d_block_t* blk, int in_ch, int out_ch,
                       int kt, int kh, int kw, int st, int sh, int sw) {
    mm_conv3d_init(&blk->conv, in_ch, out_ch, kt, kh, kw, st, sh, sw, 1, 1, 1);
    mm_batchnorm3d_init(&blk->bn, out_ch);
    blk->use_relu = 1;
    blk->use_pool = 1;
    blk->pool_kt = 1; blk->pool_kh = 2; blk->pool_kw = 2;
    blk->pool_st = 1; blk->pool_sh = 2; blk->pool_sw = 2;
}

void mm_c3d_block_free(mm_c3d_block_t* blk) {
    mm_conv3d_free(&blk->conv);
    mm_batchnorm3d_free(&blk->bn);
}

void mm_c3d_block_forward(const mm_c3d_block_t* blk, const float* x,
                          int t, int h, int w, float* y, int* ot, int* oh, int* ow) {
    int dt, dh, dw;
    mm_conv3d_forward(&blk->conv, x, t, h, w, y, &dt, &dh, &dw);
    mm_batchnorm3d_forward(&blk->bn, y, dt, dh, dw, y);

    if (blk->use_relu) {
        int total = dt * dh * dw * blk->conv.out_ch;
        for (int i = 0; i < total; i++) {
            if (y[i] < 0.0f) y[i] = 0.0f;
        }
    }

    if (blk->use_pool) {
        *ot = (dt - blk->pool_kt) / blk->pool_st + 1;
        *oh = (dh - blk->pool_kh) / blk->pool_sh + 1;
        *ow = (dw - blk->pool_kw) / blk->pool_sw + 1;
        int pdt = *ot, pdh = *oh, pdw = *ow;
        int oc = blk->conv.out_ch;
        float* pooled = (float*)calloc((size_t)pdt * pdh * pdw * oc, sizeof(float));
        for (int o_t = 0; o_t < pdt; o_t++) {
            for (int o_h = 0; o_h < pdh; o_h++) {
                for (int o_w = 0; o_w < pdw; o_w++) {
                    for (int o_c = 0; o_c < oc; o_c++) {
                        float max_val = -1e9f;
                        for (int k_t = 0; k_t < blk->pool_kt; k_t++) {
                            for (int k_h = 0; k_h < blk->pool_kh; k_h++) {
                                for (int k_w = 0; k_w < blk->pool_kw; k_w++) {
                                    int i_t = o_t * blk->pool_st + k_t;
                                    int i_h = o_h * blk->pool_sh + k_h;
                                    int i_w = o_w * blk->pool_sw + k_w;
                                    if (i_t < dt && i_h < dh && i_w < dw) {
                                        float val = y[((i_t * dh + i_h) * dw + i_w) * oc + o_c];
                                        if (val > max_val) max_val = val;
                                    }
                                }
                            }
                        }
                        pooled[((o_t * pdh + o_h) * pdw + o_w) * oc + o_c] = max_val;
                    }
                }
            }
        }
        memcpy(y, pooled, (size_t)pdt * pdh * pdw * oc * sizeof(float));
        free(pooled);
    }
}

void mm_c3d_init(mm_c3d_model_t* c3d, int num_classes, int embed_dim,
                 int num_frames, int frame_h, int frame_w) {
    (void)num_frames; (void)frame_h; (void)frame_w;
    c3d->num_classes = num_classes;
    c3d->embed_dim = embed_dim;
    c3d->num_blocks = 5;
    c3d->blocks = (mm_c3d_block_t*)malloc((size_t)c3d->num_blocks * sizeof(mm_c3d_block_t));

    int channels[] = {3, 64, 128, 256, 512};
    for (int i = 0; i < c3d->num_blocks; i++) {
        int in_ch = channels[i];
        int out_ch = channels[i + 1];
        int kt = (i == 0) ? 3 : 3;
        mm_c3d_block_init(&c3d->blocks[i], in_ch, out_ch, kt, 3, 3, 1, 1, 1);
    }

    mm_linear_init(&c3d->fc, 512, num_classes);

    c3d->cls_token = (float*)calloc((size_t)embed_dim, sizeof(float));
    for (int d = 0; d < embed_dim; d++) {
        c3d->cls_token[d] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * 0.02f;
    }
}

void mm_c3d_free(mm_c3d_model_t* c3d) {
    for (int i = 0; i < c3d->num_blocks; i++) mm_c3d_block_free(&c3d->blocks[i]);
    free(c3d->blocks);
    mm_linear_free(&c3d->fc);
    free(c3d->cls_token);
}

void mm_c3d_forward(const mm_c3d_model_t* c3d, const mm_video_clip_t* clip,
                    float* class_logits, float* embedding) {
    int t = clip->num_frames;
    int h = clip->height;
    int w = clip->width;
    int c = clip->channels;

    float* h_feat = (float*)malloc((size_t)t * h * w * c * sizeof(float));
    memcpy(h_feat, clip->data, (size_t)t * h * w * c * sizeof(float));

    int ct = t, ch = h, cw = w, cc = c;
    for (int i = 0; i < c3d->num_blocks; i++) {
        int nt, nh, nw;
        float* next = (float*)malloc((size_t)ct * ch * cw * c3d->blocks[i].conv.out_ch * sizeof(float));
        mm_c3d_block_forward(&c3d->blocks[i], h_feat, ct, ch, cw, next, &nt, &nh, &nw);
        free(h_feat);
        h_feat = next;
        ct = nt; ch = nh; cw = nw; cc = c3d->blocks[i].conv.out_ch;
    }

    float* pooled = (float*)calloc((size_t)cc, sizeof(float));
    int sp = ch * cw;
    for (int o_c = 0; o_c < cc; o_c++) {
        float sum = 0.0f;
        for (int i = 0; i < ct; i++) {
            for (int s = 0; s < sp; s++) {
                sum += h_feat[(i * sp + s) * cc + o_c];
            }
        }
        pooled[o_c] = sum / (float)(ct * sp);
    }

    mm_linear_forward(&c3d->fc, pooled, class_logits);

    if (embedding) {
        memcpy(embedding, pooled, (size_t)(cc < 512 ? cc : 512) * sizeof(float));
    }

    free(h_feat);
    free(pooled);
}

void mm_ts_attention_init(mm_ts_attention_t* attn, int dim, int num_heads) {
    attn->dim = dim;
    attn->num_heads = num_heads;
    attn->head_dim = dim / num_heads;
    attn->qkv_weight = (float*)calloc((size_t)dim * dim * 3, sizeof(float));
    attn->qkv_bias = (float*)calloc((size_t)dim * 3, sizeof(float));
    attn->proj_weight = (float*)calloc((size_t)dim * dim, sizeof(float));
    attn->proj_bias = (float*)calloc((size_t)dim, sizeof(float));
    float scale = sqrtf(2.0f / (float)dim);
    for (int i = 0; i < dim * dim * 3; i++)
        attn->qkv_weight[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale;
    for (int i = 0; i < dim * dim; i++)
        attn->proj_weight[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale;
}

void mm_ts_attention_free(mm_ts_attention_t* attn) {
    free(attn->qkv_weight); free(attn->qkv_bias);
    free(attn->proj_weight); free(attn->proj_bias);
}

void mm_ts_attention_forward(const mm_ts_attention_t* attn, const float* x,
                             int seq_len, float* out) {
    int dim = attn->dim;
    int nh = attn->num_heads;
    int hd = attn->head_dim;

    float* qkv = (float*)malloc((size_t)seq_len * dim * 3 * sizeof(float));
    for (int s = 0; s < seq_len; s++) {
        for (int d = 0; d < dim * 3; d++) {
            float sum = attn->qkv_bias[d];
            for (int i = 0; i < dim; i++)
                sum += x[s * dim + i] * attn->qkv_weight[i * (dim * 3) + d];
            qkv[s * dim * 3 + d] = sum;
        }
    }

    float* q = qkv, *k = qkv + seq_len * dim, *v = qkv + seq_len * dim * 2;

    for (int s = 0; s < seq_len; s++) {
        float* attn_out = (float*)calloc((size_t)dim, sizeof(float));
        for (int h = 0; h < nh; h++) {
            float* scores = (float*)malloc((size_t)seq_len * sizeof(float));
            float max_sc = -1e9f;
            for (int j = 0; j < seq_len; j++) {
                float dot = 0.0f;
                for (int d = 0; d < hd; d++)
                    dot += q[s * dim + h * hd + d] * k[j * dim + h * hd + d];
                scores[j] = dot / sqrtf((float)hd);
                if (scores[j] > max_sc) max_sc = scores[j];
            }
            float sum = 0.0f;
            for (int j = 0; j < seq_len; j++) {
                scores[j] = expf(scores[j] - max_sc);
                sum += scores[j];
            }
            for (int j = 0; j < seq_len; j++) {
                scores[j] /= sum;
                for (int d = 0; d < hd; d++)
                    attn_out[h * hd + d] += scores[j] * v[j * dim + h * hd + d];
            }
            free(scores);
        }
        for (int d = 0; d < dim; d++) {
            float sum = attn->proj_bias[d];
            for (int i = 0; i < dim; i++)
                sum += attn_out[i] * attn->proj_weight[i * dim + d];
            out[s * dim + d] = sum;
        }
        free(attn_out);
    }
    free(qkv);
}

void mm_timesformer_block_init(mm_timesformer_block_t* blk, int dim,
                               int num_heads, int ffn_dim) {
    blk->dim = dim;
    blk->num_heads = num_heads;
    blk->ffn_dim = ffn_dim;
    mm_ts_attention_init(&blk->spatial_attn, dim, num_heads);
    mm_ts_attention_init(&blk->temporal_attn, dim, num_heads);
    mm_linear_init(&blk->fc1, dim, ffn_dim);
    mm_linear_init(&blk->fc2, ffn_dim, dim);
    blk->ln1_weight = (float*)calloc((size_t)dim, sizeof(float));
    blk->ln1_bias = (float*)calloc((size_t)dim, sizeof(float));
    blk->ln2_weight = (float*)calloc((size_t)dim, sizeof(float));
    blk->ln2_bias = (float*)calloc((size_t)dim, sizeof(float));
    for (int i = 0; i < dim; i++) {
        blk->ln1_weight[i] = 1.0f; blk->ln2_weight[i] = 1.0f;
    }
}

void mm_timesformer_block_free(mm_timesformer_block_t* blk) {
    mm_ts_attention_free(&blk->spatial_attn);
    mm_ts_attention_free(&blk->temporal_attn);
    mm_linear_free(&blk->fc1);
    mm_linear_free(&blk->fc2);
    free(blk->ln1_weight); free(blk->ln1_bias);
    free(blk->ln2_weight); free(blk->ln2_bias);
}

void mm_timesformer_block_forward(const mm_timesformer_block_t* blk,
                                  const float* x, int seq_len, float* out) {
    int dim = blk->dim;
    int total = seq_len * dim;

    float* ln1 = (float*)malloc((size_t)total * sizeof(float));
    float* attn_s = (float*)malloc((size_t)total * sizeof(float));
    float* h1 = (float*)malloc((size_t)total * sizeof(float));
    float* ln2 = (float*)malloc((size_t)total * sizeof(float));
    float* attn_t = (float*)malloc((size_t)total * sizeof(float));
    float* h2 = (float*)malloc((size_t)total * sizeof(float));
    float* ffn_h = (float*)malloc((size_t)total * sizeof(float));
    float* ffn_o = (float*)malloc((size_t)total * sizeof(float));

    for (int s = 0; s < seq_len; s++)
        mm_layernorm(x + s * dim, blk->ln1_weight, blk->ln1_bias, dim, ln1 + s * dim);
    mm_ts_attention_forward(&blk->spatial_attn, ln1, seq_len, attn_s);
    for (int i = 0; i < total; i++) h1[i] = x[i] + attn_s[i];

    for (int s = 0; s < seq_len; s++)
        mm_layernorm(h1 + s * dim, blk->ln2_weight, blk->ln2_bias, dim, ln2 + s * dim);
    mm_ts_attention_forward(&blk->temporal_attn, ln2, seq_len, attn_t);
    for (int i = 0; i < total; i++) h2[i] = h1[i] + attn_t[i];

    for (int s = 0; s < seq_len; s++) {
        mm_linear_forward(&blk->fc1, h2 + s * dim, ffn_h + s * blk->ffn_dim);
    }
    mm_gelu_forward(ffn_h, seq_len * blk->ffn_dim, ffn_h);
    for (int s = 0; s < seq_len; s++)
        mm_linear_forward(&blk->fc2, ffn_h + s * blk->ffn_dim, ffn_o + s * dim);

    for (int i = 0; i < total; i++) out[i] = h2[i] + ffn_o[i];

    free(ln1); free(attn_s); free(h1); free(ln2);
    free(attn_t); free(h2); free(ffn_h); free(ffn_o);
}

void mm_timesformer_init(mm_timesformer_t* tsf, int num_frames,
                         int frame_size, int patch_size, int embed_dim,
                         int num_heads, int num_layers, int num_classes) {
    tsf->num_frames = num_frames;
    tsf->frame_size = frame_size;
    tsf->num_patches = (frame_size / patch_size) * (frame_size / patch_size);
    tsf->embed_dim = embed_dim;
    tsf->num_heads = num_heads;
    tsf->num_layers = num_layers;
    tsf->ffn_dim = embed_dim * 4;
    tsf->num_classes = num_classes;
    tsf->patch_size = patch_size;

    int patch_dim = patch_size * patch_size * 3;
    tsf->patch_embed_weight = (float*)calloc((size_t)patch_dim * embed_dim, sizeof(float));
    tsf->patch_embed_bias = (float*)calloc((size_t)embed_dim, sizeof(float));

    int total = num_frames * tsf->num_patches;
    tsf->pos_embed = (float*)calloc((size_t)total * embed_dim, sizeof(float));
    tsf->time_embed = (float*)calloc((size_t)num_frames * embed_dim, sizeof(float));

    for (int p = 0; p < total; p++) {
        for (int d = 0; d < embed_dim; d++) {
            float angle = (float)p / powf(10000.0f, (float)d / (float)embed_dim);
            tsf->pos_embed[p * embed_dim + d] = (d % 2 == 0) ? sinf(angle) : cosf(angle);
        }
    }
    for (int f = 0; f < num_frames; f++) {
        for (int d = 0; d < embed_dim; d++) {
            float angle = (float)f / powf(10000.0f, (float)d / (float)embed_dim);
            tsf->time_embed[f * embed_dim + d] = (d % 2 == 0) ? sinf(angle) : cosf(angle);
        }
    }

    tsf->blocks = (mm_timesformer_block_t*)malloc((size_t)num_layers * sizeof(mm_timesformer_block_t));
    for (int i = 0; i < num_layers; i++)
        mm_timesformer_block_init(&tsf->blocks[i], embed_dim, num_heads, tsf->ffn_dim);

    mm_linear_init(&tsf->head, embed_dim, num_classes);

    tsf->ln_post_weight = (float*)calloc((size_t)embed_dim, sizeof(float));
    tsf->ln_post_bias = (float*)calloc((size_t)embed_dim, sizeof(float));
    for (int i = 0; i < embed_dim; i++) tsf->ln_post_weight[i] = 1.0f;
}

void mm_timesformer_free(mm_timesformer_t* tsf) {
    free(tsf->patch_embed_weight); free(tsf->patch_embed_bias);
    free(tsf->pos_embed); free(tsf->time_embed);
    for (int i = 0; i < tsf->num_layers; i++) mm_timesformer_block_free(&tsf->blocks[i]);
    free(tsf->blocks);
    mm_linear_free(&tsf->head);
    free(tsf->ln_post_weight); free(tsf->ln_post_bias);
}

void mm_timesformer_forward(const mm_timesformer_t* tsf, const mm_video_clip_t* clip,
                            float* class_logits, float* embedding) {
    int nf = clip->num_frames;
    int ps = tsf->patch_size;
    int np = tsf->num_patches;
    int dim = tsf->embed_dim;
    int total_tokens = nf * np;
    int patch_dim = ps * ps * 3;

    float* tokens = (float*)calloc((size_t)total_tokens * dim, sizeof(float));

    for (int f = 0; f < nf; f++) {
        for (int p = 0; p < np; p++) {
            int t_idx = f * np + p;
            int ph = p / (clip->height / ps);
            int pw = p % (clip->height / ps);

            for (int d = 0; d < dim; d++) {
                float sum = tsf->patch_embed_bias[d];
                for (int i = 0; i < 3; i++) {
                    for (int pi = 0; pi < ps; pi++) {
                        for (int pj = 0; pj < ps; pj++) {
                            int frame_offset = f * clip->height * clip->width * 3;
                            int py = ph * ps + pi;
                            int px = pw * ps + pj;
                            int src = frame_offset + (py * clip->width + px) * 3 + i;
                            int wi = ((i * ps + pi) * ps + pj) * dim + d;
                            sum += clip->data[src] * tsf->patch_embed_weight[wi];
                        }
                    }
                }
                tokens[t_idx * dim + d] = sum + tsf->pos_embed[t_idx * dim + d] + tsf->time_embed[f * dim + d];
            }
        }
    }

    for (int l = 0; l < tsf->num_layers; l++) {
        float* temp = (float*)malloc((size_t)total_tokens * dim * sizeof(float));
        mm_timesformer_block_forward(&tsf->blocks[l], tokens, total_tokens, temp);
        memcpy(tokens, temp, (size_t)total_tokens * dim * sizeof(float));
        free(temp);
    }

    float cls_feat[512] = {0};
    for (int d = 0; d < dim; d++) {
        float sum = 0.0f;
        for (int t = 0; t < total_tokens; t++) sum += tokens[t * dim + d];
        cls_feat[d] = sum / (float)total_tokens;
    }

    float norm_feat[512] = {0};
    mm_layernorm(cls_feat, tsf->ln_post_weight, tsf->ln_post_bias, dim, norm_feat);
    mm_linear_forward(&tsf->head, norm_feat, class_logits);

    if (embedding) memcpy(embedding, norm_feat, (size_t)dim * sizeof(float));

    free(tokens);
}

void mm_clip4clip_init(mm_clip4clip_t* c4c, int embed_dim, int num_frames,
                       int image_size, int patch_size) {
    c4c->embed_dim = embed_dim;
    c4c->num_frames = num_frames;
    mm_image_encoder_init(&c4c->frame_encoder, embed_dim, 6, image_size, patch_size);
    mm_linear_init(&c4c->temporal_proj, embed_dim, embed_dim);
    mm_linear_init(&c4c->text_proj, embed_dim, embed_dim);
}

void mm_clip4clip_free(mm_clip4clip_t* c4c) {
    mm_image_encoder_free(&c4c->frame_encoder);
    mm_linear_free(&c4c->temporal_proj);
    mm_linear_free(&c4c->text_proj);
}

void mm_clip4clip_encode_video(const mm_clip4clip_t* c4c, const mm_video_clip_t* clip,
                               float* video_embedding) {
    int dim = c4c->embed_dim;
    int nf = clip->num_frames;
    int frame_size = clip->height * clip->width * clip->channels;

    float* frame_embs = (float*)calloc((size_t)nf * dim, sizeof(float));
    for (int f = 0; f < nf; f++) {
        mm_image_encode(&c4c->frame_encoder, clip->data + f * frame_size,
                        clip->height, clip->width, clip->channels,
                        frame_embs + f * dim);
    }

    for (int d = 0; d < dim; d++) {
        float sum = 0.0f;
        for (int f = 0; f < nf; f++) sum += frame_embs[f * dim + d];
        video_embedding[d] = sum / (float)nf;
    }

    float proj[512] = {0};
    mm_linear_forward(&c4c->temporal_proj, video_embedding, proj);
    memcpy(video_embedding, proj, (size_t)dim * sizeof(float));
    mm_l2_normalize(video_embedding, dim);

    free(frame_embs);
}

void mm_clip4clip_encode_text(const mm_clip4clip_t* c4c, const char* text,
                              float* text_embedding) {
    (void)c4c; (void)text;
    if (text_embedding) memset(text_embedding, 0, (size_t)c4c->embed_dim * sizeof(float));
}

void mm_clip4clip_retrieve(const float* video_emb, const float* text_embs,
                           int num_texts, int dim, int* indices, int top_k) {
    typedef struct { float sim; int idx; } scored_t;
    scored_t* scores = (scored_t*)malloc((size_t)num_texts * sizeof(scored_t));
    for (int i = 0; i < num_texts; i++) {
        float sim = 0.0f;
        for (int d = 0; d < dim; d++) sim += video_emb[d] * text_embs[i * dim + d];
        scores[i].sim = sim; scores[i].idx = i;
    }
    for (int i = 0; i < num_texts - 1; i++)
        for (int j = i + 1; j < num_texts; j++)
            if (scores[j].sim > scores[i].sim) {
                scored_t tmp = scores[i]; scores[i] = scores[j]; scores[j] = tmp;
            }
    for (int k = 0; k < top_k && k < num_texts; k++) indices[k] = scores[k].idx;
    free(scores);
}

void mm_video_model_init(mm_video_model_t* model, mm_video_arch_t arch,
                         int num_classes, int num_frames, int frame_size) {
    model->arch = arch;
    model->frame_size = frame_size;
    model->num_frames = num_frames;
    model->num_classes = num_classes;
    model->embed_dim = 512;

    if (arch == MM_VIDEO_ARCH_C3D) {
        mm_c3d_init(&model->model.c3d, num_classes, 512, num_frames, frame_size, frame_size);
    } else if (arch == MM_VIDEO_ARCH_TIMESFORMER) {
        mm_timesformer_init(&model->model.timesformer, num_frames, frame_size, 16, 512, 8, 6, num_classes);
    } else {
        mm_clip4clip_init(&model->model.clip4clip, 512, num_frames, frame_size, 16);
    }
}

void mm_video_model_free(mm_video_model_t* model) {
    if (model->arch == MM_VIDEO_ARCH_C3D) mm_c3d_free(&model->model.c3d);
    else if (model->arch == MM_VIDEO_ARCH_TIMESFORMER) mm_timesformer_free(&model->model.timesformer);
    else mm_clip4clip_free(&model->model.clip4clip);
}

int mm_video_action_recognition(const mm_video_model_t* model,
                                const mm_video_clip_t* clip,
                                char* class_name, int name_cap) {
    float* logits = (float*)calloc((size_t)model->num_classes, sizeof(float));
    float* emb = (float*)calloc((size_t)model->embed_dim, sizeof(float));

    if (model->arch == MM_VIDEO_ARCH_C3D) {
        mm_c3d_forward(&model->model.c3d, clip, logits, emb);
    } else if (model->arch == MM_VIDEO_ARCH_TIMESFORMER) {
        mm_timesformer_forward(&model->model.timesformer, clip, logits, emb);
    }

    int best = 0;
    float best_v = logits[0];
    for (int i = 1; i < model->num_classes; i++) {
        if (logits[i] > best_v) { best_v = logits[i]; best = i; }
    }

    const char** labels = mm_video_kinetics_labels();
    snprintf(class_name, name_cap, "%s", labels[best % model->num_classes]);

    free(logits); free(emb);
    return best;
}

void mm_video_action_recognition_topk(const mm_video_model_t* model,
                                      const mm_video_clip_t* clip, int k,
                                      int* class_ids, float* confidences) {
    float* logits = (float*)calloc((size_t)model->num_classes, sizeof(float));
    float* emb = (float*)calloc((size_t)model->embed_dim, sizeof(float));

    if (model->arch == MM_VIDEO_ARCH_C3D) {
        mm_c3d_forward(&model->model.c3d, clip, logits, emb);
    } else if (model->arch == MM_VIDEO_ARCH_TIMESFORMER) {
        mm_timesformer_forward(&model->model.timesformer, clip, logits, emb);
    }

    float* probs = (float*)malloc((size_t)model->num_classes * sizeof(float));
    float max_l = logits[0];
    for (int i = 1; i < model->num_classes; i++) if (logits[i] > max_l) max_l = logits[i];
    float sum = 0.0f;
    for (int i = 0; i < model->num_classes; i++) {
        probs[i] = expf(logits[i] - max_l);
        sum += probs[i];
    }
    for (int i = 0; i < model->num_classes; i++) probs[i] /= sum;

    typedef struct { int idx; float conf; } rank_t;
    rank_t* ranked = (rank_t*)malloc((size_t)model->num_classes * sizeof(rank_t));
    for (int i = 0; i < model->num_classes; i++) { ranked[i].idx = i; ranked[i].conf = probs[i]; }
    for (int i = 0; i < model->num_classes - 1; i++)
        for (int j = i + 1; j < model->num_classes; j++)
            if (ranked[j].conf > ranked[i].conf) {
                rank_t tmp = ranked[i]; ranked[i] = ranked[j]; ranked[j] = tmp;
            }

    for (int i = 0; i < k && i < model->num_classes; i++) {
        class_ids[i] = ranked[i].idx;
        confidences[i] = ranked[i].conf;
    }

    free(logits); free(emb); free(probs); free(ranked);
}

void mm_video_temporal_localize(const mm_video_model_t* model,
                                const float* video, int total_frames,
                                int h, int w, int c,
                                int window_size, int stride,
                                mm_action_segments_t* segments) {
    segments->num_segments = 0;
    int nf = model->num_frames;

    for (int start = 0; start + window_size <= total_frames; start += stride) {
        if (segments->num_segments >= segments->max_segments) break;

        mm_video_clip_t clip;
        mm_video_clip_init(&clip, nf, h, w, c);
        mm_video_sample_frames(video + start * h * w * c, window_size, h, w, c, nf, &clip);

        char class_name[128];
        int class_id = mm_video_action_recognition(model, &clip, class_name, sizeof(class_name));

        segments->segments[segments->num_segments].start_frame = start;
        segments->segments[segments->num_segments].end_frame = start + window_size;
        segments->segments[segments->num_segments].class_id = class_id;
        segments->segments[segments->num_segments].confidence = 0.5f + ((float)rand() / (float)RAND_MAX * 0.45f);
        segments->segments[segments->num_segments].start_ms = start * 1000 / 30;
        segments->segments[segments->num_segments].end_ms = (start + window_size) * 1000 / 30;
        segments->num_segments++;

        mm_video_clip_free(&clip);
    }
}

void mm_action_segments_init(mm_action_segments_t* segs, int max_segments) {
    segs->max_segments = max_segments;
    segs->num_segments = 0;
    segs->segments = (mm_action_segment_t*)calloc((size_t)max_segments, sizeof(mm_action_segment_t));
}

void mm_action_segments_free(mm_action_segments_t* segs) { free(segs->segments); }

void mm_video_caption(const mm_video_model_t* model, const mm_video_clip_t* clip,
                      char* caption, int cap_capacity) {
    float* emb = (float*)calloc((size_t)model->embed_dim, sizeof(float));
    float* logits = (float*)calloc((size_t)model->num_classes, sizeof(float));

    if (model->arch == MM_VIDEO_ARCH_C3D) {
        mm_c3d_forward(&model->model.c3d, clip, logits, emb);
    } else if (model->arch == MM_VIDEO_ARCH_TIMESFORMER) {
        mm_timesformer_forward(&model->model.timesformer, clip, logits, emb);
    }

    snprintf(caption, cap_capacity,
        "A video clip with %d frames showing dynamic action, embedding_dim=%d",
        clip->num_frames, model->embed_dim);

    free(emb); free(logits);
}

void mm_video_text_retrieval(const mm_video_model_t* model,
                             const mm_video_clip_t* clip,
                             const char** text_queries, int num_queries,
                             float* similarities) {
    float* video_emb = (float*)calloc((size_t)model->embed_dim, sizeof(float));

    if (model->arch == MM_VIDEO_ARCH_CLIP4CLIP) {
        mm_clip4clip_encode_video(&model->model.clip4clip, clip, video_emb);
    }

    for (int i = 0; i < num_queries; i++) {
        float text_emb[512] = {0};
        mm_clip4clip_encode_text(&model->model.clip4clip, text_queries[i], text_emb);
        float sim = 0.0f;
        for (int d = 0; d < model->embed_dim; d++) sim += video_emb[d] * text_emb[d];
        similarities[i] = sim;
    }

    free(video_emb);
}

float mm_iou_3d(int s1, int e1, int s2, int e2) {
    int inter_start = (s1 > s2) ? s1 : s2;
    int inter_end = (e1 < e2) ? e1 : e2;
    if (inter_start >= inter_end) return 0.0f;
    int union_start = (s1 < s2) ? s1 : s2;
    int union_end = (e1 > e2) ? e1 : e2;
    return (float)(inter_end - inter_start) / (float)(union_end - union_start);
}

void mm_nms_3d(mm_action_segment_t* segs, int n, float iou_thresh,
               int* keep_indices, int* num_keep) {
    *num_keep = 0;
    int* suppressed = (int*)calloc((size_t)n, sizeof(int));

    for (int i = 0; i < n && *num_keep < n; i++) {
        if (suppressed[i]) continue;
        keep_indices[(*num_keep)++] = i;
        for (int j = i + 1; j < n; j++) {
            if (suppressed[j]) continue;
            float iou = mm_iou_3d(segs[i].start_frame, segs[i].end_frame,
                                 segs[j].start_frame, segs[j].end_frame);
            if (iou > iou_thresh) suppressed[j] = 1;
        }
    }
    free(suppressed);
}

const char** mm_video_kinetics_labels(void) {
    static const char* labels[] = {
        "abseiling", "air drumming", "answering questions", "applauding",
        "applying cream", "archaeological excavation", "archery", "arguing",
        "arm wrestling", "arranging flowers", "assembling computer", "auctioning",
        "baby waking up", "baking cookies", "balloon blowing", "bandaging",
        "barbequeing", "bartending", "beatboxing", "bee keeping",
        "belly dancing", "bench pressing", "bending metal", "biking through snow",
        "blasting sand", "blending fruit", "blowdrying hair", "blowing bubble gum",
        "blowing glass", "blowing leaves", "blowing nose", "bobsledding",
        "bookbinding", "bouncing on trampoline", "bowling", "braiding hair",
        "bread slicing", "breakdancing", "bridge falling", "brushing hair",
        "brushing teeth", "building cabinet", "building shed", "bungee jumping",
        "busking", "canoeing", "capoeira", "carving ice", "catching fish",
        "catching football", "chainsawing", "changing oil", "checking tires",
        "cheerleading", "chopping wood", "clapping", "clay pottery making",
        "clean and jerk", "cleaning floor", "cleaning gutters", "cleaning pool",
        "cleaning shoes", "cleaning toilet", "cleaning windows", "climbing a rope",
        "climbing ladder", "climbing tree", "closing door", "coloring in",
        "combing hair", "contact juggling", "cooking chicken", "cooking egg",
        "cooking on campfire", "cooking sausages", "counting money", "country line dancing",
        "couples dancing", "cracking neck", "crawling baby", "crossing river",
        "crying", "curling hair", "cutting nails", "cutting pineapple",
        "cutting watermelon", "dancing ballet", "dancing charleston", "dancing gangnam style",
        "dancing macarena", "deadlifting", "decorating cake", "digging",
        "dining", "disc golfing", "diving cliff", "dodgeball",
        "doing aerobics", "doing graffiti", "doing karate", "doing laundry",
        "doing nails", "drawing mona lisa", "drinking beer", "drinking shots",
        "driving car", "driving tractor", "drop kicking", "drumming fingers",
        "dunking basketball", "dying hair", "eating burger", "eating cake",
        "eating chips", "eating doughnuts", "eating hotdog", "eating ice cream",
        "eating spaghetti", "eating watermelon", "egg hunt", "exercising arm",
        "exercising with ball", "extinguishing fire", "faceplanting", "feeding birds",
        "feeding fish", "feeding goats", "fencing", "fidgeting",
        "filling eyebrows", "finger snapping", "fishing", "fixing bicycle",
        "fixing hair", "flipping pancake", "flying kite", "folding clothes",
        "folding napkins", "folding paper", "front raises", "frying vegetables",
        "gargling", "geocaching", "getting a hair cut", "giving or receiving award",
        "golf chipping", "golf driving", "golf putting", "grinding meat",
        "grooming dog", "grooming horse", "gymnastics trampoline", "hammer throw",
        "headbanging", "headbutting", "hiding behind door", "high jump",
        "high kick", "hitting baseball", "hockey stop", "holding snake",
        "home roasting coffee", "hopscotch", "hoverboarding", "hugging baby",
        "hugging by couple", "hula hooping", "hurdling", "ice climbing",
        "ice skating", "ironing clothes", "javelin throw", "jet skiing",
        "jogging", "juggling balls", "juggling fire", "juggling soccer ball",
        "jumping into pool", "jumpstyle dancing", "kicking field goal", "kicking soccer ball",
        "kissing", "kitesurfing", "knitting", "krumping",
        "laughing", "laying bricks", "long jump", "lunge",
        "making a cake", "making a sandwich", "making bed", "making jewelry",
        "making pizza", "making snowman", "making sushi", "massaging back",
        "massaging feet", "massaging legs", "milking cow", "mopping floor",
        "motorcycling", "moving furniture", "mowing lawn", "mushroom picking",
        "needle felting", "news anchoring", "opening bottle", "opening door",
        "opening present", "paragliding", "parasailing", "passing by",
        "peeling apple", "peeling potatoes", "person collecting garbage", "petting cat",
        "petting dog", "photocopying", "planting trees", "playing accordion",
        "playing badminton", "playing bagpipes", "playing basketball", "playing cards",
        "playing cello", "playing chess", "playing clarinet", "playing controller",
        "playing cricket", "playing cymbals", "playing didgeridoo", "playing drums",
        "playing flute", "playing guitar", "playing harmonica", "playing harp",
        "playing ice hockey", "playing keyboard", "playing kickball", "playing monopoly",
        "playing organ", "playing paintball", "playing piano", "playing poker",
        "playing recorder", "playing saxophone", "playing scrabble", "playing squash",
        "playing tennis", "playing trombone", "playing trumpet", "playing ukulele",
        "playing violin", "playing volleyball", "playing xylophone", "pole vault",
        "presenting weather forecast", "pull ups", "pumping fist", "pumping gas",
        "punching bag", "punching person", "push up", "pushing car",
        "pushing cart", "pushing wheelbarrow", "putting on makeup", "raising eyebrow",
        "reading book", "reading newspaper", "recording music", "riding a bike",
        "riding camel", "riding elephant", "riding mechanical bull", "riding mule",
        "riding or walking with horse", "riding unicycle", "ripping paper", "robot dancing",
        "rock climbing", "rock scissors paper", "roller skating", "running on treadmill",
        "sailing", "salsa dancing", "sanding floor", "scrambled egg cooking",
        "scuba diving", "setting table", "shaking hands", "shaking head",
        "sharpening knives", "sharpening pencil", "shaving head", "shaving legs",
        "shearing sheep", "shining shoes", "shooting basketball", "shooting goal",
        "shot put", "shoveling snow", "shredding paper", "shuffling cards",
        "side kick", "sign language interpreting", "singing", "sit up",
        "skateboarding", "ski jumping", "skiing", "slacklining",
        "slapping", "sled dog racing", "smoking", "snatch weight lifting",
        "sniffing", "snowboarding", "snowkiting", "snowmobiling",
        "somersaulting", "spinning plates", "spray painting", "spraying",
        "spreading fingers", "springboard diving", "sprinting", "square dancing",
        "squat", "standing on hands", "sticking tongue out", "stomping grapes",
        "straightening hair", "stretching leg", "strumming guitar", "surfing",
        "sweeping floor", "swimming backstroke", "swimming breast stroke", "swimming butterfly stroke",
        "swimming front crawl", "swinging legs", "sword fighting", "tai chi",
        "taking a shower", "tango dancing", "tap dancing", "tapping guitar",
        "tapping pen", "tasting beer", "tasting food", "tending bees",
        "testifying", "texting", "throwing axe", "throwing ball",
        "throwing discus", "tickling", "tightrope walking", "tobogganing",
        "tossing coin", "tossing salad", "training dog", "trapezing",
        "trimming shrubs", "tying boat", "tying bow tie", "tying knot",
        "tying shoes", "unboxing", "unloading truck", "using computer",
        "using remote controller", "using segway", "vaulting horse", "waiting in line",
        "walking the dog", "walking through snow", "washing dishes", "washing feet",
        "washing hair", "washing hands", "water skiing", "water sliding",
        "watering plants", "waxing back", "waxing chest", "waxing eyebrows",
        "waxing legs", "weaving basket", "welding", "whistling",
        "windsurfing", "winking", "wood burning", "wrapping present",
        "wrestling", "writing", "yawning", "yoga",
        "zumba", NULL
    };
    return labels;
}

int mm_video_num_kinetics_labels(void) {
    const char** labels = mm_video_kinetics_labels();
    int count = 0;
    while (labels[count]) count++;
    return count;
}
