#include "clip_contrastive.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

float mm_l2_norm(const float* x, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += x[i] * x[i];
    return sqrtf(sum + 1e-12f);
}

void mm_l2_normalize(float* x, int n) {
    float norm = mm_l2_norm(x, n);
    float inv = 1.0f / norm;
    for (int i = 0; i < n; i++) x[i] *= inv;
}

void mm_cosine_sim_matrix(const float* img_embs, const float* txt_embs,
                          int batch_size, int dim, float* sim_matrix) {
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < batch_size; j++) {
            float dot = 0.0f;
            for (int d = 0; d < dim; d++) {
                dot += img_embs[i * dim + d] * txt_embs[j * dim + d];
            }
            sim_matrix[i * batch_size + j] = dot;
        }
    }
}

float mm_infonce_loss(const float* sim_matrix, int batch_size) {
    float loss = 0.0f;
    float scale = logf(14.2857f);

    for (int i = 0; i < batch_size; i++) {
        float row_max = sim_matrix[i * batch_size];
        for (int j = 1; j < batch_size; j++) {
            if (sim_matrix[i * batch_size + j] > row_max)
                row_max = sim_matrix[i * batch_size + j];
        }

        float sum_exp = 0.0f;
        for (int j = 0; j < batch_size; j++) {
            float v = sim_matrix[i * batch_size + j] - row_max;
            sum_exp += expf(v);
        }

        float pos = sim_matrix[i * batch_size + i] - row_max;
        loss += -pos + logf(sum_exp + 1e-12f);

        float col_max = sim_matrix[i];
        for (int j = 1; j < batch_size; j++) {
            if (sim_matrix[j * batch_size + i] > col_max)
                col_max = sim_matrix[j * batch_size + i];
        }

        float sum_exp2 = 0.0f;
        for (int j = 0; j < batch_size; j++) {
            float v = sim_matrix[j * batch_size + i] - col_max;
            sum_exp2 += expf(v);
        }

        float pos2 = sim_matrix[i * batch_size + i] - col_max;
        loss += -pos2 + logf(sum_exp2 + 1e-12f);
    }

    return loss * 0.5f / (float)batch_size;
}

void mm_infonce_grad(const float* sim_matrix, int batch_size, float* img_grad, float* txt_grad) {
    int dim = batch_size;
    (void)dim;

    for (int i = 0; i < batch_size; i++) {
        float row_max = sim_matrix[i * batch_size];
        for (int j = 1; j < batch_size; j++)
            if (sim_matrix[i * batch_size + j] > row_max)
                row_max = sim_matrix[i * batch_size + j];

        float* softmax_row = (float*)malloc(batch_size * sizeof(float));
        float sum = 0.0f;
        for (int j = 0; j < batch_size; j++) {
            softmax_row[j] = expf(sim_matrix[i * batch_size + j] - row_max);
            sum += softmax_row[j];
        }
        for (int j = 0; j < batch_size; j++) {
            softmax_row[j] /= sum;
            float grad = softmax_row[j] - (i == j ? 1.0f : 0.0f);
            img_grad[i * batch_size + j] = grad * 0.5f;
        }
        free(softmax_row);
    }

    for (int i = 0; i < batch_size; i++) {
        float col_max = sim_matrix[i];
        for (int j = 1; j < batch_size; j++)
            if (sim_matrix[j * batch_size + i] > col_max)
                col_max = sim_matrix[j * batch_size + i];

        float* softmax_col = (float*)malloc(batch_size * sizeof(float));
        float sum = 0.0f;
        for (int j = 0; j < batch_size; j++) {
            softmax_col[j] = expf(sim_matrix[j * batch_size + i] - col_max);
            sum += softmax_col[j];
        }
        for (int j = 0; j < batch_size; j++) {
            softmax_col[j] /= sum;
            float grad = softmax_col[j] - (i == j ? 1.0f : 0.0f);
            txt_grad[i * batch_size + j] = grad * 0.5f;
        }
        free(softmax_col);
    }
}

void mm_text_tokenize(const char* text, int* token_ids, int* num_tokens) {
    int count = 0;
    const char* p = text;
    const char* vocab_words[] = {
        "a", "photo", "of", "the", "is", "to", "in", "and", "that", "it",
        "with", "for", "on", "as", "this", "by", "from", "at", "or", "an",
        "cat", "dog", "bird", "fish", "car", "tree", "house", "person", "food", "sky",
        "red", "blue", "green", "large", "small", "beautiful", "ugly", "fast", "slow", "happy",
        "image", "picture", "view", "scene", "landscape", "portrait", "closeup", "drawing", "painting", NULL
    };
    int vocab_count = 0;
    while (vocab_words[vocab_count]) vocab_count++;

    while (*p && count < MM_CLIP_MAX_TEXT_LEN - 2) {
        while (*p == ' ' || *p == '\t' || *p == '\n') p++;
        if (!*p) break;

        const char* start = p;
        while (*p && *p != ' ' && *p != '\t' && *p != '\n') p++;

        int word_len = (int)(p - start);
        char word[128] = {0};
        for (int k = 0; k < word_len && k < 127; k++) word[k] = start[k];

        int found = -1;
        for (int v = 0; v < vocab_count; v++) {
            if (strcmp(word, vocab_words[v]) == 0) { found = v + 49406; break; }
        }
        if (found < 0) {
            unsigned hash = 5381;
            for (int k = 0; k < word_len; k++) hash = ((hash << 5) + hash) + (unsigned char)word[k];
            found = (int)(hash % 100) + 49456;
        }
        token_ids[count++] = found;
    }
    *num_tokens = count;
}

void mm_text_encode(const mm_text_encoder_t* enc, const int* token_ids, int num_tokens,
                    float* embedding) {
    int dim = enc->embed_dim;
    int seq_len = num_tokens;
    float* x = (float*)calloc((size_t)seq_len * dim, sizeof(float));

    for (int i = 0; i < seq_len; i++) {
        int tid = token_ids[i] % enc->vocab_size;
        for (int d = 0; d < dim; d++) {
            x[i * dim + d] = enc->token_embed.weight[tid * dim + d];
        }
        if (enc->pos_embed) {
            for (int d = 0; d < dim; d++) {
                x[i * dim + d] += enc->pos_embed[i * dim + d];
            }
        }
    }

    for (int l = 0; l < enc->num_layers; l++) {
        mm_transformer_block_forward(&enc->blocks[l], x, seq_len, x);
    }

    float* pooled = (float*)malloc((size_t)dim * sizeof(float));
    for (int d = 0; d < dim; d++) {
        float sum = 0.0f;
        for (int i = 0; i < seq_len; i++) sum += x[i * dim + d];
        pooled[d] = sum / (float)seq_len;
    }

    if (enc->final_ln_bias.data == NULL) {
        memcpy(embedding, pooled, (size_t)dim * sizeof(float));
        free(pooled);
        free(x);
        return;
    }

    for (int d = 0; d < dim; d++) {
        float s = 0.0f;
        for (int i = 0; i < seq_len; i++) {
            float v = x[i * dim + d];
            s += v * v;
        }
        float mean_sq = s / (float)seq_len;
        float inv = 1.0f / sqrtf(mean_sq + 1e-5f);
        pooled[d] = pooled[d] * inv * enc->final_ln_weight.weight[d]
                    + enc->final_ln_weight.bias[d];
    }

    if (enc->proj) {
        float temp[512] = {0};
        for (int d = 0; d < dim; d++) {
            for (int j = 0; j < dim; j++) {
                temp[d] += pooled[j] * enc->proj[j * dim + d];
            }
        }
        memcpy(pooled, temp, (size_t)dim * sizeof(float));
    }

    memcpy(embedding, pooled, (size_t)dim * sizeof(float));
    mm_l2_normalize(embedding, dim);
    free(pooled);
    free(x);
}

void mm_image_patchify(const float* image, int h, int w, int c,
                       int patch_size, float* patches) {
    int ph = h / patch_size;
    int pw = w / patch_size;
    int patch_dim = patch_size * patch_size * c;
    for (int i = 0; i < ph; i++) {
        for (int j = 0; j < pw; j++) {
            int pidx = i * pw + j;
            for (int ci = 0; ci < patch_size; ci++) {
                for (int cj = 0; cj < patch_size; cj++) {
                    for (int ch = 0; ch < c; ch++) {
                        int img_idx = ((i * patch_size + ci) * w + (j * patch_size + cj)) * c + ch;
                        int pat_idx = (ci * patch_size + cj) * c + ch;
                        patches[pidx * patch_dim + pat_idx] = image[img_idx];
                    }
                }
            }
        }
    }
}

void mm_image_encode(const mm_image_encoder_t* enc, const float* image,
                     int h, int w, int c, float* embedding) {
    int ph = h / enc->patch_size;
    int pw = w / enc->patch_size;
    int np = ph * pw;
    int patch_dim = enc->patch_size * enc->patch_size * c;
    int dim = enc->embed_dim;

    float* patches = (float*)calloc((size_t)np * patch_dim, sizeof(float));
    mm_image_patchify(image, h, w, c, enc->patch_size, patches);

    int total_tokens = np + 1;
    float* x = (float*)calloc((size_t)total_tokens * dim, sizeof(float));

    for (int d = 0; d < dim; d++) {
        if (enc->class_token) x[d] = enc->class_token[d];
    }

    for (int p = 0; p < np; p++) {
        for (int d = 0; d < dim; d++) {
            float sum = 0.0f;
            for (int pd = 0; pd < patch_dim; pd++) {
                sum += patches[p * patch_dim + pd] * enc->patch_proj.weight[pd * dim + d];
            }
            x[(p + 1) * dim + d] = sum + enc->patch_proj.bias[d];
        }
    }

    if (enc->pos_embed) {
        for (int i = 0; i < total_tokens; i++) {
            for (int d = 0; d < dim; d++) {
                x[i * dim + d] += enc->pos_embed[i * dim + d];
            }
        }
    }

    for (int l = 0; l < enc->num_layers; l++) {
        mm_transformer_block_forward(&enc->blocks[l], x, total_tokens, x);
    }

    for (int d = 0; d < dim; d++) embedding[d] = x[d];

    if (enc->proj) {
        float temp[512] = {0};
        for (int d = 0; d < dim; d++) {
            for (int j = 0; j < dim; j++) {
                temp[d] += embedding[j] * enc->proj[j * dim + d];
            }
        }
        memcpy(embedding, temp, (size_t)dim * sizeof(float));
    }

    mm_l2_normalize(embedding, dim);
    free(patches);
    free(x);
}

void mm_clip_init(mm_clip_model_t* model, int embed_dim, int num_layers) {
    model->embed_dim = embed_dim;
    model->temperature = 0.07f;
    model->use_fp16 = 0;

    mm_image_encoder_init(&model->image_encoder, embed_dim, num_layers,
                          MM_CLIP_IMAGE_SIZE, MM_CLIP_PATCH_SIZE);
    mm_text_encoder_init(&model->text_encoder, embed_dim, num_layers,
                         MM_CLIP_VOCAB_SIZE, MM_CLIP_MAX_TEXT_LEN);

    model->logit_scale = (float*)malloc(sizeof(float));
    *model->logit_scale = logf(1.0f / model->temperature);
}

void mm_clip_free(mm_clip_model_t* model) {
    mm_image_encoder_free(&model->image_encoder);
    mm_text_encoder_free(&model->text_encoder);
    free(model->logit_scale);
}

void mm_clip_encode_image(const mm_clip_model_t* model, const float* image,
                          int h, int w, int c, float* embedding) {
    mm_image_encode(&model->image_encoder, image, h, w, c, embedding);
}

void mm_clip_encode_text(const mm_clip_model_t* model, const char* text,
                         float* embedding) {
    int token_ids[MM_CLIP_MAX_TEXT_LEN];
    int num_tokens;
    mm_text_tokenize(text, token_ids, &num_tokens);
    mm_text_encode(&model->text_encoder, token_ids, num_tokens, embedding);
}

int mm_clip_zeroshot(const float* image_emb, const char** class_names,
                      int num_classes, int top_k) {
    (void)top_k;
    float best_sim = -2.0f;
    int best_idx = 0;

    for (int c = 0; c < num_classes; c++) {
        char prompt[256];
        snprintf(prompt, sizeof(prompt), "a photo of %s", class_names[c]);

        int tokens[MM_CLIP_MAX_TEXT_LEN];
        int nt;
        mm_text_tokenize(prompt, tokens, &nt);

        float txt_emb[MM_CLIP_EMBED_DIM] = {0};
        int dim = MM_CLIP_EMBED_DIM;
        for (int d = 0; d < dim; d++) {
            float sum = 0.0f;
            for (int i = 0; i < nt; i++) {
                sum += (float)(tokens[i] % dim) * 0.01f;
            }
            txt_emb[d] = sum / (float)(nt + 1);
        }
        mm_l2_normalize(txt_emb, dim);

        float sim = 0.0f;
        for (int d = 0; d < dim; d++) {
            sim += image_emb[d] * txt_emb[d];
        }

        if (sim > best_sim) {
            best_sim = sim;
            best_idx = c;
        }
    }
    return best_idx;
}

void mm_clip_retrieve(const float* query_emb, const float* gallery_embs,
                      int gallery_size, int dim, int* indices, int top_k) {
    typedef struct { float sim; int idx; } scored_t;
    scored_t* scores = (scored_t*)malloc((size_t)gallery_size * sizeof(scored_t));

    for (int i = 0; i < gallery_size; i++) {
        float sim = 0.0f;
        for (int d = 0; d < dim; d++) {
            sim += query_emb[d] * gallery_embs[i * dim + d];
        }
        scores[i].sim = sim;
        scores[i].idx = i;
    }

    for (int i = 0; i < gallery_size - 1; i++) {
        for (int j = i + 1; j < gallery_size; j++) {
            if (scores[j].sim > scores[i].sim) {
                scored_t tmp = scores[i];
                scores[i] = scores[j];
                scores[j] = tmp;
            }
        }
    }

    for (int k = 0; k < top_k && k < gallery_size; k++) {
        indices[k] = scores[k].idx;
    }
    free(scores);
}

void mm_clip_train_step(mm_clip_model_t* model,
                        const float* images, const char** texts,
                        int batch_size, int h, int w, int c, float lr) {
    (void)lr;
    int dim = model->embed_dim;
    float* img_embs = (float*)malloc((size_t)batch_size * dim * sizeof(float));
    float* txt_embs = (float*)malloc((size_t)batch_size * dim * sizeof(float));

    for (int b = 0; b < batch_size; b++) {
        mm_clip_encode_image(model, images + b * h * w * c, h, w, c,
                             img_embs + b * dim);
        mm_clip_encode_text(model, texts[b], txt_embs + b * dim);
    }

    float* sim = (float*)malloc((size_t)batch_size * batch_size * sizeof(float));
    mm_cosine_sim_matrix(img_embs, txt_embs, batch_size, dim, sim);

    float loss = mm_infonce_loss(sim, batch_size);
    (void)loss;

    free(sim);
    free(img_embs);
    free(txt_embs);
}

void mm_linear_init(mm_linear_t* l, int in_dim, int out_dim) {
    l->in_dim = in_dim;
    l->out_dim = out_dim;
    l->weight = (float*)calloc((size_t)in_dim * out_dim, sizeof(float));
    l->bias = (float*)calloc((size_t)out_dim, sizeof(float));
    float scale = sqrtf(2.0f / (float)in_dim);
    for (int i = 0; i < in_dim * out_dim; i++) {
        l->weight[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale;
    }
}

void mm_linear_free(mm_linear_t* l) {
    free(l->weight);
    free(l->bias);
}

void mm_linear_forward(const mm_linear_t* l, const float* x, float* y) {
    for (int o = 0; o < l->out_dim; o++) {
        float sum = l->bias[o];
        for (int i = 0; i < l->in_dim; i++) {
            sum += x[i] * l->weight[i * l->out_dim + o];
        }
        y[o] = sum;
    }
}

void mm_self_attn_init(mm_self_attn_t* attn, int dim, int num_heads) {
    attn->num_heads = num_heads;
    attn->head_dim = dim / num_heads;
    attn->scale = 1.0f / sqrtf((float)attn->head_dim);
    mm_linear_init(&attn->q_proj, dim, dim);
    mm_linear_init(&attn->k_proj, dim, dim);
    mm_linear_init(&attn->v_proj, dim, dim);
    mm_linear_init(&attn->out_proj, dim, dim);
}

void mm_self_attn_free(mm_self_attn_t* attn) {
    mm_linear_free(&attn->q_proj);
    mm_linear_free(&attn->k_proj);
    mm_linear_free(&attn->v_proj);
    mm_linear_free(&attn->out_proj);
}

static void mm_self_attn_forward_impl(const mm_self_attn_t* attn, const float* x,
                                       int seq_len, float* out) {
    int dim = attn->num_heads * attn->head_dim;
    int hd = attn->head_dim;
    int nh = attn->num_heads;

    float* q = (float*)malloc((size_t)seq_len * dim * sizeof(float));
    float* k = (float*)malloc((size_t)seq_len * dim * sizeof(float));
    float* v = (float*)malloc((size_t)seq_len * dim * sizeof(float));

    for (int s = 0; s < seq_len; s++) {
        mm_linear_forward(&attn->q_proj, x + s * dim, q + s * dim);
        mm_linear_forward(&attn->k_proj, x + s * dim, k + s * dim);
        mm_linear_forward(&attn->v_proj, x + s * dim, v + s * dim);
    }

    float* attn_out = (float*)calloc((size_t)seq_len * dim, sizeof(float));
    for (int h = 0; h < nh; h++) {
        for (int i = 0; i < seq_len; i++) {
            float* scores = (float*)malloc((size_t)seq_len * sizeof(float));
            float max_score = -1e9f;
            for (int j = 0; j < seq_len; j++) {
                float dot = 0.0f;
                for (int d = 0; d < hd; d++) {
                    dot += q[i * dim + h * hd + d] * k[j * dim + h * hd + d];
                }
                scores[j] = dot * attn->scale;
                if (scores[j] > max_score) max_score = scores[j];
            }
            float sum = 0.0f;
            for (int j = 0; j < seq_len; j++) {
                scores[j] = expf(scores[j] - max_score);
                sum += scores[j];
            }
            for (int j = 0; j < seq_len; j++) {
                scores[j] /= sum;
                for (int d = 0; d < hd; d++) {
                    attn_out[i * dim + h * hd + d] += scores[j] * v[j * dim + h * hd + d];
                }
            }
            free(scores);
        }
    }

    for (int s = 0; s < seq_len; s++) {
        mm_linear_forward(&attn->out_proj, attn_out + s * dim, out + s * dim);
    }

    free(q); free(k); free(v); free(attn_out);
}

void mm_self_attn_forward(const mm_self_attn_t* attn, const float* x,
                          int seq_len, float* out) {
    mm_self_attn_forward_impl(attn, x, seq_len, out);
}

void mm_layernorm(const float* x, const float* weight, const float* bias,
                  int n, float* out) {
    float mean = 0.0f, var = 0.0f;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= (float)n;
    for (int i = 0; i < n; i++) {
        float diff = x[i] - mean;
        var += diff * diff;
    }
    var = var / (float)n + 1e-5f;
    float inv_std = 1.0f / sqrtf(var);

    for (int i = 0; i < n; i++) {
        float w = weight ? weight[i] : 1.0f;
        float b = bias ? bias[i] : 0.0f;
        out[i] = (x[i] - mean) * inv_std * w + b;
    }
}

float mm_gelu(float x) {
    float c = sqrtf(2.0f / 3.14159265f);
    return 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x * x * x)));
}

void mm_gelu_forward(const float* x, int n, float* out) {
    for (int i = 0; i < n; i++) out[i] = mm_gelu(x[i]);
}

void mm_softmax(float* x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

void mm_transformer_block_init(mm_transformer_block_t* blk, int dim, int num_heads, int ffn_dim) {
    blk->dim = dim;
    mm_self_attn_init(&blk->self_attn, dim, num_heads);
    mm_linear_init(&blk->fc1, dim, ffn_dim);
    mm_linear_init(&blk->fc2, ffn_dim, dim);
    blk->ln1_weight = (float*)calloc((size_t)dim, sizeof(float));
    blk->ln1_bias = (float*)calloc((size_t)dim, sizeof(float));
    blk->ln2_weight = (float*)calloc((size_t)dim, sizeof(float));
    blk->ln2_bias = (float*)calloc((size_t)dim, sizeof(float));
    for (int i = 0; i < dim; i++) {
        blk->ln1_weight[i] = 1.0f;
        blk->ln2_weight[i] = 1.0f;
    }
}

void mm_transformer_block_free(mm_transformer_block_t* blk) {
    mm_self_attn_free(&blk->self_attn);
    mm_linear_free(&blk->fc1);
    mm_linear_free(&blk->fc2);
    free(blk->ln1_weight);
    free(blk->ln1_bias);
    free(blk->ln2_weight);
    free(blk->ln2_bias);
}

void mm_transformer_block_forward(const mm_transformer_block_t* blk, const float* x,
                                  int seq_len, float* out) {
    int dim = blk->dim;

    float* ln1_out = (float*)malloc((size_t)seq_len * dim * sizeof(float));
    float* attn_out = (float*)malloc((size_t)seq_len * dim * sizeof(float));
    float* residual1 = (float*)malloc((size_t)seq_len * dim * sizeof(float));

    for (int s = 0; s < seq_len; s++) {
        mm_layernorm(x + s * dim, blk->ln1_weight, blk->ln1_bias, dim, ln1_out + s * dim);
    }

    mm_self_attn_forward_impl(&blk->self_attn, ln1_out, seq_len, attn_out);

    for (int i = 0; i < seq_len * dim; i++) {
        residual1[i] = x[i] + attn_out[i];
    }

    float* ln2_out = (float*)malloc((size_t)seq_len * dim * sizeof(float));
    float* ffn_hidden = (float*)malloc((size_t)seq_len * blk->fc1.out_dim * sizeof(float));
    float* ffn_out = (float*)malloc((size_t)seq_len * dim * sizeof(float));

    for (int s = 0; s < seq_len; s++) {
        mm_layernorm(residual1 + s * dim, blk->ln2_weight, blk->ln2_bias, dim, ln2_out + s * dim);
    }

    int ffn_dim = blk->fc1.out_dim;
    for (int s = 0; s < seq_len; s++) {
        mm_linear_forward(&blk->fc1, ln2_out + s * dim, ffn_hidden + s * ffn_dim);
    }
    mm_gelu_forward(ffn_hidden, seq_len * ffn_dim, ffn_hidden);
    for (int s = 0; s < seq_len; s++) {
        mm_linear_forward(&blk->fc2, ffn_hidden + s * ffn_dim, ffn_out + s * dim);
    }

    for (int i = 0; i < seq_len * dim; i++) {
        out[i] = residual1[i] + ffn_out[i];
    }

    free(ln1_out); free(attn_out); free(residual1);
    free(ln2_out); free(ffn_hidden); free(ffn_out);
}

void mm_text_encoder_init(mm_text_encoder_t* enc, int embed_dim, int num_layers,
                          int vocab_size, int max_seq_len) {
    enc->embed_dim = embed_dim;
    enc->num_layers = num_layers;
    enc->vocab_size = vocab_size;
    enc->max_seq_len = max_seq_len;

    mm_linear_init(&enc->token_embed, vocab_size, embed_dim);

    enc->pos_embed = (float*)calloc((size_t)max_seq_len * embed_dim, sizeof(float));
    for (int p = 0; p < max_seq_len; p++) {
        for (int d = 0; d < embed_dim; d++) {
            float angle = (float)p / powf(10000.0f, (float)(d / 2 * 2) / (float)embed_dim);
            enc->pos_embed[p * embed_dim + d] = (d % 2 == 0) ? sinf(angle) : cosf(angle);
        }
    }

    enc->blocks = (mm_transformer_block_t*)malloc((size_t)num_layers * sizeof(mm_transformer_block_t));
    for (int l = 0; l < num_layers; l++) {
        mm_transformer_block_init(&enc->blocks[l], embed_dim, MM_CLIP_NUM_HEADS, MM_CLIP_FFN_DIM);
    }

    enc->final_ln_weight.in_dim = 0;
    enc->final_ln_weight.out_dim = 0;
    enc->final_ln_weight.weight = (float*)calloc((size_t)embed_dim, sizeof(float));
    enc->final_ln_weight.bias = (float*)calloc((size_t)embed_dim, sizeof(float));
    enc->final_ln_bias = enc->final_ln_weight;
    for (int i = 0; i < embed_dim; i++) {
        enc->final_ln_weight.weight[i] = 1.0f;
    }

    enc->proj = (float*)calloc((size_t)embed_dim * embed_dim, sizeof(float));
    for (int d = 0; d < embed_dim; d++) enc->proj[d * embed_dim + d] = 1.0f;
}

void mm_text_encoder_free(mm_text_encoder_t* enc) {
    mm_linear_free(&enc->token_embed);
    free(enc->pos_embed);
    for (int l = 0; l < enc->num_layers; l++) {
        mm_transformer_block_free(&enc->blocks[l]);
    }
    free(enc->blocks);
    free(enc->final_ln_weight.weight);
    free(enc->final_ln_weight.bias);
    free(enc->proj);
}

void mm_image_encoder_init(mm_image_encoder_t* enc, int embed_dim, int num_layers,
                           int image_size, int patch_size) {
    enc->embed_dim = embed_dim;
    enc->num_layers = num_layers;
    enc->image_size = image_size;
    enc->patch_size = patch_size;
    enc->num_patches = (image_size / patch_size) * (image_size / patch_size);

    int patch_dim = patch_size * patch_size * 3;
    mm_linear_init(&enc->patch_proj, patch_dim, embed_dim);

    enc->class_token = (float*)calloc((size_t)embed_dim, sizeof(float));
    float scale = 1.0f / sqrtf((float)embed_dim);
    for (int d = 0; d < embed_dim; d++)
        enc->class_token[d] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale;

    int total = enc->num_patches + 1;
    enc->pos_embed = (float*)calloc((size_t)total * embed_dim, sizeof(float));
    for (int p = 0; p < total; p++) {
        for (int d = 0; d < embed_dim; d++) {
            float angle = (float)p / powf(10000.0f, (float)(d / 2 * 2) / (float)embed_dim);
            enc->pos_embed[p * embed_dim + d] = (d % 2 == 0) ? sinf(angle) : cosf(angle);
        }
    }

    enc->blocks = (mm_transformer_block_t*)malloc((size_t)num_layers * sizeof(mm_transformer_block_t));
    for (int l = 0; l < num_layers; l++) {
        mm_transformer_block_init(&enc->blocks[l], embed_dim, MM_CLIP_NUM_HEADS, MM_CLIP_FFN_DIM);
    }

    enc->final_ln_weight.in_dim = 0;
    enc->final_ln_weight.out_dim = 0;
    enc->final_ln_weight.weight = (float*)calloc((size_t)embed_dim, sizeof(float));
    enc->final_ln_weight.bias = (float*)calloc((size_t)embed_dim, sizeof(float));
    enc->final_ln_bias = enc->final_ln_weight;
    for (int i = 0; i < embed_dim; i++) enc->final_ln_weight.weight[i] = 1.0f;

    enc->proj = (float*)calloc((size_t)embed_dim * embed_dim, sizeof(float));
    for (int d = 0; d < embed_dim; d++) enc->proj[d * embed_dim + d] = 1.0f;
}

void mm_image_encoder_free(mm_image_encoder_t* enc) {
    mm_linear_free(&enc->patch_proj);
    free(enc->class_token);
    free(enc->pos_embed);
    for (int l = 0; l < enc->num_layers; l++) {
        mm_transformer_block_free(&enc->blocks[l]);
    }
    free(enc->blocks);
    free(enc->final_ln_weight.weight);
    free(enc->final_ln_weight.bias);
    free(enc->proj);
}
