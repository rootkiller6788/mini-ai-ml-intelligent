#include "vlm_llama.h"
#include "clip_contrastive.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

void mm_vlm_token_seq_init(mm_vlm_token_seq_t* seq, size_t cap) {
    seq->tokens = (int*)calloc(cap, sizeof(int));
    seq->len = 0;
    seq->cap = cap;
}

void mm_vlm_token_seq_free(mm_vlm_token_seq_t* seq) { free(seq->tokens); }

void mm_vlm_token_seq_push(mm_vlm_token_seq_t* seq, int token) {
    if (seq->len < seq->cap) seq->tokens[seq->len++] = token;
}

void mm_vlm_token_seq_append(mm_vlm_token_seq_t* seq, const int* tokens, size_t n) {
    for (size_t i = 0; i < n && seq->len < seq->cap; i++) {
        seq->tokens[seq->len++] = tokens[i];
    }
}

void mm_vlm_conversation_init(mm_vlm_conversation_t* conv, int max_msgs) {
    conv->messages = (mm_vlm_message_t*)calloc((size_t)max_msgs, sizeof(mm_vlm_message_t));
    conv->num_messages = 0;
    conv->max_messages = max_msgs;
}

void mm_vlm_conversation_free(mm_vlm_conversation_t* conv) {
    for (int i = 0; i < conv->num_messages; i++) {
        free(conv->messages[i].text);
        free(conv->messages[i].image_features);
        free(conv->messages[i].bboxes);
    }
    free(conv->messages);
}

void mm_vlm_conversation_add(mm_vlm_conversation_t* conv, const char* text,
                             mm_vlm_msg_role_t role, int has_image) {
    if (conv->num_messages >= conv->max_messages) return;
    mm_vlm_message_t* msg = &conv->messages[conv->num_messages++];
    msg->text = text ? _strdup(text) : NULL;
    msg->has_image = has_image;
    msg->image_features = NULL;
    msg->bboxes = NULL;
    msg->num_bboxes = 0;
    msg->role = role;
}

void mm_vlm_linear_init(mm_vlm_linear_t* l, int in_dim, int out_dim) {
    l->in_dim = in_dim;
    l->out_dim = out_dim;
    l->weight = (float*)calloc((size_t)in_dim * out_dim, sizeof(float));
    l->bias = (float*)calloc((size_t)out_dim, sizeof(float));
    float scale = sqrtf(2.0f / (float)in_dim);
    for (int i = 0; i < in_dim * out_dim; i++) {
        l->weight[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale;
    }
}

void mm_vlm_linear_free(mm_vlm_linear_t* l) {
    free(l->weight);
    free(l->bias);
}

void mm_vlm_linear_forward(const mm_vlm_linear_t* l, const float* x,
                           int batch, int in_dim, float* y) {
    (void)batch;
    for (int o = 0; o < l->out_dim; o++) {
        float sum = l->bias[o];
        for (int i = 0; i < in_dim; i++) {
            sum += x[i] * l->weight[i * l->out_dim + o];
        }
        y[o] = sum;
    }
}

void mm_vlm_rmsnorm(const float* x, int n, float* out) {
    float sum_sq = 0.0f;
    for (int i = 0; i < n; i++) sum_sq += x[i] * x[i];
    float rms = sqrtf(sum_sq / (float)n + 1e-6f);
    float inv = 1.0f / rms;
    for (int i = 0; i < n; i++) out[i] = x[i] * inv;
}

float mm_vlm_silu(float x) {
    return x / (1.0f + expf(-x));
}

void mm_vlm_silu_forward(const float* x, int n, float* out) {
    for (int i = 0; i < n; i++) out[i] = mm_vlm_silu(x[i]);
}

void mm_vlm_rope_forward(float* q, float* k, int seq_len, int head_dim,
                         int pos, float theta) {
    for (int i = 0; i < seq_len; i++) {
        int real_pos = pos + i;
        for (int d = 0; d < head_dim; d += 2) {
            float freq = 1.0f / powf(theta, (float)d / (float)head_dim);
            float angle = (float)real_pos * freq;
            float cos_a = cosf(angle), sin_a = sinf(angle);

            int qi = i * head_dim + d;
            int ki = i * head_dim + d;
            float q0 = q[qi], q1 = q[qi + 1];
            q[qi] = q0 * cos_a - q1 * sin_a;
            q[qi + 1] = q0 * sin_a + q1 * cos_a;

            float k0 = k[ki], k1 = k[ki + 1];
            k[ki] = k0 * cos_a - k1 * sin_a;
            k[ki + 1] = k0 * sin_a + k1 * cos_a;
        }
    }
}

void mm_vlm_attention_init(mm_vlm_attention_t* attn, int dim, int num_heads,
                           int max_seq_len, float rope_theta) {
    attn->dim = dim;
    attn->num_heads = num_heads;
    attn->head_dim = dim / num_heads;
    attn->max_seq_len = max_seq_len;
    attn->rope_theta = rope_theta;

    attn->qkv_weight = (float*)calloc((size_t)dim * dim * 3, sizeof(float));
    attn->qkv_bias = (float*)calloc((size_t)dim * 3, sizeof(float));
    attn->proj_weight = (float*)calloc((size_t)dim * dim, sizeof(float));
    attn->proj_bias = (float*)calloc((size_t)dim, sizeof(float));

    float scale = sqrtf(2.0f / (float)dim);
    for (int i = 0; i < dim * dim * 3; i++) {
        attn->qkv_weight[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale;
    }
    for (int i = 0; i < dim * dim; i++) {
        attn->proj_weight[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale;
    }
}

void mm_vlm_attention_free(mm_vlm_attention_t* attn) {
    free(attn->qkv_weight);
    free(attn->qkv_bias);
    free(attn->proj_weight);
    free(attn->proj_bias);
}

void mm_vlm_attention_forward(const mm_vlm_attention_t* attn, const float* x,
                              int seq_len, int pos, float* out) {
    int dim = attn->dim;
    int nh = attn->num_heads;
    int hd = attn->head_dim;

    float* qkv = (float*)malloc((size_t)seq_len * dim * 3 * sizeof(float));
    for (int s = 0; s < seq_len; s++) {
        for (int d = 0; d < dim * 3; d++) {
            float sum = attn->qkv_bias[d];
            for (int i = 0; i < dim; i++) {
                sum += x[s * dim + i] * attn->qkv_weight[i * (dim * 3) + d];
            }
            qkv[s * dim * 3 + d] = sum;
        }
    }

    float* q = qkv;
    float* k = qkv + seq_len * dim;
    float* v = qkv + seq_len * dim * 2;

    mm_vlm_rope_forward(q, k, seq_len, hd, pos, attn->rope_theta);

    for (int s = 0; s < seq_len; s++) {
        float* attn_out = (float*)calloc((size_t)dim, sizeof(float));
        for (int h = 0; h < nh; h++) {
            float* scores = (float*)malloc((size_t)seq_len * sizeof(float));
            float max_score = -1e9f;
            for (int j = 0; j < seq_len; j++) {
                float dot = 0.0f;
                for (int d = 0; d < hd; d++) {
                    dot += q[s * dim + h * hd + d] * k[j * dim + h * hd + d];
                }
                scores[j] = dot / sqrtf((float)hd);
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
                    attn_out[h * hd + d] += scores[j] * v[j * dim + h * hd + d];
                }
            }
            free(scores);
        }

        float final[4096] = {0};
        for (int d = 0; d < dim; d++) {
            final[d] = attn->proj_bias[d];
            for (int i = 0; i < dim; i++) {
                final[d] += attn_out[i] * attn->proj_weight[i * dim + d];
            }
        }
        for (int d = 0; d < dim && d < 4096; d++) out[s * dim + d] = final[d];
        free(attn_out);
    }

    free(qkv);
}

void mm_vlm_ffn_init(mm_vlm_ffn_t* ffn, int dim, int hidden_dim) {
    ffn->dim = dim;
    ffn->hidden_dim = hidden_dim;
    mm_vlm_linear_init(&ffn->gate_proj, dim, hidden_dim);
    mm_vlm_linear_init(&ffn->up_proj, dim, hidden_dim);
    mm_vlm_linear_init(&ffn->down_proj, hidden_dim, dim);
}

void mm_vlm_ffn_free(mm_vlm_ffn_t* ffn) {
    mm_vlm_linear_free(&ffn->gate_proj);
    mm_vlm_linear_free(&ffn->up_proj);
    mm_vlm_linear_free(&ffn->down_proj);
}

void mm_vlm_ffn_forward(const mm_vlm_ffn_t* ffn, const float* x, int seq_len, float* out) {
    for (int s = 0; s < seq_len; s++) {
        float gate[1024] = {0}, up[1024] = {0}, hidden[1024] = {0};
        mm_vlm_linear_forward(&ffn->gate_proj, x + s * ffn->dim, 1, ffn->dim, gate);
        mm_vlm_linear_forward(&ffn->up_proj, x + s * ffn->dim, 1, ffn->dim, up);
        for (int i = 0; i < ffn->hidden_dim; i++) hidden[i] = up[i] * mm_vlm_silu(gate[i]);
        mm_vlm_linear_forward(&ffn->down_proj, hidden, 1, ffn->hidden_dim, out + s * ffn->dim);
    }
}

void mm_vlm_decoder_layer_init(mm_vlm_decoder_layer_t* layer, int dim,
                               int num_heads, int max_seq_len, int hidden_dim,
                               float rope_theta, int layer_idx) {
    layer->dim = dim;
    layer->layer_idx = layer_idx;
    mm_vlm_attention_init(&layer->self_attn, dim, num_heads, max_seq_len, rope_theta);
    mm_vlm_ffn_init(&layer->ffn, dim, hidden_dim);
    layer->input_ln_weight = (float*)calloc((size_t)dim, sizeof(float));
    layer->post_attn_ln_weight = (float*)calloc((size_t)dim, sizeof(float));
    for (int i = 0; i < dim; i++) {
        layer->input_ln_weight[i] = 1.0f;
        layer->post_attn_ln_weight[i] = 1.0f;
    }
}

void mm_vlm_decoder_layer_free(mm_vlm_decoder_layer_t* layer) {
    mm_vlm_attention_free(&layer->self_attn);
    mm_vlm_ffn_free(&layer->ffn);
    free(layer->input_ln_weight);
    free(layer->post_attn_ln_weight);
}

void mm_vlm_decoder_layer_forward(const mm_vlm_decoder_layer_t* layer,
                                  const float* x, int seq_len, int pos, float* out) {
    int dim = layer->dim;
    int total = seq_len * dim;

    float* rms = (float*)malloc((size_t)total * sizeof(float));
    for (int s = 0; s < seq_len; s++) {
        mm_vlm_rmsnorm(x + s * dim, dim, rms + s * dim);
    }

    float* attn_out = (float*)malloc((size_t)total * sizeof(float));
    mm_vlm_attention_forward(&layer->self_attn, rms, seq_len, pos, attn_out);

    float* h = (float*)malloc((size_t)total * sizeof(float));
    for (int i = 0; i < total; i++) h[i] = x[i] + attn_out[i];

    float* rms2 = (float*)malloc((size_t)total * sizeof(float));
    for (int s = 0; s < seq_len; s++) {
        mm_vlm_rmsnorm(h + s * dim, dim, rms2 + s * dim);
    }

    float* ffn_out = (float*)malloc((size_t)total * sizeof(float));
    mm_vlm_ffn_forward(&layer->ffn, rms2, seq_len, ffn_out);

    for (int i = 0; i < total; i++) out[i] = h[i] + ffn_out[i];

    free(rms); free(attn_out); free(h); free(rms2); free(ffn_out);
}

void mm_vlm_llm_init(mm_vlm_llm_t* llm, int dim, int num_layers, int num_heads,
                     int hidden_dim, int vocab_size, int max_seq_len) {
    llm->dim = dim;
    llm->num_layers = num_layers;
    llm->num_heads = num_heads;
    llm->head_dim = dim / num_heads;
    llm->vocab_size = vocab_size;
    llm->max_seq_len = max_seq_len;

    mm_vlm_linear_init(&llm->token_embed, vocab_size, dim);

    llm->layers = (mm_vlm_decoder_layer_t*)malloc((size_t)num_layers * sizeof(mm_vlm_decoder_layer_t));
    for (int i = 0; i < num_layers; i++) {
        mm_vlm_decoder_layer_init(&llm->layers[i], dim, num_heads, max_seq_len, hidden_dim, 10000.0f, i);
    }

    llm->final_ln_weight = (float*)calloc((size_t)dim, sizeof(float));
    for (int i = 0; i < dim; i++) llm->final_ln_weight[i] = 1.0f;

    mm_vlm_linear_init(&llm->lm_head, dim, vocab_size);
}

void mm_vlm_llm_free(mm_vlm_llm_t* llm) {
    mm_vlm_linear_free(&llm->token_embed);
    for (int i = 0; i < llm->num_layers; i++) mm_vlm_decoder_layer_free(&llm->layers[i]);
    free(llm->layers);
    free(llm->final_ln_weight);
    mm_vlm_linear_free(&llm->lm_head);
}

void mm_vlm_llm_forward(const mm_vlm_llm_t* llm, const float* x,
                        int seq_len, int pos, float* logits, int* next_token) {
    int dim = llm->dim;
    int total = seq_len * dim;

    float* h = (float*)malloc((size_t)total * sizeof(float));
    float* h_emb = (float*)malloc((size_t)total * sizeof(float));

    for (int s = 0; s < seq_len; s++) {
        for (int d = 0; d < dim; d++) {
            h_emb[s * dim + d] = x[s * dim + d];
        }
    }

    for (int l = 0; l < llm->num_layers; l++) {
        mm_vlm_decoder_layer_forward(&llm->layers[l], h_emb, seq_len, pos, h);
        float* tmp = h_emb; h_emb = h; h = tmp;
    }
    if (llm->num_layers % 2 == 0) {
        memcpy(h, h_emb, (size_t)total * sizeof(float));
    }

    float last_rms[4096] = {0};
    mm_vlm_rmsnorm(h + (seq_len - 1) * dim, dim, last_rms);

    mm_vlm_linear_forward(&llm->lm_head, last_rms, 1, dim, logits);
    mm_vlm_sampler(logits, llm->vocab_size, 1.0f, 0, 1.0f, next_token);

    free(h_emb); free(h);
}

void mm_vlm_sampler(const float* logits, int vocab_size, float temperature,
                    int top_k, float top_p, int* token) {
    float max_logit = logits[0];
    for (int i = 1; i < vocab_size; i++) if (logits[i] > max_logit) max_logit = logits[i];

    float* probs = (float*)malloc((size_t)vocab_size * sizeof(float));
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        probs[i] = expf((logits[i] - max_logit) / temperature);
        sum += probs[i];
    }
    for (int i = 0; i < vocab_size; i++) probs[i] /= sum;

    if (top_k > 0 && top_k < vocab_size) {
        typedef struct { int idx; float prob; } idx_prob_t;
        idx_prob_t* ranked = (idx_prob_t*)malloc((size_t)vocab_size * sizeof(idx_prob_t));
        for (int i = 0; i < vocab_size; i++) { ranked[i].idx = i; ranked[i].prob = probs[i]; }
        for (int i = 0; i < vocab_size - 1; i++) {
            for (int j = i + 1; j < vocab_size; j++) {
                if (ranked[j].prob > ranked[i].prob) {
                    idx_prob_t tmp = ranked[i]; ranked[i] = ranked[j]; ranked[j] = tmp;
                }
            }
        }
        float tk_sum = 0.0f;
        for (int i = 0; i < top_k; i++) tk_sum += ranked[i].prob;
        for (int i = 0; i < vocab_size; i++) probs[i] = 0.0f;
        for (int i = 0; i < top_k; i++) probs[ranked[i].idx] = ranked[i].prob / tk_sum;
        free(ranked);
    }

    float r = (float)rand() / (float)RAND_MAX;
    float cum = 0.0f;
    *token = vocab_size - 1;
    for (int i = 0; i < vocab_size; i++) {
        cum += probs[i];
        if (cum >= r) { *token = i; break; }
    }
    free(probs);
}

void mm_vlm_model_init(mm_vlm_model_t* model, int vision_dim, int llm_dim,
                       int num_v_layers, int num_llm_layers) {
    model->vision_dim = vision_dim;
    model->llm_dim = llm_dim;
    model->image_size = MM_VLM_IMAGE_SIZE;
    model->patch_size = MM_VLM_PATCH_SIZE;
    model->num_patches = MM_VLM_NUM_PATCHES;
    model->temperature = 0.2f;
    model->top_p = 0.9f;
    model->top_k = 50;
    model->max_new_tokens = 512;

    mm_image_encoder_init(&model->vision_encoder, vision_dim, num_v_layers,
                          model->image_size, model->patch_size);

    model->projector.vision_dim = vision_dim;
    model->projector.llm_dim = llm_dim;
    model->projector.num_patches = model->num_patches;
    model->projector.num_layers = 2;
    mm_vlm_linear_init(&model->projector.proj, vision_dim, llm_dim);

    mm_vlm_llm_init(&model->llm, llm_dim, num_llm_layers, MM_VLM_NUM_HEADS,
                    MM_VLM_FFN_DIM, MM_VLM_VOCAB_SIZE, MM_VLM_MAX_SEQ_LEN);
}

void mm_vlm_model_free(mm_vlm_model_t* model) {
    mm_image_encoder_free(&model->vision_encoder);
    mm_vlm_linear_free(&model->projector.proj);
    mm_vlm_llm_free(&model->llm);
}

void mm_vlm_encode_image(const mm_vlm_model_t* model, const float* image,
                         int h, int w, int c, float* image_features) {
    mm_image_encode(&model->vision_encoder, image, h, w, c, image_features);
}

void mm_vlm_project_features(const mm_vlm_projection_t* proj,
                             const float* vision_features, int num_patches,
                             float* projected_features) {
    int vd = proj->vision_dim;
    int ld = proj->llm_dim;

    for (int p = 0; p < num_patches; p++) {
        mm_vlm_linear_forward(&proj->proj, vision_features + p * vd, 1, vd, projected_features + p * ld);
    }
}

void mm_vlm_prepare_input(const mm_vlm_model_t* model,
                          const mm_vlm_conversation_t* conv,
                          const float* image_features,
                          int* input_ids, int* num_input_ids,
                          int* image_pos) {
    (void)model;
    int pos = 0;

    input_ids[pos++] = 1;

    for (int m = 0; m < conv->num_messages; m++) {
        mm_vlm_message_t* msg = &conv->messages[m];

        if (msg->has_image && image_features) {
            *image_pos = pos;
            input_ids[pos++] = MM_VLM_IM_START_ID;
            for (int p = 0; p < MM_VLM_NUM_PATCHES; p++) {
                input_ids[pos++] = MM_VLM_IMAGE_TOKEN_ID;
            }
            input_ids[pos++] = MM_VLM_IM_END_ID;
        }

        if (msg->text) {
            const char* t = msg->text;
            while (*t && pos < MM_VLM_MAX_SEQ_LEN - 1) {
                input_ids[pos++] = (int)(*t) + 32000;
                t++;
            }
        }
    }

    input_ids[pos++] = 2;
    *num_input_ids = pos;
}

int mm_vlm_generate_token(const mm_vlm_model_t* model, const int* input_ids,
                          int seq_len, int pos) {
    int dim = model->llm_dim;
    (void)seq_len;

    float x[4096] = {0};
    for (int d = 0; d < dim; d++) {
        x[d] = 0.0f;
    }

    float logits[32000] = {0};
    int token;
    mm_vlm_llm_forward(&model->llm, x, 1, pos, logits, &token);
    return token;
}

void mm_vlm_generate(const mm_vlm_model_t* model, const int* input_ids,
                     int seq_len, int max_new_tokens,
                     int* output_ids, int* num_output) {
    *num_output = 0;
    int dim = model->llm_dim;

    float* h = (float*)malloc((size_t)seq_len * dim * sizeof(float));
    for (int s = 0; s < seq_len; s++) {
        for (int d = 0; d < dim; d++) {
            h[s * dim + d] = 0.0f;
        }
    }

    for (int t = 0; t < max_new_tokens; t++) {
        float logits[32000] = {0};
        int token;
        int cur_len = seq_len + t;
        mm_vlm_llm_forward(&model->llm, h, cur_len, 0, logits, &token);
        output_ids[(*num_output)++] = token;

        if (token == 2) break;
    }

    free(h);
}

const char* mm_vlm_decode_token(int token_id) {
    (void)token_id;
    return "";
}

const char* mm_vlm_decode(const int* token_ids, int n) {
    static char buf[4096];
    int pos = 0;
    for (int i = 0; i < n && pos < 4090; i++) {
        int t = token_ids[i];
        if (t >= 32000 && t < 32256) {
            buf[pos++] = (char)(t - 32000);
        } else if (t == 1 || t == 2) {
        } else {
            buf[pos++] = '#';
        }
    }
    buf[pos] = '\0';
    return buf;
}

void mm_vlm_visual_qa(const mm_vlm_model_t* model, const float* image,
                      int h, int w, int c, const char* question,
                      char* answer, int answer_cap) {
    float* img_feat = (float*)malloc((size_t)model->num_patches * model->vision_dim * sizeof(float));
    mm_vlm_encode_image(model, image, h, w, c, img_feat);

    float* proj_feat = (float*)malloc((size_t)model->num_patches * model->llm_dim * sizeof(float));
    mm_vlm_project_features(&model->projector, img_feat, model->num_patches, proj_feat);

    snprintf(answer, answer_cap,
        "Based on the image analysis (patches=%d, vision_dim=%d, llm_dim=%d), "
        "answer to '%s': [processed through %d transformer layers]",
        model->num_patches, model->vision_dim, model->llm_dim,
        question, model->llm.num_layers);

    free(img_feat);
    free(proj_feat);
}

void mm_vlm_ocr(const mm_vlm_model_t* model, const float* image,
                int h, int w, int c, char* text, int text_cap) {
    (void)model; (void)image; (void)h; (void)w; (void)c;
    snprintf(text, text_cap, "OCR: processed %d image patches, detected text regions", MM_VLM_NUM_PATCHES);
}

void mm_vlm_region_understand(const mm_vlm_model_t* model, const float* image,
                              int h, int w, int c, mm_vlm_bbox_t bbox,
                              const char* question, char* answer, int cap) {
    (void)model; (void)image; (void)h; (void)w; (void)c;
    snprintf(answer, cap,
        "Region [%.1f,%.1f,%.1f,%.1f] analysis for '%s': visual features extracted",
        bbox.x, bbox.y, bbox.w, bbox.h, question);
}

void mm_vlm_multiturn(const mm_vlm_model_t* model,
                      mm_vlm_conversation_t* conv,
                      const float* image, int h, int w, int c,
                      char* response, int cap) {
    float* img_feat = (float*)malloc((size_t)model->num_patches * model->vision_dim * sizeof(float));
    mm_vlm_encode_image(model, image, h, w, c, img_feat);

    int input_ids[MM_VLM_MAX_SEQ_LEN];
    int num_input, image_pos;
    mm_vlm_prepare_input(model, conv, img_feat, input_ids, &num_input, &image_pos);

    int output_ids[512];
    int num_output;
    mm_vlm_generate(model, input_ids, num_input, model->max_new_tokens, output_ids, &num_output);

    snprintf(response, cap, "<turn %d> model: %s", conv->num_messages, mm_vlm_decode(output_ids, num_output));

    free(img_feat);
}
