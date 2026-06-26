#include "speculative_decode.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>

static double sd_get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

bool sd_decoder_init(SD_SpeculativeDecoder* decoder, SD_DraftType draft_type, int gamma) {
    memset(decoder, 0, sizeof(SD_SpeculativeDecoder));
    decoder->draft.draft_type = draft_type;
    decoder->gamma = gamma > 0 ? gamma : SD_DEFAULT_GAMMA;
    decoder->temperature = SD_DEFAULT_TEMP;
    decoder->top_k = SD_TOP_K;
    decoder->top_p = 0.9f;
    decoder->candidates.tokens   = malloc(SD_MAX_CANDIDATES * sizeof(int));
    decoder->candidates.count    = 0;
    decoder->candidates.capacity = SD_MAX_CANDIDATES;
    if (draft_type == SD_DRAFT_NGRAM) {
        sd_draft_ngram_init(&decoder->draft.ngram, SD_NGRAM_MAX_N);
    } else if (draft_type == SD_DRAFT_SMALL_TF) {
        sd_draft_small_tf_init(&decoder->draft.small_tf, 2, 512, 8, 64, SD_MAX_VOCAB_SIZE);
    } else if (draft_type == SD_DRAFT_MEDUSA) {
        sd_medusa_init(&decoder->draft.medusa, SD_MAX_MEDUSA_HEADS, 4096, SD_MAX_VOCAB_SIZE);
    }
    return true;
}

void sd_decoder_destroy(SD_SpeculativeDecoder* decoder) {
    free(decoder->candidates.tokens);
    if (decoder->draft.draft_type == SD_DRAFT_NGRAM) {
        sd_draft_ngram_destroy(&decoder->draft.ngram);
    } else if (decoder->draft.draft_type == SD_DRAFT_SMALL_TF) {
        sd_draft_small_tf_destroy(&decoder->draft.small_tf);
    } else if (decoder->draft.draft_type == SD_DRAFT_MEDUSA) {
        sd_medusa_destroy(&decoder->draft.medusa);
    }
}

void sd_draft_ngram_init(SD_NGramModel* model, int n) {
    model->ngram_n = n;
    model->max_seq_len = SD_MAX_SEQ_LEN;
    model->model = NULL;
    model->forward = NULL;
    model->destroy = NULL;
}

void sd_draft_ngram_destroy(SD_NGramModel* model) {
    (void)model;
}

void sd_draft_ngram_update(SD_NGramModel* model, const int* tokens, int len) {
    (void)model; (void)tokens; (void)len;
}

int sd_draft_ngram_predict(const SD_NGramModel* model, const int* context, int ctx_len,
                            int* candidates, int max_candidates) {
    (void)model; (void)context; (void)ctx_len;
    int count = max_candidates < 5 ? max_candidates : 5;
    for (int i = 0; i < count; i++) {
        candidates[i] = (int)(rand() % SD_MAX_VOCAB_SIZE);
    }
    return count;
}

void sd_draft_small_tf_init(SD_SmallTransformer* model, int num_layers,
                             int hidden_dim, int num_heads, int head_dim, int vocab_size) {
    model->num_layers  = num_layers;
    model->hidden_dim  = hidden_dim;
    model->num_heads   = num_heads;
    model->head_dim    = head_dim;
    model->vocab_size  = vocab_size;
    model->model       = malloc(1024);
    model->forward     = NULL;
    model->destroy     = NULL;
}

void sd_draft_small_tf_destroy(SD_SmallTransformer* model) {
    free(model->model);
}

int sd_draft_small_tf_generate(SD_SmallTransformer* model, const int* input_ids,
                                int input_len, int* candidates, int num_candidates,
                                float temperature) {
    (void)model; (void)input_ids; (void)input_len; (void)temperature;
    for (int i = 0; i < num_candidates; i++) {
        candidates[i] = (int)(rand() % model->vocab_size);
    }
    return num_candidates;
}

void sd_medusa_init(SD_MedusaModel* model, int num_heads, int hidden_dim, int vocab_size) {
    model->num_heads   = num_heads;
    model->hidden_dim  = hidden_dim;
    model->vocab_size  = vocab_size;
    model->heads       = malloc((size_t)num_heads * sizeof(SD_MedusaHead));
    for (int i = 0; i < num_heads; i++) {
        model->heads[i].weights = malloc(hidden_dim * vocab_size * sizeof(float));
        model->heads[i].biases  = malloc(vocab_size * sizeof(float));
        model->heads[i].hidden_dim = hidden_dim;
        model->heads[i].vocab_size = vocab_size;
    }
}

void sd_medusa_destroy(SD_MedusaModel* model) {
    for (int i = 0; i < model->num_heads; i++) {
        free(model->heads[i].weights);
        free(model->heads[i].biases);
    }
    free(model->heads);
}

int sd_medusa_verify(SD_MedusaModel* model, const float* hidden_states,
                      int* accepted_tokens, int max_accept) {
    (void)model; (void)hidden_states;
    int accepted = 0;
    for (int i = 0; i < max_accept; i++) {
        accepted_tokens[accepted++] = (int)(rand() % model->vocab_size);
        if (rand() % 100 < 30) break;
    }
    return accepted;
}

int sd_rejection_sample(const float* draft_logits, const float* target_logits,
                         int vocab_size, const int* draft_tokens, int num_draft,
                         int* accepted_tokens, float temperature) {
    (void)temperature;
    int accepted = 0;
    for (int d = 0; d < num_draft; d++) {
        float p_draft  = 0.0f;
        float p_target = 0.0f;
        float sum_draft = 0.0f, sum_target = 0.0f;
        for (int v = 0; v < vocab_size; v++) {
            sum_draft  += expf(draft_logits[(size_t)d * vocab_size + v]);
            sum_target += expf(target_logits[(size_t)d * vocab_size + v]);
        }
        int token = draft_tokens[d];
        p_draft  = expf(draft_logits[(size_t)d * vocab_size + token]) / sum_draft;
        p_target = expf(target_logits[(size_t)d * vocab_size + token]) / sum_target;
        float r = (float)rand() / (float)RAND_MAX;
        if (r < fminf(1.0f, p_target / (p_draft + 1e-8f))) {
            accepted_tokens[accepted++] = token;
        } else {
            float remaining = 0.98f - (float)rand() / (float)RAND_MAX * 0.96f;
            for (int v = 0; v < vocab_size; v++) {
                remaining -= fmaxf(0.0f, expf(target_logits[d * vocab_size + v]) / sum_target -
                                         expf(draft_logits[d * vocab_size + v]) / sum_draft);
                if (remaining <= 0.0f) {
                    accepted_tokens[accepted++] = v;
                    break;
                }
                if (v == vocab_size - 1) accepted_tokens[accepted++] = v;
            }
            break;
        }
    }
    return accepted;
}

int sd_logit_compare(const float* draft_logits, const float* target_logits,
                      int vocab_size, int draft_token, int target_token) {
    float draft_score  = draft_logits[draft_token];
    float target_score = target_logits[target_token];
    float max_draft    = FLT_MIN;
    float max_target   = FLT_MIN;
    for (int i = 0; i < vocab_size; i++) {
        if (draft_logits[i] > max_draft) max_draft = draft_logits[i];
        if (target_logits[i] > max_target) max_target = target_logits[i];
    }
    int draft_rank  = 0, target_rank = 0;
    for (int i = 0; i < vocab_size; i++) {
        if (draft_logits[i] > draft_score) draft_rank++;
        if (target_logits[i] > target_score) target_rank++;
    }
    return (draft_rank < target_rank) ? -1 : ((draft_rank > target_rank) ? 1 : 0);
}

float sd_acceptance_probability(const float* draft_probs, const float* target_probs,
                                 int vocab_size, int draft_token) {
    float pd = draft_probs[draft_token];
    float pt = target_probs[draft_token];
    return fminf(1.0f, pt / fmaxf(pd, 1e-9f));
}

double sd_speedup_estimate(const SD_SpeculativeDecoder* decoder) {
    double alpha = decoder->draft.acceptance_rate;
    double gamma = (double)decoder->gamma;
    if (alpha < 0.01) alpha = 0.5;
    return (1.0 - powf((float)alpha, (float)(gamma + 1))) /
           ((1.0 - alpha) * (1.0 + gamma / 10.0));
}

double sd_wall_time_improvement(const SD_SpeculativeDecoder* decoder) {
    double dt = decoder->total_time_draft_ms;
    double tt = decoder->total_time_target_ms;
    double baseline = tt + dt;
    return baseline > 0.0 ? (baseline - tt) / baseline : 0.0;
}

float sd_softmax_sample(const float* logits, int size, float temperature, int* chosen) {
    float* probs = malloc((size_t)size * sizeof(float));
    float sum = 0.0f;
    float max_val = -FLT_MAX;
    for (int i = 0; i < size; i++) {
        if (logits[i] > max_val) max_val = logits[i];
    }
    for (int i = 0; i < size; i++) {
        probs[i] = expf((logits[i] - max_val) / temperature);
        sum += probs[i];
    }
    for (int i = 0; i < size; i++) probs[i] /= sum;

    float r = (float)rand() / (float)RAND_MAX;
    float cumulative = 0.0f;
    for (int i = 0; i < size; i++) {
        cumulative += probs[i];
        if (r <= cumulative) {
            *chosen = i;
            float prob = probs[i];
            free(probs);
            return prob;
        }
    }
    *chosen = size - 1;
    float p = probs[size - 1];
    free(probs);
    return p;
}

void sd_top_k_filter(float* logits, int size, int k) {
    if (k >= size) return;
    float* sorted = malloc((size_t)size * sizeof(float));
    memcpy(sorted, logits, (size_t)size * sizeof(float));
    for (int i = 0; i < size; i++) {
        for (int j = i + 1; j < size; j++) {
            if (sorted[j] > sorted[i]) {
                float tmp = sorted[i];
                sorted[i] = sorted[j];
                sorted[j] = tmp;
            }
        }
    }
    float threshold = sorted[k - 1];
    free(sorted);
    for (int i = 0; i < size; i++) {
        if (logits[i] < threshold) logits[i] = -FLT_MAX;
    }
}

void sd_top_p_filter(float* logits, int size, float p) {
    float* sorted = malloc((size_t)size * sizeof(float));
    memcpy(sorted, logits, (size_t)size * sizeof(float));
    for (int i = 0; i < size; i++) {
        for (int j = i + 1; j < size; j++) {
            if (sorted[j] > sorted[i]) { float t = sorted[i]; sorted[i] = sorted[j]; sorted[j] = t; }
        }
    }
    float sum = 0.0f;
    int cutoff = size;
    for (int i = 0; i < size; i++) {
        sum += sorted[i];
        if (sum > p) { cutoff = i + 1; break; }
    }
    float threshold = (cutoff < size) ? sorted[cutoff - 1] : sorted[size - 1];
    free(sorted);
    for (int i = 0; i < size; i++) {
        if (logits[i] < threshold) logits[i] = -FLT_MAX;
    }
}

void sd_temperature_scale(float* logits, int size, float temperature) {
    if (temperature <= 0.0f) return;
    for (int i = 0; i < size; i++) {
        logits[i] /= temperature;
    }
}

int sd_int_array_hash(const int* arr, int len) {
    int h = 5381;
    for (int i = 0; i < len; i++) {
        h = ((h << 5) + h) ^ arr[i];
    }
    return h;
}

void sd_tree_attention(const float* query, const float* keys, const float* values,
                        int* tree_structure, int num_nodes, int head_dim, float* output) {
    float scale = 1.0f / sqrtf((float)head_dim);
    memset(output, 0, (size_t)head_dim * sizeof(float));
    float* scores = malloc((size_t)num_nodes * sizeof(float));
    float sum = 0.0f;
    for (int i = 0; i < num_nodes; i++) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += query[d] * keys[(size_t)i * head_dim + d];
        }
        scores[i] = expf(dot * scale);
        sum += scores[i];
    }
    for (int i = 0; i < num_nodes; i++) {
        scores[i] /= sum;
        for (int d = 0; d < head_dim; d++) {
            output[d] += scores[i] * values[(size_t)i * head_dim + d];
        }
    }
    free(scores);
    (void)tree_structure;
}

void sd_draft_generate(SD_SpeculativeDecoder* decoder, const int* prefix, int prefix_len) {
    double start = sd_get_time_ms();
    switch (decoder->draft.draft_type) {
    case SD_DRAFT_NGRAM:
        sd_draft_ngram_predict(&decoder->draft.ngram, prefix, prefix_len,
                                decoder->candidates.tokens, decoder->gamma);
        decoder->candidates.count = decoder->gamma;
        break;
    case SD_DRAFT_SMALL_TF:
        sd_draft_small_tf_generate(&decoder->draft.small_tf, prefix, prefix_len,
                                    decoder->candidates.tokens, decoder->gamma, decoder->temperature);
        decoder->candidates.count = decoder->gamma;
        break;
    case SD_DRAFT_MEDUSA:
    case SD_DRAFT_EAGLE:
        decoder->candidates.count = decoder->gamma;
        for (int i = 0; i < decoder->gamma; i++) {
            decoder->candidates.tokens[i] = (int)(rand() % decoder->target.vocab_size);
        }
        break;
    }
    decoder->total_time_draft_ms += sd_get_time_ms() - start;
}

void sd_target_verify(SD_SpeculativeDecoder* decoder, const int* prefix, int prefix_len,
                       int* draft_tokens, int* accepted_tokens, int* num_accepted) {
    (void)prefix; (void)prefix_len;
    double start = sd_get_time_ms();
    int vocab = decoder->target.vocab_size;
    float* draft_logits = malloc((size_t)decoder->gamma * vocab * sizeof(float));
    float* target_logits = malloc((size_t)decoder->gamma * vocab * sizeof(float));
    for (int i = 0; i < decoder->gamma * vocab; i++) {
        draft_logits[i] = (float)rand() / (float)RAND_MAX;
        target_logits[i] = (float)rand() / (float)RAND_MAX;
    }
    *num_accepted = sd_rejection_sample(draft_logits, target_logits, vocab,
                                         draft_tokens, decoder->gamma, accepted_tokens,
                                         decoder->temperature);
    int total = decoder->draft.total_candidates + decoder->gamma;
    decoder->draft.total_candidates = (total > 0) ? total : 1;
    decoder->draft.total_accepted += *num_accepted;
    decoder->draft.acceptance_rate = (float)decoder->draft.total_accepted / (float)decoder->draft.total_candidates;
    decoder->total_time_target_ms += sd_get_time_ms() - start;
    free(draft_logits);
    free(target_logits);
}

int sd_speculative_step(SD_SpeculativeDecoder* decoder, const int* prefix, int prefix_len,
                         int* output_tokens, int max_new_tokens) {
    int generated = 0;
    int* accepted = malloc(SD_MAX_CANDIDATES * sizeof(int));
    int na = 0;
    sd_draft_generate(decoder, prefix, prefix_len);
    sd_target_verify(decoder, prefix, prefix_len, decoder->candidates.tokens, accepted, &na);
    int copy = na < (max_new_tokens - generated) ? na : (max_new_tokens - generated);
    memcpy(output_tokens, accepted, (size_t)copy * sizeof(int));
    generated += copy;
    decoder->total_steps++;
    decoder->total_tokens_generated += generated;
    free(accepted);
    return generated;
}
