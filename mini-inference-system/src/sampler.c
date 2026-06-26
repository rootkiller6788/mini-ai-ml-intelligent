#include "sampler.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

void samp_softmax(float* probs, const float* logits, int size, float temperature) {
    float max_logit = -FLT_MAX;
    for (int i = 0; i < size; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }
    float sum = 0.0f;
    float inv_temp = (temperature > 0.0f) ? 1.0f / temperature : 1.0f;
    for (int i = 0; i < size; i++) {
        probs[i] = expf((logits[i] - max_logit) * inv_temp);
        sum += probs[i];
    }
    float inv_sum = (sum > 0.0f) ? 1.0f / sum : 1.0f;
    for (int i = 0; i < size; i++) probs[i] *= inv_sum;
}

static int samp_argmax(const float* arr, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) {
        if (arr[i] > arr[best]) best = i;
    }
    return best;
}

static int samp_random_draw(const float* probs, int n) {
    float r = (float)rand() / (float)RAND_MAX;
    float cum = 0.0f;
    for (int i = 0; i < n; i++) {
        cum += probs[i];
        if (r <= cum) return i;
    }
    return n - 1;
}

int samp_greedy(const float* logits, int vocab_size) {
    return samp_argmax(logits, vocab_size);
}

int samp_temperature(const float* logits, int vocab_size, float temperature) {
    if (temperature <= 0.0f) return samp_greedy(logits, vocab_size);
    float* probs = malloc((size_t)vocab_size * sizeof(float));
    samp_softmax(probs, logits, vocab_size, temperature);
    int chosen = samp_random_draw(probs, vocab_size);
    free(probs);
    return chosen;
}

int samp_top_k(const float* logits, int vocab_size, int k, float temperature) {
    if (k >= vocab_size) return samp_temperature(logits, vocab_size, temperature);
    if (k <= 0) return samp_greedy(logits, vocab_size);
    float* copy = malloc((size_t)vocab_size * sizeof(float));
    memcpy(copy, logits, (size_t)vocab_size * sizeof(float));
    float* sorted = malloc((size_t)k * sizeof(float));
    for (int i = 0; i < k; i++) {
        float best = -FLT_MAX;
        for (int j = 0; j < vocab_size; j++) {
            if (copy[j] > best) best = copy[j];
        }
        sorted[i] = best;
        for (int j = 0; j < vocab_size; j++) {
            if (copy[j] == best) { copy[j] = -FLT_MAX; break; }
        }
    }
    float threshold = sorted[k - 1];
    free(sorted);
    float* masked = malloc((size_t)vocab_size * sizeof(float));
    memcpy(masked, logits, (size_t)vocab_size * sizeof(float));
    for (int i = 0; i < vocab_size; i++) {
        if (logits[i] < threshold) masked[i] = -FLT_MAX;
    }
    float* probs = malloc((size_t)vocab_size * sizeof(float));
    samp_softmax(probs, masked, vocab_size, temperature);
    int chosen = samp_random_draw(probs, vocab_size);
    free(copy); free(masked); free(probs);
    return chosen;
}

int samp_top_p(const float* logits, int vocab_size, float p, float temperature) {
    if (p >= 1.0f) return samp_temperature(logits, vocab_size, temperature);
    if (p <= 0.0f) return samp_greedy(logits, vocab_size);
    float* probs = malloc((size_t)vocab_size * sizeof(float));
    samp_softmax(probs, logits, vocab_size, temperature);
    int* indices = malloc((size_t)vocab_size * sizeof(int));
    float* sorted = malloc((size_t)vocab_size * sizeof(float));
    for (int i = 0; i < vocab_size; i++) { indices[i] = i; sorted[i] = probs[i]; }
    for (int i = 0; i < vocab_size; i++) {
        for (int j = i + 1; j < vocab_size; j++) {
            if (sorted[j] > sorted[i]) {
                float tmp_f = sorted[i]; sorted[i] = sorted[j]; sorted[j] = tmp_f;
                int tmp_i = indices[i]; indices[i] = indices[j]; indices[j] = tmp_i;
            }
        }
    }
    float cum = 0.0f;
    int cutoff = vocab_size;
    for (int i = 0; i < vocab_size; i++) {
        cum += sorted[i];
        if (cum >= p) { cutoff = i + 1; break; }
    }
    float* truncated = calloc((size_t)vocab_size, sizeof(float));
    float nucleus_sum = 0.0f;
    for (int i = 0; i < cutoff; i++) {
        truncated[indices[i]] = sorted[i];
        nucleus_sum += sorted[i];
    }
    if (nucleus_sum > 0.0f) {
        for (int i = 0; i < cutoff; i++) {
            truncated[indices[i]] /= nucleus_sum;
        }
    }
    int chosen = samp_random_draw(truncated, vocab_size);
    free(probs); free(indices); free(sorted); free(truncated);
    return chosen;
}

int samp_min_p(const float* logits, int vocab_size, float min_p, float temperature) {
    if (min_p <= 0.0f) return samp_greedy(logits, vocab_size);
    float* probs = malloc((size_t)vocab_size * sizeof(float));
    samp_softmax(probs, logits, vocab_size, temperature);
    float max_p = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        if (probs[i] > max_p) max_p = probs[i];
    }
    float threshold = max_p * min_p;
    float* filtered = calloc((size_t)vocab_size, sizeof(float));
    float filtered_sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        if (probs[i] >= threshold) {
            filtered[i] = probs[i];
            filtered_sum += probs[i];
        }
    }
    if (filtered_sum > 0.0f) {
        for (int i = 0; i < vocab_size; i++) {
            if (filtered[i] > 0.0f) filtered[i] /= filtered_sum;
        }
    }
    int chosen = samp_random_draw(filtered, vocab_size);
    free(probs); free(filtered);
    return chosen;
}

int samp_typical(const float* logits, int vocab_size, float mass, float temperature) {
    float* probs = malloc((size_t)vocab_size * sizeof(float));
    samp_softmax(probs, logits, vocab_size, temperature);
    float entropy = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        if (probs[i] > 1e-9f) {
            entropy -= probs[i] * logf(probs[i]);
        }
    }
    float* typical = calloc((size_t)vocab_size, sizeof(float));
    float typical_sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        float logp = (probs[i] > 1e-9f) ? logf(probs[i]) : -20.0f;
        if (fabsf(logp + entropy) <= mass) {
            typical[i] = probs[i];
            typical_sum += probs[i];
        }
    }
    if (typical_sum > 0.0f) {
        for (int i = 0; i < vocab_size; i++) {
            if (typical[i] > 0.0f) typical[i] /= typical_sum;
        }
    }
    int chosen = samp_random_draw(typical, vocab_size);
    free(probs); free(typical);
    return chosen;
}

void samp_beam_init(Samp_BeamSearch* bs, int beam_width, int max_length,
                     int eos_token_id, int vocab_size, float length_penalty) {
    memset(bs, 0, sizeof(Samp_BeamSearch));
    bs->beam_width = beam_width;
    bs->num_beams  = 1;
    bs->max_length = max_length;
    bs->eos_token_id = eos_token_id;
    bs->length_penalty = length_penalty;
    bs->vocab_size = vocab_size;
    bs->beams = malloc((size_t)beam_width * sizeof(Samp_BeamHypothesis));
    for (int i = 0; i < beam_width; i++) {
        bs->beams[i].token_sequence = malloc((size_t)max_length * sizeof(int));
        bs->beams[i].length = 0;
        bs->beams[i].cumulative_log_prob = 0.0f;
        bs->beams[i].finished = false;
    }
}

void samp_beam_destroy(Samp_BeamSearch* bs) {
    for (int i = 0; i < bs->beam_width; i++) {
        free(bs->beams[i].token_sequence);
    }
    free(bs->beams);
}

typedef struct { float score; int beam_idx; int token; } BeamCandidate;

static int beam_cmp_desc(const void* a, const void* b) {
    float d = ((const BeamCandidate*)b)->score - ((const BeamCandidate*)a)->score;
    return (d > 0.0f) ? 1 : ((d < 0.0f) ? -1 : 0);
}

int samp_beam_step(Samp_BeamSearch* bs, const float* logits, int step) {
    if (step >= bs->max_length) return 1;
    int max_candidates = bs->beam_width * bs->beam_width;
    BeamCandidate* candidates = malloc((size_t)max_candidates * sizeof(BeamCandidate));
    int nc = 0;
    for (int b = 0; b < bs->num_beams; b++) {
        if (bs->beams[b].finished) continue;
        const float* beam_logits = logits + (size_t)b * bs->vocab_size;
        float* probs = malloc((size_t)bs->vocab_size * sizeof(float));
        samp_softmax(probs, beam_logits, bs->vocab_size, 1.0f);
        int take = bs->beam_width < bs->vocab_size ? bs->beam_width : bs->vocab_size;
        int* top_indices = malloc((size_t)take * sizeof(int));
        float* top_probs = malloc((size_t)take * sizeof(float));
        int* used = calloc((size_t)bs->vocab_size, sizeof(int));
        for (int i = 0; i < take; i++) {
            float best_p = -1.0f;
            int best_idx = 0;
            for (int j = 0; j < bs->vocab_size; j++) {
                if (!used[j] && probs[j] > best_p) {
                    best_p = probs[j];
                    best_idx = j;
                }
            }
            used[best_idx] = 1;
            top_indices[i] = best_idx;
            top_probs[i] = best_p;
        }
        free(used);
        for (int i = 0; i < take; i++) {
            if (top_probs[i] <= 1e-9f) continue;
            float logp = logf(top_probs[i]);
            float lp = powf((5.0f + (float)bs->beams[b].length + 1.0f),
                             bs->length_penalty) /
                       powf(6.0f, bs->length_penalty);
            candidates[nc].score    = (bs->beams[b].cumulative_log_prob + logp) / lp;
            candidates[nc].beam_idx = b;
            candidates[nc].token    = top_indices[i];
            nc++;
        }
        free(probs); free(top_indices); free(top_probs);
    }
    if (nc == 0) { free(candidates); return 1; }
    qsort(candidates, (size_t)nc, sizeof(BeamCandidate), beam_cmp_desc);
    int keep = nc < bs->beam_width ? nc : bs->beam_width;
    Samp_BeamHypothesis* new_beams = malloc((size_t)keep * sizeof(Samp_BeamHypothesis));
    for (int i = 0; i < keep; i++) {
        int src_b = candidates[i].beam_idx;
        new_beams[i].token_sequence = malloc((size_t)bs->max_length * sizeof(int));
        memcpy(new_beams[i].token_sequence, bs->beams[src_b].token_sequence,
               (size_t)bs->beams[src_b].length * sizeof(int));
        new_beams[i].token_sequence[bs->beams[src_b].length] = candidates[i].token;
        new_beams[i].length = bs->beams[src_b].length + 1;
        new_beams[i].cumulative_log_prob = candidates[i].score;
        new_beams[i].token_id = candidates[i].token;
        new_beams[i].finished = (candidates[i].token == bs->eos_token_id);
    }
    for (int i = 0; i < bs->num_beams; i++) {
        free(bs->beams[i].token_sequence);
    }
    memcpy(bs->beams, new_beams, (size_t)keep * sizeof(Samp_BeamHypothesis));
    bs->num_beams = keep;
    free(new_beams);
    free(candidates);
    return 0;
}

int samp_beam_best(const Samp_BeamSearch* bs) {
    if (bs->num_beams == 0) return 0;
    float best_score = -FLT_MAX;
    int best = 0;
    for (int i = 0; i < bs->num_beams; i++) {
        if (bs->beams[i].cumulative_log_prob > best_score) {
            best_score = bs->beams[i].cumulative_log_prob;
            best = i;
        }
    }
    return best;
}

bool samp_beam_all_finished(const Samp_BeamSearch* bs) {
    for (int i = 0; i < bs->num_beams; i++) {
        if (!bs->beams[i].finished) return false;
    }
    return true;
}

Samp_Config samp_config_default(void) {
    Samp_Config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.strategy      = SAMP_TOP_P;
    cfg.temperature   = SAMP_DEFAULT_TEMP;
    cfg.top_k         = SAMP_DEFAULT_TOPK;
    cfg.top_p         = SAMP_DEFAULT_TOPP;
    cfg.min_p         = 0.05f;
    cfg.typical_mass  = 0.2f;
    cfg.beam_width    = 4;
    cfg.length_penalty = 1.0f;
    cfg.eos_token_id  = 2;
    cfg.pad_token_id  = 0;
    return cfg;
}

bool samp_config_validate(const Samp_Config* cfg) {
    if (!cfg) return false;
    if (cfg->temperature < 0.0f) return false;
    if (cfg->top_k < 0) return false;
    if (cfg->top_p < 0.0f || cfg->top_p > 1.0f) return false;
    if (cfg->beam_width < 1 || cfg->beam_width > SAMP_MAX_BEAM) return false;
    return true;
}

int samp_generate(const float* logits_matrix, int seq_len, int vocab_size,
                   int* output_tokens, int max_new_tokens, const Samp_Config* cfg) {
    if (!logits_matrix || !output_tokens || !cfg) return 0;
    if (!samp_config_validate(cfg)) return 0;
    int num_generated = 0;
    if (cfg->strategy == SAMP_BEAM_SEARCH) {
        Samp_BeamSearch bs;
        samp_beam_init(&bs, cfg->beam_width, max_new_tokens, cfg->eos_token_id,
                        vocab_size, cfg->length_penalty);
        for (int step = 0; step < max_new_tokens; step++) {
            const float* step_logits = logits_matrix +
                (size_t)step * vocab_size * bs.num_beams;
            if (samp_beam_step(&bs, step_logits, step)) break;
            if (samp_beam_all_finished(&bs)) break;
            num_generated = step + 1;
        }
        int best = samp_beam_best(&bs);
        if (best >= 0 && best < bs.num_beams) {
            int copy_len = bs.beams[best].length < max_new_tokens ?
                           bs.beams[best].length : max_new_tokens;
            memcpy(output_tokens, bs.beams[best].token_sequence,
                   (size_t)copy_len * sizeof(int));
            num_generated = copy_len;
        }
        samp_beam_destroy(&bs);
    } else {
        for (int step = 0; step < max_new_tokens; step++) {
            const float* step_logits = logits_matrix + (size_t)step * vocab_size;
            int token;
            switch (cfg->strategy) {
            case SAMP_GREEDY:
                token = samp_greedy(step_logits, vocab_size); break;
            case SAMP_TEMPERATURE:
                token = samp_temperature(step_logits, vocab_size, cfg->temperature); break;
            case SAMP_TOP_K:
                token = samp_top_k(step_logits, vocab_size, cfg->top_k, cfg->temperature); break;
            case SAMP_TOP_P:
                token = samp_top_p(step_logits, vocab_size, cfg->top_p, cfg->temperature); break;
            case SAMP_MIN_P:
                token = samp_min_p(step_logits, vocab_size, cfg->min_p, cfg->temperature); break;
            case SAMP_TYPICAL:
                token = samp_typical(step_logits, vocab_size, cfg->typical_mass, cfg->temperature);
                break;
            default:
                token = samp_greedy(step_logits, vocab_size); break;
            }
            output_tokens[step] = token;
            num_generated = step + 1;
            if (token == cfg->eos_token_id) break;
        }
    }
    (void)seq_len;
    return num_generated;
}
