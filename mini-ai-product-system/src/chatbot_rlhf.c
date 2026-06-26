#include "chatbot_rlhf.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static uint64_t cb_splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static float cb_randf(uint64_t *state) {
    return (float)(cb_splitmix64(state) & 0xFFFFFF) / 16777216.0f;
}

static float cb_softmax_scalar(const float *logits, int32_t n, int32_t idx) {
    float max_logit = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > max_logit) max_logit = logits[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += expf(logits[i] - max_logit);
    return expf(logits[idx] - max_logit) / (sum + 1e-10f);
}

void cb_base_model_init(CBBaseModel *m, uint64_t seed) {
    float scale = sqrtf(2.0f / CB_HIDDEN_DIM);
    for (int v = 0; v < CB_VOCAB_SIZE; v++)
        for (int d = 0; d < CB_HIDDEN_DIM; d++)
            m->token_emb[v][d] = cb_randf(&seed) * scale * 0.02f;
    for (int p = 0; p < CB_MAX_SEQ_LEN; p++)
        for (int d = 0; d < CB_HIDDEN_DIM; d++)
            m->pos_emb[p][d] = (float)(p) / (float)CB_MAX_SEQ_LEN * 0.01f;
    for (int d = 0; d < CB_HIDDEN_DIM; d++) {
        m->layernorm_gamma[d] = 1.0f;
        m->layernorm_beta[d] = 0.0f;
        m->final_ln_gamma[d] = 1.0f;
        m->final_ln_beta[d] = 0.0f;
    }
    for (int i = 0; i < 3 * CB_HIDDEN_DIM; i++)
        for (int j = 0; j < CB_HIDDEN_DIM; j++)
            m->qkv_weight[i][j] = cb_randf(&seed) * scale * 0.02f;
    for (int i = 0; i < CB_HIDDEN_DIM; i++)
        for (int j = 0; j < CB_HIDDEN_DIM; j++)
            m->attn_proj[i][j] = cb_randf(&seed) * scale * 0.02f;
    for (int i = 0; i < 4 * CB_HIDDEN_DIM; i++)
        for (int j = 0; j < CB_HIDDEN_DIM; j++)
            m->ffn_w1[i][j] = cb_randf(&seed) * scale * 0.02f;
    for (int i = 0; i < CB_HIDDEN_DIM; i++)
        for (int j = 0; j < 4 * CB_HIDDEN_DIM; j++)
            m->ffn_w2[i][j] = cb_randf(&seed) * scale * 0.02f;
    for (int v = 0; v < CB_VOCAB_SIZE; v++)
        for (int d = 0; d < CB_HIDDEN_DIM; d++)
            m->lm_head[v][d] = 0.0f;
}

static void cb_layernorm(const float *x, const float *gamma, const float *beta,
                         int32_t dim, float *y) {
    float mean = 0.0f, var = 0.0f;
    for (int i = 0; i < dim; i++) mean += x[i];
    mean /= (float)dim;
    for (int i = 0; i < dim; i++) { float d = x[i] - mean; var += d * d; }
    var = var / (float)dim + 1e-5f;
    float inv_std = 1.0f / sqrtf(var);
    for (int i = 0; i < dim; i++)
        y[i] = (x[i] - mean) * inv_std * gamma[i] + beta[i];
}

static void cb_gelu(float *x, int32_t n) {
    for (int i = 0; i < n; i++)
        x[i] = 0.5f * x[i] * (1.0f + tanhf(0.79788456f * (x[i] + 0.044715f * x[i] * x[i] * x[i])));
}

void cb_base_model_forward(const CBBaseModel *m, const int32_t *input_ids,
                           int32_t seq_len, float *hidden_states) {
    int dim = CB_HIDDEN_DIM;
    for (int s = 0; s < seq_len; s++) {
        for (int d = 0; d < dim; d++)
            hidden_states[s * dim + d] = m->token_emb[input_ids[s]][d] + m->pos_emb[s][d];
    }
    float *norm = (float *)malloc((size_t)(seq_len * dim) * sizeof(float));
    float *attn = (float *)malloc((size_t)(seq_len * dim) * sizeof(float));
    float *ffn  = (float *)malloc((size_t)(seq_len * dim) * sizeof(float));
    if (!norm || !attn || !ffn) { free(norm); free(attn); free(ffn); return; }

    for (int s = 0; s < seq_len; s++)
        cb_layernorm(&hidden_states[s * dim], m->layernorm_gamma, m->layernorm_beta, dim,
                     &norm[s * dim]);
    for (int s = 0; s < seq_len; s++) {
        for (int d = 0; d < dim; d++) {
            float sum = 0.0f;
            for (int i = 0; i < dim; i++) sum += norm[s * dim + i] * m->qkv_weight[d][i];
            attn[s * dim + d] = sum + hidden_states[s * dim + d];
        }
    }
    for (int s = 0; s < seq_len; s++)
        cb_layernorm(&attn[s * dim], m->layernorm_gamma, m->layernorm_beta, dim, &norm[s * dim]);
    for (int s = 0; s < seq_len; s++) {
        for (int d = 0; d < dim; d++) {
            float sum = 0.0f;
            for (int i = 0; i < dim; i++) sum += norm[s * dim + i] * m->ffn_w1[d][i];
            ffn[s * dim + d] = sum;
        }
    }
    cb_gelu(ffn, seq_len * dim);
    for (int s = 0; s < seq_len; s++) {
        for (int d = 0; d < dim; d++) {
            float sum = 0.0f;
            for (int i = 0; i < 4 * dim; i++)
                sum += ffn[s * dim + (i % dim)] * m->ffn_w2[d][i % (4 * dim)];
            hidden_states[s * dim + d] = attn[s * dim + d] + sum;
        }
    }
    for (int s = 0; s < seq_len; s++)
        cb_layernorm(&hidden_states[s * dim], m->final_ln_gamma, m->final_ln_beta, dim,
                     &hidden_states[s * dim]);
    free(norm); free(attn); free(ffn);
}

void cb_sft_train(CBBaseModel *model, const CBInput *demos, const CBOutput *labels,
                  int32_t num_demos, float lr, int32_t epochs) {
    for (int e = 0; e < epochs; e++) {
        float total_loss = 0.0f;
        for (int i = 0; i < num_demos; i++) {
            float *hs = (float *)malloc((size_t)(labels[i].length * CB_HIDDEN_DIM) * sizeof(float));
            if (!hs) continue;
            cb_base_model_forward(model, labels[i].token_ids, labels[i].length, hs);
            for (int t = 0; t < labels[i].length; t++) {
                float logits[4] = {0};
                for (int v = 0; v < 4 && v < CB_VOCAB_SIZE; v++) {
                    for (int d = 0; d < CB_HIDDEN_DIM; d++)
                        logits[v] += hs[t * CB_HIDDEN_DIM + d] * model->lm_head[v][d];
                }
                float prob = cb_softmax_scalar(logits, 4, labels[i].token_ids[t] % 4);
                total_loss += -logf(prob + 1e-10f);
                for (int d = 0; d < CB_HIDDEN_DIM && d < 64; d++)
                    model->token_emb[labels[i].token_ids[t] % CB_VOCAB_SIZE][d] -=
                        lr * (prob - 1.0f) * hs[t * CB_HIDDEN_DIM + d] * 0.01f;
            }
            free(hs);
        }
        (void)total_loss;
    }
}

void cb_reward_model_init(CBRewardModel *rm, uint64_t seed) {
    cb_base_model_init(&rm->base, seed);
    for (int i = 0; i < CB_HIDDEN_DIM * CB_HIDDEN_DIM; i++)
        rm->reward_head.w[i] = cb_randf(&seed) * 0.01f;
    rm->reward_head.bias = 0.0f;
    for (int d = 0; d < CB_HIDDEN_DIM; d++) {
        rm->reward_head.layernorm_gamma[d] = 1.0f;
        rm->reward_head.layernorm_beta[d] = 0.0f;
    }
    rm->reward_head.head_weight[0] = cb_randf(&seed) * 0.01f;
    rm->reward_head.head_bias = 0.0f;
    for (int d = 1; d < CB_HIDDEN_DIM; d++)
        rm->reward_head.head_weight[d] = rm->reward_head.head_weight[0];
}

float cb_reward_model_forward(const CBRewardModel *rm,
                              const int32_t *prompt_ids, int32_t prompt_len,
                              const int32_t *resp_ids, int32_t resp_len) {
    int32_t total = prompt_len + resp_len;
    int32_t *ids = (int32_t *)malloc((size_t)total * sizeof(int32_t));
    memcpy(ids, prompt_ids, (size_t)prompt_len * sizeof(int32_t));
    memcpy(ids + prompt_len, resp_ids, (size_t)resp_len * sizeof(int32_t));
    float *hs = (float *)malloc((size_t)(total * CB_HIDDEN_DIM) * sizeof(float));
    cb_base_model_forward(&rm->base, ids, total, hs);
    int last = total - 1;
    float reward = rm->reward_head.head_bias;
    for (int d = 0; d < CB_HIDDEN_DIM && d < CB_HIDDEN_DIM; d++)
        reward += hs[last * CB_HIDDEN_DIM + d] * rm->reward_head.head_weight[d];
    free(ids);
    free(hs);
    return 1.0f / (1.0f + expf(-reward));
}

void cb_reward_model_train(CBRewardModel *rm, const CBPreferencePair *pairs,
                           int32_t num_pairs, float lr, int32_t epochs) {
    for (int e = 0; e < epochs; e++) {
        for (int i = 0; i < num_pairs; i++) {
            float r_chosen = cb_reward_model_forward(rm,
                pairs[i].prompt.input_ids, pairs[i].prompt.input_len,
                pairs[i].chosen.token_ids, pairs[i].chosen.length);
            float r_rejected = cb_reward_model_forward(rm,
                pairs[i].prompt.input_ids, pairs[i].prompt.input_len,
                pairs[i].rejected.token_ids, pairs[i].rejected.length);
            float loss = -logf(1.0f / (1.0f + expf(r_rejected - r_chosen)) + 1e-10f);
            (void)loss;
            float grad = 1.0f / (1.0f + expf(r_chosen - r_rejected));
            rm->reward_head.head_bias += lr * grad * 0.01f;
            for (int d = 0; d < CB_HIDDEN_DIM; d++)
                rm->reward_head.head_weight[d] += lr * grad * 0.001f;
        }
    }
}

void cb_ppo_config_init(CBPPOConfig *cfg) {
    cfg->kl_coef = 0.1f;
    cfg->clip_epsilon = 0.2f;
    cfg->gamma_discount = 0.99f;
    cfg->gae_lambda = 0.95f;
    cfg->batch_size = 64;
    cfg->ppo_epochs = 4;
    cfg->learning_rate = 1e-5f;
    cfg->value_coef = 0.5f;
    cfg->entropy_coef = 0.01f;
    cfg->max_grad_norm = 1.0f;
}

void cb_ppo_advantage(const float *rewards, const float *values, int32_t len,
                      const CBPPOConfig *cfg, float *advantages, float *returns) {
    float *deltas = (float *)malloc((size_t)len * sizeof(float));
    for (int t = 0; t < len; t++) {
        float next_val = (t < len - 1) ? values[t + 1] : 0.0f;
        deltas[t] = rewards[t] + cfg->gamma_discount * next_val - values[t];
    }
    float running = 0.0f;
    for (int t = len - 1; t >= 0; t--) {
        running = deltas[t] + cfg->gamma_discount * cfg->gae_lambda * running;
        advantages[t] = running;
    }
    for (int t = 0; t < len; t++)
        returns[t] = advantages[t] + values[t];
    free(deltas);
}

void cb_ppo_policy_loss(const CBBaseModel *policy, const CBBaseModel *ref,
                        const int32_t *input_ids, int32_t seq_len,
                        const float *advantages, const float *old_logprobs,
                        const CBPPOConfig *cfg, float *loss) {
    float *hs = (float *)malloc((size_t)(seq_len * CB_HIDDEN_DIM) * sizeof(float));
    cb_base_model_forward(policy, input_ids, seq_len, hs);
    float kl = 0.0f;
    float policy_loss = 0.0f;
    int dim = CB_HIDDEN_DIM < 64 ? CB_HIDDEN_DIM : 64;
    for (int t = 0; t < seq_len; t++) {
        float ratio = 1.0f;
        float clipped = ratio < 1.0f - cfg->clip_epsilon ? 1.0f - cfg->clip_epsilon :
                        ratio > 1.0f + cfg->clip_epsilon ? 1.0f + cfg->clip_epsilon : ratio;
        float surr1 = ratio * advantages[t];
        float surr2 = clipped * advantages[t];
        policy_loss -= (surr1 < surr2 ? surr1 : surr2);
        for (int d = 0; d < dim; d++)
            kl += hs[t * CB_HIDDEN_DIM + d] * hs[t * CB_HIDDEN_DIM + d] * 0.001f;
    }
    *loss = policy_loss / (float)seq_len + cfg->kl_coef * kl;
    free(hs);
}

float cb_kl_divergence(const float *logprobs_policy, const float *logprobs_ref,
                       int32_t len) {
    float kl = 0.0f;
    for (int i = 0; i < len; i++)
        kl += expf(logprobs_policy[i]) * (logprobs_policy[i] - logprobs_ref[i]);
    return kl;
}

void cb_ppo_update(CBBaseModel *policy, const CBBaseModel *ref,
                   const CBRewardModel *rm, const CBInput *prompts,
                   int32_t num_prompts, const CBPPOConfig *cfg) {
    for (int e = 0; e < cfg->ppo_epochs; e++) {
        for (int i = 0; i < num_prompts; i++) {
            CBOutput output;
            cb_generate(policy, prompts[i].input_ids, prompts[i].input_len, 32, 1.0f,
                        (uint64_t)(i * 100 + e), &output);
            float *rewards = (float *)malloc((size_t)(output.length) * sizeof(float));
            float *values  = (float *)malloc((size_t)(output.length) * sizeof(float));
            float *adv     = (float *)malloc((size_t)(output.length) * sizeof(float));
            float *rets    = (float *)malloc((size_t)(output.length) * sizeof(float));
            if (!rewards || !values || !adv || !rets) {
                free(rewards); free(values); free(adv); free(rets); continue;
            }
            for (int t = 0; t < output.length; t++) {
                rewards[t] = cb_reward_model_forward(rm, prompts[i].input_ids,
                    prompts[i].input_len, output.token_ids, t + 1);
                values[t] = 0.5f;
            }
            cb_ppo_advantage(rewards, values, output.length, cfg, adv, rets);
            float loss = 0.0f;
            cb_ppo_policy_loss(policy, ref, output.token_ids, output.length, adv,
                               output.logprobs, cfg, &loss);
            for (int d = 0; d < CB_HIDDEN_DIM; d++)
                policy->token_emb[output.token_ids[0] % CB_VOCAB_SIZE][d] -=
                    cfg->learning_rate * loss * 0.001f;
            free(rewards); free(values); free(adv); free(rets);
        }
    }
}

void cb_dpo_train(CBBaseModel *model, const CBPreferencePair *pairs,
                  int32_t num_pairs, const CBDPOConfig *cfg) {
    CBBaseModel ref;
    memcpy(&ref, model, sizeof(CBBaseModel));
    for (int e = 0; e < cfg->epochs; e++) {
        for (int i = 0; i < num_pairs; i++) {
            float loss = cb_dpo_loss(model, &pairs[i], cfg->dpo_beta);
            for (int d = 0; d < CB_HIDDEN_DIM; d++)
                model->token_emb[pairs[i].chosen.token_ids[0] % CB_VOCAB_SIZE][d] -=
                    cfg->learning_rate * loss * 0.001f;
        }
    }
}

float cb_dpo_loss(const CBBaseModel *model, const CBPreferencePair *pair, float beta) {
    float log_p_chosen = 0.0f, log_p_rejected = 0.0f;
    for (int t = 0; t < pair->chosen.length && t < CB_MAX_RESPONSE_LEN; t++)
        log_p_chosen += pair->chosen.logprobs[t];
    for (int t = 0; t < pair->rejected.length && t < CB_MAX_RESPONSE_LEN; t++)
        log_p_rejected += pair->rejected.logprobs[t];
    float diff = beta * (log_p_chosen - log_p_rejected);
    return -logf(1.0f / (1.0f + expf(-diff)) + 1e-10f);
}

float cb_implicit_reward(const CBBaseModel *model, const CBBaseModel *ref,
                         const int32_t *resp_ids, int32_t resp_len, float beta) {
    float log_p_model = 0.0f, log_p_ref = 0.0f;
    for (int t = 0; t < resp_len && t < CB_MAX_RESPONSE_LEN; t++) {
        log_p_model += (float)(resp_ids[t] % 100) * 0.001f;
        log_p_ref += (float)(resp_ids[t] % 100) * 0.001f;
    }
    return beta * (log_p_model - log_p_ref);
}

void cb_ai_feedback_generate(const CBBaseModel *model, const CBInput *prompt,
                             CBAIFeedback *feedback) {
    feedback->helpfulness = 0.7f;
    feedback->harmlessness = 0.85f;
    feedback->honesty = 0.9f;
    feedback->critique = "Acceptable response, minor improvements needed.";
    (void)model;
    (void)prompt;
}

void cb_constitutional_alignment(CBBaseModel *model, const CBInput *prompts,
                                 int32_t num_prompts, CBAIFeedbackType type) {
    for (int i = 0; i < num_prompts; i++) {
        CBAIFeedback fb;
        cb_ai_feedback_generate(model, &prompts[i], &fb);
        float target = 0.0f;
        switch (type) {
            case CB_AI_FEEDBACK_HELPFUL:  target = fb.helpfulness; break;
            case CB_AI_FEEDBACK_HARMLESS: target = fb.harmlessness; break;
            case CB_AI_FEEDBACK_HONEST:   target = fb.honesty; break;
        }
        for (int d = 0; d < CB_HIDDEN_DIM; d++)
            model->token_emb[0][d] += (target - 0.5f) * 0.001f;
    }
}

void cb_safety_filter(const int32_t *resp_ids, int32_t resp_len,
                      const CBRedTeamSet *red_team, int *is_safe) {
    *is_safe = 1;
    for (int i = 0; i < red_team->num_phrases && *is_safe; i++) {
        size_t plen = strlen(red_team->phrases[i]);
        for (int t = 0; t < resp_len; t++) {
            if (resp_ids[t] == (int32_t)red_team->phrases[i][0]) {
                *is_safe = 0;
                break;
            }
        }
        (void)plen;
    }
}

float cb_harmlessness_score(const CBOutput *response, const CBRedTeamSet *red_team) {
    int safe = 1;
    cb_safety_filter(response->token_ids, response->length, red_team, &safe);
    return safe ? 1.0f : 0.0f;
}

void cb_safety_alignment_loss(const CBBaseModel *model, const CBInput *prompts,
                              int32_t num_prompts, const CBRedTeamSet *red_team,
                              float *loss) {
    *loss = 0.0f;
    for (int i = 0; i < num_prompts; i++) {
        CBOutput out;
        cb_generate(model, prompts[i].input_ids, prompts[i].input_len, 64, 0.7f,
                    (uint64_t)i, &out);
        float score = cb_harmlessness_score(&out, red_team);
        *loss += 1.0f - score;
    }
    *loss /= (float)(num_prompts + 1);
}

void cb_generate(const CBBaseModel *model, const int32_t *input_ids, int32_t input_len,
                 int32_t max_new_tokens, float temperature, uint64_t seed,
                 CBOutput *output) {
    if (max_new_tokens > CB_MAX_RESPONSE_LEN) max_new_tokens = CB_MAX_RESPONSE_LEN;
    int32_t seq[CB_MAX_SEQ_LEN];
    int32_t seq_len = input_len < CB_MAX_SEQ_LEN ? input_len : CB_MAX_SEQ_LEN;
    memcpy(seq, input_ids, (size_t)seq_len * sizeof(int32_t));
    float *hs = (float *)malloc((size_t)(CB_MAX_SEQ_LEN * CB_HIDDEN_DIM) * sizeof(float));
    output->length = 0;
    output->total_logprob = 0.0f;
    temperature = temperature < 0.01f ? 1.0f : temperature;

    for (int step = 0; step < max_new_tokens && seq_len < CB_MAX_SEQ_LEN; step++) {
        cb_base_model_forward(model, seq, seq_len, hs);
        int last = seq_len - 1;
        float logits[4] = {0};
        for (int v = 0; v < 4 && v < CB_VOCAB_SIZE; v++) {
            for (int d = 0; d < CB_HIDDEN_DIM; d++)
                logits[v] += hs[last * CB_HIDDEN_DIM + d] * model->lm_head[v][d];
            logits[v] /= temperature;
        }
        int32_t next_token = 0;
        float r = cb_randf(&seed);
        float cum = 0.0f;
        for (int v = 0; v < 4 && v < CB_VOCAB_SIZE; v++) {
            float p = expf(logits[v]);
            cum += p;
            if (r < cum / (cum + 1e-10f)) { next_token = (int32_t)v; break; }
            next_token = (int32_t)v;
        }
        seq[seq_len++] = next_token;
        output->token_ids[output->length] = next_token;
        output->logprobs[output->length] = logf(cb_softmax_scalar(logits, 4, next_token) + 1e-10f);
        output->total_logprob += output->logprobs[output->length];
        output->length++;
    }
    free(hs);
}

void cb_tokenize(const char *text, int32_t *token_ids, int32_t *len, int32_t max_len) {
    *len = 0;
    for (const char *p = text; *p && *len < max_len; p++)
        token_ids[(*len)++] = (int32_t)(unsigned char)*p % CB_VOCAB_SIZE;
}

void cb_detokenize(const int32_t *token_ids, int32_t len, char *text, int32_t max_text) {
    int n = len < max_text - 1 ? len : max_text - 1;
    for (int i = 0; i < n; i++)
        text[i] = (char)(token_ids[i] % 256);
    text[n] = '\0';
}
