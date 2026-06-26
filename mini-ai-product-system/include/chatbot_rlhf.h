#ifndef CHATBOT_RLHF_H
#define CHATBOT_RLHF_H

#include <stddef.h>
#include <stdint.h>

#define CB_MAX_SEQ_LEN      2048
#define CB_VOCAB_SIZE       50257
#define CB_HIDDEN_DIM       768
#define CB_NUM_LAYERS       12
#define CB_NUM_HEADS        12
#define CB_MAX_PROMPT_LEN   1024
#define CB_MAX_RESPONSE_LEN 512

typedef struct {
    float token_emb[CB_VOCAB_SIZE][CB_HIDDEN_DIM];
    float pos_emb[CB_MAX_SEQ_LEN][CB_HIDDEN_DIM];
    float layernorm_gamma[CB_HIDDEN_DIM];
    float layernorm_beta[CB_HIDDEN_DIM];
    float qkv_weight[3 * CB_HIDDEN_DIM][CB_HIDDEN_DIM];
    float attn_proj[CB_HIDDEN_DIM][CB_HIDDEN_DIM];
    float ffn_w1[4 * CB_HIDDEN_DIM][CB_HIDDEN_DIM];
    float ffn_w2[CB_HIDDEN_DIM][4 * CB_HIDDEN_DIM];
    float final_ln_gamma[CB_HIDDEN_DIM];
    float final_ln_beta[CB_HIDDEN_DIM];
    float lm_head[CB_VOCAB_SIZE][CB_HIDDEN_DIM];
} CBBaseModel;

typedef struct {
    int32_t input_ids[CB_MAX_PROMPT_LEN];
    int32_t input_len;
} CBInput;

typedef struct {
    int32_t token_ids[CB_MAX_RESPONSE_LEN];
    int32_t length;
    float   logprobs[CB_MAX_RESPONSE_LEN];
    float   total_logprob;
} CBOutput;

typedef struct {
    CBInput  prompt;
    CBOutput chosen;
    CBOutput rejected;
} CBPreferencePair;

typedef struct {
    float w[CB_HIDDEN_DIM * CB_HIDDEN_DIM];
    float bias;
    float layernorm_gamma[CB_HIDDEN_DIM];
    float layernorm_beta[CB_HIDDEN_DIM];
    float head_weight[CB_HIDDEN_DIM];
    float head_bias;
} CBRewardHead;

typedef struct {
    CBBaseModel   base;
    CBRewardHead  reward_head;
} CBRewardModel;

typedef struct {
    CBBaseModel policy_model;
    CBBaseModel reference_model;
    float       kl_coef;
    float       clip_epsilon;
    float       gamma_discount;
    float       gae_lambda;
    int32_t     batch_size;
    int32_t     ppo_epochs;
    float       learning_rate;
    float       value_coef;
    float       entropy_coef;
    float       max_grad_norm;
} CBPPOConfig;

typedef struct {
    CBBaseModel model;
    float       dpo_beta;
    float       learning_rate;
    int32_t     epochs;
} CBDPOConfig;

typedef enum {
    CB_AI_FEEDBACK_HELPFUL,
    CB_AI_FEEDBACK_HARMLESS,
    CB_AI_FEEDBACK_HONEST
} CBAIFeedbackType;

typedef struct {
    CBInput     prompt;
    CBOutput    response;
    float       helpfulness;
    float       harmlessness;
    float       honesty;
    const char *critique;
} CBAIFeedback;

typedef struct {
    const char **phrases;
    int32_t      num_phrases;
} CBRedTeamSet;

void cb_base_model_init(CBBaseModel *m, uint64_t seed);
void cb_base_model_forward(const CBBaseModel *m, const int32_t *input_ids,
                           int32_t seq_len, float *hidden_states);

void cb_sft_train(CBBaseModel *model, const CBInput *demos, const CBOutput *labels,
                  int32_t num_demos, float lr, int32_t epochs);

void cb_reward_model_init(CBRewardModel *rm, uint64_t seed);
float cb_reward_model_forward(const CBRewardModel *rm,
                              const int32_t *prompt_ids, int32_t prompt_len,
                              const int32_t *resp_ids, int32_t resp_len);
void cb_reward_model_train(CBRewardModel *rm, const CBPreferencePair *pairs,
                           int32_t num_pairs, float lr, int32_t epochs);

void cb_ppo_config_init(CBPPOConfig *cfg);
void cb_ppo_advantage(const float *rewards, const float *values, int32_t len,
                      const CBPPOConfig *cfg, float *advantages, float *returns);
void cb_ppo_policy_loss(const CBBaseModel *policy, const CBBaseModel *ref,
                        const int32_t *input_ids, int32_t seq_len,
                        const float *advantages, const float *old_logprobs,
                        const CBPPOConfig *cfg, float *loss);
float cb_kl_divergence(const float *logprobs_policy, const float *logprobs_ref,
                       int32_t len);
void cb_ppo_update(CBBaseModel *policy, const CBBaseModel *ref,
                   const CBRewardModel *rm, const CBInput *prompts,
                   int32_t num_prompts, const CBPPOConfig *cfg);

void cb_dpo_train(CBBaseModel *model, const CBPreferencePair *pairs,
                  int32_t num_pairs, const CBDPOConfig *cfg);
float cb_dpo_loss(const CBBaseModel *model, const CBPreferencePair *pair,
                  float beta);
float cb_implicit_reward(const CBBaseModel *model, const CBBaseModel *ref,
                         const int32_t *resp_ids, int32_t resp_len, float beta);

void cb_ai_feedback_generate(const CBBaseModel *model, const CBInput *prompt,
                             CBAIFeedback *feedback);
void cb_constitutional_alignment(CBBaseModel *model, const CBInput *prompts,
                                 int32_t num_prompts, CBAIFeedbackType type);

void cb_safety_filter(const int32_t *resp_ids, int32_t resp_len,
                      const CBRedTeamSet *red_team, int *is_safe);
float cb_harmlessness_score(const CBOutput *response, const CBRedTeamSet *red_team);
void cb_safety_alignment_loss(const CBBaseModel *model, const CBInput *prompts,
                              int32_t num_prompts, const CBRedTeamSet *red_team,
                              float *loss);

void cb_generate(const CBBaseModel *model, const int32_t *input_ids, int32_t input_len,
                 int32_t max_new_tokens, float temperature, uint64_t seed,
                 CBOutput *output);
void cb_tokenize(const char *text, int32_t *token_ids, int32_t *len, int32_t max_len);
void cb_detokenize(const int32_t *token_ids, int32_t len, char *text, int32_t max_text);

#endif
