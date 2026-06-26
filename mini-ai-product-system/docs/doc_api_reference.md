# API Reference — mini-ai-product-system

## Common Types

All modules use C99 conventions: functions prefixed by module abbreviated name,
public types defined in headers, implementation in `.c` files.

### Naming Convention

| Module | Prefix | Header |
|--------|--------|--------|
| Recommendation | `rec_` | `recommendation.h` |
| ChatBot RLHF | `cb_` | `chatbot_rlhf.h` |
| Copilot Context | `cc_` | `copilot_context.h` |
| Eval & Monitor | `ev_` | `eval_monitor.h` |
| Model A/B | `mab_` | `model_ab_test.h` |

---

## Recommendation (`recommendation.h`)

### Embeddings

```c
void rec_embedding_init(RecEmbedding *e, uint64_t seed);
float rec_dot_product(const RecEmbedding *a, const RecEmbedding *b);
```

Initialize a `REC_EMBED_DIM`-dimensional embedding with random Xavier-normal
values. Dot product returns similarity score.

### Two-Tower Model

```c
void rec_user_tower_forward(const RecUserTower *tower, const RecFeatureVec *feat,
                            RecEmbedding *out);
void rec_item_tower_forward(const RecItemTower *tower, const RecFeatureVec *feat,
                            RecEmbedding *out);
```

Forward pass through user/item tower producing embeddings.

### Recall

```c
void rec_collab_filter(const RecUserTower *user, const RecItemTower *items,
                       int32_t num_items, int32_t k, RecCandidateList *result);
void rec_ann_search(const RecEmbedding *query, const RecItemTower *items,
                    int32_t num_items, int32_t k, RecCandidateList *result);
void rec_popular_recall(const int32_t *popular_ids, int32_t n, int32_t k,
                        RecCandidateList *result);
void rec_random_recall(const RecItemTower *items, int32_t n, int32_t k,
                       uint64_t seed, RecCandidateList *result);
void rec_multi_channel_recall(...);
```

- `rec_collab_filter`: Top-k by dot product + item bias
- `rec_ann_search`: Approximate nearest neighbor (simple top-k scan)
- `rec_popular_recall`: Pre-computed popularity list
- `rec_random_recall`: Random shuffle for exploration
- `rec_multi_channel_recall`: Weighted blend of multiple channels

### Ranking

```c
void rec_ranking_dnn_init(RecRankingDNN *dnn, const int32_t *dims,
                          int32_t num_layers, uint64_t seed);
float rec_ranking_dnn_forward(const RecRankingDNN *dnn, const RecFeatureVec *feat);
void rec_rank_candidates(const RecRankingDNN *dnn, ...);
```

DNN with configurable hidden layer dimensions. ReLU activations. Sigmoid output.

### Re-Ranking

```c
void rec_diversity_rerank(RecCandidateList *ranked, const RecItemTower *items,
                          float lambda, int32_t k);
void rec_freshness_boost(RecCandidateList *ranked, const int64_t *timestamps,
                         int64_t now, float alpha);
void rec_business_rule_filter(RecCandidateList *ranked,
                              const int32_t *blocked_ids, int32_t n_blocked);
```

- Diversity: MMR-style dedup by embedding similarity
- Freshness: Exponential time-decay boost
- Business rules: Remove blocked IDs

### Cold Start

```c
void rec_content_features(const char *title, const char *category,
                          const char **tags, int32_t n_tags, RecFeatureVec *out);
void rec_cold_start_score(const RecFeatureVec *new_item,
                          const RecFeatureVec *user_profile, float *score);
```

Content-based feature extraction and scoring for new items without interaction
history.

### A/B Testing

```c
void rec_ab_config_init(RecABConfig *cfg, int32_t control_pct,
                        int32_t treatment_pct, uint64_t seed);
int  rec_ab_decide(const RecABConfig *cfg, int32_t user_id);
void rec_ab_evaluate(double *ctrl_metrics, int32_t n_ctrl,
                     double *treat_metrics, int32_t n_treat, RecABResult *result);
```

Hash-based traffic splitting and Welch's t-test for metric comparison.

---

## ChatBot RLHF (`chatbot_rlhf.h`)

### Base Model

```c
void cb_base_model_init(CBBaseModel *m, uint64_t seed);
void cb_base_model_forward(const CBBaseModel *m, const int32_t *input_ids,
                           int32_t seq_len, float *hidden_states);
```

Transformer-style model with token + position embeddings, attention, FFN.

### SFT (Supervised Fine-Tuning)

```c
void cb_sft_train(CBBaseModel *model, const CBInput *demos,
                  const CBOutput *labels, int32_t num_demos,
                  float lr, int32_t epochs);
```

Train on (prompt, response) demonstration pairs via cross-entropy loss.

### Reward Model

```c
void  cb_reward_model_init(CBRewardModel *rm, uint64_t seed);
float cb_reward_model_forward(const CBRewardModel *rm, ...);
void  cb_reward_model_train(CBRewardModel *rm, const CBPreferencePair *pairs,
                            int32_t num_pairs, float lr, int32_t epochs);
```

Bradley-Terry preference modeling. Scalar reward from last hidden state.

### RLHF: PPO

```c
void  cb_ppo_config_init(CBPPOConfig *cfg);
void  cb_ppo_advantage(const float *rewards, const float *values,
                       int32_t len, const CBPPOConfig *cfg,
                       float *advantages, float *returns);
void  cb_ppo_policy_loss(const CBBaseModel *policy, const CBBaseModel *ref, ...);
float cb_kl_divergence(const float *logprobs_policy, const float *logprobs_ref,
                       int32_t len);
void  cb_ppo_update(CBBaseModel *policy, const CBBaseModel *ref,
                    const CBRewardModel *rm, const CBInput *prompts,
                    int32_t num_prompts, const CBPPOConfig *cfg);
```

Proximal Policy Optimization with:
- GAE (Generalized Advantage Estimation)
- Clipped surrogate objective
- KL divergence penalty from reference model

### DPO (Direct Preference Optimization)

```c
void  cb_dpo_train(CBBaseModel *model, const CBPreferencePair *pairs,
                   int32_t num_pairs, const CBDPOConfig *cfg);
float cb_dpo_loss(const CBBaseModel *model, const CBPreferencePair *pair, float beta);
float cb_implicit_reward(const CBBaseModel *model, const CBBaseModel *ref,
                         const int32_t *resp_ids, int32_t resp_len, float beta);
```

Direct optimization from preferences without a separate reward model.

### Safety

```c
void  cb_ai_feedback_generate(const CBBaseModel *model, const CBInput *prompt,
                              CBAIFeedback *feedback);
void  cb_constitutional_alignment(CBBaseModel *model, const CBInput *prompts,
                                  int32_t num_prompts, CBAIFeedbackType type);
void  cb_safety_filter(const int32_t *resp_ids, int32_t resp_len,
                       const CBRedTeamSet *red_team, int *is_safe);
float cb_harmlessness_score(const CBOutput *response, const CBRedTeamSet *red_team);
void  cb_safety_alignment_loss(const CBBaseModel *model, const CBInput *prompts,
                               int32_t num_prompts, const CBRedTeamSet *red_team,
                               float *loss);
```

- AI Feedback: Generate critique for constitutional alignment
- Safety Filter: Red-team phrase matching
- Harmlessness scoring and alignment loss

### Tokenization

```c
void cb_tokenize(const char *text, int32_t *token_ids, int32_t *len, int32_t max_len);
void cb_detokenize(const int32_t *token_ids, int32_t len, char *text, int32_t max_text);
void cb_generate(const CBBaseModel *model, const int32_t *input_ids,
                 int32_t input_len, int32_t max_new_tokens, float temperature,
                 uint64_t seed, CBOutput *output);
```

Simple byte-level tokenization (char % VOCAB_SIZE). Temperature-controlled
sampling autoregressive generation.

---

## Copilot Context (`copilot_context.h`)

### Context Gathering

```c
void cc_gather_open_files(CCContextState *state, const char **paths, int32_t n);
void cc_set_cursor(CCContextState *state, const char *path, int32_t line, int32_t col);
void cc_gather_recent_edits(CCContextState *state, ...);
void cc_gather_imports(CCContextState *state, ...);
void cc_gather_project_structure(CCContextState *state, ...);
void cc_gather_git_diff(CCContextState *state, const char *diff_text, int32_t len);
void cc_detect_language(const char *path, char *language, int32_t max_len);
```

Collects editor state: open files, cursor position, recent edits, imports,
project file tree, git diff. Language detection by file extension.

### Context Ranking

```c
float cc_snippet_relevance(const CCContextSnippet *snip, const CCCursorPos *cursor,
                           const char *active_path);
void  cc_rank_context(const CCContextState *state, CCRankedContext *ranked);
```

Sorts snippets by proximity to cursor and recency.

### Prompt Construction

```c
void cc_construct_system_message(CCPromptMessage *msg);
void cc_construct_prompt(const CCContextState *state, const CCUserRequest *request,
                         const CCRankedContext *ranked, CCConstructedPrompt *prompt);
```

Builds chat-formatted prompt: system message + context + user request.

### Completion

```c
void cc_completion_generate(const CCConstructedPrompt *prompt, const char *prefix,
                            float temperature, uint64_t seed, CCCompletion *completion);
void cc_completion_multi_line(const CCConstructedPrompt *prompt, ...);
void cc_postprocess_trim(CCCompletion *completion);
void cc_postprocess_filter_incomplete(const char *text, int32_t len, int *is_complete);
void cc_inline_chat_process(const CCUserRequest *request, const CCContextState *state,
                            CCConstructedPrompt *prompt);
```

Code suggestion generation with post-processing (whitespace trim, bracket
balance check).

---

## Evaluation & Monitoring (`eval_monitor.h`)

### Offline Metrics

```c
void ev_classification_eval(const int32_t *y_true, const int32_t *y_pred,
                            int32_t n, int32_t n_classes, EVClassificationMetrics *m);
void ev_nlg_eval(const char **references, const char **predictions,
                 int32_t n, EVNLGMetrics *metrics);
```

- Classification: accuracy, macro precision/recall, F1, confusion matrix
- NLG: BLEU 1-4 gram, ROUGE-1, ROUGE-2, ROUGE-L

### Online Metrics

```c
void ev_online_metric_init(EVOnlineMetric *m, const char *variant);
void ev_online_collect(EVOnlineMetric *m, ...);
void ev_ab_experiment_compare(const EVOnlineMetric *ctrl,
                              const EVOnlineMetric *treat, EVABExperiment *exp);
```

Running averages for engagement, CTR, task completion, session time, retention.

### LLM-as-Judge

```c
void ev_llm_judge_evaluate(EVLLMJudgeResult *judge, const char *response,
                           const char *reference, EVJudgeDimension dim);
```

Rate responses across: helpfulness, correctness, safety, coherence.

### Monitoring

```c
void ev_data_drift_detect(const double *ref, int32_t ref_len,
                          const double *cur, int32_t cur_len,
                          int32_t num_bins, EVDriftReport *report);
double ev_psi_compute(const double *ref_dist, const double *cur_dist, int32_t n);
double ev_ks_test(const double *ref, int32_t ref_len,
                  const double *cur, int32_t cur_len);
void ev_prediction_drift_monitor(..., EVDriftReport *report);
```

PSI (Population Stability Index) and KS-test for distribution drift detection.

### Dashboard & Feedback

```c
void ev_dashboard_init(EVDashboard *dash);
void ev_dashboard_update(EVDashboard *dash, const char *name, double cur, double base);
void ev_dashboard_render(const EVDashboard *dash, char *output, int32_t max_len);
void ev_feedback_collect(EVFeedbackCollection *fb, ...);
void ev_feedback_summarize(const EVFeedbackCollection *fb, ...);
```

Text-based dashboard rendering and thumbs-up/down feedback collection.

---

## Model A/B Testing (`model_ab_test.h`)

### Experiment Management

```c
void mab_experiment_init(MABExperiment *exp, const char *name, uint64_t seed);
void mab_experiment_add_variant(MABExperiment *exp, const char *name,
                                MABVariantType type, int32_t pct, void *model);
int  mab_experiment_validate(const MABExperiment *exp);
void mab_experiment_activate(MABExperiment *exp);
void mab_experiment_deactivate(MABExperiment *exp);
```

Create and manage experiments with multiple variants.

### Traffic Assignment

```c
uint64_t mab_user_hash(int32_t user_id, uint64_t seed);
int      mab_traffic_assign(int32_t user_id, const MABExperiment *exp);
```

Deterministic hash-based user-to-variant assignment.

### Metrics & Statistics

```c
void mab_metrics_init(MABVariantMetrics *m);
void mab_metrics_record_request(MABVariantMetrics *m, double latency,
                                int success, double satisfaction);
void mab_metrics_compare(const MABVariantMetrics *ctrl,
                         const MABVariantMetrics *treat, ...);
void mab_ttest_independent(const double *a, int32_t na,
                           const double *b, int32_t nb, MABTTestResult *result);
void mab_chi_square_test(double ctrl_rate, int32_t ctrl_n,
                         double treat_rate, int32_t treat_n, MABChiSquareResult *r);
int  mab_sample_size_estimate(double baseline, double mde, double alpha, double power);
```

Welch's t-test, chi-square for proportions, sample size estimation.

### Ramp-Up & Rollback

```c
void mab_ramp_up_init(MABRampUpPlan *plan, int32_t initial, int64_t step_dur);
void mab_ramp_up_execute(MABExperiment *exp, const MABRampUpPlan *plan,
                         const MABGuardrails *guardrails, MABRollbackDecision *dec);
void mab_rollback_evaluate(const MABVariantMetrics *ctrl,
                           const MABVariantMetrics *treat, ...);
void mab_rollback_execute(MABExperiment *exp);
```

Progressive traffic ramp-up (1%->5%->25%->50%->100%) with guardrail checks and
automatic rollback triggers.

### Model Registry

```c
void mab_model_registry_init(MABModelRegistry *reg, const char *name,
                             const char *version, MABModelStage stage, void *model);
void mab_model_registry_promote(MABModelRegistry *reg, MABModelStage next);
void mab_model_registry_archive(MABModelRegistry *reg);
int  mab_model_registry_can_serve(const MABModelRegistry *reg);
```

Stage transitions: INIT -> DEV -> STAGING -> CANARY -> PRODUCTION -> ARCHIVED.

### Shadow Testing

```c
void mab_shadow_test_init(MABShadowTest *st);
void mab_shadow_test_record(MABShadowTest *st, double prod, double shadow,
                            double latency, int success);
void mab_shadow_test_evaluate(MABShadowTest *st, double threshold);
```

Dark launch validation: compare shadow model outputs against production without
affecting users.

### Feature Flags

```c
void mab_feature_flag_set(const char *flag_name, int enabled);
int  mab_feature_flag_get(const char *flag_name);
int  mab_feature_flag_user(const char *flag_name, int32_t user_id, int32_t pct);
```

Global flag store with user-level rollout percentage.

### Reporting

```c
void mab_report_generate(const MABExperiment *exp,
                         const MABExperimentMetrics *metrics,
                         char *report, int32_t max_len);
```

Generate human-readable experiment report with per-variant metrics.
