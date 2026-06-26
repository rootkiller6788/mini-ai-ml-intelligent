/*
 * mini-ai-product-system — Full Demo: AI Product Systems
 *
 * Demonstrates: recommendation, chatbot RLHF, copilot context,
 *               evaluation/monitoring, A/B testing.
 */
#include "../include/recommendation.h"
#include "../include/chatbot_rlhf.h"
#include "../include/copilot_context.h"
#include "../include/eval_monitor.h"
#include "../include/model_ab_test.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(void) {
    printf("=== mini-ai-product-system: AI Product Systems Demo ===\n\n");

    /* Step 1: Recommendation System */
    printf("-- Step 1: Recommendation System --\n");
    RecEmbedding user_emb;
    rec_embedding_init(&user_emb, 42);
    RecItemTower items[4];
    for (int i = 0; i < 4; i++) {
        rec_embedding_init(&items[i].emb, (uint64_t)(i * 100));
        items[i].item_id = i;
    }
    RecCandidateList candidates;
    rec_ann_search(&user_emb, items, 4, 2, &candidates);
    printf("ANN search: query -> %d candidates\n", candidates.count);
    RecABConfig ab_cfg;
    rec_ab_config_init(&ab_cfg, 50, 50, 12345);
    int bucket = rec_ab_decide(&ab_cfg, 100);
    printf("A/B test: user 100 -> bucket %d (0=control, 1=treatment)\n", bucket);

    /* Step 2: Chatbot RLHF */
    printf("\n-- Step 2: Chatbot RLHF --\n");
    CBBaseModel model;
    cb_base_model_init(&model, 42);
    int32_t tokens[32], len;
    cb_tokenize("Hello, how are you?", tokens, &len, 32);
    printf("Tokenize: '%s' -> %d tokens\n", "Hello, how are you?", len);
    CBOutput output;
    cb_generate(&model, tokens, len, 32, 0.7f, 42, &output);
    printf("Generate: %d output tokens\n", output.length);

    CBRewardModel rm;
    cb_reward_model_init(&rm, 42);
    float reward = cb_reward_model_forward(&rm, tokens, len, output.token_ids, output.length);
    printf("Reward model score: %.4f\n", reward);

    CBRewardModel rm2;
    cb_reward_model_init(&rm2, 99);
    CBPreferencePair pair;
    memset(&pair, 0, sizeof(pair));
    pair.prompt.input_len = len;
    memcpy(pair.prompt.input_ids, tokens, len * sizeof(int32_t));
    memcpy(pair.chosen.input_ids, output.token_ids, output.length * sizeof(int32_t));
    pair.chosen.length = output.length;
    pair.chosen.total_logprob = -1.0f;
    cb_reward_model_train(&rm2, &pair, 1, 0.001f, 1);
    printf("RLHF: 1 preference pair trained\n");

    /* Step 3: Copilot Context */
    printf("\n-- Step 3: Copilot Context --\n");
    CCContextState state;
    cc_context_state_init(&state);
    const char *files[] = {"main.c", "utils.h"};
    cc_gather_open_files(&state, files, 2);
    char lang[32];
    cc_detect_language("main.c", lang, 32);
    printf("Context: %d files open, language=%s\n", state.num_open_files, lang);

    CCUserRequest req;
    memset(&req, 0, sizeof(req));
    req.intent = CC_INTENT_COMPLETE;
    strcpy(req.text, "Write a function to ");
    req.text_len = strlen(req.text);
    CCRankedContext ranked;
    cc_rank_context(&state, &ranked);
    printf("Ranked context: %d snippets\n", ranked.num_snippets);

    /* Step 4: Evaluation & Monitoring */
    printf("\n-- Step 4: Evaluation & Monitoring --\n");
    int32_t y_true[] = {0, 1, 2, 0, 1, 2};
    int32_t y_pred[] = {0, 1, 2, 0, 2, 2};
    EVClassificationMetrics metrics;
    ev_classification_eval(y_true, y_pred, 6, 3, &metrics);
    printf("Classification: acc=%.4f, f1=%.4f\n", metrics.accuracy, metrics.f1_score);
    double f1 = ev_f1_score(metrics.precision, metrics.recall);
    printf("  F1 (formula): %.4f\n", f1);

    EVOnlineMetric online;
    ev_online_metric_init(&online, "model_v1");
    ev_online_collect(&online, 0.8, 0.12, 0.65, 120.0, 1, 0);
    printf("Online metric (v1): engagement=%.2f, CTR=%.2f%%\n",
           online.engagement_rate, online.click_through_rate * 100.0);

    /* Step 5: A/B Testing */
    printf("\n-- Step 5: Model A/B Testing --\n");
    MABExperiment exp;
    mab_experiment_init(&exp, "new_model_test", MAB_HASH_SEED);
    mab_experiment_add_variant(&exp, "control", MAB_VARIANT_CONTROL, 90, NULL);
    mab_experiment_add_variant(&exp, "treatment", MAB_VARIANT_TREATMENT, 10, NULL);
    mab_experiment_validate(&exp);
    int assigned = mab_traffic_assign(999, &exp);
    printf("A/B experiment: '%s', user=999 -> variant=%d\n", exp.name, assigned);

    MABGuardrails g;
    mab_guardrails_init(&g);
    mab_guardrails_add(&g, "latency_p99_ms", 500.0f, 1);
    mab_guardrails_add(&g, "error_rate", 0.01f, 1);
    printf("Guardrails: %d configured\n", g.num_guardrails);

    MABModelRegistry reg;
    mab_model_registry_init(&reg, "my_model", "v1.2.0", MAB_STAGE_STAGING, NULL);
    printf("Model registry: %s@%s, stage=%d\n", reg.name, reg.version, reg.stage);
    mab_model_registry_promote(&reg, MAB_STAGE_CANARY);
    printf("  promoted to stage=%d\n", reg.stage);

    printf("\nAI product systems demo complete!\n");
    return 0;
}
