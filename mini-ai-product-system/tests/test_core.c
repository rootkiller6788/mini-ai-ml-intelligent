/*
 * mini-ai-product-system — Core Tests
 *
 * Unit tests for recommendation, chatbot RLHF, copilot,
 * evaluation/monitoring, A/B testing.
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

static int tests_run = 0, tests_passed = 0;

#define TEST(name) do { tests_run++; printf("  TEST %s ... ", name); } while(0)
#define PASS() do { tests_passed++; printf("PASS\n"); } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); return 1; } while(0)
#define CHECK(cond, msg) if (!(cond)) FAIL(msg)

/* ── Recommendation Tests ── */
static int test_rec_embedding_init(void) {
    TEST("rec_embedding_init");
    RecEmbedding e;
    rec_embedding_init(&e, 42);
    float dot = rec_dot_product(&e, &e);
    CHECK(dot > 0.0f, "self dot should be positive");
    PASS();
    return 0;
}

static int test_rec_ann_search(void) {
    TEST("rec_ann_search");
    RecEmbedding query, item;
    rec_embedding_init(&query, 1);
    rec_embedding_init(&item, 2);
    RecItemTower items[10];
    for (int i = 0; i < 10; i++) {
        rec_embedding_init(&items[i].emb, (uint64_t)(i + 10));
        items[i].item_id = i;
    }
    RecCandidateList result;
    rec_ann_search(&query, items, 10, 5, &result);
    CHECK(result.count > 0, "ANN search returned no results");
    PASS();
    return 0;
}

static int test_rec_ab_evaluate(void) {
    TEST("rec_ab_evaluate");
    double ctrl[] = {0.1, 0.12, 0.11, 0.13, 0.10};
    double treat[] = {0.15, 0.14, 0.16, 0.15, 0.14};
    RecABResult result;
    rec_ab_evaluate(ctrl, 5, treat, 5, &result);
    CHECK(result.ctr_control > 0.0, "control CTR should be > 0");
    CHECK(result.ctr_treatment > 0.0, "treatment CTR should be > 0");
    PASS();
    return 0;
}

/* ── Chatbot RLHF Tests ── */
static int test_cb_base_model_init(void) {
    TEST("cb_base_model_init");
    static CBBaseModel model;
    cb_base_model_init(&model, 42);
    int32_t ids[] = {1, 2, 3};
    float hidden[3 * CB_HIDDEN_DIM];
    cb_base_model_forward(&model, ids, 3, hidden);
    PASS();
    return 0;
}

static int test_cb_tokenize(void) {
    TEST("cb_tokenize_detokenize");
    int32_t tokens[32]; int32_t len;
    cb_tokenize("hello world", tokens, &len, 32);
    CHECK(len > 0, "tokenize returned 0 tokens");
    char text[256]; int32_t text_len;
    cb_detokenize(tokens, len, text, 256);
    CHECK(strlen(text) > 0, "detokenize empty");
    PASS();
    return 0;
}

static int test_cb_reward_model_init(void) {
    TEST("cb_reward_model_init");
    static CBRewardModel rm;
    cb_reward_model_init(&rm, 42);
    int32_t prompt_ids[] = {1, 2};
    int32_t resp_ids[] = {3, 4, 5};
    float score = cb_reward_model_forward(&rm, prompt_ids, 2, resp_ids, 3);
    CHECK(isfinite(score), "reward score not finite");
    PASS();
    return 0;
}

/* ── Copilot Tests ── */
static int test_cc_context_state_init(void) {
    TEST("cc_context_state_init");
    CCContextState state;
    cc_context_state_init(&state);
    CHECK(state.num_open_files == 0, "should start with no files");
    PASS();
    return 0;
}

static int test_cc_detect_language(void) {
    TEST("cc_detect_language");
    char lang[32];
    cc_detect_language("test.c", lang, 32);
    CHECK(strlen(lang) > 0, "language should be detected");
    cc_detect_language("test.py", lang, 32);
    CHECK(strlen(lang) > 0, "language should be detected");
    PASS();
    return 0;
}

/* ── Eval/Monitor Tests ── */
static int test_ev_classification_eval(void) {
    TEST("ev_classification_eval");
    int32_t y_true[] = {0, 0, 1, 1};
    int32_t y_pred[] = {0, 1, 0, 1};
    EVClassificationMetrics metrics;
    ev_classification_eval(y_true, y_pred, 4, 2, &metrics);
    CHECK(metrics.accuracy > 0.0 && metrics.accuracy < 1.0, "accuracy out of range");
    PASS();
    return 0;
}

static int test_ev_f1_score(void) {
    TEST("ev_f1_score");
    double f1 = ev_f1_score(0.8, 0.6);
    CHECK(f1 > 0.0 && f1 <= 1.0, "f1 out of range");
    PASS();
    return 0;
}

/* ── A/B Testing Tests ── */
static int test_mab_experiment_init(void) {
    TEST("mab_experiment_init");
    MABExperiment exp;
    mab_experiment_init(&exp, "test_exp", 42);
    CHECK(strcmp(exp.name, "test_exp") == 0, "name wrong");
    PASS();
    return 0;
}

static int test_mab_user_hash(void) {
    TEST("mab_user_hash");
    uint64_t h1 = mab_user_hash(12345, MAB_HASH_SEED);
    uint64_t h2 = mab_user_hash(12345, MAB_HASH_SEED);
    CHECK(h1 == h2, "hash should be deterministic");
    PASS();
    return 0;
}

static int test_mab_guardrails_init(void) {
    TEST("mab_guardrails_init");
    MABGuardrails g;
    mab_guardrails_init(&g);
    CHECK(g.num_guardrails == 0, "should start empty");
    mab_guardrails_add(&g, "latency_p99", 100.0f, 1);
    CHECK(g.num_guardrails == 1, "guardrail count wrong");
    PASS();
    return 0;
}

int main(void) {
    printf("=== mini-ai-product-system Unit Tests ===\n\n");

    int failed = 0;
    failed += test_rec_embedding_init();
    failed += test_rec_ann_search();
    failed += test_rec_ab_evaluate();
    failed += test_cb_base_model_init();
    failed += test_cb_tokenize();
    failed += test_cb_reward_model_init();
    failed += test_cc_context_state_init();
    failed += test_cc_detect_language();
    failed += test_ev_classification_eval();
    failed += test_ev_f1_score();
    failed += test_mab_experiment_init();
    failed += test_mab_user_hash();
    failed += test_mab_guardrails_init();

    printf("\n=== Results: %d/%d passed, %d failed ===\n",
           tests_passed, tests_run, failed);
    return failed > 0 ? 1 : 0;
}
