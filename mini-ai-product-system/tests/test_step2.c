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

static int test_rec_embedding_init(void) {
    TEST("rec_embedding_init");
    RecEmbedding e;
    rec_embedding_init(&e, 42);
    float dot = rec_dot_product(&e, &e);
    CHECK(dot > 0.0f, "self dot should be positive");
    PASS(); return 0;
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
    PASS(); return 0;
}
static int test_rec_ab_evaluate(void) {
    TEST("rec_ab_evaluate");
    double ctrl[] = {0.1, 0.12, 0.11, 0.13, 0.10};
    double treat[] = {0.15, 0.14, 0.16, 0.15, 0.14};
    RecABResult result;
    rec_ab_evaluate(ctrl, 5, treat, 5, &result);
    CHECK(result.ctr_control > 0.0, "control CTR should be > 0");
    PASS(); return 0;
}
static int test_cb_base_model_init(void) {
    TEST("cb_base_model_init");
    static CBBaseModel model;
    cb_base_model_init(&model, 42);
    int32_t ids[] = {1, 2, 3};
    float hidden[3 * CB_HIDDEN_DIM];
    cb_base_model_forward(&model, ids, 3, hidden);
    PASS(); return 0;
}
static int test_cb_tokenize(void) {
    TEST("cb_tokenize_detokenize");
    int32_t tokens[32]; int32_t len;
    cb_tokenize("hello world", tokens, &len, 32);
    CHECK(len > 0, "tokenize returned 0 tokens");
    char text[256];
    cb_detokenize(tokens, len, text, 256);
    CHECK(strlen(text) > 0, "detokenize empty");
    PASS(); return 0;
}
int main(void) {
    printf("=== test_step2 ===\n");
    int failed = 0;
    failed += test_rec_embedding_init();
    failed += test_rec_ann_search();
    failed += test_rec_ab_evaluate();
    failed += test_cb_base_model_init();
    failed += test_cb_tokenize();
    printf("\n=== %d/%d passed, %d failed ===\n", tests_passed, tests_run, failed);
    return failed > 0 ? 1 : 0;
}
