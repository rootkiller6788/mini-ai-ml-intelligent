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

static int test1(void) {
    TEST("rec_embedding_init");
    RecEmbedding e;
    rec_embedding_init(&e, 42);
    CHECK(rec_dot_product(&e, &e) > 0.0f, "dot");
    PASS(); return 0;
}
static int test2(void) {
    TEST("rec_ann_search");
    RecEmbedding q; rec_embedding_init(&q, 1);
    RecItemTower items[10];
    for (int i = 0; i < 10; i++) {
        rec_embedding_init(&items[i].emb, (uint64_t)(i + 10));
        items[i].item_id = i;
    }
    RecCandidateList r;
    rec_ann_search(&q, items, 10, 5, &r);
    CHECK(r.count > 0, "ann");
    PASS(); return 0;
}
static int test3(void) {
    TEST("rec_ab");
    double c[]={0.1,0.12,0.11}, t[]={0.15,0.14,0.16};
    RecABResult r; rec_ab_evaluate(c,3,t,3,&r);
    CHECK(r.ctr_control>0,"ab");
    PASS(); return 0;
}
static int test4(void) {
    TEST("cb_init");
    static CBBaseModel m;
    cb_base_model_init(&m, 42);
    int32_t ids[]={1,2,3};
    float h[3*CB_HIDDEN_DIM];
    cb_base_model_forward(&m, ids, 3, h);
    PASS(); return 0;
}
static int test5(void) {
    TEST("cb_tokenize");
    int32_t tks[32]; int32_t l;
    cb_tokenize("hello", tks, &l, 32);
    CHECK(l > 0, "tok");
    PASS(); return 0;
}
static int test6(void) {
    TEST("cb_reward");
    static CBRewardModel rm;
    cb_reward_model_init(&rm, 42);
    int32_t p[]={1,2}, r[]={3,4,5};
    float s = cb_reward_model_forward(&rm, p, 2, r, 3);
    CHECK(isfinite(s), "rm");
    PASS(); return 0;
}
static int test7(void) {
    TEST("cc_init");
    CCContextState st;
    cc_context_state_init(&st);
    CHECK(st.num_open_files == 0, "cc");
    PASS(); return 0;
}
static int test8(void) {
    TEST("cc_lang");
    char l[32];
    cc_detect_language("test.c", l, 32);
    CHECK(strlen(l) > 0, "lang");
    cc_detect_language("test.py", l, 32);
    CHECK(strlen(l) > 0, "lang2");
    PASS(); return 0;
}
static int test9(void) {
    TEST("ev_class");
    int32_t yt[]={0,0,1,1}, yp[]={0,1,0,1};
    EVClassificationMetrics m;
    ev_classification_eval(yt, yp, 4, 2, &m);
    CHECK(m.accuracy > 0.0 && m.accuracy < 1.0, "cls");
    PASS(); return 0;
}
static int test10(void) {
    TEST("ev_f1");
    double f = ev_f1_score(0.8, 0.6);
    CHECK(f > 0.0 && f <= 1.0, "f1");
    PASS(); return 0;
}
int main(void) {
    printf("=== test_step3 (10 tests) ===\n");
    int f = 0;
    f += test1(); f += test2(); f += test3(); f += test4(); f += test5();
    f += test6(); f += test7(); f += test8(); f += test9(); f += test10();
    printf("\n=== %d/%d passed, %d failed ===\n", tests_passed, tests_run, f);
    return f > 0 ? 1 : 0;
}
