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
    PASS();
    return 0;
}

int main(void) {
    printf("=== test_step ===\n");
    int failed = 0;
    failed += test_rec_embedding_init();
    printf("\n=== Results: %d/%d passed, %d failed ===\n", tests_passed, tests_run, failed);
    return failed > 0 ? 1 : 0;
}
