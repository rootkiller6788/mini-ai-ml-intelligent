/*
 * mini-ai-product-system — Core Benchmarks
 *
 * Benchmarks: recommendation, chatbot RLHF, copilot,
 *             evaluation, A/B testing.
 */
#include "../include/recommendation.h"
#include "../include/chatbot_rlhf.h"
#include "../include/copilot_context.h"
#include "../include/eval_monitor.h"
#include "../include/model_ab_test.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
    return (double)clock() / (double)(CLOCKS_PER_SEC / 1000);
}

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 5000;
    double t0, t1;
    printf("=== mini-ai-product-system Benchmarks (N=%d) ===\n\n", N);

    /* ── Rec Embedding Init ── */
    {
        t0 = now_ms();
        for (int r = 0; r < N / 10; r++) {
            RecEmbedding e;
            rec_embedding_init(&e, (uint64_t)r);
        }
        t1 = now_ms();
        printf("  rec_embedding_init:  %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 10, t1 - t0, (t1 - t0) / (double)(N / 10) * 1000.0);
    }

    /* ── Rec Dot Product ── */
    {
        RecEmbedding a, b;
        rec_embedding_init(&a, 1);
        rec_embedding_init(&b, 2);
        t0 = now_ms();
        for (int r = 0; r < N / 10; r++) {
            rec_dot_product(&a, &b);
        }
        t1 = now_ms();
        printf("  rec_dot_product:     %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 10, t1 - t0, (t1 - t0) / (double)(N / 10) * 1000.0);
    }

    /* ── Chatbot Tokenize ── */
    {
        int32_t tokens[32]; int32_t len;
        t0 = now_ms();
        for (int r = 0; r < N / 10; r++) {
            cb_tokenize("hello world test", tokens, &len, 32);
        }
        t1 = now_ms();
        printf("  cb_tokenize:         %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 10, t1 - t0, (t1 - t0) / (double)(N / 10) * 1000.0);
    }

    /* ── CB Reward Model Forward ── */
    {
        CBRewardModel rm;
        cb_reward_model_init(&rm, 42);
        int32_t p[] = {1, 2}, r[] = {3, 4, 5};
        t0 = now_ms();
        for (int i = 0; i < N / 20; i++) {
            cb_reward_model_forward(&rm, p, 2, r, 3);
        }
        t1 = now_ms();
        printf("  cb_reward_fwd:       %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 20, t1 - t0, (t1 - t0) / (double)(N / 20) * 1000.0);
    }

    /* ── CC Detect Language ── */
    {
        char lang[32];
        t0 = now_ms();
        for (int r = 0; r < N / 10; r++) {
            cc_detect_language("test.rs", lang, 32);
        }
        t1 = now_ms();
        printf("  cc_detect_language:  %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 10, t1 - t0, (t1 - t0) / (double)(N / 10) * 1000.0);
    }

    /* ── Eval Classification ── */
    {
        int32_t yt[100], yp[100];
        for (int i = 0; i < 100; i++) { yt[i] = i % 3; yp[i] = i % 3; }
        EVClassificationMetrics m;
        t0 = now_ms();
        for (int r = 0; r < N / 10; r++) {
            ev_classification_eval(yt, yp, 100, 3, &m);
        }
        t1 = now_ms();
        printf("  ev_classification:   %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 10, t1 - t0, (t1 - t0) / (double)(N / 10) * 1000.0);
    }

    /* ── MAB User Hash ── */
    {
        t0 = now_ms();
        for (int r = 0; r < N / 10; r++) {
            mab_user_hash(r, MAB_HASH_SEED);
        }
        t1 = now_ms();
        printf("  mab_user_hash:       %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 10, t1 - t0, (t1 - t0) / (double)(N / 10) * 1000.0);
    }

    /* ── MAB Experiment ── */
    {
        t0 = now_ms();
        for (int r = 0; r < N / 20; r++) {
            MABExperiment exp;
            mab_experiment_init(&exp, "test", 42);
            mab_experiment_add_variant(&exp, "control", MAB_VARIANT_CONTROL, 50, NULL);
            mab_experiment_add_variant(&exp, "treatment", MAB_VARIANT_TREATMENT, 50, NULL);
        }
        t1 = now_ms();
        printf("  mab_experiment_init: %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 20, t1 - t0, (t1 - t0) / (double)(N / 20) * 1000.0);
    }

    printf("\nDone.\n");
    return 0;
}
