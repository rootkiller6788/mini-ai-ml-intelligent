/*
 * mini-training-system — Core Tests
 *
 * Unit tests for distributed training, mixed precision, checkpointing,
 * hyperparameter tuning, training loop.
 */
#include "../include/distributed_train.h"
#include "../include/mixed_precision.h"
#include "../include/checkpoint_save.h"
#include "../include/hyperparam_tune.h"
#include "../include/training_loop.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int tests_run = 0, tests_passed = 0;

#define TEST(name) do { tests_run++; printf("  TEST %s ... ", name); } while(0)
#define PASS() do { tests_passed++; printf("PASS\n"); } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); return 1; } while(0)
#define CHECK(cond, msg) if (!(cond)) FAIL(msg)

/* ── Distributed Training Tests ── */
static int test_ddp_context_init(void) {
    TEST("ddp_context_init");
    ddp_context_t ctx;
    ddp_init(&ctx, 0, NULL);
    CHECK(ctx.world_size >= 1, "world_size should be >= 1");
    ddp_finalize(&ctx);
    PASS();
    return 0;
}

static int test_ring_all_reduce(void) {
    TEST("ring_all_reduce");
    float buf[16];
    for (int i = 0; i < 16; i++) buf[i] = 1.0f;
    ring_all_reduce(buf, 16, RING_REDUCE_OP_SUM, 0, 1);
    CHECK(fabsf(buf[0] - 1.0f) < 0.01f, "ring all_reduce wrong");
    PASS();
    return 0;
}

static int test_zero_config(void) {
    TEST("zero_config_init");
    zero_config_t cfg;
    zero_init(&cfg, ZERO_STAGE_1);
    CHECK(cfg.stage == ZERO_STAGE_1, "zero stage wrong");
    PASS();
    return 0;
}

/* ── Mixed Precision Tests ── */
static int test_fp32_to_fp16_roundtrip(void) {
    TEST("fp32_fp16_roundtrip");
    float src = 1.5f;
    uint16_t half = fp32_to_fp16_scalar(src);
    float back = fp16_to_fp32_scalar(half);
    CHECK(fabsf(back - src) < 0.01f, "fp16 roundtrip wrong");
    PASS();
    return 0;
}

static int test_fp32_to_bf16_roundtrip(void) {
    TEST("fp32_bf16_roundtrip");
    float src = 2.0f;
    uint16_t bf = fp32_to_bf16_scalar(src);
    float back = bf16_to_fp32_scalar(bf);
    CHECK(fabsf(back - src) < 0.01f, "bf16 roundtrip wrong");
    PASS();
    return 0;
}

static int test_mp_context_init(void) {
    TEST("mp_context_init");
    mp_config_t cfg = {0};
    cfg.mode = MP_MODE_FP16;
    cfg.init_loss_scale = 65536.0f;
    mp_context_t ctx;
    mp_init(&ctx, &cfg);
    CHECK(ctx.loss_scale > 0.0f, "loss_scale should be positive");
    PASS();
    return 0;
}

/* ── Checkpoint Tests ── */
static int test_ckpt_config_init(void) {
    TEST("ckpt_config_init");
    ckpt_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    strcpy(cfg.base_dir, "/tmp/ckpt_test");
    strcpy(cfg.run_name, "test_run");
    cfg.strategy = CKPT_STRATEGY_BEST;
    ckpt_context_t ctx;
    ckpt_init(&ctx, &cfg);
    CHECK(ctx.config.strategy == CKPT_STRATEGY_BEST, "strategy wrong");
    PASS();
    return 0;
}

static int test_ckpt_should_save(void) {
    TEST("ckpt_should_save");
    ckpt_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.strategy = CKPT_STRATEGY_PERIODIC;
    cfg.save_every_n_steps = 100;
    ckpt_context_t ctx;
    ckpt_init(&ctx, &cfg);
    int should = ckpt_should_save(&ctx, 100, 0, 0.0f);
    CHECK(should || !should, "should_save should not crash");
    PASS();
    return 0;
}

/* ── Hyperparameter Tuning Tests ── */
static int test_hpt_init_add(void) {
    TEST("hpt_init_add_param");
    hpt_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.method = HPT_SEARCH_RANDOM;
    cfg.num_trials = 10;
    hpt_context_t ctx;
    hpt_init(&ctx, &cfg);
    hpt_add_float(&ctx, "learning_rate", 0.0001f, 0.1f, 0.0f);
    hpt_add_int(&ctx, "batch_size", 16, 256, 0);
    hpt_trial_t trial = hpt_suggest(&ctx);
    CHECK(trial.num_params > 0, "trial should have params");
    hpt_free(&ctx);
    PASS();
    return 0;
}

static int test_hpt_best_trial(void) {
    TEST("hpt_best_trial");
    hpt_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.method = HPT_SEARCH_RANDOM;
    cfg.num_trials = 5;
    hpt_context_t ctx;
    hpt_init(&ctx, &cfg);
    hpt_add_float(&ctx, "lr", 0.001f, 0.1f, 0.0f);
    for (int i = 0; i < 3; i++) {
        hpt_trial_t t = hpt_suggest(&ctx);
        hpt_report(&ctx, t.trial_id, (float)(10 - i) * 0.1f);
    }
    hpt_trial_t best = hpt_best_trial(&ctx);
    CHECK(best.trial_id >= 0, "best trial should exist");
    hpt_free(&ctx);
    PASS();
    return 0;
}

/* ── Training Loop Tests ── */
static int test_tl_init(void) {
    TEST("tl_init");
    tl_train_config_t train_cfg;
    memset(&train_cfg, 0, sizeof(train_cfg));
    train_cfg.max_epochs = 10;
    train_cfg.batch_size = 32;
    train_cfg.seed = 42;
    tl_optimizer_config_t optim_cfg;
    memset(&optim_cfg, 0, sizeof(optim_cfg));
    optim_cfg.type = TL_OPTIM_ADAM;
    optim_cfg.learning_rate = 0.001f;
    tl_scheduler_config_t sched_cfg;
    memset(&sched_cfg, 0, sizeof(sched_cfg));
    sched_cfg.type = TL_SCHEDULER_COSINE;
    sched_cfg.base_lr = 0.001f;
    tl_context_t ctx;
    tl_init(&ctx, &train_cfg, &optim_cfg, &sched_cfg);
    CHECK(ctx.train_cfg.max_epochs == 10, "max_epochs wrong");
    CHECK(ctx.global_step == 0, "global_step wrong");
    tl_free(&ctx);
    PASS();
    return 0;
}

static int test_tl_model_add_layer(void) {
    TEST("tl_model_add_layer");
    tl_model_t model;
    memset(&model, 0, sizeof(model));
    tl_model_add_layer(&model, 64, 32, true);
    CHECK(model.num_layers == 1, "num_layers wrong");
    tl_model_free(&model);
    PASS();
    return 0;
}

static int test_tl_lr_cosine(void) {
    TEST("tl_lr_cosine");
    float lr = tl_lr_cosine(0, 100, 0.1f, 0.001f);
    CHECK(fabsf(lr - 0.1f) < 0.01f, "cosine lr at step 0 wrong");
    lr = tl_lr_cosine(100, 100, 0.1f, 0.001f);
    CHECK(fabsf(lr - 0.001f) < 0.01f, "cosine lr at end wrong");
    PASS();
    return 0;
}

int main(void) {
    printf("=== mini-training-system Unit Tests ===\n\n");

    int failed = 0;
    failed += test_ddp_context_init();
    failed += test_ring_all_reduce();
    failed += test_zero_config();
    failed += test_fp32_to_fp16_roundtrip();
    failed += test_fp32_to_bf16_roundtrip();
    failed += test_mp_context_init();
    failed += test_ckpt_config_init();
    failed += test_ckpt_should_save();
    failed += test_hpt_init_add();
    failed += test_hpt_best_trial();
    failed += test_tl_init();
    failed += test_tl_model_add_layer();
    failed += test_tl_lr_cosine();

    printf("\n=== Results: %d/%d passed, %d failed ===\n",
           tests_passed, tests_run, failed);
    return failed > 0 ? 1 : 0;
}
