#include "checkpoint_save.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define NUM_PARAMS 1024
#define DEMO_DIR "./demo_checkpoints"

typedef struct {
    float* weights;
    float* optimizer_momentum;
    float* optimizer_velocity;
    size_t num_params;
    int    step;
    float  best_loss;
} simple_model_t;

static void model_init(simple_model_t* m, size_t n) {
    m->num_params = n;
    m->weights = (float*)malloc(n * sizeof(float));
    m->optimizer_momentum = (float*)calloc(n, sizeof(float));
    m->optimizer_velocity = (float*)calloc(n, sizeof(float));
    for (size_t i = 0; i < n; i++) m->weights[i] = ((float)(i + 1)) / 100.0f;
    m->step = 0;
    m->best_loss = 1e38f;
}

static void model_free(simple_model_t* m) {
    free(m->weights); free(m->optimizer_momentum); free(m->optimizer_velocity);
    memset(m, 0, sizeof(simple_model_t));
}

static int save_weights_callback(void* userdata, FILE* f) {
    simple_model_t* m = (simple_model_t*)userdata;
    if (!m || !f) return -1;

    fwrite(&m->num_params, sizeof(size_t), 1, f);
    fwrite(m->weights, sizeof(float), m->num_params, f);
    fwrite(m->optimizer_momentum, sizeof(float), m->num_params, f);
    fwrite(m->optimizer_velocity, sizeof(float), m->num_params, f);
    fwrite(&m->step, sizeof(int), 1, f);
    fwrite(&m->best_loss, sizeof(float), 1, f);

    printf("  Saved: %zu params, step=%d, best_loss=%.4f\n",
           m->num_params, m->step, m->best_loss);
    return 0;
}

static int load_weights_callback(void* userdata, FILE* f) {
    simple_model_t* m = (simple_model_t*)userdata;
    if (!m || !f) return -1;

    size_t n;
    if (fread(&n, sizeof(size_t), 1, f) != 1) return -1;
    if (n != m->num_params) {
        printf("  Warning: param count mismatch (%zu vs %zu)\n", n, m->num_params);
    }

    size_t read_n = n < m->num_params ? n : m->num_params;
    fread(m->weights, sizeof(float), read_n, f);
    fread(m->optimizer_momentum, sizeof(float), read_n, f);
    fread(m->optimizer_velocity, sizeof(float), read_n, f);
    fread(&m->step, sizeof(int), 1, f);
    fread(&m->best_loss, sizeof(float), 1, f);

    printf("  Loaded: %zu params, step=%d, best_loss=%.4f\n",
           read_n, m->step, m->best_loss);
    return 0;
}

static float evaluate(simple_model_t* m) {
    float sum = 0.0f;
    for (size_t i = 0; i < m->num_params; i++) sum += m->weights[i] * m->weights[i];
    return sqrtf(sum);
}

static void train_one_step(simple_model_t* m) {
    float lr = 0.001f;
    for (size_t i = 0; i < m->num_params; i++) {
        float grad = 2.0f * m->weights[i];
        m->optimizer_momentum[i] = 0.9f * m->optimizer_momentum[i] + grad;
        m->optimizer_velocity[i] = 0.999f * m->optimizer_velocity[i] + grad * grad;
        m->weights[i] -= lr * m->optimizer_momentum[i] / (sqrtf(m->optimizer_velocity[i]) + 1e-8f);
    }
    m->step++;
}

static void demo_save_load_basic(void) {
    printf("\n--- Basic Save / Load ---\n");

    ckpt_config_t cfg = {0};
    strncpy(cfg.base_dir, DEMO_DIR, CKPT_MAX_PATH - 1);
    strncpy(cfg.run_name, "basic_demo", CKPT_MAX_NAME - 1);
    cfg.strategy = CKPT_STRATEGY_MANUAL;
    cfg.save_optimizer = true;

    ckpt_context_t ctx;
    ckpt_init(&ctx, &cfg);

    simple_model_t model;
    model_init(&model, NUM_PARAMS);

    for (int step = 0; step < 5; step++) {
        train_one_step(&model);
        float loss = evaluate(&model);
        printf("  Step %d: loss=%.4f\n", step, loss);

        if (step == 2) {
            ckpt_save(&ctx, step, 0, save_weights_callback, &model);
        }
    }

    simple_model_t loaded_model;
    model_init(&loaded_model, NUM_PARAMS);

    char path[CKPT_MAX_PATH];
    ckpt_build_path(&ctx, path, sizeof(path), 2, 0, NULL);
    printf("  Loading from: %s\n", path);
    int ret = ckpt_load(&ctx, path, load_weights_callback, &loaded_model);
    printf("  Load result: %s\n", ret == 0 ? "SUCCESS" : "FAILED");

    float orig_loss = evaluate(&model);
    float loaded_loss = evaluate(&loaded_model);
    printf("  Original loss=%.4f, Loaded loss=%.4f\n", orig_loss, loaded_loss);

    model_free(&model);
    model_free(&loaded_model);
}

static void demo_best_checkpoint(void) {
    printf("\n--- Best Checkpoint Strategy ---\n");

    ckpt_config_t cfg = {0};
    strncpy(cfg.base_dir, DEMO_DIR, CKPT_MAX_PATH - 1);
    strncpy(cfg.run_name, "best_demo", CKPT_MAX_NAME - 1);
    cfg.strategy = CKPT_STRATEGY_BEST;
    cfg.metric = CKPT_METRIC_LOSS;
    cfg.mode = CKPT_MODE_MIN;
    cfg.save_optimizer = true;

    ckpt_context_t ctx;
    ckpt_init(&ctx, &cfg);

    simple_model_t model;
    model_init(&model, NUM_PARAMS);

    float best_seen = 1e38f;
    for (int step = 0; step < 20; step++) {
        train_one_step(&model);
        float loss = evaluate(&model);

        if (loss < best_seen) best_seen = loss;
        bool should = ckpt_should_save(&ctx, step, 0, loss);
        if (should) {
            printf("  Step %d: loss=%.4f -> SAVING (best so far)\n", step, loss);
            ckpt_save_best(&ctx, step, 0, loss, save_weights_callback, &model);
        }
    }
    printf("  Final best loss: %.4f\n", best_seen);
    model_free(&model);
}

static void demo_periodic_checkpoint(void) {
    printf("\n--- Periodic Checkpoint Strategy ---\n");

    ckpt_config_t cfg = {0};
    strncpy(cfg.base_dir, DEMO_DIR, CKPT_MAX_PATH - 1);
    strncpy(cfg.run_name, "periodic_demo", CKPT_MAX_NAME - 1);
    cfg.strategy = CKPT_STRATEGY_PERIODIC;
    cfg.save_every_n_steps = 50;
    cfg.max_checkpoints = 3;
    cfg.keep_last_n = true;
    cfg.keep_last_n_value = 2;

    ckpt_context_t ctx;
    ckpt_init(&ctx, &cfg);

    simple_model_t model;
    model_init(&model, NUM_PARAMS);

    for (int step = 0; step < 200; step++) {
        train_one_step(&model);
        float loss = evaluate(&model);

        if (ckpt_should_save(&ctx, step, 0, loss)) {
            printf("  Step %d: Periodic save, loss=%.4f\n", step, loss);
            ckpt_save_latest(&ctx, step, 0, save_weights_callback, &model);
        }
    }
    ckpt_cleanup_old(&ctx);
    printf("  Cleaned up old checkpoints (keeping last %d)\n", cfg.keep_last_n_value);
    model_free(&model);
}

static void demo_fault_tolerance(void) {
    printf("\n--- Fault Tolerance & Recovery ---\n");

    ckpt_config_t cfg = {0};
    strncpy(cfg.base_dir, DEMO_DIR, CKPT_MAX_PATH - 1);
    strncpy(cfg.run_name, "fault_demo", CKPT_MAX_NAME - 1);
    cfg.strategy = CKPT_STRATEGY_PERIODIC;
    cfg.save_every_n_steps = 30;
    cfg.fault_tolerant = true;
    cfg.save_on_interrupt = true;

    ckpt_context_t ctx;
    ckpt_init(&ctx, &cfg);

    simple_model_t model;
    model_init(&model, NUM_PARAMS);

    printf("  Training 100 steps with periodic saves...\n");
    for (int step = 0; step < 100; step++) {
        train_one_step(&model);
        if (ckpt_should_save(&ctx, step, 0, evaluate(&model))) {
            ckpt_save(&ctx, step, 0, save_weights_callback, &model);
        }
    }

    printf("  Simulating crash at step 100...\n");

    simple_model_t recovered;
    model_init(&recovered, NUM_PARAMS);
    if (ckpt_recover_from_fault(&ctx, load_weights_callback, &recovered)) {
        printf("  Recovery SUCCESS - resumed from step %d\n", recovered.step);
    } else {
        printf("  Recovery FAILED\n");
    }

    model_free(&model);
    model_free(&recovered);
}

static void demo_async_checkpoint(void) {
    printf("\n--- Async Checkpoint Demo ---\n");

    ckpt_config_t cfg = {0};
    strncpy(cfg.base_dir, DEMO_DIR, CKPT_MAX_PATH - 1);
    strncpy(cfg.run_name, "async_demo", CKPT_MAX_NAME - 1);
    cfg.async_checkpointing = true;

    ckpt_context_t ctx;
    ckpt_init(&ctx, &cfg);

    simple_model_t model;
    model_init(&model, NUM_PARAMS);

    for (int step = 0; step < 10; step++) {
        train_one_step(&model);

        if (step == 5) {
            printf("  Starting async save at step %d...\n", step);
            ckpt_save_async_start(&ctx, step, 0, save_weights_callback, &model);
        }

        while (!ckpt_save_async_is_done(&ctx)) {
            printf("  ... training continues while saving ...\n");
            train_one_step(&model);
        }
    }
    ckpt_save_async_wait(&ctx);
    printf("  Async save complete\n");
    model_free(&model);
}

static void demo_metadata(void) {
    printf("\n--- Checkpoint Metadata ---\n");

    ckpt_config_t cfg = {0};
    strncpy(cfg.base_dir, DEMO_DIR, CKPT_MAX_PATH - 1);
    strncpy(cfg.run_name, "meta_demo", CKPT_MAX_NAME - 1);

    ckpt_context_t ctx;
    ckpt_init(&ctx, &cfg);
    ctx.state.epoch = 5;
    ctx.state.global_step = 1000;
    ctx.state.best_metric_value = 0.0423f;
    ctx.metadata.num_params = 50000;

    char meta_path[CKPT_MAX_PATH];
    snprintf(meta_path, sizeof(meta_path), "%s/meta.txt", DEMO_DIR);
    ckpt_save_metadata(&ctx, meta_path);
    printf("  Saved metadata to: %s\n", meta_path);

    ckpt_context_t loaded;
    ckpt_init(&loaded, &cfg);
    ckpt_load_metadata(&loaded, meta_path);
    printf("  Loaded: epoch=%d step=%d best=%.4f params=%zu\n",
           loaded.state.epoch, loaded.state.global_step,
           loaded.state.best_metric_value, loaded.metadata.num_params);
}

int main(void) {
    printf("============================================================\n");
    printf("  Checkpoint Save & Resume Example\n");
    printf("============================================================\n");

    demo_save_load_basic();
    demo_best_checkpoint();
    demo_periodic_checkpoint();
    demo_fault_tolerance();
    demo_async_checkpoint();
    demo_metadata();

    printf("\nAll checkpoints saved under: %s\n", DEMO_DIR);
    printf("All checkpoint demos complete.\n");
    return 0;
}
