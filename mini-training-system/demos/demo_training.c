#include "training_loop.h"
#include "mixed_precision.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define N_TRAIN 800
#define N_VAL   200
#define FEAT_DIM 10
#define N_CLASSES 3

static float s_train_data[N_TRAIN * FEAT_DIM];
static float s_train_labels[N_TRAIN * N_CLASSES];
static float s_val_data[N_VAL * FEAT_DIM];
static float s_val_labels[N_VAL * N_CLASSES];

static void gen_data(void) {
    srand(123);
    for (int i = 0; i < N_TRAIN; i++) {
        int c = i % N_CLASSES;
        float center = (float)(c - 1) * 1.5f;
        for (int j = 0; j < FEAT_DIM; j++) {
            s_train_data[i * FEAT_DIM + j] = center
                + ((float)rand() / (float)RAND_MAX - 0.5f) * 2.0f;
        }
        memset(s_train_labels + i * N_CLASSES, 0, N_CLASSES * sizeof(float));
        s_train_labels[i * N_CLASSES + c] = 1.0f;
    }
    for (int i = 0; i < N_VAL; i++) {
        int c = i % N_CLASSES;
        float center = (float)(c - 1) * 1.5f;
        for (int j = 0; j < FEAT_DIM; j++) {
            s_val_data[i * FEAT_DIM + j] = center
                + ((float)rand() / (float)RAND_MAX - 0.5f) * 2.0f;
        }
        memset(s_val_labels + i * N_CLASSES, 0, N_CLASSES * sizeof(float));
        s_val_labels[i * N_CLASSES + c] = 1.0f;
    }
}

static void demo_basic_training(void) {
    printf("\n--- Basic Training Loop ---\n");

    tl_train_config_t tcfg = {0};
    tcfg.max_epochs = 5;
    tcfg.batch_size = 32;
    tcfg.log_every_n_steps = 5;
    tcfg.eval_every_n_steps = 5;
    tcfg.max_steps = 200;
    tcfg.max_grad_norm = 1.0f;

    tl_optimizer_config_t ocfg = {0};
    ocfg.type = TL_OPTIM_SGD;
    ocfg.learning_rate = 0.01f;
    ocfg.weight_decay = 0.0001f;
    ocfg.momentum = 0.9f;

    tl_scheduler_config_t scfg = {0};
    scfg.type = TL_SCHEDULER_STEP;
    scfg.base_lr = 0.01f;
    scfg.min_lr = 0.0001f;

    tl_context_t ctx;
    tl_init(&ctx, &tcfg, &ocfg, &scfg);
    tl_set_log_file(&ctx, "training_log.txt");

    tl_register_dataset(&ctx, s_train_data, s_train_labels,
                         N_TRAIN, FEAT_DIM, N_CLASSES, false);
    tl_register_dataset(&ctx, s_val_data, s_val_labels,
                         N_VAL, FEAT_DIM, N_CLASSES, true);

    int hidden = 64;
    tl_model_add_layer(&ctx.model, FEAT_DIM, hidden, true);
    tl_model_add_layer(&ctx.model, hidden, hidden / 2, true);
    tl_model_add_layer(&ctx.model, hidden / 2, N_CLASSES, true);

    printf("  Model: %d -> %d -> %d -> %d\n", FEAT_DIM, hidden, hidden/2, N_CLASSES);
    printf("  Training %d epochs on %d samples (batch=%d)...\n",
           tcfg.max_epochs, N_TRAIN, tcfg.batch_size);

    for (int e = 0; e < tcfg.max_epochs; e++) {
        tl_train_epoch(&ctx);
        if (N_VAL > 0) tl_validate(&ctx);
        printf("  Epoch %d | train_loss=%.4f val_loss=%.4f acc=%.4f lr=%.6f\n",
               e + 1,
               ctx.metrics.train_loss.value,
               ctx.metrics.val_loss.running_avg,
               ctx.metrics.accuracy.running_avg,
               ctx.current_lr);
    }

    printf("  Final: loss=%.4f acc=%.4f\n",
           ctx.metrics.train_loss.value,
           ctx.metrics.accuracy.running_avg);
    tl_free(&ctx);
}

static void demo_lr_schedulers(void) {
    printf("\n--- LR Scheduler Comparison ---\n");

    int total_steps = 100;
    float base_lr = 0.1f;
    float min_lr = 0.001f;

    printf("  %-20s: ", "Cosine");
    for (int s = 0; s < total_steps; s += 20) {
        float lr = tl_lr_cosine(s, total_steps, base_lr, min_lr);
        printf("%.4f ", lr);
    }
    printf("\n");

    printf("  %-20s: ", "Warmup+Cosine(w=10)");
    for (int s = 0; s < total_steps; s += 20) {
        float lr = tl_lr_warmup_cosine(s, 10, total_steps, base_lr, min_lr);
        printf("%.4f ", lr);
    }
    printf("\n");

    tl_scheduler_config_t scfg_exp = {0};
    scfg_exp.type = TL_SCHEDULER_EXPONENTIAL;
    scfg_exp.base_lr = base_lr;
    scfg_exp.gamma = 0.95f;
    scfg_exp.min_lr = min_lr;

    tl_train_config_t tcfg = {0};
    tcfg.max_epochs = 10;
    tcfg.max_steps = total_steps;

    tl_optimizer_config_t ocfg = {0};
    ocfg.learning_rate = base_lr;

    tl_context_t ctx;
    tl_init(&ctx, &tcfg, &ocfg, &scfg_exp);

    printf("  %-20s: ", "Exponential(g=0.95)");
    for (int s = 0; s < total_steps; s += 20) {
        ctx.global_step = s;
        tl_scheduler_step(&ctx);
        printf("%.4f ", tl_lr_get(&ctx));
    }
    printf("\n");
    tl_free(&ctx);

    printf("  %-20s: ", "Cyclic");
    for (int s = 0; s < total_steps; s += 20) {
        float lr = tl_lr_cosine(s % 40, 40, base_lr, min_lr);
        printf("%.4f ", lr);
    }
    printf("\n");
}

static void demo_metrics(void) {
    printf("\n--- Metrics Calculation ---\n");

    size_t n = 20;
    size_t cls = 3;
    float* preds = (float*)calloc(n * cls, sizeof(float));
    float* targs = (float*)calloc(n * cls, sizeof(float));

    srand(42);
    int correct = 0;
    for (size_t b = 0; b < n; b++) {
        int true_cls = rand() % (int)cls;
        int pred_cls = (rand() % 3 == 0) ? true_cls : rand() % (int)cls;
        if (pred_cls == true_cls) correct++;

        targs[b * cls + true_cls] = 1.0f;
        preds[b * cls + pred_cls] = 1.0f;
    }

    float acc = tl_compute_accuracy(preds, targs, n, cls);
    float prec = tl_compute_precision(preds, targs, n, cls);
    float rec = tl_compute_recall(preds, targs, n, cls);
    float f1 = tl_compute_f1(prec, rec);

    printf("  Samples=%zu, Classes=%zu, Correct=%d\n", n, cls, correct);
    printf("  Accuracy:  %.4f\n", acc);
    printf("  Precision: %.4f\n", prec);
    printf("  Recall:    %.4f\n", rec);
    printf("  F1 Score:  %.4f\n", f1);

    free(preds); free(targs);
}

static void demo_gradient_techniques(void) {
    printf("\n--- Gradient Techniques ---\n");

    tl_train_config_t tcfg = {0};
    tcfg.max_epochs = 1;
    tcfg.batch_size = 64;
    tcfg.accumulation_steps = 4;
    tcfg.grad_accum_steps = 4;
    tcfg.max_grad_norm = 1.0f;
    tcfg.max_steps = 20;

    tl_optimizer_config_t ocfg = {0};
    ocfg.learning_rate = 0.01f;

    tl_scheduler_config_t scfg = {0};
    scfg.base_lr = 0.01f;

    tl_context_t ctx;
    tl_init(&ctx, &tcfg, &ocfg, &scfg);

    tl_register_dataset(&ctx, s_train_data, s_train_labels,
                         N_TRAIN, FEAT_DIM, N_CLASSES, false);

    tl_model_add_layer(&ctx.model, FEAT_DIM, 32, true);
    tl_model_add_layer(&ctx.model, 32, N_CLASSES, true);

    printf("  Gradient accumulation: %d steps\n", tcfg.accumulation_steps);

    float baseline_norm = tl_compute_grad_norm(&ctx);
    printf("  Initial grad norm: %.4f\n", baseline_norm);

    for (int s = 0; s < 8; s++) {
        size_t start = ((size_t)s * 32) % N_TRAIN;
        tl_gradient_accumulation_step(&ctx,
            s_train_data + start * FEAT_DIM,
            s_train_labels + start * N_CLASSES, 32);
    }

    float after_norm = tl_compute_grad_norm(&ctx);
    printf("  After 8 accumulation steps, norm: %.4f\n", after_norm);

    printf("  Clipping to max_norm=%.1f...\n", tcfg.max_grad_norm);
    tl_clip_grad_norm(&ctx, tcfg.max_grad_norm);
    float clipped_norm = tl_compute_grad_norm(&ctx);
    printf("  After clip, norm: %.4f\n", clipped_norm);

    tl_reset_accumulated_gradients(&ctx);

    printf("  Gradient checkpointing demo:\n");
    float* dummy_in = (float*)calloc(FEAT_DIM, sizeof(float));
    float* dummy_out = (float*)calloc(N_CLASSES, sizeof(float));
    tl_gradient_checkpoint_save(&ctx, 0, NULL);
    tl_gradient_checkpoint_recompute(&ctx, 0, dummy_in, dummy_out);
    tl_gradient_checkpoint_free(&ctx, 0);
    printf("    Recompute layer 0: OK\n");
    free(dummy_in); free(dummy_out);

    tl_free(&ctx);
}

static void demo_profiling(void) {
    printf("\n--- Training Profiling ---\n");

    tl_train_config_t tcfg = {0};
    tcfg.max_epochs = 1;
    tcfg.batch_size = 128;
    tcfg.max_steps = 10;
    tcfg.profile = true;

    tl_optimizer_config_t ocfg = {0};
    ocfg.learning_rate = 0.01f;

    tl_scheduler_config_t scfg = {0};
    scfg.base_lr = 0.01f;

    tl_context_t ctx;
    tl_init(&ctx, &tcfg, &ocfg, &scfg);

    tl_register_dataset(&ctx, s_train_data, s_train_labels,
                         N_TRAIN, FEAT_DIM, N_CLASSES, false);

    tl_model_add_layer(&ctx.model, FEAT_DIM, 64, true);
    tl_model_add_layer(&ctx.model, 64, 32, true);
    tl_model_add_layer(&ctx.model, 32, N_CLASSES, true);

    tl_profile_start(&ctx);

    clock_t t0 = clock();
    for (int s = 0; s < 10; s++) {
        size_t start = ((size_t)s * 128) % N_TRAIN;
        tl_train_step(&ctx,
            s_train_data + start * FEAT_DIM,
            s_train_labels + start * N_CLASSES, 128);
    }
    clock_t t1 = clock();
    double total_ms = 1000.0 * (double)(t1 - t0) / (double)CLOCKS_PER_SEC;

    tl_profile_stop(&ctx);
    tl_profile_summary(&ctx);

    tl_profile_entry_t entries[10];
    tl_profile_report(&ctx, entries, 10);

    printf("  Profiled 10 steps:\n");
    printf("    Total wall time: %.1f ms (%.1f ms/step)\n", total_ms, total_ms / 10.0);
    printf("    Forward:  %.1f ms/step\n", entries[0].forward_ms);
    printf("    Backward: %.1f ms/step\n", entries[0].backward_ms);
    printf("    Optim:    %.1f ms/step\n", entries[0].optim_ms);

    tl_free(&ctx);
}

static void demo_mixed_precision_training(void) {
    printf("\n--- Mixed Precision + Training Integration ---\n");

    mp_config_t mp_cfg = {0};
    mp_cfg.mode = MP_MODE_FP16;
    mp_cfg.init_loss_scale = 1024.0f;
    mp_cfg.scale_window = 100;
    mp_cfg.scale_mode = MP_LOSS_SCALE_DYNAMIC;

    mp_context_t mp_ctx;
    mp_init(&mp_ctx, &mp_cfg);

    tl_train_config_t tcfg = {0};
    tcfg.max_epochs = 3;
    tcfg.batch_size = 64;
    tcfg.max_steps = 50;
    tcfg.use_amp = true;
    tcfg.use_grad_scaler = true;
    tcfg.max_grad_norm = 1.0f;

    tl_optimizer_config_t ocfg = {0};
    ocfg.learning_rate = 0.01f;

    tl_scheduler_config_t scfg = {0};
    scfg.type = TL_SCHEDULER_COSINE;
    scfg.base_lr = 0.01f;
    scfg.min_lr = 0.0001f;

    tl_context_t ctx;
    tl_init(&ctx, &tcfg, &ocfg, &scfg);

    tl_register_dataset(&ctx, s_train_data, s_train_labels,
                         N_TRAIN, FEAT_DIM, N_CLASSES, false);

    tl_model_add_layer(&ctx.model, FEAT_DIM, 32, true);
    tl_model_add_layer(&ctx.model, 32, N_CLASSES, true);

    printf("  AMP training (FP16) with dynamic loss scaling\n");
    printf("  Initial scale: %.0f\n", mp_ctx.loss_scale);

    float scale = mp_ctx.loss_scale;
    bool overflow_flag = false;

    for (int s = 0; s < 30; s++) {
        size_t start = ((size_t)s * 64) % N_TRAIN;
        tl_train_step(&ctx,
            s_train_data + start * FEAT_DIM,
            s_train_labels + start * N_CLASSES, 64);

        bool had_overflow = (s > 0 && s % 15 == 0);
        if (had_overflow) overflow_flag = true;
        scale = mp_update_loss_scale(&mp_ctx, had_overflow);

        if (s % 10 == 0) {
            printf("    Step %2d: loss=%.4f scale=%.0f overflow=%s\n",
                   s, ctx.metrics.train_loss.value, scale,
                   had_overflow ? "YES" : "no");
        }
    }

    printf("  Final loss scale: %.0f, overflows: %d\n", scale, mp_ctx.overflow_count);

    printf("\n  Loss curve data saved:\n");
    tl_log_curve_loss(&ctx, "loss_curve.txt");
    tl_log_curve_accuracy(&ctx, "acc_curve.txt");
    printf("    - loss_curve.txt\n");
    printf("    - acc_curve.txt\n");

    tl_free(&ctx);
}

static void demo_full_pipeline(void) {
    printf("\n--- Full Training Pipeline Demo ---\n");

    tl_train_config_t tcfg = {0};
    tcfg.max_epochs = 4;
    tcfg.batch_size = 64;
    tcfg.micro_batch_size = 16;
    tcfg.grad_accum_steps = 4;
    tcfg.max_steps = 200;
    tcfg.max_grad_norm = 1.0f;
    tcfg.log_every_n_steps = 1;
    tcfg.use_grad_checkpointing = true;
    tcfg.use_amp = true;
    tcfg.use_grad_scaler = true;
    tcfg.detect_anomaly = false;
    tcfg.deterministic = false;
    tcfg.seed = 42;

    tl_optimizer_config_t ocfg = {0};
    ocfg.type = TL_OPTIM_ADAMW;
    ocfg.learning_rate = 0.001f;
    ocfg.weight_decay = 0.01f;
    ocfg.beta1 = 0.9f;
    ocfg.beta2 = 0.999f;
    ocfg.epsilon = 1e-8f;

    tl_scheduler_config_t scfg = {0};
    scfg.type = TL_SCHEDULER_COSINE_RESTARTS;
    scfg.base_lr = 0.001f;
    scfg.min_lr = 1e-6f;
    scfg.warmup_steps = 5;
    scfg.t_0 = 20;
    scfg.t_mult = 2;

    tl_context_t ctx;
    tl_init(&ctx, &tcfg, &ocfg, &scfg);
    tl_set_seed(tcfg.seed);

    tl_register_dataset(&ctx, s_train_data, s_train_labels,
                         N_TRAIN, FEAT_DIM, N_CLASSES, false);
    tl_register_dataset(&ctx, s_val_data, s_val_labels,
                         N_VAL, FEAT_DIM, N_CLASSES, true);

    int h1 = 128, h2 = 64, h3 = 32;
    tl_model_add_layer(&ctx.model, FEAT_DIM, h1, true);
    tl_model_add_layer(&ctx.model, h1, h2, true);
    tl_model_add_layer(&ctx.model, h2, h3, true);
    tl_model_add_layer(&ctx.model, h3, N_CLASSES, true);

    printf("  Config:\n");
    printf("    Model:    %d -> %d -> %d -> %d -> %d\n",
           FEAT_DIM, h1, h2, h3, N_CLASSES);
    printf("    Optim:    AdamW (lr=%.4f, wd=%.4f)\n",
           ocfg.learning_rate, ocfg.weight_decay);
    printf("    Schedule: CosineAnnealingWarmRestarts\n");
    printf("    Epochs:   %d, Batch: %d (%d micro x %d accum)\n",
           tcfg.max_epochs, tcfg.batch_size,
           tcfg.micro_batch_size, tcfg.grad_accum_steps);
    printf("    Features: AMP, GradClip(%.1f), GradCheckpoint, Seed=%d\n",
           tcfg.max_grad_norm, tcfg.seed);

    for (int e = 0; e < tcfg.max_epochs; e++) {
        tl_train_epoch(&ctx);
        if (N_VAL > 0) tl_validate(&ctx);

        float prec = tl_compute_precision(NULL, NULL, 0, N_CLASSES);
        float rec  = tl_compute_recall(NULL, NULL, 0, N_CLASSES);
        float f1   = tl_compute_f1(prec, rec);

        printf("  Epoch %d/%d | loss=%.4f val=%.4f acc=%.3f lr=%.6f | fwd=%.1fms bwd=%.1fms\n",
               e + 1, tcfg.max_epochs,
               ctx.metrics.train_loss.value,
               ctx.metrics.val_loss.running_avg,
               ctx.metrics.accuracy.running_avg,
               ctx.current_lr,
               ctx.metrics.forward_time_ms,
               ctx.metrics.backward_time_ms);
    }

    tl_grad_check(&ctx, 1e-4f, 0.01f);

    printf("\n  Final model state:\n");
    printf("    Global step: %d\n", ctx.global_step);
    printf("    LR:          %.6f\n", ctx.current_lr);
    printf("    Best val:    %.4f\n", ctx.best_val_loss);
    tl_profile_summary(&ctx);

    tl_free(&ctx);
    printf("\n  Full pipeline complete.\n");
}

int main(void) {
    printf("============================================================\n");
    printf("  Training Loop Demo\n");
    printf("============================================================\n");

    printf("\nGenerating dataset (%d train, %d val)...\n", N_TRAIN, N_VAL);
    gen_data();

    demo_basic_training();
    demo_lr_schedulers();
    demo_metrics();
    demo_gradient_techniques();
    demo_profiling();
    demo_mixed_precision_training();
    demo_full_pipeline();

    printf("\nAll training demos complete.\n");
    return 0;
}
