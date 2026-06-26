#include "hyperparam_tune.h"
#include "training_loop.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define NUM_TRAIN_SAMPLES 1000
#define NUM_VAL_SAMPLES   200
#define FEATURE_DIM       20
#define NUM_CLASSES       3
#define TUNE_TRIALS       30

static float train_data[NUM_TRAIN_SAMPLES * FEATURE_DIM];
static float train_labels[NUM_TRAIN_SAMPLES * NUM_CLASSES];
static float val_data[NUM_VAL_SAMPLES * FEATURE_DIM];
static float val_labels[NUM_VAL_SAMPLES * NUM_CLASSES];

static void generate_synthetic_data(void) {
    srand(42);
    for (int i = 0; i < NUM_TRAIN_SAMPLES; i++) {
        int c = rand() % NUM_CLASSES;
        for (int j = 0; j < FEATURE_DIM; j++) {
            train_data[i * FEATURE_DIM + j] = (float)(c == 0 ? 1.0 : c == 1 ? -1.0 : 0.0)
                + ((float)rand() / (float)RAND_MAX - 0.5f) * 0.5f;
        }
        memset(train_labels + i * NUM_CLASSES, 0, NUM_CLASSES * sizeof(float));
        train_labels[i * NUM_CLASSES + c] = 1.0f;
    }
    for (int i = 0; i < NUM_VAL_SAMPLES; i++) {
        int c = rand() % NUM_CLASSES;
        for (int j = 0; j < FEATURE_DIM; j++) {
            val_data[i * FEATURE_DIM + j] = (float)(c == 0 ? 1.0 : c == 1 ? -1.0 : 0.0)
                + ((float)rand() / (float)RAND_MAX - 0.5f) * 0.5f;
        }
        memset(val_labels + i * NUM_CLASSES, 0, NUM_CLASSES * sizeof(float));
        val_labels[i * NUM_CLASSES + c] = 1.0f;
    }
}

static float run_trial(const hpt_trial_t* trial) {
    float lr = hpt_get_float(trial, 0);
    int hidden_dim = hpt_get_int(trial, 1);
    int batch = hpt_get_int(trial, 2);

    tl_train_config_t tcfg = {0};
    tcfg.max_epochs = 5;
    tcfg.batch_size = batch;
    tcfg.max_steps = 100;
    tcfg.log_every_n_steps = 10000;
    tcfg.eval_every_n_steps = 10000;

    tl_optimizer_config_t ocfg = {0};
    ocfg.type = TL_OPTIM_ADAM;
    ocfg.learning_rate = lr;
    ocfg.weight_decay = 0.0001f;
    ocfg.beta1 = 0.9f;
    ocfg.beta2 = 0.999f;
    ocfg.epsilon = 1e-8f;

    tl_scheduler_config_t scfg = {0};
    scfg.type = TL_SCHEDULER_COSINE;
    scfg.base_lr = lr;
    scfg.min_lr = lr * 0.01f;

    tl_context_t ctx;
    tl_init(&ctx, &tcfg, &ocfg, &scfg);

    tl_register_dataset(&ctx, train_data, train_labels,
                         NUM_TRAIN_SAMPLES, FEATURE_DIM, NUM_CLASSES, false);
    tl_register_dataset(&ctx, val_data, val_labels,
                         NUM_VAL_SAMPLES, FEATURE_DIM, NUM_CLASSES, true);

    tl_model_add_layer(&ctx.model, FEATURE_DIM, hidden_dim, true);
    tl_model_add_layer(&ctx.model, hidden_dim, hidden_dim / 2, true);
    tl_model_add_layer(&ctx.model, hidden_dim / 2, NUM_CLASSES, true);

    tl_train_run(&ctx);

    float val_loss = ctx.metrics.val_loss.running_avg;
    tl_free(&ctx);
    return val_loss;
}

static void demo_grid_search(void) {
    printf("\n--- Grid Search ---\n");

    hpt_config_t cfg = {0};
    cfg.method = HPT_SEARCH_GRID;
    cfg.mode = HPT_MODE_MINIMIZE;
    cfg.num_trials = 18;
    cfg.random_seed = 42;

    hpt_context_t ctx;
    hpt_init(&ctx, &cfg);

    hpt_add_float(&ctx, "learning_rate", 0.0001f, 0.1f, 0.001f);
    hpt_add_int(&ctx, "hidden_dim", 32, 256, 32);
    hpt_add_int(&ctx, "batch_size", 16, 128, 16);

    printf("  Parameters: lr=[1e-4,0.1], hidden=[32,256], batch=[16,128]\n");

    float best_loss = 1e38f;
    hpt_trial_t best_trial;

    for (int i = 0; i < cfg.num_trials && i < 9; i++) {
        hpt_trial_t trial = hpt_suggest(&ctx);
        if (trial.trial_id < 0) break;

        float loss = run_trial(&trial);
        hpt_report(&ctx, trial.trial_id, loss);

        printf("    Trial %2d: lr=%.6f, hidden=%3d, batch=%3d, loss=%.4f",
               trial.trial_id,
               hpt_get_float(&trial, 0),
               hpt_get_int(&trial, 1),
               hpt_get_int(&trial, 2),
               loss);

        if (loss < best_loss) {
            best_loss = loss;
            best_trial = trial;
            printf(" *BEST*");
        }
        printf("\n");
    }

    printf("  Best: lr=%.6f, hidden=%d, batch=%d, loss=%.4f\n",
           hpt_get_float(&best_trial, 0),
           hpt_get_int(&best_trial, 1),
           hpt_get_int(&best_trial, 2),
           best_loss);

    hpt_free(&ctx);
}

static void demo_random_search(void) {
    printf("\n--- Random Search ---\n");

    hpt_config_t cfg = {0};
    cfg.method = HPT_SEARCH_RANDOM;
    cfg.mode = HPT_MODE_MINIMIZE;
    cfg.num_trials = 20;
    cfg.random_seed = 123;

    hpt_context_t ctx;
    hpt_init(&ctx, &cfg);
    hpt_add_float(&ctx, "learning_rate", 0.0001f, 0.1f, 0.001f);
    hpt_add_int(&ctx, "hidden_dim", 32, 256, 32);
    hpt_add_int(&ctx, "batch_size", 16, 128, 16);

    float best_loss = 1e38f;
    for (int i = 0; i < TUNE_TRIALS / 3; i++) {
        hpt_trial_t trial = hpt_suggest(&ctx);
        if (trial.trial_id < 0) break;

        float loss = run_trial(&trial);
        hpt_report(&ctx, trial.trial_id, loss);

        if ((i + 1) % 5 == 0 || loss < best_loss || i == 0) {
            printf("    Trial %2d: lr=%.6f, hidden=%3d, batch=%3d, loss=%.4f",
                   trial.trial_id,
                   hpt_get_float(&trial, 0),
                   hpt_get_int(&trial, 1),
                   hpt_get_int(&trial, 2),
                   loss);
            if (loss < best_loss) { best_loss = loss; printf(" *"); }
            printf("\n");
        }
    }
    printf("  Best loss after random search: %.4f\n", best_loss);
    hpt_free(&ctx);
}

static void demo_bayesian_optimization(void) {
    printf("\n--- Bayesian Optimization (GP Model) ---\n");

    hpt_config_t cfg = {0};
    cfg.method = HPT_SEARCH_BAYESIAN;
    cfg.mode = HPT_MODE_MINIMIZE;
    cfg.num_trials = 20;
    cfg.n_initial_points = 3;
    cfg.exploration_ratio = 0.1f;

    hpt_context_t ctx;
    hpt_init(&ctx, &cfg);
    hpt_add_float(&ctx, "learning_rate", 0.0001f, 0.1f, 0.001f);
    hpt_add_int(&ctx, "hidden_dim", 32, 256, 32);
    hpt_add_int(&ctx, "batch_size", 16, 128, 16);

    gp_model_t gp;
    gp_model_init(&gp, 3, 1e-4f);

    float best_vals[10];
    int n_best = 0;

    for (int i = 0; i < 10; i++) {
        hpt_trial_t trial = hpt_suggest(&ctx);
        float loss = run_trial(&trial);
        hpt_report(&ctx, trial.trial_id, loss);
        if (n_best < 10) best_vals[n_best++] = loss;
        if ((i + 1) % 3 == 0) {
            printf("    Trial %2d: loss=%.4f\n", trial.trial_id, loss);
        }
    }

    float mean, std;
    float test_x[3] = {0.001f, 128.0f, 64.0f};
    gp_model_predict(&gp, test_x, &mean, &std);
    printf("  GP predict(lr=0.001, hidden=128, batch=64): mean=%.4f, std=%.4f\n", mean, std);

    float ei = gp_model_acq_ei(&gp, test_x, best_vals[0]);
    printf("  Expected Improvement at test point: %.6f\n", ei);

    gp_model_free(&gp);
    hpt_free(&ctx);
}

static void demo_hyperband(void) {
    printf("\n--- Hyperband (Successive Halving) ---\n");

    hpt_config_t cfg = {0};
    cfg.method = HPT_SEARCH_HYPERBAND;
    cfg.mode = HPT_MODE_MINIMIZE;
    cfg.max_epochs = 27;
    cfg.reduction_factor = 3.0f;
    cfg.grace_period = 1;
    cfg.num_trials = 10;

    hpt_context_t ctx;
    hpt_init(&ctx, &cfg);

    hyperband_state_t hb;
    hyperband_init(&hb, cfg.max_epochs, cfg.reduction_factor);

    printf("  Max epochs: %d, Reduction factor: %.1f\n", cfg.max_epochs, cfg.reduction_factor);
    printf("  Number of brackets: %d\n", hb.num_brackets);

    for (int b = 0; b < hb.num_brackets; b++) {
        int n_configs = hyperband_get_num_configs(&hb, b, cfg.num_trials);
        printf("  Bracket %d: %d configs\n", b, n_configs);

        int n_rungs = 0;
        int budget = cfg.max_epochs;
        while (budget >= cfg.min_epochs) {
            n_rungs++;
            budget = (int)((float)budget / cfg.reduction_factor);
        }
        printf("    Rungs: %d (until min_epochs=%d)\n", n_rungs, cfg.min_epochs);
    }

    float* vals = (float*)malloc(10 * sizeof(float));
    int* indices = (int*)malloc(10 * sizeof(int));
    for (int i = 0; i < 10; i++) vals[i] = (float)(10 - i) * 0.1f;
    int top = hyperband_select_top(vals, 10, 3, indices);
    printf("  Top %d after halving: ", top);
    for (int i = 0; i < top; i++) printf("[%d]=%.2f ", indices[i], vals[indices[i]]);
    printf("\n");

    free(vals); free(indices);
    if (hb.bracket_milestones) free(hb.bracket_milestones);
    if (hb.bracket_num_configs) free(hb.bracket_num_configs);
    if (hb.bracket_budgets) free(hb.bracket_budgets);
    hpt_free(&ctx);
}

static void demo_early_stopping(void) {
    printf("\n--- Early Stopping ---\n");

    int patience = 5;
    float min_delta = 0.001f;
    float best_values[] = {1.0f, 0.8f, 0.6f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
    int n_steps = 8;

    float current_best = 1e38f;
    int counter = 0;

    for (int s = 0; s < n_steps; s++) {
        float val = best_values[s];
        if (val < current_best - min_delta) {
            current_best = val;
            counter = 0;
            printf("  Step %d: val=%.4f (improved, reset patience)\n", s, val);
        } else {
            counter++;
            printf("  Step %d: val=%.4f (no improvement, patience=%d/%d)\n",
                   s, val, counter, patience);
        }
        if (hpt_early_stopping_check(current_best, counter, patience, min_delta)) {
            printf("  EARLY STOPPING triggered at step %d!\n", s);
            break;
        }
    }
}

static void demo_optuna_tpe(void) {
    printf("\n--- TPE Sampler (Optuna-style) ---\n");

    int n_observed = 100;
    int n_dims = 2;
    float* observed = (float*)malloc((size_t)(n_observed * n_dims) * sizeof(float));

    for (int i = 0; i < n_observed; i++) {
        observed[i * n_dims + 0] = ((float)rand() / (float)RAND_MAX);
        observed[i * n_dims + 1] = ((float)rand() / (float)RAND_MAX);
    }

    printf("  Observed %d points with %d dimensions each\n", n_observed, n_dims);
    float sample = tpe_sample((const float*)observed, n_observed, 0.25f, n_dims, 42);
    printf("  TPE sample: %.4f\n", sample);

    float good[50], bad[50];
    for (int i = 0; i < 50; i++) { good[i] = (float)i / 50.0f; bad[i] = (float)(50 - i) / 50.0f; }
    float x_test[2] = {0.3f, 0.7f};
    float ratio = tpe_log_ratio(x_test, good, 50, bad, 50);
    printf("  TPE log-ratio for test point: %.4f\n", ratio);

    free(observed);
}

static void demo_experiment_tracking(void) {
    printf("\n--- Experiment Metadata & Plotting ---\n");

    hpt_config_t cfg = {0};
    cfg.method = HPT_SEARCH_RANDOM;
    cfg.num_trials = 5;
    strncpy(cfg.study_name, "demo_study", HPT_MAX_NAME - 1);

    hpt_context_t ctx;
    hpt_init(&ctx, &cfg);
    hpt_add_float(&ctx, "lr", 0.001f, 0.1f, 0.001f);
    hpt_add_int(&ctx, "hidden", 32, 128, 16);

    for (int i = 0; i < 5; i++) {
        hpt_trial_t trial = hpt_suggest(&ctx);
        float loss = run_trial(&trial);
        hpt_report(&ctx, trial.trial_id, loss);
    }

    hpt_plot_optimization_history(&ctx, "demo_opt_history.txt");
    hpt_plot_param_importance(&ctx, "demo_param_importance.txt");
    printf("  Saved optimization history to demo_opt_history.txt\n");
    printf("  Saved parameter importance to demo_param_importance.txt\n");

    hpt_trial_t best = hpt_best_trial(&ctx);
    printf("  Best trial: id=%d, value=%.4f\n", best.trial_id, best.objective_value);

    hpt_free(&ctx);
}

int main(void) {
    printf("============================================================\n");
    printf("  Hyperparameter Tuning Demo\n");
    printf("============================================================\n");

    printf("\nGenerating synthetic dataset (%d train, %d val)...\n",
           NUM_TRAIN_SAMPLES, NUM_VAL_SAMPLES);
    generate_synthetic_data();

    demo_grid_search();
    demo_random_search();
    demo_bayesian_optimization();
    demo_hyperband();
    demo_early_stopping();
    demo_optuna_tpe();
    demo_experiment_tracking();

    printf("\nAll hyperparameter tuning demos complete.\n");
    return 0;
}
