#ifndef TRAINING_LOOP_H
#define TRAINING_LOOP_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TL_MAX_LAYERS 512
#define TL_MAX_NAME 256

typedef enum {
    TL_SCHEDULER_STEP = 0,
    TL_SCHEDULER_COSINE,
    TL_SCHEDULER_COSINE_RESTARTS,
    TL_SCHEDULER_CYCLIC,
    TL_SCHEDULER_ONE_CYCLE,
    TL_SCHEDULER_EXPONENTIAL,
    TL_SCHEDULER_PLATEAU,
    TL_SCHEDULER_CUSTOM,
} tl_scheduler_type_t;

typedef enum {
    TL_OPTIM_SGD = 0,
    TL_OPTIM_ADAM,
    TL_OPTIM_ADAMW,
    TL_OPTIM_RMSPROP,
    TL_OPTIM_LAMB,
    TL_OPTIM_LION,
} tl_optimizer_type_t;

typedef enum {
    TL_METRIC_LOSS = 0,
    TL_METRIC_ACCURACY,
    TL_METRIC_PRECISION,
    TL_METRIC_RECALL,
    TL_METRIC_F1,
    TL_METRIC_AUC,
    TL_METRIC_PERLEXITY,
    TL_METRIC_CUSTOM,
} tl_metric_type_t;

typedef struct {
    tl_scheduler_type_t type;
    float base_lr;
    float max_lr;
    float min_lr;
    int   warmup_steps;
    int   total_steps;
    int   step_size;
    float gamma;
    int   t_0;
    int   t_mult;
    float plateau_factor;
    int   plateau_patience;
    float plateau_threshold;
} tl_scheduler_config_t;

typedef struct {
    tl_optimizer_type_t type;
    float learning_rate;
    float weight_decay;
    float beta1;
    float beta2;
    float epsilon;
    float momentum;
    float alpha;
    float rho;
    bool  amsgrad;
    int   grad_accum_steps;
} tl_optimizer_config_t;

typedef struct {
    int max_epochs;
    int max_steps;
    int batch_size;
    int micro_batch_size;
    int grad_accum_steps;
    int num_workers;
    int seed;
    int log_every_n_steps;
    int eval_every_n_steps;
    int save_every_n_steps;
    float max_grad_norm;
    bool  use_grad_scaler;
    bool  use_grad_checkpointing;
    bool  use_amp;
    bool  use_sync_bn;
    bool  find_unused_params;
    bool  detect_anomaly;
    bool  profile;
    int   profile_start_step;
    int   profile_end_step;
    int   profile_num_batches;
    bool  deterministic;
    int   accumulation_steps;
} tl_train_config_t;

typedef struct {
    float* weights;
    float* biases;
    size_t num_weights;
    size_t num_biases;
} tl_layer_t;

typedef struct {
    tl_layer_t layers[TL_MAX_LAYERS];
    int num_layers;
    int current_epoch;
    int global_step;
    float current_lr;
    float best_val_loss;
    float best_val_metric;
} tl_model_t;

typedef struct {
    float value;
    float sum;
    float count;
    float running_avg;
} tl_metric_tracker_t;

typedef struct {
    tl_metric_tracker_t train_loss;
    tl_metric_tracker_t val_loss;
    tl_metric_tracker_t accuracy;
    tl_metric_tracker_t precision;
    tl_metric_tracker_t recall;
    tl_metric_tracker_t f1;
    float grad_norm;
    float forward_time_ms;
    float backward_time_ms;
    float step_time_ms;
    float data_time_ms;
} tl_metrics_t;

typedef struct {
    float* data;
    float* labels;
    size_t num_samples;
    size_t feature_dim;
    size_t num_classes;
    bool   is_val;
} tl_dataset_t;

typedef struct {
    tl_train_config_t train_cfg;
    tl_optimizer_config_t optim_cfg;
    tl_scheduler_config_t sched_cfg;
    tl_model_t model;
    tl_metrics_t metrics;
    tl_dataset_t train_set;
    tl_dataset_t val_set;
    float* grad_accum_buffer;
    int    accum_counter;
    bool   is_training;
    bool   early_stop_triggered;
    FILE*  log_file;
    uint64_t start_time;
    char   run_name[TL_MAX_NAME];
} tl_context_t;

void tl_init(tl_context_t* ctx, const tl_train_config_t* train_cfg,
             const tl_optimizer_config_t* optim_cfg,
             const tl_scheduler_config_t* sched_cfg);
void tl_free(tl_context_t* ctx);

void tl_model_add_layer(tl_model_t* model, int in_features, int out_features,
                         bool has_bias);
void tl_model_free(tl_model_t* model);

void tl_register_dataset(tl_context_t* ctx, float* data, float* labels,
                          size_t n, size_t dim, size_t classes, bool is_val);

void  tl_forward(tl_context_t* ctx, const float* input, float* output);
float tl_compute_loss(tl_context_t* ctx, const float* output,
                       const float* target, size_t batch_size);
void  tl_backward(tl_context_t* ctx, const float* input,
                   const float* output, const float* target,
                   size_t batch_size);
void  tl_optimizer_step(tl_context_t* ctx);
void  tl_scheduler_step(tl_context_t* ctx);
void  tl_zero_grad(tl_context_t* ctx);

void tl_train_epoch(tl_context_t* ctx);
void tl_train_step(tl_context_t* ctx, const float* batch, const float* labels,
                    size_t batch_size);
void tl_train_run(tl_context_t* ctx);

void tl_validate(tl_context_t* ctx);
void tl_eval_mode(tl_context_t* ctx, bool enable);
void tl_no_grad_scope(tl_context_t* ctx, bool enable);

float tl_compute_accuracy(const float* output, const float* target,
                           size_t batch_size, size_t num_classes);
float tl_compute_precision(const float* output, const float* target,
                            size_t batch_size, size_t num_classes);
float tl_compute_recall(const float* output, const float* target,
                         size_t batch_size, size_t num_classes);
float tl_compute_f1(float precision, float recall);

void tl_gradient_accumulation_step(tl_context_t* ctx, const float* batch,
                                    const float* labels, size_t batch_size);
void tl_accumulate_gradients(tl_context_t* ctx, const float* grads,
                              size_t numel);
void tl_reset_accumulated_gradients(tl_context_t* ctx);

void tl_gradient_checkpoint_recompute(tl_context_t* ctx, int layer_idx,
                                       const float* input, float* output);
void tl_gradient_checkpoint_save(tl_context_t* ctx, int layer_idx,
                                  const float* activation);
void tl_gradient_checkpoint_free(tl_context_t* ctx, int layer_idx);

void tl_clip_grad_norm(tl_context_t* ctx, float max_norm);
float tl_compute_grad_norm(tl_context_t* ctx);

void tl_log_metrics(tl_context_t* ctx, int step);
void tl_log_to_file(tl_context_t* ctx, const char* msg);
void tl_set_log_file(tl_context_t* ctx, const char* path);

typedef struct {
    float forward_ms;
    float backward_ms;
    float optim_ms;
    float data_ms;
    float total_ms;
    int   step;
} tl_profile_entry_t;

void tl_profile_start(tl_context_t* ctx);
void tl_profile_stop(tl_context_t* ctx);
void tl_profile_report(tl_context_t* ctx, tl_profile_entry_t* entries,
                        int max_entries);
void tl_profile_summary(tl_context_t* ctx);

void tl_early_stopping_update(tl_context_t* ctx, float val_loss);
bool tl_early_stopping_triggered(tl_context_t* ctx);

void tl_log_curve_loss(tl_context_t* ctx, const char* filename);
void tl_log_curve_accuracy(tl_context_t* ctx, const char* filename);

float tl_lr_get(tl_context_t* ctx);
void  tl_lr_set(tl_context_t* ctx, float lr);
float tl_lr_cosine(int step, int total_steps, float base_lr, float min_lr);
float tl_lr_warmup_cosine(int step, int warmup, int total,
                           float base_lr, float min_lr);

void tl_set_seed(int seed);
void tl_grad_check(tl_context_t* ctx, float eps, float threshold);

#ifdef __cplusplus
}
#endif

#endif
