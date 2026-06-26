#ifndef CHECKPOINT_SAVE_H
#define CHECKPOINT_SAVE_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CKPT_MAGIC 0x434B5054
#define CKPT_VERSION 3
#define CKPT_MAX_PATH 1024
#define CKPT_MAX_NAME 256
#define CKPT_COMPRESS_LEVEL_DEFAULT 6

typedef enum {
    CKPT_STRATEGY_BEST = 0,
    CKPT_STRATEGY_PERIODIC,
    CKPT_STRATEGY_BOTH,
    CKPT_STRATEGY_MANUAL,
} ckpt_strategy_t;

typedef enum {
    CKPT_METRIC_LOSS = 0,
    CKPT_METRIC_ACCURACY,
    CKPT_METRIC_F1,
    CKPT_METRIC_MAP,
    CKPT_METRIC_BLEU,
    CKPT_METRIC_CUSTOM,
} ckpt_metric_t;

typedef enum {
    CKPT_MODE_MIN = 0,
    CKPT_MODE_MAX,
} ckpt_mode_t;

typedef enum {
    CKPT_FMT_RAW = 0,
    CKPT_FMT_COMPRESSED,
    CKPT_FMT_SAFETENSORS,
} ckpt_format_t;

typedef struct {
    int  epoch;
    int  global_step;
    float best_metric_value;
    float current_metric_value;
    float learning_rate;
    uint64_t samples_seen;
    uint64_t tokens_seen;
} ckpt_training_state_t;

typedef struct {
    size_t num_params;
    size_t num_optimizer_states;
    size_t num_scheduler_states;
    size_t total_bytes;
} ckpt_model_metadata_t;

typedef struct {
    char     base_dir[CKPT_MAX_PATH];
    char     run_name[CKPT_MAX_NAME];
    ckpt_strategy_t strategy;
    ckpt_metric_t   metric;
    ckpt_mode_t     mode;
    int      save_every_n_steps;
    int      save_every_n_epochs;
    int      max_checkpoints;
    bool     save_optimizer;
    bool     save_scheduler;
    bool     async_checkpointing;
    bool     compress;
    int      compress_level;
    ckpt_format_t format;
    bool     keep_last_n;
    int      keep_last_n_value;
    bool     save_on_interrupt;
    bool     fault_tolerant;
} ckpt_config_t;

typedef struct {
    ckpt_config_t config;
    ckpt_training_state_t state;
    ckpt_model_metadata_t metadata;
    float current_best;
    int   checkpoints_saved;
    int   steps_since_last_save;
    bool  is_async_save_pending;
    FILE* async_stream;
} ckpt_context_t;

typedef int (*ckpt_save_weights_fn)(void* userdata, FILE* f);
typedef int (*ckpt_load_weights_fn)(void* userdata, FILE* f);

void ckpt_init(ckpt_context_t* ctx, const ckpt_config_t* cfg);
void ckpt_reset_state(ckpt_context_t* ctx);

bool ckpt_should_save(const ckpt_context_t* ctx, int step, int epoch,
                       float metric);

int  ckpt_save(ckpt_context_t* ctx, int step, int epoch,
               ckpt_save_weights_fn save_fn, void* userdata);
int  ckpt_save_best(ckpt_context_t* ctx, int step, int epoch,
                     float metric, ckpt_save_weights_fn save_fn,
                     void* userdata);
int  ckpt_save_latest(ckpt_context_t* ctx, int step, int epoch,
                       ckpt_save_weights_fn save_fn, void* userdata);

int  ckpt_load(const ckpt_context_t* ctx, const char* path,
               ckpt_load_weights_fn load_fn, void* userdata);
int  ckpt_load_latest(ckpt_context_t* ctx, ckpt_load_weights_fn load_fn,
                       void* userdata);
int  ckpt_load_best(ckpt_context_t* ctx, ckpt_load_weights_fn load_fn,
                     void* userdata);

int  ckpt_save_async_start(ckpt_context_t* ctx, int step, int epoch,
                            ckpt_save_weights_fn save_fn, void* userdata);
bool ckpt_save_async_is_done(ckpt_context_t* ctx);
int  ckpt_save_async_wait(ckpt_context_t* ctx);

void ckpt_build_path(const ckpt_context_t* ctx, char* out, size_t out_size,
                      int step, int epoch, const char* suffix);

int  ckpt_compress_file(const char* src, const char* dst, int level);
int  ckpt_decompress_file(const char* src, const char* dst);

void ckpt_cleanup_old(ckpt_context_t* ctx);
int  ckpt_list_checkpoints(const ckpt_context_t* ctx,
                            char paths[][CKPT_MAX_PATH], int max_count);

int  ckpt_save_metadata(const ckpt_context_t* ctx, const char* path);
int  ckpt_load_metadata(ckpt_context_t* ctx, const char* path);

bool ckpt_recover_from_fault(ckpt_context_t* ctx, ckpt_load_weights_fn load_fn,
                              void* userdata);
uint64_t ckpt_crc32(const void* data, size_t len);

#ifdef __cplusplus
}
#endif

#endif
