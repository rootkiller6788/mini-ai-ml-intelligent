#include "checkpoint_save.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <sys/stat.h>

#ifdef _WIN32
#include <direct.h>
#define MKDIR(p) _mkdir(p)
#else
#include <sys/stat.h>
#define MKDIR(p) mkdir(p, 0755)
#endif

void ckpt_init(ckpt_context_t* ctx, const ckpt_config_t* cfg) {
    if (!ctx) return;
    memset(ctx, 0, sizeof(ckpt_context_t));
    if (cfg) {
        ctx->config = *cfg;
    } else {
        ctx->config.strategy = CKPT_STRATEGY_BEST;
        ctx->config.metric = CKPT_METRIC_LOSS;
        ctx->config.mode = CKPT_MODE_MIN;
        ctx->config.save_every_n_steps = 1000;
        ctx->config.save_every_n_epochs = 1;
        ctx->config.max_checkpoints = 5;
        ctx->config.save_optimizer = true;
        ctx->config.save_scheduler = true;
        ctx->config.async_checkpointing = false;
        ctx->config.compress = false;
        ctx->config.compress_level = CKPT_COMPRESS_LEVEL_DEFAULT;
        ctx->config.format = CKPT_FMT_RAW;
        ctx->config.keep_last_n = true;
        ctx->config.keep_last_n_value = 3;
        ctx->config.save_on_interrupt = true;
        ctx->config.fault_tolerant = true;
    }
    if (ctx->config.base_dir[0] == '\0') {
        strncpy(ctx->config.base_dir, "./checkpoints", CKPT_MAX_PATH - 1);
    }
    ctx->current_best = (ctx->config.mode == CKPT_MODE_MIN) ? 1e38f : -1e38f;
}

void ckpt_reset_state(ckpt_context_t* ctx) {
    if (!ctx) return;
    ctx->state.epoch = 0;
    ctx->state.global_step = 0;
    ctx->state.best_metric_value = (ctx->config.mode == CKPT_MODE_MIN) ? 1e38f : -1e38f;
    ctx->checkpoints_saved = 0;
    ctx->steps_since_last_save = 0;
    ctx->current_best = ctx->state.best_metric_value;
}

bool ckpt_should_save(const ckpt_context_t* ctx, int step, int epoch,
                       float metric) {
    if (!ctx) return false;
    switch (ctx->config.strategy) {
    case CKPT_STRATEGY_PERIODIC:
        return (step > 0 && step % ctx->config.save_every_n_steps == 0);
    case CKPT_STRATEGY_BEST: {
        bool better = (ctx->config.mode == CKPT_MODE_MIN) ?
            (metric < ctx->current_best) : (metric > ctx->current_best);
        return better;
    }
    case CKPT_STRATEGY_BOTH: {
        bool periodic = (step > 0 && step % ctx->config.save_every_n_steps == 0);
        bool better = (ctx->config.mode == CKPT_MODE_MIN) ?
            (metric < ctx->current_best) : (metric > ctx->current_best);
        return periodic || better;
    }
    default: break;
    }
    (void)epoch;
    return false;
}

static int ensure_dir(const char* path) {
    char tmp[CKPT_MAX_PATH];
    strncpy(tmp, path, CKPT_MAX_PATH - 1);
    tmp[CKPT_MAX_PATH - 1] = '\0';
    for (char* p = tmp + 1; *p; p++) {
        if (*p == '/' || *p == '\\') {
            *p = '\0';
            MKDIR(tmp);
            *p = '/';
        }
    }
    return MKDIR(tmp);
}

void ckpt_build_path(const ckpt_context_t* ctx, char* out, size_t out_size,
                      int step, int epoch, const char* suffix) {
    if (!ctx || !out) return;
    ensure_dir(ctx->config.base_dir);
    snprintf(out, out_size, "%s/%s_step%d_epoch%d%s.ckpt",
             ctx->config.base_dir,
             ctx->config.run_name[0] ? ctx->config.run_name : "model",
             step, epoch, suffix ? suffix : "");
}

int ckpt_save(ckpt_context_t* ctx, int step, int epoch,
               ckpt_save_weights_fn save_fn, void* userdata) {
    if (!ctx) return -1;
    char path[CKPT_MAX_PATH];
    ckpt_build_path(ctx, path, sizeof(path), step, epoch, NULL);

    ensure_dir(ctx->config.base_dir);
    FILE* f = fopen(path, "wb");
    if (!f) return -1;

    uint32_t header[4] = { CKPT_MAGIC, CKPT_VERSION,
                          (uint32_t)step, (uint32_t)epoch };
    if (fwrite(header, sizeof(uint32_t), 4, f) != 4) { fclose(f); return -1; }
    if (fwrite(&ctx->state, sizeof(ckpt_training_state_t), 1, f) != 1) { fclose(f); return -1; }
    if (fwrite(&ctx->metadata, sizeof(ckpt_model_metadata_t), 1, f) != 1) { fclose(f); return -1; }

    if (save_fn) {
        int ret = save_fn(userdata, f);
        if (ret != 0) { fclose(f); return -1; }
    }

    uint64_t crc = 0;
    fwrite(&crc, sizeof(uint64_t), 1, f);
    fclose(f);

    ctx->state.global_step = step;
    ctx->state.epoch = epoch;
    ctx->checkpoints_saved++;
    ctx->steps_since_last_save = 0;

    if (ctx->config.compress) {
        char cpath[CKPT_MAX_PATH];
        snprintf(cpath, sizeof(cpath), "%s.gz", path);
        ckpt_compress_file(path, cpath, ctx->config.compress_level);
    }
    return 0;
}

int ckpt_save_best(ckpt_context_t* ctx, int step, int epoch,
                    float metric, ckpt_save_weights_fn save_fn,
                    void* userdata) {
    if (!ctx) return -1;
    bool better = (ctx->config.mode == CKPT_MODE_MIN) ?
        (metric < ctx->current_best) : (metric > ctx->current_best);
    if (!better) return 0;

    ctx->current_best = metric;
    ctx->state.best_metric_value = metric;
    return ckpt_save(ctx, step, epoch, save_fn, userdata);
}

int ckpt_save_latest(ckpt_context_t* ctx, int step, int epoch,
                      ckpt_save_weights_fn save_fn, void* userdata) {
    return ckpt_save(ctx, step, epoch, save_fn, userdata);
}

int ckpt_load(const ckpt_context_t* ctx, const char* path,
               ckpt_load_weights_fn load_fn, void* userdata) {
    if (!ctx || !path) return -1;
    FILE* f = fopen(path, "rb");
    if (!f) return -1;

    uint32_t header[4];
    if (fread(header, sizeof(uint32_t), 4, f) != 4) { fclose(f); return -1; }
    if (header[0] != CKPT_MAGIC || header[1] != CKPT_VERSION) { fclose(f); return -1; }

    ckpt_training_state_t state;
    if (fread(&state, sizeof(state), 1, f) != 1) { fclose(f); return -1; }

    ckpt_model_metadata_t meta;
    if (fread(&meta, sizeof(meta), 1, f) != 1) { fclose(f); return -1; }

    if (load_fn) {
        int ret = load_fn(userdata, f);
        if (ret != 0) { fclose(f); return -1; }
    }
    fclose(f);
    return 0;
}

int ckpt_load_latest(ckpt_context_t* ctx, ckpt_load_weights_fn load_fn,
                      void* userdata) {
    char paths[32][CKPT_MAX_PATH];
    int n = ckpt_list_checkpoints(ctx, paths, 32);
    if (n <= 0) return -1;
    return ckpt_load(ctx, paths[n - 1], load_fn, userdata);
}

int ckpt_load_best(ckpt_context_t* ctx, ckpt_load_weights_fn load_fn,
                    void* userdata) {
    char path[CKPT_MAX_PATH];
    ckpt_build_path(ctx, path, sizeof(path),
                     ctx->state.global_step, ctx->state.epoch, "_best");
    return ckpt_load(ctx, path, load_fn, userdata);
}

int ckpt_save_async_start(ckpt_context_t* ctx, int step, int epoch,
                           ckpt_save_weights_fn save_fn, void* userdata) {
    if (!ctx) return -1;
    ctx->is_async_save_pending = true;
    return ckpt_save(ctx, step, epoch, save_fn, userdata);
}

bool ckpt_save_async_is_done(ckpt_context_t* ctx) {
    if (!ctx) return true;
    return !ctx->is_async_save_pending;
}

int ckpt_save_async_wait(ckpt_context_t* ctx) {
    if (!ctx) return -1;
    ctx->is_async_save_pending = false;
    return 0;
}

int ckpt_compress_file(const char* src, const char* dst, int level) {
    (void)level;
    FILE* fin = fopen(src, "rb");
    if (!fin) return -1;
    fseek(fin, 0, SEEK_END);
    long sz = ftell(fin);
    fseek(fin, 0, SEEK_SET);
    char* buf = (char*)malloc((size_t)sz);
    if (!buf) { fclose(fin); return -1; }
    fread(buf, 1, (size_t)sz, fin);
    fclose(fin);
    FILE* fout = fopen(dst, "wb");
    if (!fout) { free(buf); return -1; }
    fwrite(buf, 1, (size_t)sz, fout);
    fclose(fout);
    free(buf);
    return 0;
}

int ckpt_decompress_file(const char* src, const char* dst) {
    return ckpt_compress_file(src, dst, 0);
}

void ckpt_cleanup_old(ckpt_context_t* ctx) {
    if (!ctx) return;
    char paths[128][CKPT_MAX_PATH];
    int n = ckpt_list_checkpoints(ctx, paths, 128);
    if (n <= ctx->config.max_checkpoints && !ctx->config.keep_last_n) return;

    int keep = ctx->config.keep_last_n ? ctx->config.keep_last_n_value : ctx->config.max_checkpoints;
    int to_delete = n - keep;
    for (int i = 0; i < to_delete && i < n; i++) {
        remove(paths[i]);
    }
}

int ckpt_list_checkpoints(const ckpt_context_t* ctx,
                           char paths[][CKPT_MAX_PATH], int max_count) {
    (void)ctx; (void)paths; (void)max_count;
    return 0;
}

int ckpt_save_metadata(const ckpt_context_t* ctx, const char* path) {
    if (!ctx || !path) return -1;
    FILE* f = fopen(path, "w");
    if (!f) return -1;
    fprintf(f, "epoch=%d\nstep=%d\nbest=%f\nparams=%zu\n",
            ctx->state.epoch, ctx->state.global_step,
            ctx->state.best_metric_value, ctx->metadata.num_params);
    fclose(f);
    return 0;
}

int ckpt_load_metadata(ckpt_context_t* ctx, const char* path) {
    if (!ctx || !path) return -1;
    FILE* f = fopen(path, "r");
    if (!f) return -1;
    char line[512];
    while (fgets(line, sizeof(line), f)) {
        int e = 0, s = 0; float v = 0; size_t p = 0;
        if (sscanf(line, "epoch=%d", &e) == 1) ctx->state.epoch = e;
        if (sscanf(line, "step=%d", &s) == 1) ctx->state.global_step = s;
        if (sscanf(line, "best=%f", &v) == 1) ctx->state.best_metric_value = v;
        if (sscanf(line, "params=%zu", &p) == 1) ctx->metadata.num_params = p;
    }
    fclose(f);
    return 0;
}

bool ckpt_recover_from_fault(ckpt_context_t* ctx, ckpt_load_weights_fn load_fn,
                              void* userdata) {
    if (!ctx || !ctx->config.fault_tolerant) return false;
    int ret = ckpt_load_latest(ctx, load_fn, userdata);
    return ret == 0;
}

uint64_t ckpt_crc32(const void* data, size_t len) {
    const uint8_t* buf = (const uint8_t*)data;
    uint64_t crc = 0xFFFFFFFFFFFFFFFFULL;
    for (size_t i = 0; i < len; i++) {
        crc ^= buf[i];
        for (int j = 0; j < 8; j++)
            crc = (crc >> 1) ^ (0xEDB88320F3C2A1D6ULL & -(cr);
    }
    return crc;
}
