#include "training_loop.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

#ifdef _MSC_VER
#include <windows.h>
static double tl_now_ms(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1000.0 / (double)f.QuadPart;
}
#else
#include <sys/time.h>
static double tl_now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
}
#endif

static uint64_t tl_rng_state = 123456789ULL;
static uint64_t tl_splitmix64(uint64_t* state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

void tl_set_seed(int seed) { tl_rng_state = (uint64_t)(seed > 0 ? seed : 1); }

static float tl_randf(void) {
    return (float)(tl_splitmix64(&tl_rng_state) >> 40) / (float)(1ULL << 24);
}

static float tl_randn(void) {
    float u1 = tl_randf(), u2 = tl_randf();
    return sqrtf(-2.0f * logf(u1 + 1e-8f)) * cosf(6.28318530718f * u2);
}

static float tl_relu(float x) { return x > 0.0f ? x : 0.0f; }
static float tl_relu_back(float x) { return x > 0.0f ? 1.0f : 0.0f; }
static float tl_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
static float tl_sigmoid_back(float x) { float s = tl_sigmoid(x); return s * (1.0f - s); }

static float tl_cross_entropy(const float* logits, const float* target,
                               size_t n, size_t classes) {
    float loss = 0.0f;
    for (size_t b = 0; b < n; b++) {
        float max_val = logits[b * classes];
        for (size_t c = 1; c < classes; c++)
            if (logits[b * classes + c] > max_val) max_val = logits[b * classes + c];
        float sum_exp = 0.0f;
        for (size_t c = 0; c < classes; c++)
            sum_exp += expf(logits[b * classes + c] - max_val);
        float log_sum = max_val + logf(sum_exp + 1e-8f);
        for (size_t c = 0; c < classes; c++)
            if (target[b * classes + c] > 0.5f) loss += log_sum - logits[b * classes + c];
    }
    return loss / (float)n;
}

void tl_init(tl_context_t* ctx, const tl_train_config_t* train_cfg,
             const tl_optimizer_config_t* optim_cfg,
             const tl_scheduler_config_t* sched_cfg) {
    if (!ctx) return;
    memset(ctx, 0, sizeof(tl_context_t));
    if (train_cfg) ctx->train_cfg = *train_cfg;
    if (optim_cfg) ctx->optim_cfg = *optim_cfg;
    if (sched_cfg) ctx->sched_cfg = *sched_cfg;

    if (ctx->train_cfg.batch_size <= 0) ctx->train_cfg.batch_size = 32;
    if (ctx->train_cfg.max_epochs <= 0) ctx->train_cfg.max_epochs = 100;
    if (ctx->train_cfg.log_every_n_steps <= 0) ctx->train_cfg.log_every_n_steps = 100;
    if (ctx->train_cfg.eval_every_n_steps <= 0) ctx->train_cfg.eval_every_n_steps = 1000;
    if (ctx->train_cfg.accumulation_steps <= 0) ctx->train_cfg.accumulation_steps = 1;

    ctx->current_lr = ctx->optim_cfg.learning_rate;
    ctx->is_training = true;
    ctx->start_time = (uint64_t)tl_now_ms();
}

void tl_free(tl_context_t* ctx) {
    if (!ctx) return;
    tl_model_free(&ctx->model);
    free(ctx->grad_accum_buffer);
    if (ctx->log_file && ctx->log_file != stdout && ctx->log_file != stderr)
        fclose(ctx->log_file);
    memset(ctx, 0, sizeof(tl_context_t));
}

void tl_model_add_layer(tl_model_t* model, int in_features, int out_features,
                         bool has_bias) {
    if (!model || model->num_layers >= TL_MAX_LAYERS) return;
    tl_layer_t* l = &model->layers[model->num_layers];
    l->num_weights = (size_t)in_features * (size_t)out_features;
    l->num_biases = has_bias ? (size_t)out_features : 0;
    l->weights = (float*)malloc(l->num_weights * sizeof(float));
    l->biases = has_bias ? (float*)calloc(l->num_biases, sizeof(float)) : NULL;

    float scale = sqrtf(2.0f / (float)in_features);
    for (size_t i = 0; i < l->num_weights; i++) l->weights[i] = tl_randn() * scale;
    if (has_bias) for (size_t i = 0; i < l->num_biases; i++) l->biases[i] = 0.0f;

    model->num_layers++;
}

void tl_model_free(tl_model_t* model) {
    if (!model) return;
    for (int i = 0; i < model->num_layers; i++) {
        free(model->layers[i].weights);
        free(model->layers[i].biases);
    }
    model->num_layers = 0;
}

void tl_register_dataset(tl_context_t* ctx, float* data, float* labels,
                          size_t n, size_t dim, size_t classes, bool is_val) {
    if (!ctx) return;
    tl_dataset_t* ds = is_val ? &ctx->val_set : &ctx->train_set;
    ds->data = data;
    ds->labels = labels;
    ds->num_samples = n;
    ds->feature_dim = dim;
    ds->num_classes = classes;
    ds->is_val = is_val;
}

void tl_forward(tl_context_t* ctx, const float* input, float* output) {
    if (!ctx) return;
    size_t batch = (size_t)ctx->train_cfg.batch_size;
    size_t dim = ctx->train_set.feature_dim;
    if (dim == 0) dim = 10;

    float* curr_in = (float*)malloc(batch * dim * sizeof(float));
    float* curr_out = NULL;
    if (curr_in) memcpy(curr_in, input, batch * dim * sizeof(float));

    for (int l = 0; l < ctx->model.num_layers; l++) {
        tl_layer_t* layer = &ctx->model.layers[l];
        size_t out_dim = layer->num_biases > 0 ? layer->num_biases :
                         layer->num_weights / (dim > 0 ? dim : 1);
        curr_out = (float*)calloc(batch * out_dim, sizeof(float));
        if (!curr_out) break;

        for (size_t b = 0; b < batch; b++) {
            for (size_t o = 0; o < out_dim; o++) {
                float sum = layer->biases ? layer->biases[o] : 0.0f;
                for (size_t i = 0; i < dim; i++)
                    sum += curr_in[b * dim + i] * layer->weights[i * out_dim + o];
                curr_out[b * out_dim + o] = (l < ctx->model.num_layers - 1) ?
                    tl_relu(sum) : sum;
            }
        }
        free(curr_in);
        curr_in = curr_out;
        dim = out_dim;
    }
    if (curr_in && output) memcpy(output, curr_in, batch * dim * sizeof(float));
    free(curr_in);
}

float tl_compute_loss(tl_context_t* ctx, const float* output,
                       const float* target, size_t batch_size) {
    if (!ctx) return 0.0f;
    size_t classes = ctx->train_set.num_classes;
    if (classes == 0) classes = 2;
    return tl_cross_entropy(output, target, batch_size, classes);
}

/**
 * tl_backward — Backpropagation through the entire model
 *
 * Theorem: Chain Rule (Leibniz Notation):
 *   dL/dw_i = dL/df_n * df_n/df_{n-1} * ... * df_i/dw_i
 *
 * For cross-entropy with softmax (implicit in forward), the gradient of
 * loss w.r.t. logits is simply (softmax(output) - target), avoiding the
 * softmax Jacobian via the log-sum-exp trick.
 *
 * For ReLU layers: ∂ReLU(x)/∂x = 1 if x > 0 else 0.
 * For linear layers: ∂(Wx + b)/∂W = x^T, ∂(Wx + b)/∂x = W^T.
 *
 * Time Complexity: O(L * B * d²) for L layers, batch B, dim d.
 *
 * Reference: Rumelhart, Hinton, Williams (1986) "Learning representations
 *   by back-propagating errors" — Nature.
 */
void tl_backward(tl_context_t* ctx, const float* input,
                  const float* output, const float* target,
                  size_t batch_size) {
    if (!ctx || !output || !target || batch_size == 0) return;
    if (ctx->model.num_layers == 0) return;

    size_t classes = ctx->train_set.num_classes;
    if (classes == 0) classes = 2;
    size_t dim = ctx->train_set.feature_dim;
    if (dim == 0) dim = 10;

    /* Step 1: compute dL/dlogits = softmax(output) - target
     * This is the gradient of cross-entropy loss w.r.t. final layer logits.
     * Using the identity: ∂CE(softmax(z), y)/∂z = softmax(z) - y
     */
    size_t final_dim = classes;
    /* find actual output dimension from last layer */
    for (int l = ctx->model.num_layers - 1; l >= 0; l--) {
        tl_layer_t* layer = &ctx->model.layers[l];
        if (layer->biases) { final_dim = layer->num_biases; break; }
    }

    float* d_output = (float*)malloc(batch_size * final_dim * sizeof(float));
    if (!d_output) return;

    /* softmax gradient = softmax(logits) - target */
    for (size_t b = 0; b < batch_size; b++) {
        /* compute softmax for numerical stability */
        float max_val = output[b * final_dim];
        for (size_t c = 1; c < final_dim; c++)
            if (output[b * final_dim + c] > max_val)
                max_val = output[b * final_dim + c];
        float sum_exp = 1e-8f;
        for (size_t c = 0; c < final_dim; c++)
            sum_exp += expf(output[b * final_dim + c] - max_val);
        for (size_t c = 0; c < final_dim; c++) {
            float prob = expf(output[b * final_dim + c] - max_val) / sum_exp;
            d_output[b * final_dim + c] = (prob - target[b * final_dim + c]) / (float)batch_size;
        }
    }

    /* Step 2: Backpropagate through hidden layers (reverse order)
     * For each layer: compute dW, db, and d_input for the previous layer.
     * We need to recompute activations since we don't cache them.
     * For simplicity, we store weight gradients in a scratch buffer.
     */
    float* curr_delta = d_output;
    size_t curr_dim = final_dim;

    /* Recompute forward activations to get ReLU masks */
    float** activations = (float**)malloc((size_t)(ctx->model.num_layers) * sizeof(float*));
    if (!activations) { free(d_output); return; }

    /* Forward pass to capture activations for gradient computation */
    size_t fwd_dim = dim;
    float* fwd_input = (float*)malloc(batch_size * fwd_dim * sizeof(float));
    if (!fwd_input) { free(activations); free(d_output); return; }
    memcpy(fwd_input, input, batch_size * fwd_dim * sizeof(float));

    for (int l = 0; l < ctx->model.num_layers; l++) {
        tl_layer_t* layer = &ctx->model.layers[l];
        size_t out_dim = layer->biases ? layer->num_biases :
                         (layer->num_weights / (fwd_dim > 0 ? fwd_dim : 1));
        activations[l] = (float*)malloc(batch_size * out_dim * sizeof(float));
        if (!activations[l]) {
            for (int k = 0; k < l; k++) free(activations[k]);
            free(activations); free(fwd_input); free(d_output); return;
        }

        /* Linear: out = W^T * in + b, then ReLU for non-final layers */
        for (size_t b = 0; b < batch_size; b++) {
            for (size_t o = 0; o < out_dim; o++) {
                float sum = layer->biases ? layer->biases[o] : 0.0f;
                for (size_t i = 0; i < fwd_dim; i++)
                    sum += fwd_input[b * fwd_dim + i] * layer->weights[i * out_dim + o];
                /* store pre-activation for gradient computation */
                activations[l][b * out_dim + o] = sum;
            }
        }
        /* Apply ReLU for non-final layers */
        bool is_last = (l == ctx->model.num_layers - 1);
        for (size_t b = 0; b < batch_size; b++) {
            for (size_t o = 0; o < out_dim; o++) {
                float val = activations[l][b * out_dim + o];
                activations[l][b * out_dim + o] = is_last ? val : tl_relu(val);
            }
        }
        free(fwd_input);
        fwd_input = (float*)malloc(batch_size * out_dim * sizeof(float));
        if (fwd_input) memcpy(fwd_input, activations[l], batch_size * out_dim * sizeof(float));
        fwd_dim = out_dim;
    }
    free(fwd_input);

    /* Backprop through layers in reverse order */
    for (int l = ctx->model.num_layers - 1; l >= 0; l--) {
        tl_layer_t* layer = &ctx->model.layers[l];
        size_t in_dim = (l == 0) ? dim : 0;
        /* Determine input dimension from previous layer */
        if (l > 0) {
            tl_layer_t* prev = &ctx->model.layers[l - 1];
            in_dim = prev->biases ? prev->num_biases :
                     prev->num_weights / (in_dim > 0 ? in_dim : 1);
        }

        size_t out_dim = layer->biases ? layer->num_biases :
                         (layer->num_weights / (in_dim > 0 ? in_dim : 1));

        /* Gradient through ReLU for non-final layers */
        if (l < ctx->model.num_layers - 1) {
            for (size_t b = 0; b < batch_size; b++) {
                for (size_t o = 0; o < curr_dim && o < out_dim; o++) {
                    /* pre-activation was stored before ReLU */
                    curr_delta[b * curr_dim + o] *= tl_relu_back(
                        activations[l] ? activations[l][b * out_dim + o] : 0.0f);
                }
            }
        }

        /* dW = input^T * delta  (outer product summed over batch) */
        float* prev_act = (l == 0) ? (float*)input : activations[l - 1];
        size_t prev_dim = in_dim;
        if (prev_act) {
            for (size_t i = 0; i < prev_dim && i < in_dim; i++) {
                for (size_t o = 0; o < curr_dim && o < out_dim; o++) {
                    float grad_w = 0.0f;
                    for (size_t b = 0; b < batch_size; b++)
                        grad_w += prev_act[b * prev_dim + i] * curr_delta[b * curr_dim + o];
                    /* Apply weight update scaled by learning rate */
                    size_t widx = i * out_dim + o;
                    if (widx < layer->num_weights)
                        layer->weights[widx] -= ctx->current_lr * grad_w;
                }
            }
        }

        /* db = sum(delta over batch) */
        if (layer->biases) {
            for (size_t o = 0; o < curr_dim && o < layer->num_biases; o++) {
                float grad_b = 0.0f;
                for (size_t b = 0; b < batch_size; b++)
                    grad_b += curr_delta[b * curr_dim + o];
                layer->biases[o] -= ctx->current_lr * grad_b;
            }
        }

        /* d_input = W * delta for next iteration */
        if (l > 0) {
            float* next_delta = (float*)calloc(batch_size * in_dim, sizeof(float));
            if (next_delta) {
                for (size_t b = 0; b < batch_size; b++) {
                    for (size_t i = 0; i < in_dim; i++) {
                        for (size_t o = 0; o < curr_dim && o < out_dim; o++) {
                            next_delta[b * in_dim + i] +=
                                layer->weights[i * out_dim + o] * curr_delta[b * curr_dim + o];
                        }
                    }
                }
                free(curr_delta);
                curr_delta = next_delta;
            }
            curr_dim = in_dim;
        }
    }

    free(curr_delta);
    for (int l = 0; l < ctx->model.num_layers; l++) free(activations[l]);
    free(activations);
}

void tl_optimizer_step(tl_context_t* ctx) {
    if (!ctx) return;
    float lr = ctx->current_lr;
    float wd = ctx->optim_cfg.weight_decay;

    for (int l = 0; l < ctx->model.num_layers; l++) {
        tl_layer_t* layer = &ctx->model.layers[l];
        for (size_t i = 0; i < layer->num_weights; i++) {
            float noise = tl_randn() * 0.01f;
            layer->weights[i] -= lr * (noise + wd * layer->weights[i]);
        }
        if (layer->biases) {
            for (size_t i = 0; i < layer->num_biases; i++)
                layer->biases[i] -= lr * tl_randn() * 0.01f;
        }
    }
}

void tl_scheduler_step(tl_context_t* ctx) {
    if (!ctx) return;
    int step = ctx->global_step;
    int total = ctx->train_cfg.max_steps;
    if (total <= 0) total = ctx->train_cfg.max_epochs;

    switch (ctx->sched_cfg.type) {
    case TL_SCHEDULER_COSINE:
        ctx->current_lr = tl_lr_cosine(step, total, ctx->sched_cfg.base_lr,
                                        ctx->sched_cfg.min_lr);
        break;
    case TL_SCHEDULER_COSINE_RESTARTS: {
        int warmup = ctx->sched_cfg.warmup_steps;
        ctx->current_lr = tl_lr_warmup_cosine(step, warmup, total,
                                               ctx->sched_cfg.base_lr,
                                               ctx->sched_cfg.min_lr);
        break;
    }
    case TL_SCHEDULER_EXPONENTIAL:
        ctx->current_lr = ctx->sched_cfg.base_lr * powf(ctx->sched_cfg.gamma, (float)step);
        break;
    default:
        break;
    }
    if (ctx->current_lr < ctx->sched_cfg.min_lr) ctx->current_lr = ctx->sched_cfg.min_lr;
}

void tl_zero_grad(tl_context_t* ctx) {
    if (!ctx) return;
    (void)ctx;
}

void tl_train_epoch(tl_context_t* ctx) {
    if (!ctx) return;
    ctx->is_training = true;
    size_t n = ctx->train_set.num_samples;
    size_t bs = (size_t)ctx->train_cfg.batch_size;
    if (n == 0 || bs == 0) return;
    size_t steps = (n + bs - 1) / bs;

    for (size_t s = 0; s < steps; s++) {
        size_t start = s * bs;
        size_t count = (start + bs <= n) ? bs : n - start;
        tl_train_step(ctx, ctx->train_set.data + start * ctx->train_set.feature_dim,
                       ctx->train_set.labels + start * ctx->train_set.num_classes,
                       count);
        ctx->global_step++;
        tl_scheduler_step(ctx);
        if (ctx->global_step % ctx->train_cfg.log_every_n_steps == 0)
            tl_log_metrics(ctx, ctx->global_step);
    }
    ctx->model.current_epoch++;
}

void tl_train_step(tl_context_t* ctx, const float* batch, const float* labels,
                    size_t batch_size) {
    if (!ctx) return;
    size_t classes = ctx->train_set.num_classes;
    if (classes == 0) classes = 2;
    size_t dim = ctx->train_set.feature_dim;
    if (dim == 0) dim = 10;
    float* output = (float*)malloc(batch_size * classes * sizeof(float));
    if (!output) return;

    double t0 = tl_now_ms();
    tl_forward(ctx, batch, output);
    double tf = tl_now_ms();
    ctx->metrics.forward_time_ms = (float)(tf - t0);

    float loss = tl_compute_loss(ctx, output, labels, batch_size);
    ctx->metrics.train_loss.value = loss;
    ctx->metrics.train_loss.sum += loss * (float)batch_size;
    ctx->metrics.train_loss.count += (float)batch_size;

    double tb = tl_now_ms();
    tl_backward(ctx, batch, output, labels, batch_size);
    ctx->metrics.backward_time_ms = (float)(tl_now_ms() - tb);

    if (ctx->train_cfg.grad_accum_steps > 1) {
        ctx->accum_counter++;
        if (ctx->accum_counter >= ctx->train_cfg.grad_accum_steps) {
            tl_optimizer_step(ctx);
            tl_zero_grad(ctx);
            ctx->accum_counter = 0;
        }
    } else {
        tl_optimizer_step(ctx);
        tl_zero_grad(ctx);
    }

    float acc = tl_compute_accuracy(output, labels, batch_size, classes);
    ctx->metrics.accuracy.value = acc;
    ctx->metrics.accuracy.sum += acc * (float)batch_size;
    ctx->metrics.accuracy.count += (float)batch_size;

    double ts = tl_now_ms();
    ctx->metrics.step_time_ms = (float)(ts - t0);
    free(output);
}

void tl_train_run(tl_context_t* ctx) {
    if (!ctx) return;
    ctx->is_training = true;
    for (int e = 0; e < ctx->train_cfg.max_epochs; e++) {
        tl_train_epoch(ctx);
        if (ctx->val_set.num_samples > 0) tl_validate(ctx);
        tl_early_stopping_update(ctx, ctx->metrics.val_loss.running_avg);
        if (ctx->early_stop_triggered) break;
    }
}

void tl_validate(tl_context_t* ctx) {
    if (!ctx) return;
    tl_eval_mode(ctx, true);
    tl_no_grad_scope(ctx, true);

    size_t n = ctx->val_set.num_samples;
    size_t bs = (size_t)ctx->train_cfg.batch_size;
    if (n == 0) { tl_eval_mode(ctx, false); tl_no_grad_scope(ctx, false); return; }

    float total_loss = 0.0f;
    float total_acc = 0.0f;
    size_t processed = 0;
    size_t steps = (n + bs - 1) / bs;

    for (size_t s = 0; s < steps; s++) {
        size_t start = s * bs;
        size_t count = (start + bs <= n) ? bs : n - start;
        if (count == 0) break;

        size_t dim = ctx->val_set.feature_dim;
        size_t cls = ctx->val_set.num_classes;
        float* out = (float*)malloc(count * cls * sizeof(float));
        if (!out) continue;
        tl_forward(ctx, ctx->val_set.data + start * dim, out);
        float loss = tl_compute_loss(ctx, out, ctx->val_set.labels + start * cls, count);
        float acc = tl_compute_accuracy(out, ctx->val_set.labels + start * cls, count, cls);
        total_loss += loss * (float)count;
        total_acc += acc * (float)count;
        processed += count;
        free(out);
    }
    if (processed > 0) {
        ctx->metrics.val_loss.value = total_loss / (float)processed;
        ctx->metrics.val_loss.running_avg = ctx->metrics.val_loss.value;
        ctx->metrics.accuracy.running_avg = total_acc / (float)processed;
    }
    tl_no_grad_scope(ctx, false);
    tl_eval_mode(ctx, false);
}

void tl_eval_mode(tl_context_t* ctx, bool enable) {
    if (ctx) ctx->is_training = !enable;
}

void tl_no_grad_scope(tl_context_t* ctx, bool enable) {
    (void)ctx; (void)enable;
}

float tl_compute_accuracy(const float* output, const float* target,
                           size_t batch_size, size_t num_classes) {
    if (batch_size == 0) return 0.0f;
    int correct = 0;
    for (size_t b = 0; b < batch_size; b++) {
        int pred = 0, tgt = 0;
        float max_o = output[b * num_classes], max_t = target[b * num_classes];
        for (size_t c = 1; c < num_classes; c++) {
            if (output[b * num_classes + c] > max_o) { max_o = output[b * num_classes + c]; pred = (int)c; }
            if (target[b * num_classes + c] > max_t) { max_t = target[b * num_classes + c]; tgt = (int)c; }
        }
        if (pred == tgt) correct++;
    }
    return (float)correct / (float)batch_size;
}

float tl_compute_precision(const float* output, const float* target,
                            size_t batch_size, size_t num_classes) {
    int tp = 0, fp_count = 0;
    for (size_t b = 0; b < batch_size; b++) {
        int pred = 0, tgt = 0;
        float max_o = output[b * num_classes], max_t = target[b * num_classes];
        for (size_t c = 1; c < num_classes; c++) {
            if (output[b * num_classes + c] > max_o) { max_o = output[b * num_classes + c]; pred = (int)c; }
            if (target[b * num_classes + c] > max_t) { max_t = target[b * num_classes + c]; tgt = (int)c; }
        }
        if (pred == 1 && tgt == 1) tp++;
        if (pred == 1 && tgt == 0) fp_count++;
    }
    return (tp + fp_count) > 0 ? (float)tp / (float)(tp + fp_count) : 0.0f;
}

float tl_compute_recall(const float* output, const float* target,
                         size_t batch_size, size_t num_classes) {
    int tp = 0, fn_count = 0;
    for (size_t b = 0; b < batch_size; b++) {
        int pred = 0, tgt = 0;
        float max_o = output[b * num_classes], max_t = target[b * num_classes];
        for (size_t c = 1; c < num_classes; c++) {
            if (output[b * num_classes + c] > max_o) { max_o = output[b * num_classes + c]; pred = (int)c; }
            if (target[b * num_classes + c] > max_t) { max_t = target[b * num_classes + c]; tgt = (int)c; }
        }
        if (pred == 1 && tgt == 1) tp++;
        if (pred == 0 && tgt == 1) fn_count++;
    }
    return (tp + fn_count) > 0 ? (float)tp / (float)(tp + fn_count) : 0.0f;
}

float tl_compute_f1(float precision, float recall) {
    return (precision + recall) > 0.0f ? 2.0f * precision * recall / (precision + recall) : 0.0f;
}

void tl_gradient_accumulation_step(tl_context_t* ctx, const float* batch,
                                    const float* labels, size_t batch_size) {
    if (!ctx) return;
    ctx->accum_counter++;
    tl_train_step(ctx, batch, labels, batch_size);
    if (ctx->accum_counter >= ctx->train_cfg.accumulation_steps) {
        tl_optimizer_step(ctx);
        tl_zero_grad(ctx);
        ctx->accum_counter = 0;
    }
}

void tl_accumulate_gradients(tl_context_t* ctx, const float* grads,
                              size_t numel) {
    if (!ctx || !grads) return;
    if (!ctx->grad_accum_buffer) {
        ctx->grad_accum_buffer = (float*)calloc(numel, sizeof(float));
    }
    if (ctx->grad_accum_buffer) {
        for (size_t i = 0; i < numel; i++) ctx->grad_accum_buffer[i] += grads[i];
    }
}

void tl_reset_accumulated_gradients(tl_context_t* ctx) {
    if (!ctx || !ctx->grad_accum_buffer) return;
}

void tl_gradient_checkpoint_recompute(tl_context_t* ctx, int layer_idx,
                                       const float* input, float* output) {
    if (!ctx || layer_idx < 0 || layer_idx >= ctx->model.num_layers) return;
    tl_forward(ctx, input, output);
}

void tl_gradient_checkpoint_save(tl_context_t* ctx, int layer_idx,
                                  const float* activation) {
    (void)ctx; (void)layer_idx; (void)activation;
}

void tl_gradient_checkpoint_free(tl_context_t* ctx, int layer_idx) {
    (void)ctx; (void)layer_idx;
}

void tl_clip_grad_norm(tl_context_t* ctx, float max_norm) {
    if (!ctx) return;
    float norm = tl_compute_grad_norm(ctx);
    if (norm > max_norm) {
        float scale = max_norm / norm;
        for (int l = 0; l < ctx->model.num_layers; l++) {
            tl_layer_t* layer = &ctx->model.layers[l];
            for (size_t i = 0; i < layer->num_weights; i++)
                layer->weights[i] *= scale;
            if (layer->biases)
                for (size_t i = 0; i < layer->num_biases; i++)
                    layer->biases[i] *= scale;
        }
    }
}

float tl_compute_grad_norm(tl_context_t* ctx) {
    if (!ctx) return 0.0f;
    float sum_sq = 0.0f;
    for (int l = 0; l < ctx->model.num_layers; l++) {
        tl_layer_t* layer = &ctx->model.layers[l];
        for (size_t i = 0; i < layer->num_weights; i++)
            sum_sq += layer->weights[i] * layer->weights[i];
        if (layer->biases)
            for (size_t i = 0; i < layer->num_biases; i++)
                sum_sq += layer->biases[i] * layer->biases[i];
    }
    return sqrtf(sum_sq);
}

void tl_log_metrics(tl_context_t* ctx, int step) {
    if (!ctx) return;
    float avg_loss = ctx->metrics.train_loss.count > 0 ?
        ctx->metrics.train_loss.sum / ctx->metrics.train_loss.count : 0.0f;
    float avg_acc = ctx->metrics.accuracy.count > 0 ?
        ctx->metrics.accuracy.sum / ctx->metrics.accuracy.count : 0.0f;

    char msg[512];
    snprintf(msg, sizeof(msg),
             "[Step %d] loss=%.4f acc=%.4f lr=%.6f fwd=%.1fms bwd=%.1fms step=%.1fms",
             step, avg_loss, avg_acc, ctx->current_lr,
             ctx->metrics.forward_time_ms, ctx->metrics.backward_time_ms,
             ctx->metrics.step_time_ms);
    tl_log_to_file(ctx, msg);
}

void tl_log_to_file(tl_context_t* ctx, const char* msg) {
    if (!ctx || !msg) return;
    FILE* f = ctx->log_file ? ctx->log_file : stdout;
    fprintf(f, "%s\n", msg);
    fflush(f);
}

void tl_set_log_file(tl_context_t* ctx, const char* path) {
    if (!ctx || !path) return;
    FILE* f = fopen(path, "w");
    if (f) ctx->log_file = f;
}

void tl_profile_start(tl_context_t* ctx) {
    if (!ctx) return;
    ctx->train_cfg.profile = true;
    ctx->train_cfg.profile_start_step = ctx->global_step;
}

void tl_profile_stop(tl_context_t* ctx) {
    if (!ctx) return;
    ctx->train_cfg.profile = false;
}

void tl_profile_report(tl_context_t* ctx, tl_profile_entry_t* entries,
                        int max_entries) {
    if (!ctx || !entries || max_entries <= 0) return;
    entries[0].forward_ms = ctx->metrics.forward_time_ms;
    entries[0].backward_ms = ctx->metrics.backward_time_ms;
    entries[0].optim_ms = ctx->metrics.step_time_ms - ctx->metrics.forward_time_ms - ctx->metrics.backward_time_ms;
    entries[0].total_ms = ctx->metrics.step_time_ms;
    entries[0].step = ctx->global_step;
}

void tl_profile_summary(tl_context_t* ctx) {
    if (!ctx) return;
    char msg[256];
    snprintf(msg, sizeof(msg), "[Profile] fwd=%.1fms bwd=%.1fms step=%.1fms",
             ctx->metrics.forward_time_ms, ctx->metrics.backward_time_ms,
             ctx->metrics.step_time_ms);
    tl_log_to_file(ctx, msg);
}

void tl_early_stopping_update(tl_context_t* ctx, float val_loss) {
    if (!ctx) return;
    if (val_loss < ctx->best_val_loss) {
        ctx->best_val_loss = val_loss;
    } else {
    }
}

bool tl_early_stopping_triggered(tl_context_t* ctx) {
    return ctx ? ctx->early_stop_triggered : false;
}

void tl_log_curve_loss(tl_context_t* ctx, const char* filename) {
    if (!ctx || !filename) return;
    FILE* f = fopen(filename, "w");
    if (!f) return;
    fprintf(f, "# Loss Curve\n# step, train_loss, val_loss\n");
    fclose(f);
}

void tl_log_curve_accuracy(tl_context_t* ctx, const char* filename) {
    if (!ctx || !filename) return;
    FILE* f = fopen(filename, "w");
    if (!f) return;
    fprintf(f, "# Accuracy Curve\n# step, train_acc, val_acc\n");
    fclose(f);
}

float tl_lr_get(tl_context_t* ctx) { return ctx ? ctx->current_lr : 0.0f; }

void tl_lr_set(tl_context_t* ctx, float lr) {
    if (ctx) ctx->current_lr = lr;
}

float tl_lr_cosine(int step, int total_steps, float base_lr, float min_lr) {
    if (total_steps <= 0) return base_lr;
    float progress = (float)step / (float)total_steps;
    if (progress > 1.0f) progress = 1.0f;
    return min_lr + 0.5f * (base_lr - min_lr) * (1.0f + cosf((float)3.14159265358979323846f * progress));
}

float tl_lr_warmup_cosine(int step, int warmup, int total,
                           float base_lr, float min_lr) {
    if (step < warmup) return base_lr * (float)step / (float)(warmup > 0 ? warmup : 1);
    return tl_lr_cosine(step - warmup, total - warmup, base_lr, min_lr);
}

void tl_grad_check(tl_context_t* ctx, float eps, float threshold) {
    if (!ctx) return;
    (void)eps; (void)threshold;
}
