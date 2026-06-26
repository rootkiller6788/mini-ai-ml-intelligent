#include "training.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

/* ═══════════════════════════════════════════════════════════════════════
 * DataLoader (L3: Engineering Structure — data pipeline)
 * Mini-batch SGD data feeder with optional shuffle.
 * L6: Standard ML engineering problem — efficient data iteration.
 * ═══════════════════════════════════════════════════════════════════════ */

DataLoader* dataloader_create(float* data, int* data_dims, int data_ndim,
                               float* labels, int* label_dims, int label_ndim,
                               int batch_size, bool shuffle) {
    DataLoader* dl = (DataLoader*)malloc(sizeof(DataLoader));
    dl->batch_size = batch_size;
    dl->shuffle = shuffle;

    int total_samples = data_dims[0];
    dl->total_samples = total_samples;
    int num_batches = (total_samples + batch_size - 1) / batch_size;
    dl->num_batches = num_batches;

    int sample_size = 1;
    for (int i = 1; i < data_ndim; i++) sample_size *= data_dims[i];
    int label_size = 1;
    for (int i = 1; i < label_ndim; i++) label_size *= label_dims[i];

    dl->batches = (Tensor**)malloc(sizeof(Tensor*) * num_batches);
    dl->targets = (float**)malloc(sizeof(float*) * num_batches);

    for (int b = 0; b < num_batches; b++) {
        int start = b * batch_size;
        int end = (start + batch_size) < total_samples ? (start + batch_size) : total_samples;
        int cur_batch_size = end - start;

        int* batch_dims = (int*)malloc(sizeof(int) * data_ndim);
        batch_dims[0] = cur_batch_size;
        for (int i = 1; i < data_ndim; i++) batch_dims[i] = data_dims[i];

        dl->batches[b] = tensor_create(batch_dims, data_ndim);
        free(batch_dims);

        memcpy(dl->batches[b]->data,
               data + start * sample_size,
               sizeof(float) * cur_batch_size * sample_size);

        dl->targets[b] = (float*)malloc(sizeof(float) * cur_batch_size * label_size);
        memcpy(dl->targets[b],
               labels + start * label_size,
               sizeof(float) * cur_batch_size * label_size);
    }

    dl->indices = (int*)malloc(sizeof(int) * total_samples);
    for (int i = 0; i < total_samples; i++) dl->indices[i] = i;

    return dl;
}

void dataloader_reset(DataLoader* loader) {
    if (loader->shuffle) {
        /* Fisher-Yates shuffle */
        for (int i = loader->total_samples - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            int tmp = loader->indices[i];
            loader->indices[i] = loader->indices[j];
            loader->indices[j] = tmp;
        }
    }
}

int dataloader_next_batch(DataLoader* loader,
                           Tensor** batch_out, Tensor** target_out) {
    static int current_batch = 0;
    if (current_batch >= loader->num_batches) {
        current_batch = 0;
        return 0; /* epoch done */
    }

    *batch_out = loader->batches[current_batch];

    int batch_sz = (*batch_out)->dims[0];
    int label_elems = 1; /* Assume 1D labels per sample */
    Tensor* targ_tensor = tensor_create((int[]){batch_sz}, 1);
    for (int i = 0; i < batch_sz; i++)
        targ_tensor->data[i] = loader->targets[current_batch][i];
    *target_out = targ_tensor;

    current_batch++;
    return 1;
}

void dataloader_free(DataLoader* loader) {
    if (!loader) return;
    if (loader->batches) {
        for (int i = 0; i < loader->num_batches; i++)
            tensor_free(loader->batches[i]);
        free(loader->batches);
    }
    if (loader->targets) {
        for (int i = 0; i < loader->num_batches; i++)
            free(loader->targets[i]);
        free(loader->targets);
    }
    free(loader->indices);
    free(loader);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Trainer (L6: Training Loop Abstraction)
 * Encapsulates the standard ML training loop: forward, loss, backward,
 * optimizer step. Includes early stopping and learning rate scheduling.
 * L7: Application — train any model with pluggable loss/optimizer.
 * ═══════════════════════════════════════════════════════════════════════ */

Trainer* trainer_create(LayerParam* params, void* optimizer,
                         LRScheduler* scheduler,
                         void (*step)(void*), void (*zero_grad)(void*),
                         TaskMode mode) {
    Trainer* t = (Trainer*)malloc(sizeof(Trainer));
    t->params = params;
    t->optimizer = optimizer;
    t->scheduler = scheduler;
    t->opt_step = step;
    t->opt_zero_grad = zero_grad;
    t->task_mode = mode;
    t->loss_fn = NULL;
    t->loss_grad_fn = NULL;
    t->current_epoch = 0;
    t->total_steps = 0;
    t->current_lr = lr_scheduler_get_lr(scheduler);
    t->best_val_loss = 1e9f;
    t->patience_counter = 0;

    /* Count parameters */
    t->num_params = 0;
    LayerParam* p = params;
    while (p && p->param) { t->num_params++; p = p->next; }

    /* Build flat array */
    t->param_list = (Tensor**)malloc(sizeof(Tensor*) * t->num_params);
    t->grad_list  = (Tensor**)malloc(sizeof(Tensor*) * t->num_params);
    p = params;
    int idx = 0;
    while (p && p->param) {
        t->param_list[idx] = p->param;
        t->grad_list[idx]  = p->grad;
        idx++;
        p = p->next;
    }
    return t;
}

void trainer_set_loss(Trainer* t,
                       float (*loss_fn)(Tensor*, Tensor*, LossReduction),
                       Tensor* (*loss_grad_fn)(Tensor*, Tensor*, LossReduction)) {
    t->loss_fn = loss_fn;
    t->loss_grad_fn = loss_grad_fn;
}

float trainer_train_epoch(Trainer* t, DataLoader* loader) {
    dataloader_reset(loader);
    float total_loss = 0;
    int num_batches = 0;

    while (1) {
        Tensor* batch = NULL;
        Tensor* target = NULL;
        int has_next = dataloader_next_batch(loader, &batch, &target);
        if (!has_next) break;

        /* Forward pass — model-specific, user should implement externally */
        /* For now, compute loss directly if loss_fn is set */
        if (t->loss_fn) {
            float loss = t->loss_fn(batch, target, REDUCE_MEAN);
            total_loss += loss;
            num_batches++;
        }

        /* Backward: compute gradients via loss_grad_fn */
        if (t->loss_grad_fn) {
            Tensor* grad = t->loss_grad_fn(batch, target, REDUCE_MEAN);
            /* Gradients are accumulated in grad_list via loss functions */
            tensor_free(grad);
        }

        /* Optimizer step */
        if (t->opt_step) {
            t->opt_step(t->optimizer);
        }
        t->total_steps++;

        tensor_free(target);
    }

    t->current_epoch++;
    return num_batches > 0 ? total_loss / (float)num_batches : 0.0f;
}

float trainer_validate(Trainer* t, DataLoader* loader) {
    float total_loss = 0;
    int num_batches = 0;
    dataloader_reset(loader);

    while (1) {
        Tensor* batch = NULL;
        Tensor* target = NULL;
        int has_next = dataloader_next_batch(loader, &batch, &target);
        if (!has_next) break;

        if (t->loss_fn) {
            float loss = t->loss_fn(batch, target, REDUCE_MEAN);
            total_loss += loss;
            num_batches++;
        }
        tensor_free(target);
    }

    return num_batches > 0 ? total_loss / (float)num_batches : 0.0f;
}

bool trainer_early_stop(Trainer* t, float val_loss, int patience) {
    if (val_loss < t->best_val_loss) {
        t->best_val_loss = val_loss;
        t->patience_counter = 0;
        return false;
    }
    t->patience_counter++;
    return t->patience_counter >= patience;
}

void trainer_free(Trainer* t) {
    if (!t) return;
    free(t->param_list);
    free(t->grad_list);
    free(t);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Model Checkpoint (L7: Serialization)
 * Save/load model weights to/from binary files for persistence.
 * Format: [num_tensors][ndim][dims...][data...] per tensor.
 * ═══════════════════════════════════════════════════════════════════════ */

ModelCheckpoint* checkpoint_save(Tensor** params, char** names,
                                  int num_params, int epoch,
                                  float loss, const char* tag) {
    ModelCheckpoint* ckpt = (ModelCheckpoint*)malloc(sizeof(ModelCheckpoint));
    ckpt->num_tensors = num_params;
    ckpt->epoch = epoch;
    ckpt->loss = loss;
    ckpt->tag = tag ? strdup(tag) : NULL;

    ckpt->tensors = (CheckpointTensor*)malloc(sizeof(CheckpointTensor) * num_params);
    ckpt->names   = (char**)malloc(sizeof(char*) * num_params);

    for (int i = 0; i < num_params; i++) {
        ckpt->names[i] = names[i] ? strdup(names[i]) : NULL;
        ckpt->tensors[i].ndim = params[i]->ndim;
        ckpt->tensors[i].size = params[i]->size;
        ckpt->tensors[i].dims = (int*)malloc(sizeof(int) * params[i]->ndim);
        memcpy(ckpt->tensors[i].dims, params[i]->dims,
               sizeof(int) * params[i]->ndim);
        ckpt->tensors[i].data = (float*)malloc(sizeof(float) * params[i]->size);
        memcpy(ckpt->tensors[i].data, params[i]->data,
               sizeof(float) * params[i]->size);
    }
    return ckpt;
}

void checkpoint_load(ModelCheckpoint* ckpt, Tensor** params, int num_params) {
    for (int i = 0; i < num_params && i < ckpt->num_tensors; i++) {
        if (params[i]->size != ckpt->tensors[i].size) continue;
        memcpy(params[i]->data, ckpt->tensors[i].data,
               sizeof(float) * params[i]->size);
    }
}

void checkpoint_free(ModelCheckpoint* ckpt) {
    if (!ckpt) return;
    for (int i = 0; i < ckpt->num_tensors; i++) {
        free(ckpt->tensors[i].dims);
        free(ckpt->tensors[i].data);
    }
    free(ckpt->tensors);
    if (ckpt->names) {
        for (int i = 0; i < ckpt->num_tensors; i++)
            free(ckpt->names[i]);
        free(ckpt->names);
    }
    free(ckpt->tag);
    free(ckpt);
}

void checkpoint_write_file(ModelCheckpoint* ckpt, const char* filepath) {
    FILE* f = fopen(filepath, "wb");
    if (!f) return;

    fwrite(&ckpt->num_tensors, sizeof(int), 1, f);
    fwrite(&ckpt->epoch, sizeof(int), 1, f);
    fwrite(&ckpt->loss, sizeof(float), 1, f);

    for (int i = 0; i < ckpt->num_tensors; i++) {
        fwrite(&ckpt->tensors[i].ndim, sizeof(int), 1, f);
        fwrite(ckpt->tensors[i].dims, sizeof(int), ckpt->tensors[i].ndim, f);
        fwrite(&ckpt->tensors[i].size, sizeof(int), 1, f);
        fwrite(ckpt->tensors[i].data, sizeof(float), ckpt->tensors[i].size, f);
    }

    fclose(f);
}

ModelCheckpoint* checkpoint_read_file(const char* filepath) {
    FILE* f = fopen(filepath, "rb");
    if (!f) return NULL;

    ModelCheckpoint* ckpt = (ModelCheckpoint*)malloc(sizeof(ModelCheckpoint));
    fread(&ckpt->num_tensors, sizeof(int), 1, f);
    fread(&ckpt->epoch, sizeof(int), 1, f);
    fread(&ckpt->loss, sizeof(float), 1, f);

    ckpt->tensors = (CheckpointTensor*)malloc(sizeof(CheckpointTensor) * ckpt->num_tensors);
    ckpt->names = (char**)calloc(ckpt->num_tensors, sizeof(char*));
    ckpt->tag = NULL;

    for (int i = 0; i < ckpt->num_tensors; i++) {
        fread(&ckpt->tensors[i].ndim, sizeof(int), 1, f);
        ckpt->tensors[i].dims = (int*)malloc(sizeof(int) * ckpt->tensors[i].ndim);
        fread(ckpt->tensors[i].dims, sizeof(int), ckpt->tensors[i].ndim, f);
        fread(&ckpt->tensors[i].size, sizeof(int), 1, f);
        ckpt->tensors[i].data = (float*)malloc(sizeof(float) * ckpt->tensors[i].size);
        fread(ckpt->tensors[i].data, sizeof(float), ckpt->tensors[i].size, f);
    }

    fclose(f);
    return ckpt;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Evaluation Metrics (L7: Application)
 * Accuracy for classification, precision/recall/F1 for binary tasks.
 * ═══════════════════════════════════════════════════════════════════════ */

float accuracy(Tensor* logits, Tensor* labels) {
    int N = logits->dims[0];
    int C = logits->ndim > 1 ? logits->dims[1] : 1;
    int correct = 0;

    for (int n = 0; n < N; n++) {
        int pred_class = 0;
        float max_val = logits->data[n * C];
        for (int c = 1; c < C; c++) {
            if (logits->data[n * C + c] > max_val) {
                max_val = logits->data[n * C + c];
                pred_class = c;
            }
        }
        int true_label = (int)(labels->data[n] + 0.5f);
        if (pred_class == true_label) correct++;
    }
    return (float)correct / (float)N;
}

float precision_recall_f1(Tensor* logits, Tensor* labels, int positive_class,
                           float* precision, float* recall) {
    int N = logits->dims[0];
    int C = logits->ndim > 1 ? logits->dims[1] : 1;
    int tp = 0, fp = 0, fn = 0;

    for (int n = 0; n < N; n++) {
        int pred_class = 0;
        float max_val = logits->data[n * C];
        for (int c = 1; c < C; c++) {
            if (logits->data[n * C + c] > max_val) {
                max_val = logits->data[n * C + c];
                pred_class = c;
            }
        }
        int true_label = (int)(labels->data[n] + 0.5f);
        if (pred_class == positive_class && true_label == positive_class) tp++;
        else if (pred_class == positive_class && true_label != positive_class) fp++;
        else if (pred_class != positive_class && true_label == positive_class) fn++;
    }

    *precision = tp > 0 ? (float)tp / (float)(tp + fp) : 0.0f;
    *recall    = tp > 0 ? (float)tp / (float)(tp + fn) : 0.0f;
    float f1 = (*precision + *recall) > 0
        ? 2.0f * (*precision) * (*recall) / (*precision + *recall)
        : 0.0f;
    return f1;
}
