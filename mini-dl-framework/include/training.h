#ifndef TRAINING_H
#define TRAINING_H

#include "tensor_ops.h"
#include "nn_layers.h"
#include "optimizers.h"
#include "loss_funcs.h"
#include <stdbool.h>

/* ── L1 + L3: DataLoader Engineering Structure ───────────────── */

typedef struct {
    Tensor** batches;       /* pre-split batches */
    float** targets;         /* corresponding targets (flat float arrays) */
    int num_batches;
    int batch_size;
    int total_samples;
    bool shuffle;
    int* indices;           /* permutation indices for shuffle */
} DataLoader;

DataLoader* dataloader_create(float* data, int* data_dims, int data_ndim,
                               float* labels, int* label_dims, int label_ndim,
                               int batch_size, bool shuffle);
void dataloader_reset(DataLoader* loader);
int  dataloader_next_batch(DataLoader* loader,
                            Tensor** batch_out, Tensor** target_out);
void dataloader_free(DataLoader* loader);

/* ── L6: Training Loop Abstraction ───────────────────────────── */

typedef enum {
    MODE_CLASSIFICATION,
    MODE_REGRESSION
} TaskMode;

typedef struct {
    /* Model state */
    LayerParam* params;     /* linked list of all trainable parameters */
    int num_params;

    /* Optimizer state */
    void* optimizer;
    LRScheduler* scheduler;
    void (*opt_step)(void*);
    void (*opt_zero_grad)(void*);

    /* Loss function */
    TaskMode task_mode;
    float (*loss_fn)(Tensor*, Tensor*, LossReduction);
    Tensor* (*loss_grad_fn)(Tensor*, Tensor*, LossReduction);

    /* Training state */
    int current_epoch;
    int total_steps;
    float current_lr;
    float best_val_loss;
    int patience_counter;

    /* Metrics */
    Tensor** param_list;     /* flat array for easier grad access */
    Tensor** grad_list;
} Trainer;

Trainer* trainer_create(LayerParam* params, void* optimizer,
                         LRScheduler* scheduler,
                         void (*step)(void*), void (*zero_grad)(void*),
                         TaskMode mode);
void trainer_set_loss(Trainer* t,
                       float (*loss_fn)(Tensor*, Tensor*, LossReduction),
                       Tensor* (*loss_grad_fn)(Tensor*, Tensor*, LossReduction));
float trainer_train_epoch(Trainer* t, DataLoader* loader);
float trainer_validate(Trainer* t, DataLoader* loader);
bool trainer_early_stop(Trainer* t, float val_loss, int patience);
void trainer_free(Trainer* t);

/* ── L7: Model Checkpoint (serialization) ────────────────────── */

typedef struct {
    float* data;
    int* dims;
    int ndim;
    int size;
} CheckpointTensor;

typedef struct {
    CheckpointTensor* tensors;
    char** names;
    int num_tensors;
    int epoch;
    float loss;
    char* tag;
} ModelCheckpoint;

ModelCheckpoint* checkpoint_save(Tensor** params, char** names,
                                  int num_params, int epoch,
                                  float loss, const char* tag);
void checkpoint_load(ModelCheckpoint* ckpt, Tensor** params, int num_params);
void checkpoint_free(ModelCheckpoint* ckpt);
void checkpoint_write_file(ModelCheckpoint* ckpt, const char* filepath);
ModelCheckpoint* checkpoint_read_file(const char* filepath);

/* ── L7: Accuracy / Evaluation Metrics ──────────────────────── */

float accuracy(Tensor* logits, Tensor* labels);
float precision_recall_f1(Tensor* logits, Tensor* labels, int positive_class,
                           float* precision, float* recall);

#endif /* TRAINING_H */
