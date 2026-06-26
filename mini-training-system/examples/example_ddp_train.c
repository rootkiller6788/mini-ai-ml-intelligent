#include "distributed_train.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define NUM_GPUS 8
#define MODEL_PARAMS 1000000
#define BATCH_SIZE 256

typedef struct {
    float* weights;
    float* gradients;
    size_t num_params;
    int    rank;
    int    world_size;
    float  local_loss;
} gpu_worker_t;

static void init_model(gpu_worker_t* w, size_t n) {
    w->num_params = n;
    w->weights = (float*)malloc(n * sizeof(float));
    w->gradients = (float*)malloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++) {
        w->weights[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 0.1f;
        w->gradients[i] = 0.0f;
    }
}

static void free_model(gpu_worker_t* w) {
    free(w->weights);
    free(w->gradients);
    w->weights = NULL;
    w->gradients = NULL;
}

static void forward_pass(gpu_worker_t* w, const float* data, float* output,
                          size_t batch, size_t in_dim, size_t out_dim) {
    size_t w_idx = 0;
    for (size_t b = 0; b < batch; b++) {
        for (size_t o = 0; o < out_dim; o++) {
            float sum = 0.0f;
            for (size_t i = 0; i < in_dim; i++) {
                sum += data[b * in_dim + i] * w->weights[w_idx % w->num_params];
                w_idx++;
            }
            output[b * out_dim + o] = sum > 0.0f ? sum : 0.0f;
        }
    }
}

static float compute_loss(const float* output, const float* target,
                           size_t batch, size_t dim) {
    float loss = 0.0f;
    for (size_t b = 0; b < batch; b++) {
        for (size_t d = 0; d < dim; d++) {
            float diff = output[b * dim + d] - target[b * dim + d];
            loss += diff * diff;
        }
    }
    return loss / (float)batch;
}

static void backward_pass(gpu_worker_t* w, const float* data, const float* output,
                           const float* target, size_t batch, size_t in_dim,
                           size_t out_dim) {
    memset(w->gradients, 0, w->num_params * sizeof(float));
    size_t w_idx = 0;
    for (size_t b = 0; b < batch; b++) {
        for (size_t o = 0; o < out_dim; o++) {
            float grad = 2.0f * (output[b * out_dim + o] - target[b * out_dim + o]);
            for (size_t i = 0; i < in_dim; i++) {
                w->gradients[w_idx % w->num_params] += grad * data[b * in_dim + i];
                w_idx++;
            }
        }
    }
}

static void ddp_training_step(gpu_worker_t workers[], int num_gpus,
                               ddp_context_t* ddp_ctx, float* data,
                               float* target, size_t local_batch,
                               size_t in_dim, size_t out_dim) {
    float* outputs[NUM_GPUS];
    float* losses = (float*)calloc((size_t)num_gpus, sizeof(float));
    for (int i = 0; i < num_gpus; i++) {
        outputs[i] = (float*)malloc(local_batch * out_dim * sizeof(float));
    }

    for (int i = 0; i < num_gpus; i++) {
        forward_pass(&workers[i], data, outputs[i], local_batch, in_dim, out_dim);
        losses[i] = compute_loss(outputs[i], target, local_batch, out_dim);
        workers[i].local_loss = losses[i];
        backward_pass(&workers[i], data, outputs[i], target,
                       local_batch, in_dim, out_dim);
    }

    for (int i = 0; i < num_gpus; i++) {
        ddp_all_reduce(workers[i].gradients, workers[i].num_params,
                        RING_REDUCE_OP_AVG, ddp_ctx);
    }

    float lr = 0.001f;
    for (int i = 0; i < num_gpus; i++) {
        for (size_t j = 0; j < workers[i].num_params; j++) {
            workers[i].weights[j] -= lr * workers[i].gradients[j];
        }
    }

    for (int i = 0; i < num_gpus; i++) free(outputs[i]);
    free(losses);
}

static void demo_hybrid_3d(void) {
    printf("\n--- Hybrid 3D Parallelism Demo ---\n");
    hybrid_3d_config_t cfg = {0};
    cfg.tp_size = 2;
    cfg.pp_size = 2;
    int world_size = 8;
    hybrid_3d_init(&cfg, world_size);
    printf("3D: tp=%d pp=%d dp=%d micro_batches=%d\n",
           cfg.tp_size, cfg.pp_size, cfg.dp_size, cfg.num_micro_batches);

    for (int stage = 0; stage < cfg.pp_size; stage++) {
        printf("  Pipeline stage %d: ", stage);
        pipeline_schedule_1f1b(stage, cfg.pp_size, cfg.num_micro_batches);
        printf("1F1B scheduled\n");
    }
}

static void demo_zero_optimizer(void) {
    printf("\n--- ZeRO Optimizer Demo ---\n");
    zero_config_t cfg;
    size_t param_size = MODEL_PARAMS * sizeof(float);

    for (int s = 0; s <= ZERO_STAGE_3; s++) {
        zero_init(&cfg, (zero_stage_t)s);
        printf("Stage %d: ", s);
        switch (cfg.stage) {
        case ZERO_STAGE_0: printf("No partitioning"); break;
        case ZERO_STAGE_1: printf("Optimizer state partitioned"); break;
        case ZERO_STAGE_2: printf("+ Gradient partitioned"); break;
        case ZERO_STAGE_3: printf("+ Parameter partitioned"); break;
        }

        size_t per_gpu = param_size / (size_t)(NUM_GPUS);
        size_t states_gpu = (cfg.stage >= ZERO_STAGE_1) ? per_gpu : param_size;
        size_t grads_gpu = (cfg.stage >= ZERO_STAGE_2) ? per_gpu : param_size;
        size_t params_gpu = (cfg.stage >= ZERO_STAGE_3) ? per_gpu : param_size;

        printf(" | Mem/GPU: %.2f MB\n",
               (float)(states_gpu + grads_gpu + params_gpu) / (1024.0f * 1024.0f));
    }
}

static void demo_ring_allreduce_benchmark(void) {
    printf("\n--- Ring All-Reduce Benchmark ---\n");

    size_t sizes[] = {1024, 65536, 1048576};
    int num_sizes = 3;

    for (int si = 0; si < num_sizes; si++) {
        size_t numel = sizes[si];
        float* buf = (float*)malloc(numel * sizeof(float));
        for (size_t i = 0; i < numel; i++) buf[i] = (float)(rand() & 0xFF);

        clock_t start = clock();
        int n_repeats = (numel > 100000) ? 10 : 100;
        for (int r = 0; r < n_repeats; r++) {
            ring_all_reduce(buf, numel, RING_REDUCE_OP_SUM, 0, NUM_GPUS);
        }
        clock_t end = clock();
        double ms = 1000.0 * (double)(end - start) / (double)(CLOCKS_PER_SEC * n_repeats);
        double bw = (double)(numel * sizeof(float) * 2 * (NUM_GPUS - 1) / NUM_GPUS)
                     / (ms / 1000.0) / (1024.0 * 1024.0);

        printf("  Size=%8zu elems | Time=%.2f ms | BW=%.1f MB/s\n", numel, ms, bw);
        free(buf);
    }
}

static void demo_allgather(void) {
    printf("\n--- All-Gather Demo ---\n");

    int world = 4;
    size_t per_rank = 10;
    size_t total = per_rank * (size_t)world;
    float* buf = (float*)malloc(total * sizeof(float));

    for (int r = 0; r < world; r++) {
        for (size_t i = 0; i < per_rank; i++) {
            buf[i] = (float)(r * 100 + (int)i);
        }
        printf("  Rank %d initial: ", r);
        for (size_t i = 0; i < 3 && i < total; i++) printf("%.0f ", buf[i]);
        printf("...\n");

        allgather_ring(buf, total, r, world);
        printf("  Rank %d after AG: ", r);
        for (size_t i = 0; i < 3 && i < total; i++) printf("%.0f ", buf[i]);
        printf("...\n");
    }
    free(buf);
}

int main(void) {
    printf("============================================================\n");
    printf("  DDP Training Example - Distributed Data Parallel\n");
    printf("============================================================\n");

    srand(42);
    gpu_worker_t workers[NUM_GPUS];
    ddp_context_t ddp_ctx;

    ddp_init(&ddp_ctx, 0, NULL);
    ddp_ctx.world_size = NUM_GPUS;

    size_t params_per_gpu = MODEL_PARAMS;
    for (int i = 0; i < NUM_GPUS; i++) {
        workers[i].rank = i;
        workers[i].world_size = NUM_GPUS;
        init_model(&workers[i], params_per_gpu);
    }

    size_t local_batch = (size_t)(BATCH_SIZE / NUM_GPUS);
    size_t in_dim = 784;
    size_t out_dim = 10;

    float* data = (float*)malloc(local_batch * in_dim * sizeof(float));
    float* target = (float*)calloc(local_batch * out_dim, sizeof(float));
    for (size_t i = 0; i < local_batch * in_dim; i++) data[i] = (float)rand() / (float)RAND_MAX;
    for (size_t b = 0; b < local_batch; b++) target[b * out_dim + (rand() % out_dim)] = 1.0f;

    printf("Config: GPUs=%d, Total batch=%d, Local batch=%zu, Params=%zu\n",
           NUM_GPUS, BATCH_SIZE, local_batch, params_per_gpu);
    printf("Model: input=%zu, output=%zu\n\n", in_dim, out_dim);

    int n_steps = 5;
    for (int step = 0; step < n_steps; step++) {
        ddp_training_step(workers, NUM_GPUS, &ddp_ctx, data, target,
                          local_batch, in_dim, out_dim);

        float avg_loss = 0.0f;
        for (int i = 0; i < NUM_GPUS; i++) avg_loss += workers[i].local_loss;
        avg_loss /= (float)NUM_GPUS;

        float* grad_buf = (float*)malloc(params_per_gpu * sizeof(float));
        memcpy(grad_buf, workers[0].gradients, params_per_gpu * sizeof(float));
        ddp_all_reduce(grad_buf, params_per_gpu, RING_REDUCE_OP_AVG, &ddp_ctx);

        printf("  Step %d | Loss=%.4f | GradNorm(rank0)=%.4f | AllReduced=%.4f\n",
               step, avg_loss, sqrtf(workers[0].gradients[0] * workers[0].gradients[0]),
               sqrtf(grad_buf[0] * grad_buf[0]));
        free(grad_buf);
    }

    printf("\nDDP gradient check: ");
    bool finite = ddp_gradient_is_finite(workers[0].gradients, params_per_gpu, NUM_GPUS);
    printf("%s\n\n", finite ? "ALL FINITE" : "INF/NAN DETECTED");

    demo_ring_allreduce_benchmark();
    demo_zero_optimizer();
    demo_hybrid_3d();
    demo_allgather();

    for (int i = 0; i < NUM_GPUS; i++) free_model(&workers[i]);
    free(data);
    free(target);
    ddp_finalize(&ddp_ctx);
    return 0;
}
