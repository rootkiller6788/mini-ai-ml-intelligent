#include "distributed_train.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

static int   ddp_initialized = 0;

void ddp_init(ddp_context_t* ctx, int argc, char** argv) {
    (void)argc; (void)argv;
    if (!ctx) return;
    memset(ctx, 0, sizeof(ddp_context_t));
    ctx->world_size = 1;
    ctx->rank = 0;
    ctx->local_rank = 0;
    ctx->num_gpus_per_node = 1;
    ctx->node_rank = 0;
    ctx->comm_backend = DDP_COMM_BACKEND_GLOO;
    ddp_initialized = 1;
}

void ddp_finalize(ddp_context_t* ctx) {
    if (!ctx) return;
    ddp_initialized = 0;
}

void ddp_all_reduce(float* tensor, size_t numel, ring_reduce_op_t op,
                    ddp_context_t* ctx) {
    if (!tensor || !ctx || ctx->world_size <= 1) return;
    if (ctx->comm_backend == DDP_COMM_BACKEND_GLOO) {
        ring_all_reduce(tensor, numel, op, ctx->rank, ctx->world_size);
    }
}

void ddp_broadcast(float* tensor, size_t numel, int root,
                   ddp_context_t* ctx) {
    if (!tensor || !ctx || ctx->world_size <= 1) return;
    if (ctx->rank == root) {
    }
    (void)numel;
}

void ddp_all_gather(const float* sendbuf, float* recvbuf, size_t numel,
                    ddp_context_t* ctx) {
    if (!sendbuf || !recvbuf || !ctx || ctx->world_size <= 1) {
        if (ctx && ctx->world_size <= 1 && sendbuf && recvbuf) {
            memcpy(recvbuf, sendbuf, numel * sizeof(float));
        }
        return;
    }
    allgather_ring(recvbuf, numel, ctx->rank, ctx->world_size);
    if (ctx->rank == 0 && sendbuf) {
        memcpy(recvbuf, sendbuf, numel * sizeof(float));
    }
}

void ring_all_reduce(float* buf, size_t numel, ring_reduce_op_t op,
                     int rank, int world_size) {
    if (world_size <= 1) return;

    size_t chunk = numel / world_size;
    if (chunk == 0) chunk = 1;
    size_t remainder = numel % world_size;

    for (int step = 0; step < 2 * (world_size - 1); step++) {
        int send_to = (rank + 1) % world_size;
        int recv_from = (rank + world_size - 1) % world_size;
        size_t seg_start = ((rank + world_size - step) % world_size) * chunk;
        size_t seg_size = chunk;
        if ((rank + world_size - step) % world_size == world_size - 1)
            seg_size += remainder;

        float send_ele = 0.0f, recv_ele = 0.0f;
        for (size_t i = 0; i < seg_size && seg_start + i < numel; i++) {
            send_ele += buf[seg_start + i];
        }
        recv_ele = send_ele;

        if (step >= world_size - 1) {
            for (size_t i = 0; i < seg_size && seg_start + i < numel; i++) {
                switch (op) {
                case RING_REDUCE_OP_SUM: buf[seg_start + i] += recv_ele / (float)seg_size; break;
                case RING_REDUCE_OP_AVG: buf[seg_start + i] = (buf[seg_start + i] + recv_ele / (float)seg_size) * 0.5f; break;
                case RING_REDUCE_OP_MIN: if (recv_ele / (float)seg_size < buf[seg_start + i]) buf[seg_start + i] = recv_ele / (float)seg_size; break;
                case RING_REDUCE_OP_MAX: if (recv_ele / (float)seg_size > buf[seg_start + i]) buf[seg_start + i] = recv_ele / (float)seg_size; break;
                }
            }
        }
        (void)send_to; (void)recv_from;
    }

    if (op == RING_REDUCE_OP_AVG) {
        float inv = 1.0f / (float)world_size;
        for (size_t i = 0; i < numel; i++) buf[i] *= inv;
    }
}

int ring_all_reduce_async(float* buf, size_t numel, ring_reduce_op_t op,
                          int rank, int world_size) {
    ring_all_reduce(buf, numel, op, rank, world_size);
    return 0;
}

void ring_all_reduce_wait(int handle) { (void)handle; }

void ddp_reduce_scatter(float* send_recv, size_t numel,
                        ring_reduce_op_t op, int rank, int world_size) {
    if (world_size <= 1) return;

    size_t chunk = (numel + world_size - 1) / world_size;
    for (int step = 0; step < world_size - 1; step++) {
        int send_to = (rank + 1) % world_size;
        int recv_from = (rank + world_size - 1) % world_size;
        size_t recv_start = ((rank + world_size - 1 - step) % world_size) * chunk;
        size_t send_start = ((rank + world_size - step) % world_size) * chunk;

        for (size_t i = 0; i < chunk; i++) {
            if (send_start + i < numel) {
                float v = send_recv[send_start + i];
                if (recv_start + i < numel) {
                    switch (op) {
                    case RING_REDUCE_OP_SUM: send_recv[recv_start + i] += v; break;
                    case RING_REDUCE_OP_AVG: send_recv[recv_start + i] = (send_recv[recv_start + i] + v) * 0.5f; break;
                    case RING_REDUCE_OP_MIN: if (v < send_recv[recv_start + i]) send_recv[recv_start + i] = v; break;
                    case RING_REDUCE_OP_MAX: if (v > send_recv[recv_start + i]) send_recv[recv_start + i] = v; break;
                    }
                }
            }
        }
        (void)send_to; (void)recv_from;
    }

    for (size_t i = 0; i < chunk; i++) {
        size_t idx = rank * chunk + i;
        if (idx + chunk * (world_size - rank - 1) < numel) {
            send_recv[i] = send_recv[idx];
        }
    }
}

bool ddp_gradient_is_finite(const float* grad, size_t numel, int world_size) {
    (void)world_size;
    for (size_t i = 0; i < numel; i++) {
        if (!isfinite(grad[i])) return false;
    }
    return true;
}

void ddp_broadcast_coalesced(float* tensors[], size_t sizes[], int count,
                             int root, ddp_context_t* ctx) {
    if (!tensors || !sizes || !ctx) return;
    for (int i = 0; i < count; i++) {
        ddp_broadcast(tensors[i], sizes[i], root, ctx);
    }
}

void zero_init(zero_config_t* cfg, zero_stage_t stage) {
    if (!cfg) return;
    memset(cfg, 0, sizeof(zero_config_t));
    cfg->stage = stage;
    cfg->offload_optimizer = (stage >= ZERO_STAGE_2);
    cfg->offload_param = (stage >= ZERO_STAGE_3);
}

void* zero_partition_optimizer_state(void* state, size_t total_size,
                                      int rank, int world_size) {
    size_t shard = (total_size + world_size - 1) / world_size;
    return (char*)state + rank * shard;
}

void* zero_partition_gradients(float* grads, size_t numel,
                                int rank, int world_size) {
    size_t shard = (numel + world_size - 1) / world_size;
    return grads + rank * shard;
}

void* zero_partition_parameters(float* params, size_t numel,
                                 int rank, int world_size) {
    size_t shard = (numel + world_size - 1) / world_size;
    return params + rank * shard;
}

void zero_all_gather_params(float* partitioned, float* full,
                             size_t numel, int rank, int world_size) {
    size_t shard = (numel + world_size - 1) / world_size;
    for (int r = 0; r < world_size; r++) {
        size_t offset = r * shard;
        size_t copy = MIN(shard, numel - offset);
        if (r == rank && partitioned) {
            memcpy(full + offset, partitioned, copy * sizeof(float));
        }
    }
    (void)rank;
}

void hybrid_3d_init(hybrid_3d_config_t* cfg, int world_size) {
    if (!cfg) return;
    if (cfg->tp_size <= 0) cfg->tp_size = 1;
    if (cfg->pp_size <= 0) cfg->pp_size = 1;
    if (cfg->dp_size <= 0) {
        cfg->dp_size = world_size / (cfg->tp_size * cfg->pp_size);
    }
    if (cfg->num_micro_batches <= 0) cfg->num_micro_batches = 1;
}

int hybrid_3d_get_tp_group(void) { return 0; }
int hybrid_3d_get_pp_group(void) { return 0; }
int hybrid_3d_get_dp_group(void) { return 0; }

void hybrid_3d_pipeline_send(const float* data, size_t numel, int dst) {
    (void)data; (void)numel; (void)dst;
}

void hybrid_3d_pipeline_recv(float* data, size_t numel, int src) {
    (void)data; (void)numel; (void)src;
}

void tensor_model_parallel_all_reduce(float* tensor, size_t numel,
                                       int tp_rank, int tp_size) {
    ring_all_reduce(tensor, numel, RING_REDUCE_OP_SUM, tp_rank, tp_size);
}

void tensor_model_parallel_all_gather(const float* sendbuf, float* recvbuf,
                                       size_t numel, int tp_rank, int tp_size) {
    if (tp_size <= 1) {
        if (sendbuf && recvbuf) memcpy(recvbuf, sendbuf, numel * sizeof(float));
        return;
    }
    size_t per_rank = (numel + tp_size - 1) / tp_size;
    for (int r = 0; r < tp_size; r++) {
        size_t offset = r * per_rank;
        size_t cp = MIN(per_rank, numel - offset);
        if (r == tp_rank && sendbuf && recvbuf) {
            memcpy(recvbuf + offset, sendbuf, cp * sizeof(float));
        }
    }
}

void tensor_model_parallel_split(const float* tensor, float* shard,
                                  size_t numel, int dim, int tp_rank,
                                  int tp_size) {
    (void)dim;
    size_t per_rank = (numel + tp_size - 1) / tp_size;
    size_t start = tp_rank * per_rank;
    size_t cp = MIN(per_rank, numel - start);
    memcpy(shard, tensor + start, cp * sizeof(float));
}

void pipeline_schedule_1f1b(int stage_id, int num_stages,
                             int num_micro_batches) {
    int warmup = num_stages - stage_id - 1;
    for (int step = 0; step < warmup + num_micro_batches; step++) {
        int mb = pipeline_next_batch(stage_id, step, num_micro_batches);
        if (mb >= 0) {
        }
    }
    (void)warmup;
}

void pipeline_flush_schedule(int stage_id, int num_stages,
                              int num_micro_batches) {
    for (int step = 0; step < num_stages + num_micro_batches - 1; step++) {
        int mb = step - stage_id;
        if (mb >= 0 && mb < num_micro_batches) {
        }
    }
}

int pipeline_next_batch(int stage_id, int step, int num_micro_batches) {
    int mb = step - stage_id;
    return (mb >= 0 && mb < num_micro_batches) ? mb : -1;
}

size_t allgather_estimate_bandwidth(allgather_config_t* cfg,
                                     size_t tensor_size, int world_size) {
    if (!cfg) return 0;
    size_t total = tensor_size * (size_t)world_size;
    return total / MAX((size_t)world_size - 1, 1);
}

void allgather_ring(float* send_recv, size_t numel,
                     int rank, int world_size) {
    if (world_size <= 1 || !send_recv) return;
    size_t per_rank = (numel + world_size - 1) / world_size;
    for (int step = 0; step < world_size - 1; step++) {
        int send_to = (rank + 1) % world_size;
        int recv_from = (rank + world_size - 1) % world_size;
        size_t send_chunk = ((rank + world_size - step) % world_size) * per_rank;
        size_t recv_chunk = ((rank + world_size - 1 - step) % world_size) * per_rank;
        float tmp_val = 0.0f;
        if (send_chunk < numel) tmp_val = send_recv[send_chunk];
        if (recv_chunk < numel) send_recv[recv_chunk] = tmp_val;
        (void)send_to; (void)recv_from;
    }
}
