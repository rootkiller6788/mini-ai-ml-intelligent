#ifndef DISTRIBUTED_TRAIN_H
#define DISTRIBUTED_TRAIN_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DDP_MAX_GPUS 256
#define DDP_RING_CHUNK_SIZE (4 * 1024 * 1024)
#define DDP_COMM_BACKEND_NCCL 0
#define DDP_COMM_BACKEND_MPI  1
#define DDP_COMM_BACKEND_GLOO 2

typedef enum {
    DDP_STRATEGY_DATA_PARALLEL = 0,
    DDP_STRATEGY_MODEL_PARALLEL,
    DDP_STRATEGY_PIPELINE_PARALLEL,
    DDP_STRATEGY_HYBRID_3D,
} ddp_strategy_t;

typedef enum {
    ZERO_STAGE_0 = 0,
    ZERO_STAGE_1,
    ZERO_STAGE_2,
    ZERO_STAGE_3,
} zero_stage_t;

typedef enum {
    RING_REDUCE_OP_SUM = 0,
    RING_REDUCE_OP_AVG,
    RING_REDUCE_OP_MIN,
    RING_REDUCE_OP_MAX,
} ring_reduce_op_t;

typedef struct {
    int world_size;
    int rank;
    int local_rank;
    int num_gpus_per_node;
    int node_rank;
    int comm_backend;
} ddp_context_t;

typedef struct {
    int tp_size;
    int pp_size;
    int dp_size;
    int num_micro_batches;
    bool overlap_comm;
} hybrid_3d_config_t;

typedef struct {
    zero_stage_t stage;
    size_t param_size;
    size_t grad_size;
    size_t optimizer_state_size;
    float cpu_offload_ratio;
    bool offload_optimizer;
    bool offload_param;
} zero_config_t;

typedef enum {
    ALLGATHER_SIMPLE  = 0,
    ALLGATHER_BUCKETED,
    ALLGATHER_PIPELINED,
} allgather_algo_t;

typedef struct {
    allgather_algo_t algo;
    size_t bucket_size;
    size_t num_buckets;
} allgather_config_t;

void ddp_init(ddp_context_t* ctx, int argc, char** argv);
void ddp_finalize(ddp_context_t* ctx);

void ddp_all_reduce(float* tensor, size_t numel, ring_reduce_op_t op,
                    ddp_context_t* ctx);
void ddp_broadcast(float* tensor, size_t numel, int root,
                   ddp_context_t* ctx);
void ddp_all_gather(const float* sendbuf, float* recvbuf, size_t numel,
                    ddp_context_t* ctx);

void ring_all_reduce(float* buf, size_t numel, ring_reduce_op_t op,
                     int rank, int world_size);
int  ring_all_reduce_async(float* buf, size_t numel, ring_reduce_op_t op,
                           int rank, int world_size);
void ring_all_reduce_wait(int handle);

void ddp_reduce_scatter(float* send_recv, size_t numel,
                        ring_reduce_op_t op, int rank, int world_size);

bool ddp_gradient_is_finite(const float* grad, size_t numel, int world_size);
void ddp_broadcast_coalesced(float* tensors[], size_t sizes[], int count,
                             int root, ddp_context_t* ctx);

void zero_init(zero_config_t* cfg, zero_stage_t stage);
void* zero_partition_optimizer_state(void* state, size_t total_size,
                                      int rank, int world_size);
void* zero_partition_gradients(float* grads, size_t numel,
                                int rank, int world_size);
void* zero_partition_parameters(float* params, size_t numel,
                                 int rank, int world_size);
void  zero_all_gather_params(float* partitioned, float* full,
                              size_t numel, int rank, int world_size);

void hybrid_3d_init(hybrid_3d_config_t* cfg, int world_size);
int  hybrid_3d_get_tp_group(void);
int  hybrid_3d_get_pp_group(void);
int  hybrid_3d_get_dp_group(void);
void hybrid_3d_pipeline_send(const float* data, size_t numel, int dst);
void hybrid_3d_pipeline_recv(float* data, size_t numel, int src);

void tensor_model_parallel_all_reduce(float* tensor, size_t numel,
                                       int tp_rank, int tp_size);
void tensor_model_parallel_all_gather(const float* sendbuf, float* recvbuf,
                                       size_t numel, int tp_rank, int tp_size);
void tensor_model_parallel_split(const float* tensor, float* shard,
                                  size_t numel, int dim, int tp_rank,
                                  int tp_size);

typedef struct {
    int num_stages;
    int num_micro_batches;
    int stage_id;
    bool use_1f1b_schedule;
} pipeline_config_t;

void pipeline_schedule_1f1b(int stage_id, int num_stages,
                             int num_micro_batches);
void pipeline_flush_schedule(int stage_id, int num_stages,
                              int num_micro_batches);
int  pipeline_next_batch(int stage_id, int step, int num_micro_batches);

size_t allgather_estimate_bandwidth(allgather_config_t* cfg,
                                     size_t tensor_size, int world_size);
void   allgather_ring(float* send_recv, size_t numel,
                       int rank, int world_size);

#ifdef __cplusplus
}
#endif

#endif
