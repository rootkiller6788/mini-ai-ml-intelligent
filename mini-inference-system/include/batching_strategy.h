#ifndef BATCHING_STRATEGY_H
#define BATCHING_STRATEGY_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#define BS_MAX_BATCH_SIZE        512
#define BS_MAX_SEQ_LEN           32768
#define BS_MAX_QUEUED            4096
#define BS_MAX_ITERATION_TIME_MS 100
#define BS_DEFAULT_MAX_BATCH     32

typedef enum {
    BS_STATIC     = 0,
    BS_DYNAMIC    = 1,
    BS_CONTINUOUS = 2,
    BS_INFLIGHT   = 3,
} BS_BatchType;

typedef enum {
    BS_PHASE_PREFILL = 0,
    BS_PHASE_DECODE  = 1,
    BS_PHASE_MIXED   = 2,
} BS_InferencePhase;

typedef enum {
    BS_POLICY_NO_PREEMPTION   = 0,
    BS_POLICY_PRIORITY_PREEMPT = 1,
    BS_POLICY_FAIR_SHARE      = 2,
    BS_POLICY_SJF             = 3,
} BS_PreemptionPolicy;

typedef enum {
    BS_OBJECTIVE_TTFT   = 0,
    BS_OBJECTIVE_TPS    = 1,
    BS_OBJECTIVE_BALANCED = 2,
} BS_OptimizationObjective;

typedef struct {
    uint64_t id;
    int*     input_ids;
    int      input_len;
    int      max_output_len;
    int      generated_len;
    int      priority;
    double   arrival_time;
    double   first_token_time;
    bool     completed;
    bool     in_batch;
    int      prefill_progress;
    int      batch_slot;
    float*   logits_buffer;
    void*    kv_cache_seq;
    int      output_ids[BS_MAX_SEQ_LEN];
} BS_Request;

typedef struct {
    BS_Request* requests;
    int      count;
    int      capacity;
    int      batch_size;
} BS_Batch;

typedef struct {
    BS_Request** items;
    int      head;
    int      tail;
    int      count;
    int      capacity;
    pthread_mutex_t mutex;
    pthread_cond_t  cond;
} BS_WaitQueue;

typedef struct {
    BS_Request** items;
    int      count;
    int      capacity;
} BS_RunningBatch;

typedef struct {
    BS_BatchType batch_type;
    BS_InferencePhase phase;
    BS_PreemptionPolicy preempt_policy;
    BS_OptimizationObjective objective;

    BS_WaitQueue queue;
    BS_RunningBatch running;
    BS_Batch prefill_batch;
    BS_Batch decode_batch;

    int      max_batch_size;
    int      max_running_batches;
    double   max_queue_delay_ms;
    double   max_iteration_ms;

    int      total_prefills;
    int      total_decodes;
    double   cumulative_ttft_ms;
    double   cumulative_tpot_ms;
    double   total_runtime_ms;
} BS_BatchScheduler;

typedef struct {
    char*    model_name;
    int      bs_prefill;
    int      bs_decode;
    BS_PreemptionPolicy policy;
    int      max_running;
    double   max_delay_ms;
} BS_SchedulerConfig;

typedef void (*BS_InferCallback)(BS_Batch* batch, int phase);

void  bs_scheduler_init(BS_BatchScheduler* sched, BS_BatchType type, int max_batch);
void  bs_scheduler_destroy(BS_BatchScheduler* sched);

bool  bs_request_submit(BS_BatchScheduler* sched, BS_Request* req);
bool  bs_request_cancel(BS_BatchScheduler* sched, uint64_t request_id);
BS_Request* bs_request_create(uint64_t id, const int* input_ids, int input_len,
                               int max_output_len);

void  bs_static_batch(BS_BatchScheduler* sched, BS_InferCallback infer);
void  bs_dynamic_batch(BS_BatchScheduler* sched, BS_InferCallback infer, int max_wait_us);
void  bs_continuous_batch(BS_BatchScheduler* sched, BS_InferCallback infer);

void  bs_inflight_batch_join(BS_BatchScheduler* sched, BS_Batch* new_batch);
void  bs_inflight_batch_detach(BS_BatchScheduler* sched, BS_Request* completed);
void  bs_inflight_batch_step(BS_BatchScheduler* sched, BS_InferCallback infer);

void  bs_prefill_decode_disaggregate(BS_BatchScheduler* sched,
                                      BS_BatchScheduler* prefill_srv,
                                      BS_BatchScheduler* decode_srv,
                                      BS_InferCallback prefill_infer,
                                      BS_InferCallback decode_infer);

void  bs_preempt_by_priority(BS_BatchScheduler* sched, int threshold_priority);
void  bs_preempt_lowest(BS_BatchScheduler* sched);
void  bs_preempt_suspend(BS_BatchScheduler* sched, uint64_t request_id);
void  bs_preempt_resume(BS_BatchScheduler* sched, uint64_t request_id);

BS_Request* bs_priority_next(BS_BatchScheduler* sched);
BS_Request* bs_fifo_next(BS_BatchScheduler* sched);
BS_Request* bs_sjf_next(BS_BatchScheduler* sched);

int   bs_batch_select(BS_BatchScheduler* sched, BS_Request** selected, int max_select);
int   bs_batch_select_ttft(BS_BatchScheduler* sched, BS_Request** selected, int max_select);
int   bs_batch_select_tps(BS_BatchScheduler* sched, BS_Request** selected, int max_select);
int   bs_batch_select_balanced(BS_BatchScheduler* sched, BS_Request** selected, int max_select);

void  bs_wait_queue_push(BS_WaitQueue* q, BS_Request* req);
BS_Request* bs_wait_queue_pop(BS_WaitQueue* q);
BS_Request* bs_wait_queue_peek(const BS_WaitQueue* q);
int   bs_wait_queue_size(const BS_WaitQueue* q);

void  bs_running_add(BS_RunningBatch* rb, BS_Request* req);
void  bs_running_remove(BS_RunningBatch* rb, uint64_t request_id);
int   bs_running_count(const BS_RunningBatch* rb);

bool  bs_is_batch_full(const BS_BatchScheduler* sched);
bool  bs_should_accumulate(const BS_BatchScheduler* sched);
int   bs_max_batch_size_for_phase(const BS_BatchScheduler* sched, BS_InferencePhase phase);

void  bs_merge_batches(BS_Batch* dst, const BS_Batch* src);
void  bs_split_batch(BS_Batch* batch, BS_Request* request, BS_Batch* remainder);

double bs_estimate_ttft(const BS_BatchScheduler* sched, int input_len);
double bs_estimate_tpot(const BS_BatchScheduler* sched);
double bs_cache_utilization(const BS_BatchScheduler* sched);

void  bs_collect_metrics(BS_BatchScheduler* sched, double* avg_ttft, double* avg_tpot,
                          double* throughput, int* queue_depth);

BS_SchedulerConfig bs_config_default(void);
void  bs_config_load(BS_SchedulerConfig* config, const char* path);
void  bs_config_save(const BS_SchedulerConfig* config, const char* path);

#endif
