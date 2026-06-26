#include "batching_strategy.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define _GNU_SOURCE
#include <pthread.h>

static double bs_get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

void bs_scheduler_init(BS_BatchScheduler* sched, BS_BatchType type, int max_batch) {
    memset(sched, 0, sizeof(BS_BatchScheduler));
    sched->batch_type     = type;
    sched->phase          = BS_PHASE_MIXED;
    sched->preempt_policy = BS_POLICY_PRIORITY_PREEMPT;
    sched->objective      = BS_OBJECTIVE_BALANCED;

    sched->queue.items    = malloc(BS_MAX_QUEUED * sizeof(BS_Request*));
    sched->queue.head     = 0;
    sched->queue.tail     = 0;
    sched->queue.count    = 0;
    sched->queue.capacity = BS_MAX_QUEUED;
    pthread_mutex_init(&sched->queue.mutex, NULL);
    pthread_cond_init(&sched->queue.cond, NULL);

    sched->running.items   = malloc(BS_MAX_BATCH_SIZE * sizeof(BS_Request*));
    sched->running.count   = 0;
    sched->running.capacity = BS_MAX_BATCH_SIZE;

    sched->prefill_batch.requests = malloc(BS_MAX_BATCH_SIZE * sizeof(BS_Request));
    sched->prefill_batch.count    = 0;
    sched->prefill_batch.capacity = BS_MAX_BATCH_SIZE;
    sched->prefill_batch.batch_size = max_batch / 2;

    sched->decode_batch.requests  = malloc(BS_MAX_BATCH_SIZE * sizeof(BS_Request));
    sched->decode_batch.count     = 0;
    sched->decode_batch.capacity  = BS_MAX_BATCH_SIZE;
    sched->decode_batch.batch_size = max_batch;

    sched->max_batch_size       = max_batch;
    sched->max_running_batches  = 8;
    sched->max_queue_delay_ms   = 100.0;
    sched->max_iteration_ms     = BS_MAX_ITERATION_TIME_MS;
}

void bs_scheduler_destroy(BS_BatchScheduler* sched) {
    free(sched->queue.items);
    free(sched->running.items);
    free(sched->prefill_batch.requests);
    free(sched->decode_batch.requests);
    pthread_mutex_destroy(&sched->queue.mutex);
    pthread_cond_destroy(&sched->queue.cond);
}

BS_Request* bs_request_create(uint64_t id, const int* input_ids, int input_len,
                               int max_output_len) {
    BS_Request* req = malloc(sizeof(BS_Request));
    req->id              = id;
    req->input_ids       = malloc((size_t)input_len * sizeof(int));
    memcpy(req->input_ids, input_ids, (size_t)input_len * sizeof(int));
    req->input_len       = input_len;
    req->max_output_len  = max_output_len;
    req->generated_len   = 0;
    req->priority        = 1;
    req->arrival_time    = bs_get_time_ms();
    req->first_token_time= 0.0;
    req->completed       = false;
    req->in_batch        = false;
    req->prefill_progress = 0;
    req->batch_slot      = -1;
    req->logits_buffer   = NULL;
    req->kv_cache_seq    = NULL;
    return req;
}

void bs_wait_queue_push(BS_WaitQueue* q, BS_Request* req) {
    pthread_mutex_lock(&q->mutex);
    while (q->count >= q->capacity) {
        pthread_cond_wait(&q->cond, &q->mutex);
    }
    q->items[q->tail] = req;
    q->tail = (q->tail + 1) % q->capacity;
    q->count++;
    pthread_cond_signal(&q->cond);
    pthread_mutex_unlock(&q->mutex);
}

BS_Request* bs_wait_queue_pop(BS_WaitQueue* q) {
    pthread_mutex_lock(&q->mutex);
    while (q->count == 0) {
        pthread_cond_wait(&q->cond, &q->mutex);
    }
    BS_Request* req = q->items[q->head];
    q->head = (q->head + 1) % q->capacity;
    q->count--;
    pthread_cond_signal(&q->cond);
    pthread_mutex_unlock(&q->mutex);
    return req;
}

BS_Request* bs_wait_queue_peek(const BS_WaitQueue* q) {
    if (q->count == 0) return NULL;
    return q->items[q->head];
}

int bs_wait_queue_size(const BS_WaitQueue* q) {
    return q->count;
}

bool bs_request_submit(BS_BatchScheduler* sched, BS_Request* req) {
    bs_wait_queue_push(&sched->queue, req);
    return true;
}

bool bs_request_cancel(BS_BatchScheduler* sched, uint64_t request_id) {
    pthread_mutex_lock(&sched->queue.mutex);
    for (int i = 0; i < sched->queue.count; i++) {
        int idx = (sched->queue.head + i) % sched->queue.capacity;
        if (sched->queue.items[idx]->id == request_id) {
            BS_Request* req = sched->queue.items[idx];
            for (int j = i; j < sched->queue.count - 1; j++) {
                int from = (sched->queue.head + j + 1) % sched->queue.capacity;
                int to   = (sched->queue.head + j) % sched->queue.capacity;
                sched->queue.items[to] = sched->queue.items[from];
            }
            sched->queue.count--;
            free(req->input_ids);
            free(req);
            pthread_mutex_unlock(&sched->queue.mutex);
            return true;
        }
    }
    pthread_mutex_unlock(&sched->queue.mutex);
    return false;
}

int bs_batch_select(BS_BatchScheduler* sched, BS_Request** selected, int max_select) {
    switch (sched->objective) {
    case BS_OBJECTIVE_TTFT:   return bs_batch_select_ttft(sched, selected, max_select);
    case BS_OBJECTIVE_TPS:    return bs_batch_select_tps(sched, selected, max_select);
    case BS_OBJECTIVE_BALANCED: return bs_batch_select_balanced(sched, selected, max_select);
    default: break;
    }
    int count = 0;
    while (count < max_select && sched->queue.count > 0) {
        selected[count++] = bs_wait_queue_pop(&sched->queue);
    }
    return count;
}

int bs_batch_select_ttft(BS_BatchScheduler* sched, BS_Request** selected, int max_select) {
    int count = 0;
    int best_idx = 0;
    double best_arrival = 1e18;
    for (int r = 0; r < max_select && sched->queue.count > 0; r++) {
        best_idx = 0;
        best_arrival = 1e18;
        for (int i = 0; i < sched->queue.count; i++) {
            int idx = (sched->queue.head + i) % sched->queue.capacity;
            if (sched->queue.items[idx]->arrival_time < best_arrival) {
                best_arrival = sched->queue.items[idx]->arrival_time;
                best_idx = i;
            }
        }
        int src = (sched->queue.head + best_idx) % sched->queue.capacity;
        selected[count] = sched->queue.items[src];
        for (int j = best_idx; j < sched->queue.count - 1; j++) {
            int from = (sched->queue.head + j + 1) % sched->queue.capacity;
            int to   = (sched->queue.head + j) % sched->queue.capacity;
            sched->queue.items[to] = sched->queue.items[from];
        }
        sched->queue.count--;
        count++;
    }
    return count;
}

int bs_batch_select_tps(BS_BatchScheduler* sched, BS_Request** selected, int max_select) {
    int count = 0;
    while (count < max_select && sched->queue.count > 0) {
        int best = 0;
        int best_priority = -1;
        for (int i = 0; i < sched->queue.count; i++) {
            int idx = (sched->queue.head + i) % sched->queue.capacity;
            if (sched->queue.items[idx]->priority > best_priority) {
                best_priority = sched->queue.items[idx]->priority;
                best = i;
            }
        }
        int src = (sched->queue.head + best) % sched->queue.capacity;
        selected[count++] = sched->queue.items[src];
        for (int j = best; j < sched->queue.count - 1; j++) {
            int from = (sched->queue.head + j + 1) % sched->queue.capacity;
            int to   = (sched->queue.head + j) % sched->queue.capacity;
            sched->queue.items[to] = sched->queue.items[from];
        }
        sched->queue.count--;
    }
    return count;
}

int bs_batch_select_balanced(BS_BatchScheduler* sched, BS_Request** selected, int max_select) {
    int count = 0;
    while (count < max_select && sched->queue.count > 0) {
        selected[count++] = bs_wait_queue_pop(&sched->queue);
    }
    return count;
}

void bs_running_add(BS_RunningBatch* rb, BS_Request* req) {
    if (rb->count >= rb->capacity) return;
    rb->items[rb->count++] = req;
    req->in_batch = true;
    req->batch_slot = rb->count - 1;
}

void bs_running_remove(BS_RunningBatch* rb, uint64_t request_id) {
    for (int i = 0; i < rb->count; i++) {
        if (rb->items[i]->id == request_id) {
            rb->items[i]->completed = true;
            rb->items[i]->in_batch = false;
            for (int j = i; j < rb->count - 1; j++) {
                rb->items[j] = rb->items[j + 1];
                rb->items[j]->batch_slot = j;
            }
            rb->count--;
            return;
        }
    }
}

int bs_running_count(const BS_RunningBatch* rb) {
    return rb->count;
}

bool bs_is_batch_full(const BS_BatchScheduler* sched) {
    return bs_running_count(&sched->running) >= sched->max_batch_size;
}

bool bs_should_accumulate(const BS_BatchScheduler* sched) {
    return sched->batch_type == BS_DYNAMIC && sched->queue.count < 2;
}

void bs_static_batch(BS_BatchScheduler* sched, BS_InferCallback infer) {
    BS_Batch batch;
    batch.requests  = malloc(BS_MAX_BATCH_SIZE * sizeof(BS_Request));
    batch.count     = 0;
    batch.capacity  = BS_MAX_BATCH_SIZE;
    batch.batch_size = sched->max_batch_size;

    BS_Request* selected[BS_MAX_BATCH_SIZE];
    int n = bs_batch_select(sched, selected, sched->max_batch_size);
    for (int i = 0; i < n; i++) {
        batch.requests[batch.count++] = *selected[i];
    }

    infer(&batch, BS_PHASE_PREFILL);
    free(batch.requests);
}

void bs_dynamic_batch(BS_BatchScheduler* sched, BS_InferCallback infer, int max_wait_us) {
    struct timespec wait_ts;
    clock_gettime(CLOCK_MONOTONIC, &wait_ts);
    wait_ts.tv_nsec += max_wait_us * 1000;
    if (wait_ts.tv_nsec >= 1000000000L) {
        wait_ts.tv_sec  += wait_ts.tv_nsec / 1000000000L;
        wait_ts.tv_nsec %= 1000000000L;
    }

    pthread_mutex_lock(&sched->queue.mutex);
    int wait_count = 0;
    while (sched->queue.count < 2 && wait_count < max_wait_us) {
        wait_count += 10;
        pthread_cond_signal(&sched->queue.cond);
    }
    pthread_mutex_unlock(&sched->queue.mutex);

    BS_Request* selected[BS_MAX_BATCH_SIZE];
    int n = bs_batch_select(sched, selected, sched->max_batch_size);
    if (n == 0) return;

    BS_Batch batch;
    batch.requests  = malloc(BS_MAX_BATCH_SIZE * sizeof(BS_Request));
    batch.count     = n;
    batch.capacity  = BS_MAX_BATCH_SIZE;
    batch.batch_size = n;
    for (int i = 0; i < n; i++) {
        batch.requests[i] = *selected[i];
    }

    infer(&batch, BS_PHASE_MIXED);
    free(batch.requests);
}

void bs_continuous_batch(BS_BatchScheduler* sched, BS_InferCallback infer) {
    bs_dynamic_batch(sched, infer, 10);
}

void bs_inflight_batch_join(BS_BatchScheduler* sched, BS_Batch* new_batch) {
    (void)sched;
    for (int i = 0; i < new_batch->count; i++) {
        bs_running_add(&sched->running, &new_batch->requests[i]);
    }
}

void bs_inflight_batch_detach(BS_BatchScheduler* sched, BS_Request* completed) {
    (void)sched;
    completed->completed = true;
    completed->in_batch  = false;
}

void bs_inflight_batch_step(BS_BatchScheduler* sched, BS_InferCallback infer) {
    int running_before = bs_running_count(&sched->running);
    if (running_before < sched->max_batch_size) {
        int slots = sched->max_batch_size - running_before;
        BS_Request* selected[BS_MAX_BATCH_SIZE];
        int n = bs_batch_select(sched, selected, slots);
        for (int i = 0; i < n; i++) {
            bs_running_add(&sched->running, selected[i]);
        }
    }

    BS_Batch batch;
    batch.requests  = malloc(BS_MAX_BATCH_SIZE * sizeof(BS_Request));
    batch.count     = bs_running_count(&sched->running);
    batch.capacity  = BS_MAX_BATCH_SIZE;
    batch.batch_size = batch.count;
    for (int i = 0; i < batch.count; i++) {
        batch.requests[i] = *sched->running.items[i];
    }

    infer(&batch, BS_PHASE_MIXED);

    for (int i = batch.count - 1; i >= 0; i--) {
        if (batch.requests[i].completed) {
            bs_running_remove(&sched->running, batch.requests[i].id);
        }
    }
    free(batch.requests);
}

void bs_prefill_decode_disaggregate(BS_BatchScheduler* sched,
                                     BS_BatchScheduler* prefill_srv,
                                     BS_BatchScheduler* decode_srv,
                                     BS_InferCallback prefill_infer,
                                     BS_InferCallback decode_infer) {
    BS_Request* req;
    while ((req = bs_wait_queue_pop(&sched->queue)) != NULL) {
        if (req->input_len > 256) {
            bs_wait_queue_push(&prefill_srv->queue, req);
        } else {
            bs_wait_queue_push(&decode_srv->queue, req);
        }
    }
    prefill_infer(&prefill_srv->prefill_batch, BS_PHASE_PREFILL);
    decode_infer(&decode_srv->decode_batch, BS_PHASE_DECODE);
}

void bs_preempt_by_priority(BS_BatchScheduler* sched, int threshold_priority) {
    for (int i = sched->running.count - 1; i >= 0; i--) {
        if (sched->running.items[i]->priority < threshold_priority) {
            sched->running.items[i]->in_batch = false;
            sched->running.items[i]->prefill_progress = 0;
            bs_wait_queue_push(&sched->queue, sched->running.items[i]);
            for (int j = i; j < sched->running.count - 1; j++) {
                sched->running.items[j] = sched->running.items[j + 1];
            }
            sched->running.count--;
        }
    }
}

void bs_preempt_lowest(BS_BatchScheduler* sched) {
    if (sched->running.count == 0) return;
    int lowest = 0;
    int lowest_pri = 999;
    for (int i = 0; i < sched->running.count; i++) {
        if (sched->running.items[i]->priority < lowest_pri) {
            lowest_pri = sched->running.items[i]->priority;
            lowest = i;
        }
    }
    BS_Request* req = sched->running.items[lowest];
    req->in_batch = false;
    bs_wait_queue_push(&sched->queue, req);
    for (int j = lowest; j < sched->running.count - 1; j++) {
        sched->running.items[j] = sched->running.items[j + 1];
    }
    sched->running.count--;
}

void bs_preempt_suspend(BS_BatchScheduler* sched, uint64_t request_id) {
    bs_running_remove(&sched->running, request_id);
}

void bs_preempt_resume(BS_BatchScheduler* sched, uint64_t request_id) {
    (void)sched; (void)request_id;
}

BS_Request* bs_priority_next(BS_BatchScheduler* sched) {
    if (sched->running.count == 0) return NULL;
    BS_Request* best = sched->running.items[0];
    for (int i = 1; i < sched->running.count; i++) {
        if (sched->running.items[i]->priority > best->priority) {
            best = sched->running.items[i];
        }
    }
    return best;
}

BS_Request* bs_fifo_next(BS_BatchScheduler* sched) {
    if (sched->running.count == 0) return NULL;
    BS_Request* best = sched->running.items[0];
    for (int i = 1; i < sched->running.count; i++) {
        if (sched->running.items[i]->arrival_time < best->arrival_time) {
            best = sched->running.items[i];
        }
    }
    return best;
}

BS_Request* bs_sjf_next(BS_BatchScheduler* sched) {
    if (sched->running.count == 0) return NULL;
    BS_Request* best = sched->running.items[0];
    for (int i = 1; i < sched->running.count; i++) {
        if (sched->running.items[i]->input_len < best->input_len) {
            best = sched->running.items[i];
        }
    }
    return best;
}

void bs_merge_batches(BS_Batch* dst, const BS_Batch* src) {
    for (int i = 0; i < src->count && dst->count < dst->capacity; i++) {
        dst->requests[dst->count++] = src->requests[i];
    }
}

void bs_split_batch(BS_Batch* batch, BS_Request* request, BS_Batch* remainder) {
    remainder->count = 0;
    for (int i = 0; i < batch->count; i++) {
        if (batch->requests[i].id == request->id) {
            batch->count--;
            for (int j = i; j < batch->count; j++) {
                batch->requests[j] = batch->requests[j + 1];
            }
            remainder->requests[remainder->count++] = *request;
            return;
        }
    }
}

double bs_estimate_ttft(const BS_BatchScheduler* sched, int input_len) {
    return 50.0 + (double)input_len * 0.5 + 10.0 * (double)bs_running_count(&sched->running);
}

double bs_estimate_tpot(const BS_BatchScheduler* sched) {
    return 20.0 + 2.0 * (double)bs_running_count(&sched->running);
}

double bs_cache_utilization(const BS_BatchScheduler* sched) {
    return (double)bs_running_count(&sched->running) / (double)sched->max_batch_size;
}

void bs_collect_metrics(BS_BatchScheduler* sched, double* avg_ttft, double* avg_tpot,
                         double* throughput, int* queue_depth) {
    *avg_ttft   = sched->cumulative_ttft_ms / fmax(1.0, (double)sched->total_prefills);
    *avg_tpot   = sched->cumulative_tpot_ms / fmax(1.0, (double)sched->total_decodes);
    *throughput = (double)(sched->total_prefills + sched->total_decodes) /
                   fmax(0.001, sched->total_runtime_ms) * 1000.0;
    *queue_depth = bs_wait_queue_size(&sched->queue);
}

int bs_max_batch_size_for_phase(const BS_BatchScheduler* sched, BS_InferencePhase phase) {
    if (phase == BS_PHASE_PREFILL) return sched->max_batch_size / 2;
    return sched->max_batch_size;
}

BS_SchedulerConfig bs_config_default(void) {
    BS_SchedulerConfig cfg;
    cfg.model_name  = NULL;
    cfg.bs_prefill  = 8;
    cfg.bs_decode   = 32;
    cfg.policy      = BS_POLICY_PRIORITY_PREEMPT;
    cfg.max_running = 8;
    cfg.max_delay_ms = 100.0;
    return cfg;
}

void bs_config_load(BS_SchedulerConfig* config, const char* path) {
    (void)config; (void)path;
}

void bs_config_save(const BS_SchedulerConfig* config, const char* path) {
    (void)config; (void)path;
}
