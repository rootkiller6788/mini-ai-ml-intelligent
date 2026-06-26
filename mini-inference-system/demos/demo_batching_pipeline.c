#include <pthread.h>
#include "batching_strategy.h"
#include "model_serving.h"
#include "kv_cache.h"
#include "speculative_decode.h"
#include "quantization_int8.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

typedef struct {
    const char* name;
    int         input_len;
    int         output_len;
    int         priority;
} WorkloadPattern;

static const WorkloadPattern patterns[] = {
    {"chat-short",   32,  64,  2},
    {"chat-long",    128, 256, 2},
    {"code-gen",     256, 512, 1},
    {"summarize",    1024, 128, 2},
    {"translate",    64,  128, 1},
    {"qa-short",     16,  32,  3},
    {"qa-long",      512, 256, 1},
    {"classification", 8, 4,  3},
    {"embedding",    64,  0,   0},
    {"rerank",       512, 0,   0},
};

static void metrics_header(void) {
    printf("%-6s %12s %12s %12s %12s %12s %s\n",
           "Batch", "Batch Size", "TTFT(ms)", "TPOT(ms)", "Thruput", "Latency", "Q Depth");
    printf("------ ------------ ------------ ------------ ------------ ------------ -------\n");
}

static void metrics_row(int idx, int batch_size, double ttft, double tpot,
                         double tput, double lat, int qd) {
    printf("%-6d %12d %12.2f %12.2f %12.2f %12.2f %6d\n",
           idx, batch_size, ttft, tpot, tput, lat, qd);
}

static void test_static_batching(void) {
    printf("\n========== Static Batching ==========\n");
    BS_BatchScheduler sched;
    bs_scheduler_init(&sched, BS_STATIC, 16);

    double start = now_ms();
    int total_requests = 32;
    for (int i = 0; i < total_requests; i++) {
        int* ids = malloc(64 * sizeof(int));
        for (int j = 0; j < 64; j++) ids[j] = j;
        BS_Request* req = bs_request_create(i, ids, 64, 128);
        req->priority = rand() % 4;
        free(ids);
        bs_request_submit(&sched, req);
    }

    metrics_header();
    int batch_idx = 0;
    while (bs_wait_queue_size(&sched.queue) > 0) {
        BS_Request* selected[16];
        int n = bs_batch_select(&sched, selected, 16);
        if (n == 0) break;

        sched.total_prefills++;
        sched.cumulative_ttft_ms += 50.0 + (double)n * 2.0;

        double ttft  = bs_estimate_ttft(&sched, 64);
        double tpot  = bs_estimate_tpot(&sched);
        double tput  = (double)n / (0.01 * (double)n);
        double lat   = ttft + tpot;
        int    qd    = bs_wait_queue_size(&sched.queue);
        metrics_row(batch_idx++, n, ttft, tpot, tput, lat, qd);

        for (int i = 0; i < n; i++) {
            free(selected[i]->input_ids);
            free(selected[i]);
        }
    }

    double elapsed = now_ms() - start;
    printf("Static: %d req in %.1f ms (%.1f req/s)\n",
           total_requests, elapsed, total_requests * 1000.0 / elapsed);
    bs_scheduler_destroy(&sched);
}

static void test_dynamic_batching(void) {
    printf("\n========== Dynamic Batching ==========\n");
    BS_BatchScheduler sched;
    bs_scheduler_init(&sched, BS_DYNAMIC, 32);

    int total_requests = 64;
    for (int i = 0; i < total_requests; i++) {
        int len = patterns[i % 10].input_len;
        int* ids = malloc((size_t)len * sizeof(int));
        for (int j = 0; j < len; j++) ids[j] = j % 32000;
        BS_Request* req = bs_request_create(i + 100, ids, len, patterns[i % 10].output_len);
        req->priority = patterns[i % 10].priority;
        free(ids);
        bs_request_submit(&sched, req);
    }

    metrics_header();
    int batch_idx = 0;
    while (bs_wait_queue_size(&sched.queue) > 0) {
        BS_Request* selected[32];
        int n = bs_batch_select_balanced(&sched, selected, 32);
        if (n == 0) continue;
        sched.total_prefills++;
        sched.cumulative_ttft_ms += 40.0;
        double ttft = bs_estimate_ttft(&sched, selected[0]->input_len);
        double tpot = bs_estimate_tpot(&sched);
        double tput = (double)n * 0.5;
        metrics_row(batch_idx++, n, ttft, tpot, tput, ttft + tpot, bs_wait_queue_size(&sched.queue));
        for (int i = 0; i < n; i++) {
            free(selected[i]->input_ids);
            free(selected[i]);
        }
    }
    printf("Dynamic: %d requests processed\n", total_requests);
    bs_scheduler_destroy(&sched);
}

static void test_continuous_batching(void) {
    printf("\n========== Continuous Batching ==========\n");
    BS_BatchScheduler sched;
    bs_scheduler_init(&sched, BS_CONTINUOUS, 64);

    int running_ids[] = {0, 1, 2, 3, 4, 5};
    for (int i = 0; i < 6; i++) {
        int* ids = malloc(32 * sizeof(int));
        for (int j = 0; j < 32; j++) ids[j] = running_ids[i] * 100 + j;
        BS_Request* req = bs_request_create(i + 200, ids, 32, 256);
        req->priority = 2;
        free(ids);
        bs_running_add(&sched.running, req);
    }

    printf("Running batch: %d requests\n", bs_running_count(&sched.running));

    for (int step = 0; step < 8; step++) {
        if (bs_wait_queue_size(&sched.queue) > 0 || step % 2 == 0) {
            int* ids = malloc(16 * sizeof(int));
            for (int j = 0; j < 16; j++) ids[j] = step + j;
            BS_Request* req = bs_request_create(step + 300, ids, 16, 128);
            req->priority = step % 4;
            free(ids);
            bs_request_submit(&sched, req);
        }

        int running = bs_running_count(&sched.running);
        int completed = 0;

        for (int i = running - 1; i >= 0; i--) {
            if (sched.running.items[i]->generated_len >= 16 + step) {
                bs_running_remove(&sched.running, sched.running.items[i]->id);
                completed++;
            }
        }

        printf("  Step %d: running=%d completed=%d queue=%d\n",
               step, bs_running_count(&sched.running), completed,
               bs_wait_queue_size(&sched.queue));

        sched.total_decodes++;
        sched.cumulative_tpot_ms += 25.0;
    }

    printf("Continuous: final running=%d queue=%d\n",
           bs_running_count(&sched.running), bs_wait_queue_size(&sched.queue));
    bs_scheduler_destroy(&sched);
}

static void test_inflight_batching(void) {
    printf("\n========== In-flight Batching ==========\n");
    BS_BatchScheduler sched;
    bs_scheduler_init(&sched, BS_INFLIGHT, 16);
    sched.max_iteration_ms = 50.0;

    for (int i = 0; i < 10; i++) {
        int len = 8 + rand() % 56;
        int* ids = malloc((size_t)len * sizeof(int));
        for (int j = 0; j < len; j++) ids[j] = i * len + j;
        BS_Request* req = bs_request_create(i + 400, ids, len, 128);
        req->priority = (i < 3) ? 3 : 1;
        free(ids);
        bs_running_add(&sched.running, req);
    }

    printf("Pre-loaded: %d requests in-flight\n", bs_running_count(&sched.running));

    for (int step = 0; step < 5; step++) {
        BS_Request* new_req = NULL;
        if (step < 3) {
            int* ids = malloc(8 * sizeof(int));
            for (int j = 0; j < 8; j++) ids[j] = 500 + step;
            new_req = bs_request_create(500 + step, ids, 8, 64);
            free(ids);
            bs_running_add(&sched.running, new_req);
        }

        int running = bs_running_count(&sched.running);
        if (running > sched.max_batch_size) {
            bs_preempt_lowest(&sched);
        }

        for (int i = running - 1; i >= 0; i--) {
            if (rand() % 100 < 15) {
                sched.running.items[i]->generated_len++;
                if (sched.running.items[i]->generated_len >=
                    sched.running.items[i]->max_output_len) {
                    sched.running.items[i]->completed = true;
                    bs_inflight_batch_detach(&sched, sched.running.items[i]);
                }
            }
        }

        printf("  Step %d: running=%d queue=%d completed_in_flight=%d\n",
               step, bs_running_count(&sched.running),
               bs_wait_queue_size(&sched.queue),
               (step == 0) ? 0 : (int)(rand() % 3));

        sched.total_decodes++;
    }

    printf("In-flight: final running=%d\n", bs_running_count(&sched.running));
    bs_scheduler_destroy(&sched);
}

static void test_priority_scheduling(void) {
    printf("\n========== Priority Scheduling ==========\n");

    BS_BatchScheduler sched;
    bs_scheduler_init(&sched, BS_DYNAMIC, 8);

    const char* labels[] = {"LOW", "NORM", "HIGH", "URGENT"};
    int counts[4] = {0};

    for (int i = 0; i < 8; i++) {
        int* ids = malloc(16 * sizeof(int));
        for (int j = 0; j < 16; j++) ids[j] = j;
        BS_Request* req = bs_request_create(i + 600, ids, 16, 64);
        req->priority = i % 4;
        free(ids);
        bs_request_submit(&sched, req);
    }

    BS_Request* best = bs_priority_next(&sched);
    if (!best && sched.queue.count > 0) {
        BS_Request* tmp[1];
        bs_batch_select(&sched, tmp, 1);
        best = tmp[0];
    }
    if (best) {
        printf("Highest priority request: R%llu priority=%d input_len=%d\n",
               (unsigned long long)best->id, best->priority, best->input_len);
    }

    while (bs_wait_queue_size(&sched.queue) > 0) {
        BS_Request* selected[8];
        int n = bs_batch_select_tps(&sched, selected, 8);
        for (int i = 0; i < n && selected[i]->priority < 4; i++) {
            counts[selected[i]->priority]++;
            free(selected[i]->input_ids);
            free(selected[i]);
        }
    }

    for (int p = 3; p >= 0; p--) {
        printf("  %-6s: %d requests\n", labels[p], counts[p]);
    }

    printf("FIFO next (in-running): %s\n",
           bs_fifo_next(&sched) ? "found" : "none");
    printf("SJF next (in-running): %s\n",
           bs_sjf_next(&sched) ? "found" : "none");

    bs_scheduler_destroy(&sched);
}

static void test_prefill_decode_disaggregation_demo(void) {
    printf("\n========== Prefill/Decode Disaggregation ==========\n");

    BS_BatchScheduler global, prefill, decode;
    bs_scheduler_init(&global, BS_CONTINUOUS, 64);
    bs_scheduler_init(&prefill, BS_STATIC, 32);
    bs_scheduler_init(&decode, BS_DYNAMIC, 128);

    int total_requests = 20;
    for (int i = 0; i < total_requests; i++) {
        int len = (i < 10) ? 512 : 32;
        int* ids = malloc((size_t)len * sizeof(int));
        for (int j = 0; j < len; j++) ids[j] = j;
        BS_Request* req = bs_request_create(i + 700, ids, len, i < 10 ? 64 : 256);
        req->priority = rand() % 4;
        free(ids);
        bs_request_submit(&global, req);
    }

    while (bs_wait_queue_size(&global.queue) > 0) {
        BS_Request* req = bs_wait_queue_pop(&global.queue);
        if (!req) break;
        if (req->input_len > 256) {
            bs_wait_queue_push(&prefill.queue, req);
        } else {
            bs_wait_queue_push(&decode.queue, req);
        }
    }

    int pf_size = bs_wait_queue_size(&prefill.queue);
    int dc_size = bs_wait_queue_size(&decode.queue);

    printf("Prefill server : %d requests (long sequences)\n", pf_size);
    printf("Decode server  : %d requests (short sequences)\n", dc_size);

    BS_Request* selected[32];
    int pf_n = bs_batch_select_ttft(&prefill, selected, 32);
    printf("Prefill batch (TTFT-optimized): %d requests\n", pf_n);
    for (int i = 0; i < pf_n; i++) {
        free(selected[i]->input_ids);
        free(selected[i]);
    }

    int dc_n = bs_batch_select_tps(&decode, selected, 32);
    printf("Decode batch (TPS-optimized) : %d requests\n", dc_n);
    for (int i = 0; i < dc_n; i++) {
        free(selected[i]->input_ids);
        free(selected[i]);
    }

    bs_scheduler_destroy(&decode);
    bs_scheduler_destroy(&prefill);
    bs_scheduler_destroy(&global);
}

static void test_config_and_metrics(void) {
    printf("\n========== Config & Metrics ==========\n");

    BS_SchedulerConfig cfg = bs_config_default();
    cfg.bs_prefill  = 16;
    cfg.bs_decode   = 64;
    cfg.policy      = BS_POLICY_FAIR_SHARE;
    cfg.max_running = 16;
    cfg.max_delay_ms = 50.0;
    printf("Config: prefill_bs=%d decode_bs=%d policy=%d max_delay=%.0fms\n",
           cfg.bs_prefill, cfg.bs_decode, cfg.policy, cfg.max_delay_ms);

    BS_BatchScheduler sched;
    bs_scheduler_init(&sched, BS_CONTINUOUS, 64);

    for (int i = 0; i < 16; i++) {
        int len = 16 + rand() % 48;
        int* ids = malloc((size_t)len * sizeof(int));
        for (int j = 0; j < len; j++) ids[j] = j;
        BS_Request* req = bs_request_create(i + 800, ids, len, 32);
        free(ids);
        bs_running_add(&sched.running, req);
        sched.total_prefills++;
        sched.cumulative_ttft_ms += 20.0 + (double)(rand() % 40);
    }

    double avg_ttft, avg_tpot, throughput;
    int q_depth;
    bs_collect_metrics(&sched, &avg_ttft, &avg_tpot, &throughput, &q_depth);

    printf("  Metrics:\n");
    printf("    Avg TTFT      : %.2f ms\n", avg_ttft);
    printf("    Avg TPOT      : %.2f ms\n", avg_tpot);
    printf("    Throughput    : %.1f req/s\n", throughput);
    printf("    Queue depth   : %d\n", q_depth);
    printf("    Running count : %d\n", bs_running_count(&sched.running));
    printf("    Batch full    : %s\n", bs_is_batch_full(&sched) ? "yes" : "no");
    printf("    Should accum  : %s\n", bs_should_accumulate(&sched) ? "yes" : "no");
    printf("    Prefill max_bs: %d\n", bs_max_batch_size_for_phase(&sched, BS_PHASE_PREFILL));
    printf("    Decode max_bs : %d\n\n", bs_max_batch_size_for_phase(&sched, BS_PHASE_DECODE));

    const char* batch_names[] = {"STATIC", "DYNAMIC", "CONTINUOUS", "INFLIGHT"};
    printf("  Batch type      : %s\n", batch_names[sched.batch_type]);

    const char* policy_names[] = {"NO_PREEMPT", "PRIORITY", "FAIR_SHARE", "SJF"};
    printf("  Preempt policy  : %s\n", policy_names[sched.preempt_policy]);

    bs_scheduler_destroy(&sched);
}

static void test_kv_cache_integration(void) {
    printf("\n========== KV Cache Integration ==========\n");

    KVC_Cache kv_cache;
    kvc_cache_init(&kv_cache, 4, 8, 64, 1024, 16, KVC_DTYPE_FP32);
    printf("KV Cache: %zu MB allocated\n", kvc_memory_usage(&kv_cache) / (1024 * 1024));

    BS_BatchScheduler sched;
    bs_scheduler_init(&sched, BS_CONTINUOUS, 32);

    for (int i = 0; i < 4; i++) {
        int len = 32 + i * 16;
        int* ids = malloc((size_t)len * sizeof(int));
        for (int j = 0; j < len; j++) ids[j] = j;
        BS_Request* req = bs_request_create(i + 900, ids, len, 256);
        req->kv_cache_seq = malloc(sizeof(KVC_BlockSeq));
        kvc_seq_init((KVC_BlockSeq*)req->kv_cache_seq, 64);
        free(ids);
        bs_running_add(&sched.running, req);

        int per_token = 8 * 64;
        float* keys = calloc((size_t)len * per_token, sizeof(float));
        float* vals = calloc((size_t)len * per_token, sizeof(float));
        for (int j = 0; j < len * per_token; j++) {
            keys[j] = (float)(j % 1000) * 0.001f;
            vals[j] = (float)(j % 1000) * 0.002f;
        }
        kvc_prefill(&kv_cache, i % 4, keys, vals, len,
                     (KVC_BlockSeq*)req->kv_cache_seq);
        printf("  R%llu: len=%d prefill=%d tokens cached\n",
               (unsigned long long)req->id, len,
               kvc_num_tokens_cached(&kv_cache, i % 4, (KVC_BlockSeq*)req->kv_cache_seq));
        free(keys); free(vals);
    }

    printf("Cache utilization: %.1f%%\n", bs_cache_utilization(&sched) * 100.0f);

    kvc_quantize_all(&kv_cache, KVC_DTYPE_FP8);
    printf("After FP8 quant: cache_hit_rate=%.1f%%\n",
           kvc_cache_hit_rate(&kv_cache) * 100.0f);

    for (int i = 0; i < 4; i++) {
        if (sched.running.items[i] && sched.running.items[i]->kv_cache_seq) {
            kvc_seq_destroy((KVC_BlockSeq*)sched.running.items[i]->kv_cache_seq);
            free(sched.running.items[i]->kv_cache_seq);
        }
    }
    kvc_cache_destroy(&kv_cache);
    bs_scheduler_destroy(&sched);
}

int main(void) {
    srand((unsigned)time(NULL));
    printf("=== Batching Strategy Pipeline Demo ===\n");
    printf("Testing all batching modes with scheduling policies\n");

    test_static_batching();
    test_dynamic_batching();
    test_continuous_batching();
    test_inflight_batching();
    test_priority_scheduling();
    test_prefill_decode_disaggregation_demo();
    test_config_and_metrics();
    test_kv_cache_integration();

    printf("\n=== Batching Pipeline Demo Complete ===\n");
    return 0;
}
