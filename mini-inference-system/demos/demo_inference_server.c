#include "model_serving.h"
#include "kv_cache.h"
#include "batching_strategy.h"
#include "speculative_decode.h"
#include "quantization_int8.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define DEMO_VOCAB     32000
#define DEMO_NUM_HEADS 8
#define DEMO_HEAD_DIM  64
#define DEMO_NUM_LAYERS 4
#define DEMO_MAX_SEQ   512
#define DEMO_BLOCK     16

typedef struct {
    MS_ModelServer server;
    KVC_Cache kv_cache;
    BS_BatchScheduler scheduler;
    SD_SpeculativeDecoder spec_dec;
    int total_requests;
    int completed_requests;
    double total_latency_ms;
    double total_ttft_ms;
    double total_tokens;
} DemoServer;

static double demo_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static void demo_infer_callback(BS_Batch* batch, int phase) {
    const char* phase_name = (phase == BS_PHASE_PREFILL) ? "prefill" :
                              (phase == BS_PHASE_DECODE)  ? "decode"  : "mixed";
    printf("  [infer] phase=%s batch_size=%d\n", phase_name, batch->count);
    for (int i = 0; i < batch->count; i++) {
        batch->requests[i].output_ids[batch->requests[i].generated_len++] = 42;
    }
}

static void demo_init(DemoServer* ds) {
    memset(ds, 0, sizeof(DemoServer));
    ms_server_init(&ds->server);
    kvc_cache_init(&ds->kv_cache, DEMO_NUM_LAYERS, DEMO_NUM_HEADS, DEMO_HEAD_DIM,
                    DEMO_MAX_SEQ, DEMO_BLOCK, KVC_DTYPE_FP32);
    bs_scheduler_init(&ds->scheduler, BS_CONTINUOUS, 32);
    sd_decoder_init(&ds->spec_dec, SD_DRAFT_NGRAM, 5);
    ds->kv_cache.quantize_cache = false;
}

static void demo_destroy(DemoServer* ds) {
    sd_decoder_destroy(&ds->spec_dec);
    bs_scheduler_destroy(&ds->scheduler);
    kvc_cache_destroy(&ds->kv_cache);
    ms_server_destroy(&ds->server);
}

static void demo_load_model(DemoServer* ds) {
    printf("--- Loading model ---\n");
    MS_ModelConfig config;
    ms_config_load(&config, "demo_repo/llama-3.8B");
    config.instance_count = 2;
    config.dynamic_batching_enabled = true;
    config.max_queue_delay_us = 50;
    config.schedule_policy = MS_SCHED_PRIORITY;
    char* pbtext = ms_config_to_protobuf_text(&config);
    printf("%s\n", pbtext);
    free(pbtext);

    for (int v = 1; v <= 3; v++) {
        ms_model_load(&ds->server, "llama-3.8b", v);
        printf("  Model version %d loaded\n", v);
        ms_model_unload(&ds->server, "llama-3.8b");
    }
    ms_model_load(&ds->server, "llama-3.8b", 3);
    printf("  Final: version 3 active\n");
}

static void demo_submit_requests(DemoServer* ds, int count) {
    printf("--- Submitting %d requests ---\n", count);
    const char* priority_names[] = {"LOW", "NORMAL", "HIGH", "URGENT"};
    for (int i = 0; i < count; i++) {
        size_t in_len  = 16 + (size_t)(rand() % 48);
        size_t out_len = 1;
        MS_InferenceRequest* req = ms_request_create(i, in_len, out_len);
        req->priority = (MS_Priority)(rand() % 4);
        for (size_t j = 0; j < in_len; j++) {
            req->input_data[j] = (float)(rand() % DEMO_VOCAB) / 1000.0f;
        }
        ms_request_submit(&ds->server, req);
        printf("  R%-2d: priority=%s input_len=%zu\n",
               i, priority_names[req->priority], in_len);
        ds->total_requests++;
    }
}

static void demo_batch_serve(DemoServer* ds) {
    printf("--- Batch serving ---\n");
    BS_Request* bs_req[8];
    int input_lens[] = {32, 64, 16, 48, 128, 8, 56, 24};

    for (int i = 0; i < 8; i++) {
        int* ids = malloc(input_lens[i] * sizeof(int));
        for (int j = 0; j < input_lens[i]; j++) ids[j] = rand() % DEMO_VOCAB;
        bs_req[i] = bs_request_create(i * 100, ids, input_lens[i], 256);
        free(ids);
    }

    for (int i = 0; i < 8; i++) {
        bs_request_submit(&ds->scheduler, bs_req[i]);
    }

    printf("  Queue depth: %d\n", bs_wait_queue_size(&ds->scheduler.queue));

    BS_Request* selected[32];
    int n = bs_batch_select_ttft(&ds->scheduler, selected, 4);
    printf("  TTFT-optimized selection: %d requests\n", n);
    for (int i = 0; i < n; i++) {
        printf("    R%llu: len=%d\n",
               (unsigned long long)selected[i]->id, selected[i]->input_len);
    }

    n = bs_batch_select_tps(&ds->scheduler, selected, 3);
    printf("  TPS-optimized selection: %d requests\n", n);

    BS_Batch batch;
    batch.requests  = malloc(32 * sizeof(BS_Request));
    batch.count     = 4;
    batch.capacity  = 32;
    batch.batch_size = 4;
    demo_infer_callback(&batch, BS_PHASE_MIXED);
    free(batch.requests);

    bs_continuous_batch(&ds->scheduler, demo_infer_callback);
    printf("  Queue depth after continuous batch: %d\n",
           bs_wait_queue_size(&ds->scheduler.queue));
}

static void demo_kv_cache_workflow(DemoServer* ds) {
    printf("--- KV Cache workflow ---\n");
    KVC_BlockSeq seq;
    kvc_seq_init(&seq, 64);

    int per_token = DEMO_NUM_HEADS * DEMO_HEAD_DIM;
    float* keys   = malloc(256 * per_token * sizeof(float));
    float* values = malloc(256 * per_token * sizeof(float));
    float* query  = malloc(per_token * sizeof(float));
    float* output = malloc(per_token * sizeof(float));

    for (int i = 0; i < 256 * per_token; i++) {
        keys[i]   = cosf((float)i * 0.001f);
        values[i] = sinf((float)i * 0.001f) * 0.5f;
    }
    for (int i = 0; i < per_token; i++) {
        query[i] = (float)(i % 37) * 0.01f;
    }

    for (int l = 0; l < DEMO_NUM_LAYERS; l++) {
        kvc_prefill(&ds->kv_cache, l, keys, values, 128, &seq);
    }
    printf("  Prefill 128 tokens: blocks=%d seq_len=%d\n", seq.num_blocks, seq.seq_len);

    for (int step = 0; step < 64; step++) {
        for (int l = 0; l < DEMO_NUM_LAYERS; l++) {
            kvc_decode(&ds->kv_cache, l, query, query, &seq);
        }
    }
    printf("  Decode 64 steps: blocks=%d seq_len=%d\n", seq.num_blocks, seq.seq_len);

    kvc_attention(&ds->kv_cache, 0, query, 1, &seq, output);
    printf("  Attention result: [0]=%.4f [mid]=%.4f [last]=%.4f\n",
           output[0], output[per_token / 2], output[per_token - 1]);

    kvc_sliding_window_attention(&ds->kv_cache, 0, query, 1, &seq, 32, output);
    printf("  Sliding window (w=32): [0]=%.4f\n", output[0]);

    kvc_evict_oldest(&ds->kv_cache, 0, &seq, 2);
    printf("  After eviction: blocks=%d seq_len=%d\n", seq.num_blocks, seq.seq_len);

    kvc_paged_attention(&ds->kv_cache, 0, query, 1, &seq, output);
    printf("  PagedAttention: [0]=%.4f\n", output[0]);

    kvc_seq_destroy(&seq);
    free(keys); free(values); free(query); free(output);
}

static void demo_quantization(DemoServer* ds) {
    printf("--- Quantization demo ---\n");
    (void)ds;
    int tensor_size = 4096;
    float* data = malloc((size_t)tensor_size * sizeof(float));
    int8_t* quant = malloc((size_t)tensor_size * sizeof(int8_t));
    float* dequant = malloc((size_t)tensor_size * sizeof(float));

    for (int i = 0; i < tensor_size; i++) {
        data[i] = tanhf((float)i * 0.001f) * 10.0f;
    }

    QI8_TensorQuantParams params = qi8_calib_minmax(data, tensor_size);
    qi8_quantize_per_tensor(data, quant, tensor_size, params);
    qi8_dequantize_per_tensor(quant, dequant, tensor_size, params);

    float mse = qi8_measure_mse(data, dequant, tensor_size);
    printf("  INT8 per-tensor: MSE=%.6f scale=%.6f\n", mse, params.scale);

    QI8_TensorQuantParams* ch_params = malloc(64 * sizeof(QI8_TensorQuantParams));
    for (int c = 0; c < 64; c++) {
        qi8_symmetric_params(data + c * 64, 64, &ch_params[c].scale);
        ch_params[c].zero_point = 0;
    }
    qi8_quantize_per_channel(data, quant, 64, 64, ch_params);
    qi8_dequantize_per_channel(quant, dequant, 64, 64, ch_params);
    mse = qi8_measure_mse(data, dequant, tensor_size);
    printf("  INT8 per-channel: MSE=%.6f\n", mse);

    qi8_fused_quant_relu(data, quant, tensor_size, params);
    qi8_dequantize_per_tensor(quant, dequant, tensor_size, params);
    mse = qi8_measure_mse(data, dequant, tensor_size);
    printf("  Fused ReLU+Quant: MSE=%.6f\n", mse);

    qi8_fused_quant_gelu(data, quant, tensor_size, params);
    printf("  Fused GELU+Quant: done\n");

    int8_t A[64] = {1}, B[64] = {2};
    int32_t C[64] = {0};
    qi8_gemm_int8(A, B, C, 8, 8, 8, 0.01f, 0.01f, 0.001f, NULL);
    printf("  INT8 GEMM 8x8x8: C[0]=%d C[63]=%d\n", C[0], C[63]);

    printf("  Cache hit rate: %.1f%%\n", kvc_cache_hit_rate(&ds->kv_cache) * 100.0f);

    free(data); free(quant); free(dequant); free(ch_params);
}

static void demo_speculative_decode(DemoServer* ds) {
    printf("--- Speculative decoding demo ---\n");
    ds->spec_dec.target.vocab_size = DEMO_VOCAB;
    ds->spec_dec.target.max_seq_len = 2048;
    ds->spec_dec.gamma = 5;

    int prefix[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int prefix_len = 10;
    int output[SD_MAX_CANDIDATES * 10];
    double start = demo_time_ms();

    for (int iter = 0; iter < 20; iter++) {
        int n = sd_speculative_step(&ds->spec_dec, prefix, prefix_len,
                                     output + iter * SD_MAX_CANDIDATES, SD_MAX_CANDIDATES);
        printf("  Step %d: generated %d tokens\n", iter, n);
        ds->total_tokens += n;
    }

    double elapsed = demo_time_ms() - start;
    printf("  20 speculative steps: %.2f ms total\n", elapsed);
    printf("  Tokens generated: %d\n", ds->spec_dec.total_tokens_generated);
    printf("  Draft acceptance rate: %.1f%%\n",
           ds->spec_dec.draft.acceptance_rate * 100.0f);
    printf("  Estimated speedup: %.2fx\n", sd_speedup_estimate(&ds->spec_dec));

    SD_SpeculativeDecoder medusa_dec;
    sd_decoder_init(&medusa_dec, SD_DRAFT_MEDUSA, 4);
    medusa_dec.target.vocab_size = DEMO_VOCAB;
    int medusa_out[32];
    sd_speculative_step(&medusa_dec, prefix, prefix_len, medusa_out, 8);
    printf("  Medusa draft: %d heads\n", medusa_dec.draft.medusa.num_heads);
    sd_decoder_destroy(&medusa_dec);
}

static void demo_prefill_decode_disaggregation(DemoServer* ds) {
    printf("--- Prefill/Decode disaggregation ---\n");
    BS_BatchScheduler prefill_srv, decode_srv;
    bs_scheduler_init(&prefill_srv, BS_STATIC, 8);
    bs_scheduler_init(&decode_srv, BS_DYNAMIC, 32);

    for (int i = 0; i < 6; i++) {
        int* ids = malloc(64 * sizeof(int));
        for (int j = 0; j < 64; j++) ids[j] = j;
        BS_Request* req = bs_request_create(i + 500, ids, 64, 128);
        free(ids);
        bs_request_submit(&ds->scheduler, req);
    }

    bs_prefill_decode_disaggregate(&ds->scheduler, &prefill_srv, &decode_srv,
                                    demo_infer_callback, demo_infer_callback);
    printf("  Prefill server queue: %d\n", bs_wait_queue_size(&prefill_srv.queue));
    printf("  Decode server queue: %d\n", bs_wait_queue_size(&decode_srv.queue));

    bs_scheduler_destroy(&prefill_srv);
    bs_scheduler_destroy(&decode_srv);
}

static void demo_preemption(DemoServer* ds) {
    printf("--- Priority preemption ---\n");
    for (int i = 0; i < 4; i++) {
        int* ids = malloc(16 * sizeof(int));
        for (int j = 0; j < 16; j++) ids[j] = j;
        BS_Request* req = bs_request_create(i + 600, ids, 16, 64);
        req->priority = (i < 2) ? 3 : 0;
        free(ids);
        bs_running_add(&ds->scheduler.running, req);
    }
    printf("  Running before preempt: %d\n", bs_running_count(&ds->scheduler.running));
    bs_preempt_by_priority(&ds->scheduler, 2);
    printf("  Running after preempt (pri < 2 evicted): %d\n", bs_running_count(&ds->scheduler.running));
    printf("  Queue depth: %d\n", bs_wait_queue_size(&ds->scheduler.queue));
}

static void demo_metrics(DemoServer* ds) {
    printf("--- Metrics ---\n");
    double avg_ttft, avg_tpot, throughput;
    int q_depth;
    bs_collect_metrics(&ds->scheduler, &avg_ttft, &avg_tpot, &throughput, &q_depth);
    printf("  Avg TTFT: %.2f ms\n", avg_ttft);
    printf("  Avg TPOT: %.2f ms\n", avg_tpot);
    printf("  Throughput: %.1f tok/s\n", throughput);
    printf("  Queue depth: %d\n", q_depth);
    printf("  KV cache mem: %zu MB\n", kvc_memory_usage(&ds->kv_cache) / (1024 * 1024));
    printf("  Cache hit rate: %.1f%%\n", kvc_cache_hit_rate(&ds->kv_cache) * 100.0f);

    double ttft_est = bs_estimate_ttft(&ds->scheduler, 128);
    double tpot_est = bs_estimate_tpot(&ds->scheduler);
    printf("  Est TTFT (128 tok): %.2f ms\n", ttft_est);
    printf("  Est TPOT: %.2f ms\n", tpot_est);
    printf("  Cache utilization: %.1f%%\n",
           bs_cache_utilization(&ds->scheduler) * 100.0f);
}

int main(void) {
    srand((unsigned)time(NULL));

    DemoServer ds;
    demo_init(&ds);

    demo_load_model(&ds);
    demo_submit_requests(&ds, 6);
    demo_batch_serve(&ds);
    demo_kv_cache_workflow(&ds);
    demo_quantization(&ds);
    demo_speculative_decode(&ds);
    demo_prefill_decode_disaggregation(&ds);
    demo_preemption(&ds);
    demo_metrics(&ds);

    printf("\n=== Demo complete ===\n");
    printf("Total served: %d requests\n", ds.total_requests);
    printf("Total tokens: %.0f\n", ds.total_tokens);
    printf("Avg acceptance: %.1f%%\n", ds.spec_dec.draft.acceptance_rate * 100.0f);

    demo_destroy(&ds);
    return 0;
}
