/*
 * mini-inference-system — Core Benchmarks
 *
 * Benchmarks: model serving, INT8 quantization, KV cache,
 *             speculative decoding, batching strategies.
 */
#include <pthread.h>
#include "../include/model_serving.h"
#include "../include/quantization_int8.h"
#include "../include/kv_cache.h"
#include "../include/speculative_decode.h"
#include "../include/batching_strategy.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
    return (double)clock() / (double)(CLOCKS_PER_SEC / 1000);
}

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 5000;
    double t0, t1;
    printf("=== mini-inference-system Benchmarks (N=%d) ===\n\n", N);

    /* ── Model Server Init/Destroy ── */
    {
        t0 = now_ms();
        for (int r = 0; r < N / 20; r++) {
            MS_ModelServer server;
            ms_server_init(&server);
            ms_server_destroy(&server);
        }
        t1 = now_ms();
        printf("  ms_server_init:      %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 20, t1 - t0, (t1 - t0) / (double)(N / 20) * 1000.0);
    }

    /* ── Request Create/Push/Pop ── */
    {
        MS_RequestQueue q;
        ms_queue_init(&q, 256);
        t0 = now_ms();
        for (int r = 0; r < N / 20; r++) {
            MS_InferenceRequest *req = ms_request_create(r, 128, 64);
            ms_queue_push(&q, req);
            MS_InferenceRequest *popped = ms_queue_pop(&q);
            ms_request_destroy(popped);
        }
        t1 = now_ms();
        printf("  ms_queue_push+pop:   %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 20, t1 - t0, (t1 - t0) / (double)(N / 20) * 1000.0);
        ms_queue_destroy(&q);
    }

    /* ── INT8 Quantize/Dequantize ── */
    {
        float src[4096];
        int8_t quant[4096];
        float dst[4096];
        for (int i = 0; i < 4096; i++) src[i] = (float)i / 100.0f - 20.0f;
        QI8_TensorQuantParams params = qi8_calib_minmax(src, 4096);
        t0 = now_ms();
        for (int r = 0; r < N / 10; r++) {
            qi8_quantize_per_tensor(src, quant, 4096, params);
            qi8_dequantize_per_tensor(quant, dst, 4096, params);
        }
        t1 = now_ms();
        printf("  qi8_quant+dequant:   %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 10, t1 - t0, (t1 - t0) / (double)(N / 10) * 1000.0);
    }

    /* ── INT8 Calibration ── */
    {
        float data[4096];
        for (int i = 0; i < 4096; i++) data[i] = (float)(rand() % 1000) / 100.0f - 5.0f;
        t0 = now_ms();
        for (int r = 0; r < N / 10; r++) {
            qi8_calib_minmax(data, 4096);
            qi8_calib_mse(data, 4096, 256);
        }
        t1 = now_ms();
        printf("  qi8_calib_minmax+mse:%d ops in %.1f ms  (%.1f µs/op)\n",
               N / 10, t1 - t0, (t1 - t0) / (double)(N / 10) * 1000.0);
    }

    /* ── KV Cache Init/Destroy ── */
    {
        t0 = now_ms();
        for (int r = 0; r < N / 20; r++) {
            KVC_Cache cache;
            kvc_cache_init(&cache, 8, 16, 64, 4096, 16, KVC_DTYPE_FP32);
            kvc_cache_destroy(&cache);
        }
        t1 = now_ms();
        printf("  kvc_cache_init:      %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 20, t1 - t0, (t1 - t0) / (double)(N / 20) * 1000.0);
    }

    /* ── KV Cache Block Ops ── */
    {
        KVC_Cache cache;
        kvc_cache_init(&cache, 4, 8, 64, 1024, 16, KVC_DTYPE_FP32);
        t0 = now_ms();
        for (int r = 0; r < N / 10; r++) {
            int blk = kvc_block_alloc(&cache, 0);
            kvc_block_free(&cache, 0, blk);
        }
        t1 = now_ms();
        printf("  kvc_block_alloc+free:%d ops in %.1f ms  (%.1f µs/op)\n",
               N / 10, t1 - t0, (t1 - t0) / (double)(N / 10) * 1000.0);
        kvc_cache_destroy(&cache);
    }

    /* ── NGram Predict ── */
    {
        SD_NGramModel model;
        sd_draft_ngram_init(&model, 3);
        int tokens[] = {1, 2, 3, 4, 5};
        sd_draft_ngram_update(&model, tokens, 5);
        int candidates[4];
        t0 = now_ms();
        for (int r = 0; r < N / 10; r++) {
            sd_draft_ngram_predict(&model, tokens, 5, candidates, 4);
        }
        t1 = now_ms();
        printf("  ngram_predict:       %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 10, t1 - t0, (t1 - t0) / (double)(N / 10) * 1000.0);
        sd_draft_ngram_destroy(&model);
    }

    /* ── Batching Scheduler Init ── */
    {
        t0 = now_ms();
        for (int r = 0; r < N / 20; r++) {
            BS_BatchScheduler sched;
            bs_scheduler_init(&sched, BS_DYNAMIC, 32);
            bs_scheduler_destroy(&sched);
        }
        t1 = now_ms();
        printf("  bs_scheduler_init:   %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 20, t1 - t0, (t1 - t0) / (double)(N / 20) * 1000.0);
    }

    /* ── Request Submit ── */
    {
        BS_BatchScheduler sched;
        bs_scheduler_init(&sched, BS_DYNAMIC, 32);
        t0 = now_ms();
        for (int r = 0; r < N / 20; r++) {
            int input[] = {1, 2, 3};
            BS_Request *req = bs_request_create(r, input, 3, 64);
            bs_request_submit(&sched, req);
            free(req);
        }
        t1 = now_ms();
        printf("  bs_request_submit:   %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 20, t1 - t0, (t1 - t0) / (double)(N / 20) * 1000.0);
        bs_scheduler_destroy(&sched);
    }

    printf("\nDone.\n");
    return 0;
}
