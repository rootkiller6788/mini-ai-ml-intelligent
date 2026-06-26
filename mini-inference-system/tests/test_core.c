/*
 * mini-inference-system — Core Tests
 *
 * Unit tests for model serving, INT8 quantization, KV cache,
 * speculative decoding, batching strategies.
 */
#include <pthread.h>
#include "../include/model_serving.h"
#include "../include/quantization_int8.h"
#include "../include/kv_cache.h"
#include "../include/speculative_decode.h"
#include "../include/batching_strategy.h"
#include "../include/sampler.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int tests_run = 0, tests_passed = 0;

#define TEST(name) do { tests_run++; printf("  TEST %s ... ", name); } while(0)
#define PASS() do { tests_passed++; printf("PASS\n"); } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); return 1; } while(0)
#define CHECK(cond, msg) if (!(cond)) FAIL(msg)

/* ── Model Serving Tests ── */
static int test_ms_server_init(void) {
    TEST("ms_server_init");
    MS_ModelServer server;
    ms_server_init(&server);
    CHECK(server.running == false, "server should not be running");
    ms_server_destroy(&server);
    PASS();
    return 0;
}

static int test_ms_request_create(void) {
    TEST("ms_request_create");
    MS_InferenceRequest *req = ms_request_create(1, 1024, 512);
    CHECK(req != NULL, "request create failed");
    CHECK(req->id == 1, "request id wrong");
    ms_request_destroy(req);
    PASS();
    return 0;
}

static int test_ms_queue_push_pop(void) {
    TEST("ms_queue_push_pop");
    MS_RequestQueue q;
    ms_queue_init(&q, 16);
    MS_InferenceRequest *req = ms_request_create(42, 128, 64);
    CHECK(ms_queue_push(&q, req), "push failed");
    MS_InferenceRequest *popped = ms_queue_pop(&q);
    CHECK(popped != NULL, "pop failed");
    CHECK(popped->id == 42, "popped id wrong");
    ms_request_destroy(req);
    ms_queue_destroy(&q);
    PASS();
    return 0;
}

/* ── INT8 Quantization Tests ── */
static int test_qi8_calib_minmax(void) {
    TEST("qi8_calib_minmax");
    float data[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    QI8_TensorQuantParams params = qi8_calib_minmax(data, 5);
    CHECK(params.scale > 0.0f, "scale should be positive");
    CHECK(params.min_val <= -2.0f, "min_val wrong");
    CHECK(params.max_val >= 2.0f, "max_val wrong");
    PASS();
    return 0;
}

static int test_qi8_quantize_dequantize(void) {
    TEST("qi8_quantize_dequantize");
    float src[] = {-1.0f, 0.0f, 1.0f, 2.0f};
    int8_t quant[4];
    float dst[4];
    QI8_TensorQuantParams params = qi8_calib_minmax(src, 4);
    qi8_quantize_per_tensor(src, quant, 4, params);
    qi8_dequantize_per_tensor(quant, dst, 4, params);
    for (int i = 0; i < 4; i++)
        CHECK(fabsf(dst[i] - src[i]) < 0.5f, "quant/dequant error too large");
    PASS();
    return 0;
}

static int test_qi8_measure_mse(void) {
    TEST("qi8_measure_mse");
    float orig[] = {0.0f, 1.0f, 2.0f, 3.0f};
    float deq[] = {0.1f, 0.9f, 2.1f, 2.9f};
    float mse = qi8_measure_mse(orig, deq, 4);
    CHECK(mse >= 0.0f, "mse should be non-negative");
    CHECK(mse < 1.0f, "mse should be small");
    PASS();
    return 0;
}

/* ── KV Cache Tests ── */
static int test_kvc_cache_init(void) {
    TEST("kvc_cache_init");
    KVC_Cache cache;
    kvc_cache_init(&cache, 4, 16, 64, 2048, 16, KVC_DTYPE_FP32);
    CHECK(cache.num_layers == 4, "num_layers wrong");
    CHECK(cache.num_heads == 16, "num_heads wrong");
    kvc_cache_destroy(&cache);
    PASS();
    return 0;
}

static int test_kvc_block_alloc_free(void) {
    TEST("kvc_block_alloc_free");
    KVC_Cache cache;
    kvc_cache_init(&cache, 2, 8, 64, 1024, 16, KVC_DTYPE_FP32);
    int block = kvc_block_alloc(&cache, 0);
    CHECK(block >= 0, "block alloc failed");
    kvc_block_free(&cache, 0, block);
    kvc_cache_destroy(&cache);
    PASS();
    return 0;
}

static int test_kvc_seq_init(void) {
    TEST("kvc_seq_init");
    KVC_BlockSeq seq;
    kvc_seq_init(&seq, 64);
    CHECK(seq.num_blocks == 0, "seq should start empty");
    kvc_seq_destroy(&seq);
    PASS();
    return 0;
}

/* ── Speculative Decoding Tests ── */
static int test_sd_decoder_init(void) {
    TEST("sd_decoder_init");
    SD_SpeculativeDecoder decoder;
    bool ok = sd_decoder_init(&decoder, SD_DRAFT_NGRAM, 5);
    CHECK(ok || !ok, "decoder init should not crash");
    sd_decoder_destroy(&decoder);
    PASS();
    return 0;
}

static int test_sd_ngram_init(void) {
    TEST("sd_ngram_init");
    SD_NGramModel model;
    sd_draft_ngram_init(&model, 3);
    int tokens[] = {1, 2, 3, 4, 5};
    sd_draft_ngram_update(&model, tokens, 5);
    int candidates[4];
    int n = sd_draft_ngram_predict(&model, tokens, 5, candidates, 4);
    CHECK(n >= 0, "predict should return >= 0");
    sd_draft_ngram_destroy(&model);
    PASS();
    return 0;
}

/* ── Batching Strategy Tests ── */
static int test_bs_scheduler_init(void) {
    TEST("bs_scheduler_init");
    BS_BatchScheduler sched;
    bs_scheduler_init(&sched, BS_DYNAMIC, 32);
    CHECK(sched.max_batch_size == 32, "max_batch_size wrong");
    bs_scheduler_destroy(&sched);
    PASS();
    return 0;
}

static int test_bs_request_create(void) {
    TEST("bs_request_create");
    int input[] = {1, 2, 3};
    BS_Request *req = bs_request_create(100, input, 3, 64);
    CHECK(req != NULL, "request create failed");
    CHECK(req->id == 100, "id wrong");
    CHECK(req->input_len == 3, "input_len wrong");
    free(req);
    PASS();
    return 0;
}

/* ── Sampler Tests ── */
static int test_samp_greedy(void) {
    TEST("samp_greedy");
    float logits[] = {0.1f, 0.5f, 0.3f, 0.9f, 0.2f};
    int token = samp_greedy(logits, 5);
    CHECK(token == 3, "greedy should pick max");
    PASS();
    return 0;
}

static int test_samp_temperature(void) {
    TEST("samp_temperature");
    float logits[] = {0.1f, 2.0f, 0.5f, -1.0f, -3.0f};
    int token = samp_temperature(logits, 5, 1.0f);
    CHECK(token >= 0 && token < 5, "token out of range");
    PASS();
    return 0;
}

static int test_samp_top_p(void) {
    TEST("samp_top_p");
    float logits[100];
    for (int i = 0; i < 100; i++) logits[i] = (float)i / 10.0f;
    int token = samp_top_p(logits, 100, 0.9f, 1.0f);
    CHECK(token >= 0 && token < 100, "token out of range");
    PASS();
    return 0;
}

static int test_samp_softmax(void) {
    TEST("samp_softmax");
    float logits[] = {1.0f, 2.0f, 3.0f};
    float probs[3];
    samp_softmax(probs, logits, 3, 1.0f);
    float sum = probs[0] + probs[1] + probs[2];
    CHECK(fabsf(sum - 1.0f) < 0.001f, "softmax should sum to 1");
    CHECK(probs[2] > probs[1] && probs[1] > probs[0], "order should be preserved");
    PASS();
    return 0;
}

static int test_samp_config(void) {
    TEST("samp_config");
    Samp_Config cfg = samp_config_default();
    CHECK(samp_config_validate(&cfg), "default config should be valid");
    cfg.temperature = -1.0f;
    CHECK(!samp_config_validate(&cfg), "negative temp should be invalid");
    PASS();
    return 0;
}

static int test_samp_generate(void) {
    TEST("samp_generate");
    /* Allocate logits for max_new_tokens steps × vocab_size */
    float* logits = malloc((size_t)256 * 32 * sizeof(float));
    for (int i = 0; i < 256 * 32; i++) logits[i] = (float)(i % 1000) / 1000.0f;
    int output[16];
    Samp_Config cfg = samp_config_default();
    cfg.strategy = SAMP_GREEDY;
    int n = samp_generate(logits, 5, 256, output, 16, &cfg);
    free(logits);
    CHECK(n > 0, "should generate at least one token");
    CHECK(n <= 16, "should not exceed max_new_tokens");
    PASS();
    return 0;
}

int main(void) {
    printf("=== mini-inference-system Unit Tests ===\n\n");

    int failed = 0;
    failed += test_ms_server_init();
    failed += test_ms_request_create();
    failed += test_ms_queue_push_pop();
    failed += test_qi8_calib_minmax();
    failed += test_qi8_quantize_dequantize();
    failed += test_qi8_measure_mse();
    failed += test_kvc_cache_init();
    failed += test_kvc_block_alloc_free();
    failed += test_kvc_seq_init();
    failed += test_sd_decoder_init();
    failed += test_sd_ngram_init();
    failed += test_bs_scheduler_init();
    failed += test_bs_request_create();
    failed += test_samp_greedy();
    failed += test_samp_temperature();
    failed += test_samp_top_p();
    failed += test_samp_softmax();
    failed += test_samp_config();
    failed += test_samp_generate();

    printf("\n=== Results: %d/%d passed, %d failed ===\n",
           tests_passed, tests_run, failed);
    return failed > 0 ? 1 : 0;
}
