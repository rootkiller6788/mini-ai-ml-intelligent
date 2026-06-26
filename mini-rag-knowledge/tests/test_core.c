/*
 * mini-rag-knowledge — Core Tests
 *
 * Unit tests for embedding retrieval, chunking, reranking,
 * hybrid search, guardrails.
 */
#include "../include/embedding_retrieve.h"
#include "../include/chunking_strategy.h"
#include "../include/reranking_model.h"
#include "../include/hybrid_search.h"
#include "../include/guardrails_gen.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int tests_run = 0, tests_passed = 0;

#define TEST(name) do { tests_run++; printf("  TEST %s ... ", name); } while(0)
#define PASS() do { tests_passed++; printf("PASS\n"); } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); return 1; } while(0)
#define CHECK(cond, msg) if (!(cond)) FAIL(msg)

/* ── Embedding/Retrieval Tests ── */
static int test_rag_create(void) {
    TEST("rag_create");
    RagConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.embedding_dim = 384;
    cfg.chunk_size = 512;
    cfg.top_k = 5;
    RagContext *r = rag_create(&cfg);
    CHECK(r != NULL, "rag create failed");
    rag_destroy(r);
    PASS();
    return 0;
}

static int test_embed_model_create(void) {
    TEST("embed_model_create");
    EmbeddingModel *m = embed_model_create(256, EMBED_SIM_COSINE);
    CHECK(m != NULL, "embed model create failed");
    CHECK(m->dim == 256, "dim wrong");
    embed_model_destroy(m);
    PASS();
    return 0;
}

static int test_vec_cosine_sim(void) {
    TEST("vec_cosine_sim");
    float a[] = {1, 0, 0}, b[] = {1, 0, 0};
    float s = vec_cosine_sim(a, b, 3);
    CHECK(fabsf(s - 1.0f) < 0.01f, "cosine sim != 1");
    float c[] = {0, 1, 0};
    s = vec_cosine_sim(a, c, 3);
    CHECK(fabsf(s) < 0.01f, "orthogonal cosine != 0");
    PASS();
    return 0;
}

static int test_index_create_add(void) {
    TEST("index_create_add_search");
    VectorIndex *idx = index_create(64, 16, INDEX_COSINE);
    CHECK(idx != NULL, "index create failed");
    float v[16];
    for (int i = 0; i < 16; i++) v[i] = 1.0f;
    index_add(idx, v, 0);
    index_add(idx, v, 1);
    index_build(idx);
    RetrieveResult *res = index_search(idx, v, 2);
    CHECK(res != NULL, "search failed");
    CHECK(res->count > 0, "no results");
    retrieve_destroy(res);
    index_destroy(idx);
    PASS();
    return 0;
}

/* ── Chunking Tests ── */
static int test_chunk_char_split(void) {
    TEST("chunk_char_split");
    const char *text = "Hello world. This is a test. More text here.";
    ChunkList *cl = chunk_split_characters(text, strlen(text), 16, 4);
    CHECK(cl != NULL, "chunk split failed");
    CHECK(cl->count > 0, "zero chunks");
    chunklist_destroy(cl);
    PASS();
    return 0;
}

static int test_count_tokens_approx(void) {
    TEST("count_tokens_approx");
    size_t n = count_tokens_approx("hello world test", 15);
    CHECK(n == 3 || n > 0, "token count wrong");
    PASS();
    return 0;
}

/* ── Reranking Tests ── */
static int test_cross_encoder_create(void) {
    TEST("cross_encoder_create");
    CrossEncoder *ce = cross_encoder_create(64, 32);
    CHECK(ce != NULL, "cross encoder create failed");
    float qe[64], de[64];
    memset(qe, 0, sizeof(qe));
    memset(de, 0, sizeof(de));
    float score = cross_encoder_score(ce, qe, de);
    CHECK(isfinite(score), "score should be finite");
    cross_encoder_destroy(ce);
    PASS();
    return 0;
}

static int test_rrf_score(void) {
    TEST("rrf_score");
    float s = rrf_score(1, 2, 60.0f);
    CHECK(s > 0.0f, "rrf score positive");
    CHECK(s < 1.0f, "rrf score < 1");
    PASS();
    return 0;
}

/* ── Hybrid Search Tests ── */
static int test_bm25_create_add(void) {
    TEST("bm25_create_add");
    BM25Index *bi = bm25_create(128);
    CHECK(bi != NULL, "bm25 create failed");
    bm25_add_document(bi, "hello world", 11);
    bm25_build(bi);
    CHECK(bi->num_docs == 1, "doc count wrong");
    bm25_destroy(bi);
    PASS();
    return 0;
}

static int test_hybrid_results_create(void) {
    TEST("hybrid_results_create");
    HybridResults *hr = hybrid_results_create(32);
    CHECK(hr != NULL, "hybrid results create failed");
    hybrid_results_destroy(hr);
    PASS();
    return 0;
}

/* ── Guardrails Tests ── */
static int test_guard_result_create(void) {
    TEST("guard_result_create");
    GuardResult *gr = guard_result_create();
    CHECK(gr != NULL, "guard result create failed");
    CHECK(gr->passed, "should pass by default");
    guard_result_destroy(gr);
    PASS();
    return 0;
}

static int test_guard_add_violation(void) {
    TEST("guard_add_violation");
    GuardResult *gr = guard_result_create();
    guard_add_violation(gr, GUARD_INPUT_INJECTION, SEVERITY_CRITICAL, "SQL injection", 0, 10, 0.95f);
    CHECK(gr->count == 1, "violation count wrong");
    CHECK(!gr->passed, "should not pass with critical violation");
    guard_result_destroy(gr);
    PASS();
    return 0;
}

int main(void) {
    printf("=== mini-rag-knowledge Unit Tests ===\n\n");

    int failed = 0;
    failed += test_rag_create();
    failed += test_embed_model_create();
    failed += test_vec_cosine_sim();
    failed += test_index_create_add();
    failed += test_chunk_char_split();
    failed += test_count_tokens_approx();
    failed += test_cross_encoder_create();
    failed += test_rrf_score();
    failed += test_bm25_create_add();
    failed += test_hybrid_results_create();
    failed += test_guard_result_create();
    failed += test_guard_add_violation();

    printf("\n=== Results: %d/%d passed, %d failed ===\n",
           tests_passed, tests_run, failed);
    return failed > 0 ? 1 : 0;
}
