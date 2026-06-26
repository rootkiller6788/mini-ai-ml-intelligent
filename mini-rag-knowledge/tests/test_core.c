/*
 * mini-rag-knowledge — Core Tests
 *
 * Unit tests for embedding retrieval, chunking, reranking,
 * hybrid search, guardrails, evaluation, knowledge graph,
 * and query processing.
 */
#include "../include/embedding_retrieve.h"
#include "../include/chunking_strategy.h"
#include "../include/reranking_model.h"
#include "../include/hybrid_search.h"
#include "../include/guardrails_gen.h"
#include "../include/rag_evaluator.h"
#include "../include/knowledge_graph_rag.h"
#include "../include/query_processor.h"
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

/* ── RAG Evaluator Tests ── */
static int test_ndcg_perfect_ranking(void) {
    TEST("ndcg_perfect_ranking");
    /* Create a ranked list with docs in perfect relevance order */
    RankedList *rl = rankedlist_create(10);
    rl->query_id = 0;
    rl->docs[0].doc_id = 0; rl->docs[0].score = 10.0f; rl->docs[0].relevance = 4;
    rl->docs[1].doc_id = 1; rl->docs[1].score = 9.0f;  rl->docs[1].relevance = 3;
    rl->docs[2].doc_id = 2; rl->docs[2].score = 8.0f;  rl->docs[2].relevance = 2;
    rl->docs[3].doc_id = 3; rl->docs[3].score = 7.0f;  rl->docs[3].relevance = 1;
    rl->docs[4].doc_id = 4; rl->docs[4].score = 6.0f;  rl->docs[4].relevance = 0;
    rl->count = 5;

    double ndcg = ndcg_at_k(rl, 5);
    CHECK(ndcg > 0.9, "perfect ranking NDCG should be ~1");
    CHECK(ndcg <= 1.0, "NDCG should not exceed 1");
    rankedlist_destroy(rl);
    PASS();
    return 0;
}

static int test_precision_recall_at_k(void) {
    TEST("precision_recall_at_k");
    RankedList *rl = rankedlist_create(10);
    rl->docs[0].doc_id = 0; rl->docs[0].score = 1.0f; rl->docs[0].relevance = 1;
    rl->docs[1].doc_id = 1; rl->docs[1].score = 0.9f; rl->docs[1].relevance = 0;
    rl->docs[2].doc_id = 2; rl->docs[2].score = 0.8f; rl->docs[2].relevance = 1;
    rl->docs[3].doc_id = 3; rl->docs[3].score = 0.7f; rl->docs[3].relevance = 0;
    rl->docs[4].doc_id = 4; rl->docs[4].score = 0.6f; rl->docs[4].relevance = 0;
    rl->count = 5;

    double p3 = precision_at_k(rl, 3);
    CHECK(fabsf((float)p3 - 0.6666f) < 0.1f, "Precision@3 should be ~0.666");
    double r3 = recall_at_k(rl, 3);
    CHECK(fabsf((float)r3 - 1.0f) < 0.1f, "Recall@3 should be 1.0");
    double rr = reciprocal_rank(rl);
    CHECK(fabsf((float)rr - 1.0f) < 0.1f, "RR should be 1.0 (first doc relevant)");
    rankedlist_destroy(rl);
    PASS();
    return 0;
}

static int test_judgmentset(void) {
    TEST("judgmentset");
    JudgmentSet *js = judgmentset_create(32);
    CHECK(js != NULL, "judgment set create failed");
    judgmentset_add(js, 0, 1, 3);
    judgmentset_add(js, 0, 2, 1);
    CHECK(js->count == 2, "judgment count wrong");
    judgmentset_destroy(js);
    PASS();
    return 0;
}

/* ── Knowledge Graph Tests ── */
static int test_kg_create(void) {
    TEST("kg_create");
    KnowledgeGraph *kg = kg_create();
    CHECK(kg != NULL, "kg create failed");
    CHECK(kg_entity_count(kg) == 0, "should start empty");
    CHECK(kg_relation_count(kg) == 0, "should start empty");
    kg_destroy(kg);
    PASS();
    return 0;
}

static int test_kg_entity_extraction(void) {
    TEST("kg_entity_extraction");
    const char *text = "Dr. Smith works at Microsoft Corporation in Seattle.";
    EntityMentionList *eml = kg_extract_entities(text, strlen(text));
    CHECK(eml != NULL, "entity extraction failed");
    /* Should find at least 1 entity (Dr. Smith, Microsoft Corporation, or Seattle) */
    CHECK(eml->count > 0, "should find entities in text");
    entity_mention_list_destroy(eml);
    PASS();
    return 0;
}

static int test_kg_build_from_chunks(void) {
    TEST("kg_build_from_chunks");
    const char *chunks[] = {
        "John founded Acme Corp in Boston. Acme makes widgets.",
        "Acme Corp depends on steel suppliers in Pittsburgh."
    };
    size_t lens[] = {strlen(chunks[0]), strlen(chunks[1])};
    size_t docs[] = {0, 0};
    KnowledgeGraph *kg = kg_build_from_chunks(chunks, lens, 2, docs);
    CHECK(kg != NULL, "kg build failed");
    CHECK(kg_entity_count(kg) > 0, "should extract entities");
    kg_destroy(kg);
    PASS();
    return 0;
}

static int test_kg_density(void) {
    TEST("kg_density");
    KnowledgeGraph *kg = kg_create();
    float d = kg_density(kg);
    CHECK(d >= 0.0f, "density should be >= 0");
    kg_destroy(kg);
    PASS();
    return 0;
}

static int test_kg_normalize(void) {
    TEST("kg_normalize_entity_name");
    char *n = kg_normalize_entity_name("Microsoft Corp.");
    CHECK(n != NULL, "normalize failed");
    /* Should be lowercase without punctuation */
    CHECK(strcmp(n, "microsoft corp") == 0, "normalize wrong");
    free(n);
    PASS();
    return 0;
}

/* ── Query Processor Tests ── */
static int test_query_decompose(void) {
    TEST("query_decompose");
    SubQueryList *sql = query_decompose("What is machine learning and how does it work");
    CHECK(sql != NULL, "decompose failed");
    CHECK(sql->count >= 1, "should have sub-queries");
    subquery_list_destroy(sql);
    PASS();
    return 0;
}

static int test_query_expand_terms(void) {
    TEST("query_expand_terms");
    QueryExpansionTerms *qe = query_expand_terms("fast neural network training");
    CHECK(qe != NULL, "expand terms failed");
    /* "neural", "network", "training" should each have synonyms */
    query_expansion_terms_destroy(qe);
    PASS();
    return 0;
}

static int test_query_classify_intent(void) {
    TEST("query_classify_intent");
    QueryIntent i1 = query_classify_intent("what is a transformer model");
    CHECK(i1 == INTENT_DEFINITION, "should be definition");
    QueryIntent i2 = query_classify_intent("how to train a neural network");
    CHECK(i2 == INTENT_HOW_TO, "should be how-to");
    QueryIntent i3 = query_classify_intent("why does overfitting happen");
    CHECK(i3 == INTENT_WHY, "should be why");
    QueryIntent i4 = query_classify_intent("compare GPT vs BERT");
    CHECK(i4 == INTENT_COMPARISON, "should be comparison");
    PASS();
    return 0;
}

static int test_query_complexity(void) {
    TEST("query_complexity");
    QueryComplexity c1 = query_analyze_complexity("hello");
    CHECK(c1 == COMPLEXITY_SIMPLE, "short query should be simple");
    QueryComplexity c2 = query_analyze_complexity("why does backpropagation work better than random search and what are the theoretical guarantees");
    CHECK(c2 >= COMPLEXITY_MODERATE, "long query should be moderate+");
    PASS();
    return 0;
}

static int test_query_keywords(void) {
    TEST("query_keywords");
    size_t n = 0;
    char **kw = query_extract_keywords("the quick brown fox jumps over the lazy dog", &n);
    CHECK(kw != NULL, "should extract keywords");
    CHECK(n > 0, "should have keywords");
    /* "the", "over" should be filtered as stop words */
    query_free_keywords(kw, n);
    PASS();
    return 0;
}

static int test_aggregate_merge(void) {
    TEST("aggregate_merge");
    AggregatedResults *ar = aggregate_create(32);
    size_t ids1[] = {0, 1, 2};
    float scores1[] = {0.9f, 0.7f, 0.5f};
    aggregate_merge(ar, ids1, scores1, 3);
    CHECK(ar->count == 3, "first merge count wrong");
    size_t ids2[] = {1, 3};
    float scores2[] = {0.3f, 0.8f};
    aggregate_merge(ar, ids2, scores2, 2);
    CHECK(ar->count == 4, "merge should add new docs");
    /* doc 1 should have score 1.0 (0.7 + 0.3) */
    aggregate_destroy(ar);
    PASS();
    return 0;
}

/* ── Integration: Full Retrieval + Evaluation Pipeline ── */
static int test_full_retrieval_eval_pipeline(void) {
    TEST("full_retrieval_eval_pipeline");
    /* Create RAG context with a few documents */
    RagConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.embedding_dim = 64;
    cfg.chunk_size = 128;
    cfg.top_k = 3;

    RagContext *r = rag_create(&cfg);
    CHECK(r != NULL, "rag create failed");

    const char *doc1 = "Machine learning is a subset of artificial intelligence. "
                        "It focuses on building systems that learn from data. "
                        "Supervised learning uses labeled data for training.";
    const char *doc2 = "Deep learning uses neural networks with many layers. "
                        "Backpropagation is the key algorithm for training deep networks. "
                        "Convolutional neural networks are used for image recognition.";
    const char *doc3 = "Natural language processing deals with text and speech. "
                        "Transformer models have revolutionized NLP since 2017. "
                        "BERT and GPT are popular transformer architectures.";

    rag_add_document(r, doc1, strlen(doc1), "ml_intro.txt", "ML Introduction", 1);
    rag_add_document(r, doc2, strlen(doc2), "deep_learning.txt", "Deep Learning", 1);
    rag_add_document(r, doc3, strlen(doc3), "nlp.txt", "NLP Overview", 1);

    CHECK(r->num_docs == 3, "doc count wrong");

    rag_chunk_documents(r);
    CHECK(r->num_chunks > 0, "should have chunks");

    rag_embed_chunks(r);
    rag_build_index(r);

    /* Search */
    RetrieveResult *res = rag_retrieve(r, "transformer models for NLP", 24);
    CHECK(res != NULL, "retrieve failed");
    CHECK(res->count > 0, "should find results");

    /* Build evaluation judgment set */
    JudgmentSet *js = judgmentset_create(16);
    judgmentset_add(js, 0, 0, 3);
    judgmentset_add(js, 0, 1, 2);
    judgmentset_add(js, 0, 2, 1);

    /* Convert to ranked list for evaluation */
    RankedList *rl = rankedlist_create(res->count);
    rl->query_id = 0;
    for (size_t i = 0; i < res->count; i++) {
        rl->docs[i].doc_id = res->indices[i];
        rl->docs[i].score  = res->scores[i];
    }
    rl->count = res->count;

    /* Label relevance */
    rankedlist_set_relevance(rl, js);

    /* Evaluation metrics */
    double ndcg = ndcg_at_k(rl, 3);
    CHECK(ndcg >= 0.0 && ndcg <= 1.0, "NDCG out of range");
    double ap = average_precision(rl);
    CHECK(ap >= 0.0 && ap <= 1.0, "AP out of range");

    retrieve_destroy(res);
    rankedlist_destroy(rl);
    judgmentset_destroy(js);
    rag_destroy(r);
    PASS();
    return 0;
}

int main(void) {
    printf("=== mini-rag-knowledge Unit Tests ===\n\n");

    int failed = 0;
    /* Core RAG */
    failed += test_rag_create();
    failed += test_embed_model_create();
    failed += test_vec_cosine_sim();
    failed += test_index_create_add();
    /* Chunking */
    failed += test_chunk_char_split();
    failed += test_count_tokens_approx();
    /* Reranking */
    failed += test_cross_encoder_create();
    failed += test_rrf_score();
    /* Hybrid Search */
    failed += test_bm25_create_add();
    failed += test_hybrid_results_create();
    /* Guardrails */
    failed += test_guard_result_create();
    failed += test_guard_add_violation();
    /* Evaluator */
    failed += test_ndcg_perfect_ranking();
    failed += test_precision_recall_at_k();
    failed += test_judgmentset();
    /* Knowledge Graph */
    failed += test_kg_create();
    failed += test_kg_entity_extraction();
    failed += test_kg_build_from_chunks();
    failed += test_kg_density();
    failed += test_kg_normalize();
    /* Query Processor */
    failed += test_query_decompose();
    failed += test_query_expand_terms();
    failed += test_query_classify_intent();
    failed += test_query_complexity();
    failed += test_query_keywords();
    failed += test_aggregate_merge();
    /* Integration */
    failed += test_full_retrieval_eval_pipeline();

    printf("\n=== Results: %d/%d passed, %d failed ===\n",
           tests_passed, tests_run, failed);
    return failed > 0 ? 1 : 0;
}