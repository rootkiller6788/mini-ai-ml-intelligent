/*
 * mini-rag-knowledge — Core Benchmarks
 *
 * Benchmarks: embedding retrieval, chunking, reranking,
 *             hybrid search, guardrails.
 */
#include "../include/embedding_retrieve.h"
#include "../include/chunking_strategy.h"
#include "../include/reranking_model.h"
#include "../include/hybrid_search.h"
#include "../include/guardrails_gen.h"
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
    printf("=== mini-rag-knowledge Benchmarks (N=%d) ===\n\n", N);

    /* ── RAG Context Create ── */
    {
        RagConfig cfg;
        memset(&cfg, 0, sizeof(cfg));
        cfg.embedding_dim = 384;
        cfg.chunk_size = 512;
        cfg.top_k = 5;
        t0 = now_ms();
        for (int r = 0; r < N / 20; r++) {
            RagContext *rag = rag_create(&cfg);
            rag_destroy(rag);
        }
        t1 = now_ms();
        printf("  rag_create+destroy:  %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 20, t1 - t0, (t1 - t0) / (double)(N / 20) * 1000.0);
    }

    /* ── Embedding Model Create ── */
    {
        t0 = now_ms();
        for (int r = 0; r < N / 20; r++) {
            EmbeddingModel *m = embed_model_create(384, EMBED_SIM_COSINE);
            embed_model_destroy(m);
        }
        t1 = now_ms();
        printf("  embed_model_create:  %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 20, t1 - t0, (t1 - t0) / (double)(N / 20) * 1000.0);
    }

    /* ── Vector Index Add/Search ── */
    {
        VectorIndex *idx = index_create(1024, 64, INDEX_COSINE);
        float v[64];
        for (int i = 0; i < 64; i++) v[i] = 1.0f;
        for (int i = 0; i < 512; i++) index_add(idx, v, i);
        index_build(idx);
        t0 = now_ms();
        for (int r = 0; r < N / 10; r++) {
            RetrieveResult *res = index_search(idx, v, 5);
            retrieve_destroy(res);
        }
        t1 = now_ms();
        printf("  index_search(k=5):   %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 10, t1 - t0, (t1 - t0) / (double)(N / 10) * 1000.0);
        index_destroy(idx);
    }

    /* ── Chunking ── */
    {
        const char *text = "This is a test document. It contains multiple sentences. "
                           "We use it to benchmark chunking strategies. Each chunk "
                           "should be processed efficiently.";
        t0 = now_ms();
        for (int r = 0; r < N / 10; r++) {
            ChunkList *cl = chunk_split_characters(text, strlen(text), 32, 8);
            chunklist_destroy(cl);
        }
        t1 = now_ms();
        printf("  chunk_split_char:    %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 10, t1 - t0, (t1 - t0) / (double)(N / 10) * 1000.0);
    }

    /* ── Reranker ── */
    {
        CrossEncoder *ce = cross_encoder_create(64, 32);
        t0 = now_ms();
        for (int r = 0; r < N / 10; r++) {
            float qe[64], de[64];
            memset(qe, 0, sizeof(qe));
            memset(de, 0, sizeof(de));
            cross_encoder_score(ce, qe, de);
        }
        t1 = now_ms();
        printf("  cross_encoder_score: %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 10, t1 - t0, (t1 - t0) / (double)(N / 10) * 1000.0);
        cross_encoder_destroy(ce);
    }

    /* ── BM25 ── */
    {
        BM25Index *bi = bm25_create(1024);
        const char *doc = "the quick brown fox jumps over the lazy dog";
        bm25_add_document(bi, doc, strlen(doc));
        bm25_build(bi);
        const char *terms[] = {"quick", "fox"};
        t0 = now_ms();
        for (int r = 0; r < N / 10; r++) {
            bm25_score(bi, 0, terms, 2);
        }
        t1 = now_ms();
        printf("  bm25_score:          %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 10, t1 - t0, (t1 - t0) / (double)(N / 10) * 1000.0);
        bm25_destroy(bi);
    }

    /* ── Guardrails ── */
    {
        t0 = now_ms();
        for (int r = 0; r < N / 10; r++) {
            GuardResult *gr = guard_result_create();
            guard_add_violation(gr, GUARD_INPUT_INJECTION, SEVERITY_SAFE, "ok", 0, 0, 0.0f);
            guard_result_destroy(gr);
        }
        t1 = now_ms();
        printf("  guard_result:        %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 10, t1 - t0, (t1 - t0) / (double)(N / 10) * 1000.0);
    }

    printf("\nDone.\n");
    return 0;
}
