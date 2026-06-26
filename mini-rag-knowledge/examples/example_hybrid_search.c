#include "hybrid_search.h"
#include "embedding_retrieve.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#define NUM_DOCS 6
#define IMG_REPR_DIM 128

int main(void) {
    printf("=== Hybrid Search (Dense + Sparse) Demo ===\n\n");

    const char *documents[NUM_DOCS] = {
        "machine learning is a subset of artificial intelligence",
        "deep learning uses neural networks with many layers",
        "python is a popular programming language for data science",
        "vector databases store embeddings for similarity search",
        "bm25 is a probabilistic ranking function for text retrieval",
        "attention mechanisms improved neural machine translation",
    };

    size_t dim = 64;
    /* Allocate document embeddings (random for demo) */
    float **doc_embs = malloc(NUM_DOCS * sizeof(float*));
    for (size_t i = 0; i < NUM_DOCS; i++) {
        doc_embs[i] = malloc(dim * sizeof(float));
        for (size_t j = 0; j < dim; j++) {
            doc_embs[i][j] = sinf((float)(i * dim + j) * 0.1f) * 0.5f +
                             cosf((float)((i + 1) * j) * 0.07f) * 0.3f;
        }
        vec_normalize(doc_embs[i], dim);
    }

    /* Build BM25 index */
    BM25Index *bm25 = bm25_create(NUM_DOCS);
    for (size_t i = 0; i < NUM_DOCS; i++) {
        bm25_add_document(bm25, documents[i], strlen(documents[i]));
    }
    bm25_build(bm25);
    printf("BM25 index built: %zu docs, avg_len=%.2f\n",
           bm25->num_docs, bm25->avg_doc_len);

    /* Query */
    const char *query = "neural network deep learning";
    const char *query_terms[] = {"neural", "network", "deep", "learning"};
    size_t num_terms = 4;

    /* Create query embedding (simple mock) */
    float *query_emb = malloc(dim * sizeof(float));
    for (size_t j = 0; j < dim; j++) {
        query_emb[j] = cosf((float)(j + 42) * 0.13f) * 0.5f;
    }
    vec_normalize(query_emb, dim);

    printf("\n--- Dense-only Search ---\n");
    HybridResults *dense = dense_search((const float**)doc_embs, query_emb,
                                         dim, NUM_DOCS, NUM_DOCS);
    for (size_t i = 0; i < dense->count; i++) {
        printf("  rank %zu: doc[%zu] score=%.4f\n",
               i + 1, dense->results[i].doc_id, dense->results[i].dense_score);
    }
    hybrid_results_destroy(dense);

    printf("\n--- BM25-only Search ---\n");
    float bm25_scores[NUM_DOCS];
    for (size_t i = 0; i < NUM_DOCS; i++) {
        bm25_scores[i] = bm25_score(bm25, i, query_terms, num_terms);
    }
    typedef struct { float s; size_t id; } Sc;
    Sc sc[NUM_DOCS];
    for (size_t i = 0; i < NUM_DOCS; i++) { sc[i].s = bm25_scores[i]; sc[i].id = i; }
    for (size_t i = 0; i < NUM_DOCS; i++) {
        for (size_t j = i + 1; j < NUM_DOCS; j++) {
            if (sc[j].s > sc[i].s) { Sc t = sc[i]; sc[i] = sc[j]; sc[j] = t; }
        }
    }
    for (size_t i = 0; i < NUM_DOCS; i++) {
        printf("  rank %zu: doc[%zu] score=%.4f  \"%s\"\n",
               i + 1, sc[i].id, sc[i].s, documents[sc[i].id]);
    }

    printf("\n--- Weighted Sum Fusion (alpha=0.6 dense, 0.4 sparse) ---\n");
    HybridResults *weighted = hybrid_weighted_sum(
        (const float**)doc_embs, query_emb, bm25,
        query_terms, num_terms, dim, NUM_DOCS, 0.6f, NUM_DOCS);
    for (size_t i = 0; i < weighted->count; i++) {
        printf("  rank %zu: doc[%zu] dense=%.4f sparse=%.4f combined=%.4f\n",
               i + 1, weighted->results[i].doc_id,
               weighted->results[i].dense_score,
               weighted->results[i].sparse_score,
               weighted->results[i].combined_score);
    }
    hybrid_results_destroy(weighted);

    printf("\n--- RRF Fusion (k=60) ---\n");
    HybridResults *rrf = hybrid_rrf(
        (const float**)doc_embs, query_emb, bm25,
        query_terms, num_terms, dim, NUM_DOCS, 60.0f, NUM_DOCS);
    for (size_t i = 0; i < rrf->count; i++) {
        HybridResult *r = &rrf->results[i];
        printf("  rank %zu: doc[%zu] dense_r=%zu sparse_r=%zu rrf=%.4f  \"%s\"\n",
               i + 1, r->doc_id, r->dense_rank, r->sparse_rank,
               r->combined_score, documents[r->doc_id]);
    }
    hybrid_results_destroy(rrf);

    /* ColBERT demo */
    printf("\n--- ColBERT Late Interaction (mock) ---\n");
    ColBERTRepr *q_repr = colbert_repr_create(3, dim);
    ColBERTRepr *d_repr = colbert_repr_create(5, dim);
    for (size_t t = 0; t < 3; t++)
        for (size_t j = 0; j < dim; j++)
            q_repr->token_embs[t][j] = sinf((float)(t * j + 1) * 0.1f);
    for (size_t t = 0; t < 5; t++)
        for (size_t j = 0; j < dim; j++)
            d_repr->token_embs[t][j] = cosf((float)(t * j + 2) * 0.1f);
    float colbert_score = colbert_late_interaction(q_repr, d_repr);
    printf("  ColBERT score: %.4f\n", colbert_score);
    colbert_repr_destroy(q_repr);
    colbert_repr_destroy(d_repr);

    /* HyDE demo */
    printf("\n--- HyDE Hypothetical Document Embedding ---\n");
    float *hyde_emb = hyde_generate(query, dim);
    float hyde_norm = 0.0f;
    for (size_t j = 0; j < dim; j++) hyde_norm += hyde_emb[j] * hyde_emb[j];
    printf("  HyDE embedding norm: %.4f (first 4: %.3f %.3f %.3f %.3f)\n",
           sqrtf(hyde_norm), hyde_emb[0], hyde_emb[1], hyde_emb[2], hyde_emb[3]);
    free(hyde_emb);

    /* Multi-modal demo */
    printf("\n--- Multi-modal Fusion ---\n");
    MultiModalQuery *mmq = multimodal_create(dim);
    for (size_t j = 0; j < dim; j++) {
        mmq->text_emb[j]  = sinf((float)j * 0.01f);
        mmq->image_emb[j] = cosf((float)j * 0.01f);
    }
    multimodal_fuse(mmq, 0.7f);
    printf("  Fused embedding (first 4): %.3f %.3f %.3f %.3f\n",
           mmq->fused_emb[0], mmq->fused_emb[1],
           mmq->fused_emb[2], mmq->fused_emb[3]);
    multimodal_destroy(mmq);

    /* Cleanup */
    bm25_destroy(bm25);
    for (size_t i = 0; i < NUM_DOCS; i++) free(doc_embs[i]);
    free(doc_embs);
    free(query_emb);

    printf("\nAll hybrid search demos complete.\n");
    return 0;
}
