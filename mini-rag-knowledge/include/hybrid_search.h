#ifndef HYBRID_SEARCH_H
#define HYBRID_SEARCH_H

#include <stdbool.h>
#include <stddef.h>

/* ──────────────────────────────────────────────
   Hybrid Search Configuration
   ────────────────────────────────────────────── */
typedef enum {
    HYBRID_WEIGHTED_SUM,
    HYBRID_RRF,
    HYBRID_COLBERT,
    HYBRID_SCORE_COMBINE
} HybridFusion;

typedef struct {
    HybridFusion fusion;
    float         alpha;          /* dense weight (1-alpha = sparse)      */
    float         rrf_k;          /* RRF constant (default 60)            */
    size_t        top_k;
    bool          normalize_scores;
    bool          enable_query_expansion;
    bool          enable_hyde;
} HybridConfig;

/* ──────────────────────────────────────────────
   Sparse Vectors (BM25 representation)
   ────────────────────────────────────────────── */
typedef struct {
    size_t  *indices;
    float   *values;
    size_t   length;
    size_t   capacity;
} SparseVector;

typedef struct {
    SparseVector  vec;
    size_t        doc_id;
    size_t        doc_len;
    float         avgdl;
    size_t        total_docs;
    float         k1;
    float         b;
} BM25Doc;

/* ──────────────────────────────────────────────
   BM25 Index
   ────────────────────────────────────────────── */
typedef struct {
    BM25Doc *documents;
    size_t   num_docs;
    size_t   capacity;
    float    avg_doc_len;
    size_t   total_docs;
    float    k1;
    float    b;
} BM25Index;

/* ──────────────────────────────────────────────
   Hybrid Search Results
   ────────────────────────────────────────────── */
typedef struct {
    size_t  doc_id;
    float   dense_score;
    float   sparse_score;
    float   combined_score;
    size_t  dense_rank;
    size_t  sparse_rank;
} HybridResult;

typedef struct {
    HybridResult *results;
    size_t        count;
    size_t        capacity;
} HybridResults;

/* ──────────────────────────────────────────────
   Multi-vector Late Interaction (ColBERT-style)
   ────────────────────────────────────────────── */
typedef struct {
    float  **token_embs;
    size_t   num_tokens;
    size_t   dim;
} ColBERTRepr;

/* ──────────────────────────────────────────────
   Query Expansion Types
   ────────────────────────────────────────────── */
typedef struct {
    char  **expanded_queries;
    size_t  count;
} QueryExpansion;

/* ──────────────────────────────────────────────
   Lifecycle
   ────────────────────────────────────────────── */
HybridResults*  hybrid_results_create(size_t capacity);
void            hybrid_results_destroy(HybridResults *hr);

/* ──────────────────────────────────────────────
   BM25 (Sparse Retrieval)
   ────────────────────────────────────────────── */
BM25Index*  bm25_create(size_t capacity);
void        bm25_destroy(BM25Index *idx);
void        bm25_add_document(BM25Index *idx, const char *text, size_t text_len);
void        bm25_build(BM25Index *idx);
float       bm25_score(const BM25Index *idx, size_t doc_id,
                       const char **query_terms, size_t num_terms);

/* ──────────────────────────────────────────────
   Dense Search (Vector Similarity)
   ────────────────────────────────────────────── */
HybridResults*  dense_search(const float **doc_vectors,
                             const float *query_vector,
                             size_t dim, size_t num_docs, size_t top_k);

/* ──────────────────────────────────────────────
   Weighted Sum Fusion
   ────────────────────────────────────────────── */
HybridResults*  hybrid_weighted_sum(const float **doc_vectors,
                                    const float *query_vector,
                                    const BM25Index *bm25,
                                    const char **query_terms, size_t num_terms,
                                    size_t dim, size_t num_docs,
                                    float alpha, size_t top_k);

/* ──────────────────────────────────────────────
   RRF Fusion
   ────────────────────────────────────────────── */
HybridResults*  hybrid_rrf(const float **doc_vectors,
                           const float *query_vector,
                           const BM25Index *bm25,
                           const char **query_terms, size_t num_terms,
                           size_t dim, size_t num_docs,
                           float k, size_t top_k);

/* ──────────────────────────────────────────────
   ColBERT Late Interaction
   ────────────────────────────────────────────── */
float  colbert_late_interaction(const ColBERTRepr *query_repr,
                                const ColBERTRepr *doc_repr);

ColBERTRepr*  colbert_repr_create(size_t num_tokens, size_t dim);
void          colbert_repr_destroy(ColBERTRepr *r);

/* ──────────────────────────────────────────────
   Query Expansion
   ────────────────────────────────────────────── */
QueryExpansion*  query_expand(const char *query,
                              const EmbeddingModel *model,
                              size_t num_expansions);
void             query_expansion_destroy(QueryExpansion *qe);

/* ──────────────────────────────────────────────
   Hypothetical Document Embedding (HyDE)
   ────────────────────────────────────────────── */
float*  hyde_generate(const char *query, size_t dim);

/* ──────────────────────────────────────────────
   Multi-modal Retrieval Helper
   ────────────────────────────────────────────── */
typedef struct {
    float *text_emb;
    float *image_emb;
    float *fused_emb;
    size_t dim;
} MultiModalQuery;

MultiModalQuery*  multimodal_create(size_t dim);
void              multimodal_destroy(MultiModalQuery *mq);
void              multimodal_fuse(MultiModalQuery *mq, float text_weight);

#endif /* HYBRID_SEARCH_H */
