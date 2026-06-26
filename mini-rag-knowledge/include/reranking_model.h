#ifndef RERANKING_MODEL_H
#define RERANKING_MODEL_H

#include <stdbool.h>
#include <stddef.h>

/* ──────────────────────────────────────────────
   Reranking Configuration
   ────────────────────────────────────────────── */
typedef enum {
    RERANK_CROSS_ENCODER,
    RERANK_RRF,
    RERANK_MMR,
    RERANK_COMBINED
} RerankStrategy;

typedef struct {
    RerankStrategy strategy;
    size_t         initial_k;        /* top-k from first retrieval    */
    size_t         final_k;          /* output top-k after reranking  */
    size_t         hidden_dim;       /* cross-encoder hidden size     */
    float          rrf_k;            /* RRF smoothing constant        */
    float          mmr_lambda;       /* λ: relevance vs diversity     */
    bool           lost_in_middle;   /* reorder best docs to edges     */
} RerankConfig;

/* ──────────────────────────────────────────────
   Cross-Encoder (query+doc → relevance score)
   ────────────────────────────────────────────── */
typedef struct {
    size_t  embedding_dim;
    size_t  hidden_dim;
    float  *W_query;
    float  *W_doc;
    float  *W_out;
    float  *b_hidden;
    float   b_out;
} CrossEncoder;

/* ──────────────────────────────────────────────
   Retrieval Results (candidate documents)
   ────────────────────────────────────────────── */
typedef struct {
    char   *doc_text;
    size_t  doc_id;
    float   bm25_score;
    float   vector_score;
    float   final_score;
    bool    selected;
} RerankCandidate;

typedef struct {
    RerankCandidate *candidates;
    size_t           count;
    size_t           capacity;
} CandidatePool;

/* ──────────────────────────────────────────────
   Lifecycle
   ────────────────────────────────────────────── */
CandidatePool*  candidate_pool_create(size_t capacity);
void            candidate_pool_destroy(CandidatePool *pool);
void            candidate_pool_add(CandidatePool *pool, const char *text,
                                   size_t doc_id, float bm25, float vec_score);

/* ──────────────────────────────────────────────
   Cross-Encoder Operations
   ────────────────────────────────────────────── */
CrossEncoder*  cross_encoder_create(size_t emb_dim, size_t hidden_dim);
void           cross_encoder_destroy(CrossEncoder *ce);
float          cross_encoder_score(const CrossEncoder *ce,
                                   const float *query_emb,
                                   const float *doc_emb);
void           cross_encoder_rank(const CrossEncoder *ce,
                                  const float *query_emb,
                                  CandidatePool *pool);

/* ──────────────────────────────────────────────
   Reciprocal Rank Fusion (RRF)
   ────────────────────────────────────────────── */
float  rrf_score(size_t dense_rank, size_t sparse_rank, float k);
void   rrf_fuse(CandidatePool *pool,
                const size_t *dense_ranks,
                const size_t *sparse_ranks,
                float k);

/* ──────────────────────────────────────────────
   Maximal Marginal Relevance (MMR) — diversity
   ────────────────────────────────────────────── */
float  mmr_relevance(const float *query_emb, const float *doc_emb, size_t dim);
float  mmr_similarity(const float *a, const float *b, size_t dim);
void   mmr_rerank(CandidatePool *pool,
                  const float *query_emb,
                  const float **doc_embs,
                  size_t dim,
                  float lambda,
                  size_t output_k);

/* ──────────────────────────────────────────────
   Lost-in-the-Middle Reordering
   ────────────────────────────────────────────── */
void  lost_in_middle_reorder(CandidatePool *pool, size_t output_k);

/* ──────────────────────────────────────────────
   Full Reranking Pipeline
   ────────────────────────────────────────────── */
void  rerank_pipeline(CandidatePool *pool,
                      const RerankConfig *cfg,
                      const CrossEncoder *ce,
                      const float *query_emb,
                      const float **doc_embs,
                      size_t dim);

#endif /* RERANKING_MODEL_H */
