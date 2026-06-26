#ifndef EMBEDDING_RETRIEVE_H
#define EMBEDDING_RETRIEVE_H

#include <stdbool.h>
#include <stddef.h>

/* ──────────────────────────────────────────────
   RAG Pipeline Configuration
   ────────────────────────────────────────────── */
#define RAG_MAX_CHUNKS     65536
#define RAG_MAX_DOCS       16384
#define RAG_MAX_TOPK           50
#define RAG_DEFAULT_DIM       768
#define RAG_DEFAULT_CHUNK     512
#define RAG_DEFAULT_OVERLAP    64
#define RAG_DEFAULT_TOPK        5

typedef struct {
    size_t embedding_dim;
    size_t chunk_size;
    size_t chunk_overlap;
    size_t max_chunks_per_doc;
    size_t top_k;
    float  similarity_threshold;
    bool   use_metadata_filter;
    bool   normalize_embeddings;
} RagConfig;

/* ──────────────────────────────────────────────
   Document & Chunk Types
   ────────────────────────────────────────────── */
typedef struct {
    char  *text;
    size_t text_len;
    char  *source;
    char  *title;
    int    page;
    void  *metadata;
} Document;

typedef struct {
    char   *text;
    size_t  text_len;
    size_t  doc_index;
    size_t  chunk_index;
    size_t  start_offset;
    size_t  end_offset;
    float  *embedding;
    size_t  embedding_dim;
    char   *source;
    int     page;
} Chunk;

typedef struct {
    size_t *indices;
    float  *scores;
    size_t  count;
} RetrieveResult;

/* ──────────────────────────────────────────────
   Embedding Model (BERT-like encoder simulation)
   ────────────────────────────────────────────── */
typedef enum {
    EMBED_SIM_COSINE,
    EMBED_SIM_SIMPLE_PROJ,
    EMBED_SIM_RANDOM_PROJ
} EmbedSimType;

typedef struct {
    size_t       dim;
    size_t       vocab_size;
    float       *token_embed;
    float       *pos_embed;
    float       *proj_weight;
    EmbedSimType sim_type;
} EmbeddingModel;

/* ──────────────────────────────────────────────
   Vector Index (Flat-L2 / Cosine)
   ────────────────────────────────────────────── */
typedef enum {
    INDEX_FLAT,
    INDEX_L2,
    INDEX_COSINE
} IndexType;

typedef struct {
    float    **vectors;
    size_t     *ids;
    size_t      count;
    size_t      capacity;
    size_t      dim;
    IndexType   type;
    bool        is_built;
} VectorIndex;

/* ──────────────────────────────────────────────
   RAG Context (main pipeline state)
   ────────────────────────────────────────────── */
typedef struct {
    RagConfig       config;
    Document       *documents;
    size_t          num_docs;
    Chunk          *chunks;
    size_t          num_chunks;
    EmbeddingModel *model;
    VectorIndex    *index;
    size_t         *small_to_big_map;
} RagContext;

/* ──────────────────────────────────────────────
   Lifecycle
   ────────────────────────────────────────────── */
RagContext*  rag_create(const RagConfig *cfg);
void         rag_destroy(RagContext *r);

/* ──────────────────────────────────────────────
   Document Operations
   ────────────────────────────────────────────── */
void  rag_add_document(RagContext *r, const char *text, size_t text_len,
                       const char *source, const char *title, int page);
void  rag_add_documents(RagContext *r, Document *docs, size_t n);

/* ──────────────────────────────────────────────
   Chunking
   ────────────────────────────────────────────── */
void  rag_chunk_documents(RagContext *r);

/* ──────────────────────────────────────────────
   Embedding
   ────────────────────────────────────────────── */
EmbeddingModel*  embed_model_create(size_t dim, EmbedSimType sim_type);
void             embed_model_destroy(EmbeddingModel *m);
void             embed_encode(const EmbeddingModel *m, const char *text,
                              size_t text_len, float *out);
void             embed_encode_batch(const EmbeddingModel *m, Chunk *chunks,
                                    size_t n);
void             rag_embed_chunks(RagContext *r);

/* ──────────────────────────────────────────────
   Indexing
   ────────────────────────────────────────────── */
VectorIndex*  index_create(size_t capacity, size_t dim, IndexType type);
void          index_destroy(VectorIndex *idx);
void          index_add(VectorIndex *idx, const float *vec, size_t id);
void          index_build(VectorIndex *idx);
void          rag_build_index(RagContext *r);

/* ──────────────────────────────────────────────
   Retrieval
   ────────────────────────────────────────────── */
RetrieveResult*  retrieve_create(void);
void             retrieve_destroy(RetrieveResult *res);
RetrieveResult*  index_search(const VectorIndex *idx, const float *query,
                              size_t top_k);
RetrieveResult*  rag_retrieve(RagContext *r, const char *query,
                              size_t query_len);

/* ──────────────────────────────────────────────
   Prompt Construction
   ────────────────────────────────────────────── */
char*  rag_build_prompt(const RagContext *r, const RetrieveResult *result,
                        const char *query, const char *system_prompt);

/* ──────────────────────────────────────────────
   Utilities
   ────────────────────────────────────────────── */
float  vec_l2_dist(const float *a, const float *b, size_t dim);
float  vec_cosine_sim(const float *a, const float *b, size_t dim);
float  vec_dot(const float *a, const float *b, size_t dim);
void   vec_normalize(float *v, size_t dim);

#endif /* EMBEDDING_RETRIEVE_H */
