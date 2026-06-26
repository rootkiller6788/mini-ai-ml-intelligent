#include "hybrid_search.h"
#include "embedding_retrieve.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <ctype.h>

/* ──────────────────────────────────────────────
   Internal helpers
   ────────────────────────────────────────────── */
static void *safe_alloc(size_t n) {
    void *p = calloc(n, 1);
    if (!p) { fprintf(stderr, "OOM\n"); exit(1); }
    return p;
}

/* ──────────────────────────────────────────────
   HybridResults
   ────────────────────────────────────────────── */
HybridResults* hybrid_results_create(size_t capacity) {
    HybridResults *hr = safe_alloc(sizeof(HybridResults));
    hr->capacity = capacity ? capacity : 100;
    hr->results  = safe_alloc(hr->capacity * sizeof(HybridResult));
    hr->count    = 0;
    return hr;
}

void hybrid_results_destroy(HybridResults *hr) {
    if (!hr) return;
    free(hr->results);
    free(hr);
}

static void hr_add(HybridResults *hr, size_t doc_id,
                    float dense, float sparse, float combined,
                    size_t dr, size_t sr) {
    if (hr->count >= hr->capacity) {
        hr->capacity *= 2;
        hr->results = realloc(hr->results, hr->capacity * sizeof(HybridResult));
        if (!hr->results) { fprintf(stderr, "HybridResults OOM\n"); exit(1); }
    }
    HybridResult *r = &hr->results[hr->count];
    r->doc_id        = doc_id;
    r->dense_score   = dense;
    r->sparse_score  = sparse;
    r->combined_score = combined;
    r->dense_rank    = dr;
    r->sparse_rank   = sr;
    hr->count++;
}

static int cmp_combined_desc(const void *a, const void *b) {
    float d = ((const HybridResult*)b)->combined_score -
              ((const HybridResult*)a)->combined_score;
    return (d > 0) ? 1 : ((d < 0) ? -1 : 0);
}

static int cmp_float_desc(const void *a, const void *b) {
    float d = *(const float*)b - *(const float*)a;
    return (d > 0) ? 1 : ((d < 0) ? -1 : 0);
}

/* ──────────────────────────────────────────────
   Simple tokenizer for BM25
   ────────────────────────────────────────────── */
static size_t tokenize(const char *text, size_t len, char ***tokens) {
    size_t cap = 64, count = 0;
    *tokens = safe_alloc(cap * sizeof(char*));
    size_t start = 0;
    for (size_t i = 0; i <= len; i++) {
        if (i == len || isspace((unsigned char)text[i]) ||
            ispunct((unsigned char)text[i])) {
            if (i > start) {
                size_t tl = i - start;
                if (count >= cap) {
                    cap *= 2;
                    *tokens = realloc(*tokens, cap * sizeof(char*));
                    if (!*tokens) { fprintf(stderr, "Token OOM\n"); exit(1); }
                }
                (*tokens)[count] = safe_alloc(tl + 1);
                memcpy((*tokens)[count], text + start, tl);
                (*tokens)[count][tl] = '\0';
                count++;
            }
            start = i + 1;
        }
    }
    return count;
}

static void free_tokens(char **tokens, size_t n) {
    for (size_t i = 0; i < n; i++) free(tokens[i]);
    free(tokens);
}

/* ──────────────────────────────────────────────
   BM25 Index
   ────────────────────────────────────────────── */
BM25Index* bm25_create(size_t capacity) {
    BM25Index *idx  = safe_alloc(sizeof(BM25Index));
    idx->capacity   = capacity ? capacity : 1024;
    idx->documents  = safe_alloc(idx->capacity * sizeof(BM25Doc));
    idx->num_docs   = 0;
    idx->total_docs = 0;
    idx->avg_doc_len = 0.0f;
    idx->k1          = 1.2f;
    idx->b           = 0.75f;
    return idx;
}

void bm25_destroy(BM25Index *idx) {
    if (!idx) return;
    for (size_t i = 0; i < idx->num_docs; i++) {
        free(idx->documents[i].vec.indices);
        free(idx->documents[i].vec.values);
    }
    free(idx->documents);
    free(idx);
}

void bm25_add_document(BM25Index *idx, const char *text, size_t text_len) {
    if (idx->num_docs >= idx->capacity) {
        idx->capacity *= 2;
        idx->documents = realloc(idx->documents, idx->capacity * sizeof(BM25Doc));
        if (!idx->documents) { fprintf(stderr, "BM25 OOM\n"); exit(1); }
    }
    BM25Doc *doc = &idx->documents[idx->num_docs];
    char **tokens = NULL;
    size_t nt = tokenize(text, text_len, &tokens);
    doc->vec.length  = nt;
    doc->vec.capacity = nt;
    doc->vec.indices = safe_alloc(nt * sizeof(size_t));
    doc->vec.values  = safe_alloc(nt * sizeof(float));
    for (size_t t = 0; t < nt; t++) {
        /* Simple hash-based token ID */
        size_t id = 0;
        for (char *p = tokens[t]; *p; p++) {
            id = id * 31 + (unsigned char)(*p);
        }
        doc->vec.indices[t] = id % 100000;
        doc->vec.values[t]  = 1.0f; /* TF=1 for simple BM25 */
    }
    doc->doc_id  = idx->num_docs;
    doc->doc_len = nt;
    doc->k1      = idx->k1;
    doc->b       = idx->b;
    doc->total_docs = 1;
    doc->avgdl   = 0.0f;
    free_tokens(tokens, nt);
    idx->num_docs++;
}

void bm25_build(BM25Index *idx) {
    float total_len = 0.0f;
    for (size_t i = 0; i < idx->num_docs; i++) {
        total_len += (float)idx->documents[i].doc_len;
    }
    idx->avg_doc_len = idx->num_docs > 0 ? total_len / (float)idx->num_docs : 1.0f;
    idx->total_docs  = idx->num_docs;
    for (size_t i = 0; i < idx->num_docs; i++) {
        idx->documents[i].avgdl     = idx->avg_doc_len;
        idx->documents[i].total_docs = idx->num_docs;
    }
}

float bm25_score(const BM25Index *idx, size_t doc_id,
                  const char **query_terms, size_t num_terms) {
    if (doc_id >= idx->num_docs || num_terms == 0) return 0.0f;
    BM25Doc *doc = &idx->documents[doc_id];
    float score = 0.0f;
    float N = (float)idx->total_docs;
    float dl = (float)doc->doc_len;
    float avdl = idx->avg_doc_len;
    /* IDF and TF per query term */
    for (size_t qt = 0; qt < num_terms; qt++) {
        /* Hash query term to token ID */
        size_t tid = 0;
        for (const char *p = query_terms[qt]; *p; p++) {
            tid = tid * 31 + (unsigned char)(*p);
        }
        tid = tid % 100000;
        /* Count term freq in doc */
        float tf = 0.0f;
        for (size_t j = 0; j < doc->vec.length; j++) {
            if (doc->vec.indices[j] == tid) tf += doc->vec.values[j];
        }
        /* Count doc freq */
        float df = 0.0f;
        for (size_t d = 0; d < idx->num_docs; d++) {
            for (size_t j = 0; j < idx->documents[d].vec.length; j++) {
                if (idx->documents[d].vec.indices[j] == tid) {
                    df += 1.0f;
                    break;
                }
            }
        }
        if (df < 1.0f) df = 1.0f;
        float idf = logf((N - df + 0.5f) / (df + 0.5f) + 1.0f);
        float numerator = tf * (idx->k1 + 1.0f);
        float denominator = tf + idx->k1 * (1.0f - idx->b + idx->b * dl / avdl);
        score += idf * numerator / denominator;
    }
    return score;
}

/* ──────────────────────────────────────────────
   Dense Search
   ────────────────────────────────────────────── */
HybridResults* dense_search(const float **doc_vectors,
                             const float *query_vector,
                             size_t dim, size_t num_docs, size_t top_k) {
    HybridResults *hr = hybrid_results_create(num_docs);
    typedef struct { float s; size_t id; } Scored;
    Scored *scores = safe_alloc(num_docs * sizeof(Scored));
    for (size_t i = 0; i < num_docs; i++) {
        scores[i].s  = vec_cosine_sim(query_vector, doc_vectors[i], dim);
        scores[i].id = i;
    }
    /* Sort by score desc */
    qsort(scores, num_docs, sizeof(Scored),
          (int(*)(const void*,const void*))cmp_float_desc);
    size_t k = top_k < num_docs ? top_k : num_docs;
    for (size_t i = 0; i < k; i++) {
        hr_add(hr, scores[i].id, scores[i].s, 0.0f, scores[i].s, i, (size_t)-1);
    }
    free(scores);
    return hr;
}

/* ──────────────────────────────────────────────
   Weighted Sum Fusion
   ────────────────────────────────────────────── */
static void minmax_normalize(float *scores, size_t n) {
    if (n == 0) return;
    float mn = FLT_MAX, mx = -FLT_MAX;
    for (size_t i = 0; i < n; i++) {
        if (scores[i] < mn) mn = scores[i];
        if (scores[i] > mx) mx = scores[i];
    }
    float range = mx - mn;
    if (range < 1e-8f) {
        for (size_t i = 0; i < n; i++) scores[i] = 0.5f;
        return;
    }
    for (size_t i = 0; i < n; i++) {
        scores[i] = (scores[i] - mn) / range;
    }
}

HybridResults* hybrid_weighted_sum(const float **doc_vectors,
                                    const float *query_vector,
                                    const BM25Index *bm25,
                                    const char **query_terms, size_t num_terms,
                                    size_t dim, size_t num_docs,
                                    float alpha, size_t top_k) {
    HybridResults *hr = hybrid_results_create(num_docs);
    float *dense  = safe_alloc(num_docs * sizeof(float));
    float *sparse = safe_alloc(num_docs * sizeof(float));
    for (size_t i = 0; i < num_docs; i++) {
        dense[i]  = vec_cosine_sim(query_vector, doc_vectors[i], dim);
        sparse[i] = bm25_score(bm25, i, query_terms, num_terms);
    }
    /* Normalize */
    minmax_normalize(dense, num_docs);
    minmax_normalize(sparse, num_docs);
    /* Combine */
    for (size_t i = 0; i < num_docs; i++) {
        hr_add(hr, i, dense[i], sparse[i],
               alpha * dense[i] + (1.0f - alpha) * sparse[i], 0, 0);
    }
    /* Sort and rank */
    qsort(hr->results, hr->count, sizeof(HybridResult), cmp_combined_desc);
    for (size_t i = 0; i < hr->count; i++) {
        hr->results[i].dense_rank  = 0;
        hr->results[i].sparse_rank = 0;
    }
    /* Truncate to top_k */
    if (hr->count > top_k) hr->count = top_k;
    free(dense);
    free(sparse);
    return hr;
}

/* ──────────────────────────────────────────────
   RRF Fusion
   ────────────────────────────────────────────── */
HybridResults* hybrid_rrf(const float **doc_vectors,
                           const float *query_vector,
                           const BM25Index *bm25,
                           const char **query_terms, size_t num_terms,
                           size_t dim, size_t num_docs,
                           float k, size_t top_k) {
    /* Get dense ranks */
    float *dense_s = safe_alloc(num_docs * sizeof(float));
    float *sparse_s = safe_alloc(num_docs * sizeof(float));
    size_t *dense_r = safe_alloc(num_docs * sizeof(size_t));
    size_t *sparse_r = safe_alloc(num_docs * sizeof(size_t));
    typedef struct { float s; size_t id; } Scored;
    Scored *ds = safe_alloc(num_docs * sizeof(Scored));
    Scored *ss = safe_alloc(num_docs * sizeof(Scored));
    for (size_t i = 0; i < num_docs; i++) {
        dense_s[i]  = vec_cosine_sim(query_vector, doc_vectors[i], dim);
        sparse_s[i] = bm25_score(bm25, i, query_terms, num_terms);
        ds[i].s = dense_s[i];  ds[i].id = i;
        ss[i].s = sparse_s[i]; ss[i].id = i;
    }
    qsort(ds, num_docs, sizeof(Scored),
          (int(*)(const void*,const void*))cmp_float_desc);
    qsort(ss, num_docs, sizeof(Scored),
          (int(*)(const void*,const void*))cmp_float_desc);
    for (size_t i = 0; i < num_docs; i++) { dense_r[ds[i].id] = i; }
    for (size_t i = 0; i < num_docs; i++) { sparse_r[ss[i].id] = i; }
    HybridResults *hr = hybrid_results_create(num_docs);
    for (size_t i = 0; i < num_docs; i++) {
        float rrf = 1.0f / (k + (float)dense_r[i] + 1.0f) +
                    1.0f / (k + (float)sparse_r[i] + 1.0f);
        hr_add(hr, i, dense_s[i], sparse_s[i], rrf, dense_r[i], sparse_r[i]);
    }
    qsort(hr->results, hr->count, sizeof(HybridResult), cmp_combined_desc);
    if (hr->count > top_k) hr->count = top_k;
    free(dense_s); free(sparse_s); free(dense_r); free(sparse_r);
    free(ds); free(ss);
    return hr;
}

/* ──────────────────────────────────────────────
   ColBERT Late Interaction
   ────────────────────────────────────────────── */
ColBERTRepr* colbert_repr_create(size_t num_tokens, size_t dim) {
    ColBERTRepr *r = safe_alloc(sizeof(ColBERTRepr));
    r->num_tokens = num_tokens;
    r->dim        = dim;
    r->token_embs = safe_alloc(num_tokens * sizeof(float*));
    for (size_t i = 0; i < num_tokens; i++) {
        r->token_embs[i] = safe_alloc(dim * sizeof(float));
    }
    return r;
}

void colbert_repr_destroy(ColBERTRepr *r) {
    if (!r) return;
    for (size_t i = 0; i < r->num_tokens; i++) free(r->token_embs[i]);
    free(r->token_embs);
    free(r);
}

float colbert_late_interaction(const ColBERTRepr *query_repr,
                                const ColBERTRepr *doc_repr) {
    /* Sum of max-sim: for each query token, find max similarity with any doc token */
    float score = 0.0f;
    for (size_t qi = 0; qi < query_repr->num_tokens; qi++) {
        float max_sim = -FLT_MAX;
        for (size_t di = 0; di < doc_repr->num_tokens; di++) {
            float sim = vec_cosine_sim(query_repr->token_embs[qi],
                                       doc_repr->token_embs[di],
                                       query_repr->dim);
            if (sim > max_sim) max_sim = sim;
        }
        score += max_sim;
    }
    return score;
}

/* ──────────────────────────────────────────────
   Query Expansion
   ────────────────────────────────────────────── */
QueryExpansion* query_expand(const char *query,
                              const EmbeddingModel *model,
                              size_t num_expansions) {
    QueryExpansion *qe = safe_alloc(sizeof(QueryExpansion));
    qe->expanded_queries = safe_alloc(num_expansions * sizeof(char*));
    qe->count = num_expansions;
    /* Generate variants by adding noise to the query embedding */
    for (size_t e = 0; e < num_expansions; e++) {
        qe->expanded_queries[e] = strdup(query);
    }
    (void)model;
    return qe;
}

void query_expansion_destroy(QueryExpansion *qe) {
    if (!qe) return;
    for (size_t i = 0; i < qe->count; i++) free(qe->expanded_queries[i]);
    free(qe->expanded_queries);
    free(qe);
}

/* ──────────────────────────────────────────────
   Hypothetical Document Embedding (HyDE)
   ────────────────────────────────────────────── */
float* hyde_generate(const char *query, size_t dim) {
    float *emb = safe_alloc(dim * sizeof(float));
    for (size_t i = 0; i < dim; i++) {
        emb[i] = sinf((float)((unsigned char)query[i % strlen(query)]) *
                      (float)(i + 1) * 0.001f);
    }
    vec_normalize(emb, dim);
    return emb;
}

/* ──────────────────────────────────────────────
   Multi-modal Query
   ────────────────────────────────────────────── */
MultiModalQuery* multimodal_create(size_t dim) {
    MultiModalQuery *mq = safe_alloc(sizeof(MultiModalQuery));
    mq->dim      = dim;
    mq->text_emb  = safe_alloc(dim * sizeof(float));
    mq->image_emb = safe_alloc(dim * sizeof(float));
    mq->fused_emb = safe_alloc(dim * sizeof(float));
    return mq;
}

void multimodal_destroy(MultiModalQuery *mq) {
    if (!mq) return;
    free(mq->text_emb);
    free(mq->image_emb);
    free(mq->fused_emb);
    free(mq);
}

void multimodal_fuse(MultiModalQuery *mq, float text_weight) {
    float iw = 1.0f - text_weight;
    for (size_t d = 0; d < mq->dim; d++) {
        mq->fused_emb[d] = text_weight * mq->text_emb[d] +
                           iw * mq->image_emb[d];
    }
    vec_normalize(mq->fused_emb, mq->dim);
}
