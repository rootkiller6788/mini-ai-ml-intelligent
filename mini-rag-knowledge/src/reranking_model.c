#include "reranking_model.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

/* ──────────────────────────────────────────────
   Internal helpers
   ────────────────────────────────────────────── */
static void *safe_alloc(size_t n) {
    void *p = calloc(n, 1);
    if (!p) { fprintf(stderr, "OOM\n"); exit(1); }
    return p;
}

/* ──────────────────────────────────────────────
   Candidate Pool
   ────────────────────────────────────────────── */
CandidatePool* candidate_pool_create(size_t capacity) {
    CandidatePool *pool = safe_alloc(sizeof(CandidatePool));
    pool->capacity    = capacity ? capacity : 100;
    pool->candidates  = safe_alloc(pool->capacity * sizeof(RerankCandidate));
    pool->count       = 0;
    return pool;
}

void candidate_pool_destroy(CandidatePool *pool) {
    if (!pool) return;
    for (size_t i = 0; i < pool->count; i++) free(pool->candidates[i].doc_text);
    free(pool->candidates);
    free(pool);
}

void candidate_pool_add(CandidatePool *pool, const char *text,
                         size_t doc_id, float bm25, float vec_score) {
    if (pool->count >= pool->capacity) {
        pool->capacity *= 2;
        pool->candidates = realloc(pool->candidates,
                                   pool->capacity * sizeof(RerankCandidate));
        if (!pool->candidates) { fprintf(stderr, "Pool OOM\n"); exit(1); }
    }
    RerankCandidate *c = &pool->candidates[pool->count];
    c->doc_text     = text ? strdup(text) : NULL;
    c->doc_id       = doc_id;
    c->bm25_score   = bm25;
    c->vector_score = vec_score;
    c->final_score  = 0.0f;
    c->selected     = false;
    pool->count++;
}

/* ──────────────────────────────────────────────
   Cross-Encoder
   ────────────────────────────────────────────── */
CrossEncoder* cross_encoder_create(size_t emb_dim, size_t hidden_dim) {
    CrossEncoder *ce = safe_alloc(sizeof(CrossEncoder));
    ce->embedding_dim = emb_dim;
    ce->hidden_dim    = hidden_dim;
    ce->W_query = safe_alloc(emb_dim * hidden_dim * sizeof(float));
    ce->W_doc   = safe_alloc(emb_dim * hidden_dim * sizeof(float));
    ce->W_out   = safe_alloc(hidden_dim * sizeof(float));
    ce->b_hidden = safe_alloc(hidden_dim * sizeof(float));
    /* Random init */
    for (size_t i = 0; i < emb_dim * hidden_dim; i++) {
        ce->W_query[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
        ce->W_doc[i]   = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
    }
    for (size_t i = 0; i < hidden_dim; i++) {
        ce->W_out[i]   = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
        ce->b_hidden[i] = 0.0f;
    }
    ce->b_out = 0.0f;
    return ce;
}

void cross_encoder_destroy(CrossEncoder *ce) {
    if (!ce) return;
    free(ce->W_query);
    free(ce->W_doc);
    free(ce->W_out);
    free(ce->b_hidden);
    free(ce);
}

static float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

float cross_encoder_score(const CrossEncoder *ce,
                           const float *query_emb,
                           const float *doc_emb) {
    size_t ed = ce->embedding_dim;
    size_t hd = ce->hidden_dim;
    float *hidden = safe_alloc(hd * sizeof(float));
    /* hidden = relu(query * W_q + doc * W_d + b) */
    for (size_t j = 0; j < hd; j++) {
        float s = ce->b_hidden[j];
        for (size_t i = 0; i < ed; i++) {
            s += query_emb[i] * ce->W_query[i * hd + j];
            s += doc_emb[i]   * ce->W_doc[i * hd + j];
        }
        hidden[j] = fmaxf(0.0f, s);
    }
    float score = ce->b_out;
    for (size_t j = 0; j < hd; j++) {
        score += hidden[j] * ce->W_out[j];
    }
    free(hidden);
    return sigmoidf(score);
}

void cross_encoder_rank(const CrossEncoder *ce,
                         const float *query_emb,
                         CandidatePool *pool) {
    for (size_t i = 0; i < pool->count; i++) {
        pool->candidates[i].final_score = 0.0f;
        pool->candidates[i].selected = false;
    }
}

/* ──────────────────────────────────────────────
   Reciprocal Rank Fusion
   ────────────────────────────────────────────── */
float rrf_score(size_t dense_rank, size_t sparse_rank, float k) {
    return 1.0f / (k + (float)dense_rank + 1.0f) +
           1.0f / (k + (float)sparse_rank + 1.0f);
}

void rrf_fuse(CandidatePool *pool,
              const size_t *dense_ranks,
              const size_t *sparse_ranks,
              float k) {
    for (size_t i = 0; i < pool->count; i++) {
        pool->candidates[i].final_score =
            rrf_score(dense_ranks[i], sparse_ranks[i], k);
    }
}

/* ──────────────────────────────────────────────
   Maximal Marginal Relevance (MMR)
   ────────────────────────────────────────────── */
float mmr_relevance(const float *query_emb, const float *doc_emb, size_t dim) {
    float dot = 0.0f, nq = 0.0f, nd = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        dot += query_emb[i] * doc_emb[i];
        nq  += query_emb[i] * query_emb[i];
        nd  += doc_emb[i]   * doc_emb[i];
    }
    if (nq < 1e-12f || nd < 1e-12f) return 0.0f;
    return dot / sqrtf(nq * nd);
}

float mmr_similarity(const float *a, const float *b, size_t dim) {
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    if (na < 1e-12f || nb < 1e-12f) return 0.0f;
    return dot / sqrtf(na * nb);
}

void mmr_rerank(CandidatePool *pool,
                const float *query_emb,
                const float **doc_embs,
                size_t dim,
                float lambda,
                size_t output_k) {
    if (pool->count == 0) return;
    bool *selected = safe_alloc(pool->count * sizeof(bool));
    size_t *order  = safe_alloc(output_k * sizeof(size_t));
    size_t order_count = 0;
    /* Greedy MMR selection */
    for (size_t step = 0; step < output_k && step < pool->count; step++) {
        float best_mmr = -FLT_MAX;
        size_t best_i  = 0;
        for (size_t i = 0; i < pool->count; i++) {
            if (selected[i]) continue;
            float rel = mmr_relevance(query_emb, doc_embs[i], dim);
            float max_sim = 0.0f;
            for (size_t j = 0; j < order_count; j++) {
                float sim = mmr_similarity(doc_embs[i], doc_embs[order[j]], dim);
                if (sim > max_sim) max_sim = sim;
            }
            float mmr = lambda * rel - (1.0f - lambda) * max_sim;
            if (mmr > best_mmr) { best_mmr = mmr; best_i = i; }
        }
        selected[best_i] = true;
        order[order_count++] = best_i;
    }
    /* Reorder pool */
    CandidatePool *temp = candidate_pool_create(pool->count);
    for (size_t i = 0; i < order_count; i++) {
        RerankCandidate *c = &pool->candidates[order[i]];
        candidate_pool_add(temp, c->doc_text, c->doc_id,
                           c->bm25_score, c->vector_score);
        temp->candidates[i].final_score = (float)i;
        temp->candidates[i].selected = true;
    }
    /* Remaining unselected */
    for (size_t i = 0; i < pool->count; i++) {
        if (!selected[i]) {
            RerankCandidate *c = &pool->candidates[i];
            candidate_pool_add(temp, c->doc_text, c->doc_id,
                               c->bm25_score, c->vector_score);
        }
    }
    /* Swap */
    for (size_t i = 0; i < pool->count; i++) free(pool->candidates[i].doc_text);
    free(pool->candidates);
    pool->candidates = temp->candidates;
    pool->count      = temp->count;
    pool->capacity   = temp->capacity;
    free(temp);
    free(selected);
    free(order);
}

/* ──────────────────────────────────────────────
   Lost-in-the-Middle Reordering
   ────────────────────────────────────────────── */
void lost_in_middle_reorder(CandidatePool *pool, size_t output_k) {
    if (pool->count < 2) return;
    size_t n = output_k < pool->count ? output_k : pool->count;
    RerankCandidate *reordered = safe_alloc(n * sizeof(RerankCandidate));
    /* Place best docs at edges, worst in middle */
    size_t left = 0, right = n - 1;
    for (size_t i = 0; i < n; i++) {
        if (i % 2 == 0) {
            reordered[left++] = pool->candidates[i];
        } else {
            reordered[right--] = pool->candidates[i];
        }
    }
    for (size_t i = 0; i < n; i++) {
        pool->candidates[i] = reordered[i];
    }
    free(reordered);
}

/* ──────────────────────────────────────────────
   Full Reranking Pipeline
   ────────────────────────────────────────────── */
static int cmp_final_desc(const void *a, const void *b) {
    float d = ((const RerankCandidate*)b)->final_score -
              ((const RerankCandidate*)a)->final_score;
    return (d > 0) ? 1 : ((d < 0) ? -1 : 0);
}

void rerank_pipeline(CandidatePool *pool,
                     const RerankConfig *cfg,
                     const CrossEncoder *ce,
                     const float *query_emb,
                     const float **doc_embs,
                     size_t dim) {
    if (!pool || pool->count == 0) return;

    if (cfg->strategy == RERANK_CROSS_ENCODER && ce) {
        for (size_t i = 0; i < pool->count; i++) {
            pool->candidates[i].final_score =
                cross_encoder_score(ce, query_emb, doc_embs[i]);
        }
    } else if (cfg->strategy == RERANK_MMR) {
        mmr_rerank(pool, query_emb, doc_embs, dim,
                   cfg->mmr_lambda, cfg->final_k);
        return;
    }
    /* Sort by final score */
    qsort(pool->candidates, pool->count, sizeof(RerankCandidate), cmp_final_desc);
    /* Lost-in-middle reorder */
    if (cfg->lost_in_middle) {
        lost_in_middle_reorder(pool, cfg->final_k);
    }
}
