#include "embedding_retrieve.h"
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
    if (!p) { fprintf(stderr, "OOM: alloc %zu bytes\n", n); exit(1); }
    return p;
}

static float random_float(void) {
    return (float)rand() / (float)RAND_MAX;
}

/* ──────────────────────────────────────────────
   Vector operations
   ────────────────────────────────────────────── */
float vec_dot(const float *a, const float *b, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; i++) sum += a[i] * b[i];
    return sum;
}

float vec_l2_dist(const float *a, const float *b, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sqrtf(sum);
}

float vec_cosine_sim(const float *a, const float *b, size_t dim) {
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if (na < 1e-12f || nb < 1e-12f) return 0.0f;
    return dot / sqrtf(na * nb);
}

void vec_normalize(float *v, size_t dim) {
    float n = 0.0f;
    for (size_t i = 0; i < dim; i++) n += v[i] * v[i];
    if (n < 1e-12f) return;
    n = sqrtf(n);
    for (size_t i = 0; i < dim; i++) v[i] /= n;
}

/* ──────────────────────────────────────────────
   Embedding Model
   ────────────────────────────────────────────── */
EmbeddingModel* embed_model_create(size_t dim, EmbedSimType sim_type) {
    EmbeddingModel *m = safe_alloc(sizeof(EmbeddingModel));
    m->dim      = dim;
    m->sim_type = sim_type;
    m->vocab_size = 30522;
    m->token_embed = NULL;
    m->pos_embed   = NULL;
    m->proj_weight = safe_alloc(dim * dim * sizeof(float));
    for (size_t i = 0; i < dim * dim; i++) {
        m->proj_weight[i] = (random_float() - 0.5f) * 0.02f;
    }
    return m;
}

void embed_model_destroy(EmbeddingModel *m) {
    if (!m) return;
    free(m->token_embed);
    free(m->pos_embed);
    free(m->proj_weight);
    free(m);
}

void embed_encode(const EmbeddingModel *m, const char *text,
                  size_t text_len, float *out) {
    (void)text_len;
    /* Simple projection-based embedding: character-level hash + projection */
    if (m->sim_type == EMBED_SIM_SIMPLE_PROJ) {
        float *temp = safe_alloc(m->dim * sizeof(float));
        for (size_t i = 0; i < m->dim; i++) temp[i] = 0.0f;
        for (size_t c = 0; text[c] && c < 512; c++) {
            size_t idx = ((unsigned char)text[c] * 2654435761ULL) % m->dim;
            temp[idx] += 1.0f;
        }
        for (size_t i = 0; i < m->dim; i++) {
            out[i] = 0.0f;
            for (size_t j = 0; j < m->dim; j++) {
                out[i] += temp[j] * m->proj_weight[j * m->dim + i];
            }
        }
        free(temp);
    } else if (m->sim_type == EMBED_SIM_RANDOM_PROJ) {
        for (size_t i = 0; i < m->dim; i++) {
            out[i] = (random_float() - 0.5f) * 0.1f;
        }
        for (size_t c = 0; text[c] && c < 512; c++) {
            size_t seed = (unsigned char)text[c] * 0x9E3779B9;
            for (size_t i = 0; i < m->dim; i++) {
                out[i] += sinf((float)(seed + i) * 0.001f) * 0.01f;
            }
        }
    } else {
        /* Cosine-sim based mock: each char contributes a pattern */
        for (size_t i = 0; i < m->dim; i++) out[i] = 0.0f;
        for (size_t c = 0; text[c] && c < 512; c++) {
            float phase = (float)((unsigned char)text[c]) * 0.024543f;
            for (size_t i = 0; i < m->dim; i++) {
                out[i] += cosf(phase * (float)(i + 1)) * 0.01f;
            }
        }
    }
    if (vec_dot(out, out, m->dim) > 1e-6f) vec_normalize(out, m->dim);
}

void embed_encode_batch(const EmbeddingModel *m, Chunk *chunks, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (!chunks[i].embedding) {
            chunks[i].embedding = safe_alloc(m->dim * sizeof(float));
        }
        chunks[i].embedding_dim = m->dim;
        embed_encode(m, chunks[i].text, chunks[i].text_len, chunks[i].embedding);
    }
}

/* ──────────────────────────────────────────────
   Vector Index
   ────────────────────────────────────────────── */
VectorIndex* index_create(size_t capacity, size_t dim, IndexType type) {
    VectorIndex *idx = safe_alloc(sizeof(VectorIndex));
    idx->capacity = capacity ? capacity : 1024;
    idx->dim      = dim;
    idx->type     = type;
    idx->count    = 0;
    idx->is_built = false;
    idx->vectors  = safe_alloc(idx->capacity * sizeof(float*));
    idx->ids      = safe_alloc(idx->capacity * sizeof(size_t));
    return idx;
}

void index_destroy(VectorIndex *idx) {
    if (!idx) return;
    for (size_t i = 0; i < idx->count; i++) free(idx->vectors[i]);
    free(idx->vectors);
    free(idx->ids);
    free(idx);
}

void index_add(VectorIndex *idx, const float *vec, size_t id) {
    if (idx->count >= idx->capacity) {
        idx->capacity *= 2;
        idx->vectors = realloc(idx->vectors, idx->capacity * sizeof(float*));
        idx->ids     = realloc(idx->ids, idx->capacity * sizeof(size_t));
        if (!idx->vectors || !idx->ids) { fprintf(stderr, "Index OOM\n"); exit(1); }
    }
    idx->vectors[idx->count] = safe_alloc(idx->dim * sizeof(float));
    memcpy(idx->vectors[idx->count], vec, idx->dim * sizeof(float));
    idx->ids[idx->count] = id;
    idx->count++;
}

void index_build(VectorIndex *idx) {
    idx->is_built = true;
}

RetrieveResult* retrieve_create(void) {
    RetrieveResult *r = safe_alloc(sizeof(RetrieveResult));
    r->indices = safe_alloc(RAG_MAX_TOPK * sizeof(size_t));
    r->scores  = safe_alloc(RAG_MAX_TOPK * sizeof(float));
    r->count   = 0;
    return r;
}

void retrieve_destroy(RetrieveResult *res) {
    if (!res) return;
    free(res->indices);
    free(res->scores);
    free(res);
}

typedef struct { float score; size_t idx; } HeapEntry;
static int heap_cmp_desc(const void *a, const void *b) {
    float d = ((const HeapEntry*)b)->score - ((const HeapEntry*)a)->score;
    return (d > 0) ? 1 : ((d < 0) ? -1 : 0);
}

RetrieveResult* index_search(const VectorIndex *idx, const float *query,
                              size_t top_k) {
    RetrieveResult *r = retrieve_create();
    if (idx->count == 0) return r;
    HeapEntry *heap = safe_alloc(idx->count * sizeof(HeapEntry));
    for (size_t i = 0; i < idx->count; i++) {
        float s = (idx->type == INDEX_COSINE)
            ? vec_cosine_sim(query, idx->vectors[i], idx->dim)
            : -vec_l2_dist(query, idx->vectors[i], idx->dim);
        heap[i].score = s;
        heap[i].idx   = i;
    }
    qsort(heap, idx->count, sizeof(HeapEntry), heap_cmp_desc);
    size_t k = top_k < idx->count ? top_k : idx->count;
    for (size_t i = 0; i < k; i++) {
        r->indices[i] = heap[i].idx;
        r->scores[i]  = heap[i].score;
    }
    r->count = k;
    free(heap);
    return r;
}

/* ──────────────────────────────────────────────
   RAG Context
   ────────────────────────────────────────────── */
RagContext* rag_create(const RagConfig *cfg) {
    RagContext *r = safe_alloc(sizeof(RagContext));
    r->config = *cfg;
    r->documents = safe_alloc(RAG_MAX_DOCS * sizeof(Document));
    r->num_docs  = 0;
    r->chunks    = safe_alloc(RAG_MAX_CHUNKS * sizeof(Chunk));
    r->num_chunks = 0;
    r->model      = embed_model_create(cfg->embedding_dim ? cfg->embedding_dim : RAG_DEFAULT_DIM,
                                       EMBED_SIM_SIMPLE_PROJ);
    r->index      = index_create(RAG_MAX_CHUNKS, r->model->dim, INDEX_COSINE);
    r->small_to_big_map = NULL;
    return r;
}

void rag_destroy(RagContext *r) {
    if (!r) return;
    for (size_t i = 0; i < r->num_docs; i++) {
        free(r->documents[i].text);
        free(r->documents[i].source);
        free(r->documents[i].title);
    }
    for (size_t i = 0; i < r->num_chunks; i++) {
        free(r->chunks[i].text);
        free(r->chunks[i].embedding);
        free(r->chunks[i].source);
    }
    free(r->documents);
    free(r->chunks);
    free(r->small_to_big_map);
    embed_model_destroy(r->model);
    index_destroy(r->index);
    free(r);
}

void rag_add_document(RagContext *r, const char *text, size_t text_len,
                      const char *source, const char *title, int page) {
    if (r->num_docs >= RAG_MAX_DOCS) return;
    Document *d = &r->documents[r->num_docs];
    d->text     = safe_alloc(text_len + 1);
    memcpy(d->text, text, text_len);
    d->text[text_len] = '\0';
    d->text_len = text_len;
    d->source   = source ? strdup(source) : NULL;
    d->title    = title  ? strdup(title)  : NULL;
    d->page     = page;
    d->metadata = NULL;
    r->num_docs++;
}

void rag_add_documents(RagContext *r, Document *docs, size_t n) {
    for (size_t i = 0; i < n; i++) {
        rag_add_document(r, docs[i].text, docs[i].text_len,
                         docs[i].source, docs[i].title, docs[i].page);
    }
}

/* ──────────────────────────────────────────────
   Chunking
   ────────────────────────────────────────────── */
void rag_chunk_documents(RagContext *r) {
    r->num_chunks = 0;
    size_t cs = r->config.chunk_size ? r->config.chunk_size : RAG_DEFAULT_CHUNK;
    size_t ov = r->config.chunk_overlap;
    for (size_t d = 0; d < r->num_docs; d++) {
        Document *doc = &r->documents[d];
        size_t pos = 0;
        size_t ci  = 0;
        while (pos < doc->text_len && r->num_chunks < RAG_MAX_CHUNKS) {
            size_t end = pos + cs;
            if (end > doc->text_len) end = doc->text_len;
            Chunk *ch = &r->chunks[r->num_chunks];
            ch->text_len     = end - pos;
            ch->text         = safe_alloc(ch->text_len + 1);
            memcpy(ch->text, doc->text + pos, ch->text_len);
            ch->text[ch->text_len] = '\0';
            ch->doc_index    = d;
            ch->chunk_index  = ci++;
            ch->start_offset = pos;
            ch->end_offset   = end;
            ch->embedding    = NULL;
            ch->embedding_dim = 0;
            ch->source       = doc->source ? strdup(doc->source) : NULL;
            ch->page         = doc->page;
            r->num_chunks++;
            if (end >= doc->text_len) break;
            pos = end - ov;
            if (pos >= end) pos = end;
        }
    }
}

void rag_embed_chunks(RagContext *r) {
    embed_encode_batch(r->model, r->chunks, r->num_chunks);
}

void rag_build_index(RagContext *r) {
    for (size_t i = 0; i < r->num_chunks; i++) {
        index_add(r->index, r->chunks[i].embedding, i);
    }
    index_build(r->index);
}

RetrieveResult* rag_retrieve(RagContext *r, const char *query, size_t query_len) {
    float *qemb = safe_alloc(r->model->dim * sizeof(float));
    embed_encode(r->model, query, query_len, qemb);
    RetrieveResult *res = index_search(r->index, qemb,
                                       r->config.top_k ? r->config.top_k : RAG_DEFAULT_TOPK);
    free(qemb);
    return res;
}

char* rag_build_prompt(const RagContext *r, const RetrieveResult *result,
                       const char *query, const char *system_prompt) {
    size_t cap = 4096;
    char *prompt = safe_alloc(cap);
    size_t off = 0;
    if (system_prompt) {
        off += snprintf(prompt + off, cap - off, "%s\n\n", system_prompt);
    }
    off += snprintf(prompt + off, cap - off, "Context:\n");
    for (size_t i = 0; i < result->count; i++) {
        size_t ci = result->indices[i];
        if (ci < r->num_chunks) {
            off += snprintf(prompt + off, cap - off,
                            "[%zu] %s\n\n", i + 1, r->chunks[ci].text);
        }
    }
    off += snprintf(prompt + off, cap - off,
                    "Question: %s\n\nAnswer:", query);
    return prompt;
}
