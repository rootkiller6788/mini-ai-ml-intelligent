#include "chunking_strategy.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* ──────────────────────────────────────────────
   Internal helpers
   ────────────────────────────────────────────── */
static void *safe_alloc(size_t n) {
    void *p = calloc(n, 1);
    if (!p) { fprintf(stderr, "OOM\n"); exit(1); }
    return p;
}

static bool is_separator(char c, SeparatorLevel level) {
    switch (level) {
    case SEP_PARAGRAPH: return c == '\n';
    case SEP_SENTENCE:  return c == '.' || c == '!' || c == '?' || c == '\n';
    case SEP_WORD:      return c == ' ' || c == '\t' || c == '\n';
    default:            return false;
    }
}

static size_t find_nearest_sep(const char *text, size_t len,
                                size_t target, SeparatorLevel level) {
    if (target >= len) return len;
    for (size_t off = 0; off < len; off++) {
        size_t left  = (target > off) ? target - off : 0;
        size_t right = target + off;
        if (right < len && is_separator(text[right], level)) return right;
        if (left > 0 && is_separator(text[left], level))  return left;
    }
    return len;
}

/* ──────────────────────────────────────────────
   ChunkList
   ────────────────────────────────────────────── */
ChunkList* chunklist_create(size_t capacity) {
    ChunkList *cl = safe_alloc(sizeof(ChunkList));
    cl->capacity = capacity ? capacity : 256;
    cl->chunks   = safe_alloc(cl->capacity * sizeof(TextChunk));
    cl->count    = 0;
    return cl;
}

void chunklist_destroy(ChunkList *cl) {
    if (!cl) return;
    for (size_t i = 0; i < cl->count; i++) {
        free(cl->chunks[i].text);
        free(cl->chunks[i].meta.source);
        free(cl->chunks[i].meta.title);
    }
    free(cl->chunks);
    free(cl);
}

static void chunklist_add(ChunkList *cl, const char *text, size_t len,
                           size_t start, size_t end, size_t num) {
    if (cl->count >= cl->capacity) {
        cl->capacity *= 2;
        cl->chunks = realloc(cl->chunks, cl->capacity * sizeof(TextChunk));
        if (!cl->chunks) { fprintf(stderr, "ChunkList OOM\n"); exit(1); }
    }
    TextChunk *ch = &cl->chunks[cl->count];
    ch->text     = safe_alloc(len + 1);
    memcpy(ch->text, text, len);
    ch->text[len] = '\0';
    ch->text_len  = len;
    ch->meta.source     = NULL;
    ch->meta.title      = NULL;
    ch->meta.page       = 0;
    ch->meta.start_char = start;
    ch->meta.end_char   = end;
    ch->meta.chunk_num  = num;
    cl->count++;
}

/* ──────────────────────────────────────────────
   Character Split
   ────────────────────────────────────────────── */
ChunkList* chunk_split_characters(const char *text, size_t text_len,
                                   size_t chunk_size, size_t overlap) {
    ChunkList *cl = chunklist_create((text_len / chunk_size) + 4);
    size_t pos = 0, ci = 0;
    while (pos < text_len) {
        size_t end = pos + chunk_size;
        if (end > text_len) end = text_len;
        chunklist_add(cl, text + pos, end - pos, pos, end, ci++);
        if (end >= text_len) break;
        pos = end - overlap;
        if (pos >= end) pos = end;
    }
    return cl;
}

/* ──────────────────────────────────────────────
   Recursive Character Split
   ────────────────────────────────────────────── */
static void recursive_split_inner(const char *text, size_t len,
                                   size_t chunk_size, size_t overlap,
                                   size_t base_offset, size_t *ci,
                                   ChunkList *cl) {
    if (len <= chunk_size) {
        chunklist_add(cl, text, len, base_offset, base_offset + len, (*ci)++);
        return;
    }
    /* Try paragraph boundaries first */
    size_t sep = find_nearest_sep(text, len, chunk_size, SEP_PARAGRAPH);
    if (sep > 0 && sep < len) {
        recursive_split_inner(text, sep, chunk_size, overlap,
                              base_offset, ci, cl);
        size_t next_start = (sep + 1 < len) ? sep + 1 : sep;
        recursive_split_inner(text + next_start, len - next_start,
                              chunk_size, overlap,
                              base_offset + next_start, ci, cl);
        return;
    }
    /* Try sentence boundaries */
    sep = find_nearest_sep(text, len, chunk_size - overlap, SEP_SENTENCE);
    if (sep > 0 && sep < len) {
        size_t end_pos = sep + 1;
        chunklist_add(cl, text, end_pos, base_offset,
                      base_offset + end_pos, (*ci)++);
        size_t next = (end_pos > overlap) ? end_pos - overlap : 0;
        recursive_split_inner(text + next, len - next,
                              chunk_size, overlap,
                              base_offset + next, ci, cl);
        return;
    }
    /* Fallback: split at word boundary or character */
    sep = find_nearest_sep(text, len, chunk_size, SEP_WORD);
    if (sep == len || sep == 0) sep = chunk_size;
    chunklist_add(cl, text, sep, base_offset, base_offset + sep, (*ci)++);
    if (sep < len) {
        size_t next = (sep > overlap) ? sep - overlap : 0;
        recursive_split_inner(text + next, len - next,
                              chunk_size, overlap,
                              base_offset + next, ci, cl);
    }
}

ChunkList* chunk_split_recursive(const char *text, size_t text_len,
                                  size_t chunk_size, size_t overlap) {
    ChunkList *cl = chunklist_create(64);
    size_t ci = 0;
    recursive_split_inner(text, text_len, chunk_size, overlap, 0, &ci, cl);
    return cl;
}

/* ──────────────────────────────────────────────
   Token-based Split
   ────────────────────────────────────────────── */
size_t count_tokens_approx(const char *text, size_t text_len) {
    size_t tokens = 0;
    bool in_word = false;
    for (size_t i = 0; i < text_len; i++) {
        if (text[i] == ' ' || text[i] == '\t' || text[i] == '\n') {
            if (in_word) { tokens++; in_word = false; }
        } else {
            in_word = true;
        }
    }
    if (in_word) tokens++;
    return tokens;
}

size_t count_sentences(const char *text, size_t text_len) {
    size_t sentences = 0;
    for (size_t i = 0; i < text_len; i++) {
        if (text[i] == '.' || text[i] == '!' || text[i] == '?') sentences++;
    }
    return sentences;
}

ChunkList* chunk_split_tokens(const char *text, size_t text_len,
                               size_t tokens_per_chunk, size_t overlap_tokens) {
    ChunkList *cl = chunklist_create(64);
    size_t pos = 0, ci = 0;
    while (pos < text_len) {
        size_t tcount = 0, end = pos;
        bool in_word = false;
        while (end < text_len && tcount < tokens_per_chunk) {
            if (text[end] == ' ' || text[end] == '\t' || text[end] == '\n') {
                if (in_word) { tcount++; in_word = false; }
            } else {
                in_word = true;
            }
            end++;
        }
        if (in_word) tcount++;
        chunklist_add(cl, text + pos, end - pos, pos, end, ci++);
        if (end >= text_len) break;
        size_t back = end;
        size_t ot = 0;
        while (back > pos && ot < overlap_tokens) {
            back--;
            if (text[back] == ' ' || text[back] == '\t' || text[back] == '\n') ot++;
        }
        pos = back;
    }
    return cl;
}

/* ──────────────────────────────────────────────
   Semantic Split (embedding similarity threshold)
   ────────────────────────────────────────────── */
ChunkList* chunk_split_semantic(const char *text, size_t text_len,
                                 const float *sentence_embeddings,
                                 size_t num_sentences, size_t dim,
                                 float threshold) {
    ChunkList *cl = chunklist_create(num_sentences + 4);
    size_t start = 0, ci = 0;
    for (size_t i = 1; i < num_sentences; i++) {
        float sim = 0.0f;
        for (size_t d = 0; d < dim; d++) {
            sim += sentence_embeddings[(i - 1) * dim + d] *
                   sentence_embeddings[i * dim + d];
        }
        if (sim < threshold) {
            size_t end = 0;
            size_t sc = 0;
            for (size_t p = 0; p < text_len && sc <= i; p++) {
                if (text[p] == '.' || text[p] == '!' || text[p] == '?') sc++;
                if (sc == i) end = p + 1;
            }
            if (end > 0 && end > start) {
                chunklist_add(cl, text + start, end - start, start, end, ci++);
            }
            start = end;
        }
    }
    /* Last chunk from last boundary to end */
    if (start < text_len) {
        chunklist_add(cl, text + start, text_len - start,
                      start, text_len, ci++);
    }
    /* If no splits occurred, return single chunk */
    if (cl->count == 0) {
        chunklist_add(cl, text, text_len, 0, text_len, 0);
    }
    return cl;
}

/* ──────────────────────────────────────────────
   Chunk with Metadata
   ────────────────────────────────────────────── */
ChunkList* chunk_with_metadata(const ChunkList *base, const ChunkMetadata *meta) {
    if (!base || !meta) return NULL;
    for (size_t i = 0; i < base->count; i++) {
        TextChunk *ch = &base->chunks[i];
        if (meta->source) ch->meta.source = strdup(meta->source);
        if (meta->title)  ch->meta.title  = strdup(meta->title);
        ch->meta.page = meta->page;
    }
    return (ChunkList*)base;
}

/* ──────────────────────────────────────────────
   Small-to-Big Index
   ────────────────────────────────────────────── */
SmallBigIndex* smallbig_create(TextChunk *big_chunks, size_t num_big,
                                size_t small_chunk_size, size_t small_overlap) {
    SmallBigIndex *sbi = safe_alloc(sizeof(SmallBigIndex));
    sbi->big_chunks    = big_chunks;
    sbi->num_big       = num_big;
    /* Create small chunks from all big chunks */
    size_t total_small = 0;
    for (size_t i = 0; i < num_big; i++) {
        total_small += (big_chunks[i].text_len / small_chunk_size) + 1;
    }
    sbi->small_chunks  = safe_alloc(total_small * sizeof(TextChunk));
    sbi->mappings      = safe_alloc(total_small * sizeof(SmallBigMapping));
    sbi->num_small     = 0;
    sbi->num_mappings  = 0;
    for (size_t bi = 0; bi < num_big; bi++) {
        size_t pos = 0;
        while (pos < big_chunks[bi].text_len) {
            size_t end = pos + small_chunk_size;
            if (end > big_chunks[bi].text_len) end = big_chunks[bi].text_len;
            TextChunk *sc = &sbi->small_chunks[sbi->num_small];
            sc->text     = safe_alloc(end - pos + 1);
            memcpy(sc->text, big_chunks[bi].text + pos, end - pos);
            sc->text[end - pos] = '\0';
            sc->text_len = end - pos;
            SmallBigMapping *m = &sbi->mappings[sbi->num_mappings++];
            m->small_id       = sbi->num_small;
            m->big_id         = bi;
            m->offset_in_big  = pos;
            m->is_small_chunk = true;
            sbi->num_small++;
            if (end >= big_chunks[bi].text_len) break;
            pos = end - small_overlap;
            if (pos >= end) pos = end;
        }
    }
    return sbi;
}

void smallbig_destroy(SmallBigIndex *sbi) {
    if (!sbi) return;
    for (size_t i = 0; i < sbi->num_small; i++) free(sbi->small_chunks[i].text);
    free(sbi->small_chunks);
    free(sbi->mappings);
    free(sbi);
}

TextChunk* smallbig_expand(const SmallBigIndex *sbi, size_t small_id) {
    if (!sbi || small_id >= sbi->num_mappings) return NULL;
    SmallBigMapping *m = &sbi->mappings[small_id];
    return &sbi->big_chunks[m->big_id];
}
