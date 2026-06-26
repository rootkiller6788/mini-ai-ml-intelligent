#ifndef CHUNKING_STRATEGY_H
#define CHUNKING_STRATEGY_H

#include <stdbool.h>
#include <stddef.h>

/* ──────────────────────────────────────────────
   Chunking Strategy Selection
   ────────────────────────────────────────────── */
typedef enum {
    CHUNK_CHARACTER_SPLIT,
    CHUNK_RECURSIVE_CHAR,
    CHUNK_TOKEN_SPLIT,
    CHUNK_SEMANTIC_SPLIT
} ChunkStrategy;

typedef enum {
    SEP_PARAGRAPH,   /* \n\n       */
    SEP_SENTENCE,    /* .!? + \n   */
    SEP_WORD         /* space      */
} SeparatorLevel;

/* ──────────────────────────────────────────────
   Chunk Configuration
   ────────────────────────────────────────────── */
typedef struct {
    ChunkStrategy strategy;
    size_t        chunk_size;         /* 512–2048 chars / tokens    */
    size_t        chunk_overlap;      /* 10–20% of chunk_size       */
    float         similarity_threshold; /* for semantic splitting    */
    size_t        max_chunks;
    bool          add_metadata;
} ChunkConfig;

/* ──────────────────────────────────────────────
   Chunk Metadata
   ────────────────────────────────────────────── */
typedef struct {
    char   *source;
    char   *title;
    int     page;
    size_t  start_char;
    size_t  end_char;
    size_t  chunk_num;
} ChunkMetadata;

/* ──────────────────────────────────────────────
   Chunk Result
   ────────────────────────────────────────────── */
typedef struct {
    char   *text;
    size_t  text_len;
    ChunkMetadata meta;
} TextChunk;

typedef struct {
    TextChunk *chunks;
    size_t     count;
    size_t     capacity;
} ChunkList;

/* ──────────────────────────────────────────────
   Small-to-Big Retrieval Mapping
   ────────────────────────────────────────────── */
typedef struct {
    size_t  small_id;
    size_t  big_id;
    size_t  offset_in_big;
    bool    is_small_chunk;
} SmallBigMapping;

typedef struct {
    TextChunk       *big_chunks;
    size_t           num_big;
    TextChunk       *small_chunks;
    size_t           num_small;
    SmallBigMapping *mappings;
    size_t           num_mappings;
} SmallBigIndex;

/* ──────────────────────────────────────────────
   Lifecycle
   ────────────────────────────────────────────── */
ChunkList*  chunklist_create(size_t capacity);
void        chunklist_destroy(ChunkList *cl);

/* ──────────────────────────────────────────────
   Character-based Splitting
   ────────────────────────────────────────────── */
ChunkList*  chunk_split_characters(const char *text, size_t text_len,
                                   size_t chunk_size, size_t overlap);

/* ──────────────────────────────────────────────
   Recursive Character Splitting
   ────────────────────────────────────────────── */
ChunkList*  chunk_split_recursive(const char *text, size_t text_len,
                                  size_t chunk_size, size_t overlap);

/* ──────────────────────────────────────────────
   Token-based Splitting
   ────────────────────────────────────────────── */
ChunkList*  chunk_split_tokens(const char *text, size_t text_len,
                               size_t tokens_per_chunk, size_t overlap_tokens);

/* ──────────────────────────────────────────────
   Semantic Splitting (embedding similarity threshold)
   ────────────────────────────────────────────── */
ChunkList*  chunk_split_semantic(const char *text, size_t text_len,
                                 const float *sentence_embeddings,
                                 size_t num_sentences, size_t dim,
                                 float threshold);

/* ──────────────────────────────────────────────
   Chunk with Metadata
   ────────────────────────────────────────────── */
ChunkList*  chunk_with_metadata(const ChunkList *base, const ChunkMetadata *meta);

/* ──────────────────────────────────────────────
   Small-to-Big Index Construction
   ────────────────────────────────────────────── */
SmallBigIndex*  smallbig_create(TextChunk *big_chunks, size_t num_big,
                                size_t small_chunk_size, size_t small_overlap);
void            smallbig_destroy(SmallBigIndex *sbi);
TextChunk*      smallbig_expand(const SmallBigIndex *sbi, size_t small_id);

/* ──────────────────────────────────────────────
   Utility
   ────────────────────────────────────────────── */
size_t  count_tokens_approx(const char *text, size_t text_len);
size_t  count_sentences(const char *text, size_t text_len);

#endif /* CHUNKING_STRATEGY_H */
