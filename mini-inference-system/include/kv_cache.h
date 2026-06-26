#ifndef KV_CACHE_H
#define KV_CACHE_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#define KVC_MAX_BATCH_SIZE     256
#define KVC_MAX_SEQ_LEN        32768
#define KVC_MAX_HEADS          64
#define KVC_MAX_HEAD_DIM       256
#define KVC_MAX_BLOCKS         4096
#define KVC_BLOCK_SIZE         16
#define KVC_MAX_LAYERS         128
#define KVC_DEFAULT_BLOCK_SIZE 16
#define KVC_MIN_BLOCK_SIZE     8

typedef enum {
    KVC_DTYPE_FP32 = 0,
    KVC_DTYPE_FP16 = 1,
    KVC_DTYPE_FP8  = 2,
    KVC_DTYPE_INT8 = 3,
} KVC_DataType;

typedef enum {
    KVC_ATTN_VANILLA      = 0,
    KVC_ATTN_SLIDING_WINDOW = 1,
    KVC_ATTN_PAGED        = 2,
} KVC_AttentionType;

typedef struct {
    float*   data;
    int      head_count;
    int      head_dim;
    int      seq_len;
    KVC_DataType dtype;
    size_t   bytes_alloc;
} KVC_Tensor;

typedef struct {
    KVC_Tensor* key;
    KVC_Tensor* value;
    int      block_idx;
    int      token_count;
    bool     in_use;
} KVC_Block;

typedef struct {
    KVC_Block* blocks;
    int      num_blocks;
    int      block_size;
    int      free_blocks;
    int*     free_list;
    int      free_list_top;
} KVC_BlockTable;

typedef struct {
    int*     block_ids;
    int      num_blocks;
    int      seq_len;
    bool     in_use;
} KVC_BlockSeq;

typedef struct {
    int      num_heads;
    int      head_dim;
    int      max_seq_len;
    KVC_BlockTable* tables;
    int      num_layers;
    int      num_tables;
    KVC_DataType dtype;
    KVC_AttentionType attn_type;
    int      sliding_window;
    size_t   total_bytes;
    bool     quantize_cache;
    KVC_DataType quant_dtype;
} KVC_Cache;

typedef struct {
    char*    prefix_text;
    size_t   prefix_len;
    int*     block_ids;
    int      num_blocks;
    int      hash;
    int      ref_count;
    double   last_access;
} KVC_PrefixEntry;

typedef struct {
    KVC_PrefixEntry* entries;
    int      num_entries;
    int      capacity;
    pthread_mutex_t mutex;
} KVC_PrefixCache;

void  kvc_cache_init(KVC_Cache* cache, int num_layers, int num_heads, int head_dim,
                      int max_seq_len, int block_size, KVC_DataType dtype);
void  kvc_cache_destroy(KVC_Cache* cache);

int   kvc_block_alloc(KVC_Cache* cache, int layer);
void  kvc_block_free(KVC_Cache* cache, int layer, int block_id);
void  kvc_block_free_seq(KVC_Cache* cache, int layer, KVC_BlockSeq* seq);

bool  kvc_kv_append(KVC_Cache* cache, int layer, float* key_data, float* value_data,
                     int num_tokens, KVC_BlockSeq* seq);

void  kvc_prefill(KVC_Cache* cache, int layer, const float* key_data,
                   const float* value_data, int seq_len, KVC_BlockSeq* seq);
void  kvc_decode(KVC_Cache* cache, int layer, const float* new_key,
                  const float* new_value, KVC_BlockSeq* seq);

void  kvc_attention(const KVC_Cache* cache, int layer, const float* query,
                     int query_len, const KVC_BlockSeq* seq, float* output);

void  kvc_sliding_window_attention(const KVC_Cache* cache, int layer,
                                    const float* query, int query_len,
                                    const KVC_BlockSeq* seq, int window_size,
                                    float* output);

void  kvc_paged_attention(const KVC_Cache* cache, int layer, const float* query,
                           int query_len, const KVC_BlockSeq* seq, float* output);

void  kvc_attention_with_rope(const KVC_Cache* cache, int layer, const float* query,
                               int query_len, const KVC_BlockSeq* seq,
                               const float* rope_freqs, float* output);

void  kvc_quantize_block(KVC_Cache* cache, int layer, int block_id, KVC_DataType dst_dtype);
void  kvc_dequantize_block(const KVC_Cache* cache, int layer, int block_id, float* output);
void  kvc_quantize_all(KVC_Cache* cache, KVC_DataType dst_dtype);

void  kvc_prefix_cache_init(KVC_PrefixCache* pcache, int capacity);
void  kvc_prefix_cache_destroy(KVC_PrefixCache* pcache);
int   kvc_prefix_cache_lookup(KVC_PrefixCache* pcache, const char* prefix, size_t len);
void  kvc_prefix_cache_insert(KVC_PrefixCache* pcache, const char* prefix, size_t len,
                                int* block_ids, int num_blocks);
bool  kvc_prefix_share_blocks(KVC_Cache* cache, KVC_PrefixCache* pcache,
                               const char* prompt, KVC_BlockSeq* seq);

size_t kvc_memory_usage(const KVC_Cache* cache);
int    kvc_num_tokens_cached(const KVC_Cache* cache, int layer, const KVC_BlockSeq* seq);
float  kvc_cache_hit_rate(const KVC_Cache* cache);
void   kvc_evict_oldest(KVC_Cache* cache, int layer, KVC_BlockSeq* seq, int num_to_remove);

void  kvc_seq_init(KVC_BlockSeq* seq, int max_blocks);
void  kvc_seq_destroy(KVC_BlockSeq* seq);

void  kvc_rope_apply(float* q, float* k, int head_dim, int seq_pos, const float* freqs);
void  kvc_flash_attention(const float* Q, const float* K, const float* V,
                           float* O, int seq_len, int num_heads, int head_dim);

#endif
