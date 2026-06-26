#include "kv_cache.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NUM_HEADS   8
#define HEAD_DIM    64
#define MAX_SEQ     512
#define BLOCK_SIZE  16
#define NUM_LAYERS  4

int main(void) {
    KVC_Cache cache;
    kvc_cache_init(&cache, NUM_LAYERS, NUM_HEADS, HEAD_DIM, MAX_SEQ, BLOCK_SIZE, KVC_DTYPE_FP32);
    printf("KV Cache init: %zu MB allocated\n", kvc_memory_usage(&cache) / (1024 * 1024));

    KVC_BlockSeq seq;
    kvc_seq_init(&seq, 64);

    int per_token = NUM_HEADS * HEAD_DIM;
    float* keys   = malloc(128 * per_token * sizeof(float));
    float* values = malloc(128 * per_token * sizeof(float));
    for (int i = 0; i < 128 * per_token; i++) {
        keys[i]   = (float)(i % 1000) * 0.001f;
        values[i] = (float)(i % 1000) * 0.002f;
    }

    for (int l = 0; l < NUM_LAYERS; l++) {
        kvc_prefill(&cache, l, keys, values, 64, &seq);
    }
    printf("Prefill: 64 tokens → seq_len=%d, num_blocks=%d\n",
           seq.seq_len, seq.num_blocks);

    float* new_key   = malloc(per_token * sizeof(float));
    float* new_value = malloc(per_token * sizeof(float));
    for (int i = 0; i < per_token; i++) {
        new_key[i]   = (float)(i + 100) * 0.001f;
        new_value[i] = (float)(i + 100) * 0.002f;
    }

    for (int step = 0; step < 32; step++) {
        for (int l = 0; l < NUM_LAYERS; l++) {
            kvc_decode(&cache, l, new_key, new_value, &seq);
        }
    }
    printf("Decode: 32 steps → seq_len=%d, num_blocks=%d\n",
           seq.seq_len, seq.num_blocks);

    float* query  = malloc(per_token * sizeof(float));
    float* output = malloc(per_token * sizeof(float));
    for (int i = 0; i < per_token; i++) query[i] = (float)i * 0.0001f;

    kvc_attention(&cache, 0, query, 1, &seq, output);
    printf("Attention: output[0]=%.4f, output[%d]=%.4f\n",
           output[0], per_token - 1, output[per_token - 1]);

    kvc_sliding_window_attention(&cache, 0, query, 1, &seq, 32, output);
    printf("Sliding window (w=32): output[0]=%.4f\n", output[0]);

    kvc_paged_attention(&cache, 0, query, 1, &seq, output);
    printf("PagedAttention: output[0]=%.4f\n", output[0]);

    KVC_PrefixCache pcache;
    kvc_prefix_cache_init(&pcache, 16);
    const char* prefix = "The capital of France is";
    kvc_prefix_cache_insert(&pcache, prefix, strlen(prefix), seq.block_ids, seq.num_blocks);

    int hit = kvc_prefix_cache_lookup(&pcache, prefix, strlen(prefix));
    printf("Prefix cache lookup: hit=%d\n", hit);

    float rope_q[HEAD_DIM], rope_k[HEAD_DIM];
    float rope_freqs[HEAD_DIM / 2];
    for (int j = 0; j < HEAD_DIM / 2; j++) {
        rope_freqs[j] = 1.0f / powf(10000.0f, (2.0f * (float)j) / (float)HEAD_DIM);
    }
    for (int j = 0; j < HEAD_DIM; j++) {
        rope_q[j] = (float)(j + 1);
        rope_k[j] = (float)(j + 100);
    }
    kvc_rope_apply(rope_q, rope_k, HEAD_DIM, 50, rope_freqs);
    printf("RoPE applied at pos 50: q[0]=%.4f, k[0]=%.4f\n", rope_q[0], rope_k[0]);

    kvc_prefix_cache_destroy(&pcache);
    kvc_seq_destroy(&seq);
    kvc_cache_destroy(&cache);

    free(keys); free(values);
    free(new_key); free(new_value);
    free(query); free(output);

    printf("KV Cache example complete.\n");
    return 0;
}
