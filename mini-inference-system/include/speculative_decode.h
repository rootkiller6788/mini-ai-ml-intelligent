#ifndef SPECULATIVE_DECODE_H
#define SPECULATIVE_DECODE_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#define SD_MAX_CANDIDATES     16
#define SD_MAX_SEQ_LEN        2048
#define SD_MAX_VOCAB_SIZE     128256
#define SD_MAX_MEDUSA_HEADS   8
#define SD_TOP_K              50
#define SD_DEFAULT_GAMMA      5
#define SD_DEFAULT_TEMP       1.0f
#define SD_NGRAM_MAX_N        4

typedef enum {
    SD_DRAFT_NGRAM       = 0,
    SD_DRAFT_SMALL_TF    = 1,
    SD_DRAFT_MEDUSA      = 2,
    SD_DRAFT_EAGLE       = 3,
} SD_DraftType;

typedef struct {
    void*    model;
    int      ngram_n;
    int      max_seq_len;
    float    (*forward)(void* model, const int* input_ids, int len, float* logits);
    void     (*destroy)(void* model);
} SD_NGramModel;

typedef struct {
    void*    model;
    int      num_layers;
    int      hidden_dim;
    int      num_heads;
    int      head_dim;
    int      vocab_size;
    float    (*forward)(void* model, const int* input_ids, int len, float* logits);
    void     (*destroy)(void* model);
} SD_SmallTransformer;

typedef struct {
    float*   weights;
    float*   biases;
    int      hidden_dim;
    int      vocab_size;
} SD_MedusaHead;

typedef struct {
    SD_MedusaHead* heads;
    int      num_heads;
    int      hidden_dim;
    int      vocab_size;
} SD_MedusaModel;

typedef struct {
    void*    model;
    void*    tokenizer;
    int      vocab_size;
    int      max_seq_len;
    float    temperature;
    SD_DraftType draft_type;
    SD_NGramModel ngram;
    SD_SmallTransformer small_tf;
    SD_MedusaModel medusa;
    int      candidate_count;
    float    acceptance_rate;
    double   avg_speedup;
    int      total_candidates;
    int      total_accepted;
} SD_DraftModel;

typedef struct {
    void*    model;
    int      vocab_size;
    int      max_seq_len;
    float    (*forward)(void* model, const int* input_ids, int len, float* logits);
    void     (*destroy)(void* model);
} SD_TargetModel;

typedef struct {
    int*     tokens;
    int      count;
    int      capacity;
} SD_CandidateSequence;

typedef struct {
    SD_DraftModel  draft;
    SD_TargetModel target;
    SD_CandidateSequence candidates;
    int      gamma;
    float    temperature;
    int      top_k;
    float    top_p;
    double   total_time_draft_ms;
    double   total_time_target_ms;
    int      total_steps;
    int      total_tokens_generated;
} SD_SpeculativeDecoder;

bool  sd_decoder_init(SD_SpeculativeDecoder* decoder, SD_DraftType draft_type, int gamma);
void  sd_decoder_destroy(SD_SpeculativeDecoder* decoder);

void  sd_draft_ngram_init(SD_NGramModel* model, int n);
void  sd_draft_ngram_destroy(SD_NGramModel* model);
void  sd_draft_ngram_update(SD_NGramModel* model, const int* tokens, int len);
int   sd_draft_ngram_predict(const SD_NGramModel* model, const int* context, int ctx_len,
                              int* candidates, int max_candidates);

void  sd_draft_small_tf_init(SD_SmallTransformer* model, int num_layers,
                              int hidden_dim, int num_heads, int head_dim, int vocab_size);
void  sd_draft_small_tf_destroy(SD_SmallTransformer* model);
int   sd_draft_small_tf_generate(SD_SmallTransformer* model, const int* input_ids,
                                  int input_len, int* candidates, int num_candidates,
                                  float temperature);

void  sd_medusa_init(SD_MedusaModel* model, int num_heads, int hidden_dim, int vocab_size);
void  sd_medusa_destroy(SD_MedusaModel* model);
int   sd_medusa_verify(SD_MedusaModel* model, const float* hidden_states,
                        int* accepted_tokens, int max_accept);

void  sd_draft_generate(SD_SpeculativeDecoder* decoder, const int* prefix, int prefix_len);

void  sd_target_verify(SD_SpeculativeDecoder* decoder, const int* prefix, int prefix_len,
                        int* draft_tokens, int* accepted_tokens, int* num_accepted);

int   sd_speculative_step(SD_SpeculativeDecoder* decoder, const int* prefix, int prefix_len,
                           int* output_tokens, int max_new_tokens);

int   sd_rejection_sample(const float* draft_logits, const float* target_logits,
                           int vocab_size, const int* draft_tokens, int num_draft,
                           int* accepted_tokens, float temperature);

int   sd_logit_compare(const float* draft_logits, const float* target_logits,
                        int vocab_size, int draft_token, int target_token);

float sd_acceptance_probability(const float* draft_probs, const float* target_probs,
                                  int vocab_size, int draft_token);

double sd_speedup_estimate(const SD_SpeculativeDecoder* decoder);
double sd_wall_time_improvement(const SD_SpeculativeDecoder* decoder);

float sd_softmax_sample(const float* logits, int size, float temperature, int* chosen);
void  sd_top_k_filter(float* logits, int size, int k);
void  sd_top_p_filter(float* logits, int size, float p);
void  sd_temperature_scale(float* logits, int size, float temperature);

int   sd_int_array_hash(const int* arr, int len);
void  sd_tree_attention(const float* query, const float* keys, const float* values,
                         int* tree_structure, int num_nodes, int head_dim, float* output);

#endif
