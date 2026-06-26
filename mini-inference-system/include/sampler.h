#ifndef SAMPLER_H
#define SAMPLER_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#define SAMP_MAX_VOCAB      128256
#define SAMP_MAX_BEAM        16
#define SAMP_MAX_SEQ         4096
#define SAMP_DEFAULT_TEMP    1.0f
#define SAMP_DEFAULT_TOPK    50
#define SAMP_DEFAULT_TOPP    0.9f

typedef enum {
    SAMP_GREEDY      = 0,
    SAMP_TEMPERATURE = 1,
    SAMP_TOP_K       = 2,
    SAMP_TOP_P       = 3,
    SAMP_MIN_P       = 4,
    SAMP_TYPICAL     = 5,
    SAMP_BEAM_SEARCH = 6,
} Samp_Strategy;

typedef struct {
    int     token_id;
    float   log_prob;
    float   cumulative_log_prob;
    int*    token_sequence;
    int     length;
    bool    finished;
} Samp_BeamHypothesis;

typedef struct {
    Samp_BeamHypothesis* beams;
    int     num_beams;
    int     beam_width;
    int     max_length;
    int     eos_token_id;
    float   length_penalty;
    int     vocab_size;
} Samp_BeamSearch;

typedef struct {
    Samp_Strategy strategy;
    float   temperature;
    int     top_k;
    float   top_p;
    float   min_p;
    float   typical_mass;
    int     beam_width;
    float   length_penalty;
    int     eos_token_id;
    int     pad_token_id;
} Samp_Config;

int   samp_greedy(const float* logits, int vocab_size);
int   samp_temperature(const float* logits, int vocab_size, float temperature);
int   samp_top_k(const float* logits, int vocab_size, int k, float temperature);
int   samp_top_p(const float* logits, int vocab_size, float p, float temperature);
int   samp_min_p(const float* logits, int vocab_size, float min_p, float temperature);
int   samp_typical(const float* logits, int vocab_size, float mass, float temperature);
void  samp_softmax(float* probs, const float* logits, int size, float temperature);
void  samp_beam_init(Samp_BeamSearch* bs, int beam_width, int max_length,
                      int eos_token_id, int vocab_size, float length_penalty);
void  samp_beam_destroy(Samp_BeamSearch* bs);
int   samp_beam_step(Samp_BeamSearch* bs, const float* logits, int step);
int   samp_beam_best(const Samp_BeamSearch* bs);
bool  samp_beam_all_finished(const Samp_BeamSearch* bs);
Samp_Config samp_config_default(void);
bool  samp_config_validate(const Samp_Config* cfg);
int   samp_generate(const float* logits_matrix, int seq_len, int vocab_size,
                     int* output_tokens, int max_new_tokens, const Samp_Config* cfg);

#endif
