#ifndef AUDIO_WHISPER_H
#define AUDIO_WHISPER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MM_WHISPER_SAMPLE_RATE     16000
#define MM_WHISPER_N_MELS          80
#define MM_WHISPER_HOP_LENGTH      160
#define MM_WHISPER_N_FFT           400
#define MM_WHISPER_MAX_AUDIO_SEC   30
#define MM_WHISPER_MAX_AUDIO_LEN   (MM_WHISPER_SAMPLE_RATE * MM_WHISPER_MAX_AUDIO_SEC)
#define MM_WHISPER_MAX_MEL_FRAMES  3000
#define MM_WHISPER_ENCODER_DIM     512
#define MM_WHISPER_ENCODER_HEADS   8
#define MM_WHISPER_ENCODER_LAYERS  6
#define MM_WHISPER_DECODER_DIM     512
#define MM_WHISPER_DECODER_HEADS   8
#define MM_WHISPER_DECODER_LAYERS  6
#define MM_WHISPER_FFN_DIM         2048
#define MM_WHISPER_VOCAB_SIZE      51865
#define MM_WHISPER_MAX_DEC_LEN     448
#define MM_WHISPER_NUM_LANGS       99
#define MM_WHISPER_NUM_TASKS       3

typedef enum {
    MM_WHISPER_TASK_TRANSCRIBE   = 0,
    MM_WHISPER_TASK_TRANSLATE    = 1,
    MM_WHISPER_TASK_LANG_DETECT  = 2,
} mm_whisper_task_t;

typedef enum {
    MM_WHISPER_TOKEN_SOT         = 50257,
    MM_WHISPER_TOKEN_SOT_PREV    = 50360,
    MM_WHISPER_TOKEN_SOLM        = 50361,
    MM_WHISPER_TOKEN_NOT         = 50362,
    MM_WHISPER_TOKEN_BEG_TIM     = 50363,
    MM_WHISPER_TOKEN_NO_TIM      = 50364,
    MM_WHISPER_TOKEN_LANG_BASE   = 50258,
    MM_WHISPER_TOKEN_TRANSLATE   = 50358,
    MM_WHISPER_TOKEN_TRANSCRIBE  = 50359,
    MM_WHISPER_TOKEN_EOT         = 50256,
    MM_WHISPER_TOKEN_TIME_BASE   = 50364,
} mm_whisper_special_token_t;

typedef struct {
    int     start_ms;
    int     end_ms;
    char    text[256];
    float   prob;
} mm_whisper_segment_t;

typedef struct {
    mm_whisper_segment_t* segments;
    int                   num_segments;
    int                   max_segments;
    char                  detected_lang[8];
    float                 lang_prob;
} mm_whisper_result_t;

typedef struct {
    float* data;
    int    n_frames;
    int    n_mels;
} mm_mel_spectrogram_t;

typedef struct {
    float*           mel_filters;
    int              n_mels;
    int              n_fft;
    int              hop_length;
    int              sample_rate;
} mm_mel_filterbank_t;

typedef struct {
    float*           qkv_weight;
    float*           qkv_bias;
    float*           proj_weight;
    float*           proj_bias;
    int              num_heads;
    int              head_dim;
    int              dim;
} mm_whisper_attn_t;

typedef struct {
    float*           q_weight;
    float*           q_bias;
    float*           k_weight;
    float*           k_bias;
    float*           v_weight;
    float*           v_bias;
    float*           proj_weight;
    float*           proj_bias;
    int              num_heads;
    int              head_dim;
    int              dim;
    int              cross_dim;
} mm_whisper_cross_attn_t;

typedef struct {
    float*           fc1_weight;
    float*           fc1_bias;
    float*           fc2_weight;
    float*           fc2_bias;
    int              dim;
    int              ffn_dim;
} mm_whisper_ffn_t;

typedef struct {
    mm_whisper_attn_t self_attn;
    mm_whisper_ffn_t  ffn;
    float*            ln1_weight;
    float*            ln1_bias;
    float*            ln2_weight;
    float*            ln2_bias;
    int               dim;
} mm_whisper_encoder_layer_t;

typedef struct {
    mm_whisper_attn_t        self_attn;
    mm_whisper_cross_attn_t  cross_attn;
    mm_whisper_ffn_t         ffn;
    float*                   ln1_weight;
    float*                   ln1_bias;
    float*                   ln2_weight;
    float*                   ln2_bias;
    float*                   ln3_weight;
    float*                   ln3_bias;
    int                      dim;
} mm_whisper_decoder_layer_t;

typedef struct {
    mm_conv2d_t                conv1;
    mm_conv2d_t                conv2;
    float*                     pos_embed;
    mm_whisper_encoder_layer_t* layers;
    float*                     ln_post_weight;
    float*                     ln_post_bias;
    int                        num_layers;
    int                        dim;
} mm_whisper_encoder_t;

typedef struct {
    mm_linear_t                token_embed;
    float*                     pos_embed;
    mm_whisper_decoder_layer_t* layers;
    float*                     ln_final_weight;
    float*                     ln_final_bias;
    mm_linear_t                lm_head;
    int                        num_layers;
    int                        dim;
    int                        vocab_size;
    int                        max_seq_len;
} mm_whisper_decoder_t;

typedef struct {
    mm_whisper_encoder_t   encoder;
    mm_whisper_decoder_t   decoder;
    mm_mel_filterbank_t    mel_filterbank;
    int                    sample_rate;
    int                    n_mels;
    int                    encoder_dim;
    int                    decoder_dim;
} mm_whisper_model_t;

void mm_mel_filterbank_init(mm_mel_filterbank_t* fbank, int n_mels, int n_fft,
                            int sample_rate);
void mm_mel_filterbank_free(mm_mel_filterbank_t* fbank);

void mm_audio_stft(const float* audio, int audio_len, int n_fft, int hop_len,
                   float* spec_real, float* spec_imag, int* n_frames, int* n_freqs);
void mm_audio_mel_spectrogram(const float* audio, int audio_len,
                              const mm_mel_filterbank_t* fbank,
                              mm_mel_spectrogram_t* mel);
void mm_mel_spectrogram_free(mm_mel_spectrogram_t* mel);

float mm_vad_energy(const float* audio, int len);
int   mm_vad_is_speech(const float* audio, int len, float threshold_ms);
void  mm_vad_split(const float* audio, int total_len, int chunk_ms,
                   float threshold, int** speech_starts, int** speech_ends,
                   int* num_segments);

void mm_whisper_model_init(mm_whisper_model_t* model, int encoder_dim,
                           int decoder_dim, int encoder_layers,
                           int decoder_layers, int n_mels);
void mm_whisper_model_free(mm_whisper_model_t* model);

void mm_whisper_encoder_forward(const mm_whisper_encoder_t* enc,
                                const mm_mel_spectrogram_t* mel,
                                float* encoder_hidden);
void mm_whisper_decoder_forward(const mm_whisper_decoder_t* dec,
                                const float* encoder_hidden, int enc_len,
                                const int* prev_tokens, int num_prev,
                                float* logits);

void mm_whisper_transcribe(const mm_whisper_model_t* model,
                           const float* audio, int audio_len,
                           const char* language,
                           mm_whisper_task_t task,
                           mm_whisper_result_t* result);
void mm_whisper_translate(const mm_whisper_model_t* model,
                          const float* audio, int audio_len,
                          mm_whisper_result_t* result);
void mm_whisper_detect_language(const mm_whisper_model_t* model,
                                const float* audio, int audio_len,
                                char* lang, float* prob);

void mm_whisper_result_init(mm_whisper_result_t* result, int max_segments);
void mm_whisper_result_free(mm_whisper_result_t* result);

void mm_whisper_attn_init(mm_whisper_attn_t* attn, int dim, int num_heads);
void mm_whisper_attn_free(mm_whisper_attn_t* attn);
void mm_whisper_attn_forward(const mm_whisper_attn_t* attn, const float* x,
                             int seq_len, float* out);

void mm_whisper_cross_attn_init(mm_whisper_cross_attn_t* attn, int dim,
                                int cross_dim, int num_heads);
void mm_whisper_cross_attn_free(mm_whisper_cross_attn_t* attn);
void mm_whisper_cross_attn_forward(const mm_whisper_cross_attn_t* attn,
                                   const float* x, const float* context,
                                   int seq_len, int ctx_len, float* out);

void mm_whisper_encoder_layer_init(mm_whisper_encoder_layer_t* layer, int dim,
                                   int num_heads, int ffn_dim);
void mm_whisper_encoder_layer_free(mm_whisper_encoder_layer_t* layer);
void mm_whisper_encoder_layer_forward(const mm_whisper_encoder_layer_t* layer,
                                      const float* x, int seq_len, float* out);

void mm_whisper_decoder_layer_init(mm_whisper_decoder_layer_t* layer, int dim,
                                   int num_heads, int ffn_dim);
void mm_whisper_decoder_layer_free(mm_whisper_decoder_layer_t* layer);
void mm_whisper_decoder_layer_forward(const mm_whisper_decoder_layer_t* layer,
                                      const float* x, const float* enc_out,
                                      int seq_len, int enc_len, float* out);

void mm_whisper_encode_token(int timestamp_ms, int* token);
int  mm_whisper_decode_time_token(int token);
const char* mm_whisper_lang_name(int lang_id);
int  mm_whisper_lang_id(const char* lang);

void mm_whisper_ffn_init(mm_whisper_ffn_t* ffn, int dim, int ffn_dim);
void mm_whisper_ffn_free(mm_whisper_ffn_t* ffn);
void mm_whisper_ffn_forward(const mm_whisper_ffn_t* ffn, const float* x,
                            int seq_len, float* out);

#ifdef __cplusplus
}
#endif

#endif
