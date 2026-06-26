#ifndef VLM_LLAMA_H
#define VLM_LLAMA_H

#include <stddef.h>
#include <stdint.h>
#include "clip_contrastive.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MM_VLM_IMAGE_SIZE      336
#define MM_VLM_PATCH_SIZE      14
#define MM_VLM_NUM_PATCHES     ((MM_VLM_IMAGE_SIZE / MM_VLM_PATCH_SIZE) * (MM_VLM_IMAGE_SIZE / MM_VLM_PATCH_SIZE))
#define MM_VLM_VISION_DIM      1024
#define MM_VLM_LLM_DIM         4096
#define MM_VLM_MAX_SEQ_LEN     4096
#define MM_VLM_NUM_LAYERS      32
#define MM_VLM_NUM_HEADS       32
#define MM_VLM_HEAD_DIM        (MM_VLM_LLM_DIM / MM_VLM_NUM_HEADS)
#define MM_VLM_FFN_DIM         11008
#define MM_VLM_VOCAB_SIZE      32000
#define MM_VLM_NUM_IMG_TOKENS  (MM_VLM_NUM_PATCHES)
#define MM_VLM_IMAGE_TOKEN_ID  32001
#define MM_VLM_IM_START_ID     32002
#define MM_VLM_IM_END_ID       32003

typedef struct {
    int*   tokens;
    size_t len;
    size_t cap;
} mm_vlm_token_seq_t;

typedef struct {
    float  x, y, w, h;
} mm_vlm_bbox_t;

typedef enum {
    MM_VLM_MSG_USER       = 0,
    MM_VLM_MSG_ASSISTANT  = 1,
    MM_VLM_MSG_SYSTEM     = 2,
} mm_vlm_msg_role_t;

typedef struct {
    char*             text;
    float*            image_features;
    int               has_image;
    mm_vlm_bbox_t*    bboxes;
    int               num_bboxes;
    mm_vlm_msg_role_t role;
} mm_vlm_message_t;

typedef struct {
    mm_vlm_message_t* messages;
    int               num_messages;
    int               max_messages;
} mm_vlm_conversation_t;

typedef struct {
    float* weight;
    float* bias;
    int    in_dim, out_dim;
} mm_vlm_linear_t;

typedef struct {
    float*           qkv_weight;
    float*           qkv_bias;
    float*           proj_weight;
    float*           proj_bias;
    float*           rope_freqs;
    int              num_heads;
    int              head_dim;
    int              dim;
    int              max_seq_len;
    float            rope_theta;
} mm_vlm_attention_t;

typedef struct {
    mm_vlm_linear_t  gate_proj;
    mm_vlm_linear_t  up_proj;
    mm_vlm_linear_t  down_proj;
    int              dim;
    int              hidden_dim;
} mm_vlm_ffn_t;

typedef struct {
    mm_vlm_attention_t  self_attn;
    mm_vlm_ffn_t        ffn;
    float*              input_ln_weight;
    float*              post_attn_ln_weight;
    int                 dim;
    int                 layer_idx;
} mm_vlm_decoder_layer_t;

typedef struct {
    mm_vlm_linear_t           token_embed;
    mm_vlm_decoder_layer_t*   layers;
    float*                    final_ln_weight;
    mm_vlm_linear_t           lm_head;
    int                       num_layers;
    int                       dim;
    int                       vocab_size;
    int                       max_seq_len;
    int                       num_heads;
    int                       head_dim;
} mm_vlm_llm_t;

typedef struct {
    mm_vlm_linear_t           proj;
    int                       num_layers;
    int                       num_patches;
    int                       vision_dim;
    int                       llm_dim;
} mm_vlm_projection_t;

typedef struct {
    mm_image_encoder_t     vision_encoder;
    mm_vlm_projection_t    projector;
    mm_vlm_llm_t           llm;
    int                    vision_dim;
    int                    llm_dim;
    int                    image_size;
    int                    patch_size;
    int                    num_patches;
    float                  temperature;
    int                    top_p;
    int                    top_k;
    int                    max_new_tokens;
} mm_vlm_model_t;

void mm_vlm_token_seq_init(mm_vlm_token_seq_t* seq, size_t cap);
void mm_vlm_token_seq_free(mm_vlm_token_seq_t* seq);
void mm_vlm_token_seq_push(mm_vlm_token_seq_t* seq, int token);
void mm_vlm_token_seq_append(mm_vlm_token_seq_t* seq, const int* tokens, size_t n);

void mm_vlm_conversation_init(mm_vlm_conversation_t* conv, int max_msgs);
void mm_vlm_conversation_free(mm_vlm_conversation_t* conv);
void mm_vlm_conversation_add(mm_vlm_conversation_t* conv, const char* text,
                             mm_vlm_msg_role_t role, int has_image);

void mm_vlm_model_init(mm_vlm_model_t* model, int vision_dim, int llm_dim,
                       int num_v_layers, int num_llm_layers);
void mm_vlm_model_free(mm_vlm_model_t* model);

void mm_vlm_encode_image(const mm_vlm_model_t* model, const float* image,
                         int h, int w, int c, float* image_features);
void mm_vlm_project_features(const mm_vlm_projection_t* proj,
                             const float* vision_features, int num_patches,
                             float* projected_features);

void mm_vlm_prepare_input(const mm_vlm_model_t* model,
                          const mm_vlm_conversation_t* conv,
                          const float* image_features,
                          int* input_ids, int* num_input_ids,
                          int* image_pos);

int  mm_vlm_generate_token(const mm_vlm_model_t* model, const int* input_ids,
                           int seq_len, int pos);
void mm_vlm_generate(const mm_vlm_model_t* model, const int* input_ids,
                     int seq_len, int max_new_tokens,
                     int* output_ids, int* num_output);

const char* mm_vlm_decode_token(int token_id);
const char* mm_vlm_decode(const int* token_ids, int n);

void mm_vlm_visual_qa(const mm_vlm_model_t* model, const float* image,
                      int h, int w, int c, const char* question,
                      char* answer, int answer_cap);

void mm_vlm_ocr(const mm_vlm_model_t* model, const float* image,
                int h, int w, int c, char* text, int text_cap);

void mm_vlm_region_understand(const mm_vlm_model_t* model, const float* image,
                              int h, int w, int c, mm_vlm_bbox_t bbox,
                              const char* question, char* answer, int cap);

void mm_vlm_multiturn(const mm_vlm_model_t* model,
                      mm_vlm_conversation_t* conv,
                      const float* image, int h, int w, int c,
                      char* response, int cap);

void mm_vlm_linear_init(mm_vlm_linear_t* l, int in_dim, int out_dim);
void mm_vlm_linear_free(mm_vlm_linear_t* l);
void mm_vlm_linear_forward(const mm_vlm_linear_t* l, const float* x,
                           int batch, int in_dim, float* y);

void mm_vlm_attention_init(mm_vlm_attention_t* attn, int dim, int num_heads,
                           int max_seq_len, float rope_theta);
void mm_vlm_attention_free(mm_vlm_attention_t* attn);
void mm_vlm_attention_forward(const mm_vlm_attention_t* attn, const float* x,
                              int seq_len, int pos, float* out);

void mm_vlm_ffn_init(mm_vlm_ffn_t* ffn, int dim, int hidden_dim);
void mm_vlm_ffn_free(mm_vlm_ffn_t* ffn);
void mm_vlm_ffn_forward(const mm_vlm_ffn_t* ffn, const float* x, int seq_len, float* out);

void mm_vlm_decoder_layer_init(mm_vlm_decoder_layer_t* layer, int dim,
                               int num_heads, int max_seq_len, int hidden_dim,
                               float rope_theta, int layer_idx);
void mm_vlm_decoder_layer_free(mm_vlm_decoder_layer_t* layer);
void mm_vlm_decoder_layer_forward(const mm_vlm_decoder_layer_t* layer,
                                  const float* x, int seq_len, int pos, float* out);

void mm_vlm_llm_init(mm_vlm_llm_t* llm, int dim, int num_layers, int num_heads,
                     int hidden_dim, int vocab_size, int max_seq_len);
void mm_vlm_llm_free(mm_vlm_llm_t* llm);
void mm_vlm_llm_forward(const mm_vlm_llm_t* llm, const float* x,
                        int seq_len, int pos, float* logits, int* next_token);

void mm_vlm_rmsnorm(const float* x, int n, float* out);

void mm_vlm_rope_forward(float* q, float* k, int seq_len, int head_dim,
                         int pos, float theta);

float mm_vlm_silu(float x);
void  mm_vlm_silu_forward(const float* x, int n, float* out);

void mm_vlm_sampler(const float* logits, int vocab_size, float temperature,
                    int top_k, float top_p, int* token);

#ifdef __cplusplus
}
#endif

#endif
