#ifndef CLIP_CONTRASTIVE_H
#define CLIP_CONTRASTIVE_H

#include <stddef.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MM_CLIP_MAX_TEXT_LEN   77
#define MM_CLIP_EMBED_DIM      512
#define MM_CLIP_VOCAB_SIZE     49408
#define MM_CLIP_NUM_HEADS      8
#define MM_CLIP_NUM_LAYERS     12
#define MM_CLIP_FFN_DIM        2048
#define MM_CLIP_PATCH_SIZE     16
#define MM_CLIP_IMAGE_SIZE     224
#define MM_CLIP_NUM_PATCHES    ((MM_CLIP_IMAGE_SIZE / MM_CLIP_PATCH_SIZE) * (MM_CLIP_IMAGE_SIZE / MM_CLIP_PATCH_SIZE))

typedef struct {
    float* weight;
    float* bias;
    int    in_dim;
    int    out_dim;
} mm_linear_t;

typedef struct {
    mm_linear_t q_proj, k_proj, v_proj, out_proj;
    int         num_heads;
    int         head_dim;
    float       scale;
} mm_self_attn_t;

typedef struct {
    mm_self_attn_t self_attn;
    mm_linear_t    fc1, fc2;
    float*         ln1_weight;
    float*         ln1_bias;
    float*         ln2_weight;
    float*         ln2_bias;
    int            dim;
} mm_transformer_block_t;

typedef struct {
    mm_linear_t           token_embed;
    float*                pos_embed;
    mm_transformer_block_t* blocks;
    mm_linear_t           final_ln_weight;
    mm_linear_t           final_ln_bias;
    float*                proj;
    int                   num_layers;
    int                   embed_dim;
    int                   vocab_size;
    int                   max_seq_len;
} mm_text_encoder_t;

typedef struct {
    float* class_token;
    float* pos_embed;
    mm_linear_t patch_proj;
    mm_transformer_block_t* blocks;
    mm_linear_t final_ln_weight;
    mm_linear_t final_ln_bias;
    float*                proj;
    int                   num_layers;
    int                   embed_dim;
    int                   image_size;
    int                   patch_size;
    int                   num_patches;
} mm_image_encoder_t;

typedef struct {
    mm_image_encoder_t image_encoder;
    mm_text_encoder_t  text_encoder;
    float*             logit_scale;
    int                embed_dim;
    float              temperature;
    int                use_fp16;
} mm_clip_model_t;

typedef struct {
    float* data;
    int    n;
} mm_embedding_t;

typedef struct {
    mm_embedding_t   image_emb;
    mm_embedding_t   text_emb;
    float            similarity;
} mm_clip_pair_t;

float mm_l2_norm(const float* x, int n);
void  mm_l2_normalize(float* x, int n);
void  mm_cosine_sim_matrix(const float* img_embs, const float* txt_embs,
                           int batch_size, int dim, float* sim_matrix);
float mm_infonce_loss(const float* sim_matrix, int batch_size);
void  mm_infonce_grad(const float* sim_matrix, int batch_size, float* img_grad, float* txt_grad);

void mm_text_tokenize(const char* text, int* token_ids, int* num_tokens);
void mm_text_encode(const mm_text_encoder_t* enc, const int* token_ids, int num_tokens,
                    float* embedding);

void mm_image_patchify(const float* image, int h, int w, int c,
                       int patch_size, float* patches);
void mm_image_encode(const mm_image_encoder_t* enc, const float* image,
                     int h, int w, int c, float* embedding);

void mm_clip_init(mm_clip_model_t* model, int embed_dim, int num_layers);
void mm_clip_free(mm_clip_model_t* model);
void mm_clip_encode_image(const mm_clip_model_t* model, const float* image,
                          int h, int w, int c, float* embedding);
void mm_clip_encode_text(const mm_clip_model_t* model, const char* text,
                         float* embedding);
int  mm_clip_zeroshot(const float* image_emb, const char** class_names,
                       int num_classes, int top_k);
void mm_clip_retrieve(const float* query_emb, const float* gallery_embs,
                      int gallery_size, int dim, int* indices, int top_k);

void mm_clip_train_step(mm_clip_model_t* model,
                        const float* images, const char** texts,
                        int batch_size, int h, int w, int c, float lr);

void mm_linear_init(mm_linear_t* l, int in_dim, int out_dim);
void mm_linear_free(mm_linear_t* l);
void mm_linear_forward(const mm_linear_t* l, const float* x, float* y);

void mm_self_attn_init(mm_self_attn_t* attn, int dim, int num_heads);
void mm_self_attn_free(mm_self_attn_t* attn);
void mm_self_attn_forward(const mm_self_attn_t* attn, const float* x,
                          int seq_len, float* out);

void mm_transformer_block_init(mm_transformer_block_t* blk, int dim, int num_heads, int ffn_dim);
void mm_transformer_block_free(mm_transformer_block_t* blk);
void mm_transformer_block_forward(const mm_transformer_block_t* blk, const float* x,
                                  int seq_len, float* out);

void mm_layernorm(const float* x, const float* weight, const float* bias,
                  int n, float* out);

void mm_softmax(float* x, int n);
float mm_gelu(float x);
void mm_gelu_forward(const float* x, int n, float* out);

void mm_text_encoder_init(mm_text_encoder_t* enc, int embed_dim, int num_layers,
                          int vocab_size, int max_seq_len);
void mm_text_encoder_free(mm_text_encoder_t* enc);

void mm_image_encoder_init(mm_image_encoder_t* enc, int embed_dim, int num_layers,
                           int image_size, int patch_size);
void mm_image_encoder_free(mm_image_encoder_t* enc);

#ifdef __cplusplus
}
#endif

#endif
