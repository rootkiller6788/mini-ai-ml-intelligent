#ifndef VIDEO_UNDERSTANDING_H
#define VIDEO_UNDERSTANDING_H

#include <stddef.h>
#include <math.h>
#include "clip_contrastive.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MM_VIDEO_NUM_FRAMES       16
#define MM_VIDEO_FRAME_SIZE       224
#define MM_VIDEO_FRAME_CHANNELS   3
#define MM_VIDEO_C3D_CHANNELS     {3, 64, 128, 256, 512}
#define MM_VIDEO_MAX_DURATION_SEC 300
#define MM_VIDEO_NUM_CLASSES      400
#define MM_VIDEO_EMBED_DIM        512
#define MM_VIDEO_MAX_CAP_LEN      64

typedef struct {
    float* data;
    int    num_frames;
    int    height;
    int    width;
    int    channels;
} mm_video_clip_t;

typedef struct {
    float* weight;
    float* bias;
    int    in_ch;
    int    out_ch;
    int    kt, kh, kw;
    int    st, sh, sw;
    int    pt, ph, pw;
} mm_conv3d_t;

typedef struct {
    float* weight;
    float* bias;
    int    num_features;
} mm_batchnorm3d_t;

typedef struct {
    mm_conv3d_t       conv;
    mm_batchnorm3d_t  bn;
    int               use_relu;
    int               use_pool;
    int               pool_kt, pool_kh, pool_kw;
    int               pool_st, pool_sh, pool_sw;
} mm_c3d_block_t;

typedef struct {
    mm_c3d_block_t*   blocks;
    int               num_blocks;
    mm_linear_t       fc;
    float*            cls_token;
    int               num_classes;
    int               embed_dim;
} mm_c3d_model_t;

typedef struct {
    float*           qkv_weight;
    float*           qkv_bias;
    float*           proj_weight;
    float*           proj_bias;
    int              num_heads;
    int              head_dim;
    int              dim;
} mm_ts_attention_t;

typedef struct {
    mm_ts_attention_t spatial_attn;
    mm_ts_attention_t temporal_attn;
    mm_linear_t       fc1, fc2;
    float*            ln1_weight;
    float*            ln1_bias;
    float*            ln2_weight;
    float*            ln2_bias;
    int               dim;
    int               num_heads;
    int               ffn_dim;
} mm_timesformer_block_t;

typedef struct {
    float*                   patch_embed_weight;
    float*                   patch_embed_bias;
    float*                   pos_embed;
    float*                   time_embed;
    mm_timesformer_block_t*  blocks;
    mm_linear_t              head;
    float*                   ln_post_weight;
    float*                   ln_post_bias;
    int                      num_layers;
    int                      num_frames;
    int                      num_patches;
    int                      embed_dim;
    int                      num_heads;
    int                      ffn_dim;
    int                      num_classes;
    int                      frame_size;
    int                      patch_size;
} mm_timesformer_t;

typedef struct {
    mm_linear_t fc1, fc2;
    int         input_dim;
    int         hidden_dim;
    int         output_dim;
} mm_action_head_t;

typedef struct {
    mm_image_encoder_t frame_encoder;
    mm_linear_t        temporal_proj;
    mm_linear_t        text_proj;
    int                embed_dim;
    int                num_frames;
} mm_clip4clip_t;

typedef struct {
    int    start_frame;
    int    end_frame;
    int    class_id;
    float  confidence;
    int    start_ms;
    int    end_ms;
} mm_action_segment_t;

typedef struct {
    mm_action_segment_t* segments;
    int                  num_segments;
    int                  max_segments;
} mm_action_segments_t;

typedef enum {
    MM_VIDEO_ARCH_C3D        = 0,
    MM_VIDEO_ARCH_TIMESFORMER = 1,
    MM_VIDEO_ARCH_CLIP4CLIP  = 2,
} mm_video_arch_t;

typedef struct {
    mm_video_arch_t arch;
    union {
        mm_c3d_model_t     c3d;
        mm_timesformer_t   timesformer;
        mm_clip4clip_t     clip4clip;
    } model;
    int frame_size;
    int num_frames;
    int num_classes;
    int embed_dim;
} mm_video_model_t;

void mm_video_clip_init(mm_video_clip_t* clip, int num_frames, int h, int w, int c);
void mm_video_clip_free(mm_video_clip_t* clip);

void mm_video_sample_frames(const float* video, int total_frames,
                            int h, int w, int c, int num_sample,
                            mm_video_clip_t* clip);
void mm_video_uniform_sample(const float* video, int total_frames,
                             int h, int w, int c, int num_sample,
                             int* frame_indices);

void mm_conv3d_init(mm_conv3d_t* c3d, int in_ch, int out_ch,
                    int kt, int kh, int kw, int st, int sh, int sw,
                    int pt, int ph, int pw);
void mm_conv3d_free(mm_conv3d_t* c3d);
void mm_conv3d_forward(const mm_conv3d_t* c3d, const float* x,
                       int t, int h, int w, float* y, int* ot, int* oh, int* ow);

void mm_batchnorm3d_init(mm_batchnorm3d_t* bn, int nf);
void mm_batchnorm3d_free(mm_batchnorm3d_t* bn);
void mm_batchnorm3d_forward(const mm_batchnorm3d_t* bn, const float* x,
                            int t, int h, int w, float* y);

void mm_c3d_block_init(mm_c3d_block_t* blk, int in_ch, int out_ch,
                       int kt, int kh, int kw, int st, int sh, int sw);
void mm_c3d_block_free(mm_c3d_block_t* blk);
void mm_c3d_block_forward(const mm_c3d_block_t* blk, const float* x,
                          int t, int h, int w, float* y, int* ot, int* oh, int* ow);

void mm_c3d_init(mm_c3d_model_t* c3d, int num_classes, int embed_dim,
                 int num_frames, int frame_h, int frame_w);
void mm_c3d_free(mm_c3d_model_t* c3d);
void mm_c3d_forward(const mm_c3d_model_t* c3d, const mm_video_clip_t* clip,
                    float* class_logits, float* embedding);

void mm_timesformer_init(mm_timesformer_t* tsf, int num_frames,
                         int frame_size, int patch_size, int embed_dim,
                         int num_heads, int num_layers, int num_classes);
void mm_timesformer_free(mm_timesformer_t* tsf);
void mm_timesformer_forward(const mm_timesformer_t* tsf, const mm_video_clip_t* clip,
                            float* class_logits, float* embedding);

void mm_clip4clip_init(mm_clip4clip_t* c4c, int embed_dim, int num_frames,
                       int image_size, int patch_size);
void mm_clip4clip_free(mm_clip4clip_t* c4c);
void mm_clip4clip_encode_video(const mm_clip4clip_t* c4c, const mm_video_clip_t* clip,
                               float* video_embedding);
void mm_clip4clip_encode_text(const mm_clip4clip_t* c4c, const char* text,
                              float* text_embedding);
void mm_clip4clip_retrieve(const float* video_emb, const float* text_embs,
                           int num_texts, int dim, int* indices, int top_k);

void mm_video_model_init(mm_video_model_t* model, mm_video_arch_t arch,
                         int num_classes, int num_frames, int frame_size);
void mm_video_model_free(mm_video_model_t* model);

int  mm_video_action_recognition(const mm_video_model_t* model,
                                 const mm_video_clip_t* clip,
                                 char* class_name, int name_cap);
void mm_video_action_recognition_topk(const mm_video_model_t* model,
                                      const mm_video_clip_t* clip, int k,
                                      int* class_ids, float* confidences);

void mm_video_temporal_localize(const mm_video_model_t* model,
                                const float* video, int total_frames,
                                int h, int w, int c,
                                int window_size, int stride,
                                mm_action_segments_t* segments);
void mm_action_segments_init(mm_action_segments_t* segs, int max_segments);
void mm_action_segments_free(mm_action_segments_t* segs);

void mm_video_caption(const mm_video_model_t* model, const mm_video_clip_t* clip,
                      char* caption, int cap_capacity);

void mm_video_text_retrieval(const mm_video_model_t* model,
                             const mm_video_clip_t* clip,
                             const char** text_queries, int num_queries,
                             float* similarities);

void mm_ts_attention_init(mm_ts_attention_t* attn, int dim, int num_heads);
void mm_ts_attention_free(mm_ts_attention_t* attn);
void mm_ts_attention_forward(const mm_ts_attention_t* attn, const float* x,
                             int seq_len, float* out);

void mm_timesformer_block_init(mm_timesformer_block_t* blk, int dim,
                               int num_heads, int ffn_dim);
void mm_timesformer_block_free(mm_timesformer_block_t* blk);
void mm_timesformer_block_forward(const mm_timesformer_block_t* blk,
                                  const float* x, int seq_len, float* out);

float mm_iou_3d(int s1, int e1, int s2, int e2);
void  mm_nms_3d(mm_action_segment_t* segs, int n, float iou_thresh,
                int* keep_indices, int* num_keep);

const char** mm_video_kinetics_labels(void);
int          mm_video_num_kinetics_labels(void);

#ifdef __cplusplus
}
#endif

#endif
