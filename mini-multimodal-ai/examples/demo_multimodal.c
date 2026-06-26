#include "clip_contrastive.h"
#include "image_generation.h"
#include "vlm_llama.h"
#include "audio_whisper.h"
#include "video_understanding.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define DEMO_IMG_H  224
#define DEMO_IMG_W  224
#define DEMO_IMG_C  3

static float test_images[2][DEMO_IMG_H * DEMO_IMG_W * DEMO_IMG_C];

static void init_test_data(void) {
    for (int b = 0; b < 2; b++)
        for (int i = 0; i < DEMO_IMG_H * DEMO_IMG_W * DEMO_IMG_C; i++)
            test_images[b][i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
}

static void demo_clip(void) {
    printf("\n%s\n", "============================================================");
    printf("%s\n", "  MODULE 1: CLIP Contrastive Learning");
    printf("%s\n\n", "============================================================");

    mm_clip_model_t clip;
    mm_clip_init(&clip, 256, 2);

    float img_emb[256] = {0};
    mm_clip_encode_image(&clip, test_images[0], DEMO_IMG_H, DEMO_IMG_W, DEMO_IMG_C, img_emb);

    float text_emb[256] = {0};
    mm_clip_encode_text(&clip, "a beautiful sunset over mountains", text_emb);

    float sim = 0.0f;
    for (int d = 0; d < 256; d++) sim += img_emb[d] * text_emb[d];
    printf("  Cosine similarity (image-text): %.6f\n", sim);

    printf("  Image embedding (first 8): [");
    for (int i = 0; i < 8; i++) printf("%.3f%s", img_emb[i], i < 7 ? "," : "");
    printf("]\n");
    printf("  Text embedding  (first 8): [");
    for (int i = 0; i < 8; i++) printf("%.3f%s", text_emb[i], i < 7 ? "," : "");
    printf("]\n");

    printf("  Embedding L2 norms: img=%.4f, text=%.4f\n",
           mm_l2_norm(img_emb, 256), mm_l2_norm(text_emb, 256));

    const char* classes[] = {"cat", "dog", "car", "tree", "sunset", "mountain", "beach", "city"};
    int pred = mm_clip_zeroshot(img_emb, classes, 8, 1);
    printf("  Zero-shot classification: %s (index=%d)\n", classes[pred], pred);

    float gallery[3][256];
    const char* gallery_texts[] = {"sunset", "cat", "mountain"};
    for (int i = 0; i < 3; i++) mm_clip_encode_text(&clip, gallery_texts[i], gallery[i]);
    int indices[3];
    mm_clip_retrieve(img_emb, (float*)gallery, 3, 256, indices, 3);
    printf("  Image-text retrieval top-3: ");
    for (int i = 0; i < 3; i++) printf("%s ", gallery_texts[indices[i]]);
    printf("\n");

    float imgs2[2][256];
    const char* texts2[] = {"a cat", "a dog"};
    mm_clip_train_step(&clip, (float*)test_images, texts2, 2, DEMO_IMG_H, DEMO_IMG_W, DEMO_IMG_C, 1e-3f);
    printf("  Training step completed (batch=2)\n");

    mm_clip_free(&clip);
}

static void demo_sd(void) {
    printf("\n%s\n", "============================================================");
    printf("%s\n", "  MODULE 2: Image Generation (Stable Diffusion)");
    printf("%s\n\n", "============================================================");

    printf("  [VAE Module]\n");
    int vae_blocks = 3;
    mm_vae_t vae;
    mm_vae_init(&vae, 4, 32, vae_blocks);
    printf("  VAE: latent_dim=%d, base_channels=%d, blocks=%d\n", 4, 32, vae_blocks);

    int img_size = 128;
    int vae_scale = 1 << (vae_blocks - 1);  /* N-1 downsamplings */
    int ls = img_size / vae_scale;
    int n_img = img_size * img_size * 3;
    int n_lat = ls * ls * 4;

    float* img_in = (float*)malloc((size_t)n_img * sizeof(float));
    for (int i = 0; i < n_img; i++) img_in[i] = (float)rand() / (float)RAND_MAX;

    float* mu = (float*)calloc((size_t)n_lat, sizeof(float));
    float* lv = (float*)calloc((size_t)n_lat, sizeof(float));
    mm_vae_encode(&vae, img_in, img_size, img_size, 3, mu, lv);

    float* lt = (float*)malloc((size_t)n_lat * sizeof(float));
    mm_vae_sample(mu, lv, n_lat, lt);

    float* rec = (float*)malloc((size_t)n_img * sizeof(float));
    mm_vae_decode(&vae, lt, rec, img_size, img_size, 3);
    printf("  Encode->Decode: latent_shape=%dx%dx4, reconstruction computed\n", ls, ls);

    printf("\n  [UNet Denoiser]\n");
    mm_unet_t unet;
    mm_unet_init(&unet, 32, 1);
    printf("  UNet: base_ch=%d, down_blocks=%d, up_blocks=%d\n", 32, unet.num_down, unet.num_up);

    float* unet_in = (float*)malloc((size_t)n_lat * sizeof(float));
    for (int i = 0; i < n_lat; i++) unet_in[i] = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;

    float t_arr[1] = {0.3f};
    float ctx[768] = {0};
    for (int i = 0; i < 768; i++) ctx[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 0.02f;
    mm_unet_forward(&unet, unet_in, t_arr, ctx, ls, ls, 768 / 768, lt);

    float unet_mean = 0.0f;
    for (int i = 0; i < n_lat; i++) unet_mean += lt[i];
    unet_mean /= (float)n_lat;
    printf("  UNet forward output mean: %.6f\n", unet_mean);

    printf("\n  [Diffusion Schedule & Sampling]\n");
    mm_schedule_t betas, alphas, ac;
    mm_sd_beta_schedule(&betas, 1000, 0.00085f, 0.012f, MM_SD_SCHEDULE_LINEAR);
    mm_sd_alphas_from_betas(&betas, &alphas, &ac);
    printf("  Linear schedule: beta[0]=%.5f, beta[999]=%.5f, alpha_cumprod[999]=%.6f\n",
           betas.data[0], betas.data[999], ac.data[999]);

    float* noise = (float*)malloc((size_t)n_lat * sizeof(float));
    for (int i = 0; i < n_lat; i++) noise[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 2.0f;
    mm_sd_add_noise(lt, noise, sqrtf(ac.data[500]), sqrtf(1.0f - ac.data[500]), n_lat, lt);
    printf("  Added noise at t=500, alpha_cumprod=%.4f\n", ac.data[500]);

    float* dd_out = (float*)malloc((size_t)n_lat * sizeof(float));
    mm_sd_ddim_step(lt, noise, 500, 450, &ac, n_lat, 0.0f, dd_out);
    printf("  DDIM step t=500->450 completed\n");

    printf("\n  [Full Pipeline]\n");
    mm_stable_diffusion_t sd;
    mm_sd_init(&sd, 256, 4, 32, 4);
    printf("  SD model: size=%d, latent=%d, timesteps=%d\n", 256, sd.latent_size, 1000);

    float* gen = (float*)malloc((size_t)256 * 256 * 3 * sizeof(float));
    mm_sd_generate(&sd, ctx, 1, 15, MM_SD_SAMPLER_DPM_PP_2M, 256, 256, 3, gen);

    float gen_min = 1e9f, gen_max = -1e9f;
    for (int i = 0; i < 256 * 256 * 3; i++) {
        if (gen[i] < gen_min) gen_min = gen[i];
        if (gen[i] > gen_max) gen_max = gen[i];
    }
    printf("  Generated image: min=%.4f, max=%.4f (DPM++ 2M, 15 steps)\n", gen_min, gen_max);

    free(img_in); free(mu); free(lv); free(lt); free(rec);
    free(unet_in); free(noise); free(dd_out); free(gen);
    mm_vae_free(&vae);
    mm_unet_free(&unet);
    mm_sd_free(&sd);
    mm_schedule_free(&betas);
    mm_schedule_free(&alphas);
    mm_schedule_free(&ac);
}

static void demo_llava(void) {
    printf("\n%s\n", "============================================================");
    printf("%s\n", "  MODULE 3: Vision-Language Model (LLaVA)");
    printf("%s\n\n", "============================================================");

    mm_vlm_model_t vlm;
    mm_vlm_model_init(&vlm, 256, 512, 2, 2);

    int np = vlm.num_patches;
    printf("  LLaVA: vision_dim=%d, llm_dim=%d, patches=%d\n", vlm.vision_dim, vlm.llm_dim, np);
    printf("  LLM: layers=%d, heads=%d, head_dim=%d, ffn_dim=%d\n",
           vlm.llm.num_layers, vlm.llm.num_heads, vlm.llm.head_dim,
           vlm.llm.layers[0].ffn.hidden_dim);

    float* img = (float*)malloc((size_t)336 * 336 * 3 * sizeof(float));
    for (int i = 0; i < 336 * 336 * 3; i++) img[i] = (float)rand() / (float)RAND_MAX;

    float* vfeat = (float*)malloc((size_t)np * vlm.vision_dim * sizeof(float));
    mm_vlm_encode_image(&vlm, img, 336, 336, 3, vfeat);
    printf("  Vision encoding: %dx%d patches -> %d features\n", 24, 24, np);

    float* pfeat = (float*)malloc((size_t)np * vlm.llm_dim * sizeof(float));
    mm_vlm_project_features(&vlm.projector, vfeat, np, pfeat);
    printf("  Projection: %dD -> %dD\n", vlm.vision_dim, vlm.llm_dim);

    char qa[512];
    mm_vlm_visual_qa(&vlm, img, 336, 336, 3, "Describe this image in detail.", qa, sizeof(qa));
    printf("  Visual QA: %s\n", qa);

    mm_vlm_conversation_t conv;
    mm_vlm_conversation_init(&conv, 5);
    mm_vlm_conversation_add(&conv, "Hello! What do you see?", MM_VLM_MSG_USER, 1);
    mm_vlm_conversation_add(&conv, "I see a beautiful landscape.", MM_VLM_MSG_ASSISTANT, 0);
    mm_vlm_conversation_add(&conv, "Can you describe it more?", MM_VLM_MSG_USER, 1);
    printf("  Conversation: %d turns\n", conv.num_messages);

    char resp[512];
    mm_vlm_multiturn(&vlm, &conv, img, 336, 336, 3, resp, sizeof(resp));
    printf("  Multi-turn response: %s\n", resp);

    char ocr[512];
    mm_vlm_ocr(&vlm, img, 336, 336, 3, ocr, sizeof(ocr));
    printf("  OCR: %s\n", ocr);

    mm_vlm_bbox_t bbox = {50, 80, 120, 200};
    mm_vlm_region_understand(&vlm, img, 336, 336, 3, bbox, "What is the main object?", qa, sizeof(qa));
    printf("  Region understanding [%.0f,%.0f,%.0f,%.0f]: %s\n", bbox.x, bbox.y, bbox.w, bbox.h, qa);

    free(img); free(vfeat); free(pfeat);
    mm_vlm_conversation_free(&conv);
    mm_vlm_model_free(&vlm);
}

static void demo_whisper(void) {
    printf("\n%s\n", "============================================================");
    printf("%s\n", "  MODULE 4: Audio Understanding (Whisper)");
    printf("%s\n\n", "============================================================");

    mm_whisper_model_t whisper;
    mm_whisper_model_init(&whisper, 256, 256, 3, 3, 80);
    printf("  Whisper: encoder_dim=%d, decoder_dim=%d, enc_layers=%d, dec_layers=%d\n",
           256, 256, 3, 3);

    int audio_len = 16000 * 3;
    float* audio = (float*)malloc((size_t)audio_len * sizeof(float));
    for (int i = 0; i < audio_len; i++) {
        audio[i] = sinf(2.0f * 3.14159f * 440.0f * (float)i / 16000.0f) * 0.5f;
        audio[i] += sinf(2.0f * 3.14159f * 880.0f * (float)i / 16000.0f) * 0.3f;
        audio[i] += ((float)rand() / (float)RAND_MAX - 0.5f) * 0.05f;
    }

    float energy = mm_vad_energy(audio, audio_len);
    int is_speech = mm_vad_is_speech(audio, audio_len, 0.001f);
    printf("  Audio: %d samples (%.1fs), energy=%.6f, is_speech=%d\n",
           audio_len, 3.0, energy, is_speech);

    mm_mel_spectrogram_t mel;
    mm_audio_mel_spectrogram(audio, audio_len, &whisper.mel_filterbank, &mel);
    printf("  Mel spectrogram: %d frames x %d mels\n", mel.n_frames, mel.n_mels);

    int enc_len = mel.n_frames;
    float* enc_hidden = (float*)malloc((size_t)enc_len * 256 * sizeof(float));
    mm_whisper_encoder_forward(&whisper.encoder, &mel, enc_hidden);
    printf("  Encoder output: %d frames x 256 dim\n", enc_len);

    char lang[8];
    float prob;
    mm_whisper_detect_language(&whisper, audio, audio_len, lang, &prob);
    printf("  Detected language: %s (prob=%.3f)\n", lang, prob);

    mm_whisper_result_t result;
    mm_whisper_result_init(&result, 10);
    mm_whisper_transcribe(&whisper, audio, audio_len, "en", MM_WHISPER_TASK_TRANSCRIBE, &result);
    printf("  Transcription: %d segments\n", result.num_segments);
    for (int i = 0; i < result.num_segments; i++) {
        printf("    [%d] %d-%dms: %s (p=%.2f)\n", i,
               result.segments[i].start_ms, result.segments[i].end_ms,
               result.segments[i].text, result.segments[i].prob);
    }

    int* starts, *ends, nseg;
    mm_vad_split(audio, audio_len, 500, 0.0005f, &starts, &ends, &nseg);
    printf("  VAD: %d speech segments detected\n", nseg);

    int time_token;
    mm_whisper_encode_token(1500, &time_token);
    printf("  Time token: %dms -> token %d -> decode: %dms\n", 1500, time_token,
           mm_whisper_decode_time_token(time_token));
    printf("  Language ID: 'en'=%d, 'zh'=%d, 'de'=%d\n",
           mm_whisper_lang_id("en"), mm_whisper_lang_id("zh"), mm_whisper_lang_id("de"));

    mm_whisper_result_free(&result);
    free(audio); free(enc_hidden); free(starts); free(ends);
    mm_mel_spectrogram_free(&mel);
    mm_whisper_model_free(&whisper);
}

static void demo_video(void) {
    printf("\n%s\n", "============================================================");
    printf("%s\n", "  MODULE 5: Video Understanding");
    printf("%s\n\n", "============================================================");

    int num_kinetics = mm_video_num_kinetics_labels();
    printf("  Kinetics labels: %d classes\n", num_kinetics);

    printf("\n  [C3D Model]\n");
    mm_video_model_t c3d_model;
    mm_video_model_init(&c3d_model, MM_VIDEO_ARCH_C3D, 100, 16, 112);
    printf("  C3D: num_frames=%d, frame_size=%d, embed_dim=%d\n", 16, 112, c3d_model.embed_dim);

    mm_video_clip_t clip;
    mm_video_clip_init(&clip, 16, 112, 112, 3);
    for (int i = 0; i < 16 * 112 * 112 * 3; i++)
        clip.data[i] = (float)rand() / (float)RAND_MAX;

    float* logits3d = (float*)calloc((size_t)100, sizeof(float));
    float* emb3d = (float*)calloc((size_t)c3d_model.embed_dim, sizeof(float));
    mm_c3d_forward(&c3d_model.model.c3d, &clip, logits3d, emb3d);
    printf("  C3D forward completed, logits[0..3]: %.3f, %.3f, %.3f, %.3f\n",
           logits3d[0], logits3d[1], logits3d[2], logits3d[3]);

    char class_name[128];
    int pred = mm_video_action_recognition(&c3d_model, &clip, class_name, sizeof(class_name));
    printf("  Action recognition: %s (class_id=%d)\n", class_name, pred);

    int top_ids[5];
    float top_confs[5];
    mm_video_action_recognition_topk(&c3d_model, &clip, 5, top_ids, top_confs);
    printf("  Top-5: ");
    const char** labels = mm_video_kinetics_labels();
    for (int i = 0; i < 5; i++) printf("%s(%.2f) ", labels[top_ids[i]], top_confs[i]);
    printf("\n");

    char cap[256];
    mm_video_caption(&c3d_model, &clip, cap, sizeof(cap));
    printf("  Caption: %s\n", cap);

    printf("\n  [TimeSformer]\n");
    mm_video_model_t tsf_model;
    mm_video_model_init(&tsf_model, MM_VIDEO_ARCH_TIMESFORMER, 100, 8, 112);
    printf("  TimeSformer: num_frames=%d, frame_size=%d, layers=%d\n", 8, 112, 6);

    mm_video_clip_t clip8;
    mm_video_clip_init(&clip8, 8, 112, 112, 3);
    for (int i = 0; i < 8 * 112 * 112 * 3; i++) clip8.data[i] = (float)rand() / (float)RAND_MAX;

    float* logits_tsf = (float*)calloc((size_t)100, sizeof(float));
    float* emb_tsf = (float*)calloc((size_t)tsf_model.embed_dim, sizeof(float));
    mm_timesformer_forward(&tsf_model.model.timesformer, &clip8, logits_tsf, emb_tsf);
    printf("  TimeSformer forward completed\n");

    pred = mm_video_action_recognition(&tsf_model, &clip8, class_name, sizeof(class_name));
    printf("  Action recognition: %s\n", class_name);

    printf("\n  [CLIP4Clip - Video-Text Retrieval]\n");
    mm_video_model_t c4c_model;
    mm_video_model_init(&c4c_model, MM_VIDEO_ARCH_CLIP4CLIP, 100, 8, 112);
    printf("  CLIP4Clip: embed_dim=%d, frames=%d\n", c4c_model.embed_dim, 8);

    float* vemb = (float*)malloc((size_t)c4c_model.embed_dim * sizeof(float));
    mm_clip4clip_encode_video(&c4c_model.model.clip4clip, &clip8, vemb);
    printf("  Video embedding:");
    for (int i = 0; i < 5; i++) printf(" %.3f", vemb[i]);
    printf(" ...\n");

    const char* queries[] = {"playing guitar", "riding bike", "cooking food", "swimming pool"};
    float sims[4];
    mm_video_text_retrieval(&c4c_model, &clip8, queries, 4, sims);
    printf("  Text retrieval sims: ");
    for (int i = 0; i < 4; i++) printf("%s=%.3f ", queries[i], sims[i]);
    printf("\n");

    printf("\n  [Temporal Localization]\n");
    int total_f = 90;
    float* video_data = (float*)malloc((size_t)total_f * 112 * 112 * 3 * sizeof(float));
    for (int i = 0; i < total_f * 112 * 112 * 3; i++) video_data[i] = (float)rand() / (float)RAND_MAX;

    mm_action_segments_t segs;
    mm_action_segments_init(&segs, 20);
    mm_video_temporal_localize(&c3d_model, video_data, total_f, 112, 112, 3, 30, 15, &segs);
    printf("  Temporal localization: %d segments found\n", segs.num_segments);
    for (int i = 0; i < segs.num_segments && i < 3; i++) {
        printf("    [%d] frames %d-%d, class=%d, conf=%.2f, time=%d-%dms\n",
               i, segs.segments[i].start_frame, segs.segments[i].end_frame,
               segs.segments[i].class_id, segs.segments[i].confidence,
               segs.segments[i].start_ms, segs.segments[i].end_ms);
    }

    float iou = mm_iou_3d(0, 30, 20, 50);
    printf("  IoU test ([0,30] vs [20,50]) = %.4f\n", iou);

    int keep[20], nkeep;
    mm_nms_3d(segs.segments, segs.num_segments, 0.5f, keep, &nkeep);
    printf("  NMS: %d -> %d segments after IoU>0.5 filtering\n", segs.num_segments, nkeep);

    free(logits3d); free(emb3d); free(logits_tsf); free(emb_tsf);
    free(vemb); free(video_data);
    mm_video_clip_free(&clip);
    mm_video_clip_free(&clip8);
    mm_action_segments_free(&segs);
    mm_video_model_free(&c3d_model);
    mm_video_model_free(&tsf_model);
    mm_video_model_free(&c4c_model);
}

static void demo_cross_modal(void) {
    printf("\n%s\n", "============================================================");
    printf("%s\n", "  CROSS-MODAL: Unified Pipeline Demo");
    printf("%s\n\n", "============================================================");

    printf("  Demonstrating cross-modal alignment:\n");

    mm_clip_model_t clip;
    mm_clip_init(&clip, 128, 1);

    float img_emb[128] = {0};
    mm_clip_encode_image(&clip, test_images[0], DEMO_IMG_H, DEMO_IMG_W, DEMO_IMG_C, img_emb);

    float txt_emb[128] = {0};
    mm_clip_encode_text(&clip, "a person playing guitar", txt_emb);

    float cossim = 0.0f;
    for (int d = 0; d < 128; d++) cossim += img_emb[d] * txt_emb[d];
    printf("  Image-Text cosine sim: %.4f\n", cossim);

    float txt2_emb[128] = {0};
    mm_clip_encode_text(&clip, "a person riding bicycle", txt2_emb);
    float cossim2 = 0.0f;
    for (int d = 0; d < 128; d++) cossim2 += img_emb[d] * txt2_emb[d];
    printf("  Same image vs different text: %.4f (vs ^)\n", cossim2);

    mm_video_clip_t vclip;
    mm_video_clip_init(&vclip, 8, 112, 112, 3);
    for (int i = 0; i < 8 * 112 * 112 * 3; i++) vclip.data[i] = (float)rand() / (float)RAND_MAX;

    mm_video_model_t vc3d;
    mm_video_model_init(&vc3d, MM_VIDEO_ARCH_C3D, 50, 8, 112);
    char vc_name[128];
    int vc_pred = mm_video_action_recognition(&vc3d, &vclip, vc_name, sizeof(vc_name));
    printf("  Video action (C3D): class=%d, label=%s\n", vc_pred, vc_name);

    mm_video_model_t vtsf;
    mm_video_model_init(&vtsf, MM_VIDEO_ARCH_TIMESFORMER, 50, 8, 112);
    int vt_pred = mm_video_action_recognition(&vtsf, &vclip, vc_name, sizeof(vc_name));
    printf("  Video action (TimeSformer): class=%d, label=%s\n", vt_pred, vc_name);

    mm_video_clip_free(&vclip);
    mm_video_model_free(&vc3d);
    mm_video_model_free(&vtsf);
    mm_clip_free(&clip);
}

int main(void) {
    srand((unsigned)time(NULL));
    init_test_data();

    printf("%s\n", "################################################################");
    printf("%s\n", "#                                                              #");
    printf("%s\n", "#   mini-multimodal-ai - Complete Demo Suite                   #");
    printf("%s\n", "#   5 Modules | CLIP | SD | LLaVA | Whisper | Video            #");
    printf("%s\n", "#                                                              #");
    printf("%s\n", "################################################################");

    demo_clip();
    demo_sd();
    demo_llava();
    demo_whisper();
    demo_video();
    demo_cross_modal();

    printf("\n%s\n", "################################################################");
    printf("%s\n\n", "  All 5 modules + cross-modal demo completed successfully.");

    return 0;
}
