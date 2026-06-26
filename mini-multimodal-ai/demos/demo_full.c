/*
 * mini-multimodal-ai — Full Demo: Multimodal AI
 *
 * Demonstrates: CLIP, Stable Diffusion, VLM (LLaVA-style),
 *               Whisper (speech), video understanding.
 */
#include "../include/clip_contrastive.h"
#include "../include/image_generation.h"
#include "../include/vlm_llama.h"
#include "../include/audio_whisper.h"
#include "../include/video_understanding.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(void) {
    printf("=== mini-multimodal-ai: Multimodal AI Demo ===\n\n");

    /* Step 1: CLIP — Contrastive Language-Image Pre-training */
    printf("-- Step 1: CLIP (Image-Text Alignment) --\n");
    mm_clip_model_t clip;
    mm_clip_init(&clip, 256, 3);
    printf("CLIP init: embed_dim=%d, layers=%d\n", clip.embed_dim, 3);
    float dummy_img[3 * 64 * 64];
    for (int i = 0; i < 3 * 64 * 64; i++) dummy_img[i] = 0.5f;
    float img_emb[256];
    mm_clip_encode_image(&clip, dummy_img, 64, 64, 3, img_emb);
    float txt_emb[256];
    mm_clip_encode_text(&clip, "a photo of a cat", txt_emb);
    float sim[1];
    mm_cosine_sim_matrix(img_emb, txt_emb, 1, 256, sim);
    printf("  image-text cosine similarity: %.4f\n", sim[0]);
    mm_clip_free(&clip);

    /* Step 2: Stable Diffusion Components */
    printf("\n-- Step 2: Stable Diffusion (UNet + VAE) --\n");
    mm_vae_t vae;
    mm_vae_init(&vae, 4, 64, 1);
    float img[3 * 64 * 64];
    for (int i = 0; i < 3 * 64 * 64; i++) img[i] = 0.5f;
    float mu[4 * 8 * 8], lv[4 * 8 * 8];
    mm_vae_encode(&vae, img, 64, 64, 3, mu, lv);
    printf("VAE: 3x64x64 -> latent 4x8x8 (mean[0]=%.4f)\n", mu[0]);
    mm_vae_free(&vae);

    mm_unet_t unet;
    mm_unet_init(&unet, 64, 1);
    printf("UNet: base_channels=%d, num_res_blocks=%d\n", 64, 1);
    mm_unet_free(&unet);

    mm_schedule_t sched;
    mm_schedule_init(&sched, 100);
    printf("Diffusion schedule: %d steps\n", sched.n);
    mm_schedule_free(&sched);

    /* Step 3: VLM — Vision Language Model */
    printf("\n-- Step 3: VLM (Vision-Language Model) --\n");
    mm_vlm_model_t vlm;
    mm_vlm_model_init(&vlm, 256, 512, 3, 6);
    printf("VLM: vision_dim=%d, llm_dim=%d, temp=%.1f\n", vlm.vision_dim, vlm.llm_dim, vlm.temperature);
    mm_vlm_conversation_t conv;
    mm_vlm_conversation_init(&conv, 4);
    mm_vlm_conversation_add(&conv, "What is in this image?", MM_VLM_MSG_USER, 1);
    printf("VLM conversation: %d messages\n", conv.num_messages);
    mm_vlm_conversation_free(&conv);
    mm_vlm_model_free(&vlm);

    /* Step 4: Whisper — Speech Recognition */
    printf("\n-- Step 4: Whisper (Speech-to-Text) --\n");
    mm_mel_filterbank_t fb;
    mm_mel_filterbank_init(&fb, 80, 400, 16000);
    printf("Mel filterbank: %d mels, n_fft=%d, sr=%d\n", fb.n_mels, 400, 16000);
    mm_mel_filterbank_free(&fb);

    mm_whisper_model_t whisper;
    mm_whisper_model_init(&whisper, 256, 256, 3, 3, 80);
    printf("Whisper: enc_layers=%d, dec_layers=%d\n", 3, 3);
    mm_whisper_result_t result;
    mm_whisper_result_init(&result, 8);
    printf("  result buffer: max_segments=%d\n", result.max_segments);
    mm_whisper_result_free(&result);
    mm_whisper_model_free(&whisper);

    /* Step 5: Video Understanding */
    printf("\n-- Step 5: Video Understanding --\n");
    mm_video_model_t vmodel;
    mm_video_model_init(&vmodel, MM_VIDEO_ARCH_TIMESFORMER, 400, 8, 112);
    printf("Video model: arch=TimeSformer, classes=%d, frames=%d\n", 400, 8);
    mm_video_model_free(&vmodel);

    mm_video_clip_t vclip;
    mm_video_clip_init(&vclip, 8, 112, 112, 3);
    printf("Video clip: %d frames @ %dx%dx%d\n", vclip.num_frames, vclip.height, vclip.width, vclip.channels);
    mm_video_clip_free(&vclip);

    mm_conv3d_t c3d;
    mm_conv3d_init(&c3d, 3, 64, 3, 3, 3, 1, 1, 1, 1, 1, 1);
    printf("Conv3D: in=%d, out=%d, kernel=3x3x3\n", c3d.in_ch, c3d.out_ch);
    mm_conv3d_free(&c3d);

    mm_action_segments_t segs;
    mm_action_segments_init(&segs, 16);
    printf("Action segments buffer: max=%d\n", segs.max_segments);
    mm_action_segments_free(&segs);

    printf("\nMultimodal AI demo complete!\n");
    return 0;
}
