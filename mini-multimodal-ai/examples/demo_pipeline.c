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

#define PIPE_INPUT_H  224
#define PIPE_INPUT_W  224
#define PIPE_INPUT_C  3
#define PIPE_AUDIO_S  16000

typedef enum {
    PIPE_CLIP         = 0,
    PIPE_SD           = 1,
    PIPE_LLAVA        = 2,
    PIPE_WHISPER      = 3,
    PIPE_VIDEO_C3D    = 4,
    PIPE_VIDEO_TSF    = 5,
    PIPE_VIDEO_CLIP4C = 6,
    PIPE_TEST_COUNT   = 7
} pipe_test_id_t;

typedef struct {
    const char* name;
    pipe_test_id_t id;
    void* state;
    int    configured;
} pipe_module_t;

typedef struct {
    pipe_module_t modules[10];
    int           num_modules;
    int           total_params;
    float         elapsed_ms;
    char          log[4096];
} pipe_registry_t;

static float test_image[PIPE_INPUT_H * PIPE_INPUT_W * PIPE_INPUT_C];
static float test_audio[PIPE_AUDIO_S * 3];
static float test_video[90 * 112 * 112 * 3];

static void init_pipe_data(void) {
    for (int i = 0; i < PIPE_INPUT_H * PIPE_INPUT_W * PIPE_INPUT_C; i++)
        test_image[i] = (float)rand() / (float)RAND_MAX;
    for (int i = 0; i < PIPE_AUDIO_S * 3; i++)
        test_audio[i] = sinf(2.0f * 3.14159f * 440.0f * (float)i / 16000.0f) * 0.5f;
    for (int i = 0; i < 90 * 112 * 112 * 3; i++)
        test_video[i] = (float)rand() / (float)RAND_MAX;
}

static void pipe_registry_init(pipe_registry_t* reg) {
    reg->num_modules = 0;
    reg->total_params = 0;
    reg->elapsed_ms = 0.0f;
    reg->log[0] = '\0';
}

static void pipe_register(pipe_registry_t* reg, const char* name,
                          pipe_test_id_t id, void* state) {
    if (reg->num_modules >= 10) return;
    pipe_module_t* m = &reg->modules[reg->num_modules++];
    m->name   = name;
    m->id     = id;
    m->state  = state;
    m->configured = 0;
    snprintf(reg->log + strlen(reg->log), sizeof(reg->log) - strlen(reg->log),
             "  REGISTER: %s (id=%d)\n", name, (int)id);
}

static clock_t pipe_tick(void) {
    return clock();
}

static double pipe_tock(clock_t start, pipe_registry_t* reg) {
    double ms = (double)(clock() - start) * 1000.0 / (double)CLOCKS_PER_SEC;
    reg->elapsed_ms += (float)ms;
    return ms;
}

static void run_clip_pipeline(pipe_registry_t* reg) {
    printf("\n  ---- CLIP Pipeline ----\n");
    clock_t t0 = pipe_tick();

    mm_clip_model_t* clip = (mm_clip_model_t*)malloc(sizeof(mm_clip_model_t));
    mm_clip_init(clip, 128, 2);
    pipe_register(reg, "CLIP", PIPE_CLIP, clip);

    float img_emb[128], txt_emb[128];
    mm_clip_encode_image(clip, test_image, PIPE_INPUT_H, PIPE_INPUT_W, PIPE_INPUT_C, img_emb);
    mm_clip_encode_text(clip, "a photo of a sunset", txt_emb);

    float sim = 0.0f;
    for (int d = 0; d < 128; d++) sim += img_emb[d] * txt_emb[d];
    printf("  Cosine similarity: %.4f\n", sim);

    const char* classes[] = {"sunset", "mountain", "beach", "forest"};
    int pred = mm_clip_zeroshot(img_emb, classes, 4, 1);
    printf("  Zero-shot result: %s\n", classes[pred]);

    float loss = 0.0f;
    float sim_mat[16];
    float img_batch[4][128], txt_batch[4][128];
    for (int b = 0; b < 4; b++) {
        for (int d = 0; d < 128; d++) {
            img_batch[b][d] = (float)rand() / (float)RAND_MAX;
            txt_batch[b][d] = (float)rand() / (float)RAND_MAX;
        }
        mm_l2_normalize(img_batch[b], 128);
        mm_l2_normalize(txt_batch[b], 128);
    }
    mm_cosine_sim_matrix((float*)img_batch, (float*)txt_batch, 4, 128, sim_mat);
    loss = mm_infonce_loss(sim_mat, 4);
    printf("  InfoNCE loss (batch=4): %.6f\n", loss);

    int indices[4];
    mm_clip_retrieve(img_batch[0], (float*)txt_batch, 4, 128, indices, 4);
    printf("  Retrieval top-4: %d %d %d %d\n", indices[0], indices[1], indices[2], indices[3]);

    reg->total_params += 128 * 128 * 4;
    double elapsed = pipe_tock(t0, reg);
    printf("  CLIP pipeline: %.2f ms\n", elapsed);

    mm_clip_free(clip);
    free(clip);
}

static void run_sd_pipeline(pipe_registry_t* reg) {
    printf("\n  ---- Stable Diffusion Pipeline ----\n");
    clock_t t0 = pipe_tick();

    mm_stable_diffusion_t* sd = (mm_stable_diffusion_t*)malloc(sizeof(mm_stable_diffusion_t));
    int sd_blocks = 3;
    mm_sd_init(sd, 128, 4, 32, sd_blocks);
    pipe_register(reg, "StableDiffusion", PIPE_SD, sd);

    printf("  Model: %dx%d images, latent=%dx%d, base_ch=%d\n",
           128, 128, sd->latent_size, sd->latent_size, 32);

    float* gen = (float*)malloc((size_t)128 * 128 * 3 * sizeof(float));
    float ctx[768] = {0};
    for (int i = 0; i < 768; i++) ctx[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 0.02f;

    mm_sd_generate(sd, ctx, 1, 10, MM_SD_SAMPLER_DDIM, 128, 128, 3, gen);
    printf("  Generated 128x128 image (DDIM, 10 steps)\n");

    float* mask = (float*)malloc((size_t)128 * 128 * 3 * sizeof(float));
    for (int i = 0; i < 128 * 128 * 3; i++) {
        int px = (i / 3) % 128;
        int py = (i / 3) / 128;
        mask[i] = (px > 32 && px < 96 && py > 32 && py < 96) ? 1.0f : 0.0f;
    }
    float* inpainted = (float*)malloc((size_t)128 * 128 * 3 * sizeof(float));
    mm_sd_inpaint(sd, gen, mask, ctx, 1, 128, 128, 3, 10, inpainted);
    printf("  Inpainting completed (mask: center 64x64 region)\n");

    int vae_blocks = 3;
    mm_vae_t* vae = (mm_vae_t*)malloc(sizeof(mm_vae_t));
    mm_vae_init(vae, 4, 32, vae_blocks);
    int vae_scl = 1 << (vae_blocks - 1);
    int lat_sz = 128 / vae_scl;
    int lat_n = lat_sz * lat_sz * 4;
    float* latent = (float*)malloc((size_t)lat_n * sizeof(float));
    float* logvar = (float*)malloc((size_t)lat_n * sizeof(float));
    mm_vae_encode(vae, test_image, 128, 128, 3, latent, logvar);
    float* sampled = (float*)malloc((size_t)lat_n * sizeof(float));
    mm_vae_sample(latent, logvar, lat_n, sampled);
    float* reconstructed = (float*)malloc((size_t)128 * 128 * 3 * sizeof(float));
    mm_vae_decode(vae, sampled, reconstructed, 128, 128, 3);
    printf("  VAE encode-decode roundtrip completed\n");

    reg->total_params += 32 * 32 * 64 * 4;

    mm_vae_free(vae);
    free(vae);
    free(gen); free(mask); free(inpainted);
    free(latent); free(logvar); free(sampled); free(reconstructed);

    double elapsed = pipe_tock(t0, reg);
    printf("  SD pipeline: %.2f ms\n", elapsed);

    mm_sd_free(sd);
    free(sd);
}

static void run_llava_pipeline(pipe_registry_t* reg) {
    printf("\n  ---- LLaVA Pipeline ----\n");
    clock_t t0 = pipe_tick();

    mm_vlm_model_t* vlm = (mm_vlm_model_t*)malloc(sizeof(mm_vlm_model_t));
    mm_vlm_model_init(vlm, 128, 256, 2, 2);
    pipe_register(reg, "LLaVA", PIPE_LLAVA, vlm);

    printf("  Model: vision=%dD, llm=%dD, patches=%d, llm_layers=%d\n",
           vlm->vision_dim, vlm->llm_dim, vlm->num_patches, vlm->llm.num_layers);

    float* img336 = (float*)malloc((size_t)336 * 336 * 3 * sizeof(float));
    for (int i = 0; i < 336 * 336 * 3; i++) img336[i] = (float)rand() / (float)RAND_MAX;

    float* vfeat = (float*)malloc((size_t)vlm->num_patches * vlm->vision_dim * sizeof(float));
    mm_vlm_encode_image(vlm, img336, 336, 336, 3, vfeat);
    printf("  Vision encoding: %d patches\n", vlm->num_patches);

    float* pfeat = (float*)malloc((size_t)vlm->num_patches * vlm->llm_dim * sizeof(float));
    mm_vlm_project_features(&vlm->projector, vfeat, vlm->num_patches, pfeat);
    printf("  Projection to LLM space: %dD -> %dD\n", vlm->vision_dim, vlm->llm_dim);

    char answer[512];
    mm_vlm_visual_qa(vlm, img336, 336, 336, 3, "What do you see?", answer, sizeof(answer));
    printf("  Visual QA: '%s'\n", answer);

    mm_vlm_conversation_t conv;
    mm_vlm_conversation_init(&conv, 4);
    mm_vlm_conversation_add(&conv, "Describe the image.", MM_VLM_MSG_USER, 1);
    mm_vlm_conversation_add(&conv, "It shows a landscape.", MM_VLM_MSG_ASSISTANT, 0);
    mm_vlm_conversation_add(&conv, "What colors are present?", MM_VLM_MSG_USER, 1);
    char resp[512];
    mm_vlm_multiturn(vlm, &conv, img336, 336, 336, 3, resp, sizeof(resp));
    printf("  Multi-turn (%d msgs): %s\n", conv.num_messages, resp);

    reg->total_params += vlm->llm.dim * vlm->llm.num_layers * 4;

    mm_vlm_conversation_free(&conv);
    free(img336); free(vfeat); free(pfeat);

    double elapsed = pipe_tock(t0, reg);
    printf("  LLaVA pipeline: %.2f ms\n", elapsed);

    mm_vlm_model_free(vlm);
    free(vlm);
}

static void run_whisper_pipeline(pipe_registry_t* reg) {
    printf("\n  ---- Whisper Pipeline ----\n");
    clock_t t0 = pipe_tick();

    mm_whisper_model_t* wh = (mm_whisper_model_t*)malloc(sizeof(mm_whisper_model_t));
    mm_whisper_model_init(wh, 128, 128, 2, 2, 80);
    pipe_register(reg, "Whisper", PIPE_WHISPER, wh);

    int audio_len = 16000 * 5;
    float* audio = (float*)malloc((size_t)audio_len * sizeof(float));
    for (int i = 0; i < audio_len; i++) {
        audio[i] = sinf(2.0f * 3.14159f * 300.0f * (float)i / 16000.0f) * 0.3f;
        audio[i] += sinf(2.0f * 3.14159f * 600.0f * (float)i / 16000.0f) * 0.2f;
    }

    float energy = mm_vad_energy(audio, audio_len);
    int is_speech = mm_vad_is_speech(audio, audio_len, 0.0005f);
    printf("  Audio: %.1fs, energy=%.6f, VAD=%d\n", 5.0, energy, is_speech);

    int* starts, *ends, nseg;
    mm_vad_split(audio, audio_len, 500, 0.0003f, &starts, &ends, &nseg);
    printf("  VAD split: %d chunks\n", nseg);

    mm_mel_spectrogram_t mel;
    mm_audio_mel_spectrogram(audio, audio_len, &wh->mel_filterbank, &mel);
    printf("  Mel: %d frames, %d mels\n", mel.n_frames, mel.n_mels);

    float* enc_hidden = (float*)malloc((size_t)mel.n_frames * 128 * sizeof(float));
    mm_whisper_encoder_forward(&wh->encoder, &mel, enc_hidden);
    printf("  Encoder: %d frames -> hidden\n", mel.n_frames);

    char lang[8];
    float lprob;
    mm_whisper_detect_language(wh, audio, audio_len, lang, &lprob);
    printf("  Language: %s (prob=%.3f)\n", lang, lprob);

    mm_whisper_result_t result;
    mm_whisper_result_init(&result, 10);
    mm_whisper_transcribe(wh, audio, audio_len, "en", MM_WHISPER_TASK_TRANSCRIBE, &result);
    printf("  Transcription: %d segments\n", result.num_segments);
    for (int i = 0; i < result.num_segments && i < 3; i++) {
        printf("    [%d] %s\n", i, result.segments[i].text);
    }

    mm_whisper_result_free(&result);
    free(audio); free(starts); free(ends); free(enc_hidden);
    mm_mel_spectrogram_free(&mel);

    reg->total_params += 128 * 128 * 6;

    double elapsed = pipe_tock(t0, reg);
    printf("  Whisper pipeline: %.2f ms\n", elapsed);

    mm_whisper_model_free(wh);
    free(wh);
}

static void run_video_pipeline(pipe_registry_t* reg) {
    printf("\n  ---- Video Pipeline ----\n");
    clock_t t0 = pipe_tick();

    int nf = 16;
    int fs = 112;
    mm_video_clip_t clip;
    mm_video_clip_init(&clip, nf, fs, fs, 3);
    for (int i = 0; i < nf * fs * fs * 3; i++) clip.data[i] = (float)rand() / (float)RAND_MAX;

    printf("  Sample: %d frames, %dx%dx%d\n", nf, fs, fs, 3);

    mm_video_model_t c3d, tsf, c4c;
    mm_video_model_init(&c3d, MM_VIDEO_ARCH_C3D, 100, nf, fs);
    pipe_register(reg, "Video-C3D", PIPE_VIDEO_C3D, &c3d);

    char name[128];
    int c3d_p = mm_video_action_recognition(&c3d, &clip, name, sizeof(name));
    printf("  C3D action: %s (id=%d)\n", name, c3d_p);

    mm_video_model_init(&tsf, MM_VIDEO_ARCH_TIMESFORMER, 100, 8, 112);
    pipe_register(reg, "Video-TimeSformer", PIPE_VIDEO_TSF, &tsf);

    mm_video_clip_t clip8;
    mm_video_clip_init(&clip8, 8, 112, 112, 3);
    for (int i = 0; i < 8 * 112 * 112 * 3; i++) clip8.data[i] = (float)rand() / (float)RAND_MAX;

    int tsf_p = mm_video_action_recognition(&tsf, &clip8, name, sizeof(name));
    printf("  TimeSformer action: %s (id=%d)\n", name, tsf_p);

    mm_video_model_init(&c4c, MM_VIDEO_ARCH_CLIP4CLIP, 100, 8, 112);
    pipe_register(reg, "Video-CLIP4Clip", PIPE_VIDEO_CLIP4C, &c4c);

    float vemb[512];
    mm_clip4clip_encode_video(&c4c.model.clip4clip, &clip8, vemb);
    printf("  CLIP4Clip embedding: 512-D\n");

    const char* queries[] = {"playing guitar", "riding a bike", "cooking"};
    float sims[3];
    mm_video_text_retrieval(&c4c, &clip8, queries, 3, sims);
    printf("  Video-text retrieval: ");
    for (int i = 0; i < 3; i++) printf("%s=%.3f ", queries[i], sims[i]);
    printf("\n");

    mm_action_segments_t segs;
    mm_action_segments_init(&segs, 15);
    mm_video_temporal_localize(&c3d, test_video, 90, 112, 112, 3, 30, 15, &segs);
    printf("  Temporal localization: %d candidate segments\n", segs.num_segments);

    char cap_text[256];
    mm_video_caption(&c3d, &clip, cap_text, sizeof(cap_text));
    printf("  Video captioning: %s\n", cap_text);

    printf("  Kinetics-400 labels loaded: %d\n", mm_video_num_kinetics_labels());

    reg->total_params += 512 * 16 * 3;

    mm_action_segments_free(&segs);
    mm_video_clip_free(&clip);
    mm_video_clip_free(&clip8);
    mm_video_model_free(&c3d);
    mm_video_model_free(&tsf);
    mm_video_model_free(&c4c);

    double elapsed = pipe_tock(t0, reg);
    printf("  Video pipeline: %.2f ms\n", elapsed);
}

static void print_pipeline_summary(pipe_registry_t* reg) {
    printf("\n%s\n", "============================================================");
    printf("%s\n", "  PIPELINE SUMMARY");
    printf("%s\n\n", "============================================================");

    printf("  Modules registered: %d\n\n", reg->num_modules);
    for (int i = 0; i < reg->num_modules; i++) {
        printf("  %2d. %-20s  (id=%d)\n",
               i + 1, reg->modules[i].name, reg->modules[i].id);
    }

    printf("\n  Total estimated parameters: %d\n", reg->total_params);
    printf("  Total pipeline time: %.2f ms\n\n", reg->elapsed_ms);

    printf("  Pipeline Log:\n%s\n", reg->log);
}

static void benchmark_matrix_mul(void) {
    printf("\n  ---- Bench: Matrix Multiply (C=AxB) ----\n");
    int N = 256;
    float* A = (float*)malloc((size_t)N * N * sizeof(float));
    float* B = (float*)malloc((size_t)N * N * sizeof(float));
    float* C = (float*)calloc((size_t)N * N, sizeof(float));

    for (int i = 0; i < N * N; i++) {
        A[i] = (float)rand() / (float)RAND_MAX;
        B[i] = (float)rand() / (float)RAND_MAX;
    }

    clock_t t0 = clock();
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            float aik = A[i * N + k];
            for (int j = 0; j < N; j++) {
                C[i * N + j] += aik * B[k * N + j];
            }
        }
    }
    double ms = (double)(clock() - t0) * 1000.0 / (double)CLOCKS_PER_SEC;
    float gflops = (2.0f * (float)N * (float)N * (float)N) / ((float)ms * 1e6f);

    float checksum = 0.0f;
    for (int i = 0; i < N * N; i++) checksum += C[i];
    printf("  MatMul %dx%d: %.2f ms, %.2f GFLOPS, checksum=%.4f\n",
           N, N, ms, gflops, checksum);

    free(A); free(B); free(C);
}

int main(void) {
    srand((unsigned)time(NULL));
    init_pipe_data();

    printf("%s\n", "################################################################");
    printf("%s\n", "#                                                              #");
    printf("%s\n", "#   mini-multimodal-ai - Training/Inference Pipeline Demo       #");
    printf("%s\n", "#   5 Pipelines + Benchmarks                                   #");
    printf("%s\n", "#                                                              #");
    printf("%s\n", "################################################################\n");

    printf("System info: C99, %zu-bit pointers, %zu-bit float\n",
           sizeof(void*) * 8, sizeof(float) * 8);

    benchmark_matrix_mul();

    pipe_registry_t reg;
    pipe_registry_init(&reg);

    run_clip_pipeline(&reg);
    run_sd_pipeline(&reg);
    run_llava_pipeline(&reg);
    run_whisper_pipeline(&reg);
    run_video_pipeline(&reg);

    print_pipeline_summary(&reg);

    printf("%s\n", "Pipeline registration log:");
    printf("  %s", reg.log);

    printf("\n%s\n", "################################################################");
    printf("%s\n\n", "  Pipeline demo complete - all 5 modules benchmarked.");

    return 0;
}
