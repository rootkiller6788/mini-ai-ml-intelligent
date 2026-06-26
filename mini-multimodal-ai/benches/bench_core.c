/*
 * mini-multimodal-ai — Core Benchmarks
 *
 * Benchmarks: CLIP, Stable Diffusion, VLM, Whisper, video understanding.
 */
#include "../include/clip_contrastive.h"
#include "../include/image_generation.h"
#include "../include/vlm_llama.h"
#include "../include/audio_whisper.h"
#include "../include/video_understanding.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
    return (double)clock() / (double)(CLOCKS_PER_SEC / 1000);
}

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 5000;
    double t0, t1;
    printf("=== mini-multimodal-ai Benchmarks (N=%d) ===\n\n", N);

    /* ── CLIP Init/Destroy ── */
    {
        t0 = now_ms();
        for (int r = 0; r < N / 20; r++) {
            mm_clip_model_t model;
            mm_clip_init(&model, 256, 4);
            mm_clip_free(&model);
        }
        t1 = now_ms();
        printf("  clip_init+free:      %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 20, t1 - t0, (t1 - t0) / (double)(N / 20) * 1000.0);
    }

    /* ── Cosine Similarity ── */
    {
        float ie[512], te[512], sim[1];
        for (int i = 0; i < 512; i++) { ie[i] = 1.0f; te[i] = 0.5f; }
        t0 = now_ms();
        for (int r = 0; r < N / 10; r++) {
            mm_cosine_sim_matrix(ie, te, 1, 512, sim);
        }
        t1 = now_ms();
        printf("  cosine_sim:          %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 10, t1 - t0, (t1 - t0) / (double)(N / 10) * 1000.0);
    }

    /* ── Conv2D (SD) ── */
    {
        mm_conv2d_t c;
        mm_conv2d_init(&c, 3, 16, 3, 1, 1, 1);
        float x[3 * 32 * 32], y[16 * 32 * 32];
        memset(x, 0, sizeof(x));
        t0 = now_ms();
        for (int r = 0; r < N / 20; r++) {
            mm_conv2d_forward(&c, x, 32, 32, y);
        }
        t1 = now_ms();
        printf("  mm_conv2d_fwd:       %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 20, t1 - t0, (t1 - t0) / (double)(N / 20) * 1000.0);
        mm_conv2d_free(&c);
    }

    /* ── SD Schedule ── */
    {
        t0 = now_ms();
        for (int r = 0; r < N / 10; r++) {
            mm_schedule_t s;
            mm_schedule_init(&s, 200);
            mm_schedule_free(&s);
        }
        t1 = now_ms();
        printf("  mm_schedule_init:    %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 10, t1 - t0, (t1 - t0) / (double)(N / 10) * 1000.0);
    }

    /* ── VLM Token Seq ── */
    {
        t0 = now_ms();
        for (int r = 0; r < N / 10; r++) {
            mm_vlm_token_seq_t seq;
            mm_vlm_token_seq_init(&seq, 128);
            mm_vlm_token_seq_push(&seq, 42);
            mm_vlm_token_seq_free(&seq);
        }
        t1 = now_ms();
        printf("  vlm_token_seq:       %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 10, t1 - t0, (t1 - t0) / (double)(N / 10) * 1000.0);
    }

    /* ── Mel Filterbank ── */
    {
        t0 = now_ms();
        for (int r = 0; r < N / 20; r++) {
            mm_mel_filterbank_t fb;
            mm_mel_filterbank_init(&fb, 80, 400, 16000);
            mm_mel_filterbank_free(&fb);
        }
        t1 = now_ms();
        printf("  mel_filterbank_init: %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 20, t1 - t0, (t1 - t0) / (double)(N / 20) * 1000.0);
    }

    /* ── Whisper Model Init ── */
    {
        t0 = now_ms();
        for (int r = 0; r < N / 20; r++) {
            mm_whisper_model_t model;
            mm_whisper_model_init(&model, 256, 256, 3, 3, 40);
            mm_whisper_model_free(&model);
        }
        t1 = now_ms();
        printf("  whisper_model_init:  %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 20, t1 - t0, (t1 - t0) / (double)(N / 20) * 1000.0);
    }

    /* ── Video Clip Init ── */
    {
        t0 = now_ms();
        for (int r = 0; r < N / 10; r++) {
            mm_video_clip_t clip;
            mm_video_clip_init(&clip, 8, 112, 112, 3);
            mm_video_clip_free(&clip);
        }
        t1 = now_ms();
        printf("  video_clip_init:     %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 10, t1 - t0, (t1 - t0) / (double)(N / 10) * 1000.0);
    }

    /* ── Conv3D ── */
    {
        mm_conv3d_t c3d;
        mm_conv3d_init(&c3d, 3, 64, 3, 3, 3, 1, 1, 1, 1, 1, 1);
        float x[3 * 16 * 112 * 112], y[64 * 16 * 112 * 112];
        memset(x, 0, sizeof(x));
        int ot, oh, ow;
        t0 = now_ms();
        for (int r = 0; r < N / 20; r++) {
            mm_conv3d_forward(&c3d, x, 16, 112, 112, y, &ot, &oh, &ow);
        }
        t1 = now_ms();
        printf("  conv3d_fwd:          %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 20, t1 - t0, (t1 - t0) / (double)(N / 20) * 1000.0);
        mm_conv3d_free(&c3d);
    }

    printf("\nDone.\n");
    return 0;
}
