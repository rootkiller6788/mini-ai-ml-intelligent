/*
 * mini-multimodal-ai — Core Tests
 *
 * Unit tests for CLIP, image generation (Stable Diffusion), VLM,
 * audio (Whisper), video understanding.
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

static int tests_run = 0, tests_passed = 0;

#define TEST(name) do { tests_run++; printf("  TEST %s ... ", name); } while(0)
#define PASS() do { tests_passed++; printf("PASS\n"); } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); return 1; } while(0)
#define CHECK(cond, msg) if (!(cond)) FAIL(msg)

/* ── CLIP Tests ── */
static int test_clip_init(void) {
    TEST("clip_init");
    mm_clip_model_t model;
    mm_clip_init(&model, 512, 6);
    CHECK(model.embed_dim == 512, "embed_dim wrong");
    mm_clip_free(&model);
    PASS();
    return 0;
}

static int test_clip_cosine_sim(void) {
    TEST("clip_cosine_sim");
    float ie[4] = {1, 0, 0, 0}, te[4] = {1, 0, 0, 0};
    float sim[1];
    mm_cosine_sim_matrix(ie, te, 1, 4, sim);
    CHECK(fabsf(sim[0] - 1.0f) < 0.01f, "cosine sim should be ~1.0");
    PASS();
    return 0;
}

static int test_clip_l2_norm(void) {
    TEST("clip_l2_norm");
    float x[] = {3.0f, 4.0f};
    float n = mm_l2_norm(x, 2);
    CHECK(fabsf(n - 5.0f) < 0.01f, "L2 norm wrong");
    mm_l2_normalize(x, 2);
    CHECK(fabsf(mm_l2_norm(x, 2) - 1.0f) < 0.01f, "normalized L2 != 1");
    PASS();
    return 0;
}

/* ── Image Generation Tests ── */
static int test_sd_init(void) {
    TEST("sd_init");
    mm_stable_diffusion_t sd;
    mm_sd_init(&sd, 256, 4, 64, 1);
    CHECK(sd.image_size == 256, "image_size wrong");
    mm_sd_free(&sd);
    PASS();
    return 0;
}

static int test_schedule_create(void) {
    TEST("schedule_create");
    mm_schedule_t s;
    mm_schedule_init(&s, 100);
    CHECK(s.n == 100, "schedule size wrong");
    mm_schedule_free(&s);
    PASS();
    return 0;
}

static int test_conv2d_init_forward(void) {
    TEST("mm_conv2d_init_forward");
    mm_conv2d_t c;
    mm_conv2d_init(&c, 3, 16, 3, 1, 1, 1);
    float x[3 * 32 * 32], y[16 * 32 * 32];
    memset(x, 0, sizeof(x));
    mm_conv2d_forward(&c, x, 32, 32, y);
    mm_conv2d_free(&c);
    PASS();
    return 0;
}

/* ── VLM Tests ── */
static int test_vlm_token_seq(void) {
    TEST("vlm_token_seq");
    mm_vlm_token_seq_t seq;
    mm_vlm_token_seq_init(&seq, 64);
    mm_vlm_token_seq_push(&seq, 42);
    CHECK(seq.len == 1, "token_seq len wrong");
    mm_vlm_token_seq_free(&seq);
    PASS();
    return 0;
}

static int test_vlm_conversation(void) {
    TEST("vlm_conversation");
    mm_vlm_conversation_t conv;
    mm_vlm_conversation_init(&conv, 4);
    mm_vlm_conversation_add(&conv, "Hello", MM_VLM_MSG_USER, 0);
    CHECK(conv.num_messages == 1, "conv message count wrong");
    mm_vlm_conversation_free(&conv);
    PASS();
    return 0;
}

/* ── Audio/Whisper Tests ── */
static int test_mel_filterbank(void) {
    TEST("mel_filterbank_init");
    mm_mel_filterbank_t fb;
    mm_mel_filterbank_init(&fb, 80, 400, 16000);
    mm_mel_filterbank_free(&fb);
    PASS();
    return 0;
}

static int test_whisper_model_init(void) {
    TEST("whisper_model_init");
    mm_whisper_model_t model;
    mm_whisper_model_init(&model, 512, 512, 6, 6, 80);
    CHECK(model.encoder_dim == 512, "encoder_dim wrong");
    mm_whisper_model_free(&model);
    PASS();
    return 0;
}

static int test_whisper_result(void) {
    TEST("whisper_result_init");
    mm_whisper_result_t result;
    mm_whisper_result_init(&result, 8);
    CHECK(result.max_segments == 8, "max_segments wrong");
    mm_whisper_result_free(&result);
    PASS();
    return 0;
}

/* ── Video Understanding Tests ── */
static int test_video_clip_init(void) {
    TEST("video_clip_init");
    mm_video_clip_t clip;
    mm_video_clip_init(&clip, 16, 224, 224, 3);
    CHECK(clip.num_frames == 16, "num_frames wrong");
    mm_video_clip_free(&clip);
    PASS();
    return 0;
}

static int test_conv3d_init(void) {
    TEST("conv3d_init");
    mm_conv3d_t c3d;
    mm_conv3d_init(&c3d, 3, 64, 3, 3, 3, 1, 1, 1, 1, 1, 1);
    CHECK(c3d.in_ch == 3, "in_ch wrong");
    mm_conv3d_free(&c3d);
    PASS();
    return 0;
}

static int test_video_model_init(void) {
    TEST("video_model_init");
    mm_video_model_t model;
    mm_video_model_init(&model, MM_VIDEO_ARCH_C3D, 400, 16, 112);
    CHECK(model.num_classes == 400, "num_classes wrong");
    mm_video_model_free(&model);
    PASS();
    return 0;
}

static int test_iou_3d(void) {
    TEST("iou_3d");
    float iou = mm_iou_3d(0, 10, 5, 15);
    CHECK(iou >= 0.0f && iou <= 1.0f, "IoU out of range");
    PASS();
    return 0;
}

int main(void) {
    printf("=== mini-multimodal-ai Unit Tests ===\n\n");

    int failed = 0;
    failed += test_clip_init();
    failed += test_clip_cosine_sim();
    failed += test_clip_l2_norm();
    failed += test_sd_init();
    failed += test_schedule_create();
    failed += test_conv2d_init_forward();
    failed += test_vlm_token_seq();
    failed += test_vlm_conversation();
    failed += test_mel_filterbank();
    failed += test_whisper_model_init();
    failed += test_whisper_result();
    failed += test_video_clip_init();
    failed += test_conv3d_init();
    failed += test_video_model_init();
    failed += test_iou_3d();

    printf("\n=== Results: %d/%d passed, %d failed ===\n",
           tests_passed, tests_run, failed);
    return failed > 0 ? 1 : 0;
}
