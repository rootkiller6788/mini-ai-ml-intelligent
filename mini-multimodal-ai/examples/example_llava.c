#include "vlm_llama.h"
#include "clip_contrastive.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#define TEST_IMG_H  336
#define TEST_IMG_W  336
#define TEST_IMG_C  3

int main(void) {
    srand((unsigned)time(NULL));

    printf("=== LLaVA Vision-Language Model Demo ===\n\n");

    printf("--- Model Initialization ---\n");
    mm_vlm_model_t model;
    int vision_dim = 512;
    int llm_dim = 1024;
    int num_v_layers = 3;
    int num_llm_layers = 4;
    mm_vlm_model_init(&model, vision_dim, llm_dim, num_v_layers, num_llm_layers);
    printf("LLaVA initialized: vision=%d, llm=%d, v_layers=%d, llm_layers=%d\n",
           vision_dim, llm_dim, num_v_layers, num_llm_layers);
    printf("  num_patches: %d, image_token: %d, max_seq_len: %d\n\n",
           model.num_patches, MM_VLM_IMAGE_TOKEN_ID, model.llm.max_seq_len);

    printf("--- Vision Encoding ---\n");
    float* test_img = (float*)malloc((size_t)TEST_IMG_H * TEST_IMG_W * TEST_IMG_C * sizeof(float));
    for (int i = 0; i < TEST_IMG_H * TEST_IMG_W * TEST_IMG_C; i++)
        test_img[i] = (float)rand() / (float)RAND_MAX;

    float* img_feat = (float*)malloc((size_t)model.num_patches * vision_dim * sizeof(float));
    mm_vlm_encode_image(&model, test_img, TEST_IMG_H, TEST_IMG_W, TEST_IMG_C, img_feat);
    printf("Image encoded: %d patches x %d dim\n", model.num_patches, vision_dim);
    printf("  features[0..4]: ");
    for (int i = 0; i < 5; i++) printf("%.3f ", img_feat[i]);
    printf("...\n\n");

    printf("--- Projection Layer ---\n");
    float* proj_feat = (float*)malloc((size_t)model.num_patches * llm_dim * sizeof(float));
    mm_vlm_project_features(&model.projector, img_feat, model.num_patches, proj_feat);
    printf("Features projected: %d tokens x %d dim\n", model.num_patches, llm_dim);
    printf("  proj[0..4]: ");
    for (int i = 0; i < 5; i++) printf("%.3f ", proj_feat[i]);
    printf("...\n\n");

    printf("--- Conversation ---\n");
    mm_vlm_conversation_t conv;
    mm_vlm_conversation_init(&conv, 10);
    mm_vlm_conversation_add(&conv, "What is in this image?", MM_VLM_MSG_USER, 1);
    mm_vlm_conversation_add(&conv, "I see a cat sitting on a chair.", MM_VLM_MSG_ASSISTANT, 0);
    mm_vlm_conversation_add(&conv, "What color is the cat?", MM_VLM_MSG_USER, 1);
    printf("Conversation: %d messages\n", conv.num_messages);
    for (int i = 0; i < conv.num_messages; i++) {
        printf("  [%s] %s %s\n",
               conv.messages[i].role == MM_VLM_MSG_USER ? "USER" : "ASSISTANT",
               conv.messages[i].text,
               conv.messages[i].has_image ? "(with image)" : "");
    }
    printf("\n");

    printf("--- Input Preparation ---\n");
    int input_ids[MM_VLM_MAX_SEQ_LEN];
    int num_input;
    int image_pos = -1;
    mm_vlm_prepare_input(&model, &conv, img_feat, input_ids, &num_input, &image_pos);
    printf("Prepared input: %d tokens, image_pos=%d\n", num_input, image_pos);
    printf("  input_ids[0..8]: ");
    for (int i = 0; i < 9 && i < num_input; i++) printf("%d ", input_ids[i]);
    printf("...\n\n");

    printf("--- Token Sequence ---\n");
    mm_vlm_token_seq_t seq;
    mm_vlm_token_seq_init(&seq, 100);
    for (int i = 0; i < num_input && i < 100; i++) mm_vlm_token_seq_push(&seq, input_ids[i]);
    printf("Token sequence: len=%zu, cap=%zu\n\n", seq.len, seq.cap);

    printf("--- Visual Question Answering ---\n");
    char answer[512];
    mm_vlm_visual_qa(&model, test_img, TEST_IMG_H, TEST_IMG_W, TEST_IMG_C, "How many objects?", answer, sizeof(answer));
    printf("Q: 'How many objects?' -> A: '%s'\n\n", answer);

    printf("--- OCR ---\n");
    char ocr_text[512];
    mm_vlm_ocr(&model, test_img, TEST_IMG_H, TEST_IMG_W, TEST_IMG_C, ocr_text, sizeof(ocr_text));
    printf("OCR result: %s\n\n", ocr_text);

    printf("--- Region Understanding ---\n");
    mm_vlm_bbox_t bbox = {10.0f, 20.0f, 100.0f, 150.0f};
    char region_answer[512];
    mm_vlm_region_understand(&model, test_img, TEST_IMG_H, TEST_IMG_W, TEST_IMG_C, bbox, "What object?", region_answer, sizeof(region_answer));
    printf("Region Q: 'What object?' -> A: '%s'\n\n", region_answer);

    printf("--- Multi-turn Conversation ---\n");
    char response[512];
    mm_vlm_multiturn(&model, &conv, test_img, TEST_IMG_H, TEST_IMG_W, TEST_IMG_C, response, sizeof(response));
    printf("Multi-turn response: %s\n\n", response);

    printf("--- RMSNorm Test ---\n");
    float rms_in[8] = {1.0f, 2.0f, -3.0f, 4.0f, -1.0f, 0.5f, -2.0f, 1.5f};
    float rms_out[8];
    mm_vlm_rmsnorm(rms_in, 8, rms_out);
    printf("RMSNorm: ");
    for (int i = 0; i < 8; i++) printf("%.3f ", rms_out[i]);
    printf("\n\n");

    printf("--- SiLU Activation ---\n");
    float silu_in[5] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    float silu_out[5];
    mm_vlm_silu_forward(silu_in, 5, silu_out);
    printf("SiLU: ");
    for (int i = 0; i < 5; i++) printf("%.3f ", silu_out[i]);
    printf("\n\n");

    printf("--- LLM Forward ---\n");
    float* x_input = (float*)calloc((size_t)llm_dim * 4, sizeof(float));
    for (int i = 0; i < llm_dim * 4; i++) x_input[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 0.02f;
    float logits[32000] = {0};
    int next_token;
    mm_vlm_llm_forward(&model.llm, x_input, 4, 0, logits, &next_token);
    printf("LLM next token: %d\n\n", next_token);

    printf("--- Generation ---\n");
    int output_ids[64];
    int num_output;
    mm_vlm_generate(&model, input_ids, num_input, 32, output_ids, &num_output);
    printf("Generated %d tokens: %s\n\n", num_output, mm_vlm_decode(output_ids, num_output));

    printf("=== Demo Complete ===\n");

    mm_vlm_token_seq_free(&seq);
    mm_vlm_conversation_free(&conv);
    free(test_img);
    free(img_feat);
    free(proj_feat);
    free(x_input);
    mm_vlm_model_free(&model);

    return 0;
}
