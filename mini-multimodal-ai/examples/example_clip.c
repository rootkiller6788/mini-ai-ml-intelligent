#include "clip_contrastive.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#define BATCH    4
#define IMG_H    224
#define IMG_W    224
#define IMG_C    3

static float dummy_image[BATCH][IMG_H * IMG_W * IMG_C];

static void generate_dummy_images(void) {
    for (int b = 0; b < BATCH; b++) {
        for (int i = 0; i < IMG_H * IMG_W * IMG_C; i++) {
            dummy_image[b][i] = (float)rand() / (float)RAND_MAX;
        }
    }
}

int main(void) {
    srand((unsigned)time(NULL));
    generate_dummy_images();

    printf("=== CLIP Zero-Shot Classification Demo ===\n\n");

    mm_clip_model_t model;
    mm_clip_init(&model, MM_CLIP_EMBED_DIM, 3);
    printf("CLIP model initialized: embed_dim=%d, layers=%d\n\n", model.embed_dim, 3);

    float img_emb[MM_CLIP_EMBED_DIM] = {0};
    mm_clip_encode_image(&model, dummy_image[0], IMG_H, IMG_W, IMG_C, img_emb);

    printf("Image encoded to embedding: [");
    for (int i = 0; i < 5; i++) printf("%.3f%s", img_emb[i], i < 4 ? ", " : "");
    printf(" ...]\n");
    printf("Embedding L2 norm: %f\n\n", mm_l2_norm(img_emb, MM_CLIP_EMBED_DIM));

    const char* classes[] = {"cat", "dog", "bird", "fish", "car", "tree", "house", "person"};
    int num_classes = 8;

    printf("--- Zero-Shot Classification ---\n");
    int predicted = mm_clip_zeroshot(img_emb, classes, num_classes, 1);
    printf("Predicted class: %s\n\n", classes[predicted]);

    printf("--- Text Encoding ---\n");
    const char* prompts[] = {
        "a photo of a cat",
        "a photo of a dog",
        "a photo of a bird",
        "a photo of a fish"
    };
    float text_embs[4][MM_CLIP_EMBED_DIM];
    for (int i = 0; i < 4; i++) {
        mm_clip_encode_text(&model, prompts[i], text_embs[i]);
        float sim = 0.0f;
        for (int d = 0; d < MM_CLIP_EMBED_DIM; d++) sim += img_emb[d] * text_embs[i][d];
        printf("  '%s' -> similarity: %.4f\n", prompts[i], sim);
    }
    printf("\n");

    printf("--- Contrastive Loss (InfoNCE) ---\n");
    float sim_matrix[BATCH * BATCH];
    float img_embs[BATCH][MM_CLIP_EMBED_DIM];
    float txt_embs[BATCH][MM_CLIP_EMBED_DIM];

    const char* batch_texts[] = {"a cat", "a dog", "a bird", "a fish"};
    for (int b = 0; b < BATCH; b++) {
        mm_clip_encode_image(&model, dummy_image[b], IMG_H, IMG_W, IMG_C, img_embs[b]);
        mm_clip_encode_text(&model, batch_texts[b], txt_embs[b]);
    }

    mm_cosine_sim_matrix((float*)img_embs, (float*)txt_embs, BATCH, MM_CLIP_EMBED_DIM, sim_matrix);
    float loss = mm_infonce_loss(sim_matrix, BATCH);
    printf("InfoNCE loss: %.6f\n\n", loss);

    printf("--- Image-Text Retrieval ---\n");
    float query_emb[MM_CLIP_EMBED_DIM];
    memcpy(query_emb, img_embs[0], MM_CLIP_EMBED_DIM * sizeof(float));

    int indices[BATCH];
    mm_clip_retrieve(query_emb, (float*)txt_embs, BATCH, MM_CLIP_EMBED_DIM, indices, BATCH);
    printf("Query image 0 matches: ");
    for (int i = 0; i < BATCH; i++) printf("text_%d(%.3f) ", indices[i],
        query_emb[0] * txt_embs[indices[i]][0]);
    printf("\n\n");

    printf("--- Training Step ---\n");
    mm_clip_train_step(&model, (float*)dummy_image, batch_texts, BATCH, IMG_H, IMG_W, IMG_C, 1e-4f);
    printf("One training step completed.\n\n");

    printf("--- Gradient Computation (InfoNCE) ---\n");
    float img_grad[BATCH][MM_CLIP_EMBED_DIM];
    float txt_grad[BATCH][MM_CLIP_EMBED_DIM];
    memset(img_grad, 0, sizeof(img_grad));
    memset(txt_grad, 0, sizeof(txt_grad));
    mm_infonce_grad(sim_matrix, BATCH, (float*)img_grad, (float*)txt_grad);
    printf("Gradients computed for %d pairs.\n", BATCH);

    mm_clip_free(&model);
    printf("\n=== Demo Complete ===\n");
    return 0;
}
