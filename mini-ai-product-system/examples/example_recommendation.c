#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "recommendation.h"

int main(void) {
    uint64_t seed = 42;
    printf("=== Recommendation System Demo ===\n\n");

    RecUserTower user_tower;
    rec_embedding_init(&user_tower.emb, seed);
    user_tower.bias = 0.1f;

    RecItemTower *items = (RecItemTower *)malloc(200 * sizeof(RecItemTower));
    for (int i = 0; i < 200; i++) {
        rec_embedding_init(&items[i].emb, seed + (uint64_t)(i + 1));
        items[i].bias = (float)(i % 10) * 0.02f;
        items[i].item_id = i + 1000;
    }

    RecCandidateList candidates;
    rec_collab_filter(&user_tower, items, 200, 20, &candidates);
    printf("Collaborative Filtering Top-10:\n");
    for (int i = 0; i < 10 && i < candidates.count; i++)
        printf("  item_%d  score=%.4f\n", candidates.items[i].item_id, candidates.items[i].score);

    RecCandidateList ann_result;
    rec_ann_search(&user_tower.emb, items, 200, 10, &ann_result);
    printf("\nANN Search Top-5:\n");
    for (int i = 0; i < 5 && i < ann_result.count; i++)
        printf("  item_%d  score=%.4f\n", ann_result.items[i].item_id, ann_result.items[i].score);

    RecRecallChannel channels[3];
    channels[0].source = REC_SOURCE_COLLAB;  channels[0].weight = 1.0f;
    channels[1].source = REC_SOURCE_POPULAR;  channels[1].weight = 0.7f;
    channels[2].source = REC_SOURCE_RANDOM;   channels[2].weight = 0.3f;
    RecRecallResult recall;
    rec_multi_channel_recall(&user_tower, items, 200, channels, 3, seed, &recall);
    printf("\nMulti-channel Recall (%d candidates):\n", recall.candidates.count);
    for (int i = 0; i < 5 && i < recall.candidates.count; i++)
        printf("  item_%d  score=%.4f\n", recall.candidates.items[i].item_id,
               recall.candidates.items[i].score);

    RecFeatureVec user_feat;
    user_feat.num_features = 8;
    for (int i = 0; i < 8; i++) user_feat.features[i] = (float)(i + 1) / 10.0f;
    RecFeatureVec session_feat;
    session_feat.num_features = 4;
    for (int i = 0; i < 4; i++) session_feat.features[i] = (float)(i * 3) / 10.0f;

    int32_t dims[3] = {32, 16, 1};
    RecRankingDNN dnn;
    rec_ranking_dnn_init(&dnn, dims, 3, seed + 100);
    RecCandidateList ranked;
    rec_rank_candidates(&dnn, &user_feat, &session_feat, items, &candidates, &ranked);
    printf("\nRanked Top-5:\n");
    for (int i = 0; i < 5 && i < ranked.count; i++)
        printf("  item_%d  score=%.4f\n", ranked.items[i].item_id, ranked.items[i].score);

    int64_t timestamps[REC_MAX_CANDIDATES];
    for (int i = 0; i < ranked.count; i++)
        timestamps[i] = (int64_t)time(NULL) - (int64_t)(i * 3600);
    rec_freshness_boost(&ranked, timestamps, (int64_t)time(NULL), 0.5f);
    printf("\nFreshness Boost Top-5:\n");
    for (int i = 0; i < 5 && i < ranked.count; i++)
        printf("  item_%d  score=%.4f\n", ranked.items[i].item_id, ranked.items[i].score);

    int32_t blocked[] = {1005, 1012, 1020};
    rec_business_rule_filter(&ranked, blocked, 3);
    printf("\nAfter Business Rule Filter (%d items):\n", ranked.count);
    for (int i = 0; i < 5 && i < ranked.count; i++)
        printf("  item_%d  score=%.4f\n", ranked.items[i].item_id, ranked.items[i].score);

    rec_diversity_rerank(&ranked, items, 0.5f, 5);
    printf("\nDiversity Re-rank (%d items):\n", ranked.count);
    for (int i = 0; i < ranked.count && i < 5; i++)
        printf("  item_%d  score=%.4f\n", ranked.items[i].item_id, ranked.items[i].score);

    const char *tags[] = {"tech", "ai", "coding"};
    RecFeatureVec content_feat;
    rec_content_features("Neural Networks Guide", "education", tags, 3, &content_feat);
    printf("\nContent Features (%d dims):\n", content_feat.num_features);
    for (int i = 0; i < 5 && i < content_feat.num_features; i++)
        printf("  f[%d] = %.4f\n", i, content_feat.features[i]);

    float cs_score;
    rec_cold_start_score(&content_feat, &user_feat, &cs_score);
    printf("\nCold Start Score: %.4f\n", cs_score);

    RecABConfig ab_cfg;
    rec_ab_config_init(&ab_cfg, 50, 50, seed);
    printf("\nA/B User Assignments:\n");
    for (int uid = 100; uid < 110; uid++)
        printf("  user_%d -> %s\n", uid, rec_ab_decide(&ab_cfg, uid) ? "treatment" : "control");

    double ctrl[50], treat[50];
    for (int i = 0; i < 50; i++) {
        ctrl[i] = 0.10 + (rand() % 10) * 0.005;
        treat[i] = 0.12 + (rand() % 10) * 0.005;
    }
    RecABResult ab_result;
    rec_ab_evaluate(ctrl, 50, treat, 50, &ab_result);
    printf("\nA/B Result: ctrl=%.4f treat=%.4f p=%.4f sig=%d\n",
           ab_result.ctr_control, ab_result.ctr_treatment,
           ab_result.p_value, ab_result.significant);

    free(items);
    printf("\nDone.\n");
    return 0;
}
