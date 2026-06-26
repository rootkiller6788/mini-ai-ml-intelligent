#ifndef RECOMMENDATION_H
#define RECOMMENDATION_H

#include <stddef.h>
#include <stdint.h>

#define REC_MAX_USERS      100000
#define REC_MAX_ITEMS      500000
#define REC_EMBED_DIM      64
#define REC_MAX_CANDIDATES 1000
#define REC_MAX_FEATURES   256
#define REC_TOPK           100

typedef struct {
    float data[REC_EMBED_DIM];
} RecEmbedding;

typedef struct {
    int32_t item_id;
    float   score;
} RecCandidate;

typedef struct {
    RecCandidate items[REC_MAX_CANDIDATES];
    int32_t count;
} RecCandidateList;

typedef struct {
    RecEmbedding emb;
    float        bias;
} RecUserTower;

typedef struct {
    RecEmbedding emb;
    float        bias;
    int32_t      item_id;
} RecItemTower;

typedef struct {
    float features[REC_MAX_FEATURES];
    int32_t num_features;
} RecFeatureVec;

typedef enum {
    REC_SOURCE_COLLAB,
    REC_SOURCE_ANN,
    REC_SOURCE_POPULAR,
    REC_SOURCE_RANDOM,
    REC_SOURCE_CONTENT
} RecRecallSource;

typedef struct {
    RecRecallSource source;
    float           weight;
} RecRecallChannel;

typedef struct {
    RecCandidateList candidates;
    RecRecallChannel channels[5];
    int32_t          num_channels;
} RecRecallResult;

typedef struct {
    float w[REC_MAX_FEATURES];
    float bias;
} RecDNNLayer;

typedef struct {
    RecDNNLayer layers[4];
    int32_t     num_layers;
    int32_t     hidden_dims[4];
} RecRankingDNN;

typedef struct {
    int32_t  control_pct;
    int32_t  treatment_pct;
    uint64_t seed;
} RecABConfig;

typedef struct {
    double ctr_control;
    double ctr_treatment;
    double p_value;
    int    significant;
} RecABResult;

void    rec_embedding_init(RecEmbedding *e, uint64_t seed);
float   rec_dot_product(const RecEmbedding *a, const RecEmbedding *b);

void    rec_user_tower_forward(const RecUserTower *tower, const RecFeatureVec *feat,
                               RecEmbedding *out);
void    rec_item_tower_forward(const RecItemTower *tower, const RecFeatureVec *feat,
                               RecEmbedding *out);

void    rec_ann_search(const RecEmbedding *query, const RecItemTower *items,
                       int32_t num_items, int32_t k, RecCandidateList *result);
void    rec_popular_recall(const int32_t *popular_ids, int32_t n, int32_t k,
                           RecCandidateList *result);
void    rec_random_recall(const RecItemTower *items, int32_t n, int32_t k,
                          uint64_t seed, RecCandidateList *result);

void    rec_collab_filter(const RecUserTower *user, const RecItemTower *items,
                          int32_t num_items, int32_t k, RecCandidateList *result);
void    rec_multi_channel_recall(const RecUserTower *user, const RecItemTower *items,
                                 int32_t num_items, const RecRecallChannel *chans,
                                 int32_t num_chans, uint64_t seed,
                                 RecRecallResult *result);

void    rec_ranking_dnn_init(RecRankingDNN *dnn, const int32_t *dims, int32_t num_layers,
                             uint64_t seed);
float   rec_ranking_dnn_forward(const RecRankingDNN *dnn, const RecFeatureVec *feat);
void    rec_rank_candidates(const RecRankingDNN *dnn, const RecFeatureVec *user_feat,
                            const RecFeatureVec *session_feat,
                            const RecItemTower *items, const RecCandidateList *cands,
                            RecCandidateList *ranked);

void    rec_diversity_rerank(RecCandidateList *ranked, const RecItemTower *items,
                             float lambda, int32_t k);
void    rec_freshness_boost(RecCandidateList *ranked, const int64_t *timestamps,
                            int64_t now, float alpha);
void    rec_business_rule_filter(RecCandidateList *ranked,
                                 const int32_t *blocked_ids, int32_t n_blocked);

void    rec_content_features(const char *title, const char *category,
                             const char **tags, int32_t n_tags, RecFeatureVec *out);
void    rec_cold_start_score(const RecFeatureVec *new_item,
                             const RecFeatureVec *user_profile, float *score);

void    rec_ab_config_init(RecABConfig *cfg, int32_t control_pct, int32_t treatment_pct,
                           uint64_t seed);
int     rec_ab_decide(const RecABConfig *cfg, int32_t user_id);
void    rec_ab_evaluate(double *ctrl_metrics, int32_t n_ctrl,
                        double *treat_metrics, int32_t n_treat,
                        RecABResult *result);

#endif
