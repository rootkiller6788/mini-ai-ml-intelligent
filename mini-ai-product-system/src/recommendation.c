#include "recommendation.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static uint64_t rec_splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static float rec_randf(uint64_t *state) {
    return (float)(rec_splitmix64(state) & 0xFFFFFF) / 16777216.0f;
}

static int rec_cmp_candidate_desc(const void *a, const void *b) {
    const RecCandidate *ca = (const RecCandidate *)a;
    const RecCandidate *cb = (const RecCandidate *)b;
    if (ca->score > cb->score) return -1;
    if (ca->score < cb->score) return 1;
    return 0;
}

static int rec_cmp_candidate_asc(const void *a, const void *b) {
    return -rec_cmp_candidate_desc(a, b);
}

void rec_embedding_init(RecEmbedding *e, uint64_t seed) {
    float scale = 1.0f / sqrtf((float)REC_EMBED_DIM);
    for (int i = 0; i < REC_EMBED_DIM; i++) {
        e->data[i] = (rec_randf(&seed) * 2.0f - 1.0f) * scale;
    }
}

float rec_dot_product(const RecEmbedding *a, const RecEmbedding *b) {
    float dp = 0.0f;
    for (int i = 0; i < REC_EMBED_DIM; i++) {
        dp += a->data[i] * b->data[i];
    }
    return dp;
}

void rec_user_tower_forward(const RecUserTower *tower, const RecFeatureVec *feat,
                            RecEmbedding *out) {
    int nf = feat->num_features;
    for (int d = 0; d < REC_EMBED_DIM; d++) {
        float sum = 0.0f;
        for (int f = 0; f < nf && f < REC_MAX_FEATURES; f++) {
            sum += feat->features[f] * 0.1f;
        }
        out->data[d] = tanhf(sum) + tower->emb.data[d] * 0.01f;
    }
}

void rec_item_tower_forward(const RecItemTower *tower, const RecFeatureVec *feat,
                            RecEmbedding *out) {
    int nf = feat->num_features;
    for (int d = 0; d < REC_EMBED_DIM; d++) {
        float sum = 0.0f;
        for (int f = 0; f < nf && f < REC_MAX_FEATURES; f++) {
            sum += feat->features[f] * 0.1f;
        }
        out->data[d] = tanhf(sum) + tower->emb.data[d] * 0.01f;
    }
}

void rec_ann_search(const RecEmbedding *query, const RecItemTower *items,
                    int32_t num_items, int32_t k, RecCandidateList *result) {
    if (k > REC_MAX_CANDIDATES) k = REC_MAX_CANDIDATES;
    result->count = 0;
    for (int i = 0; i < num_items && i < k; i++) {
        result->items[i].item_id = items[i].item_id;
        result->items[i].score = rec_dot_product(query, &items[i].emb);
        result->count++;
    }
    qsort(result->items, result->count, sizeof(RecCandidate), rec_cmp_candidate_desc);
}

void rec_popular_recall(const int32_t *popular_ids, int32_t n, int32_t k,
                        RecCandidateList *result) {
    if (k > REC_MAX_CANDIDATES) k = REC_MAX_CANDIDATES;
    int m = n < k ? n : k;
    result->count = m;
    for (int i = 0; i < m; i++) {
        result->items[i].item_id = popular_ids[i];
        result->items[i].score = (float)(n - i) / (float)n;
    }
}

void rec_random_recall(const RecItemTower *items, int32_t n, int32_t k,
                       uint64_t seed, RecCandidateList *result) {
    if (k > REC_MAX_CANDIDATES) k = REC_MAX_CANDIDATES;
    int m = n < k ? n : k;
    int *indices = (int *)malloc((size_t)n * sizeof(int));
    for (int i = 0; i < n; i++) indices[i] = i;
    for (int i = n - 1; i > 0; i--) {
        int j = (int)(rec_randf(&seed) * (i + 1));
        int tmp = indices[i]; indices[i] = indices[j]; indices[j] = tmp;
    }
    result->count = m;
    for (int i = 0; i < m; i++) {
        int idx = indices[i];
        result->items[i].item_id = items[idx].item_id;
        result->items[i].score = 0.1f;
    }
    free(indices);
}

void rec_collab_filter(const RecUserTower *user, const RecItemTower *items,
                       int32_t num_items, int32_t k, RecCandidateList *result) {
    if (k > REC_MAX_CANDIDATES) k = REC_MAX_CANDIDATES;
    if (k > num_items) k = num_items;
    result->count = 0;
    for (int i = 0; i < num_items; i++) {
        if (result->count >= k) {
            if (i < k) continue;
            float s = rec_dot_product(&user->emb, &items[i].emb) + items[i].bias;
            float min_score = result->items[k - 1].score;
            if (s <= min_score) continue;
            int pos = k - 1;
            while (pos > 0 && result->items[pos - 1].score < s) pos--;
            memmove(&result->items[pos + 1], &result->items[pos],
                    (size_t)(k - 1 - pos) * sizeof(RecCandidate));
            result->items[pos].item_id = items[i].item_id;
            result->items[pos].score = s;
        } else {
            int idx = result->count;
            result->items[idx].item_id = items[i].item_id;
            result->items[idx].score = rec_dot_product(&user->emb, &items[i].emb) + items[i].bias;
            result->count++;
        }
    }
    if (result->count > 1) {
        qsort(result->items, result->count, sizeof(RecCandidate), rec_cmp_candidate_desc);
    }
}

void rec_multi_channel_recall(const RecUserTower *user, const RecItemTower *items,
                              int32_t num_items, const RecRecallChannel *chans,
                              int32_t num_chans, uint64_t seed,
                              RecRecallResult *result) {
    result->candidates.count = 0;
    result->num_channels = num_chans < 5 ? num_chans : 5;
    for (int c = 0; c < result->num_channels; c++) {
        result->channels[c] = chans[c];
        RecCandidateList tmp;
        int k_per_channel = REC_TOPK / num_chans;
        if (k_per_channel < 1) k_per_channel = 1;
        switch (chans[c].source) {
            case REC_SOURCE_COLLAB:
                rec_collab_filter(user, items, num_items, k_per_channel, &tmp);
                break;
            case REC_SOURCE_ANN:
                rec_ann_search(&user->emb, items, num_items, k_per_channel, &tmp);
                break;
            case REC_SOURCE_POPULAR:
                {
                    int32_t pop_ids[REC_MAX_CANDIDATES];
                    for (int i = 0; i < num_items && i < REC_MAX_CANDIDATES; i++)
                        pop_ids[i] = items[i].item_id;
                    rec_popular_recall(pop_ids, num_items, k_per_channel, &tmp);
                }
                break;
            case REC_SOURCE_RANDOM:
                rec_random_recall(items, num_items, k_per_channel, seed + (uint64_t)c, &tmp);
                break;
            case REC_SOURCE_CONTENT:
                rec_random_recall(items, num_items, k_per_channel, seed + 1000ULL + (uint64_t)c, &tmp);
                break;
            default:
                tmp.count = 0;
                break;
        }
        for (int j = 0; j < tmp.count; j++) {
            tmp.items[j].score *= chans[c].weight;
            if (result->candidates.count < REC_MAX_CANDIDATES) {
                result->candidates.items[result->candidates.count++] = tmp.items[j];
            }
        }
    }
    qsort(result->candidates.items, result->candidates.count,
          sizeof(RecCandidate), rec_cmp_candidate_desc);
}

void rec_ranking_dnn_init(RecRankingDNN *dnn, const int32_t *dims, int32_t num_layers,
                          uint64_t seed) {
    dnn->num_layers = num_layers < 4 ? num_layers : 4;
    for (int l = 0; l < dnn->num_layers; l++) {
        dnn->hidden_dims[l] = dims[l];
        int in_dim = (l == 0) ? REC_MAX_FEATURES : dims[l - 1];
        int out_dim = dims[l];
        for (int i = 0; i < out_dim && i < REC_MAX_FEATURES; i++) {
            for (int j = 0; j < in_dim && j < REC_MAX_FEATURES; j++) {
                dnn->layers[l].w[i * REC_MAX_FEATURES + j] =
                    (rec_randf(&seed) * 2.0f - 1.0f) * sqrtf(2.0f / (float)in_dim);
            }
        }
        dnn->layers[l].bias = 0.0f;
    }
}

static float rec_relu(float x) { return x > 0.0f ? x : 0.0f; }

float rec_ranking_dnn_forward(const RecRankingDNN *dnn, const RecFeatureVec *feat) {
    float hidden[REC_MAX_FEATURES];
    float next[REC_MAX_FEATURES];
    int in_dim = feat->num_features < REC_MAX_FEATURES ? feat->num_features : REC_MAX_FEATURES;
    memcpy(hidden, feat->features, (size_t)in_dim * sizeof(float));

    for (int l = 0; l < dnn->num_layers; l++) {
        int out_dim = dnn->hidden_dims[l];
        int prev_dim = (l == 0) ? in_dim : dnn->hidden_dims[l - 1];
        for (int o = 0; o < out_dim && o < REC_MAX_FEATURES - 1; o++) {
            float sum = dnn->layers[l].bias;
            for (int i = 0; i < prev_dim && i < REC_MAX_FEATURES; i++) {
                sum += hidden[i] * dnn->layers[l].w[o * REC_MAX_FEATURES + i];
            }
            next[o] = rec_relu(sum);
        }
        memcpy(hidden, next, (size_t)out_dim * sizeof(float));
    }
    float score = 0.0f;
    for (int i = 0; i < dnn->hidden_dims[dnn->num_layers - 1] && i < REC_MAX_FEATURES; i++) {
        score += hidden[i] * 0.1f;
    }
    return 1.0f / (1.0f + expf(-score));
}

void rec_rank_candidates(const RecRankingDNN *dnn, const RecFeatureVec *user_feat,
                         const RecFeatureVec *session_feat,
                         const RecItemTower *items, const RecCandidateList *cands,
                         RecCandidateList *ranked) {
    RecFeatureVec combined;
    combined.num_features = 0;
    for (int i = 0; i < user_feat->num_features && combined.num_features < REC_MAX_FEATURES; i++)
        combined.features[combined.num_features++] = user_feat->features[i];
    for (int i = 0; i < session_feat->num_features && combined.num_features < REC_MAX_FEATURES; i++)
        combined.features[combined.num_features++] = session_feat->features[i];
    int n = cands->count;
    ranked->count = n < REC_MAX_CANDIDATES ? n : REC_MAX_CANDIDATES;
    for (int i = 0; i < ranked->count; i++) {
        ranked->items[i] = cands->items[i];
        RecFeatureVec item_feat;
        item_feat.num_features = REC_EMBED_DIM < REC_MAX_FEATURES ? REC_EMBED_DIM : REC_MAX_FEATURES;
        for (int d = 0; d < item_feat.num_features; d++)
            item_feat.features[d] = items[cands->items[i].item_id % REC_MAX_ITEMS].emb.data[d];
        RecFeatureVec all_feat;
        all_feat.num_features = 0;
        for (int j = 0; j < combined.num_features && all_feat.num_features < REC_MAX_FEATURES; j++)
            all_feat.features[all_feat.num_features++] = combined.features[j];
        for (int j = 0; j < item_feat.num_features && all_feat.num_features < REC_MAX_FEATURES; j++)
            all_feat.features[all_feat.num_features++] = item_feat.features[j];
        ranked->items[i].score = rec_ranking_dnn_forward(dnn, &all_feat);
    }
    qsort(ranked->items, ranked->count, sizeof(RecCandidate), rec_cmp_candidate_desc);
}

void rec_diversity_rerank(RecCandidateList *ranked, const RecItemTower *items,
                          float lambda, int32_t k) {
    (void)lambda;
    if (ranked->count <= k) return;
    RecCandidateList diverse;
    diverse.count = 0;
    int *used = (int *)calloc((size_t)ranked->count, sizeof(int));
    if (!used) return;
    for (int i = 0; i < ranked->count && diverse.count < k; i++) {
        if (used[i]) continue;
        diverse.items[diverse.count++] = ranked->items[i];
        used[i] = 1;
        for (int j = i + 1; j < ranked->count && diverse.count < k; j++) {
            if (used[j]) continue;
            int too_similar = 0;
            for (int d = 0; d < diverse.count; d++) {
                float dp = rec_dot_product(&items[diverse.items[d].item_id % REC_MAX_ITEMS].emb,
                                           &items[ranked->items[j].item_id % REC_MAX_ITEMS].emb);
                if (dp > 0.95f) { too_similar = 1; break; }
            }
            if (!too_similar) {
                diverse.items[diverse.count++] = ranked->items[j];
                used[j] = 1;
            }
        }
    }
    ranked->count = diverse.count;
    memcpy(ranked->items, diverse.items, (size_t)diverse.count * sizeof(RecCandidate));
    free(used);
}

void rec_freshness_boost(RecCandidateList *ranked, const int64_t *timestamps,
                         int64_t now, float alpha) {
    for (int i = 0; i < ranked->count; i++) {
        int64_t age = now - timestamps[i];
        float freshness = expf(-alpha * (float)age / 86400.0f);
        ranked->items[i].score *= (1.0f + freshness);
    }
    qsort(ranked->items, ranked->count, sizeof(RecCandidate), rec_cmp_candidate_desc);
}

void rec_business_rule_filter(RecCandidateList *ranked,
                              const int32_t *blocked_ids, int32_t n_blocked) {
    RecCandidateList filtered;
    filtered.count = 0;
    for (int i = 0; i < ranked->count; i++) {
        int blocked = 0;
        for (int j = 0; j < n_blocked; j++) {
            if (ranked->items[i].item_id == blocked_ids[j]) { blocked = 1; break; }
        }
        if (!blocked) {
            filtered.items[filtered.count++] = ranked->items[i];
        }
    }
    ranked->count = filtered.count;
    memcpy(ranked->items, filtered.items, (size_t)filtered.count * sizeof(RecCandidate));
}

void rec_content_features(const char *title, const char *category,
                          const char **tags, int32_t n_tags, RecFeatureVec *out) {
    out->num_features = 0;
    if (title) {
        size_t len = strlen(title);
        out->features[out->num_features++] = (float)len / 100.0f;
    }
    if (category) {
        unsigned long h = 5381;
        for (const char *p = category; *p; p++) h = ((h << 5) + h) + (unsigned char)*p;
        out->features[out->num_features++] = (float)(h % 1000) / 1000.0f;
    }
    if (tags) {
        for (int i = 0; i < n_tags && out->num_features + 1 < REC_MAX_FEATURES; i++) {
            unsigned long h = 5381;
            for (const char *p = tags[i]; *p; p++) h = ((h << 5) + h) + (unsigned char)*p;
            out->features[out->num_features++] = (float)(h % 1000) / 1000.0f;
        }
    }
    while (out->num_features < REC_MAX_FEATURES) {
        out->features[out->num_features++] = 0.0f;
    }
}

void rec_cold_start_score(const RecFeatureVec *new_item,
                          const RecFeatureVec *user_profile, float *score) {
    float dp = 0.0f;
    int n = new_item->num_features < user_profile->num_features ?
            new_item->num_features : user_profile->num_features;
    for (int i = 0; i < n; i++) {
        dp += new_item->features[i] * user_profile->features[i];
    }
    *score = 1.0f / (1.0f + expf(-dp));
}

void rec_ab_config_init(RecABConfig *cfg, int32_t control_pct, int32_t treatment_pct,
                        uint64_t seed) {
    cfg->control_pct = control_pct;
    cfg->treatment_pct = treatment_pct;
    cfg->seed = seed;
}

int rec_ab_decide(const RecABConfig *cfg, int32_t user_id) {
    uint64_t h = (uint64_t)(user_id * 2654435761U) ^ cfg->seed;
    h = (h ^ (h >> 16)) * 0x85EBCA6BULL;
    h = (h ^ (h >> 13)) * 0xC2B2AE35ULL;
    h = h ^ (h >> 16);
    float bucket = (float)(h % 10000) / 100.0f;
    if (bucket < (float)cfg->treatment_pct) return 1;
    return 0;
}

static double rec_mean(const double *x, int32_t n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += x[i];
    return s / (double)n;
}

static double rec_stdev(const double *x, int32_t n, double mean) {
    double s = 0.0;
    for (int i = 0; i < n; i++) { double d = x[i] - mean; s += d * d; }
    return sqrt(s / (double)(n - 1));
}

void rec_ab_evaluate(double *ctrl_metrics, int32_t n_ctrl,
                     double *treat_metrics, int32_t n_treat,
                     RecABResult *result) {
    double m1 = rec_mean(ctrl_metrics, n_ctrl);
    double m2 = rec_mean(treat_metrics, n_treat);
    double s1 = rec_stdev(ctrl_metrics, n_ctrl, m1);
    double s2 = rec_stdev(treat_metrics, n_treat, m2);
    double se = sqrt(s1 * s1 / n_ctrl + s2 * s2 / n_treat);
    double t = (se > 1e-10) ? (m2 - m1) / se : 0.0;
    result->ctr_control = m1;
    result->ctr_treatment = m2;
    result->p_value = 2.0 * (1.0 - 0.5 * (1.0 + erf(fabs(t) / sqrt(2.0))));
    result->significant = (result->p_value < 0.05) ? 1 : 0;
}
