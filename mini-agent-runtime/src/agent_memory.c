#include "agent_memory.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

agent_memory_t* agent_memory_create(void) {
    agent_memory_t *mem = (agent_memory_t*)calloc(1, sizeof(agent_memory_t));
    if (!mem) return NULL;
    mem->working.capacity = MEMORY_MAX_WORKING_ITEMS;
    mem->working.decay_rate = 0.98f;
    mem->short_term.capacity = MEMORY_MAX_SHORT_TERM_ITEMS;
    mem->short_term.max_tokens = 8192;
    mem->short_term.ttl_seconds = 3600.0f;
    mem->long_term.capacity = MEMORY_MAX_VECTOR_ITEMS;
    return mem;
}

void agent_memory_destroy(agent_memory_t *mem) {
    if (!mem) return;
    for (int i = 0; i < mem->episodic.count; i++) {
        if (mem->episodic.episodes[i].episode_items) {
            free(mem->episodic.episodes[i].episode_items);
        }
    }
    free(mem);
}

void agent_memory_clear(agent_memory_t *mem, memory_type_t type) {
    if (!mem) return;
    switch (type) {
        case MEMORY_TYPE_WORKING:
            memset(&mem->working, 0, sizeof(working_memory_t));
            mem->working.capacity = MEMORY_MAX_WORKING_ITEMS;
            mem->working.decay_rate = 0.98f;
            break;
        case MEMORY_TYPE_SHORT_TERM:
            memset(&mem->short_term, 0, sizeof(short_term_memory_t));
            mem->short_term.capacity = MEMORY_MAX_SHORT_TERM_ITEMS;
            break;
        case MEMORY_TYPE_LONG_TERM:
            memset(&mem->long_term, 0, sizeof(long_term_memory_t));
            break;
        case MEMORY_TYPE_EPISODIC:
            for (int i = 0; i < mem->episodic.count; i++)
                free(mem->episodic.episodes[i].episode_items);
            memset(&mem->episodic, 0, sizeof(episodic_memory_t));
            break;
        case MEMORY_TYPE_SUMMARY:
            memset(&mem->summary, 0, sizeof(summary_memory_t));
            break;
        case MEMORY_TYPE_REFLECTION:
            memset(&mem->reflection, 0, sizeof(reflection_memory_t));
            break;
    }
}

bool working_memory_add(working_memory_t *wm, memory_role_t role, const char *content) {
    if (!wm || !content || wm->count >= wm->capacity) return false;
    memory_item_t *item = &wm->items[wm->count];
    item->role = role;
    strncpy(item->content, content, MEMORY_MAX_CONTENT_LEN - 1);
    item->timestamp = time(NULL);
    item->importance = 0.5f;
    item->recency = 1.0f;
    item->token_count = memory_estimate_tokens(content);
    wm->count++;
    return true;
}

bool working_memory_evict(working_memory_t *wm, int index) {
    if (!wm || index < 0 || index >= wm->count) return false;
    for (int i = index; i < wm->count - 1; i++) {
        wm->items[i] = wm->items[i + 1];
    }
    wm->count--;
    return true;
}

void working_memory_decay(working_memory_t *wm) {
    if (!wm) return;
    for (int i = 0; i < wm->count; i++) {
        wm->items[i].recency *= wm->decay_rate;
    }
}

const memory_item_t* working_memory_get_recent(const working_memory_t *wm, int n, int *out_count) {
    if (!wm || !out_count) return NULL;
    int count = n < wm->count ? n : wm->count;
    *out_count = count;
    if (count == 0) return NULL;
    return &wm->items[wm->count - count];
}

const char* working_memory_format_for_llm(const working_memory_t *wm, char *buf, size_t buf_size) {
    if (!wm || !buf) return NULL;
    int offset = 0;
    for (int i = 0; i < wm->count; i++) {
        const memory_item_t *item = &wm->items[i];
        const char *role_str = "User";
        switch (item->role) {
            case MEMORY_ROLE_ASSISTANT: role_str = "Assistant"; break;
            case MEMORY_ROLE_SYSTEM:    role_str = "System"; break;
            case MEMORY_ROLE_TOOL:      role_str = "Tool"; break;
            default: role_str = "User"; break;
        }
        offset += snprintf(buf + offset, buf_size - offset,
                          "%s: %s\n", role_str, item->content);
    }
    return buf;
}

bool short_term_memory_add(short_term_memory_t *stm, memory_role_t role, const char *content) {
    if (!stm || !content || stm->count >= stm->capacity) return false;
    memory_item_t *item = &stm->items[stm->count];
    item->role = role;
    strncpy(item->content, content, MEMORY_MAX_CONTENT_LEN - 1);
    item->timestamp = time(NULL);
    item->importance = 0.5f;
    item->recency = 1.0f;
    item->token_count = memory_estimate_tokens(content);
    stm->current_tokens += item->token_count;
    stm->count++;
    return true;
}

bool short_term_memory_is_full(const short_term_memory_t *stm) {
    return stm && (stm->count >= stm->capacity || stm->current_tokens >= stm->max_tokens);
}

const char* short_term_memory_compress(short_term_memory_t *stm, agent_memory_t *mem, char *buf, size_t buf_size) {
    if (!stm || !mem || !buf) return NULL;
    summary_memory_create_summary(&mem->summary, stm->items, stm->count);
    int offset = snprintf(buf, buf_size, "Summary of previous conversation:\n");
    for (int i = 0; i < mem->summary.count && i < 3; i++) {
        offset += snprintf(buf + offset, buf_size - offset,
                          "- %s\n", mem->summary.summaries[i].summary);
    }
    memset(stm, 0, sizeof(short_term_memory_t));
    stm->capacity = MEMORY_MAX_SHORT_TERM_ITEMS;
    stm->max_tokens = 8192;
    return buf;
}

void short_term_memory_manage_ttl(short_term_memory_t *stm) {
    if (!stm) return;
    time_t now = time(NULL);
    int write_idx = 0;
    for (int i = 0; i < stm->count; i++) {
        double age = difftime(now, stm->items[i].timestamp);
        if (age < stm->ttl_seconds) {
            if (write_idx != i) stm->items[write_idx] = stm->items[i];
            write_idx++;
        }
    }
    stm->count = write_idx;
}

bool long_term_memory_store(long_term_memory_t *ltm, const char *content, const char *key) {
    if (!ltm || !content || ltm->count >= ltm->capacity) return false;
    vector_memory_entry_t *entry = &ltm->entries[ltm->count];
    strncpy(entry->content, content, MEMORY_MAX_CONTENT_LEN - 1);
    if (key) strncpy(entry->key, key, MEMORY_MAX_KEY_LEN - 1);
    entry->timestamp = time(NULL);
    entry->importance = 0.5f;
    entry->access_count = 0;
    long_term_memory_embed(content, entry->vector, MEMORY_VEC_DIM);
    ltm->count++;
    return true;
}

bool long_term_memory_store_vector(long_term_memory_t *ltm, const char *content, const float *vector, const char *key) {
    if (!ltm || !content || ltm->count >= ltm->capacity) return false;
    vector_memory_entry_t *entry = &ltm->entries[ltm->count];
    strncpy(entry->content, content, MEMORY_MAX_CONTENT_LEN - 1);
    if (key) strncpy(entry->key, key, MEMORY_MAX_KEY_LEN - 1);
    entry->timestamp = time(NULL);
    entry->importance = 0.5f;
    if (vector) memcpy(entry->vector, vector, MEMORY_VEC_DIM * sizeof(float));
    ltm->count++;
    return true;
}

const vector_memory_entry_t* long_term_memory_retrieve(long_term_memory_t *ltm, const char *key) {
    if (!ltm || !key) return NULL;
    for (int i = 0; i < ltm->count; i++) {
        if (strcmp(ltm->entries[i].key, key) == 0) {
            ltm->entries[i].access_count++;
            return &ltm->entries[i];
        }
    }
    return NULL;
}

int long_term_memory_search_similar(const long_term_memory_t *ltm, const float *query_vec, memory_retrieval_t *results, int max_results) {
    if (!ltm || !query_vec || !results || ltm->count == 0) return 0;
    int found = 0;
    for (int i = 0; i < ltm->count && found < max_results; i++) {
        float sim = long_term_memory_cosine_similarity(query_vec, ltm->entries[i].vector, MEMORY_VEC_DIM);
        if (sim > 0.5f) {
            results[found].key[0] = '\0';
            strncpy(results[found].key, ltm->entries[i].key, MEMORY_MAX_KEY_LEN - 1);
            results[found].index = i;
            results[found].score.relevance = sim;
            found++;
        }
    }
    return found;
}

float long_term_memory_cosine_similarity(const float *a, const float *b, int dim) {
    float dot = 0.0f, mag_a = 0.0f, mag_b = 0.0f;
    for (int i = 0; i < dim; i++) {
        dot += a[i] * b[i];
        mag_a += a[i] * a[i];
        mag_b += b[i] * b[i];
    }
    if (mag_a < 1e-9f || mag_b < 1e-9f) return 0.0f;
    return dot / (sqrtf(mag_a) * sqrtf(mag_b));
}

void long_term_memory_embed(const char *text, float *vec, int dim) {
    if (!text || !vec) return;
    unsigned int hash = 5381;
    int c;
    while ((c = (unsigned char)*text++)) hash = ((hash << 5) + hash) + c;
    for (int i = 0; i < dim; i++) {
        hash = ((hash << 5) + hash) + i + 1;
        vec[i] = ((float)(hash % 10000) / 5000.0f) - 1.0f;
    }
    float mag = 0.0f;
    for (int i = 0; i < dim; i++) mag += vec[i] * vec[i];
    if (mag > 1e-9f) {
        mag = sqrtf(mag);
        for (int i = 0; i < dim; i++) vec[i] /= mag;
    }
}

bool summary_memory_create_summary(summary_memory_t *sm, const memory_item_t *items, int count) {
    if (!sm || !items || sm->count >= MEMORY_MAX_SUMMARY_ITEMS) return false;
    summary_t *s = &sm->summaries[sm->count];
    int key_point_count = 0;
    char orig_content[MEMORY_MAX_CONTENT_LEN * 2] = {0};
    for (int i = 0; i < count && i < 5; i++) {
        strncat(orig_content, items[i].content, MEMORY_MAX_CONTENT_LEN);
        strncat(orig_content, " ", 1);
    }
    snprintf(s->summary, MEMORY_MAX_CONTENT_LEN,
             "[Summary of %d messages] %s...", count,
             orig_content[0] ? orig_content : "empty");
    s->created_at = time(NULL);
    s->importance_rank = count;
    s->episode_length = count;
    sm->count++;
    (void)key_point_count;
    return true;
}

bool summary_memory_merge_summaries(summary_memory_t *sm, int idx1, int idx2) {
    if (!sm || idx1 >= sm->count || idx2 >= sm->count) return false;
    summary_t *a = &sm->summaries[idx1];
    summary_t *b = &sm->summaries[idx2];
    snprintf(a->summary, MEMORY_MAX_CONTENT_LEN, "%s + %s", a->summary, b->summary);
    a->importance_rank += b->importance_rank;
    memmove(&sm->summaries[idx2], &sm->summaries[idx2 + 1],
            (sm->count - idx2 - 1) * sizeof(summary_t));
    sm->count--;
    return true;
}

const char* summary_memory_get_context(const summary_memory_t *sm, char *buf, size_t buf_size) {
    if (!sm || !buf) return NULL;
    int offset = 0;
    for (int i = 0; i < sm->count; i++) {
        offset += snprintf(buf + offset, buf_size - offset,
                          "[Summary %d] %s\n", i, sm->summaries[i].summary);
    }
    return buf;
}

bool episodic_memory_store(episodic_memory_t *em, const episode_t *ep) {
    if (!em || !ep || em->count >= MEMORY_MAX_EPISODIC_ITEMS) return false;
    em->episodes[em->count] = *ep;
    em->count++;
    return true;
}

episode_t* episodic_memory_begin_episode(episodic_memory_t *em) {
    if (!em || em->count >= MEMORY_MAX_EPISODIC_ITEMS) return NULL;
    episode_t *ep = &em->episodes[em->count];
    memset(ep, 0, sizeof(episode_t));
    ep->episode_id = em->count;
    ep->start_time = time(NULL);
    ep->success_rating = 0.0f;
    return ep;
}

bool episodic_memory_end_episode(episodic_memory_t *em, int episode_id, const char *outcome, float success) {
    if (!em || episode_id >= em->count || episode_id < 0) return false;
    episode_t *ep = &em->episodes[episode_id];
    ep->end_time = time(NULL);
    if (outcome) strncpy(ep->outcome, outcome, MEMORY_MAX_CONTENT_LEN - 1);
    ep->success_rating = success;
    em->count++;
    return true;
}

bool episodic_memory_add_event(episodic_memory_t *em, int episode_id, memory_role_t role, const char *content) {
    if (!em || !content || episode_id < 0 || episode_id >= MEMORY_MAX_EPISODIC_ITEMS) return false;
    episode_t *ep = &em->episodes[episode_id];
    if (ep->event_count >= 32) return false;
    memory_item_t *ev = &ep->events[ep->event_count];
    ev->role = role;
    strncpy(ev->content, content, MEMORY_MAX_CONTENT_LEN - 1);
    ev->timestamp = time(NULL);
    ep->event_count++;
    return true;
}

const episode_t* episodic_memory_find_similar(const episodic_memory_t *em, const char *situation, int *count, episode_t *out, int max_out) {
    if (!em || !count || !out) return NULL;
    *count = 0;
    for (int i = 0; i < em->count && *count < max_out; i++) {
        if (situation && strstr(em->episodes[i].outcome, situation)) {
            out[*count] = em->episodes[i];
            (*count)++;
        } else if (!situation && *count < max_out) {
            out[*count] = em->episodes[i];
            (*count)++;
        }
    }
    return *count > 0 ? out : NULL;
}

bool reflection_memory_extract(reflection_memory_t *rm, const agent_memory_t *mem, const char *prompt_context) {
    if (!rm || !mem || rm->count >= MEMORY_MAX_REFLECTION_ITEMS) return false;
    reflection_item_t *item = &rm->items[rm->count];
    item->confidence = 0.5f;
    item->created_at = time(NULL);
    snprintf(item->insight, MEMORY_MAX_CONTENT_LEN,
             "Reflection on session context: %s [%d working items, %d short-term items]",
             prompt_context ? prompt_context : "general",
             mem->working.count, mem->short_term.count);
    rm->count++;
    return true;
}

const char* reflection_memory_get_insights(const reflection_memory_t *rm, char *buf, size_t buf_size) {
    if (!rm || !buf) return NULL;
    int offset = 0;
    for (int i = 0; i < rm->count; i++) {
        offset += snprintf(buf + offset, buf_size - offset,
                          "[Insight %d, conf=%.2f] %s\n", i,
                          rm->items[i].confidence, rm->items[i].insight);
    }
    return buf;
}

reflection_item_t reflection_memory_generate(agent_memory_t *mem, const memory_item_t *items, int count) {
    reflection_item_t ri;
    memset(&ri, 0, sizeof(ri));
    if (!mem || !items) return ri;
    ri.confidence = 0.6f;
    ri.created_at = time(NULL);
    snprintf(ri.insight, MEMORY_MAX_CONTENT_LEN,
             "Generated insight from %d conversation items", count);
    snprintf(ri.evidence, sizeof(ri.evidence),
             "Evidence: %d items analyzed, %d working memory, %d short-term memory",
             count, mem->working.count, mem->short_term.count);
    ri.source_count = count < 8 ? count : 8;
    return ri;
}

memory_score_t memory_compute_score(const memory_item_t *item, time_t now) {
    memory_score_t score = {0.0f, 0.0f, 0.0f, 0.0f};
    if (!item) return score;
    double age_seconds = difftime(now, item->timestamp);
    double age_hours = age_seconds / 3600.0;
    score.recency = age_hours > 0 ? (float)(1.0 / (1.0 + log(1.0 + age_hours))) : 1.0f;
    score.importance = item->importance;
    score.relevance = 0.5f;
    score.composite = 0.4f * score.recency + 0.3f * score.importance + 0.3f * score.relevance;
    return score;
}

memory_retrieval_t* memory_retrieve(agent_memory_t *mem, const char *query, int limit, int *out_count) {
    if (!mem || !out_count) return NULL;
    *out_count = 0;
    memory_retrieval_t *results = (memory_retrieval_t*)malloc(limit * sizeof(memory_retrieval_t));
    if (!results) return NULL;
    time_t now = time(NULL);
    int found = 0;
    for (int i = 0; i < mem->short_term.count && found < limit; i++) {
        memory_score_t sc = memory_compute_score(&mem->short_term.items[i], now);
        if (sc.composite > 0.1f) {
            results[found].index = i;
            results[found].score = sc;
            snprintf(results[found].key, MEMORY_MAX_KEY_LEN, "stm_%d", i);
            found++;
        }
    }
    *out_count = found;
    return results;
}

void memory_retrieval_sort(memory_retrieval_t *results, int count) {
    if (!results || count <= 1) return;
    for (int i = 0; i < count - 1; i++) {
        for (int j = 0; j < count - i - 1; j++) {
            if (results[j].score.composite < results[j + 1].score.composite) {
                memory_retrieval_t tmp = results[j];
                results[j] = results[j + 1];
                results[j + 1] = tmp;
            }
        }
    }
}

const char* memory_retrieval_format(const memory_retrieval_t *results, int count, char *buf, size_t buf_size) {
    if (!results || !buf) return NULL;
    int offset = 0;
    for (int i = 0; i < count; i++) {
        offset += snprintf(buf + offset, buf_size - offset,
                          "[%d] score=%.3f (key=%s)\n", i,
                          results[i].score.composite, results[i].key);
    }
    return buf;
}

int memory_estimate_tokens(const char *text) {
    if (!text) return 0;
    int tokens = 0;
    bool in_word = false;
    for (int i = 0; text[i]; i++) {
        if ((text[i] == ' ' || text[i] == '\n' || text[i] == '\t')) {
            if (in_word) { tokens++; in_word = false; }
        } else {
            in_word = true;
        }
    }
    if (in_word) tokens++;
    return tokens + 2;
}

void memory_manage_capacity(agent_memory_t *mem) {
    if (!mem) return;
    if (mem->short_term.count > mem->short_term.capacity * 3 / 4) {
        char buf[MEMORY_MAX_CONTENT_LEN];
        short_term_memory_compress(&mem->short_term, mem, buf, sizeof(buf));
    }
    if (mem->working.count > mem->working.capacity * 3 / 4) {
        working_memory_decay(&mem->working);
    }
}
