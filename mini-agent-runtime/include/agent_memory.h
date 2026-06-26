#ifndef AGENT_MEMORY_H
#define AGENT_MEMORY_H

#include <stddef.h>
#include <stdbool.h>
#include <time.h>

#define MEMORY_MAX_WORKING_ITEMS    256
#define MEMORY_MAX_SHORT_TERM_ITEMS 1024
#define MEMORY_MAX_VECTOR_ITEMS     10000
#define MEMORY_MAX_EPISODIC_ITEMS   512
#define MEMORY_MAX_SUMMARY_ITEMS    64
#define MEMORY_MAX_REFLECTION_ITEMS 128
#define MEMORY_MAX_CONTENT_LEN      8192
#define MEMORY_MAX_KEY_LEN          256
#define MEMORY_VEC_DIM              384
#define MEMORY_MAX_DECAY_RATE       0.99f

typedef enum {
    MEMORY_TYPE_WORKING,
    MEMORY_TYPE_SHORT_TERM,
    MEMORY_TYPE_LONG_TERM,
    MEMORY_TYPE_EPISODIC,
    MEMORY_TYPE_SUMMARY,
    MEMORY_TYPE_REFLECTION
} memory_type_t;

typedef enum {
    MEMORY_ROLE_USER,
    MEMORY_ROLE_ASSISTANT,
    MEMORY_ROLE_SYSTEM,
    MEMORY_ROLE_TOOL
} memory_role_t;

typedef struct {
    memory_role_t role;
    char content[MEMORY_MAX_CONTENT_LEN];
    time_t timestamp;
    float importance;
    float recency;
    int token_count;
} memory_item_t;

typedef struct {
    memory_item_t items[MEMORY_MAX_WORKING_ITEMS];
    int count;
    int capacity;
    float decay_rate;
} working_memory_t;

typedef struct {
    memory_item_t items[MEMORY_MAX_SHORT_TERM_ITEMS];
    int count;
    int capacity;
    int max_tokens;
    int current_tokens;
    float ttl_seconds;
} short_term_memory_t;

typedef struct {
    char content[MEMORY_MAX_CONTENT_LEN];
    float vector[MEMORY_VEC_DIM];
    char key[MEMORY_MAX_KEY_LEN];
    time_t timestamp;
    float importance;
    int access_count;
} vector_memory_entry_t;

typedef struct {
    vector_memory_entry_t entries[MEMORY_MAX_VECTOR_ITEMS];
    int count;
    int capacity;
} long_term_memory_t;

typedef struct {
    int original_episode_id;
    memory_item_t *episode_items;
    int episode_length;
    char summary[MEMORY_MAX_CONTENT_LEN];
    char key_points[4096];
    time_t created_at;
    int importance_rank;
} summary_t;

typedef struct {
    summary_t summaries[MEMORY_MAX_SUMMARY_ITEMS];
    int count;
} summary_memory_t;

typedef struct {
    char description[MEMORY_MAX_CONTENT_LEN];
    memory_item_t *episode_items;
    int episode_length;
    memory_item_t events[32];
    int event_count;
    time_t start_time;
    time_t end_time;
    char outcome[MEMORY_MAX_CONTENT_LEN];
    float success_rating;
    int episode_id;
} episode_t;

typedef struct {
    episode_t episodes[MEMORY_MAX_EPISODIC_ITEMS];
    int count;
} episodic_memory_t;

typedef struct {
    char insight[MEMORY_MAX_CONTENT_LEN];
    char evidence[4096];
    int source_episodes[8];
    int source_count;
    time_t created_at;
    float confidence;
} reflection_item_t;

typedef struct {
    reflection_item_t items[MEMORY_MAX_REFLECTION_ITEMS];
    int count;
} reflection_memory_t;

typedef struct {
    float relevance;
    float recency;
    float importance;
    float composite;
} memory_score_t;

typedef struct {
    char key[MEMORY_MAX_KEY_LEN];
    int index;
    memory_score_t score;
} memory_retrieval_t;

typedef struct {
    working_memory_t working;
    short_term_memory_t short_term;
    long_term_memory_t long_term;
    summary_memory_t summary;
    episodic_memory_t episodic;
    reflection_memory_t reflection;
    int total_tokens;
} agent_memory_t;

agent_memory_t* agent_memory_create(void);
void agent_memory_destroy(agent_memory_t *mem);
void agent_memory_clear(agent_memory_t *mem, memory_type_t type);

bool working_memory_add(working_memory_t *wm, memory_role_t role, const char *content);
bool working_memory_evict(working_memory_t *wm, int index);
void working_memory_decay(working_memory_t *wm);
const memory_item_t* working_memory_get_recent(const working_memory_t *wm, int n, int *out_count);
const char* working_memory_format_for_llm(const working_memory_t *wm, char *buf, size_t buf_size);

bool short_term_memory_add(short_term_memory_t *stm, memory_role_t role, const char *content);
bool short_term_memory_is_full(const short_term_memory_t *stm);
const char* short_term_memory_compress(short_term_memory_t *stm, agent_memory_t *mem, char *buf, size_t buf_size);
void short_term_memory_manage_ttl(short_term_memory_t *stm);

bool long_term_memory_store(long_term_memory_t *ltm, const char *content, const char *key);
bool long_term_memory_store_vector(long_term_memory_t *ltm, const char *content, const float *vector, const char *key);
const vector_memory_entry_t* long_term_memory_retrieve(long_term_memory_t *ltm, const char *key);
int long_term_memory_search_similar(const long_term_memory_t *ltm, const float *query_vec, memory_retrieval_t *results, int max_results);
float long_term_memory_cosine_similarity(const float *a, const float *b, int dim);
void long_term_memory_embed(const char *text, float *vec, int dim);

bool summary_memory_create_summary(summary_memory_t *sm, const memory_item_t *items, int count);
bool summary_memory_merge_summaries(summary_memory_t *sm, int idx1, int idx2);
const char* summary_memory_get_context(const summary_memory_t *sm, char *buf, size_t buf_size);

bool episodic_memory_store(episodic_memory_t *em, const episode_t *ep);
episode_t* episodic_memory_begin_episode(episodic_memory_t *em);
bool episodic_memory_end_episode(episodic_memory_t *em, int episode_id, const char *outcome, float success);
bool episodic_memory_add_event(episodic_memory_t *em, int episode_id, memory_role_t role, const char *content);
const episode_t* episodic_memory_find_similar(const episodic_memory_t *em, const char *situation, int *count, episode_t *out, int max_out);

bool reflection_memory_extract(reflection_memory_t *rm, const agent_memory_t *mem, const char *prompt_context);
const char* reflection_memory_get_insights(const reflection_memory_t *rm, char *buf, size_t buf_size);
reflection_item_t reflection_memory_generate(agent_memory_t *mem, const memory_item_t *items, int count);

memory_score_t memory_compute_score(const memory_item_t *item, time_t now);
memory_retrieval_t* memory_retrieve(agent_memory_t *mem, const char *query, int limit, int *out_count);
void memory_retrieval_sort(memory_retrieval_t *results, int count);
const char* memory_retrieval_format(const memory_retrieval_t *results, int count, char *buf, size_t buf_size);

int memory_estimate_tokens(const char *text);
void memory_manage_capacity(agent_memory_t *mem);

#endif
