#include "agent_memory.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void print_separator(const char *title) {
    printf("\n========================================\n");
    printf("  %s\n", title);
    printf("========================================\n");
}

static void print_memory_scores(const memory_score_t *score) {
    printf("  relevance=%.4f  recency=%.4f  importance=%.4f  composite=%.4f\n",
           score->relevance, score->recency, score->importance, score->composite);
}

int main(void) {
    printf("========== Agent Memory System Demo ==========\n\n");
    printf("Demonstrating all memory subsystems:\n");
    printf("  1. Working Memory\n");
    printf("  2. Short-Term Memory\n");
    printf("  3. Long-Term Memory (Vector Store)\n");
    printf("  4. Summary/Compression Memory\n");
    printf("  5. Episodic Memory\n");
    printf("  6. Reflection Memory\n");
    printf("  7. Memory Scoring and Retrieval\n");

    agent_memory_t *mem = agent_memory_create();
    if (!mem) { printf("Failed to create memory\n"); return 1; }
    printf("\nMemory system initialized.\n");

    print_separator("1. Working Memory");

    working_memory_add(&mem->working, MEMORY_ROLE_USER, "Hello, I need help with data analysis.");
    working_memory_add(&mem->working, MEMORY_ROLE_ASSISTANT, "I can help with that. What kind of data?");
    working_memory_add(&mem->working, MEMORY_ROLE_USER, "It's a CSV file with sales data from Q3.");
    working_memory_add(&mem->working, MEMORY_ROLE_ASSISTANT, "I'll analyze it. One moment please.");
    working_memory_add(&mem->working, MEMORY_ROLE_TOOL, "[calculator] Result: 15000 total units sold");
    working_memory_add(&mem->working, MEMORY_ROLE_ASSISTANT, "Total units sold in Q3: 15,000. Want details?");
    working_memory_add(&mem->working, MEMORY_ROLE_USER, "Yes, break it down by region.");
    working_memory_add(&mem->working, MEMORY_ROLE_ASSISTANT, "North: 5,000 | South: 4,000 | East: 3,500 | West: 2,500");

    printf("Working memory items: %d\n", mem->working.count);

    int recent_count = 0;
    const memory_item_t *recent = working_memory_get_recent(&mem->working, 3, &recent_count);
    printf("Last %d recent items:\n", recent_count);
    for (int i = 0; i < recent_count; i++) {
        printf("  [%.15s...] %.60s...\n",
               recent[i].role == MEMORY_ROLE_USER ? "User" :
               recent[i].role == MEMORY_ROLE_ASSISTANT ? "Assistant" : "Tool",
               recent[i].content);
    }

    printf("\nApplying memory decay (rate=%.2f)...\n", mem->working.decay_rate);
    working_memory_decay(&mem->working);

    memory_score_t score0 = memory_compute_score(&mem->working.items[0], time(NULL));
    memory_score_t score7 = memory_compute_score(&mem->working.items[7], time(NULL));
    printf("First item score: ");
    print_memory_scores(&score0);
    printf("Last item score:  ");
    print_memory_scores(&score7);

    print_separator("2. Short-Term Memory");

    short_term_memory_add(&mem->short_term, MEMORY_ROLE_USER, "Let's discuss the marketing strategy.");
    short_term_memory_add(&mem->short_term, MEMORY_ROLE_ASSISTANT, "Current strategy: digital ads + email campaigns.");
    short_term_memory_add(&mem->short_term, MEMORY_ROLE_USER, "What's the conversion rate for Q3?");
    short_term_memory_add(&mem->short_term, MEMORY_ROLE_ASSISTANT, "Q3 conversion rate: 3.2%, up from 2.8% in Q2.");
    short_term_memory_add(&mem->short_term, MEMORY_ROLE_USER, "Great. What channels perform best?");
    short_term_memory_add(&mem->short_term, MEMORY_ROLE_ASSISTANT, "Email: 4.1% | Social: 2.5% | Search: 3.8% | Display: 1.9%");

    printf("Short-term items: %d\n", mem->short_term.count);
    printf("Current token count: %d / %d\n", mem->short_term.current_tokens, mem->short_term.max_tokens);
    printf("Is full: %s\n", short_term_memory_is_full(&mem->short_term) ? "yes" : "no");

    for (int i = 0; i < mem->short_term.count; i++) {
        memory_item_t *item = &mem->short_term.items[i];
        printf("  [%d] %.50s... (tokens: %d)\n", i, item->content, item->token_count);
    }

    short_term_memory_manage_ttl(&mem->short_term);
    printf("After TTL management: %d items remain\n", mem->short_term.count);

    print_separator("3. Long-Term Memory (Vector Store)");

    long_term_memory_store(&mem->long_term, "AI agents use ReAct loops for tool-based reasoning.", "react_concept");
    long_term_memory_store(&mem->long_term, "Vector embeddings enable semantic search over knowledge.", "embedding_concept");
    long_term_memory_store(&mem->long_term, "Planning systems decompose complex tasks into subtasks.", "planning_concept");
    long_term_memory_store(&mem->long_term, "Multi-agent systems enable collaborative problem solving.", "multiagent_concept");
    long_term_memory_store(&mem->long_term, "Memory systems help agents maintain context over time.", "memory_concept");
    long_term_memory_store(&mem->long_term, "Tool use extends LLM capabilities through function calling.", "tooluse_concept");
    long_term_memory_store(&mem->long_term, "Reflection enables agents to self-improve their outputs.", "reflection_concept");
    long_term_memory_store(&mem->long_term, "Working memory holds recent conversation turns.", "working_memory_concept");

    printf("Vector store entries: %d\n", mem->long_term.count);

    const vector_memory_entry_t *entry = long_term_memory_retrieve(&mem->long_term, "react_concept");
    if (entry) {
        printf("Retrieved by key 'react_concept': %.80s...\n", entry->content);
        printf("  Vector (first 5 dims): %.3f, %.3f, %.3f, %.3f, %.3f\n",
               entry->vector[0], entry->vector[1], entry->vector[2],
               entry->vector[3], entry->vector[4]);
    }

    entry = long_term_memory_retrieve(&mem->long_term, "multiagent_concept");
    if (entry) {
        printf("Retrieved by key 'multiagent_concept': %.80s...\n", entry->content);
        printf("  Access count: %d\n", entry->access_count);
    }

    float query_vec[MEMORY_VEC_DIM];
    const char *query_text = "agent planning and tool usage";
    long_term_memory_embed(query_text, query_vec, MEMORY_VEC_DIM);

    int max_retrieval = 5;
    memory_retrieval_t *sim_results = (memory_retrieval_t*)calloc(max_retrieval, sizeof(memory_retrieval_t));
    int sim_count = long_term_memory_search_similar(&mem->long_term, query_vec, sim_results, max_retrieval);
    printf("\nSemantic search for '%s': found %d similar entries\n", query_text, sim_count);
    for (int i = 0; i < sim_count; i++) {
        vector_memory_entry_t *e = &mem->long_term.entries[sim_results[i].index];
        printf("  [%.3f] %s: %.60s...\n",
               sim_results[i].score.relevance, e->key, e->content);
    }
    free(sim_results);

    printf("\nCosine similarity tests:\n");
    float v1[MEMORY_VEC_DIM], v2[MEMORY_VEC_DIM];
    long_term_memory_embed("hello world", v1, MEMORY_VEC_DIM);
    long_term_memory_embed("hello world", v2, MEMORY_VEC_DIM);
    printf("  Same text: %.4f\n", long_term_memory_cosine_similarity(v1, v2, MEMORY_VEC_DIM));

    long_term_memory_embed("hello world", v1, MEMORY_VEC_DIM);
    long_term_memory_embed("goodbye universe", v2, MEMORY_VEC_DIM);
    printf("  Different text: %.4f\n", long_term_memory_cosine_similarity(v1, v2, MEMORY_VEC_DIM));

    print_separator("4. Summary Memory (Compression)");

    printf("Before compression: %d short-term items\n", mem->short_term.count);
    char compress_buf[4096];
    short_term_memory_compress(&mem->short_term, mem, compress_buf, sizeof(compress_buf));
    printf("After compression: %d short-term items\n", mem->short_term.count);
    printf("Generated summaries: %d\n", mem->summary.count);
    printf("Compressed output: %.300s...\n", compress_buf);

    summary_memory_create_summary(&mem->summary,
        mem->working.items, mem->working.count);
    summary_memory_create_summary(&mem->summary,
        mem->short_term.items, mem->short_term.count);
    printf("Total summaries: %d\n", mem->summary.count);

    for (int i = 0; i < mem->summary.count; i++) {
        printf("  Summary %d (rank=%d): %.100s...\n", i,
               mem->summary.summaries[i].importance_rank,
               mem->summary.summaries[i].summary);
    }

    if (mem->summary.count >= 2) {
        summary_memory_merge_summaries(&mem->summary, 0, 1);
        printf("After merge: %d summaries\n", mem->summary.count);
    }

    char ctx_buf[4096];
    summary_memory_get_context(&mem->summary, ctx_buf, sizeof(ctx_buf));
    printf("Summary context:\n%s", ctx_buf);

    print_separator("5. Episodic Memory");

    episode_t *ep1 = episodic_memory_begin_episode(&mem->episodic);
    if (ep1) {
        episodic_memory_add_event(&mem->episodic, ep1->episode_id, MEMORY_ROLE_USER, "Task: analyze sales data");
        episodic_memory_add_event(&mem->episodic, ep1->episode_id, MEMORY_ROLE_ASSISTANT, "Loading Q3 sales CSV...");
        episodic_memory_add_event(&mem->episodic, ep1->episode_id, MEMORY_ROLE_TOOL, "CSV parsed: 1200 rows, 15 columns");
        episodic_memory_add_event(&mem->episodic, ep1->episode_id, MEMORY_ROLE_ASSISTANT, "Analysis complete. Key insights ready.");
        printf("Episode %d started with %d events\n", ep1->episode_id, ep1->event_count);
    }

    episode_t *ep2 = episodic_memory_begin_episode(&mem->episodic);
    if (ep2) {
        episodic_memory_add_event(&mem->episodic, ep2->episode_id, MEMORY_ROLE_USER, "Task: generate weather report");
        episodic_memory_add_event(&mem->episodic, ep2->episode_id, MEMORY_ROLE_ASSISTANT, "Fetching weather data...");
        episodic_memory_add_event(&mem->episodic, ep2->episode_id, MEMORY_ROLE_TOOL, "[weather] Beijing: 22C, Shanghai: 25C, Shenzhen: 28C");
        episodic_memory_add_event(&mem->episodic, ep2->episode_id, MEMORY_ROLE_ASSISTANT, "Weather report generated. Average temp: 25C");
        printf("Episode %d started with %d events\n", ep2->episode_id, ep2->event_count);
    }

    episodic_memory_end_episode(&mem->episodic, 0, "Successfully completed sales analysis", 0.95f);
    episodic_memory_end_episode(&mem->episodic, 1, "Weather report generated correctly", 0.90f);

    printf("Total episodes: %d\n", mem->episodic.count);

    int found_episodes = 0;
    episode_t similar_eps[4];
    episodic_memory_find_similar(&mem->episodic, "sales", &found_episodes, similar_eps, 4);
    printf("Episodes matching 'sales': %d\n", found_episodes);
    for (int i = 0; i < found_episodes; i++) {
        printf("  Episode %d: %.100s (success: %.2f)\n",
               similar_eps[i].episode_id, similar_eps[i].outcome, similar_eps[i].success_rating);
    }

    print_separator("6. Reflection Memory");

    reflection_memory_extract(&mem->reflection, mem, "data analysis assistant");
    reflection_memory_extract(&mem->reflection, mem, "weather reporting agent");
    reflection_memory_extract(&mem->reflection, mem, "general performance review");

    printf("Reflections extracted: %d\n", mem->reflection.count);

    reflection_item_t ri = reflection_memory_generate(mem, mem->working.items, mem->working.count);
    printf("Generated reflection:\n  %s\n", ri.insight);

    char insights_buf[4096];
    reflection_memory_get_insights(&mem->reflection, insights_buf, sizeof(insights_buf));
    printf("\nAll insights:\n%s", insights_buf);

    print_separator("7. Memory Scoring and Retrieval");

    memory_retrieval_t *results = memory_retrieve(mem, "sales analysis data", 10, &recent_count);
    if (results && recent_count > 0) {
        printf("Retrieved %d relevant items:\n", recent_count);
        memory_retrieval_sort(results, recent_count);

        char retrieval_buf[4096];
        memory_retrieval_format(results, recent_count, retrieval_buf, sizeof(retrieval_buf));
        printf("%s", retrieval_buf);

        free(results);
    }

    printf("\nMemory capacity management:\n");
    printf("  Before: working=%d/%d shortterm=%d/%d\n",
           mem->working.count, mem->working.capacity,
           mem->short_term.count, mem->short_term.capacity);

    for (int i = 0; i < 200; i++) {
        char buf[128];
        snprintf(buf, sizeof(buf), "Bulk message %d for capacity testing.", i);
        short_term_memory_add(&mem->short_term, MEMORY_ROLE_SYSTEM, buf);
    }
    memory_manage_capacity(mem);
    printf("  After bulk insert + management: working=%d shortterm=%d\n",
           mem->working.count, mem->short_term.count);

    printf("\nToken estimation:\n");
    printf("  'hello world': %d tokens\n", memory_estimate_tokens("hello world"));
    printf("  'The quick brown fox jumps over the lazy dog.': %d tokens\n",
           memory_estimate_tokens("The quick brown fox jumps over the lazy dog."));

    print_separator("Memory State Summary");
    printf("  Working memory:    %d items\n", mem->working.count);
    printf("  Short-term memory: %d items (%d tokens)\n", mem->short_term.count, mem->short_term.current_tokens);
    printf("  Long-term memory:  %d vectors\n", mem->long_term.count);
    printf("  Summary memory:    %d summaries\n", mem->summary.count);
    printf("  Episodic memory:   %d episodes\n", mem->episodic.count);
    printf("  Reflection memory: %d reflections\n", mem->reflection.count);
    printf("  Total tokens:      %d\n", mem->total_tokens);

    agent_memory_destroy(mem);
    printf("\nDemo completed successfully.\n");
    return 0;
}
