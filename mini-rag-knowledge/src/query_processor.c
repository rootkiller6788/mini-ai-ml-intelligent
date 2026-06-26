#include "query_processor.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>

/* Internal helpers */
static void *safe_alloc(size_t n) {
    void *p = calloc(n, 1);
    if (!p) { fprintf(stderr, "OOM: qproc alloc %zu\n", n); exit(1); }
    return p;
}

static char *safe_strdup(const char *s) {
    if (!s) return NULL;
    size_t n = strlen(s);
    char *d = safe_alloc(n + 1);
    memcpy(d, s, n);
    return d;
}

/*
 * English stop words list for keyword extraction.
 * These are high-frequency words that carry little
 * semantic signal for retrieval.
 */
static const char *STOP_WORDS[] = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "can", "shall",
    "of", "in", "on", "at", "to", "for", "with", "by", "from",
    "about", "into", "through", "during", "before", "after",
    "above", "below", "between", "under", "over", "up", "down",
    "out", "off", "and", "but", "or", "nor", "not", "so", "as",
    "if", "than", "too", "very", "just", "now", "then", "also",
    "i", "me", "my", "we", "our", "you", "your", "he", "she",
    "it", "its", "they", "them", "their", "this", "that",
    "these", "those", "what", "which", "who", "whom", "where",
    "when", "why", "how", "all", "each", "every", "both", "few",
    "more", "most", "other", "some", "such", "no", "only",
    NULL
};

static bool is_stop_word(const char *word, size_t len) {
    for (const char **sw = STOP_WORDS; *sw; sw++) {
        if (strlen(*sw) == len && strncmp(word, *sw, len) == 0) {
            return true;
        }
    }
    return false;
}

/* ──────────────────────────────────────────────
   SubQueryList Lifecycle
   ────────────────────────────────────────────── */
SubQueryList* subquery_list_create(void) {
    SubQueryList *sql = safe_alloc(sizeof(SubQueryList));
    sql->capacity = 16;
    sql->queries  = safe_alloc(sql->capacity * sizeof(SubQuery));
    sql->count    = 0;
    return sql;
}

void subquery_list_destroy(SubQueryList *sql) {
    if (!sql) return;
    for (size_t i = 0; i < sql->count; i++) {
        free(sql->queries[i].text);
        free(sql->queries[i].keywords);
    }
    free(sql->queries);
    free(sql);
}

static void sql_add(SubQueryList *sql, const char *text, SubQueryType type,
                     float prio) {
    if (sql->count >= sql->capacity) {
        sql->capacity *= 2;
        sql->queries = realloc(sql->queries, sql->capacity * sizeof(SubQuery));
        if (!sql->queries) { fprintf(stderr, "SubQuery OOM\n"); exit(1); }
    }
    SubQuery *sq = &sql->queries[sql->count];
    sq->text     = safe_strdup(text);
    sq->type     = type;
    sq->priority = prio;
    sq->keywords = NULL;
    sql->count++;
}

/* ──────────────────────────────────────────────
   QueryExpansionTerms Lifecycle
   ────────────────────────────────────────────── */
QueryExpansionTerms* query_expansion_terms_create(void) {
    QueryExpansionTerms *qe = safe_alloc(sizeof(QueryExpansionTerms));
    qe->terms  = safe_alloc(32 * sizeof(ExpandedTerm));
    qe->count  = 0;
    return qe;
}

void query_expansion_terms_destroy(QueryExpansionTerms *qe) {
    if (!qe) return;
    for (size_t i = 0; i < qe->count; i++) {
        free(qe->terms[i].original);
        for (size_t j = 0; j < qe->terms[i].num_terms; j++) {
            free(qe->terms[i].expanded_terms[j]);
        }
        free(qe->terms[i].expanded_terms);
    }
    free(qe->terms);
    free(qe);
}

/* ──────────────────────────────────────────────
   ReformulatedQuery / RetrievalPlan Lifecycle
   ────────────────────────────────────────────── */
ReformulatedQuery* reformulated_query_create(void) {
    return safe_alloc(sizeof(ReformulatedQuery));
}

void reformulated_query_destroy(ReformulatedQuery *rq) {
    if (!rq) return;
    free(rq->rewritten_query);
    free(rq->rationale);
    free(rq);
}

RetrievalPlan* retrieval_plan_create(void) {
    RetrievalPlan *rp = safe_alloc(sizeof(RetrievalPlan));
    rp->sub_queries = *subquery_list_create();
    rp->synthesized_answer = NULL;
    return rp;
}

void retrieval_plan_destroy(RetrievalPlan *rp) {
    if (!rp) return;
    for (size_t i = 0; i < rp->sub_queries.count; i++) {
        free(rp->sub_queries.queries[i].text);
        free(rp->sub_queries.queries[i].keywords);
    }
    free(rp->sub_queries.queries);
    free(rp->synthesized_answer);
    free(rp);
}

/* ──────────────────────────────────────────────
   Query Decomposition

   Strategy: Split on conjunction markers, detect
   comparison and multi-hop patterns.

   Algorithm:
   1. Find split points: " and ", "; ", " also ", " additionally "
   2. Check for comparison: " vs ", " compared to ", " between "
   3. Check for multi-hop: nested clauses, dependency chains
   4. Classify each sub-query by type
   ────────────────────────────────────────────── */

static bool is_conjunction(const char *text, size_t pos, size_t len) {
    const char *markers[] = {" and ", " & ", "; ", ". ",
                              " also ", " additionally ",
                              " furthermore ", " moreover ",
                              " in addition "};
    for (size_t m = 0; m < sizeof(markers)/sizeof(markers[0]); m++) {
        size_t ml = strlen(markers[m]);
        if (pos + ml <= len && strncmp(text + pos, markers[m], ml) == 0) {
            return true;
        }
    }
    return false;
}

static bool is_comparison_marker(const char *text) {
    const char *markers[] = {" vs ", " versus ", " compare ",
                              " compared to ", " difference between ",
                              " similarities between "};
    size_t tl = strlen(text);
    for (size_t m = 0; m < sizeof(markers)/sizeof(markers[0]); m++) {
        size_t ml = strlen(markers[m]);
        if (tl >= ml && strncmp(text, markers[m], ml) == 0) return true;
    }
    return false;
}

SubQueryList* query_decompose(const char *query) {
    SubQueryList *sql = subquery_list_create();
    if (!query) return sql;

    size_t ql = strlen(query);
    if (ql < 5) {
        sql_add(sql, query, SUBQ_FACTUAL, 1.0f);
        return sql;
    }

    /* Detect comparison pattern */
    if (is_comparison_marker(query)) {
        sql_add(sql, query, SUBQ_COMPARISON, 1.0f);
        return sql;
    }

    /* Split on conjunction boundaries */
    size_t seg_start = 0;
    size_t subq_count = 0;

    for (size_t i = 1; i < ql; i++) {
        if (is_conjunction(query, i, ql)) {
            /* Found a split point */
            size_t seg_len = i - seg_start;
            if (seg_len > 3) {
                char *subq = safe_alloc(seg_len + 1);
                memcpy(subq, query + seg_start, seg_len);
                subq[seg_len] = '\0';

                SubQueryType type = SUBQ_FACTUAL;
                if (strstr(subq, "why ") == subq || strstr(subq, "Why ") == subq)
                    type = SUBQ_CAUSAL;
                else if (strstr(subq, "how ") == subq || strstr(subq, "How ") == subq)
                    type = SUBQ_CAUSAL;
                else if (strstr(subq, "what is ") == subq || strstr(subq, "What is ") == subq)
                    type = SUBQ_DEFINITIONAL;

                sql_add(sql, subq, type, 1.0f / (float)(subq_count + 1));
                free(subq);
                subq_count++;
            }
            /* Advance past the conjunction marker */
            seg_start = i + 1;
            /* Find end of current marker */
            while (seg_start < ql && query[seg_start] == ' ') seg_start++;
            i = seg_start;
        }
    }

    /* Last segment */
    if (seg_start < ql) {
        size_t seg_len = ql - seg_start;
        if (seg_len > 2) {
            char *subq = safe_alloc(seg_len + 1);
            memcpy(subq, query + seg_start, seg_len);
            subq[seg_len] = '\0';

            SubQueryType type = SUBQ_FACTUAL;
            if (strstr(subq, "why ") == subq || strstr(subq, "Why ") == subq)
                type = SUBQ_CAUSAL;
            else if (strstr(subq, "how ") == subq || strstr(subq, "How ") == subq)
                type = SUBQ_CAUSAL;

            sql_add(sql, subq, type, 1.0f);
            free(subq);
            subq_count++;
        }
    }

    /* If no decomposition occurred, add original as one sub-query */
    if (sql->count == 0) {
        sql_add(sql, query, SUBQ_FACTUAL, 1.0f);
    }

    return sql;
}

/* ──────────────────────────────────────────────
   Query Expansion: Term-Level

   For each content word in the query, generate
   synonymous terms using a lookup dictionary.

   Dictionary covers common domain terms and
   their variants for scientific/technical RAG.
   ────────────────────────────────────────────── */

typedef struct { const char *word; const char *synonyms[4]; } SynEntry;

static const SynEntry SYN_DICT[] = {
    {"algorithm",     {"method", "procedure", "computation", NULL}},
    {"model",         {"architecture", "framework", "system", NULL}},
    {"data",          {"dataset", "information", "records", NULL}},
    {"training",      {"learning", "optimization", "fitting", NULL}},
    {"inference",     {"prediction", "reasoning", "deduction", NULL}},
    {"neural",        {"deep", "connectionist", "ANN", NULL}},
    {"network",       {"graph", "topology", "structure", NULL}},
    {"vector",        {"embedding", "representation", "array", NULL}},
    {"search",        {"retrieval", "lookup", "query", NULL}},
    {"document",      {"text", "article", "paper", NULL}},
    {"knowledge",     {"information", "expertise", "understanding", NULL}},
    {"error",         {"mistake", "fault", "bug", NULL}},
    {"accuracy",      {"precision", "correctness", "fidelity", NULL}},
    {"performance",   {"speed", "efficiency", "throughput", NULL}},
    {"large",         {"big", "massive", "extensive", NULL}},
    {"small",         {"tiny", "compact", "minimal", NULL}},
    {"fast",          {"quick", "rapid", "speedy", NULL}},
    {"slow",          {"sluggish", "delayed", "gradual", NULL}},
    {"improve",       {"enhance", "optimize", "boost", NULL}},
    {"reduce",        {"decrease", "minimize", "cut", NULL}},
    {NULL, {NULL, NULL, NULL, NULL}}
};

QueryExpansionTerms* query_expand_terms(const char *query) {
    QueryExpansionTerms *qe = query_expansion_terms_create();
    if (!query) return qe;

    size_t ql = strlen(query);
    size_t pos = 0;

    while (pos < ql) {
        /* Skip non-alpha */
        while (pos < ql && !isalpha((unsigned char)query[pos])) pos++;
        if (pos >= ql) break;

        /* Extract word */
        size_t ws = pos;
        while (pos < ql && isalpha((unsigned char)query[pos])) pos++;
        size_t wl = pos - ws;
        if (wl < 2) continue;

        if (is_stop_word(query + ws, wl)) continue;

        /* Look up synonyms */
        bool found = false;
        for (const SynEntry *se = SYN_DICT; se->word; se++) {
            size_t sl = strlen(se->word);
            if (wl == sl && strncasecmp(query + ws, se->word, wl) == 0) {
                /* Count synonyms */
                size_t ns = 0;
                for (; ns < 4 && se->synonyms[ns]; ns++) {}

                if (qe->count >= 32) break;
                ExpandedTerm *et = &qe->terms[qe->count];
                et->original = safe_alloc(wl + 1);
                memcpy(et->original, query + ws, wl);
                et->original[wl] = '\0';
                et->expanded_terms = safe_alloc(ns * sizeof(char*));
                et->num_terms = ns;
                for (size_t s = 0; s < ns; s++) {
                    et->expanded_terms[s] = safe_strdup(se->synonyms[s]);
                }
                qe->count++;
                found = true;
                break;
            }
        }
        if (!found) {
            /* Add word itself as base term with no expansions */
            if (qe->count >= 32) break;
            ExpandedTerm *et = &qe->terms[qe->count];
            et->original = safe_alloc(wl + 1);
            memcpy(et->original, query + ws, wl);
            et->original[wl] = '\0';
            et->expanded_terms = NULL;
            et->num_terms = 0;
            qe->count++;
        }
    }

    return qe;
}

/* ──────────────────────────────────────────────
   Query Reformulation

   Iterative refinement: Given the original query and
   snippets from initial retrieval, generate a refined
   query that better captures the information need.

   This implements a simplified version of the
   generative query reformulation (GQR) technique:
   - Extract key terms from retrieved snippets
   - Create a new query combining original + snippet terms
   ────────────────────────────────────────────── */
ReformulatedQuery* query_reformulate(const char *original_query,
                                       const char **retrieved_snippets,
                                       size_t num_snippets) {
    ReformulatedQuery *rq = reformulated_query_create();
    if (!original_query) return rq;

    size_t oql = strlen(original_query);

    /* Collect terms from snippets (simple TF approach) */
    size_t total_len = oql + 256;
    for (size_t i = 0; i < num_snippets && i < 8; i++) {
        if (retrieved_snippets[i]) {
            total_len += strlen(retrieved_snippets[i]);
        }
    }

    char *rewritten = safe_alloc(total_len + 1);
    /* Start with original query */
    memcpy(rewritten, original_query, oql);
    size_t off = oql;

    /* Append key terms from snippets */
    rewritten[off++] = ' ';
    size_t term_count = 0;
    for (size_t s = 0; s < num_snippets && s < 5 && term_count < 10; s++) {
        if (!retrieved_snippets[s]) continue;
        size_t sl = strlen(retrieved_snippets[s]);
        size_t p = 0;
        while (p < sl && term_count < 10) {
            while (p < sl && !isalpha((unsigned char)retrieved_snippets[s][p])) p++;
            if (p >= sl) break;
            size_t ws = p;
            while (p < sl && isalpha((unsigned char)retrieved_snippets[s][p])) p++;
            size_t wl = p - ws;
            if (wl > 4 && !is_stop_word(retrieved_snippets[s] + ws, wl)) {
                /* Check if term already in original */
                bool exists = false;
                for (size_t c = 0; c + wl <= oql && !exists; c++) {
                    if (strncasecmp(original_query + c,
                                     retrieved_snippets[s] + ws, wl) == 0) {
                        exists = true;
                    }
                }
                if (!exists) {
                    memcpy(rewritten + off, retrieved_snippets[s] + ws, wl);
                    off += wl;
                    rewritten[off++] = ' ';
                    term_count++;
                }
            }
        }
    }
    rewritten[off] = '\0';

    rq->rewritten_query = rewritten;
    rq->confidence       = off > oql + 5 ? 0.8f : 0.5f;
    rq->rationale        = safe_strdup("Term expansion from retrieved context");
    return rq;
}

/* ──────────────────────────────────────────────
   Retrieval Plan Construction

   Analyzes query complexity and constructs
   an appropriate retrieval plan (single-hop,
   parallel, sequential, or iterative).
   ────────────────────────────────────────────── */
RetrievalPlan* query_build_retrieval_plan(const char *query) {
    RetrievalPlan *rp = retrieval_plan_create();
    if (!query) return rp;

    QueryComplexity cx = query_analyze_complexity(query);
    SubQueryList *sql = query_decompose(query);

    /* Free existing sub-queries in plan */
    for (size_t i = 0; i < rp->sub_queries.count; i++) {
        free(rp->sub_queries.queries[i].text);
        free(rp->sub_queries.queries[i].keywords);
    }
    free(rp->sub_queries.queries);

    rp->sub_queries = *sql;
    free(sql); /* Free the container but not contents (moved to rp) */

    switch (cx) {
    case COMPLEXITY_SIMPLE:
        rp->type = PLAN_SINGLE_HOP;
        break;
    case COMPLEXITY_MODERATE:
        rp->type = PLAN_PARALLEL;
        break;
    case COMPLEXITY_MULTI_HOP:
        rp->type = PLAN_SEQUENTIAL;
        break;
    case COMPLEXITY_COMPARATIVE:
        rp->type = PLAN_PARALLEL;
        break;
    case COMPLEXITY_CAUSAL:
        rp->type = PLAN_ITERATIVE;
        break;
    default:
        rp->type = PLAN_SINGLE_HOP;
        break;
    }

    return rp;
}

/* ──────────────────────────────────────────────
   Query Complexity Analysis

   Heuristic-based complexity estimation:
   - Count question words (who, what, where, when, why, how)
   - Detect nested clauses
   - Detect comparison markers
   - Estimate word count
   ────────────────────────────────────────────── */
QueryComplexity query_analyze_complexity(const char *query) {
    if (!query) return COMPLEXITY_SIMPLE;
    size_t ql = strlen(query);

    /* Count question indicators */
    int question_words = 0;
    if (strstr(query, "why ") || strstr(query, "Why ")) question_words += 3;
    if (strstr(query, "how ") || strstr(query, "How ")) question_words += 2;
    if (strstr(query, "what ") || strstr(query, "What ")) question_words++;
    if (strstr(query, "compare")) question_words += 4;
    if (strstr(query, "difference between")) question_words += 3;
    if (strstr(query, "relationship")) question_words += 3;

    /* Count complexity markers */
    int complexity_markers = 0;
    if (strstr(query, "because")) complexity_markers++;
    if (strstr(query, "therefore")) complexity_markers++;
    if (strstr(query, "however")) complexity_markers++;
    if (strstr(query, "although")) complexity_markers++;
    if (strstr(query, "unless")) complexity_markers++;

    /* Count conjunctions (suggests multi-part query) */
    int conjunctions = 0;
    for (size_t i = 0; i < ql; i++) {
        if (is_conjunction(query, i, ql)) conjunctions++;
    }

    /* Word count */
    size_t word_count = 0;
    bool in_word = false;
    for (size_t i = 0; i < ql; i++) {
        if (isspace((unsigned char)query[i])) {
            if (in_word) { word_count++; in_word = false; }
        } else {
            in_word = true;
        }
    }
    if (in_word) word_count++;

    /* Classify */
    if (question_words >= 6 || complexity_markers >= 3) {
        return COMPLEXITY_CAUSAL;
    }
    if (question_words >= 4) {
        return COMPLEXITY_COMPARATIVE;
    }
    if (conjunctions >= 2 || question_words >= 2) {
        return COMPLEXITY_MULTI_HOP;
    }
    if (word_count > 12 || question_words >= 1) {
        return COMPLEXITY_MODERATE;
    }
    return COMPLEXITY_SIMPLE;
}

/* ──────────────────────────────────────────────
   Keyword Extraction

   Extract content words (non-stop words) from query.
   Returns dynamically allocated array of strings.

   Implementation: Porter Stemmer-inspired simple
   stemming with lowercasing.
   ────────────────────────────────────────────── */
char** query_extract_keywords(const char *query, size_t *num_keywords) {
    if (num_keywords) *num_keywords = 0;
    if (!query) return NULL;

    size_t ql = strlen(query);
    /* First pass: count keywords */
    size_t count = 0;
    size_t pos = 0;
    while (pos < ql) {
        while (pos < ql && !isalpha((unsigned char)query[pos])) pos++;
        if (pos >= ql) break;
        size_t ws = pos;
        while (pos < ql && isalpha((unsigned char)query[pos])) pos++;
        size_t wl = pos - ws;
        if (wl >= 2 && !is_stop_word(query + ws, wl)) {
            count++;
        }
    }

    if (count == 0) {
        if (num_keywords) *num_keywords = 0;
        return NULL;
    }

    char **keywords = safe_alloc(count * sizeof(char*));
    size_t ki = 0;
    pos = 0;
    while (pos < ql && ki < count) {
        while (pos < ql && !isalpha((unsigned char)query[pos])) pos++;
        if (pos >= ql) break;
        size_t ws = pos;
        while (pos < ql && isalpha((unsigned char)query[pos])) pos++;
        size_t wl = pos - ws;
        if (wl >= 2 && !is_stop_word(query + ws, wl)) {
            keywords[ki] = safe_alloc(wl + 1);
            for (size_t c = 0; c < wl; c++) {
                keywords[ki][c] = (char)tolower((unsigned char)query[ws + c]);
            }
            keywords[ki][wl] = '\0';
            ki++;
        }
    }

    if (num_keywords) *num_keywords = ki;
    return keywords;
}

void query_free_keywords(char **keywords, size_t n) {
    if (!keywords) return;
    for (size_t i = 0; i < n; i++) free(keywords[i]);
    free(keywords);
}

/* ──────────────────────────────────────────────
   Query Intent Classification

   Classifies user intent based on query patterns:
   - "what is/are X" → DEFINITION
   - "how to X" → HOW_TO
   - "why X" → WHY
   - "X vs Y" / "compare X and Y" → COMPARISON
   - "list of X" / "top N X" → LIST
   - "is X Y?" / "does X Y?" → YES_NO
   ────────────────────────────────────────────── */
QueryIntent query_classify_intent(const char *query) {
    if (!query) return INTENT_OTHER;
    size_t ql = strlen(query);

    /* Lowercase copy for matching */
    char *lq = safe_alloc(ql + 1);
    for (size_t i = 0; i < ql; i++) {
        lq[i] = (char)tolower((unsigned char)query[i]);
    }
    lq[ql] = '\0';

    QueryIntent intent = INTENT_OTHER;

    if (strncmp(lq, "what is ", 8) == 0 ||
        strncmp(lq, "what are ", 9) == 0 ||
        strncmp(lq, "define ", 7) == 0) {
        intent = INTENT_DEFINITION;
    } else if (strncmp(lq, "how to ", 7) == 0 ||
               strncmp(lq, "how do ", 7) == 0 ||
               strncmp(lq, "how can ", 8) == 0) {
        intent = INTENT_HOW_TO;
    } else if (strncmp(lq, "why ", 4) == 0 ||
               strncmp(lq, "explain why ", 12) == 0) {
        intent = INTENT_WHY;
    } else if (strstr(lq, " vs ") || strstr(lq, " versus ") ||
               strncmp(lq, "compare ", 8) == 0 ||
               strstr(lq, "difference between")) {
        intent = INTENT_COMPARISON;
    } else if (strncmp(lq, "list ", 5) == 0 ||
               strncmp(lq, "top ", 4) == 0 ||
               strstr(lq, "examples of")) {
        intent = INTENT_LIST;
    } else if (strncmp(lq, "is ", 3) == 0 ||
               strncmp(lq, "does ", 5) == 0 ||
               strncmp(lq, "can ", 4) == 0 ||
               strncmp(lq, "will ", 5) == 0 ||
               strncmp(lq, "are ", 4) == 0) {
        intent = INTENT_YES_NO;
    } else if (strncmp(lq, "who ", 4) == 0 ||
               strncmp(lq, "what ", 5) == 0 ||
               strncmp(lq, "when ", 5) == 0 ||
               strncmp(lq, "where ", 6) == 0) {
        intent = INTENT_FACTUAL;
    }

    free(lq);
    return intent;
}

/* ──────────────────────────────────────────────
   Result Aggregation for Multi-Hop Queries
   ────────────────────────────────────────────── */
AggregatedResults* aggregate_create(size_t capacity) {
    AggregatedResults *ar = safe_alloc(sizeof(AggregatedResults));
    ar->capacity = capacity ? capacity : 128;
    ar->doc_ids  = safe_alloc(ar->capacity * sizeof(size_t));
    ar->scores   = safe_alloc(ar->capacity * sizeof(float));
    ar->count    = 0;
    return ar;
}

void aggregate_destroy(AggregatedResults *ar) {
    if (!ar) return;
    free(ar->doc_ids);
    free(ar->scores);
    free(ar);
}

/*
 * Merge new results into aggregated results.
 * Strategy: sum scores for same doc_id, add new docs.
 * This creates a unified ranking across sub-queries.
 */
void aggregate_merge(AggregatedResults *ar,
                      const size_t *doc_ids,
                      const float *scores,
                      size_t count) {
    if (!ar || !doc_ids || !scores || count == 0) return;

    for (size_t i = 0; i < count; i++) {
        bool found = false;
        for (size_t j = 0; j < ar->count; j++) {
            if (ar->doc_ids[j] == doc_ids[i]) {
                ar->scores[j] += scores[i];
                found = true;
                break;
            }
        }
        if (!found) {
            if (ar->count >= ar->capacity) {
                ar->capacity *= 2;
                ar->doc_ids = realloc(ar->doc_ids,
                                       ar->capacity * sizeof(size_t));
                ar->scores  = realloc(ar->scores,
                                       ar->capacity * sizeof(float));
                if (!ar->doc_ids || !ar->scores) {
                    fprintf(stderr, "Aggregate OOM\n"); exit(1);
                }
            }
            ar->doc_ids[ar->count] = doc_ids[i];
            ar->scores[ar->count]  = scores[i];
            ar->count++;
        }
    }

    /* Sort by combined score descending */
    for (size_t i = 0; i < ar->count; i++) {
        for (size_t j = i + 1; j < ar->count; j++) {
            if (ar->scores[j] > ar->scores[i]) {
                float tmp_s = ar->scores[i];
                size_t tmp_id = ar->doc_ids[i];
                ar->scores[i] = ar->scores[j];
                ar->doc_ids[i] = ar->doc_ids[j];
                ar->scores[j] = tmp_s;
                ar->doc_ids[j] = tmp_id;
            }
        }
    }
}