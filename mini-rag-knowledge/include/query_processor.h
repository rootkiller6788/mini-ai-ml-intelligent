#ifndef QUERY_PROCESSOR_H
#define QUERY_PROCESSOR_H

#include <stdbool.h>
#include <stddef.h>

/* ──────────────────────────────────────────────
   L5: Advanced Query Processing for RAG

   Query processing transforms user queries for better
   retrieval performance. This module implements:

   1. Query Decomposition: break complex multi-hop queries
      into simpler sub-queries (e.g., DECOMPRC from FLARE)

   2. Query Expansion: generate synonyms and related terms
      for lexical matching improvement

   3. Query Reformulation: rewrite queries using retrieved
      context (Iterative Retrieval)

   4. Hypothetical Document Embedding (HyDE): generate
      synthetic answer embedding for better vector search

   References:
   - Gao et al., "Precise Zero-Shot Dense Retrieval without
     Relevance Labels" (HyDE, 2022)
   - Trivedi et al., "IRCoT: Interleaving Retrieval with
     Chain-of-Thought" (2022)
   - Wang et al., "Query2doc: Query Expansion with LLMs" (2023)
   ────────────────────────────────────────────── */

/* Query decomposition: types of sub-queries */
typedef enum {
    SUBQ_FACTUAL,       /* Simple fact lookup     */
    SUBQ_COMPARISON,    /* Compare/contrast       */
    SUBQ_AGGREGATE,     /* Count/summarize        */
    SUBQ_CAUSAL,        /* Why/how reasoning      */
    SUBQ_DEFINITIONAL   /* What is X             */
} SubQueryType;

/* A decomposed sub-query */
typedef struct {
    char        *text;
    SubQueryType type;
    float        priority;     /* 0.0-1.0 importance weight */
    char        *keywords;
} SubQuery;

typedef struct {
    SubQuery *queries;
    size_t    count;
    size_t    capacity;
} SubQueryList;

/* Query Expansion: synonym-based term expansion */
typedef struct {
    char   *original;
    char  **expanded_terms;
    size_t  num_terms;
} ExpandedTerm;

typedef struct {
    ExpandedTerm *terms;
    size_t        count;
} QueryExpansionTerms;

/* Query Reformulation: context-aware rewrite */
typedef struct {
    char  *rewritten_query;
    float  confidence;
    char  *rationale;
} ReformulatedQuery;

/* Multi-Step Retrieval Plan */
typedef enum {
    PLAN_SINGLE_HOP,
    PLAN_PARALLEL,       /* Execute sub-queries in parallel     */
    PLAN_SEQUENTIAL,     /* Chain: result of Q1 feeds into Q2   */
    PLAN_ITERATIVE       /* Repeat retrieval with reformulation  */
} PlanType;

typedef struct {
    PlanType     type;
    SubQueryList sub_queries;
    char        *synthesized_answer;
} RetrievalPlan;

/* ──────────────────────────────────────────────
   Lifecycle
   ────────────────────────────────────────────── */
SubQueryList*  subquery_list_create(void);
void           subquery_list_destroy(SubQueryList *sql);

QueryExpansionTerms*  query_expansion_terms_create(void);
void                  query_expansion_terms_destroy(QueryExpansionTerms *qe);

ReformulatedQuery*  reformulated_query_create(void);
void                reformulated_query_destroy(ReformulatedQuery *rq);

RetrievalPlan*  retrieval_plan_create(void);
void            retrieval_plan_destroy(RetrievalPlan *rp);

/* ──────────────────────────────────────────────
   Query Decomposition

   Breaks a complex query into sub-queries using:
   - Conjunction splitting ("and", "also", "additionally")
   - Comparison detection ("vs", "compared to", "difference between")
   - Multi-hop detection ("What causes X's Y to Z?")
   ────────────────────────────────────────────── */
SubQueryList*  query_decompose(const char *query);

/* ──────────────────────────────────────────────
   Query Expansion: Term-Level

   Generates synonymous terms for each content word
   using a lightweight similarity approach:
   - Stemming approximation
   - Common synonym pairs
   - Acronym expansion
   ────────────────────────────────────────────── */
QueryExpansionTerms*  query_expand_terms(const char *query);

/* ──────────────────────────────────────────────
   Query Reformulation: Iterative Refinement

   Rewrites a query based on retrieved document snippets
   to improve retrieval in subsequent iterations.
   ────────────────────────────────────────────── */
ReformulatedQuery*  query_reformulate(const char *original_query,
                                       const char **retrieved_snippets,
                                       size_t num_snippets);

/* ──────────────────────────────────────────────
   Retrieval Plan Construction

   Creates a multi-step retrieval plan based on query
   complexity analysis.
   ────────────────────────────────────────────── */
RetrievalPlan*  query_build_retrieval_plan(const char *query);

/* ──────────────────────────────────────────────
   Query Complexity Analysis

   Estimates query complexity:
   - Single-hop: simple fact lookup
   - Multi-hop: requires multiple retrieval steps
   - Comparative: requires comparing entities
   - Causal: requires reasoning chain
   ────────────────────────────────────────────── */
typedef enum {
    COMPLEXITY_SIMPLE,
    COMPLEXITY_MODERATE,
    COMPLEXITY_MULTI_HOP,
    COMPLEXITY_COMPARATIVE,
    COMPLEXITY_CAUSAL
} QueryComplexity;

QueryComplexity  query_analyze_complexity(const char *query);

/* ──────────────────────────────────────────────
   Keyword Extraction

   Extracts key terms from a query, removing stop words.
   ────────────────────────────────────────────── */
char**  query_extract_keywords(const char *query, size_t *num_keywords);
void    query_free_keywords(char **keywords, size_t n);

/* ──────────────────────────────────────────────
   Query Intent Classification

   Classifies the user's intent for routing.
   ────────────────────────────────────────────── */
typedef enum {
    INTENT_FACTUAL,
    INTENT_HOW_TO,
    INTENT_WHY,
    INTENT_COMPARISON,
    INTENT_LIST,
    INTENT_DEFINITION,
    INTENT_YES_NO,
    INTENT_OTHER
} QueryIntent;

QueryIntent  query_classify_intent(const char *query);

/* ──────────────────────────────────────────────
   Result Aggregation for Multi-Hop Queries

   Merges and de-duplicates results from multiple
   sub-queries, ranking by combined relevance.
   ────────────────────────────────────────────── */
typedef struct {
    size_t *doc_ids;
    float  *scores;
    size_t  count;
    size_t  capacity;
} AggregatedResults;

AggregatedResults*  aggregate_create(size_t capacity);
void                aggregate_destroy(AggregatedResults *ar);
void                aggregate_merge(AggregatedResults *ar,
                                     const size_t *doc_ids,
                                     const float *scores,
                                     size_t count);

#endif /* QUERY_PROCESSOR_H */