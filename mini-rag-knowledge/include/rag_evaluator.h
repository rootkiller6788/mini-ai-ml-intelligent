#ifndef RAG_EVALUATOR_H
#define RAG_EVALUATOR_H

#include <stdbool.h>
#include <stddef.h>

/* ──────────────────────────────────────────────
   L4: Information Retrieval Evaluation Theorems

   NDCG (Normalized Discounted Cumulative Gain):
     DCG@K = Σᵢ₌₁ᴷ (2^relᵢ - 1) / log₂(i+1)
     NDCG@K = DCG@K / IDCG@K

   MAP (Mean Average Precision):
     AP = Σₖ (P@k * rel(k)) / |Relevant|
     MAP = (1/|Q|) Σ AP(q)

   MRR (Mean Reciprocal Rank):
     MRR = (1/|Q|) Σ (1 / rank₁)

   References:
   - Manning, Raghavan, Schütze, "Intro to IR" (Cambridge, 2008)
   - Järvelin & Kekäläinen, "Cumulated Gain-Based Evaluation" (ACM TOIS, 2002)
   ────────────────────────────────────────────── */

typedef enum {
    REL_NOT_RELEVANT = 0,
    REL_MARGINAL     = 1,
    REL_RELEVANT     = 2,
    REL_HIGHLY_REL   = 3,
    REL_PERFECT      = 4
} RelevanceGrade;

typedef struct {
    size_t  query_id;
    size_t  doc_id;
    int     relevance;
} RelevanceJudgment;

typedef struct {
    RelevanceJudgment *judgments;
    size_t             count;
    size_t             capacity;
} JudgmentSet;

typedef struct {
    size_t  doc_id;
    float   score;
    int     relevance;
} RankedDoc;

typedef struct {
    RankedDoc *docs;
    size_t     count;
    size_t     capacity;
    size_t     query_id;
} RankedList;

typedef struct {
    double  ndcg_at_k;
    double  map;
    double  mrr;
    double  precision_at_k;
    double  recall_at_k;
    double  f1_at_k;
    size_t  k;
    size_t  num_queries;
} EvalResult;

/* Lifecycle */
JudgmentSet*  judgmentset_create(size_t capacity);
void          judgmentset_destroy(JudgmentSet *js);
void          judgmentset_add(JudgmentSet *js, size_t qid, size_t did, int rel);

RankedList*   rankedlist_create(size_t capacity);
void          rankedlist_destroy(RankedList *rl);
void          rankedlist_set_relevance(RankedList *rl, const JudgmentSet *js);

/* DCG/NDCG — Järvelin & Kekäläinen (2002) */
double  dcg_at_k(const RankedList *rl, size_t k);
double  idcg_at_k(const RankedList *rl, size_t k);
double  ndcg_at_k(const RankedList *rl, size_t k);

/* AP/MAP */
double  average_precision(const RankedList *rl);
double  mean_average_precision(RankedList **queries, size_t num_queries);

/* RR/MRR — Voorhees (1999) */
double  reciprocal_rank(const RankedList *rl);
double  mean_reciprocal_rank(RankedList **queries, size_t num_queries);

/* Precision/Recall/F1 at K */
double  precision_at_k(const RankedList *rl, size_t k);
double  recall_at_k(const RankedList *rl, size_t k);
double  f1_at_k(const RankedList *rl, size_t k);

/* Full batch evaluation */
EvalResult  evaluate_queries(RankedList **queries, size_t num_queries,
                              const JudgmentSet *js, size_t k);

/* Helpers */
size_t  count_relevant(const RankedList *rl, size_t k);
size_t  count_total_relevant(const RankedList *rl);

#endif /* RAG_EVALUATOR_H */
