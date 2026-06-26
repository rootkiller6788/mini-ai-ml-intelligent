#include "rag_evaluator.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* Internal helpers */
static void *safe_alloc(size_t n) {
    void *p = calloc(n, 1);
    if (!p) { fprintf(stderr, "OOM: eval alloc %zu\n", n); exit(1); }
    return p;
}

static int cmp_ranked_score_desc(const void *a, const void *b) {
    float d = ((const RankedDoc*)b)->score - ((const RankedDoc*)a)->score;
    return (d > 0) ? 1 : ((d < 0) ? -1 : 0);
}

static int cmp_ranked_relevance_desc(const void *a, const void *b) {
    int d = ((const RankedDoc*)b)->relevance - ((const RankedDoc*)a)->relevance;
    return d;
}

/* JudgmentSet Lifecycle */
JudgmentSet* judgmentset_create(size_t capacity) {
    JudgmentSet *js = safe_alloc(sizeof(JudgmentSet));
    js->capacity   = capacity ? capacity : 256;
    js->judgments  = safe_alloc(js->capacity * sizeof(RelevanceJudgment));
    js->count      = 0;
    return js;
}

void judgmentset_destroy(JudgmentSet *js) {
    if (!js) return;
    free(js->judgments);
    free(js);
}

void judgmentset_add(JudgmentSet *js, size_t qid, size_t did, int rel) {
    if (js->count >= js->capacity) {
        js->capacity *= 2;
        js->judgments = realloc(js->judgments,
                                 js->capacity * sizeof(RelevanceJudgment));
        if (!js->judgments) { fprintf(stderr, "JudgmentSet OOM\n"); exit(1); }
    }
    js->judgments[js->count].query_id  = qid;
    js->judgments[js->count].doc_id    = did;
    js->judgments[js->count].relevance = rel;
    js->count++;
}

/* RankedList Lifecycle */
RankedList* rankedlist_create(size_t capacity) {
    RankedList *rl = safe_alloc(sizeof(RankedList));
    rl->capacity = capacity ? capacity : 256;
    rl->docs     = safe_alloc(rl->capacity * sizeof(RankedDoc));
    rl->count    = 0;
    rl->query_id = 0;
    return rl;
}

void rankedlist_destroy(RankedList *rl) {
    if (!rl) return;
    free(rl->docs);
    free(rl);
}

void rankedlist_set_relevance(RankedList *rl, const JudgmentSet *js) {
    if (!rl || !js) return;
    for (size_t i = 0; i < rl->count; i++) {
        rl->docs[i].relevance = 0;
        for (size_t j = 0; j < js->count; j++) {
            if (js->judgments[j].query_id == rl->query_id &&
                js->judgments[j].doc_id == rl->docs[i].doc_id) {
                rl->docs[i].relevance = js->judgments[j].relevance;
                break;
            }
        }
    }
    qsort(rl->docs, rl->count, sizeof(RankedDoc), cmp_ranked_score_desc);
}

/* Count helpers */
size_t count_relevant(const RankedList *rl, size_t k) {
    if (!rl) return 0;
    size_t n = k < rl->count ? k : rl->count;
    size_t cnt = 0;
    for (size_t i = 0; i < n; i++) {
        if (rl->docs[i].relevance > 0) cnt++;
    }
    return cnt;
}

size_t count_total_relevant(const RankedList *rl) {
    if (!rl) return 0;
    size_t cnt = 0;
    for (size_t i = 0; i < rl->count; i++) {
        if (rl->docs[i].relevance > 0) cnt++;
    }
    return cnt;
}

/*
 * DCG@K - Discounted Cumulative Gain
 *
 * Formula: DCG@K = sum_{i=1}^{K} (2^{rel_i} - 1) / log_2(i + 1)
 *
 * Gain = 2^rel - 1  (exponential gain for higher relevance)
 * Discount = 1 / log_2(i+2)  (later positions discounted more)
 *
 * Reference: Jarvelin & Kekalainen (2002)
 * "Cumulated Gain-Based Evaluation of IR Techniques", ACM TOIS
 */
double dcg_at_k(const RankedList *rl, size_t k) {
    if (!rl || rl->count == 0 || k == 0) return 0.0;
    size_t n = k < rl->count ? k : rl->count;
    double dcg = 0.0;
    for (size_t i = 0; i < n; i++) {
        int rel = rl->docs[i].relevance;
        if (rel < 0) rel = 0;
        if (rel > 4) rel = 4;
        double gain = (double)((1 << rel) - 1);
        double discount = log2((double)(i + 2));
        if (discount > 0.0) dcg += gain / discount;
    }
    return dcg;
}

/*
 * IDCG@K - Ideal DCG
 *
 * Sort all documents by ground-truth relevance descending,
 * then compute DCG@K. This is the oracle upper bound.
 */
double idcg_at_k(const RankedList *rl, size_t k) {
    if (!rl || rl->count == 0 || k == 0) return 0.0;
    RankedList *ideal = rankedlist_create(rl->count);
    ideal->query_id = rl->query_id;
    ideal->count    = rl->count;
    for (size_t i = 0; i < rl->count; i++) {
        ideal->docs[i] = rl->docs[i];
    }
    qsort(ideal->docs, ideal->count, sizeof(RankedDoc),
          cmp_ranked_relevance_desc);
    double idcg = dcg_at_k(ideal, k);
    rankedlist_destroy(ideal);
    return idcg;
}

/*
 * NDCG@K - Normalized DCG
 *
 * NDCG@K = DCG@K / IDCG@K
 * Range [0, 1]. 1 = perfect ranking.
 */
double ndcg_at_k(const RankedList *rl, size_t k) {
    double d = dcg_at_k(rl, k);
    double i = idcg_at_k(rl, k);
    if (i < 1e-12) return 0.0;
    return d / i;
}

/*
 * Average Precision (AP)
 *
 * AP = (1/|R|) * sum_k P@k * rel(k)
 *
 * Where |R| = total relevant docs, P@k = precision at cutoff k,
 * rel(k) = 1 if doc at position k is relevant.
 *
 * Reference: Manning, Raghavan, Schutze, "Intro to IR", Ch. 8
 */
double average_precision(const RankedList *rl) {
    if (!rl || rl->count == 0) return 0.0;
    size_t total_rel = count_total_relevant(rl);
    if (total_rel == 0) return 0.0;
    double ap = 0.0;
    size_t rel_seen = 0;
    for (size_t k = 0; k < rl->count; k++) {
        if (rl->docs[k].relevance > 0) {
            rel_seen++;
            double p_at_k = (double)rel_seen / (double)(k + 1);
            ap += p_at_k;
        }
    }
    return ap / (double)total_rel;
}

/*
 * MAP - Mean Average Precision
 *
 * MAP = (1/|Q|) * sum_{q in Q} AP(q)
 *
 * Arithmetic mean of AP over all queries. The primary IR
 * evaluation metric for ranked retrieval.
 */
double mean_average_precision(RankedList **queries, size_t num_queries) {
    if (!queries || num_queries == 0) return 0.0;
    double sum = 0.0;
    for (size_t i = 0; i < num_queries; i++) {
        sum += average_precision(queries[i]);
    }
    return sum / (double)num_queries;
}

/*
 * Reciprocal Rank (RR)
 *
 * RR = 1 / rank_1
 *
 * rank_1 = 1-indexed position of first relevant document.
 *
 * Reference: Voorhees (1999), TREC-8 Question Answering Track
 */
double reciprocal_rank(const RankedList *rl) {
    if (!rl || rl->count == 0) return 0.0;
    for (size_t i = 0; i < rl->count; i++) {
        if (rl->docs[i].relevance > 0) {
            return 1.0 / (double)(i + 1);
        }
    }
    return 0.0;
}

/*
 * MRR - Mean Reciprocal Rank
 *
 * MRR = (1/|Q|) * sum_{q in Q} RR(q)
 */
double mean_reciprocal_rank(RankedList **queries, size_t num_queries) {
    if (!queries || num_queries == 0) return 0.0;
    double sum = 0.0;
    for (size_t i = 0; i < num_queries; i++) {
        sum += reciprocal_rank(queries[i]);
    }
    return sum / (double)num_queries;
}

/*
 * Precision@K
 *
 * P@K = |{relevant in top K}| / K
 */
double precision_at_k(const RankedList *rl, size_t k) {
    if (!rl || rl->count == 0 || k == 0) return 0.0;
    size_t n = k < rl->count ? k : rl->count;
    size_t rel = count_relevant(rl, n);
    return (double)rel / (double)n;
}

/*
 * Recall@K
 *
 * R@K = |{relevant in top K}| / |{all relevant}|
 */
double recall_at_k(const RankedList *rl, size_t k) {
    if (!rl || rl->count == 0) return 0.0;
    size_t total_rel = count_total_relevant(rl);
    if (total_rel == 0) return 0.0;
    size_t n = k < rl->count ? k : rl->count;
    size_t rel = count_relevant(rl, n);
    return (double)rel / (double)total_rel;
}

/*
 * F1@K
 *
 * F1 = 2 * (P * R) / (P + R)
 *
 * Harmonic mean of Precision@K and Recall@K.
 */
double f1_at_k(const RankedList *rl, size_t k) {
    double p = precision_at_k(rl, k);
    double r = recall_at_k(rl, k);
    if (p + r < 1e-12) return 0.0;
    return 2.0 * p * r / (p + r);
}

/*
 * Full batch evaluation over multiple queries.
 * Computes average NDCG@K, MAP, MRR, P@K, R@K, F1@K.
 */
EvalResult evaluate_queries(RankedList **queries, size_t num_queries,
                             const JudgmentSet *js, size_t k) {
    EvalResult er;
    memset(&er, 0, sizeof(er));
    er.k = k;
    er.num_queries = num_queries;
    if (!queries || num_queries == 0) return er;

    for (size_t q = 0; q < num_queries; q++) {
        if (queries[q]) {
            rankedlist_set_relevance(queries[q], js);
        }
    }

    for (size_t q = 0; q < num_queries; q++) {
        if (queries[q]) {
            er.ndcg_at_k += ndcg_at_k(queries[q], k);
        }
    }
    er.ndcg_at_k /= (double)num_queries;

    er.map = mean_average_precision(queries, num_queries);
    er.mrr = mean_reciprocal_rank(queries, num_queries);

    for (size_t q = 0; q < num_queries; q++) {
        if (queries[q]) {
            er.precision_at_k += precision_at_k(queries[q], k);
            er.recall_at_k    += recall_at_k(queries[q], k);
            er.f1_at_k        += f1_at_k(queries[q], k);
        }
    }
    er.precision_at_k /= (double)num_queries;
    er.recall_at_k    /= (double)num_queries;
    er.f1_at_k        /= (double)num_queries;

    return er;
}