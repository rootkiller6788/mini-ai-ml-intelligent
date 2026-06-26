#ifndef AGENT_METRICS_H
#define AGENT_METRICS_H

#include <stddef.h>
#include <stdbool.h>

/**
 * agent_metrics.h — Agent Evaluation Metrics Framework
 *
 * L4 Standards: Implements standard NLP evaluation metrics (BLEU, ROUGE-L, F1)
 * used to quantify agent output quality. Based on:
 *   - BLEU: Papineni et al., ACL 2002 — Bilingual Evaluation Understudy
 *   - ROUGE: Lin, ACL 2004 — Recall-Oriented Understudy for Gisting Evaluation
 *   - F1 score: Standard harmonic mean of precision and recall
 *
 * L7 Application: Agent evaluation in production — A/B testing, regression detection
 *
 * Course mapping:
 *   - Stanford CS 224N (NLP): BLEU/ROUGE metrics
 *   - MIT 6.8610 (Quantitative Methods): Statistical testing
 *   - CMU 11-711 (Advanced NLP): Evaluation methodology
 *
 * Theorem (BLEU brevity penalty):
 *   BP = min(1, exp(1 - r/c)) where r = reference length, c = candidate length
 *
 * Theorem (ROUGE-L F1):
 *   F1 = 2 * P_lcs * R_lcs / (P_lcs + R_lcs)
 *   where P_lcs = LCS_len / candidate_len, R_lcs = LCS_len / ref_len
 *
 * Theorem (Bootstrap CI):
 *   CI = [x̄ - z_{α/2} * SE, x̄ + z_{α/2} * SE]
 *   where SE = σ_bootstrap / sqrt(B), B = bootstrap iterations
 */

#define METRICS_MAX_NGRAM       4
#define METRICS_MAX_RESULT_LEN  4096
#define METRICS_MAX_TOKENS      1024
#define METRICS_MAX_BENCHMARKS  64

/* ── L1: Core type definitions ── */

typedef enum {
    METRIC_BLEU,
    METRIC_ROUGE_L,
    METRIC_EXACT_MATCH,
    METRIC_F1,
    METRIC_PASS_AT_K,
    METRIC_CUSTOM,
    METRIC_COUNT
} metric_type_t;

typedef struct {
    metric_type_t type;
    double bleu_score;
    double rouge_l_precision;
    double rouge_l_recall;
    double rouge_l_f1;
    bool exact_match;
    double f1_precision;
    double f1_recall;
    double f1_score;
    double pass_at_k;
    int k_value;
    char name[METRICS_MAX_RESULT_LEN];
} metric_result_t;

typedef struct {
    const char *reference;
    const char *candidate;
    metric_result_t results[METRIC_COUNT];
    int result_count;
} evaluation_pair_t;

typedef struct {
    evaluation_pair_t *pairs;
    int pair_count;
    int capacity;
    double aggregate_bleu;
    double aggregate_rouge_l;
    double aggregate_exact_match_rate;
    double aggregate_f1;
    char dataset_name[256];
} benchmark_suite_t;

/* ── L1: API declarations ── */

/* Single metric evaluation */
metric_result_t metrics_evaluate_bleu(const char *reference, const char *candidate, int max_ngram);
metric_result_t metrics_evaluate_rouge_l(const char *reference, const char *candidate);
metric_result_t metrics_evaluate_exact_match(const char *reference, const char *candidate);
metric_result_t metrics_evaluate_f1(const char *reference, const char *candidate);
metric_result_t metrics_evaluate_pass_at_k(int *correct_samples, int total, int k);

/* Composite evaluation — runs all metrics */
evaluation_pair_t* metrics_evaluate_pair(const char *reference, const char *candidate);
void metrics_evaluation_pair_free(evaluation_pair_t *pair);

/* Benchmark suite */
benchmark_suite_t* metrics_benchmark_create(const char *dataset_name, int capacity);
void metrics_benchmark_destroy(benchmark_suite_t *suite);
int metrics_benchmark_add(benchmark_suite_t *suite, const char *reference, const char *candidate);
void metrics_benchmark_compute(benchmark_suite_t *suite);
const char* metrics_benchmark_report(const benchmark_suite_t *suite, char *buf, size_t buf_size);

/* L4: Bootstrap confidence interval
 * Theorem: CI = x̄ ± z_{α/2} * σ_boot / sqrt(B)
 * Uses standard normal approximation for z-scores
 */
typedef struct {
    double mean;
    double std_dev;
    double ci_lower;
    double ci_upper;
    int bootstrap_samples;
    double confidence_level;
} metrics_confidence_interval_t;

metrics_confidence_interval_t metrics_bootstrap_ci(const double *scores, int n,
                                                     int bootstrap_iter, double confidence);

/* L8: Paired bootstrap test for comparing two agent systems
 * H₀: system_a and system_b have equal mean performance
 * Two-sided test with bootstrap resampling
 */
typedef struct {
    double p_value;
    double delta_mean;
    bool significant;
    int bootstrap_iterations;
} metrics_system_comparison_t;

metrics_system_comparison_t metrics_compare_systems(const double *system_a, const double *system_b,
                                                      int n, int bootstrap_iter);

/* L5: Algorithm helpers */

/* N-gram precision: P_n = Σ_{ngram∈C} min(count_C(ngram), count_R(ngram)) / Σ_{ngram∈C} count_C(ngram) */
double metrics_ngram_precision(const char *reference, const char *candidate, int n);

/* Brevity penalty: BP = 1 if c > r, else exp(1 - r/c) */
double metrics_brevity_penalty(const char *reference, const char *candidate);

/* Longest Common Subsequence (dynamic programming, O(mn)) */
int metrics_lcs_length(const char *a, const char *b);
int metrics_lcs_dp(const char *a, const char *b, int la, int lb);

/* Tokenization: splits text into word tokens by whitespace/punctuation */
int metrics_tokenize(const char *text, char tokens[][64], int max_tokens);
int metrics_count_words(const char *text);

/* Utility */
const char* metrics_type_name(metric_type_t type);
const char* metrics_format_result(const metric_result_t *result, char *buf, size_t buf_size);

#endif
