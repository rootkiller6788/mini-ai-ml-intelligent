#include "agent_metrics.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

int metrics_count_words(const char *text) {
    if (!text) return 0;
    int count = 0, in_word = 0;
    for (int i = 0; text[i]; i++) {
        if (text[i] == 32 || text[i] == 10 || text[i] == 9 || text[i] == 13) {
            if (in_word) { count++; in_word = 0; }
        } else { in_word = 1; }
    }
    if (in_word) count++;
    return count;
}

/* ================================================================
 * agent_metrics.c -- Agent Evaluation Metrics Implementation
 *
 * L4 Standards: BLEU (Papineni et al., ACL 2002)
 *               ROUGE-L (Lin, ACL 2004)
 *               F1 Score (van Rijsbergen, 1979)
 *               Pass@k (Chen et al., 2021)
 *
 * L5 Algorithms:
 *   - N-gram precision with clipping
 *   - LCS via dynamic programming O(mn) time, O(min(m,n)) space
 *   - Bootstrap CI (Efron, 1979)
 *   - Paired bootstrap test (Koehn, EMNLP 2004)
 *
 * L8: Two-system comparison via paired bootstrap
 * ================================================================ */

int metrics_tokenize(const char *text, char tokens[][64], int max_tokens) {
    if (!text || !tokens || max_tokens <= 0) return 0;
    int count = 0;
    const char *p = text;
    while (*p && count < max_tokens) {
        while (*p == ' ' || *p == '\n' || *p == '\t' || *p == '\r') p++;
        if (!*p) break;
        const char *start = p;
        while (*p && *p != ' ' && *p != '\n' && *p != '\t' && *p != '\r') p++;
        size_t len = (size_t)(p - start);
        if (len >= 64) len = 63;
        memcpy(tokens[count], start, len);
        tokens[count][len] = '\0';
        count++;
    }
    return count;
}

/* N-gram precision: P_n = clipped / total candidate n-grams
 * Theorem: Modified n-gram precision penalizes over-generation */

double metrics_ngram_precision(const char *reference, const char *candidate, int n) {
    if (!reference || !candidate || n < 1) return 0.0;
    char ref_tokens[1024][64], cand_tokens[1024][64];
    int ref_count = metrics_tokenize(reference, ref_tokens, 1024);
    int cand_count = metrics_tokenize(candidate, cand_tokens, 1024);
    if (cand_count < n) return 0.0;
    int total_clipped = 0, total_ngrams = 0;
    for (int i = 0; i <= cand_count - n; i++) {
        total_ngrams++;
        int cand_cnt = 0;
        for (int ci = 0; ci <= i; ci++) {
            int ok = 1;
            for (int k = 0; k < n; k++)
                if (strcmp(cand_tokens[ci + k], cand_tokens[i + k]) != 0) { ok = 0; break; }
            if (ok) cand_cnt++;
        }
        int ref_cnt = 0;
        for (int ri = 0; ri <= ref_count - n; ri++) {
            int ok = 1;
            for (int k = 0; k < n; k++)
                if (strcmp(ref_tokens[ri + k], cand_tokens[i + k]) != 0) { ok = 0; break; }
            if (ok) ref_cnt++;
        }
        if (cand_cnt > 0 && ref_cnt > 0)
            total_clipped += (cand_cnt < ref_cnt) ? cand_cnt : ref_cnt;
    }
    if (total_ngrams == 0) return 0.0;
    return (double)total_clipped / (double)total_ngrams;
}

/* Brevity Penalty: BP = 1 if c > r else exp(1 - r/c) */

double metrics_brevity_penalty(const char *reference, const char *candidate) {
    if (!reference || !candidate) return 0.0;
    int ref_len = metrics_count_words(reference);
    int cand_len = metrics_count_words(candidate);
    if (ref_len == 0 || cand_len == 0) return 0.0;
    if (cand_len > ref_len) return 1.0;
    return exp(1.0 - (double)ref_len / (double)cand_len);
}

/* BLEU = BP * exp(sum_{n=1..N} w_n * log(p_n)), w_n = 1/N */

metric_result_t metrics_evaluate_bleu(const char *reference, const char *candidate, int max_ngram) {
    metric_result_t res;
    memset(&res, 0, sizeof(res));
    res.type = METRIC_BLEU;
    snprintf(res.name, sizeof(res.name), "BLEU-%d", max_ngram);
    if (!reference || !candidate) return res;
    if (max_ngram > METRICS_MAX_NGRAM) max_ngram = METRICS_MAX_NGRAM;
    double bp = metrics_brevity_penalty(reference, candidate);
    double log_sum = 0.0, weight = 1.0 / (double)max_ngram;
    int all_zero = 1;
    for (int n = 1; n <= max_ngram; n++) {
        double p_n = metrics_ngram_precision(reference, candidate, n);
        if (p_n < 1e-10) p_n = 1e-10;
        log_sum += weight * log(p_n);
        if (p_n > 1e-10) all_zero = 0;
    }
    res.bleu_score = all_zero ? 0.0 : bp * exp(log_sum);
    return res;
}

/* LCS via DP: O(mn) time, O(min(m,n)) space */

int metrics_lcs_dp(const char *a, const char *b, int la, int lb) {
    if (!a || !b || la == 0 || lb == 0) return 0;
    int *prev = (int*)calloc((size_t)(la + 1), sizeof(int));
    int *curr = (int*)calloc((size_t)(la + 1), sizeof(int));
    if (!prev || !curr) { free(prev); free(curr); return 0; }
    for (int j = 1; j <= lb; j++) {
        curr[0] = 0;
        for (int i = 1; i <= la; i++) {
            if (a[i - 1] == b[j - 1]) curr[i] = prev[i - 1] + 1;
            else curr[i] = (prev[i] > curr[i - 1]) ? prev[i] : curr[i - 1];
        }
        memcpy(prev, curr, (size_t)(la + 1) * sizeof(int));
    }
    int ans = prev[la];
    free(prev); free(curr);
    return ans;
}

int metrics_lcs_length(const char *a, const char *b) {
    if (!a || !b) return 0;
    return metrics_lcs_dp(a, b, (int)strlen(a), (int)strlen(b));
}

/* ROUGE-L: F1 = (1+b^2)*P*R/(R+b^2*P) with word-level LCS */

metric_result_t metrics_evaluate_rouge_l(const char *reference, const char *candidate) {
    metric_result_t res;
    memset(&res, 0, sizeof(res));
    res.type = METRIC_ROUGE_L;
    strncpy(res.name, "ROUGE-L", sizeof(res.name) - 1);
    if (!reference || !candidate) return res;
    char ref_words[1024][64], cand_words[1024][64];
    int ref_cnt = metrics_tokenize(reference, ref_words, 1024);
    int cand_cnt = metrics_tokenize(candidate, cand_words, 1024);
    int *prev = (int*)calloc((size_t)(ref_cnt + 1), sizeof(int));
    int *curr = (int*)calloc((size_t)(ref_cnt + 1), sizeof(int));
    if (!prev || !curr) { free(prev); free(curr); return res; }
    for (int j = 1; j <= cand_cnt; j++) {
        curr[0] = 0;
        for (int i = 1; i <= ref_cnt; i++) {
            if (strcmp(ref_words[i - 1], cand_words[j - 1]) == 0) curr[i] = prev[i - 1] + 1;
            else curr[i] = (prev[i] > curr[i - 1]) ? prev[i] : curr[i - 1];
        }
        memcpy(prev, curr, (size_t)(ref_cnt + 1) * sizeof(int));
    }
    int lcs_cnt = prev[ref_cnt];
    free(prev); free(curr);
    res.rouge_l_precision = (cand_cnt > 0) ? (double)lcs_cnt / (double)cand_cnt : 0.0;
    res.rouge_l_recall    = (ref_cnt > 0)  ? (double)lcs_cnt / (double)ref_cnt  : 0.0;
    double beta = 1.0, denom = res.rouge_l_recall + beta * beta * res.rouge_l_precision;
    res.rouge_l_f1 = (denom > 1e-10)
        ? (1.0 + beta * beta) * res.rouge_l_recall * res.rouge_l_precision / denom : 0.0;
    return res;
}

/* Exact Match: string equality after normalization */

metric_result_t metrics_evaluate_exact_match(const char *reference, const char *candidate) {
    metric_result_t res;
    memset(&res, 0, sizeof(res));
    res.type = METRIC_EXACT_MATCH;
    strncpy(res.name, "Exact Match", sizeof(res.name) - 1);
    if (!reference || !candidate) return res;
    res.exact_match = (strcmp(reference, candidate) == 0);
    return res;
}

/* F1 Score: word-level precision/recall harmonic mean */

metric_result_t metrics_evaluate_f1(const char *reference, const char *candidate) {
    metric_result_t res;
    memset(&res, 0, sizeof(res));
    res.type = METRIC_F1;
    strncpy(res.name, "F1 Score", sizeof(res.name) - 1);
    if (!reference || !candidate) return res;
    char ref_words[1024][64], cand_words[1024][64];
    int ref_cnt = metrics_tokenize(reference, ref_words, 1024);
    int cand_cnt = metrics_tokenize(candidate, cand_words, 1024);
    int intersect = 0;
    int *ref_used = (int*)calloc((size_t)ref_cnt, sizeof(int));
    if (!ref_used) return res;
    for (int i = 0; i < cand_cnt; i++) {
        for (int j = 0; j < ref_cnt; j++) {
            if (!ref_used[j] && strcmp(cand_words[i], ref_words[j]) == 0) {
                intersect++; ref_used[j] = 1; break;
            }
        }
    }
    free(ref_used);
    res.f1_precision = (cand_cnt > 0) ? (double)intersect / (double)cand_cnt : 0.0;
    res.f1_recall    = (ref_cnt > 0)  ? (double)intersect / (double)ref_cnt  : 0.0;
    double denom = res.f1_precision + res.f1_recall;
    res.f1_score = (denom > 1e-10)
        ? 2.0 * res.f1_precision * res.f1_recall / denom : 0.0;
    return res;
}

/* Pass@k: unbiased estimator = 1 - C(n-c,k)/C(n,k) */

static double n_choose_k(int n, int k) {
    if (k > n || k < 0) return 0.0;
    if (k == 0 || k == n) return 1.0;
    double val = 1.0;
    for (int i = 1; i <= k; i++) val *= (double)(n - k + i) / (double)i;
    return val;
}

metric_result_t metrics_evaluate_pass_at_k(int *correct_samples, int total, int k) {
    metric_result_t res;
    memset(&res, 0, sizeof(res));
    res.type = METRIC_PASS_AT_K;
    res.k_value = k;
    snprintf(res.name, sizeof(res.name), "Pass@%d", k);
    if (!correct_samples || total <= 0 || k <= 0) return res;
    int correct = 0;
    for (int i = 0; i < total; i++) if (correct_samples[i]) correct++;
    double denom = n_choose_k(total, k);
    if (denom < 1e-10) return res;
    double numer = n_choose_k(total - correct, k);
    res.pass_at_k = 1.0 - numer / denom;
    if (res.pass_at_k < 0.0) res.pass_at_k = 0.0;
    if (res.pass_at_k > 1.0) res.pass_at_k = 1.0;
    return res;
}

/* Composite evaluation */

evaluation_pair_t* metrics_evaluate_pair(const char *reference, const char *candidate) {
    evaluation_pair_t *pair = (evaluation_pair_t*)calloc(1, sizeof(evaluation_pair_t));
    if (!pair) return NULL;
    pair->reference = reference;
    pair->candidate = candidate;
    pair->results[0] = metrics_evaluate_bleu(reference, candidate, 4);
    pair->results[1] = metrics_evaluate_rouge_l(reference, candidate);
    pair->results[2] = metrics_evaluate_exact_match(reference, candidate);
    pair->results[3] = metrics_evaluate_f1(reference, candidate);
    pair->result_count = 4;
    return pair;
}

void metrics_evaluation_pair_free(evaluation_pair_t *pair) {
    if (pair) free(pair);
}

/* Benchmark suite */

benchmark_suite_t* metrics_benchmark_create(const char *dataset_name, int capacity) {
    benchmark_suite_t *suite = (benchmark_suite_t*)calloc(1, sizeof(benchmark_suite_t));
    if (!suite) return NULL;
    suite->capacity = capacity > 0 ? capacity : METRICS_MAX_BENCHMARKS;
    suite->pairs = (evaluation_pair_t*)calloc((size_t)suite->capacity, sizeof(evaluation_pair_t));
    if (!suite->pairs) { free(suite); return NULL; }
    if (dataset_name) strncpy(suite->dataset_name, dataset_name, sizeof(suite->dataset_name) - 1);
    return suite;
}

void metrics_benchmark_destroy(benchmark_suite_t *suite) {
    if (!suite) return;
    free(suite->pairs);
    free(suite);
}

int metrics_benchmark_add(benchmark_suite_t *suite, const char *reference, const char *candidate) {
    if (!suite || !reference || !candidate || suite->pair_count >= suite->capacity) return 0;
    evaluation_pair_t *ep = metrics_evaluate_pair(reference, candidate);
    if (!ep) return 0;
    suite->pairs[suite->pair_count] = *ep;
    suite->pair_count++;
    free(ep);
    return 1;
}

void metrics_benchmark_compute(benchmark_suite_t *suite) {
    if (!suite || suite->pair_count == 0) return;
    double s_bleu = 0.0, s_rouge = 0.0, s_f1 = 0.0;
    int em_cnt = 0;
    for (int i = 0; i < suite->pair_count; i++) {
        s_bleu  += suite->pairs[i].results[0].bleu_score;
        s_rouge += suite->pairs[i].results[1].rouge_l_f1;
        if (suite->pairs[i].results[2].exact_match) em_cnt++;
        s_f1    += suite->pairs[i].results[3].f1_score;
    }
    suite->aggregate_bleu = s_bleu / suite->pair_count;
    suite->aggregate_rouge_l = s_rouge / suite->pair_count;
    suite->aggregate_exact_match_rate = (double)em_cnt / suite->pair_count;
    suite->aggregate_f1 = s_f1 / suite->pair_count;
}

const char* metrics_benchmark_report(const benchmark_suite_t *suite, char *buf, size_t buf_size) {
    if (!suite || !buf) return NULL;
    snprintf(buf, buf_size,
        "=== Benchmark: %s ===\n"
        "  Pairs: %d | BLEU-4: %.4f | ROUGE-L: %.4f | EM: %.4f | F1: %.4f\n",
        suite->dataset_name, suite->pair_count,
        suite->aggregate_bleu, suite->aggregate_rouge_l,
        suite->aggregate_exact_match_rate, suite->aggregate_f1);
    return buf;
}

/* Bootstrap CI: non-parametric bootstrap (Efron 1979)
 * Algorithm: resample with replacement B times, compute quantiles */

static int cmp_dbl_asc(const void *a, const void *b) {
    double da = *(const double*)a, db = *(const double*)b;
    return (da > db) - (da < db);
}

metrics_confidence_interval_t metrics_bootstrap_ci(const double *scores, int n,
                                                     int boot_iter, double confidence) {
    metrics_confidence_interval_t ci;
    memset(&ci, 0, sizeof(ci));
    ci.bootstrap_samples = boot_iter;
    ci.confidence_level = confidence;
    if (!scores || n <= 0 || boot_iter <= 0) return ci;
    double sum = 0.0, sum_sq = 0.0;
    for (int i = 0; i < n; i++) { sum += scores[i]; sum_sq += scores[i] * scores[i]; }
    ci.mean = sum / n;
    ci.std_dev = (n > 1) ? sqrt((sum_sq - sum * sum / n) / (n - 1)) : 0.0;
    double *boot_means = (double*)malloc((size_t)boot_iter * sizeof(double));
    if (!boot_means) return ci;
    unsigned long long seed = 12345ULL;
    for (int b = 0; b < boot_iter; b++) {
        double bsum = 0.0;
        for (int i = 0; i < n; i++) {
            seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
            int idx = (int)(seed % (unsigned)n);
            bsum += scores[idx];
        }
        boot_means[b] = bsum / n;
    }
    qsort(boot_means, (size_t)boot_iter, sizeof(double), cmp_dbl_asc);
    double alpha = (1.0 - confidence) / 2.0;
    int lo = (int)(alpha * boot_iter);
    int hi = (int)((1.0 - alpha) * boot_iter);
    if (lo < 0) lo = 0;
    if (hi >= boot_iter) hi = boot_iter - 1;
    ci.ci_lower = boot_means[lo];
    ci.ci_upper = boot_means[hi];
    free(boot_means);
    return ci;
}

/* Paired Bootstrap Test: H0: mean_A == mean_B */

metrics_system_comparison_t metrics_compare_systems(const double *sys_a, const double *sys_b,
                                                      int n, int boot_iter) {
    metrics_system_comparison_t comp;
    memset(&comp, 0, sizeof(comp));
    comp.bootstrap_iterations = boot_iter;
    if (!sys_a || !sys_b || n <= 0 || boot_iter <= 0) return comp;
    double sa = 0.0, sb = 0.0;
    for (int i = 0; i < n; i++) { sa += sys_a[i]; sb += sys_b[i]; }
    double ma = sa / n, mb = sb / n;
    comp.delta_mean = ma - mb;
    double *deltas = (double*)malloc((size_t)boot_iter * sizeof(double));
    if (!deltas) return comp;
    unsigned long long seed = 67890ULL;
    for (int b = 0; b < boot_iter; b++) {
        double bsa = 0.0, bsb = 0.0;
        for (int i = 0; i < n; i++) {
            seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
            int idx = (int)(seed % (unsigned)n);
            bsa += sys_a[idx];
            bsb += sys_b[idx];
        }
        deltas[b] = (bsa / n - bsb / n) - comp.delta_mean;
    }
    int extreme = 0;
    double abs_d = fabs(comp.delta_mean);
    for (int b = 0; b < boot_iter; b++)
        if (fabs(deltas[b]) >= abs_d) extreme++;
    comp.p_value = (double)extreme / (double)boot_iter;
    comp.significant = (comp.p_value < 0.05);
    free(deltas);
    return comp;
}

const char* metrics_type_name(metric_type_t type) {
    switch (type) {
        case METRIC_BLEU: return "BLEU";
        case METRIC_ROUGE_L: return "ROUGE-L";
        case METRIC_EXACT_MATCH: return "Exact Match";
        case METRIC_F1: return "F1";
        case METRIC_PASS_AT_K: return "Pass@k";
        case METRIC_CUSTOM: return "Custom";
        default: return "Unknown";
    }
}

const char* metrics_format_result(const metric_result_t *r, char *buf, size_t sz) {
    if (!r || !buf) return NULL;
    switch (r->type) {
        case METRIC_BLEU:
            snprintf(buf, sz, "%s: %.4f", r->name, r->bleu_score); break;
        case METRIC_ROUGE_L:
            snprintf(buf, sz, "%s: P=%.4f R=%.4f F1=%.4f",
                     r->name, r->rouge_l_precision, r->rouge_l_recall, r->rouge_l_f1); break;
        case METRIC_EXACT_MATCH:
            snprintf(buf, sz, "%s: %s", r->name, r->exact_match ? "true" : "false"); break;
        case METRIC_F1:
            snprintf(buf, sz, "%s: P=%.4f R=%.4f F1=%.4f",
                     r->name, r->f1_precision, r->f1_recall, r->f1_score); break;
        case METRIC_PASS_AT_K:
            snprintf(buf, sz, "%s: %.4f", r->name, r->pass_at_k); break;
        default:
            snprintf(buf, sz, "%s: N/A", r->name); break;
    }
    return buf;
}
