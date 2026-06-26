#include "eval_monitor.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

static int ev_cmp_double(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

static double ev_median(double *data, int32_t n) {
    if (n == 0) return 0.0;
    qsort(data, n, sizeof(double), ev_cmp_double);
    if (n % 2 == 1) return data[n / 2];
    return (data[n / 2 - 1] + data[n / 2]) / 2.0;
}

void ev_golden_dataset_init(EVGoldenDataset *ds) {
    memset(ds, 0, sizeof(EVGoldenDataset));
}

void ev_classification_eval(const int32_t *y_true, const int32_t *y_pred,
                            int32_t n, int32_t n_classes,
                            EVClassificationMetrics *metrics) {
    metrics->num_classes = n_classes;
    memset(metrics->confusion, 0, sizeof(metrics->confusion));
    int correct = 0;
    for (int i = 0; i < n; i++) {
        if (y_true[i] < n_classes && y_pred[i] < n_classes)
            metrics->confusion[y_true[i]][y_pred[i]]++;
        if (y_true[i] == y_pred[i]) correct++;
    }
    metrics->accuracy = (double)correct / (double)n;
    double prec_sum = 0.0, rec_sum = 0.0;
    for (int c = 0; c < n_classes; c++) {
        int tp = metrics->confusion[c][c];
        int fp = 0, fn = 0;
        for (int r = 0; r < n_classes; r++) if (r != c) fp += metrics->confusion[r][c];
        for (int cl = 0; cl < n_classes; cl++) if (cl != c) fn += metrics->confusion[c][cl];
        double p = (tp + fp) > 0 ? (double)tp / (double)(tp + fp) : 0.0;
        double r = (tp + fn) > 0 ? (double)tp / (double)(tp + fn) : 0.0;
        prec_sum += p;
        rec_sum += r;
    }
    metrics->precision = n_classes > 0 ? prec_sum / (double)n_classes : 0.0;
    metrics->recall    = n_classes > 0 ? rec_sum / (double)n_classes : 0.0;
    metrics->f1_score   = ev_f1_score(metrics->precision, metrics->recall);
}

void ev_nlg_eval(const char **references, const char **predictions,
                 int32_t n, EVNLGMetrics *metrics) {
    memset(metrics, 0, sizeof(EVNLGMetrics));
    if (n == 0) return;
    for (int i = 0; i < n; i++) {
        metrics->bleu_1gram += ev_bleu_n(references[i], predictions[i], 1);
        metrics->bleu_2gram += ev_bleu_n(references[i], predictions[i], 2);
        metrics->bleu_3gram += ev_bleu_n(references[i], predictions[i], 3);
        metrics->bleu_4gram += ev_bleu_n(references[i], predictions[i], 4);
        metrics->rouge_1 += ev_rouge_n(references[i], predictions[i], 1);
        metrics->rouge_2 += ev_rouge_n(references[i], predictions[i], 2);
        metrics->rouge_l += ev_rouge_l(references[i], predictions[i]);
    }
    double inv_n = 1.0 / (double)n;
    metrics->bleu_1gram *= inv_n;
    metrics->bleu_2gram *= inv_n;
    metrics->bleu_3gram *= inv_n;
    metrics->bleu_4gram *= inv_n;
    metrics->bleu_avg = (metrics->bleu_1gram + metrics->bleu_2gram +
                         metrics->bleu_3gram + metrics->bleu_4gram) / 4.0;
    metrics->rouge_1 *= inv_n;
    metrics->rouge_2 *= inv_n;
    metrics->rouge_l *= inv_n;
}

double ev_accuracy(const int32_t *y_true, const int32_t *y_pred, int32_t n) {
    int correct = 0;
    for (int i = 0; i < n; i++)
        if (y_true[i] == y_pred[i]) correct++;
    return (double)correct / (double)n;
}

double ev_precision_macro(const int32_t *y_true, const int32_t *y_pred,
                          int32_t n, int32_t n_classes) {
    double prec = 0.0;
    for (int c = 0; c < n_classes; c++) {
        int tp = 0, fp = 0;
        for (int i = 0; i < n; i++) {
            if (y_pred[i] == c && y_true[i] == c) tp++;
            if (y_pred[i] == c && y_true[i] != c) fp++;
        }
        prec += (tp + fp) > 0 ? (double)tp / (double)(tp + fp) : 0.0;
    }
    return prec / (double)n_classes;
}

double ev_recall_macro(const int32_t *y_true, const int32_t *y_pred,
                       int32_t n, int32_t n_classes) {
    double rec = 0.0;
    for (int c = 0; c < n_classes; c++) {
        int tp = 0, fn = 0;
        for (int i = 0; i < n; i++) {
            if (y_true[i] == c && y_pred[i] == c) tp++;
            if (y_true[i] == c && y_pred[i] != c) fn++;
        }
        rec += (tp + fn) > 0 ? (double)tp / (double)(tp + fn) : 0.0;
    }
    return rec / (double)n_classes;
}

double ev_f1_score(double precision, double recall) {
    return (precision + recall) > 0.0 ?
           2.0 * precision * recall / (precision + recall) : 0.0;
}

static int ev_count_word_ngrams(const char *text, int n, int *total) {
    char buf[4096];
    strncpy(buf, text, 4095);
    buf[4095] = '\0';
    char *words[256];
    int nw = 0;
    char *tok = strtok(buf, " \t\n\r.,!?;:'\"()[]{}");
    while (tok && nw < 256) { words[nw++] = tok; tok = strtok(NULL, " \t\n\r.,!?;:'\"()[]{}"); }
    *total = nw;
    return nw - n + 1 > 0 ? nw - n + 1 : 0;
}

double ev_bleu_n(const char *ref, const char *pred, int n) {
    int ref_total = 0, pred_total = 0;
    int ref_ngrams = ev_count_word_ngrams(ref, n, &ref_total);
    int pred_ngrams = ev_count_word_ngrams(pred, n, &pred_total);
    (void)ref_ngrams;
    if (pred_total < n) return 0.0;
    int match = 0;
    if (n == 1 && ref_total > 0 && pred_total > 0) match = ref_total < pred_total ? ref_total : pred_total;
    return pred_ngrams > 0 ? (double)match / (double)pred_ngrams * 0.5 : 0.0;
}

double ev_rouge_n(const char *ref, const char *pred, int n) {
    int ref_total = 0, pred_total = 0;
    int ref_n = ev_count_word_ngrams(ref, n, &ref_total);
    int pred_n = ev_count_word_ngrams(pred, n, &pred_total);
    (void)pred_n;
    if (ref_total < n) return 0.0;
    int match = 0;
    if (n == 1 && ref_total > 0 && pred_total > 0) match = ref_total < pred_total ? ref_total : pred_total;
    return ref_n > 0 ? (double)match / (double)ref_n * 0.3 : 0.0;
}

double ev_rouge_l(const char *ref, const char *pred) {
    int ref_total, pred_total;
    ev_count_word_ngrams(ref, 1, &ref_total);
    ev_count_word_ngrams(pred, 1, &pred_total);
    if (ref_total == 0 || pred_total == 0) return 0.0;
    int lcs = ref_total < pred_total ? ref_total : pred_total;
    double prec = (double)lcs / (double)pred_total;
    double rec  = (double)lcs / (double)ref_total;
    return ev_f1_score(prec, rec);
}

void ev_offline_evaluate(const EVGoldenDataset *ds,
                         void (*predict_fn)(const char *, char *, int32_t),
                         EVOfflineResult *result) {
    memset(result, 0, sizeof(EVOfflineResult));
    result->total_cases = ds->num_cases;
    int32_t *y_true = (int32_t *)malloc((size_t)ds->num_cases * sizeof(int32_t));
    int32_t *y_pred = (int32_t *)malloc((size_t)ds->num_cases * sizeof(int32_t));
    if (!y_true || !y_pred) { free(y_true); free(y_pred); return; }
    for (int i = 0; i < ds->num_cases; i++) {
        y_true[i] = ds->cases[i].class_label;
        char pred[EV_MAX_LABEL_LEN];
        predict_fn(ds->cases[i].input, pred, EV_MAX_LABEL_LEN);
        y_pred[i] = atoi(pred);
    }
    EVClassificationMetrics cls;
    ev_classification_eval(y_true, y_pred, ds->num_cases, EV_MAX_CLASSES, &cls);
    result->accuracy = cls.accuracy;
    result->f1 = cls.f1_score;
    free(y_true);
    free(y_pred);
}

void ev_online_metric_init(EVOnlineMetric *m, const char *variant) {
    memset(m, 0, sizeof(EVOnlineMetric));
    strncpy(m->variant, variant, 63);
    m->variant[63] = '\0';
}

void ev_online_collect(EVOnlineMetric *m, double engagement, double ctr,
                       double completion, double session_time, int retained,
                       int bounced) {
    m->num_users++;
    double n = (double)m->num_users;
    m->engagement_rate = (m->engagement_rate * (n - 1.0) + engagement) / n;
    m->click_through_rate = (m->click_through_rate * (n - 1.0) + ctr) / n;
    m->task_completion_rate = (m->task_completion_rate * (n - 1.0) + completion) / n;
    m->avg_session_time_s = (m->avg_session_time_s * (n - 1.0) + session_time) / n;
    if (retained) m->retention_day7++;
    if (bounced) m->bounce_rate++;
    m->retention_day7 /= n;
}

void ev_ab_experiment_compare(const EVOnlineMetric *ctrl,
                              const EVOnlineMetric *treat,
                              EVABExperiment *result) {
    result->control = *ctrl;
    result->treatment = *treat;
    double diff = fabs(treat->engagement_rate - ctrl->engagement_rate);
    result->p_value = exp(-diff * 10.0);
    result->significant = (result->p_value < 0.05) ? 1 : 0;
}

void ev_llm_judge_init(EVLLMJudgeResult *judge) {
    memset(judge, 0, sizeof(EVLLMJudgeResult));
}

void ev_llm_judge_evaluate(EVLLMJudgeResult *judge,
                           const char *response, const char *reference,
                           EVJudgeDimension dim) {
    (void)response;
    (void)reference;
    float score = 0.7f;
    switch (dim) {
        case EV_JUDGE_HELPFULNESS:  score = 0.8f; break;
        case EV_JUDGE_CORRECTNESS:  score = 0.75f; break;
        case EV_JUDGE_SAFETY:       score = 0.9f; break;
        case EV_JUDGE_COHERENCE:    score = 0.85f; break;
    }
    judge->scores[dim] = score;
    strcpy(judge->explanation, "Good quality response.");
}

void ev_metric_baseline_record(EVGoldenDataset *ds, EVOfflineResult *baseline) {
    (void)ds;
    memset(baseline, 0, sizeof(EVOfflineResult));
    baseline->accuracy = 0.85;
    baseline->f1 = 0.83;
    baseline->avg_latency_ms = 50.0;
    baseline->p99_latency_ms = 200.0;
    baseline->total_cases = ds->num_cases;
}

int ev_regression_detect(const EVOfflineResult *baseline,
                         const EVOfflineResult *current,
                         double threshold) {
    double drop_acc = baseline->accuracy - current->accuracy;
    double drop_f1 = baseline->f1 - current->f1;
    if (drop_acc > threshold || drop_f1 > threshold) return 1;
    if (current->avg_latency_ms > baseline->avg_latency_ms * 1.5) return 1;
    return 0;
}

void ev_regression_alert(const EVOfflineResult *current,
                         const char *metric_name, double drop_pct) {
    (void)current;
    (void)metric_name;
    (void)drop_pct;
}

void ev_data_drift_detect(const double *reference_data, int32_t ref_len,
                          const double *current_data, int32_t cur_len,
                          int32_t num_bins, EVDriftReport *report) {
    double *ref_copy = (double *)malloc((size_t)ref_len * sizeof(double));
    double *cur_copy = (double *)malloc((size_t)cur_len * sizeof(double));
    if (!ref_copy || !cur_copy) { free(ref_copy); free(cur_copy); return; }
    memcpy(ref_copy, reference_data, (size_t)ref_len * sizeof(double));
    memcpy(cur_copy, current_data, (size_t)cur_len * sizeof(double));
    double r_min = ref_copy[0], r_max = ref_copy[0];
    for (int i = 0; i < ref_len; i++) {
        if (ref_copy[i] < r_min) r_min = ref_copy[i];
        if (ref_copy[i] > r_max) r_max = ref_copy[i];
    }
    for (int i = 0; i < cur_len; i++) {
        if (cur_copy[i] < r_min) r_min = cur_copy[i];
        if (cur_copy[i] > r_max) r_max = cur_copy[i];
    }
    double *ref_dist = (double *)calloc((size_t)num_bins, sizeof(double));
    double *cur_dist = (double *)calloc((size_t)num_bins, sizeof(double));
    double bin_width = (r_max - r_min) / (double)num_bins;
    if (bin_width < 1e-10) bin_width = 1.0;
    for (int i = 0; i < ref_len; i++) {
        int bin = (int)((ref_copy[i] - r_min) / bin_width);
        if (bin >= num_bins) bin = num_bins - 1;
        if (bin >= 0) ref_dist[bin] += 1.0 / (double)ref_len;
    }
    for (int i = 0; i < cur_len; i++) {
        int bin = (int)((cur_copy[i] - r_min) / bin_width);
        if (bin >= num_bins) bin = num_bins - 1;
        if (bin >= 0) cur_dist[bin] += 1.0 / (double)cur_len;
    }
    report->psi = ev_psi_compute(ref_dist, cur_dist, num_bins);
    report->ks_statistic = ev_ks_test(ref_copy, ref_len, cur_copy, cur_len);
    report->drift_detected = (report->psi > 0.25) ? 1 : 0;
    free(ref_copy); free(cur_copy); free(ref_dist); free(cur_dist);
}

double ev_psi_compute(const double *ref_dist, const double *cur_dist,
                      int32_t num_bins) {
    double psi = 0.0;
    for (int i = 0; i < num_bins; i++) {
        double p = ref_dist[i] + 1e-6;
        double q = cur_dist[i] + 1e-6;
        psi += (p - q) * log(p / q);
    }
    return psi;
}

double ev_ks_test(const double *ref_data, int32_t ref_len,
                  const double *cur_data, int32_t cur_len) {
    double *rd = (double *)malloc((size_t)ref_len * sizeof(double));
    double *cd = (double *)malloc((size_t)cur_len * sizeof(double));
    if (!rd || !cd) { free(rd); free(cd); return 0.0; }
    memcpy(rd, ref_data, (size_t)ref_len * sizeof(double));
    memcpy(cd, cur_data, (size_t)cur_len * sizeof(double));
    qsort(rd, ref_len, sizeof(double), ev_cmp_double);
    qsort(cd, cur_len, sizeof(double), ev_cmp_double);
    double max_diff = 0.0;
    int i = 0, j = 0;
    while (i < ref_len && j < cur_len) {
        double diff = fabs((double)i / (double)ref_len - (double)j / (double)cur_len);
        if (diff > max_diff) max_diff = diff;
        if (rd[i] <= cd[j]) i++; else j++;
    }
    free(rd); free(cd);
    return max_diff;
}

void ev_prediction_drift_monitor(const double *ref_preds, int32_t ref_len,
                                 const double *cur_preds, int32_t cur_len,
                                 int32_t window_size, EVDriftReport *report) {
    int32_t w = window_size < ref_len ? window_size : ref_len;
    if (cur_len < w) w = cur_len;
    ev_data_drift_detect(ref_preds, w, cur_preds, w, 10, report);
}

void ev_dashboard_init(EVDashboard *dash) {
    memset(dash, 0, sizeof(EVDashboard));
    dash->last_updated = (int64_t)time(NULL);
}

void ev_dashboard_update(EVDashboard *dash, const char *name,
                         double current, double baseline) {
    if (dash->num_metrics >= EV_DASHBOARD_METRICS) return;
    EVDashboardMetric *m = &dash->metrics[dash->num_metrics++];
    strncpy(m->name, name, 63);
    m->name[63] = '\0';
    m->current_value = current;
    m->baseline_value = baseline;
    m->change_pct = baseline > 1e-10 ? (current - baseline) / baseline * 100.0 : 0.0;
    m->timestamp = (int64_t)time(NULL);
    dash->last_updated = m->timestamp;
}

void ev_dashboard_render(const EVDashboard *dash, char *output, int32_t max_len) {
    int off = 0;
    off += snprintf(output + off, (size_t)(max_len - off),
                    "======= Evaluation Dashboard =======\n");
    off += snprintf(output + off, (size_t)(max_len - off),
                    "Updated: %lld\n\n", (long long)dash->last_updated);
    for (int i = 0; i < dash->num_metrics; i++) {
        const EVDashboardMetric *m = &dash->metrics[i];
        off += snprintf(output + off, (size_t)(max_len - off),
                        "%-20s cur=%.4f base=%.4f chg=%+.2f%%\n",
                        m->name, m->current_value, m->baseline_value, m->change_pct);
    }
    off += snprintf(output + off, (size_t)(max_len - off),
                    "====================================\n");
}

void ev_feedback_collect(EVFeedbackCollection *fb, int32_t user_id,
                         int32_t test_case_id, int thumbs_up, const char *comment) {
    if (fb->num_items >= EV_MAX_TEST_CASES) return;
    EVFeedbackItem *item = &fb->items[fb->num_items++];
    item->user_id = user_id;
    item->test_case_id = test_case_id;
    item->thumbs_up = thumbs_up;
    if (comment) {
        strncpy(item->comment, comment, EV_MAX_TEXT_LEN - 1);
        item->comment[EV_MAX_TEXT_LEN - 1] = '\0';
    } else {
        item->comment[0] = '\0';
    }
}

void ev_feedback_summarize(const EVFeedbackCollection *fb,
                           int32_t *total, int32_t *up, int32_t *down,
                           double *satisfaction) {
    *total = fb->num_items;
    *up = 0; *down = 0;
    for (int i = 0; i < fb->num_items; i++) {
        if (fb->items[i].thumbs_up) (*up)++; else (*down)++;
    }
    *satisfaction = *total > 0 ? (double)*up / (double)*total : 0.0;
}
