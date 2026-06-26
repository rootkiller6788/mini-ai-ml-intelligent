#ifndef EVAL_MONITOR_H
#define EVAL_MONITOR_H

#include <stddef.h>
#include <stdint.h>

#define EV_MAX_TEST_CASES   10000
#define EV_MAX_CLASSES      100
#define EV_MAX_LABEL_LEN    256
#define EV_MAX_TEXT_LEN     4096
#define EV_MAX_DRIFT_WINDOW 1000
#define EV_DASHBOARD_METRICS 32

typedef struct {
    int32_t id;
    char    input[EV_MAX_TEXT_LEN];
    char    expected[EV_MAX_LABEL_LEN];
    char    prediction[EV_MAX_LABEL_LEN];
    int32_t class_label;
    int32_t pred_label;
} EVTestCase;

typedef struct {
    EVTestCase cases[EV_MAX_TEST_CASES];
    int32_t     num_cases;
} EVGoldenDataset;

typedef struct {
    double accuracy;
    double precision;
    double recall;
    double f1_score;
    int32_t confusion[EV_MAX_CLASSES][EV_MAX_CLASSES];
    int32_t num_classes;
} EVClassificationMetrics;

typedef struct {
    double bleu_1gram;
    double bleu_2gram;
    double bleu_3gram;
    double bleu_4gram;
    double bleu_avg;
    double rouge_1;
    double rouge_2;
    double rouge_l;
    double avg_len_ratio;
} EVNLGMetrics;

typedef struct {
    double   accuracy;
    double   f1;
    EVNLGMetrics nlg;
    double   avg_latency_ms;
    double   p50_latency_ms;
    double   p99_latency_ms;
    double   throughput_rps;
    int32_t  total_cases;
} EVOfflineResult;

typedef struct {
    char    variant[64];
    int32_t num_users;
    double  engagement_rate;
    double  click_through_rate;
    double  task_completion_rate;
    double  avg_session_time_s;
    double  retention_day7;
    double  bounce_rate;
    double  user_satisfaction;
    int32_t thumbs_up;
    int32_t thumbs_down;
} EVOnlineMetric;

typedef struct {
    EVOnlineMetric control;
    EVOnlineMetric treatment;
    double         p_value;
    int            significant;
} EVABExperiment;

typedef enum {
    EV_JUDGE_HELPFULNESS,
    EV_JUDGE_CORRECTNESS,
    EV_JUDGE_SAFETY,
    EV_JUDGE_COHERENCE
} EVJudgeDimension;

typedef struct {
    int32_t test_case_id;
    char    response[EV_MAX_TEXT_LEN];
    char    reference[EV_MAX_TEXT_LEN];
    float   scores[4];
    char    explanation[EV_MAX_TEXT_LEN];
} EVLLMJudgeResult;

typedef struct {
    double mean;
    double std;
    double min;
    double max;
    double median;
    double p25;
    double p75;
} EVDataDriftStats;

typedef struct {
    EVDataDriftStats reference;
    EVDataDriftStats current;
    double           psi;
    double           ks_statistic;
    int              drift_detected;
} EVDriftReport;

typedef struct {
    char    name[64];
    double  current_value;
    double  baseline_value;
    double  change_pct;
    int64_t timestamp;
} EVDashboardMetric;

typedef struct {
    EVDashboardMetric metrics[EV_DASHBOARD_METRICS];
    int32_t           num_metrics;
    int64_t           last_updated;
} EVDashboard;

typedef struct {
    int32_t user_id;
    int32_t test_case_id;
    int     thumbs_up;
    char    comment[EV_MAX_TEXT_LEN];
} EVFeedbackItem;

typedef struct {
    EVFeedbackItem items[EV_MAX_TEST_CASES];
    int32_t        num_items;
} EVFeedbackCollection;

void ev_golden_dataset_init(EVGoldenDataset *ds);

void ev_classification_eval(const int32_t *y_true, const int32_t *y_pred,
                            int32_t n, int32_t n_classes,
                            EVClassificationMetrics *metrics);
void ev_nlg_eval(const char **references, const char **predictions,
                 int32_t n, EVNLGMetrics *metrics);

double ev_accuracy(const int32_t *y_true, const int32_t *y_pred, int32_t n);
double ev_precision_macro(const int32_t *y_true, const int32_t *y_pred,
                          int32_t n, int32_t n_classes);
double ev_recall_macro(const int32_t *y_true, const int32_t *y_pred,
                       int32_t n, int32_t n_classes);
double ev_f1_score(double precision, double recall);

double ev_bleu_n(const char *ref, const char *pred, int n);
double ev_rouge_n(const char *ref, const char *pred, int n);
double ev_rouge_l(const char *ref, const char *pred);

void ev_offline_evaluate(const EVGoldenDataset *ds,
                         void (*predict_fn)(const char *, char *, int32_t),
                         EVOfflineResult *result);

void ev_online_metric_init(EVOnlineMetric *m, const char *variant);
void ev_online_collect(EVOnlineMetric *m, double engagement, double ctr,
                       double completion, double session_time, int retained,
                       int bounced);
void ev_ab_experiment_compare(const EVOnlineMetric *ctrl,
                              const EVOnlineMetric *treat,
                              EVABExperiment *exp);

void ev_llm_judge_init(EVLLMJudgeResult *judge);
void ev_llm_judge_evaluate(EVLLMJudgeResult *judge,
                           const char *response, const char *reference,
                           EVJudgeDimension dim);

void ev_metric_baseline_record(EVGoldenDataset *ds, EVOfflineResult *baseline);
int  ev_regression_detect(const EVOfflineResult *baseline,
                          const EVOfflineResult *current,
                          double threshold);
void ev_regression_alert(const EVOfflineResult *current,
                         const char *metric_name, double drop_pct);

void ev_data_drift_detect(const double *reference_data, int32_t ref_len,
                          const double *current_data, int32_t cur_len,
                          int32_t num_bins, EVDriftReport *report);
double ev_psi_compute(const double *ref_dist, const double *cur_dist,
                      int32_t num_bins);
double ev_ks_test(const double *ref_data, int32_t ref_len,
                  const double *cur_data, int32_t cur_len);
void ev_prediction_drift_monitor(const double *ref_preds, int32_t ref_len,
                                 const double *cur_preds, int32_t cur_len,
                                 int32_t window_size, EVDriftReport *report);

void ev_dashboard_init(EVDashboard *dash);
void ev_dashboard_update(EVDashboard *dash, const char *name,
                         double current, double baseline);
void ev_dashboard_render(const EVDashboard *dash, char *output, int32_t max_len);

void ev_feedback_collect(EVFeedbackCollection *fb, int32_t user_id,
                         int32_t test_case_id, int thumbs_up, const char *comment);
void ev_feedback_summarize(const EVFeedbackCollection *fb,
                           int32_t *total, int32_t *up, int32_t *down,
                           double *satisfaction);

#endif
