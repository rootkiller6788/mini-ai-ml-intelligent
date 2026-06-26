#ifndef MODEL_AB_TEST_H
#define MODEL_AB_TEST_H

#include <stddef.h>
#include <stdint.h>
#include <time.h>

#define MAB_MAX_EXPERIMENTS   64
#define MAB_MAX_VARIANTS      8
#define MAB_MAX_METRICS       32
#define MAB_MAX_STAGES        8
#define MAB_HASH_SEED         0xDEADBEEF

typedef enum {
    MAB_VARIANT_CONTROL,
    MAB_VARIANT_TREATMENT,
    MAB_VARIANT_SHADOW
} MABVariantType;

typedef struct {
    char          name[64];
    MABVariantType type;
    int32_t       traffic_pct;
    void         *model_ptr;
} MABVariant;

typedef struct {
    char       name[128];
    MABVariant variants[MAB_MAX_VARIANTS];
    int32_t    num_variants;
    int64_t    start_time;
    int64_t    end_time;
    int        is_active;
    uint64_t   hash_seed;
} MABExperiment;

typedef struct {
    double latency_ms;
    double success_rate;
    double user_satisfaction;
    double engagement;
    double ctr;
    double completion_rate;
    double error_rate;
    double avg_tokens;
    int32_t total_requests;
} MABVariantMetrics;

typedef struct {
    MABVariantMetrics per_variant[MAB_MAX_VARIANTS];
    int32_t           num_variants;
} MABExperimentMetrics;

typedef struct {
    double t_statistic;
    double p_value;
    double effect_size;
    double ci_lower;
    double ci_upper;
    int    significant;
    int    df;
} MABTTestResult;

typedef struct {
    double chi2_stat;
    double p_value;
    int    df;
    int    significant;
    int32_t observed_ctrl;
    int32_t observed_treat;
    int32_t total_ctrl;
    int32_t total_treat;
} MABChiSquareResult;

typedef enum {
    MAB_STAGE_INIT,
    MAB_STAGE_DEV,
    MAB_STAGE_STAGING,
    MAB_STAGE_CANARY,
    MAB_STAGE_PRODUCTION,
    MAB_STAGE_DEPRECATED,
    MAB_STAGE_ARCHIVED
} MABModelStage;

typedef struct {
    char         name[128];
    char         version[32];
    MABModelStage stage;
    int64_t       promoted_at;
    int64_t       last_served;
    int32_t       request_count;
    void        *model_ptr;
} MABModelRegistry;

typedef struct {
    char         name[64];
    float        threshold;
    int          is_critical;
} MABGuardrailMetric;

typedef struct {
    MABGuardrailMetric guardrails[MAB_MAX_METRICS];
    int32_t            num_guardrails;
    double             current_values[MAB_MAX_METRICS];
} MABGuardrails;

typedef struct {
    int32_t initial_pct;
    int32_t ramp_steps[4];
    int32_t ramp_sizes[4];
    int32_t num_steps;
    int64_t step_duration_s;
} MABRampUpPlan;

typedef struct {
    int     should_rollback;
    char    reason[256];
    double  degradation_pct;
    int     auto_rollback;
} MABRollbackDecision;

typedef struct {
    MABVariantMetrics shadow_metrics;
    double            latency_ms;
    double            error_rate;
    int32_t           total_requests;
    int               passed;
    double            score_delta;
} MABShadowTest;

uint64_t mab_user_hash(int32_t user_id, uint64_t seed);
int      mab_traffic_assign(int32_t user_id, const MABExperiment *exp);

void     mab_experiment_init(MABExperiment *exp, const char *name, uint64_t seed);
void     mab_experiment_add_variant(MABExperiment *exp, const char *name,
                                    MABVariantType type, int32_t traffic_pct,
                                    void *model_ptr);
int      mab_experiment_validate(const MABExperiment *exp);
void     mab_experiment_activate(MABExperiment *exp);
void     mab_experiment_deactivate(MABExperiment *exp);

void     mab_metrics_init(MABVariantMetrics *m);
void     mab_metrics_record_request(MABVariantMetrics *m, double latency,
                                    int success, double satisfaction);
void     mab_metrics_collect(const MABExperiment *exp,
                             const MABVariantMetrics *per_variant,
                             MABExperimentMetrics *result);
void     mab_metrics_compare(const MABVariantMetrics *ctrl,
                             const MABVariantMetrics *treat,
                             const char *metric_name,
                             MABTTestResult *result);

void     mab_ttest_independent(const double *a, int32_t na,
                               const double *b, int32_t nb,
                               MABTTestResult *result);
void     mab_chi_square_test(double ctrl_rate, int32_t ctrl_n,
                             double treat_rate, int32_t treat_n,
                             MABChiSquareResult *result);
int      mab_significance_check(const MABTTestResult *t, double alpha);
int      mab_sample_size_estimate(double baseline_rate, double min_detectable_effect,
                                  double alpha, double power);

void     mab_ramp_up_init(MABRampUpPlan *plan, int32_t initial, int64_t step_dur);
void     mab_ramp_up_step(const MABRampUpPlan *plan, int32_t step_idx,
                          MABExperiment *exp);
void     mab_ramp_up_execute(MABExperiment *exp, const MABRampUpPlan *plan,
                             const MABGuardrails *guardrails,
                             MABRollbackDecision *decision);

void     mab_rollback_evaluate(const MABVariantMetrics *ctrl,
                               const MABVariantMetrics *treat,
                               const MABGuardrails *guardrails,
                               MABRollbackDecision *decision);
void     mab_rollback_execute(MABExperiment *exp);

void     mab_model_registry_init(MABModelRegistry *reg, const char *name,
                                 const char *version, MABModelStage stage,
                                 void *model_ptr);
void     mab_model_registry_promote(MABModelRegistry *reg, MABModelStage next);
void     mab_model_registry_archive(MABModelRegistry *reg);
int      mab_model_registry_can_serve(const MABModelRegistry *reg);

void     mab_guardrails_init(MABGuardrails *g);
void     mab_guardrails_add(MABGuardrails *g, const char *name,
                            float threshold, int critical);
int      mab_guardrails_check(const MABGuardrails *g,
                              const MABVariantMetrics *m);
void     mab_guardrails_alert(const MABGuardrails *g, const char *variant_name,
                              char *alert_msg, int32_t max_len);

void     mab_shadow_test_init(MABShadowTest *st);
void     mab_shadow_test_record(MABShadowTest *st, double prod_score,
                                double shadow_score, double latency,
                                int success);
void     mab_shadow_test_evaluate(MABShadowTest *st, double threshold);

void     mab_feature_flag_set(const char *flag_name, int enabled);
int      mab_feature_flag_get(const char *flag_name);
int      mab_feature_flag_user(const char *flag_name, int32_t user_id,
                               int32_t rollout_pct);

void     mab_report_generate(const MABExperiment *exp,
                             const MABExperimentMetrics *metrics,
                             char *report, int32_t max_len);

#endif
