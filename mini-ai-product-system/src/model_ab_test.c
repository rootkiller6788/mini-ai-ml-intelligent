#include "model_ab_test.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define MAB_FLAGS_MAX 128

static struct { char name[64]; int enabled; } mab_feature_flags[MAB_FLAGS_MAX];
static int mab_num_flags = 0;

static double mab_mean(const double *x, int32_t n) {
    if (n == 0) return 0.0;
    double s = 0.0;
    for (int i = 0; i < n; i++) s += x[i];
    return s / (double)n;
}

static double mab_var(const double *x, int32_t n, double mean) {
    if (n < 2) return 1.0;
    double s = 0.0;
    for (int i = 0; i < n; i++) { double d = x[i] - mean; s += d * d; }
    return s / (double)(n - 1);
}

static double mab_t_dist_cdf(double t, int df) {
    double x = df / (df + t * t);
    double a = 1.0, b = 1.0;
    for (int j = df - 2; j >= 2; j -= 2) a = 1.0 + a * (double)(j - 1) / (2.0 * (double)j) * x;
    if (df % 2 == 0) return 0.5 + 0.5 * sqrt(x) * a;
    double c = 1.0;
    for (int j = df - 2; j >= 1; j -= 2) c = 1.0 + c * (double)j / (double)(j + 1) * x;
    return 0.5 + t * sqrt(x) * c / (M_PI * 1.0);
}

static double mab_chi2_cdf(double x, int df) {
    if (x < 0) return 0.0;
    double sum = 0.0, term = 1.0;
    int k = df / 2;
    for (int i = 0; i < 100; i++) {
        if (k + i >= 1000) break;
        double num = pow(x / 2.0, (double)(k + i));
        double den = tgamma((double)(k + i + 1));
        sum += term;
        term *= x / (2.0 * (k + i + 1.0));
        (void)num; (void)den;
    }
    return sum < 1.0 ? sum : 1.0;
}

uint64_t mab_user_hash(int32_t user_id, uint64_t seed) {
    uint64_t h = (uint64_t)(user_id) ^ seed;
    h = (h ^ (h >> 30)) * 0xBF58476D1CE4E5B9ULL;
    h = (h ^ (h >> 27)) * 0x94D049BB133111EBULL;
    return h ^ (h >> 31);
}

int mab_traffic_assign(int32_t user_id, const MABExperiment *exp) {
    if (!exp->is_active) return 0;
    uint64_t h = mab_user_hash(user_id, exp->hash_seed);
    float bucket = (float)(h % 10000) / 100.0f;
    float cum = 0.0f;
    for (int i = 0; i < exp->num_variants; i++) {
        cum += (float)exp->variants[i].traffic_pct;
        if (bucket < cum) return i;
    }
    return 0;
}

void mab_experiment_init(MABExperiment *exp, const char *name, uint64_t seed) {
    memset(exp, 0, sizeof(MABExperiment));
    strncpy(exp->name, name, 127);
    exp->name[127] = '\0';
    exp->num_variants = 0;
    exp->is_active = 0;
    exp->hash_seed = seed;
    exp->start_time = (int64_t)time(NULL);
}

void mab_experiment_add_variant(MABExperiment *exp, const char *name,
                                MABVariantType type, int32_t traffic_pct,
                                void *model_ptr) {
    if (exp->num_variants >= MAB_MAX_VARIANTS) return;
    MABVariant *v = &exp->variants[exp->num_variants++];
    strncpy(v->name, name, 63);
    v->name[63] = '\0';
    v->type = type;
    v->traffic_pct = traffic_pct;
    v->model_ptr = model_ptr;
}

int mab_experiment_validate(const MABExperiment *exp) {
    int32_t total = 0;
    for (int i = 0; i < exp->num_variants; i++)
        total += exp->variants[i].traffic_pct;
    return total == 100 ? 1 : 0;
}

void mab_experiment_activate(MABExperiment *exp) {
    exp->is_active = 1;
    exp->start_time = (int64_t)time(NULL);
}

void mab_experiment_deactivate(MABExperiment *exp) {
    exp->is_active = 0;
    exp->end_time = (int64_t)time(NULL);
}

void mab_metrics_init(MABVariantMetrics *m) {
    memset(m, 0, sizeof(MABVariantMetrics));
}

void mab_metrics_record_request(MABVariantMetrics *m, double latency,
                                int success, double satisfaction) {
    m->total_requests++;
    double n = (double)m->total_requests;
    m->latency_ms = (m->latency_ms * (n - 1.0) + latency) / n;
    m->success_rate = (m->success_rate * (n - 1.0) + (success ? 1.0 : 0.0)) / n;
    m->user_satisfaction = (m->user_satisfaction * (n - 1.0) + satisfaction) / n;
}

void mab_metrics_collect(const MABExperiment *exp,
                         const MABVariantMetrics *per_variant,
                         MABExperimentMetrics *result) {
    result->num_variants = exp->num_variants;
    for (int i = 0; i < exp->num_variants; i++)
        result->per_variant[i] = per_variant[i];
}

void mab_metrics_compare(const MABVariantMetrics *ctrl,
                         const MABVariantMetrics *treat,
                         const char *metric_name,
                         MABTTestResult *result) {
    (void)metric_name;
    double c_val = ctrl->success_rate;
    double t_val = treat->success_rate;
    int32_t cn = ctrl->total_requests > 0 ? ctrl->total_requests : 1;
    int32_t tn = treat->total_requests > 0 ? treat->total_requests : 1;
    double *ca = (double *)malloc((size_t)cn * sizeof(double));
    double *ta = (double *)malloc((size_t)tn * sizeof(double));
    for (int i = 0; i < cn; i++) ca[i] = c_val;
    for (int i = 0; i < tn; i++) ta[i] = t_val;
    mab_ttest_independent(ca, cn, ta, tn, result);
    free(ca); free(ta);
}

void mab_ttest_independent(const double *a, int32_t na,
                           const double *b, int32_t nb,
                           MABTTestResult *result) {
    double ma = mab_mean(a, na);
    double mb = mab_mean(b, nb);
    double va = mab_var(a, na, ma);
    double vb = mab_var(b, nb, mb);
    double se = sqrt(va / (double)na + vb / (double)nb);
    result->t_statistic = (se > 1e-10) ? (mb - ma) / se : 0.0;
    int df = na + nb - 2;
    if (df < 1) df = 1;
    result->df = df;
    result->effect_size = (mb - ma) / sqrt((va + vb) / 2.0);
    result->ci_lower = (mb - ma) - 1.96 * se;
    result->ci_upper = (mb - ma) + 1.96 * se;
    double t_abs = fabs(result->t_statistic);
    result->p_value = 2.0 * (1.0 - mab_t_dist_cdf(t_abs, df));
    result->significant = (result->p_value < 0.05) ? 1 : 0;
}

void mab_chi_square_test(double ctrl_rate, int32_t ctrl_n,
                         double treat_rate, int32_t treat_n,
                         MABChiSquareResult *result) {
    result->observed_ctrl = (int32_t)(ctrl_rate * ctrl_n);
    result->observed_treat = (int32_t)(treat_rate * treat_n);
    result->total_ctrl = ctrl_n;
    result->total_treat = treat_n;
    double total_rate = (ctrl_rate * ctrl_n + treat_rate * treat_n) /
                        (double)(ctrl_n + treat_n);
    double exp_ctrl = total_rate * ctrl_n;
    double exp_treat = total_rate * treat_n;
    double exp_no_ctrl = ctrl_n - exp_ctrl;
    double exp_no_treat = treat_n - exp_treat;
    double obs_no_ctrl = ctrl_n - result->observed_ctrl;
    double obs_no_treat = treat_n - result->observed_treat;
    double chi2 = 0.0;
    if (exp_ctrl > 0) chi2 += pow(result->observed_ctrl - exp_ctrl, 2) / exp_ctrl;
    if (exp_treat > 0) chi2 += pow(result->observed_treat - exp_treat, 2) / exp_treat;
    if (exp_no_ctrl > 0) chi2 += pow(obs_no_ctrl - exp_no_ctrl, 2) / exp_no_ctrl;
    if (exp_no_treat > 0) chi2 += pow(obs_no_treat - exp_no_treat, 2) / exp_no_treat;
    result->chi2_stat = chi2;
    result->df = 1;
    result->p_value = 1.0 - mab_chi2_cdf(chi2, 1);
    result->significant = (result->p_value < 0.05) ? 1 : 0;
}

int mab_significance_check(const MABTTestResult *t, double alpha) {
    return t->p_value < alpha ? 1 : 0;
}

int mab_sample_size_estimate(double baseline_rate, double min_detectable_effect,
                              double alpha, double power) {
    double z_alpha = 1.96;
    double z_power = 0.84;
    (void)alpha; (void)power;
    double p = baseline_rate;
    double delta = min_detectable_effect;
    double n = 2.0 * p * (1.0 - p) * pow((z_alpha + z_power) / delta, 2);
    return (int)(n + 0.5);
}

void mab_ramp_up_init(MABRampUpPlan *plan, int32_t initial, int64_t step_dur) {
    plan->initial_pct = initial;
    plan->step_duration_s = step_dur;
    plan->ramp_steps[0] = 1;
    plan->ramp_steps[1] = 5;
    plan->ramp_steps[2] = 25;
    plan->ramp_steps[3] = 50;
    plan->ramp_sizes[0] = plan->ramp_steps[0];
    plan->ramp_sizes[1] = plan->ramp_steps[1] - plan->ramp_steps[0];
    plan->ramp_sizes[2] = plan->ramp_steps[2] - plan->ramp_steps[1];
    plan->ramp_sizes[3] = plan->ramp_steps[3] - plan->ramp_steps[2];
    plan->num_steps = 4;
}

void mab_ramp_up_step(const MABRampUpPlan *plan, int32_t step_idx,
                      MABExperiment *exp) {
    (void)plan;
    if (step_idx >= 0 && step_idx < exp->num_variants) {
        int32_t target = 1;
        if (step_idx == 1) target = 5;
        else if (step_idx == 2) target = 25;
        else if (step_idx == 3) target = 50;
        if (exp->variants[1].traffic_pct < target)
            exp->variants[1].traffic_pct = target;
    }
}

void mab_ramp_up_execute(MABExperiment *exp, const MABRampUpPlan *plan,
                         const MABGuardrails *guardrails,
                         MABRollbackDecision *decision) {
    decision->should_rollback = 0;
    decision->auto_rollback = 0;
    strcpy(decision->reason, "");
    for (int s = 0; s < plan->num_steps; s++) {
        mab_ramp_up_step(plan, s, exp);
        MABVariantMetrics dummy;
        mab_metrics_init(&dummy);
        dummy.latency_ms = 100.0;
        mab_guardrails_check(guardrails, &dummy);
    }
}

void mab_rollback_evaluate(const MABVariantMetrics *ctrl,
                           const MABVariantMetrics *treat,
                           const MABGuardrails *guardrails,
                           MABRollbackDecision *decision) {
    decision->should_rollback = 0;
    decision->auto_rollback = 0;
    strcpy(decision->reason, "");
    if (treat->error_rate > ctrl->error_rate * 2.0) {
        decision->should_rollback = 1;
        decision->degradation_pct = treat->error_rate - ctrl->error_rate;
        strcpy(decision->reason, "Error rate doubled");
    }
    if (treat->latency_ms > ctrl->latency_ms * 1.5) {
        decision->should_rollback = 1;
        decision->degradation_pct = (treat->latency_ms - ctrl->latency_ms) / ctrl->latency_ms;
        strcpy(decision->reason, "Latency increase > 50%");
    }
    if (!mab_guardrails_check(guardrails, treat)) {
        decision->should_rollback = 1;
        strcpy(decision->reason, "Guardrail violation");
    }
    (void)guardrails;
}

void mab_rollback_execute(MABExperiment *exp) {
    if (exp->variants[1].traffic_pct > 0)
        exp->variants[1].traffic_pct = 0;
    exp->variants[0].traffic_pct = 100;
}

void mab_model_registry_init(MABModelRegistry *reg, const char *name,
                             const char *version, MABModelStage stage,
                             void *model_ptr) {
    memset(reg, 0, sizeof(MABModelRegistry));
    strncpy(reg->name, name, 127);
    reg->name[127] = '\0';
    strncpy(reg->version, version, 31);
    reg->version[31] = '\0';
    reg->stage = stage;
    reg->model_ptr = model_ptr;
    reg->promoted_at = (int64_t)time(NULL);
}

void mab_model_registry_promote(MABModelRegistry *reg, MABModelStage next) {
    if (next > reg->stage) {
        reg->stage = next;
        reg->promoted_at = (int64_t)time(NULL);
    }
}

void mab_model_registry_archive(MABModelRegistry *reg) {
    reg->stage = MAB_STAGE_ARCHIVED;
}

int mab_model_registry_can_serve(const MABModelRegistry *reg) {
    return (reg->stage == MAB_STAGE_PRODUCTION ||
            reg->stage == MAB_STAGE_STAGING ||
            reg->stage == MAB_STAGE_CANARY) ? 1 : 0;
}

void mab_guardrails_init(MABGuardrails *g) {
    memset(g, 0, sizeof(MABGuardrails));
}

void mab_guardrails_add(MABGuardrails *g, const char *name,
                        float threshold, int critical) {
    if (g->num_guardrails >= MAB_MAX_METRICS) return;
    MABGuardrailMetric *gm = &g->guardrails[g->num_guardrails++];
    strncpy(gm->name, name, 63);
    gm->name[63] = '\0';
    gm->threshold = threshold;
    gm->is_critical = critical;
}

int mab_guardrails_check(const MABGuardrails *g,
                         const MABVariantMetrics *m) {
    (void)g; (void)m;
    return 1;
}

void mab_guardrails_alert(const MABGuardrails *g, const char *variant_name,
                          char *alert_msg, int32_t max_len) {
    snprintf(alert_msg, (size_t)max_len, "GUARDRAIL ALERT for %s", variant_name);
    (void)g;
}

void mab_shadow_test_init(MABShadowTest *st) {
    memset(st, 0, sizeof(MABShadowTest));
    st->passed = 1;
}

void mab_shadow_test_record(MABShadowTest *st, double prod_score,
                            double shadow_score, double latency,
                            int success) {
    st->total_requests++;
    double n = (double)st->total_requests;
    st->shadow_metrics.success_rate = (st->shadow_metrics.success_rate * (n - 1.0) +
                                        (success ? 1.0 : 0.0)) / n;
    st->latency_ms = (st->latency_ms * (n - 1.0) + latency) / n;
    st->score_delta = (st->score_delta * (n - 1.0) + (shadow_score - prod_score)) / n;
    (void)prod_score;
    (void)shadow_score;
}

void mab_shadow_test_evaluate(MABShadowTest *st, double threshold) {
    st->passed = (fabs(st->score_delta) < threshold) ? 1 : 0;
    if (st->error_rate > 0.01) st->passed = 0;
}

void mab_feature_flag_set(const char *flag_name, int enabled) {
    for (int i = 0; i < mab_num_flags; i++) {
        if (strcmp(mab_feature_flags[i].name, flag_name) == 0) {
            mab_feature_flags[i].enabled = enabled;
            return;
        }
    }
    if (mab_num_flags < MAB_FLAGS_MAX) {
        strncpy(mab_feature_flags[mab_num_flags].name, flag_name, 63);
        mab_feature_flags[mab_num_flags].enabled = enabled;
        mab_num_flags++;
    }
}

int mab_feature_flag_get(const char *flag_name) {
    for (int i = 0; i < mab_num_flags; i++)
        if (strcmp(mab_feature_flags[i].name, flag_name) == 0)
            return mab_feature_flags[i].enabled;
    return 0;
}

int mab_feature_flag_user(const char *flag_name, int32_t user_id,
                          int32_t rollout_pct) {
    if (!mab_feature_flag_get(flag_name)) return 0;
    uint64_t h = mab_user_hash(user_id, MAB_HASH_SEED);
    return (int)(h % 100) < rollout_pct ? 1 : 0;
}

void mab_report_generate(const MABExperiment *exp,
                         const MABExperimentMetrics *metrics,
                         char *report, int32_t max_len) {
    int off = 0;
    off += snprintf(report + off, (size_t)(max_len - off),
                    "===== A/B Test Report: %s =====\n", exp->name);
    off += snprintf(report + off, (size_t)(max_len - off),
                    "Active: %s  Start: %lld\n",
                    exp->is_active ? "YES" : "NO", (long long)exp->start_time);
    for (int i = 0; i < metrics->num_variants; i++) {
        const MABVariantMetrics *m = &metrics->per_variant[i];
        off += snprintf(report + off, (size_t)(max_len - off),
                        "  %s: req=%d lat=%.2fms succ=%.4f sat=%.4f\n",
                        exp->variants[i].name, m->total_requests,
                        m->latency_ms, m->success_rate, m->user_satisfaction);
    }
    off += snprintf(report + off, (size_t)(max_len - off),
                    "====================================\n");
}
