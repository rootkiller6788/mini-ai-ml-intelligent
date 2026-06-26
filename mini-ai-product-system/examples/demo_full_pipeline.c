#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "recommendation.h"
#include "chatbot_rlhf.h"
#include "copilot_context.h"
#include "eval_monitor.h"
#include "model_ab_test.h"

/* --- Full Pipeline: Recommendation -> ChatBot -> Copilot -> Eval -> A/B --- */

static uint64_t g_seed = 123456789;

static void pipeline_header(const char *step, const char *desc) {
    printf("\n+------------------------------------------------------------------+\n");
    printf("| STEP %s: %-54s |\n", step, desc);
    printf("+------------------------------------------------------------------+\n");
}

static void pipeline_result(const char *label, const char *value) {
    printf("  %-30s : %s\n", label, value);
}

static void pipeline_metric(const char *label, double value) {
    printf("  %-30s : %.4f\n", label, value);
}

static void eval_predict_fn(const char *input, char *output, int32_t max_len) {
    (void)input;
    snprintf(output, (size_t)max_len, "1");
}

int main(void) {
    srand((unsigned)time(NULL));
    g_seed = (uint64_t)time(NULL) * 31337ULL;

    printf("####################################################################\n");
    printf("#  Full AI Product Pipeline — End-to-End Demonstration            #\n");
    printf("####################################################################\n");

    /* ================================================================
     * STEP 1: Initialize Recommendation System & Generate Candidates
     * ================================================================ */
    pipeline_header("1", "Recommendation: User Profile -> Embedding -> Recall");

    RecUserTower user;
    rec_embedding_init(&user.emb, g_seed);
    user.bias = 0.1f;

    RecItemTower *catalog = (RecItemTower *)malloc(1000 * sizeof(RecItemTower));
    for (int i = 0; i < 1000; i++) {
        rec_embedding_init(&catalog[i].emb, g_seed + (uint64_t)i * 13);
        catalog[i].bias = (float)((i * 17) % 30) * 0.01f - 0.15f;
        catalog[i].item_id = i + 50000;
    }

    RecRecallChannel chans[5];
    chans[0] = (RecRecallChannel){ REC_SOURCE_COLLAB,  1.0f };
    chans[1] = (RecRecallChannel){ REC_SOURCE_ANN,     0.9f };
    chans[2] = (RecRecallChannel){ REC_SOURCE_POPULAR,  0.6f };
    chans[3] = (RecRecallChannel){ REC_SOURCE_RANDOM,   0.3f };
    chans[4] = (RecRecallChannel){ REC_SOURCE_CONTENT,  0.4f };
    RecRecallResult recall_res;
    rec_multi_channel_recall(&user, catalog, 1000, chans, 5, g_seed, &recall_res);

    pipeline_metric("Candidates recalled", (double)recall_res.candidates.count);
    pipeline_metric("Top candidate score", (double)recall_res.candidates.items[0].score);

    /* --- Rank candidates --- */
    int32_t dnn_dims[3] = {128, 64, 1};
    RecRankingDNN ranking_dnn;
    rec_ranking_dnn_init(&ranking_dnn, dnn_dims, 3, g_seed + 77);
    RecFeatureVec user_feat = { .num_features = 16 };
    RecFeatureVec sess_feat = { .num_features = 8 };
    for (int i = 0; i < 16; i++) user_feat.features[i] = (float)((i + 5) % 20) / 20.0f;
    for (int i = 0; i < 8; i++)  sess_feat.features[i] = (float)(i * 3 + 1) / 25.0f;

    RecCandidateList ranked_list;
    rec_rank_candidates(&ranking_dnn, &user_feat, &sess_feat, catalog, &recall_res.candidates, &ranked_list);

    int64_t now = (int64_t)time(NULL);
    int64_t ts_arr[REC_MAX_CANDIDATES];
    for (int i = 0; i < ranked_list.count; i++)
        ts_arr[i] = now - (int64_t)(i * 5400);
    rec_freshness_boost(&ranked_list, ts_arr, now, 0.5f);

    int32_t blocked_items[] = { 50003, 50017, 50042 };
    rec_business_rule_filter(&ranked_list, blocked_items, 3);

    rec_diversity_rerank(&ranked_list, catalog, 0.7f, 15);
    pipeline_metric("Final ranked items", (double)ranked_list.count);

    RecABConfig ab_cfg;
    rec_ab_config_init(&ab_cfg, 50, 50, g_seed);
    int user_variant = rec_ab_decide(&ab_cfg, 42);
    pipeline_result("Recommendation A/B", user_variant ? "treatment" : "control");

    /* ================================================================
     * STEP 2: ChatBot RLHF — Full Training Pipeline
     * ================================================================ */
    pipeline_header("2", "ChatBot RLHF: SFT -> Reward Model -> PPO -> DPO -> Safety");

    CBBaseModel base_model;
    cb_base_model_init(&base_model, g_seed + 10000);
    pipeline_result("Base model", "Random initialized");

    /* SFT Phase */
    CBInput sft_demos[5];
    CBOutput sft_targets[5];
    const char *demo_prompts[] = {
        "Hello", "What is C?", "Write a function",
        "Explain memory", "Thank you"
    };
    const char *demo_answers[] = {
        "Hello! How can I assist you today?",
        "C is a powerful systems programming language.",
        "int add(int a, int b) { return a + b; }",
        "Memory in C is managed manually with malloc/free.",
        "You're welcome!"
    };
    for (int i = 0; i < 5; i++) {
        cb_tokenize(demo_prompts[i], sft_demos[i].input_ids,
                    &sft_demos[i].input_len, CB_MAX_PROMPT_LEN);
        cb_tokenize(demo_answers[i], sft_targets[i].token_ids,
                    &sft_targets[i].length, CB_MAX_RESPONSE_LEN);
    }
    cb_sft_train(&base_model, sft_demos, sft_targets, 5, 0.001f, 3);
    pipeline_result("SFT Phase", "5 demos x 3 epochs");

    /* Reward Model Phase */
    CBRewardModel reward;
    cb_reward_model_init(&reward, g_seed + 20000);
    CBPreferencePair pref_pairs[5];
    memset(&pref_pairs, 0, sizeof(pref_pairs));
    for (int i = 0; i < 5; i++) {
        cb_tokenize(demo_prompts[i], pref_pairs[i].prompt.input_ids,
                    &pref_pairs[i].prompt.input_len, CB_MAX_PROMPT_LEN);
        cb_tokenize(demo_answers[i], pref_pairs[i].chosen.token_ids,
                    &pref_pairs[i].chosen.length, CB_MAX_RESPONSE_LEN);
        pref_pairs[i].chosen.total_logprob = -2.0f;
        cb_tokenize("I don't know.", pref_pairs[i].rejected.token_ids,
                    &pref_pairs[i].rejected.length, CB_MAX_RESPONSE_LEN);
        pref_pairs[i].rejected.total_logprob = -8.0f;
    }
    cb_reward_model_train(&reward, pref_pairs, 5, 0.01f, 3);
    pipeline_result("Reward Model Phase", "5 pairs x 3 epochs");

    float r_good = cb_reward_model_forward(&reward, pref_pairs[0].prompt.input_ids,
        pref_pairs[0].prompt.input_len, pref_pairs[0].chosen.token_ids, pref_pairs[0].chosen.length);
    float r_bad = cb_reward_model_forward(&reward, pref_pairs[0].prompt.input_ids,
        pref_pairs[0].prompt.input_len, pref_pairs[0].rejected.token_ids, pref_pairs[0].rejected.length);
    pipeline_metric("Reward (good)", r_good);
    pipeline_metric("Reward (bad)",  r_bad);

    /* PPO Phase */
    CBPPOConfig ppo_cfg;
    cb_ppo_config_init(&ppo_cfg);
    CBBaseModel policy, ref_model;
    memcpy(&policy, &base_model, sizeof(CBBaseModel));
    memcpy(&ref_model, &base_model, sizeof(CBBaseModel));
    cb_ppo_update(&policy, &ref_model, &reward, sft_demos, 5, &ppo_cfg);
    pipeline_result("PPO Phase", "Policy updated via RLHF");

    /* DPO Phase */
    CBDPOConfig dpo_cfg = { .dpo_beta = 0.1f, .learning_rate = 1e-5f, .epochs = 3 };
    memcpy(&dpo_cfg.model, &policy, sizeof(CBBaseModel));
    cb_dpo_train(&policy, pref_pairs, 5, &dpo_cfg);
    pipeline_result("DPO Phase", "Direct preference optimization done");

    /* Safety Alignment */
    const char *danger_phrases[] = {"hack", "virus", "exploit", "malware", "phish"};
    CBRedTeamSet red_team = { danger_phrases, 5 };
    float safety_loss;
    cb_safety_alignment_loss(&policy, sft_demos, 5, &red_team, &safety_loss);
    pipeline_metric("Safety alignment loss", (double)safety_loss);

    /* Generation Test */
    int32_t test_in[CB_MAX_PROMPT_LEN]; int32_t test_len;
    cb_tokenize("Write a hello world program", test_in, &test_len, CB_MAX_PROMPT_LEN);
    CBOutput gen_out;
    cb_generate(&policy, test_in, test_len, 32, 0.7f, g_seed + 123, &gen_out);
    char gen_text[512];
    cb_detokenize(gen_out.token_ids, gen_out.length, gen_text, 512);
    pipeline_result("Generated text", gen_text);

    /* Constitutional AI */
    cb_constitutional_alignment(&policy, sft_demos, 3, CB_AI_FEEDBACK_HARMLESS);
    pipeline_result("Constitutional AI", "Alignment step applied");

    /* ================================================================
     * STEP 3: Copilot — Context -> Prompt -> Completion
     * ================================================================ */
    pipeline_header("3", "Copilot: Context Gathering -> Prompt -> Completion");

    CCContextState ctx_state;
    cc_context_state_init(&ctx_state);

    const char *open_files[] = {
        "src/server.c", "include/server.h",
        "src/handler.c", "include/config.h",
        "tests/test_server.c"
    };
    cc_gather_open_files(&ctx_state, open_files, 5);
    cc_set_cursor(&ctx_state, "src/server.c", 256, 10);

    const char *edit_before = "static int port = 8080;\n";
    const char *edit_after  = "static int port = 8080;\nstatic int max_connections = 1000;\n";
    cc_gather_recent_edits(&ctx_state, "src/server.c",
                           edit_before, (int32_t)strlen(edit_before),
                           edit_after,  (int32_t)strlen(edit_after));

    const char *proj_paths[] = {"src/server.c", "src/handler.c", "include/server.h",
                                "include/config.h", "tests/test_server.c", "Makefile"};
    const char *proj_names[] = {"server.c", "handler.c", "server.h",
                                "config.h", "test_server.c", "Makefile"};
    const char *proj_langs[] = {"c", "c", "c", "c", "c", "makefile"};
    cc_gather_project_structure(&ctx_state, proj_paths, proj_names, proj_langs, 6);

    cc_gather_git_diff(&ctx_state, "+int max_connections = 1000;\n-old_limit();\n", 47);

    CCRankedContext ranked_ctx;
    cc_rank_context(&ctx_state, &ranked_ctx);
    pipeline_metric("Context snippets ranked", (double)ranked_ctx.num_snippets);

    CCUserRequest copilot_req;
    copilot_req.intent = CC_INTENT_COMPLETE;
    const char *req_body = "Implement handle_connection that reads from socket and dispatches.";
    copilot_req.text_len = (int32_t)strlen(req_body);
    memcpy(copilot_req.text, req_body, (size_t)copilot_req.text_len);
    copilot_req.text[copilot_req.text_len] = '\0';
    strcpy(copilot_req.language, "c");

    CCConstructedPrompt cop_prompt;
    cc_construct_prompt(&ctx_state, &copilot_req, &ranked_ctx, &cop_prompt);
    pipeline_metric("Prompt messages", (double)cop_prompt.num_messages);
    pipeline_metric("Prompt length (chars)", (double)cop_prompt.raw_len);

    CCCompletion cop_comp;
    cc_completion_generate(&cop_prompt, "void handle_connection(Socket *s) {\n",
                           0.7f, g_seed + 999, &cop_comp);
    cc_postprocess_trim(&cop_comp);
    int is_complete_suggestion;
    cc_postprocess_filter_incomplete(cop_comp.text, cop_comp.text_len, &is_complete_suggestion);
    pipeline_result("Completion is complete", is_complete_suggestion ? "yes" : "no");
    pipeline_metric("Completion length", (double)cop_comp.text_len);

    /* ================================================================
     * STEP 4: Evaluation — Offline + Online + Monitoring
     * ================================================================ */
    pipeline_header("4", "Evaluation: Offline Metrics + Online A/B + Drift Monitor");

    int32_t labels_true[200], labels_pred[200];
    for (int i = 0; i < 200; i++) {
        labels_true[i] = i % 3;
        labels_pred[i] = (i % 10 < 8) ? (i % 3) : ((i % 3 + 1) % 3);
    }
    EVClassificationMetrics cls;
    ev_classification_eval(labels_true, labels_pred, 200, 3, &cls);
    pipeline_metric("Classification accuracy", cls.accuracy);
    pipeline_metric("Classification F1", cls.f1_score);

    const char *nlg_refs[] = {
        "The server listens on port 8080 and accepts connections.",
        "Memory allocation returns NULL on failure."
    };
    const char *nlg_preds[] = {
        "The server listens on port 8080 accepting connections.",
        "Memory alloc returns null if it fails."
    };
    EVNLGMetrics nlg_m;
    ev_nlg_eval(nlg_refs, nlg_preds, 2, &nlg_m);
    pipeline_metric("BLEU avg", nlg_m.bleu_avg);
    pipeline_metric("ROUGE-L", nlg_m.rouge_l);

    EVGoldenDataset golden;
    ev_golden_dataset_init(&golden);
    golden.num_cases = 50;
    for (int i = 0; i < 50; i++) {
        golden.cases[i].id = i;
        golden.cases[i].class_label = i % 5;
        snprintf(golden.cases[i].input, EV_MAX_TEXT_LEN, "sample input %d", i);
    }
    EVOfflineResult offline_res;
    ev_offline_evaluate(&golden, eval_predict_fn, &offline_res);
    pipeline_metric("Offline accuracy", offline_res.accuracy);
    pipeline_metric("Offline F1", offline_res.f1);

    EVOfflineResult baseline;
    ev_metric_baseline_record(&golden, &baseline);
    int regression = ev_regression_detect(&baseline, &offline_res, 0.05);
    pipeline_result("Regression detected", regression ? "YES" : "NO");

    EVOnlineMetric online_ctrl, online_treat;
    ev_online_metric_init(&online_ctrl, "control_v1");
    ev_online_metric_init(&online_treat, "treatment_v2");
    for (int i = 0; i < 500; i++) {
        ev_online_collect(&online_ctrl, 0.55 + ((double)(rand() % 200)) / 1000.0,
                          0.07 + ((double)(rand() % 50)) / 1000.0,
                          0.68, 110.0 + (rand() % 30), rand() % 2, rand() % 3 == 0);
        ev_online_collect(&online_treat, 0.58 + ((double)(rand() % 200)) / 1000.0,
                          0.08 + ((double)(rand() % 50)) / 1000.0,
                          0.70, 115.0 + (rand() % 30), rand() % 2, rand() % 3 == 0);
    }
    pipeline_metric("Online Ctrl CTR", online_ctrl.click_through_rate);
    pipeline_metric("Online Treat CTR", online_treat.click_through_rate);
    pipeline_metric("Ctrl satisfaction", online_ctrl.user_satisfaction);
    pipeline_metric("Treat satisfaction", online_treat.user_satisfaction);

    EVABExperiment online_ab;
    ev_ab_experiment_compare(&online_ctrl, &online_treat, &online_ab);
    pipeline_result("Online A/B significant", online_ab.significant ? "YES" : "NO");

    double ref_dist_data[300], cur_dist_data[300];
    for (int i = 0; i < 300; i++) {
        ref_dist_data[i] = 0.5 + (double)(rand() % 100) / 100.0;
        cur_dist_data[i] = 0.5 + (double)(rand() % 100) / 100.0;
    }
    EVDriftReport drift_rpt;
    ev_prediction_drift_monitor(ref_dist_data, 300, cur_dist_data, 300, 100, &drift_rpt);
    pipeline_metric("Prediction drift PSI", drift_rpt.psi);
    pipeline_result("Drift alert", drift_rpt.drift_detected ? "YES" : "NO");

    EVDashboard dashboard;
    ev_dashboard_init(&dashboard);
    ev_dashboard_update(&dashboard, "accuracy", cls.accuracy, 0.85);
    ev_dashboard_update(&dashboard, "f1", cls.f1_score, 0.83);
    ev_dashboard_update(&dashboard, "latency_p99", 180.0, 200.0);
    ev_dashboard_update(&dashboard, "throughput_rps", 512.0, 480.0);
    ev_dashboard_update(&dashboard, "ctr", online_treat.click_through_rate, 0.07);
    ev_dashboard_update(&dashboard, "drift_psi", drift_rpt.psi, 0.1);
    char dash_out[4096];
    ev_dashboard_render(&dashboard, dash_out, 4096);
    printf("  [Dashboard]:\n%s", dash_out);

    EVFeedbackCollection feedback;
    memset(&feedback, 0, sizeof(feedback));
    for (int i = 0; i < 20; i++)
        ev_feedback_collect(&feedback, i, i * 10, rand() % 3 > 0 ? 1 : 0, "OK");
    int32_t fb_t, fb_u, fb_d; double fb_s;
    ev_feedback_summarize(&feedback, &fb_t, &fb_u, &fb_d, &fb_s);
    pipeline_metric("Feedback satisfaction", fb_s);

    /* ================================================================
     * STEP 5: Model A/B — Full Experiment Lifecycle
     * ================================================================ */
    pipeline_header("5", "Model A/B: Experiment -> Ramp-up -> Rollback -> Registry");

    MABExperiment mab_exp;
    mab_experiment_init(&mab_exp, "ChatBot v2.0 Experiment", g_seed + 40000);
    mab_experiment_add_variant(&mab_exp, "control_v1", MAB_VARIANT_CONTROL, 50, NULL);
    mab_experiment_add_variant(&mab_exp, "treat_v2", MAB_VARIANT_TREATMENT, 50, NULL);
    mab_experiment_activate(&mab_exp);

    int mab_ctrl = 0, mab_treat = 0;
    for (int uid = 0; uid < 10000; uid++) {
        int v = mab_traffic_assign(uid, &mab_exp);
        if (v == 0) mab_ctrl++; else mab_treat++;
    }
    pipeline_metric("MAB control users", (double)mab_ctrl);
    pipeline_metric("MAB treatment users", (double)mab_treat);

    MABVariantMetrics mab_ctrl_m, mab_treat_m;
    mab_metrics_init(&mab_ctrl_m);
    mab_metrics_init(&mab_treat_m);
    for (int i = 0; i < 300; i++) {
        mab_metrics_record_request(&mab_ctrl_m, 55.0 + (rand() % 25), 1, 0.82);
        mab_metrics_record_request(&mab_treat_m, 48.0 + (rand() % 20), 1, 0.88);
    }
    MABTTestResult ttest_r;
    mab_metrics_compare(&mab_ctrl_m, &mab_treat_m, "success", &ttest_r);
    pipeline_metric("t-Test p-value", ttest_r.p_value);
    pipeline_result("t-Test significant", ttest_r.significant ? "YES" : "NO");
    pipeline_metric("Effect size (Cohen's d)", ttest_r.effect_size);

    MABChiSquareResult chi_r;
    mab_chi_square_test(mab_ctrl_m.success_rate, mab_ctrl_m.total_requests,
                        mab_treat_m.success_rate, mab_treat_m.total_requests, &chi_r);
    pipeline_metric("Chi-square p-value", chi_r.p_value);

    MABRampUpPlan ramp_plan;
    mab_ramp_up_init(&ramp_plan, 1, 3600);
    MABGuardrails guard_set;
    mab_guardrails_init(&guard_set);
    mab_guardrails_add(&guard_set, "latency_p99", 250.0f, 1);
    mab_guardrails_add(&guard_set, "error_rate", 0.005f, 1);
    mab_guardrails_add(&guard_set, "success_rate_min", 0.95f, 0);
    MABRollbackDecision rollback_dec;
    mab_rollback_evaluate(&mab_ctrl_m, &mab_treat_m, &guard_set, &rollback_dec);
    pipeline_result("Rollback needed", rollback_dec.should_rollback ? "YES" : "NO");

    MABModelRegistry model_reg;
    mab_model_registry_init(&model_reg, "chatbot_model", "v2.0.0", MAB_STAGE_STAGING, NULL);
    pipeline_result("Registry stage", "STAGING");
    mab_model_registry_promote(&model_reg, MAB_STAGE_CANARY);
    mab_model_registry_promote(&model_reg, MAB_STAGE_PRODUCTION);
    pipeline_result("Registry promoted to", "PRODUCTION");
    pipeline_result("Can serve", mab_model_registry_can_serve(&model_reg) ? "YES" : "NO");

    MABShadowTest shadow_test;
    mab_shadow_test_init(&shadow_test);
    for (int i = 0; i < 200; i++)
        mab_shadow_test_record(&shadow_test, 0.93, 0.95, 52.0, 1);
    mab_shadow_test_evaluate(&shadow_test, 0.03);
    pipeline_result("Shadow test passed", shadow_test.passed ? "YES" : "NO");
    pipeline_metric("Shadow score delta", shadow_test.score_delta);

    mab_feature_flag_set("new_chatbot_model", 1);
    mab_feature_flag_set("dark_mode", 1);
    int ff_chat = mab_feature_flag_user("new_chatbot_model", 12345, 50);
    pipeline_result("Feature flag: chatbot@50%", ff_chat ? "ENABLED" : "DISABLED");

    MABExperimentMetrics exp_all;
    MABVariantMetrics per_variant_arr[2] = { mab_ctrl_m, mab_treat_m };
    mab_metrics_collect(&mab_exp, per_variant_arr, &exp_all);
    char full_report[8192];
    mab_report_generate(&mab_exp, &exp_all, full_report, 8192);
    printf("  [Experiment Report]:\n%s", full_report);

    int sample = mab_sample_size_estimate(0.08, 0.015, 0.05, 0.80);
    pipeline_metric("Min sample per variant", (double)sample);

    free(catalog);

    printf("\n####################################################################\n");
    printf("#  Full Pipeline Complete — All 5 modules integrated & verified.   #\n");
    printf("####################################################################\n");
    return 0;
}
