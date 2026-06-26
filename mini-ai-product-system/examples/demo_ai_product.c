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

static void demo_section(const char *title) {
    printf("\n%s\n", title);
    for (size_t i = 0; i < strlen(title); i++) putchar('=');
    printf("\n");
}

static void print_classification_metrics(const EVClassificationMetrics *m) {
    printf("  Accuracy:  %.4f\n", m->accuracy);
    printf("  Precision: %.4f\n", m->precision);
    printf("  Recall:    %.4f\n", m->recall);
    printf("  F1-Score:  %.4f\n", m->f1_score);
}

static void print_nlg_metrics(const EVNLGMetrics *m) {
    printf("  BLEU: 1g=%.4f 2g=%.4f 3g=%.4f 4g=%.4f avg=%.4f\n",
           m->bleu_1gram, m->bleu_2gram, m->bleu_3gram, m->bleu_4gram, m->bleu_avg);
    printf("  ROUGE: 1=%.4f 2=%.4f L=%.4f\n", m->rouge_1, m->rouge_2, m->rouge_l);
}

static void print_variant_metrics(const MABVariantMetrics *m, const char *name) {
    printf("  %-12s: req=%-5d lat=%-8.2f succ=%-6.4f sat=%-6.4f\n",
           name, m->total_requests, m->latency_ms, m->success_rate, m->user_satisfaction);
}

int main(void) {
    uint64_t seed = 31337;
    srand((unsigned)time(NULL));
    printf("================================================================\n");
    printf("   mini-ai-product-system — AI Product System Demo\n");
    printf("================================================================\n");

    /* =========================== 1. Recommendation =========================== */
    demo_section("1. Recommendation System");
    {
        RecUserTower user_tower;
        rec_embedding_init(&user_tower.emb, seed);
        user_tower.bias = 0.05f;

        RecItemTower *items = (RecItemTower *)malloc(500 * sizeof(RecItemTower));
        for (int i = 0; i < 500; i++) {
            rec_embedding_init(&items[i].emb, seed + (uint64_t)(i + 1) * 7);
            items[i].bias = (float)((i * 7) % 20) * 0.01f - 0.1f;
            items[i].item_id = 10000 + i;
        }

        RecCandidateList cf_result;
        rec_collab_filter(&user_tower, items, 500, 20, &cf_result);
        printf("  [Collaborative Filtering] Top-5:\n");
        for (int i = 0; i < 5; i++)
            printf("    item_%d  score=%.4f\n", cf_result.items[i].item_id,
                   cf_result.items[i].score);

        RecRecallChannel chans[4];
        chans[0].source = REC_SOURCE_COLLAB;  chans[0].weight = 1.0f;
        chans[1].source = REC_SOURCE_ANN;     chans[1].weight = 0.8f;
        chans[2].source = REC_SOURCE_POPULAR;  chans[2].weight = 0.5f;
        chans[3].source = REC_SOURCE_RANDOM;   chans[3].weight = 0.2f;
        RecRecallResult recall;
        rec_multi_channel_recall(&user_tower, items, 500, chans, 4, seed, &recall);
        printf("  [Multi-channel Recall] %d candidates\n", recall.candidates.count);

        int32_t dims[4] = {64, 32, 16, 1};
        RecRankingDNN dnn;
        rec_ranking_dnn_init(&dnn, dims, 4, seed + 42);
        RecFeatureVec ufeat, sfeat;
        ufeat.num_features = 10; sfeat.num_features = 6;
        for (int i = 0; i < 10; i++) ufeat.features[i] = (float)(i + 3) / 15.0f;
        for (int i = 0; i < 6; i++)  sfeat.features[i] = (float)(i * 2) / 15.0f;
        RecCandidateList ranked;
        rec_rank_candidates(&dnn, &ufeat, &sfeat, items, &cf_result, &ranked);
        printf("  [Ranked] Top-5:\n");
        for (int i = 0; i < 5; i++)
            printf("    item_%d  score=%.4f\n", ranked.items[i].item_id,
                   ranked.items[i].score);

        int64_t now = (int64_t)time(NULL);
        int64_t timestamps[REC_MAX_CANDIDATES];
        for (int i = 0; i < ranked.count; i++)
            timestamps[i] = now - (int64_t)(i * 7200);
        rec_freshness_boost(&ranked, timestamps, now, 0.3f);
        printf("  [Freshness Boost] Applied to %d items\n", ranked.count);

        int32_t blocked[] = {10003, 10007, 10015, 10021};
        rec_business_rule_filter(&ranked, blocked, 4);
        printf("  [Business Rule Filter] %d items remain after blocking 4 items\n",
               ranked.count);

        rec_diversity_rerank(&ranked, items, 0.6f, 10);
        printf("  [Diversity Re-rank] %d items after diversity\n", ranked.count);

        const char *tags[] = {"machine-learning", "python", "tutorial"};
        RecFeatureVec content_fv;
        rec_content_features("Deep Learning with PyTorch", "ai/ml", tags, 3, &content_fv);
        printf("  [Content Features] %d dims for new item\n", content_fv.num_features);

        float cs;
        rec_cold_start_score(&content_fv, &ufeat, &cs);
        printf("  [Cold Start] Score for new item: %.4f\n", cs);

        RecABConfig ab_cfg;
        rec_ab_config_init(&ab_cfg, 50, 50, seed);
        int treat_count = 0;
        for (int uid = 0; uid < 1000; uid++)
            treat_count += rec_ab_decide(&ab_cfg, uid);
        printf("  [A/B Split] Treatment: ~%d/1000\n", treat_count);

        double ctrl_vals[200], treat_vals[200];
        for (int i = 0; i < 200; i++) {
            ctrl_vals[i] = 0.12 + (rand() % 20) * 0.002;
            treat_vals[i] = 0.14 + (rand() % 20) * 0.002;
        }
        RecABResult ab_r;
        rec_ab_evaluate(ctrl_vals, 200, treat_vals, 200, &ab_r);
        printf("  [A/B Result] ctrl=%.4f treat=%.4f p=%.4f significant=%d\n",
               ab_r.ctr_control, ab_r.ctr_treatment,
               ab_r.p_value, ab_r.significant);

        free(items);
    }

    /* =========================== 2. ChatBot RLHF =========================== */
    demo_section("2. ChatBot RLHF Pipeline");
    {
        CBBaseModel base;
        cb_base_model_init(&base, seed + 1000);

        CBInput sft_in[4];
        CBOutput sft_out[4];
        const char *sft_texts[] = {
            "Hello!", "How can I help?", "Here is your answer.",
            "Thank you for asking."
        };
        for (int i = 0; i < 4; i++) {
            cb_tokenize(sft_texts[i], sft_in[i].input_ids,
                        &sft_in[i].input_len, CB_MAX_PROMPT_LEN);
            sft_out[i].length = sft_in[i].input_len;
            memcpy(sft_out[i].token_ids, sft_in[i].input_ids,
                   (size_t)sft_in[i].input_len * sizeof(int32_t));
        }
        cb_sft_train(&base, sft_in, sft_out, 4, 0.001f, 3);
        printf("  [SFT] Trained on 4 demonstrations, 3 epochs\n");

        CBRewardModel rm;
        cb_reward_model_init(&rm, seed + 2000);
        CBPreferencePair pairs[3];
        memset(&pairs, 0, sizeof(pairs));
        const char *prompts[3] = {"What is C?", "Write hello world", "Explain pointers"};
        const char *chosen[3]  = {"C is a programming language.", "printf(\"hello\");",
                                   "Pointers store memory addresses."};
        const char *rejected[3] = {"I don't know.", "...", "It's complicated."};
        for (int i = 0; i < 3; i++) {
            cb_tokenize(prompts[i], pairs[i].prompt.input_ids,
                        &pairs[i].prompt.input_len, CB_MAX_PROMPT_LEN);
            cb_tokenize(chosen[i], pairs[i].chosen.token_ids,
                        &pairs[i].chosen.length, CB_MAX_RESPONSE_LEN);
            pairs[i].chosen.total_logprob = -1.0f - (float)i * 0.5f;
            cb_tokenize(rejected[i], pairs[i].rejected.token_ids,
                        &pairs[i].rejected.length, CB_MAX_RESPONSE_LEN);
            pairs[i].rejected.total_logprob = -4.0f - (float)i;
        }
        cb_reward_model_train(&rm, pairs, 3, 0.01f, 2);
        printf("  [Reward Model] Trained on 3 preference pairs, 2 epochs\n");

        float r_chosen = cb_reward_model_forward(&rm, pairs[0].prompt.input_ids,
            pairs[0].prompt.input_len, pairs[0].chosen.token_ids, pairs[0].chosen.length);
        float r_rejected = cb_reward_model_forward(&rm, pairs[0].prompt.input_ids,
            pairs[0].prompt.input_len, pairs[0].rejected.token_ids, pairs[0].rejected.length);
        printf("  [Reward Scores] chosen=%.4f  rejected=%.4f\n", r_chosen, r_rejected);

        CBPPOConfig ppo_cfg;
        cb_ppo_config_init(&ppo_cfg);
        CBBaseModel policy, ref;
        memcpy(&policy, &base, sizeof(CBBaseModel));
        memcpy(&ref, &base, sizeof(CBBaseModel));
        cb_ppo_update(&policy, &ref, &rm, pairs[0].prompt.input_ids ? &pairs[0].prompt : NULL,
                      1, &ppo_cfg);
        printf("  [PPO] Policy update completed\n");

        CBDPOConfig dpo_cfg = { .dpo_beta = 0.1f, .learning_rate = 1e-5f, .epochs = 2 };
        memcpy(&dpo_cfg.model, &base, sizeof(CBBaseModel));
        cb_dpo_train(&base, pairs, 3, &dpo_cfg);
        printf("  [DPO] Direct preference optimization completed\n");

        int32_t gen_in[CB_MAX_PROMPT_LEN]; int32_t gen_len;
        cb_tokenize("Hello, world!", gen_in, &gen_len, CB_MAX_PROMPT_LEN);
        CBOutput gen_out;
        cb_generate(&base, gen_in, gen_len, 16, 0.8f, seed + 5000, &gen_out);
        char gen_txt[CB_MAX_RESPONSE_LEN];
        cb_detokenize(gen_out.token_ids, gen_out.length, gen_txt, CB_MAX_RESPONSE_LEN);
        printf("  [Generation] '%s' (%d tokens, logprob=%.2f)\n",
               gen_txt, gen_out.length, gen_out.total_logprob);

        const char *bad[] = {"hack", "virus", "exploit", "malware"};
        CBRedTeamSet rt = { bad, 4 };
        float harm = cb_harmlessness_score(&gen_out, &rt);
        printf("  [Safety] Harmlessness: %.2f\n", harm);
    }

    /* =========================== 3. Copilot Context =========================== */
    demo_section("3. Copilot Context System");
    {
        CCContextState ctx;
        cc_context_state_init(&ctx);

        const char *files[] = {"src/app.c", "include/app.h", "tests/test.c"};
        cc_gather_open_files(&ctx, files, 3);
        cc_set_cursor(&ctx, "src/app.c", 128, 22);

        const char *imports = "#include \"app.h\"\n#include <string.h>\n";
        cc_gather_imports(&ctx, "src/app.c", imports, (int32_t)strlen(imports));

        const char *p_paths[] = {"src/app.c", "include/app.h"};
        const char *p_names[] = {"app.c", "app.h"};
        const char *p_langs[] = {"c", "c"};
        cc_gather_project_structure(&ctx, p_paths, p_names, p_langs, 2);

        cc_gather_git_diff(&ctx, "+int new_feature(void);\n", 22);

        CCUserRequest req;
        req.intent = CC_INTENT_COMPLETE;
        const char *rtext = "Complete the init_app function.";
        req.text_len = (int32_t)strlen(rtext);
        memcpy(req.text, rtext, (size_t)req.text_len);
        req.text[req.text_len] = '\0';
        strcpy(req.language, "c");

        CCRankedContext rc;
        cc_rank_context(&ctx, &rc);
        CCConstructedPrompt ccp;
        cc_construct_prompt(&ctx, &req, &rc, &ccp);

        CCCompletion comp;
        cc_completion_generate(&ccp, "void init_app(void) {\n", 0.7f, seed, &comp);
        cc_postprocess_trim(&comp);
        printf("  [Completion] %d chars generated\n", comp.text_len);

        CCMultiLineCompletion mlc;
        cc_completion_multi_line(&ccp, "", 4, 0.8f, seed, &mlc);
        printf("  [Multi-line] %d lines\n", mlc.num_lines);
    }

    /* =========================== 4. Eval & Monitor =========================== */
    demo_section("4. Evaluation & Monitoring");
    {
        int32_t y_true[100], y_pred[100];
        int acc_count = 0;
        for (int i = 0; i < 100; i++) {
            y_true[i] = i % 5;
            y_pred[i] = (i % 5 == 0) ? (i % 5) : ((i % 5 + (rand() % 2 == 0 ? 0 : 1)) % 5);
            if (y_true[i] == y_pred[i]) acc_count++;
        }

        EVClassificationMetrics cls_metrics;
        ev_classification_eval(y_true, y_pred, 100, 5, &cls_metrics);
        printf("  [Classification]:\n");
        print_classification_metrics(&cls_metrics);

        const char *refs[] = {"The quick brown fox jumps over the lazy dog."};
        const char *preds[] = {"A fast brown fox leaps over a sleepy dog."};
        EVNLGMetrics nlg;
        ev_nlg_eval(refs, preds, 1, &nlg);
        printf("  [NLG Metrics]:\n");
        print_nlg_metrics(&nlg);

        EVOnlineMetric ctrl_om, treat_om;
        ev_online_metric_init(&ctrl_om, "control");
        ev_online_metric_init(&treat_om, "treatment");
        for (int i = 0; i < 100; i++) {
            ev_online_collect(&ctrl_om, 0.6, 0.08, 0.7, 120.0, 1, 0);
            ev_online_collect(&treat_om, 0.65, 0.09, 0.72, 130.0, 1, 0);
        }
        printf("  [Online Ctrl] engage=%.4f ctr=%.4f users=%d\n",
               ctrl_om.engagement_rate, ctrl_om.click_through_rate, ctrl_om.num_users);
        printf("  [Online Treat] engage=%.4f ctr=%.4f users=%d\n",
               treat_om.engagement_rate, treat_om.click_through_rate, treat_om.num_users);

        EVABExperiment ab_exp;
        ev_ab_experiment_compare(&ctrl_om, &treat_om, &ab_exp);
        printf("  [A/B Test] p=%.4f significant=%d\n", ab_exp.p_value, ab_exp.significant);

        EVLLMJudgeResult judge;
        ev_llm_judge_init(&judge);
        ev_llm_judge_evaluate(&judge, "Good response", "Reference text", EV_JUDGE_HELPFULNESS);
        printf("  [LLM Judge] Helpfulness: %.2f\n", judge.scores[EV_JUDGE_HELPFULNESS]);

        double ref_data[200], cur_data[200];
        for (int i = 0; i < 200; i++) {
            ref_data[i] = 0.5 + ((double)(rand() % 100)) / 100.0;
            cur_data[i] = 0.5 + ((double)(rand() % 100)) / 100.0;
        }
        EVDriftReport drift;
        ev_data_drift_detect(ref_data, 200, cur_data, 200, 10, &drift);
        printf("  [Drift] PSI=%.4f drift=%d\n", drift.psi, drift.drift_detected);

        EVDashboard dash;
        ev_dashboard_init(&dash);
        ev_dashboard_update(&dash, "accuracy", 0.87, 0.85);
        ev_dashboard_update(&dash, "f1_score", 0.84, 0.83);
        ev_dashboard_update(&dash, "latency_p99", 180.0, 200.0);
        ev_dashboard_update(&dash, "throughput", 500.0, 480.0);
        char dash_buf[2048];
        ev_dashboard_render(&dash, dash_buf, 2048);
        printf("  [Dashboard]:\n%s", dash_buf);

        EVFeedbackCollection fb;
        memset(&fb, 0, sizeof(fb));
        ev_feedback_collect(&fb, 1, 101, 1, "Great!");
        ev_feedback_collect(&fb, 2, 102, 1, "Good");
        ev_feedback_collect(&fb, 3, 103, 0, "Wrong answer");
        int32_t fb_total, fb_up, fb_down; double fb_sat;
        ev_feedback_summarize(&fb, &fb_total, &fb_up, &fb_down, &fb_sat);
        printf("  [Feedback] total=%d up=%d down=%d satisfaction=%.2f\n",
               fb_total, fb_up, fb_down, fb_sat);
    }

    /* =========================== 5. Model A/B =========================== */
    demo_section("5. Model A/B Testing");
    {
        MABExperiment exp;
        mab_experiment_init(&exp, "Model v2 Experiment", seed + 9000);
        mab_experiment_add_variant(&exp, "control", MAB_VARIANT_CONTROL, 50, NULL);
        mab_experiment_add_variant(&exp, "treat", MAB_VARIANT_TREATMENT, 50, NULL);
        printf("  [Experiment] %s: %d variants, valid=%s\n", exp.name,
               exp.num_variants, mab_experiment_validate(&exp) ? "yes" : "no");

        mab_experiment_activate(&exp);
        int ctrl_cnt = 0, treat_cnt = 0;
        for (int uid = 1000; uid < 2000; uid++) {
            int v = mab_traffic_assign(uid, &exp);
            if (v == 0) ctrl_cnt++; else treat_cnt++;
        }
        printf("  [Traffic Split] control=%d treatment=%d\n", ctrl_cnt, treat_cnt);

        MABVariantMetrics ctrl_m, treat_m;
        mab_metrics_init(&ctrl_m);
        mab_metrics_init(&treat_m);
        for (int i = 0; i < 500; i++) {
            mab_metrics_record_request(&ctrl_m, 50.0 + (rand() % 20), 1, 0.8);
            mab_metrics_record_request(&treat_m, 45.0 + (rand() % 15), 1, 0.85);
        }
        printf("  [Metrics]:\n");
        print_variant_metrics(&ctrl_m, "control");
        print_variant_metrics(&treat_m, "treat");

        MABTTestResult tt;
        mab_metrics_compare(&ctrl_m, &treat_m, "success_rate", &tt);
        printf("  [t-Test] t=%.4f p=%.4f significant=%d df=%d\n",
               tt.t_statistic, tt.p_value, tt.significant, tt.df);

        MABChiSquareResult chi;
        mab_chi_square_test(ctrl_m.success_rate, ctrl_m.total_requests,
                            treat_m.success_rate, treat_m.total_requests, &chi);
        printf("  [Chi-Square] chi2=%.4f p=%.4f significant=%d\n",
               chi.chi2_stat, chi.p_value, chi.significant);

        MABRampUpPlan ramp;
        mab_ramp_up_init(&ramp, 1, 3600);
        MABGuardrails guards;
        mab_guardrails_init(&guards);
        mab_guardrails_add(&guards, "latency", 200.0f, 1);
        mab_guardrails_add(&guards, "error_rate", 0.01f, 1);
        MABRollbackDecision rb;
        mab_ramp_up_execute(&exp, &ramp, &guards, &rb);
        printf("  [Ramp-Up] Rollback needed: %s\n", rb.should_rollback ? "YES" : "NO");

        mab_rollback_evaluate(&ctrl_m, &treat_m, &guards, &rb);
        printf("  [Rollback Eval] Should rollback: %s (%s)\n",
               rb.should_rollback ? "YES" : "NO", rb.reason);

        MABModelRegistry reg;
        mab_model_registry_init(&reg, "recommendation_model", "v3.2.1",
                                MAB_STAGE_STAGING, NULL);
        printf("  [Registry] %s v%s stage=%d can_serve=%d\n",
               reg.name, reg.version, reg.stage, mab_model_registry_can_serve(&reg));
        mab_model_registry_promote(&reg, MAB_STAGE_PRODUCTION);
        printf("  [Registry] Promoted to stage=%d can_serve=%d\n",
               reg.stage, mab_model_registry_can_serve(&reg));

        MABShadowTest shadow;
        mab_shadow_test_init(&shadow);
        for (int i = 0; i < 100; i++)
            mab_shadow_test_record(&shadow, 0.92, 0.94, 55.0, 1);
        mab_shadow_test_evaluate(&shadow, 0.05);
        printf("  [Shadow Test] passed=%s score_delta=%.4f req=%d\n",
               shadow.passed ? "YES" : "NO", shadow.score_delta, shadow.total_requests);

        mab_feature_flag_set("new_recommender", 1);
        int ff = mab_feature_flag_user("new_recommender", 42, 50);
        printf("  [Feature Flag] new_recommender for user 42 at 50%%: %s\n",
               ff ? "ENABLED" : "DISABLED");

        MABExperimentMetrics exp_metrics;
        MABVariantMetrics per_var[2] = { ctrl_m, treat_m };
        mab_metrics_collect(&exp, per_var, &exp_metrics);
        char report[4096];
        mab_report_generate(&exp, &exp_metrics, report, 4096);
        printf("  [Report]:\n%s", report);

        int sample_n = mab_sample_size_estimate(0.10, 0.02, 0.05, 0.80);
        printf("  [Sample Estimate] ~%d per variant\n", sample_n);
    }

    printf("\n================================================================\n");
    printf("  All systems demonstrated successfully.\n");
    printf("================================================================\n");
    return 0;
}
