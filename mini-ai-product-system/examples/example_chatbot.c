#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "chatbot_rlhf.h"

int main(void) {
    uint64_t seed = 12345;
    printf("=== ChatBot RLHF Pipeline Demo ===\n\n");

    CBBaseModel base_model;
    cb_base_model_init(&base_model, seed);
    printf("Base model initialized (vocab=%d, hidden=%d)\n", CB_VOCAB_SIZE, CB_HIDDEN_DIM);

    CBInput demos[3];
    CBOutput labels[3];
    const char *demo_texts[] = {
        "Hello, how can I help you?",
        "The capital of France is Paris.",
        "Here is the code you requested."
    };
    for (int i = 0; i < 3; i++) {
        cb_tokenize(demo_texts[i], demos[i].input_ids, &demos[i].input_len, CB_MAX_PROMPT_LEN);
        labels[i] = (CBOutput){0};
        labels[i].length = demos[i].input_len;
        memcpy(labels[i].token_ids, demos[i].input_ids,
               (size_t)demos[i].input_len * sizeof(int32_t));
    }
    cb_sft_train(&base_model, demos, labels, 3, 0.001f, 2);
    printf("SFT training completed (3 demos, 2 epochs)\n");

    CBRewardModel reward_model;
    cb_reward_model_init(&reward_model, seed + 1);
    printf("Reward model initialized\n");

    CBPreferencePair pref_pairs[2];
    memset(&pref_pairs, 0, sizeof(pref_pairs));
    cb_tokenize("What is Python?", pref_pairs[0].prompt.input_ids,
                &pref_pairs[0].prompt.input_len, CB_MAX_PROMPT_LEN);
    cb_tokenize("Python is a programming language. It's great for AI.",
                pref_pairs[0].chosen.token_ids, &pref_pairs[0].chosen.length, CB_MAX_RESPONSE_LEN);
    pref_pairs[0].chosen.total_logprob = -1.5f;
    cb_tokenize("Python is a snake.",
                pref_pairs[0].rejected.token_ids, &pref_pairs[0].rejected.length, CB_MAX_RESPONSE_LEN);
    pref_pairs[0].rejected.total_logprob = -3.0f;
    cb_tokenize("Write a function", pref_pairs[1].prompt.input_ids,
                &pref_pairs[1].prompt.input_len, CB_MAX_PROMPT_LEN);
    cb_tokenize("def hello(): return 'world'",
                pref_pairs[1].chosen.token_ids, &pref_pairs[1].chosen.length, CB_MAX_RESPONSE_LEN);
    pref_pairs[1].chosen.total_logprob = -2.0f;
    cb_tokenize("I don't know.",
                pref_pairs[1].rejected.token_ids, &pref_pairs[1].rejected.length, CB_MAX_RESPONSE_LEN);
    pref_pairs[1].rejected.total_logprob = -5.0f;
    cb_reward_model_train(&reward_model, pref_pairs, 2, 0.01f, 3);
    printf("Reward model trained (2 pairs, 3 epochs)\n");

    float r_base = cb_reward_model_forward(&reward_model,
        pref_pairs[0].prompt.input_ids, pref_pairs[0].prompt.input_len,
        pref_pairs[0].chosen.token_ids, pref_pairs[0].chosen.length);
    float r_rej = cb_reward_model_forward(&reward_model,
        pref_pairs[0].prompt.input_ids, pref_pairs[0].prompt.input_len,
        pref_pairs[0].rejected.token_ids, pref_pairs[0].rejected.length);
    printf("Reward scores: chosen=%.4f rejected=%.4f\n", r_base, r_rej);

    CBPPOConfig ppo_cfg;
    cb_ppo_config_init(&ppo_cfg);
    CBBaseModel policy_model;
    memcpy(&policy_model, &base_model, sizeof(CBBaseModel));
    CBBaseModel ref_model;
    memcpy(&ref_model, &base_model, sizeof(CBBaseModel));
    CBInput ppo_prompts[2];
    cb_tokenize("Tell me a joke", ppo_prompts[0].input_ids,
                &ppo_prompts[0].input_len, CB_MAX_PROMPT_LEN);
    cb_tokenize("Explain gravity", ppo_prompts[1].input_ids,
                &ppo_prompts[1].input_len, CB_MAX_PROMPT_LEN);
    cb_ppo_update(&policy_model, &ref_model, &reward_model, ppo_prompts, 2, &ppo_cfg);
    printf("PPO training completed\n");

    CBDPOConfig dpo_cfg = { .dpo_beta = 0.1f, .learning_rate = 1e-5f, .epochs = 2 };
    memcpy(&dpo_cfg.model, &base_model, sizeof(CBBaseModel));
    cb_dpo_train(&base_model, pref_pairs, 2, &dpo_cfg);
    printf("DPO training completed\n");

    float dpo_loss = cb_dpo_loss(&base_model, &pref_pairs[0], 0.1f);
    printf("DPO Loss: %.4f\n", dpo_loss);

    float implicit_r = cb_implicit_reward(&base_model, &ref_model,
        pref_pairs[0].chosen.token_ids, pref_pairs[0].chosen.length, 0.1f);
    printf("Implicit Reward: %.4f\n", implicit_r);

    printf("\n--- Generation ---\n");
    int32_t prompt_ids[CB_MAX_PROMPT_LEN];
    int32_t prompt_len;
    cb_tokenize("Hello", prompt_ids, &prompt_len, CB_MAX_PROMPT_LEN);
    CBOutput gen_output;
    cb_generate(&base_model, prompt_ids, prompt_len, 20, 0.8f, seed, &gen_output);
    char gen_text[CB_MAX_RESPONSE_LEN + 1] = {0};
    cb_detokenize(gen_output.token_ids, gen_output.length, gen_text, CB_MAX_RESPONSE_LEN + 1);
    printf("Generated (%d tokens, logprob=%.2f): '%s'\n",
           gen_output.length, gen_output.total_logprob, gen_text);

    printf("\n--- AI Feedback ---\n");
    CBAIFeedback feedback;
    cb_ai_feedback_generate(&base_model, &pref_pairs[0].prompt, &feedback);
    printf("Helpfulness: %.2f  Harmlessness: %.2f  Honesty: %.2f\n",
           feedback.helpfulness, feedback.harmlessness, feedback.honesty);
    printf("Critique: %s\n", feedback.critique);

    printf("\n--- Constitutional Alignment ---\n");
    cb_constitutional_alignment(&base_model, ppo_prompts, 2, CB_AI_FEEDBACK_HARMLESS);
    printf("Alignment step completed\n");

    printf("\n--- Safety ---\n");
    const char *bad_phrases[] = {"hack", "exploit", "malware"};
    CBRedTeamSet red_team = { bad_phrases, 3 };
    int is_safe;
    int32_t safe_tokens[] = {72, 101, 108, 108, 111}; /* "Hello" */
    cb_safety_filter(safe_tokens, 5, &red_team, &is_safe);
    printf("Safety filter (hello): %s\n", is_safe ? "SAFE" : "UNSAFE");
    int32_t bad_tokens[] = {104, 97, 99, 107}; /* "hack" */
    cb_safety_filter(bad_tokens, 4, &red_team, &is_safe);
    printf("Safety filter (hack):  %s\n", is_safe ? "SAFE" : "UNSAFE");

    float harm_score = cb_harmlessness_score(&gen_output, &red_team);
    printf("Harmlessness score: %.2f\n", harm_score);

    float safety_loss;
    cb_safety_alignment_loss(&base_model, ppo_prompts, 2, &red_team, &safety_loss);
    printf("Safety alignment loss: %.4f\n", safety_loss);

    printf("\nDone.\n");
    return 0;
}
