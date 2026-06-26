#ifndef AGENT_SAFETY_H
#define AGENT_SAFETY_H

#include <stddef.h>
#include <stdbool.h>

/**
 * agent_safety.h -- Agent Safety and Guardrails Framework
 *
 * L2 Core Concepts: Content safety, prompt injection defense, rate limiting
 * L4 Standards: OWASP Top 10 for LLM Applications (2023, 2025)
 *   - LLM01: Prompt Injection
 *   - LLM02: Insecure Output Handling
 *   - LLM03: Training Data Poisoning
 *   - LLM04: Model Denial of Service
 * L7 Application: AI safety guardrails for production agent deployment
 * L8 Advanced: Statistical anomaly detection, adversarial prompt detection
 * L9 Industry Frontiers: RLHF safety training, Constitutional AI principles
 *
 * Course mapping:
 *   - MIT 6.858 (Computer Security): Adversarial input handling
 *   - Stanford CS 329S (ML Systems): ML safety and security
 *   - Berkeley CS 294 (AI Systems): AI alignment and safety
 *   - CMU 17-537 (AI Safety): Content moderation systems
 */

#define SAFETY_MAX_KEYWORDS      256
#define SAFETY_MAX_PATTERNS      128
#define SAFETY_MAX_BLOCKED_WORDS 512
#define SAFETY_MAX_MSG_LEN       16384
#define SAFETY_MAX_RATE_WINDOW   64
#define SAFETY_MAX_PII_PATTERNS  32
#define SAFETY_MAX_ALLOWED_DOMAINS 128
#define SAFETY_MAX_MODEL_RESPONSE 65536
#define SAFETY_MAX_INJECTION_SIGS 64
#define SAFETY_DEFAULT_RATE_LIMIT 60
#define SAFETY_DEFAULT_TIME_WINDOW 60

/* ── L1: Core type definitions ── */

typedef enum {
    SAFETY_LEVEL_NONE,
    SAFETY_LEVEL_LOW,
    SAFETY_LEVEL_MODERATE,
    SAFETY_LEVEL_HIGH,
    SAFETY_LEVEL_STRICT
} safety_level_t;

typedef enum {
    SAFETY_CAT_HARASSMENT,
    SAFETY_CAT_HATE_SPEECH,
    SAFETY_CAT_SEXUAL_CONTENT,
    SAFETY_CAT_VIOLENCE,
    SAFETY_CAT_SELF_HARM,
    SAFETY_CAT_PROMPT_INJECTION,
    SAFETY_CAT_PII_LEAK,
    SAFETY_CAT_JAILBREAK,
    SAFETY_CAT_MISINFORMATION,
    SAFETY_CAT_DOS_ATTACK,
    SAFETY_CAT_CODE_INJECTION,
    SAFETY_CAT_UNSAFE_OUTPUT,
    SAFETY_CAT_COUNT
} safety_category_t;

typedef enum {
    SAFETY_CHECK_PASS,
    SAFETY_CHECK_FLAG,
    SAFETY_CHECK_BLOCK,
    SAFETY_CHECK_ERROR
} safety_check_t;

typedef struct {
    safety_category_t category;
    safety_check_t result;
    double confidence;
    char reason[1024];
    char matched_pattern[512];
    int severity;
} safety_check_result_t;

typedef struct {
    safety_check_result_t results[16];
    int result_count;
    safety_check_t overall;
} safety_report_t;

typedef struct {
    char pattern[256];
    safety_category_t category;
    int severity;
    bool is_regex;
} safety_pattern_t;

typedef struct {
    safety_pattern_t patterns[SAFETY_MAX_PATTERNS];
    int pattern_count;
} safety_rule_set_t;

typedef struct {
    time_t timestamp;
    int request_count;
} rate_limit_entry_t;

typedef struct {
    rate_limit_entry_t entries[SAFETY_MAX_RATE_WINDOW];
    int entry_count;
    int max_requests;
    int time_window_seconds;
} rate_limiter_t;

typedef struct {
    char blocked_words[SAFETY_MAX_BLOCKED_WORDS][64];
    int blocked_count;
} content_filter_t;

typedef struct {
    char pattern[128];
    char description[256];
    safety_category_t category;
} pii_pattern_t;

typedef struct {
    pii_pattern_t patterns[SAFETY_MAX_PII_PATTERNS];
    int pattern_count;
} pii_detector_t;

typedef struct {
    char signatures[SAFETY_MAX_INJECTION_SIGS][128];
    int signature_count;
} injection_detector_t;

typedef struct {
    safety_rule_set_t rules;
    content_filter_t filter;
    pii_detector_t pii_detect;
    injection_detector_t injection;
    rate_limiter_t rate_limit;
    safety_level_t level;
    bool enabled;
    int blocks_total;
    int flags_total;
} agent_safety_t;

/* ── L1: API declarations ── */

agent_safety_t* agent_safety_create(safety_level_t level);
void agent_safety_destroy(agent_safety_t *safety);

/* L2: Core safety checks */
safety_report_t agent_safety_check_input(agent_safety_t *safety, const char *input);
safety_report_t agent_safety_check_output(agent_safety_t *safety, const char *output,
                                           const char *input_context);
safety_report_t agent_safety_audit_all(agent_safety_t *safety, const char *input,
                                        const char *output);

/* L5: Prompt Injection Detection (signature-based + heuristic)
 * Theorem: Detects known injection patterns using substring matching (KMP-like)
 * Reference: Perez & Ribeiro, "Ignore Previous Prompt" (2022)
 *            Liu et al., "Prompt Injection Attacks and Defenses" (2024)
 */
safety_check_result_t safety_detect_injection(injection_detector_t *det, const char *text);
int safety_add_injection_signature(injection_detector_t *det, const char *sig);
int injection_detector_load_defaults(injection_detector_t *det);

/* L5: Content filtering with blocked word list
 * Time complexity: O(n*m) where n = text length, m = num blocked words
 * Space complexity: O(1) additional
 */
safety_check_result_t safety_filter_content(content_filter_t *cf, const char *text,
                                              safety_category_t category);
bool safety_add_blocked_word(content_filter_t *cf, const char *word);
void content_filter_load_defaults(content_filter_t *cf);

/* L5: PII (Personally Identifiable Information) detection
 * Detects: email, phone, SSN, credit card, IP address patterns
 * Reference: NIST SP 800-122 (Guide to PII Protection)
 */
safety_check_result_t safety_detect_pii(pii_detector_t *pd, const char *text);
bool safety_add_pii_pattern(pii_detector_t *pd, const char *pattern, safety_category_t cat,
                              const char *desc);
void pii_detector_load_defaults(pii_detector_t *pd);

/* L3: Rate limiter using sliding window
 * Theoretical basis: Token bucket algorithm generalization
 * Reference: TPC Benchmark specifications
 */
bool rate_limiter_check(rate_limiter_t *rl);
void rate_limiter_reset(rate_limiter_t *rl);
int rate_limiter_remaining(const rate_limiter_t *rl);

/* L3: Safety rule matching engine */
safety_check_result_t safety_rule_match(safety_rule_set_t *rules, const char *text);
bool safety_add_rule(safety_rule_set_t *rules, const char *pattern,
                     safety_category_t cat, int severity);

/* L8: Jailbreak detection heuristics
 * Detects: role-play subversion, multi-turn attacks, encoding tricks
 * Reference: Wei et al., "Jailbroken: How Does LLM Safety Training Fail?" (2023)
 */
safety_check_result_t safety_detect_jailbreak(const char *text);

/* L8: Statistical anomaly detection for input patterns
 * Uses: z-score analysis on input length, entropy, and token distribution
 */
typedef struct {
    double input_mean_len;
    double input_std_len;
    double entropy_threshold;
    int sample_count;
} anomaly_detector_t;

bool safety_detect_anomaly(anomaly_detector_t *ad, const char *input);
void anomaly_detector_update(anomaly_detector_t *ad, const char *input);

/* Utility */
const char* safety_category_name(safety_category_t cat);
const char* safety_check_name(safety_check_t check);
const char* safety_report_format(const safety_report_t *report, char *buf, size_t buf_size);
bool safety_is_safe(const safety_report_t *report);

#endif
