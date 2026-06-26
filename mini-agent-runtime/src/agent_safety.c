#include "agent_safety.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>
#include <time.h>
#include <math.h>

/* ================================================================
 * agent_safety.c -- Agent Safety and Guardrails Implementation
 *
 * L2 Core Concepts: Content safety, prompt injection defense, rate limiting
 * L4 Standards: OWASP Top 10 for LLM Applications
 * L5 Algorithms: Signature-based injection detection, content filtering,
 *                PII pattern matching, sliding window rate limiter
 * L7 Application: Production AI safety guardrails
 * L8 Advanced: Jailbreak heuristics, anomaly detection (Welford online)
 * ================================================================ */

agent_safety_t* agent_safety_create(safety_level_t level) {
    agent_safety_t *s = (agent_safety_t*)calloc(1, sizeof(agent_safety_t));
    if (!s) return NULL;
    s->level = level;
    s->enabled = true;
    s->rate_limit.max_requests = SAFETY_DEFAULT_RATE_LIMIT;
    s->rate_limit.time_window_seconds = SAFETY_DEFAULT_TIME_WINDOW;
    content_filter_load_defaults(&s->filter);
    injection_detector_load_defaults(&s->injection);
    pii_detector_load_defaults(&s->pii_detect);
    return s;
}

void agent_safety_destroy(agent_safety_t *safety) {
    if (safety) free(safety);
}

/* ---- Core safety checks ---- */

safety_report_t agent_safety_check_input(agent_safety_t *safety, const char *input) {
    safety_report_t report;
    memset(&report, 0, sizeof(report));
    report.overall = SAFETY_CHECK_PASS;
    if (!safety || !safety->enabled || !input) return report;

    if (safety->level >= SAFETY_LEVEL_LOW) {
        safety_check_result_t r = safety_filter_content(&safety->filter, input,
                                                          SAFETY_CAT_HARASSMENT);
        if (r.result >= SAFETY_CHECK_FLAG) {
            report.results[report.result_count++] = r;
            if (r.result == SAFETY_CHECK_BLOCK) report.overall = SAFETY_CHECK_BLOCK;
        }
    }

    if (safety->level >= SAFETY_LEVEL_MODERATE) {
        safety_check_result_t r = safety_detect_injection(&safety->injection, input);
        if (r.result >= SAFETY_CHECK_FLAG) {
            report.results[report.result_count++] = r;
            if (r.result == SAFETY_CHECK_BLOCK) report.overall = SAFETY_CHECK_BLOCK;
        }
        r = safety_detect_pii(&safety->pii_detect, input);
        if (r.result >= SAFETY_CHECK_FLAG) {
            report.results[report.result_count++] = r;
            if (r.result == SAFETY_CHECK_BLOCK)
                report.overall = SAFETY_CHECK_BLOCK;
            else if (report.overall < SAFETY_CHECK_FLAG)
                report.overall = SAFETY_CHECK_FLAG;
        }
    }

    if (safety->level >= SAFETY_LEVEL_HIGH) {
        safety_check_result_t r = safety_detect_jailbreak(input);
        if (r.result >= SAFETY_CHECK_FLAG) {
            report.results[report.result_count++] = r;
            if (r.result == SAFETY_CHECK_BLOCK) report.overall = SAFETY_CHECK_BLOCK;
        }
    }

    if (report.result_count == 0) report.overall = SAFETY_CHECK_PASS;
    return report;
}

safety_report_t agent_safety_check_output(agent_safety_t *safety, const char *output,
                                           const char *input_context) {
    (void)input_context;
    safety_report_t report;
    memset(&report, 0, sizeof(report));
    report.overall = SAFETY_CHECK_PASS;
    if (!safety || !safety->enabled || !output) return report;

    safety_check_result_t r = safety_filter_content(&safety->filter, output,
                                                      SAFETY_CAT_UNSAFE_OUTPUT);
    if (r.result >= SAFETY_CHECK_FLAG) {
        report.results[report.result_count++] = r;
        if (r.result == SAFETY_CHECK_BLOCK) report.overall = SAFETY_CHECK_BLOCK;
    }

    if (safety->level >= SAFETY_LEVEL_HIGH) {
        r = safety_detect_pii(&safety->pii_detect, output);
        if (r.result >= SAFETY_CHECK_FLAG)
            report.results[report.result_count++] = r;
    }

    if (report.result_count == 0) report.overall = SAFETY_CHECK_PASS;
    return report;
}

safety_report_t agent_safety_audit_all(agent_safety_t *safety, const char *input,
                                        const char *output) {
    safety_report_t in_report = agent_safety_check_input(safety, input);
    safety_report_t out_report = agent_safety_check_output(safety, output, input);
    safety_report_t combined = in_report;
    for (int i = 0; i < out_report.result_count && combined.result_count < 16; i++)
        combined.results[combined.result_count++] = out_report.results[i];
    if (out_report.overall == SAFETY_CHECK_BLOCK) combined.overall = SAFETY_CHECK_BLOCK;
    return combined;
}

/* ---- Prompt Injection Detection ---- */

safety_check_result_t safety_detect_injection(injection_detector_t *det, const char *text) {
    safety_check_result_t res;
    memset(&res, 0, sizeof(res));
    res.category = SAFETY_CAT_PROMPT_INJECTION;
    res.result = SAFETY_CHECK_PASS;
    if (!det || !text) return res;

    for (int i = 0; i < det->signature_count; i++) {
        if (strstr(text, det->signatures[i])) {
            res.result = SAFETY_CHECK_BLOCK;
            res.confidence = 0.9;
            res.severity = 5;
            strncpy(res.matched_pattern, det->signatures[i],
                    sizeof(res.matched_pattern) - 1);
            snprintf(res.reason, sizeof(res.reason),
                     "Detected injection signature: %s", det->signatures[i]);
            return res;
        }
    }

    /* Heuristic: role-rebinding */
    const char *sys = strstr(text, "system:");
    if (!sys) sys = strstr(text, "System:");
    if (sys) {
        const char *nw = sys + 7;
        while (*nw == ' ') nw++;
        if (strncmp(nw, "you are", 7) == 0 ||
            strncmp(nw, "now you", 7) == 0 ||
            strncmp(nw, "prompt", 6) == 0) {
            res.result = SAFETY_CHECK_BLOCK;
            res.confidence = 0.85;
            res.severity = 4;
            strncpy(res.matched_pattern, "role-rebinding",
                    sizeof(res.matched_pattern) - 1);
            snprintf(res.reason, sizeof(res.reason),
                     "Potential role-rebinding attempt");
            return res;
        }
    }

    /* Heuristic: excessive delimiters */
    int delim = 0;
    for (int i = 0; text[i]; i++)
        if (text[i] == '-' || text[i] == '=' || text[i] == '#') delim++;
    if (delim > 20) {
        res.result = SAFETY_CHECK_FLAG;
        res.confidence = 0.6;
        res.severity = 2;
        strncpy(res.matched_pattern, "excessive-delimiters",
                sizeof(res.matched_pattern) - 1);
        snprintf(res.reason, sizeof(res.reason),
                 "Excessive delimiter chars (%d)", delim);
        return res;
    }
    return res;
}

int safety_add_injection_signature(injection_detector_t *det, const char *sig) {
    if (!det || !sig || det->signature_count >= SAFETY_MAX_INJECTION_SIGS) return 0;
    strncpy(det->signatures[det->signature_count], sig,
            sizeof(det->signatures[0]) - 1);
    det->signature_count++;
    return 1;
}

int injection_detector_load_defaults(injection_detector_t *det) {
    if (!det) return 0;
    const char *def[] = {
        "ignore previous instructions", "ignore all previous",
        "disregard previous", "forget your instructions",
        "your new instructions", "you are now DAN", "DAN mode",
        "jailbreak", "system prompt:", "you are an unrestricted",
        "pretend you are", "act as if you are", "from now on you are",
        "[INST]", "do anything now", "without any restrictions",
        "bypass your safety", "new system prompt", NULL
    };
    int c = 0;
    for (int i = 0; def[i]; i++)
        if (safety_add_injection_signature(det, def[i])) c++;
    return c;
}

/* ---- Content Filtering ---- */

static int ci_strstr(const char *h, const char *n) {
    if (!h || !n) return 0;
    size_t nl = strlen(n);
    if (nl == 0) return 0;
    for (size_t i = 0; h[i]; i++) {
        size_t j;
        for (j = 0; j < nl && h[i+j]; j++)
            if (tolower((unsigned char)h[i+j]) != tolower((unsigned char)n[j])) break;
        if (j == nl) return 1;
    }
    return 0;
}

safety_check_result_t safety_filter_content(content_filter_t *cf, const char *text,
                                              safety_category_t category) {
    safety_check_result_t res;
    memset(&res, 0, sizeof(res));
    res.category = category;
    res.result = SAFETY_CHECK_PASS;
    if (!cf || !text) return res;
    for (int i = 0; i < cf->blocked_count; i++) {
        if (ci_strstr(text, cf->blocked_words[i])) {
            res.result = SAFETY_CHECK_BLOCK;
            res.confidence = 0.95;
            res.severity = 4;
            strncpy(res.matched_pattern, cf->blocked_words[i],
                    sizeof(res.matched_pattern) - 1);
            snprintf(res.reason, sizeof(res.reason),
                     "Blocked content: %s", cf->blocked_words[i]);
            return res;
        }
    }
    return res;
}

bool safety_add_blocked_word(content_filter_t *cf, const char *word) {
    if (!cf || !word || cf->blocked_count >= SAFETY_MAX_BLOCKED_WORDS) return false;
    strncpy(cf->blocked_words[cf->blocked_count], word, 63);
    cf->blocked_count++;
    return true;
}

void content_filter_load_defaults(content_filter_t *cf) {
    if (!cf) return;
    const char *w[] = {"hack", "exploit", "backdoor", "trojan", "keylogger",
                       "ransomware", "malware", "phishing", "spyware", NULL};
    for (int i = 0; w[i]; i++) safety_add_blocked_word(cf, w[i]);
}

/* ---- PII Detection ---- */

static int is_email_pat(const char *s) {
    const char *at = strchr(s, '@');
    if (!at || at == s) return 0;
    const char *dot = strchr(at, '.');
    return dot && dot != at + 1;
}

static int is_ssn_pat(const char *s) {
    int d = 0, h = 0;
    for (int i = 0; s[i]; i++) {
        if (isdigit((unsigned char)s[i])) d++;
        else if (s[i] == '-') h++;
    }
    return d == 9 && h == 2;
}

static int is_cc_pat(const char *s) {
    int d = 0, sep = 0;
    for (int i = 0; s[i]; i++) {
        if (isdigit((unsigned char)s[i])) d++;
        else if (s[i] == '-' || s[i] == ' ') sep++;
    }
    return d >= 13 && d <= 19 && (sep == 3);
}

static int is_phone_pat(const char *s) {
    int d = 0, sep = 0;
    for (int i = 0; s[i]; i++) {
        if (isdigit((unsigned char)s[i])) d++;
        else if (s[i] == '-' || s[i] == '(' || s[i] == ')') sep++;
    }
    return d >= 10 && d <= 15 && sep > 0;
}

static int is_ip_pat(const char *s) {
    int dot = 0, dg = 0;
    for (int i = 0; s[i]; i++) {
        if (isdigit((unsigned char)s[i])) dg++;
        else if (s[i] == '.') dot++;
        else return 0;
    }
    return dot == 3 && dg >= 4 && dg <= 12;
}

safety_check_result_t safety_detect_pii(pii_detector_t *pd, const char *text) {
    safety_check_result_t res;
    memset(&res, 0, sizeof(res));
    res.category = SAFETY_CAT_PII_LEAK;
    res.result = SAFETY_CHECK_PASS;
    if (!pd || !text) return res;

    char buf[SAFETY_MAX_MSG_LEN];
    strncpy(buf, text, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';
    char *tok = strtok(buf, " \t\n\r,;:()[]{}<>\"'");
    while (tok) {
        if (is_email_pat(tok)) {
            res.result = SAFETY_CHECK_BLOCK; res.confidence = 0.95; res.severity = 5;
            strncpy(res.matched_pattern, tok, sizeof(res.matched_pattern) - 1);
            snprintf(res.reason, sizeof(res.reason), "PII: email address");
            return res;
        }
        if (is_ssn_pat(tok)) {
            res.result = SAFETY_CHECK_BLOCK; res.confidence = 0.9; res.severity = 5;
            strncpy(res.matched_pattern, "SSN", sizeof(res.matched_pattern) - 1);
            snprintf(res.reason, sizeof(res.reason), "PII: SSN pattern");
            return res;
        }
        if (is_cc_pat(tok)) {
            res.result = SAFETY_CHECK_BLOCK; res.confidence = 0.85; res.severity = 5;
            strncpy(res.matched_pattern, "CCN", sizeof(res.matched_pattern) - 1);
            snprintf(res.reason, sizeof(res.reason), "PII: credit card number");
            return res;
        }
        if (is_phone_pat(tok)) {
            res.result = SAFETY_CHECK_FLAG; res.confidence = 0.7; res.severity = 3;
            strncpy(res.matched_pattern, tok, sizeof(res.matched_pattern) - 1);
            snprintf(res.reason, sizeof(res.reason), "PII: phone number");
            return res;
        }
        if (is_ip_pat(tok)) {
            res.result = SAFETY_CHECK_FLAG; res.confidence = 0.6; res.severity = 2;
            strncpy(res.matched_pattern, tok, sizeof(res.matched_pattern) - 1);
            snprintf(res.reason, sizeof(res.reason), "PII: IP address");
            return res;
        }
        tok = strtok(NULL, " \t\n\r,;:()[]{}<>\"'");
    }
    return res;
}

bool safety_add_pii_pattern(pii_detector_t *pd, const char *pattern,
                              safety_category_t cat, const char *desc) {
    if (!pd || !pattern || pd->pattern_count >= SAFETY_MAX_PII_PATTERNS) return false;
    pii_pattern_t *pp = &pd->patterns[pd->pattern_count];
    strncpy(pp->pattern, pattern, sizeof(pp->pattern) - 1);
    pp->category = cat;
    if (desc) strncpy(pp->description, desc, sizeof(pp->description) - 1);
    pd->pattern_count++;
    return true;
}

void pii_detector_load_defaults(pii_detector_t *pd) {
    if (!pd) return;
    safety_add_pii_pattern(pd, "email", SAFETY_CAT_PII_LEAK, "Email address");
    safety_add_pii_pattern(pd, "ssn", SAFETY_CAT_PII_LEAK, "US Social Security Number");
    safety_add_pii_pattern(pd, "credit_card", SAFETY_CAT_PII_LEAK, "Credit card number");
    safety_add_pii_pattern(pd, "phone", SAFETY_CAT_PII_LEAK, "Phone number");
    safety_add_pii_pattern(pd, "ip_address", SAFETY_CAT_PII_LEAK, "IP address");
}

/* ---- Rate Limiter (Sliding Window) ---- */

bool rate_limiter_check(rate_limiter_t *rl) {
    if (!rl) return true;
    time_t now = time(NULL), cutoff = now - rl->time_window_seconds;
    int recent = 0;
    for (int i = 0; i < rl->entry_count; i++)
        if (rl->entries[i].timestamp >= cutoff)
            recent += rl->entries[i].request_count;
    if (recent >= rl->max_requests) return false;
    if (rl->entry_count < SAFETY_MAX_RATE_WINDOW) {
        rl->entries[rl->entry_count].timestamp = now;
        rl->entries[rl->entry_count].request_count = 1;
        rl->entry_count++;
    } else {
        int oldest = 0;
        for (int i = 1; i < rl->entry_count; i++)
            if (rl->entries[i].timestamp < rl->entries[oldest].timestamp) oldest = i;
        rl->entries[oldest].timestamp = now;
        rl->entries[oldest].request_count = 1;
    }
    return true;
}

void rate_limiter_reset(rate_limiter_t *rl) {
    if (rl) rl->entry_count = 0;
}

int rate_limiter_remaining(const rate_limiter_t *rl) {
    if (!rl) return 0;
    time_t now = time(NULL), cutoff = now - rl->time_window_seconds;
    int recent = 0;
    for (int i = 0; i < rl->entry_count; i++)
        if (rl->entries[i].timestamp >= cutoff)
            recent += rl->entries[i].request_count;
    int rem = rl->max_requests - recent;
    return rem > 0 ? rem : 0;
}

/* ---- Safety Rule Matching ---- */

safety_check_result_t safety_rule_match(safety_rule_set_t *rules, const char *text) {
    safety_check_result_t res;
    memset(&res, 0, sizeof(res));
    res.result = SAFETY_CHECK_PASS;
    if (!rules || !text) return res;
    for (int i = 0; i < rules->pattern_count; i++) {
        if (strstr(text, rules->patterns[i].pattern)) {
            res.result = SAFETY_CHECK_FLAG;
            res.category = rules->patterns[i].category;
            res.severity = rules->patterns[i].severity;
            res.confidence = 0.8;
            strncpy(res.matched_pattern, rules->patterns[i].pattern,
                    sizeof(res.matched_pattern) - 1);
            snprintf(res.reason, sizeof(res.reason),
                     "Rule matched: %s", rules->patterns[i].pattern);
            return res;
        }
    }
    return res;
}

bool safety_add_rule(safety_rule_set_t *rules, const char *pattern,
                     safety_category_t cat, int severity) {
    if (!rules || !pattern || rules->pattern_count >= SAFETY_MAX_PATTERNS) return false;
    safety_pattern_t *sp = &rules->patterns[rules->pattern_count];
    strncpy(sp->pattern, pattern, sizeof(sp->pattern) - 1);
    sp->category = cat;
    sp->severity = severity;
    sp->is_regex = false;
    rules->pattern_count++;
    return true;
}

/* ---- Jailbreak Detection ---- */

safety_check_result_t safety_detect_jailbreak(const char *text) {
    safety_check_result_t res;
    memset(&res, 0, sizeof(res));
    res.category = SAFETY_CAT_JAILBREAK;
    res.result = SAFETY_CHECK_PASS;
    if (!text) return res;

    const char *tokens[] = {
        "DAN", "STAN", "DUDE", "Developer Mode", "jailbreak", "jail break",
        "God mode", "no restrictions", "no limits", "you are no longer",
        "you are now free", "pretend you are", "act as if", "imagine you are",
        "you are a different", "you have been reprogrammed", "ignore safety",
        "bypass content", "override safety", "without ethical",
        "without constraints", "do anything", "anything now", NULL
    };

    int hits = 0;
    for (int i = 0; tokens[i]; i++)
        if (ci_strstr(text, tokens[i])) {
            hits++;
            if (hits == 1)
                strncpy(res.matched_pattern, tokens[i], sizeof(res.matched_pattern) - 1);
        }

    if (hits >= 3) {
        res.result = SAFETY_CHECK_BLOCK;
        res.confidence = 0.92; res.severity = 5;
        snprintf(res.reason, sizeof(res.reason),
                 "Jailbreak: %d tokens detected", hits);
    } else if (hits >= 1) {
        res.result = SAFETY_CHECK_FLAG;
        res.confidence = 0.7; res.severity = 3;
        snprintf(res.reason, sizeof(res.reason),
                 "Potential jailbreak: %d token(s)", hits);
    }

    /* Encoding trick detection */
    int b64c = 0, hexc = 0, tlen = (int)strlen(text);
    for (int i = 0; text[i]; i++) {
        unsigned char c = (unsigned char)text[i];
        if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
            (c >= '0' && c <= '9') || c == '+' || c == '/' || c == '=') b64c++;
        if ((c >= '0' && c <= '9') || (c >= 'A' && c <= 'F') ||
            (c >= 'a' && c <= 'f') || c == ' ' || c == 'x') hexc++;
    }
    if (tlen > 20 && b64c > tlen * 0.9) {
        res.result = SAFETY_CHECK_FLAG; res.confidence = 0.6; res.severity = 2;
        strncpy(res.matched_pattern, "base64", sizeof(res.matched_pattern) - 1);
        snprintf(res.reason, sizeof(res.reason), "Possible encoded payload (base64)");
    } else if (tlen > 10 && hexc > tlen * 0.95) {
        res.result = SAFETY_CHECK_FLAG; res.confidence = 0.55; res.severity = 2;
        strncpy(res.matched_pattern, "hex", sizeof(res.matched_pattern) - 1);
        snprintf(res.reason, sizeof(res.reason), "Possible encoded payload (hex)");
    }
    return res;
}

/* ---- Anomaly Detection (Welford Online Algorithm) ---- */

bool safety_detect_anomaly(anomaly_detector_t *ad, const char *input) {
    if (!ad || !input || ad->sample_count < 10) return false;
    double l = (double)strlen(input);
    double z = (l - ad->input_mean_len) / (ad->input_std_len + 1e-10);
    return fabs(z) > 3.0;
}

void anomaly_detector_update(anomaly_detector_t *ad, const char *input) {
    if (!ad || !input) return;
    double x = (double)strlen(input);
    ad->sample_count++;
    double delta = x - ad->input_mean_len;
    ad->input_mean_len += delta / ad->sample_count;
    double old_var = ad->input_std_len * ad->input_std_len;
    ad->input_std_len = sqrt((old_var * (ad->sample_count - 1) +
                              delta * (x - ad->input_mean_len)) /
                             (ad->sample_count > 1 ? (ad->sample_count - 1) : 1));
    if (ad->input_std_len < 1.0) ad->input_std_len = 1.0;
}

/* ---- Utility ---- */

const char* safety_category_name(safety_category_t cat) {
    switch (cat) {
        case SAFETY_CAT_HARASSMENT:       return "Harassment";
        case SAFETY_CAT_HATE_SPEECH:      return "Hate Speech";
        case SAFETY_CAT_SEXUAL_CONTENT:   return "Sexual Content";
        case SAFETY_CAT_VIOLENCE:         return "Violence";
        case SAFETY_CAT_SELF_HARM:        return "Self-Harm";
        case SAFETY_CAT_PROMPT_INJECTION: return "Prompt Injection";
        case SAFETY_CAT_PII_LEAK:         return "PII Leak";
        case SAFETY_CAT_JAILBREAK:        return "Jailbreak";
        case SAFETY_CAT_MISINFORMATION:   return "Misinformation";
        case SAFETY_CAT_DOS_ATTACK:       return "DoS Attack";
        case SAFETY_CAT_CODE_INJECTION:   return "Code Injection";
        case SAFETY_CAT_UNSAFE_OUTPUT:    return "Unsafe Output";
        default: return "Unknown";
    }
}

const char* safety_check_name(safety_check_t check) {
    switch (check) {
        case SAFETY_CHECK_PASS:  return "PASS";
        case SAFETY_CHECK_FLAG:  return "FLAG";
        case SAFETY_CHECK_BLOCK: return "BLOCK";
        case SAFETY_CHECK_ERROR: return "ERROR";
        default: return "?";
    }
}

const char* safety_report_format(const safety_report_t *report, char *buf, size_t buf_size) {
    if (!report || !buf) return NULL;
    int off = snprintf(buf, buf_size, "Safety Report [%s]\n",
                       safety_check_name(report->overall));
    for (int i = 0; i < report->result_count && i < 16; i++) {
        off += snprintf(buf + off, buf_size - off,
                        "  [%s] %s: %s (%.0f%%)\n",
                        safety_check_name(report->results[i].result),
                        safety_category_name(report->results[i].category),
                        report->results[i].reason,
                        report->results[i].confidence * 100.0);
    }
    return buf;
}

bool safety_is_safe(const safety_report_t *report) {
    return report && report->overall == SAFETY_CHECK_PASS;
}
