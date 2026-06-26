#include "guardrails_gen.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <ctype.h>
#include <regex.h>

/* ──────────────────────────────────────────────
   Internal helpers
   ────────────────────────────────────────────── */
static void *safe_alloc(size_t n) {
    void *p = calloc(n, 1);
    if (!p) { fprintf(stderr, "OOM\n"); exit(1); }
    return p;
}

static bool str_contains_nocase(const char *haystack, const char *needle) {
    size_t hl = strlen(haystack), nl = strlen(needle);
    if (nl > hl) return false;
    for (size_t i = 0; i <= hl - nl; i++) {
        bool match = true;
        for (size_t j = 0; j < nl; j++) {
            if (tolower((unsigned char)haystack[i + j]) !=
                tolower((unsigned char)needle[j])) {
                match = false;
                break;
            }
        }
        if (match) return true;
    }
    return false;
}

/* ──────────────────────────────────────────────
   Guard Result
   ────────────────────────────────────────────── */
GuardResult* guard_result_create(void) {
    GuardResult *gr = safe_alloc(sizeof(GuardResult));
    gr->violations = safe_alloc(16 * sizeof(GuardViolation));
    gr->capacity   = 16;
    gr->count      = 0;
    gr->passed     = true;
    return gr;
}

void guard_result_destroy(GuardResult *gr) {
    if (!gr) return;
    for (size_t i = 0; i < gr->count; i++) free(gr->violations[i].message);
    free(gr->violations);
    free(gr);
}

void guard_add_violation(GuardResult *gr, GuardType type,
                          Severity sev, const char *msg,
                          size_t start, size_t end, float conf) {
    if (gr->count >= gr->capacity) {
        gr->capacity *= 2;
        gr->violations = realloc(gr->violations,
                                 gr->capacity * sizeof(GuardViolation));
        if (!gr->violations) { fprintf(stderr, "Guard OOM\n"); exit(1); }
    }
    GuardViolation *v = &gr->violations[gr->count];
    v->type           = type;
    v->severity       = sev;
    v->message        = msg ? strdup(msg) : NULL;
    v->position_start = start;
    v->position_end   = end;
    v->confidence     = conf;
    gr->count++;
    if (sev >= SEVERITY_HIGH) gr->passed = false;
}

/* ──────────────────────────────────────────────
   Guard Context
   ────────────────────────────────────────────── */
GuardContext* guard_context_create(void) {
    GuardContext *gc = safe_alloc(sizeof(GuardContext));
    gc->results     = guard_result_create();
    gc->should_block = false;
    gc->block_reason = NULL;
    return gc;
}

void guard_context_destroy(GuardContext *gc) {
    if (!gc) return;
    guard_result_destroy(gc->results);
    citation_list_destroy(&gc->citations);
    free(gc->block_reason);
    free(gc);
}

/* ──────────────────────────────────────────────
   Input: Prompt Injection Detection
   ────────────────────────────────────────────── */
static const char *INJECTION_PATTERNS[] = {
    "ignore previous",
    "ignore all instructions",
    "forget your",
    "you are now",
    "pretend you are",
    "system prompt:",
    "<<<",
    ">>>",
    "{{",
    "}}",
    "[system]",
    "[INST]",
    "<<SYS>>",
    "override",
    "disregard",
    "do not follow",
    "new instructions:",
    "your new task",
    "from now on",
    NULL
};

GuardResult* guard_check_injection(const char *input, size_t len) {
    GuardResult *gr = guard_result_create();
    for (const char **p = INJECTION_PATTERNS; *p; p++) {
        if (str_contains_nocase(input, *p)) {
            guard_add_violation(gr, GUARD_INPUT_INJECTION,
                                SEVERITY_HIGH, "Prompt injection detected",
                                0, len, 0.9f);
            break;
        }
    }
    /* Check for delimiter-based injection */
    const char *delims[] = {"===", "---", "###", "<<<", ">>>"};
    int delim_count = 0;
    for (size_t d = 0; d < 5; d++) {
        if (strstr(input, delims[d])) delim_count++;
    }
    if (delim_count >= 2) {
        guard_add_violation(gr, GUARD_INPUT_INJECTION,
                            SEVERITY_MEDIUM, "Suspicious delimiters detected",
                            0, len, 0.6f);
    }
    return gr;
}

/* ──────────────────────────────────────────────
   Input: Toxicity Check
   ────────────────────────────────────────────── */
static const char *TOXIC_PATTERNS[] = {
    "kill yourself", "kys", "hate you",
    "die in a fire", "worthless", "pathetic",
    NULL
};

GuardResult* guard_check_toxicity(const char *input, size_t len) {
    GuardResult *gr = guard_result_create();
    for (const char **p = TOXIC_PATTERNS; *p; p++) {
        if (str_contains_nocase(input, *p)) {
            guard_add_violation(gr, GUARD_INPUT_TOXICITY,
                                SEVERITY_MEDIUM, "Toxic content detected",
                                0, len, 0.7f);
            break;
        }
    }
    return gr;
}

/* ──────────────────────────────────────────────
   Input: PII Detection
   ────────────────────────────────────────────── */
static bool match_pii_email(const char *text, size_t len,
                             size_t *start, size_t *end) {
    for (size_t i = 0; i < len; i++) {
        if (text[i] == '@') {
            size_t s = i, e = i + 1;
            while (s > 0 && isalnum((unsigned char)text[s - 1])) s--;
            while (e < len && (isalnum((unsigned char)text[e]) || text[e] == '.')) e++;
            if (s < i && e > i + 1) {
                *start = s; *end = e;
                return true;
            }
        }
    }
    return false;
}

static bool match_pii_phone(const char *text, size_t len,
                             size_t *start, size_t *end) {
    int digits = 0;
    size_t run_start = 0;
    for (size_t i = 0; i < len; i++) {
        if (isdigit((unsigned char)text[i]) || text[i] == '-' ||
            text[i] == ' ' || text[i] == '(' || text[i] == ')' || text[i] == '+') {
            if (digits == 0) run_start = i;
            if (isdigit((unsigned char)text[i])) digits++;
        } else {
            if (digits >= 10) {
                *start = run_start; *end = i;
                return true;
            }
            digits = 0;
        }
    }
    if (digits >= 10) {
        *start = run_start; *end = len;
        return true;
    }
    return false;
}

static bool match_pii_credit_card(const char *text, size_t len,
                                   size_t *start, size_t *end) {
    int d = 0;
    size_t s = 0;
    for (size_t i = 0; i < len; i++) {
        if (isdigit((unsigned char)text[i]) || text[i] == '-' || text[i] == ' ') {
            if (d == 0) s = i;
            if (isdigit((unsigned char)text[i])) d++;
        } else {
            if (d >= 13 && d <= 19) {
                *start = s; *end = i;
                return true;
            }
            d = 0;
        }
    }
    if (d >= 13 && d <= 19) { *start = s; *end = len; return true; }
    return false;
}

GuardResult* guard_check_pii(const char *input, size_t len) {
    GuardResult *gr = guard_result_create();
    size_t s, e;
    if (match_pii_email(input, len, &s, &e)) {
        guard_add_violation(gr, GUARD_INPUT_PII, SEVERITY_HIGH,
                            "Email address detected", s, e, 0.95f);
    }
    if (match_pii_phone(input, len, &s, &e)) {
        guard_add_violation(gr, GUARD_INPUT_PII, SEVERITY_MEDIUM,
                            "Phone number detected", s, e, 0.8f);
    }
    if (match_pii_credit_card(input, len, &s, &e)) {
        guard_add_violation(gr, GUARD_INPUT_PII, SEVERITY_CRITICAL,
                            "Credit card number detected", s, e, 0.99f);
    }
    return gr;
}

GuardResult* guard_check_pii_type(const char *input, size_t len, PIIType pii_type) {
    GuardResult *gr = guard_result_create();
    size_t s, e;
    switch (pii_type) {
    case PII_EMAIL:
        if (match_pii_email(input, len, &s, &e))
            guard_add_violation(gr, GUARD_INPUT_PII, SEVERITY_HIGH,
                                "Email detected", s, e, 0.95f);
        break;
    case PII_PHONE:
        if (match_pii_phone(input, len, &s, &e))
            guard_add_violation(gr, GUARD_INPUT_PII, SEVERITY_MEDIUM,
                                "Phone detected", s, e, 0.8f);
        break;
    case PII_CREDIT_CARD:
        if (match_pii_credit_card(input, len, &s, &e))
            guard_add_violation(gr, GUARD_INPUT_PII, SEVERITY_CRITICAL,
                                "Credit card detected", s, e, 0.99f);
        break;
    default:
        break;
    }
    return gr;
}

/* ──────────────────────────────────────────────
   Context: Hallucination Detection
   ────────────────────────────────────────────── */
float guard_entailment_score(const char *claim, const char *evidence) {
    float score = 0.0f;
    /* Simple overlap-based NLI: word overlap / claim length */
    if (!claim || !evidence) return 0.0f;
    size_t cl = strlen(claim), el = strlen(evidence);
    if (cl == 0 || el == 0) return 0.0f;
    size_t matches = 0;
    for (size_t i = 0; i < cl; i++) {
        char cc = tolower((unsigned char)claim[i]);
        for (size_t j = 0; j < el; j++) {
            if (cc == tolower((unsigned char)evidence[j])) {
                matches++;
                break;
            }
        }
    }
    score = (float)matches / (float)cl;
    return score < 0.0f ? 0.0f : (score > 1.0f ? 1.0f : score);
}

GuardResult* guard_check_hallucination(const char *generated,
                                        const char **retrieved_docs,
                                        size_t num_docs,
                                        float threshold) {
    GuardResult *gr = guard_result_create();
    if (num_docs == 0 || !generated) return gr;
    /* For each sentence in generated, check against all docs */
    size_t gl = strlen(generated);
    size_t sent_start = 0;
    for (size_t i = 0; i <= gl; i++) {
        if (i == gl || generated[i] == '.' || generated[i] == '!' ||
            generated[i] == '?' || generated[i] == '\n') {
            size_t slen = i - sent_start;
            if (slen > 5) {
                float best_sim = 0.0f;
                for (size_t d = 0; d < num_docs; d++) {
                    float sim = guard_entailment_score(generated + sent_start,
                                                       retrieved_docs[d]);
                    if (sim > best_sim) best_sim = sim;
                }
                if (best_sim < threshold) {
                    guard_add_violation(gr, GUARD_CONTEXT_HALLUCIN,
                                        SEVERITY_MEDIUM,
                                        "Potential hallucination detected",
                                        sent_start, i, 1.0f - best_sim);
                }
            }
            sent_start = i + 1;
        }
    }
    return gr;
}

GuardResult* guard_verify_fact(const char *statement,
                                const char *reference_text) {
    GuardResult *gr = guard_result_create();
    float score = guard_entailment_score(statement, reference_text);
    if (score < 0.5f) {
        guard_add_violation(gr, GUARD_CONTEXT_HALLUCIN,
                            SEVERITY_LOW, "Fact not verified in reference",
                            0, strlen(statement), 1.0f - score);
    }
    return gr;
}

/* ──────────────────────────────────────────────
   Output: Factual Consistency (NLI)
   ────────────────────────────────────────────── */
NLILabel guard_nli_classify(const char *premise, const char *hypothesis) {
    float score = guard_entailment_score(hypothesis, premise);
    if (score > 0.7f) return NLI_ENTAILMENT;
    if (score < 0.3f) return NLI_CONTRADICTION;
    return NLI_NEUTRAL;
}

GuardResult* guard_check_factual_consistency(const char *generated,
                                              const char **retrieved_docs,
                                              size_t num_docs) {
    GuardResult *gr = guard_result_create();
    if (num_docs == 0 || !generated) return gr;
    size_t gl = strlen(generated);
    size_t sent_start = 0;
    for (size_t i = 0; i <= gl; i++) {
        if (i == gl || generated[i] == '.' || generated[i] == '!' ||
            generated[i] == '?') {
            size_t slen = i - sent_start;
            if (slen > 10) {
                bool supported = false;
                for (size_t d = 0; d < num_docs; d++) {
                    NLILabel label = guard_nli_classify(retrieved_docs[d],
                                                        generated + sent_start);
                    if (label == NLI_ENTAILMENT) { supported = true; break; }
                }
                if (!supported) {
                    guard_add_violation(gr, GUARD_OUTPUT_NLI,
                                        SEVERITY_MEDIUM,
                                        "Statement not supported by context",
                                        sent_start, i, 0.8f);
                }
            }
            sent_start = i + 1;
        }
    }
    return gr;
}

GuardResult* guard_refuse_harmful(const char *output, size_t len) {
    GuardResult *gr = guard_result_create();
    for (const char **p = TOXIC_PATTERNS; *p; p++) {
        if (str_contains_nocase(output, *p)) {
            guard_add_violation(gr, GUARD_OUTPUT_REFUSAL,
                                SEVERITY_CRITICAL,
                                "Harmful content in output", 0, len, 0.95f);
            break;
        }
    }
    return gr;
}

/* ──────────────────────────────────────────────
   Citation Operations
   ────────────────────────────────────────────── */
CitationList* citation_extract(const char *generated, const char **sources,
                                const char **source_texts, size_t num_sources) {
    CitationList *cl = safe_alloc(sizeof(CitationList));
    cl->entries = safe_alloc(num_sources * sizeof(Citation));
    cl->count   = 0;
    /* Look for [N] or [source-N] patterns in generated text */
    for (size_t s = 0; s < num_sources; s++) {
        char pattern[64];
        snprintf(pattern, sizeof(pattern), "[%zu]", s + 1);
        if (strstr(generated, pattern)) {
            Citation *c = &cl->entries[cl->count];
            c->source_id   = s;
            c->source_text = sources[s] ? strdup(sources[s]) : NULL;
            c->excerpt     = source_texts[s] ? strdup(source_texts[s]) : NULL;
            c->relevance   = 1.0f;
            cl->count++;
        }
        snprintf(pattern, sizeof(pattern), "(source %zu)", s + 1);
        if (str_contains_nocase(generated, pattern)) {
            Citation *c = &cl->entries[cl->count];
            c->source_id   = s;
            c->source_text = sources[s] ? strdup(sources[s]) : NULL;
            c->excerpt     = source_texts[s] ? strdup(source_texts[s]) : NULL;
            c->relevance   = 1.0f;
            cl->count++;
        }
    }
    return cl;
}

void citation_list_destroy(CitationList *cl) {
    if (!cl) return;
    for (size_t i = 0; i < cl->count; i++) {
        free(cl->entries[i].source_text);
        free(cl->entries[i].excerpt);
    }
    free(cl->entries);
    free(cl);
}

GuardResult* guard_check_citations(const CitationList *expected,
                                    const CitationList *found) {
    GuardResult *gr = guard_result_create();
    if (!found || found->count == 0) {
        guard_add_violation(gr, GUARD_CITATION_MISSING,
                            SEVERITY_MEDIUM,
                            "No citations found in output",
                            0, 0, 1.0f);
        return gr;
    }
    if (expected && found->count < expected->count) {
        guard_add_violation(gr, GUARD_CITATION_MISSING,
                            SEVERITY_LOW,
                            "Fewer citations than expected",
                            0, 0, 0.5f);
    }
    return gr;
}

/* ──────────────────────────────────────────────
   Full Guard Chain
   ────────────────────────────────────────────── */
GuardResult* guard_chain_input(GuardContext *gc, const GuardConfig *cfg) {
    GuardResult *gr = guard_result_create();
    if (cfg->active_guards & GUARD_INPUT_INJECTION) {
        GuardResult *r = guard_check_injection(gc->user_query, gc->query_len);
        for (size_t i = 0; i < r->count; i++) {
            guard_add_violation(gr, r->violations[i].type,
                                r->violations[i].severity,
                                r->violations[i].message,
                                r->violations[i].position_start,
                                r->violations[i].position_end,
                                r->violations[i].confidence);
        }
        if (cfg->block_on_violation && !r->passed) {
            gc->should_block = true;
            gc->block_reason = strdup("Input guard: prompt injection");
        }
        guard_result_destroy(r);
    }
    if (cfg->active_guards & GUARD_INPUT_TOXICITY) {
        GuardResult *r = guard_check_toxicity(gc->user_query, gc->query_len);
        for (size_t i = 0; i < r->count; i++) {
            guard_add_violation(gr, r->violations[i].type,
                                r->violations[i].severity,
                                r->violations[i].message,
                                r->violations[i].position_start,
                                r->violations[i].position_end,
                                r->violations[i].confidence);
        }
        if (cfg->block_on_violation && !r->passed) {
            gc->should_block = true;
            gc->block_reason = strdup("Input guard: toxicity");
        }
        guard_result_destroy(r);
    }
    if (cfg->active_guards & GUARD_INPUT_PII) {
        GuardResult *r = guard_check_pii(gc->user_query, gc->query_len);
        for (size_t i = 0; i < r->count; i++) {
            guard_add_violation(gr, r->violations[i].type,
                                r->violations[i].severity,
                                r->violations[i].message,
                                r->violations[i].position_start,
                                r->violations[i].position_end,
                                r->violations[i].confidence);
        }
        if (cfg->block_on_violation && !r->passed) {
            gc->should_block = true;
            gc->block_reason = strdup("Input guard: PII detected");
        }
        guard_result_destroy(r);
    }
    return gr;
}

GuardResult* guard_chain_context(GuardContext *gc, const GuardConfig *cfg) {
    GuardResult *gr = guard_result_create();
    if (cfg->active_guards & GUARD_CONTEXT_HALLUCIN && gc->generated_text) {
        GuardResult *r = guard_check_hallucination(
            gc->generated_text,
            (const char**)gc->doc_texts,
            gc->num_docs,
            cfg->hallucination_threshold > 0 ? cfg->hallucination_threshold : 0.3f);
        for (size_t i = 0; i < r->count; i++) {
            guard_add_violation(gr, r->violations[i].type,
                                r->violations[i].severity,
                                r->violations[i].message,
                                r->violations[i].position_start,
                                r->violations[i].position_end,
                                r->violations[i].confidence);
        }
        guard_result_destroy(r);
    }
    return gr;
}

GuardResult* guard_chain_output(GuardContext *gc, const GuardConfig *cfg) {
    GuardResult *gr = guard_result_create();
    if (!gc->generated_text) return gr;
    if (cfg->active_guards & GUARD_OUTPUT_NLI) {
        GuardResult *r = guard_check_factual_consistency(
            gc->generated_text,
            (const char**)gc->doc_texts,
            gc->num_docs);
        for (size_t i = 0; i < r->count; i++) {
            guard_add_violation(gr, r->violations[i].type,
                                r->violations[i].severity,
                                r->violations[i].message,
                                r->violations[i].position_start,
                                r->violations[i].position_end,
                                r->violations[i].confidence);
        }
        guard_result_destroy(r);
    }
    if (cfg->active_guards & GUARD_OUTPUT_REFUSAL) {
        GuardResult *r = guard_refuse_harmful(gc->generated_text,
                                               gc->generated_len);
        for (size_t i = 0; i < r->count; i++) {
            guard_add_violation(gr, r->violations[i].type,
                                r->violations[i].severity,
                                r->violations[i].message,
                                r->violations[i].position_start,
                                r->violations[i].position_end,
                                r->violations[i].confidence);
        }
        if (cfg->block_on_violation && !r->passed) {
            gc->should_block = true;
            gc->block_reason = strdup("Output guard: harmful content");
        }
        guard_result_destroy(r);
    }
    return gr;
}

GuardResult* guard_chain_full(GuardContext *gc, const GuardConfig *cfg) {
    GuardResult *gr = guard_result_create();
    GuardResult *input_r   = guard_chain_input(gc, cfg);
    GuardResult *context_r = guard_chain_context(gc, cfg);
    GuardResult *output_r  = guard_chain_output(gc, cfg);
    for (size_t i = 0; i < input_r->count; i++)
        guard_add_violation(gr, input_r->violations[i].type,
                            input_r->violations[i].severity,
                            input_r->violations[i].message,
                            input_r->violations[i].position_start,
                            input_r->violations[i].position_end,
                            input_r->violations[i].confidence);
    for (size_t i = 0; i < context_r->count; i++)
        guard_add_violation(gr, context_r->violations[i].type,
                            context_r->violations[i].severity,
                            context_r->violations[i].message,
                            context_r->violations[i].position_start,
                            context_r->violations[i].position_end,
                            context_r->violations[i].confidence);
    for (size_t i = 0; i < output_r->count; i++)
        guard_add_violation(gr, output_r->violations[i].type,
                            output_r->violations[i].severity,
                            output_r->violations[i].message,
                            output_r->violations[i].position_start,
                            output_r->violations[i].position_end,
                            output_r->violations[i].confidence);
    guard_result_destroy(input_r);
    guard_result_destroy(context_r);
    guard_result_destroy(output_r);
    if (gc->should_block) gr->passed = false;
    return gr;
}
