#ifndef GUARDRAILS_GEN_H
#define GUARDRAILS_GEN_H

#include <stdbool.h>
#include <stddef.h>

/* ──────────────────────────────────────────────
   Guardrail Configuration
   ────────────────────────────────────────────── */
typedef enum {
    GUARD_INPUT_INJECTION   = 1 << 0,
    GUARD_INPUT_TOXICITY    = 1 << 1,
    GUARD_INPUT_PII         = 1 << 2,
    GUARD_CONTEXT_HALLUCIN  = 1 << 3,
    GUARD_OUTPUT_NLI        = 1 << 4,
    GUARD_OUTPUT_REFUSAL    = 1 << 5,
    GUARD_CITATION_MISSING  = 1 << 6,
    GUARD_ALL               = 0x7F
} GuardType;

typedef enum {
    SEVERITY_SAFE,
    SEVERITY_LOW,
    SEVERITY_MEDIUM,
    SEVERITY_HIGH,
    SEVERITY_CRITICAL
} Severity;

/* ──────────────────────────────────────────────
   Guard Violation Result
   ────────────────────────────────────────────── */
typedef struct {
    GuardType  type;
    Severity   severity;
    char      *message;
    size_t     position_start;
    size_t     position_end;
    float      confidence;
} GuardViolation;

typedef struct {
    GuardViolation *violations;
    size_t          count;
    size_t          capacity;
    bool            passed;
} GuardResult;

/* ──────────────────────────────────────────────
   PII Pattern Types
   ────────────────────────────────────────────── */
typedef enum {
    PII_EMAIL,
    PII_PHONE,
    PII_SSN,
    PII_CREDIT_CARD,
    PII_API_KEY,
    PII_IP_ADDRESS
} PIIType;

/* ──────────────────────────────────────────────
   NLI Label (Natural Language Inference)
   ────────────────────────────────────────────── */
typedef enum {
    NLI_ENTAILMENT,
    NLI_CONTRADICTION,
    NLI_NEUTRAL
} NLILabel;

/* ──────────────────────────────────────────────
   Citation Entry
   ────────────────────────────────────────────── */
typedef struct {
    size_t  source_id;
    char   *source_text;
    char   *excerpt;
    float   relevance;
} Citation;

typedef struct {
    Citation *entries;
    size_t    count;
} CitationList;

/* ──────────────────────────────────────────────
   Guard Context (carries state through pipeline)
   ────────────────────────────────────────────── */
typedef struct {
    const char  *user_query;
    size_t       query_len;
    const char  *retrieved_docs;
    size_t       num_docs;
    char       **doc_texts;
    const char  *generated_text;
    size_t       generated_len;
    GuardResult *results;
    CitationList citations;
    bool         should_block;
    char        *block_reason;
} GuardContext;

/* ──────────────────────────────────────────────
   Lifecycle
   ────────────────────────────────────────────── */
GuardResult*   guard_result_create(void);
void           guard_result_destroy(GuardResult *gr);
void           guard_add_violation(GuardResult *gr, GuardType type,
                                   Severity sev, const char *msg,
                                   size_t start, size_t end, float conf);
GuardContext*  guard_context_create(void);
void           guard_context_destroy(GuardContext *gc);

/* ──────────────────────────────────────────────
   Input Guards
   ────────────────────────────────────────────── */
GuardResult*  guard_check_injection(const char *input, size_t len);
GuardResult*  guard_check_toxicity(const char *input, size_t len);
GuardResult*  guard_check_pii(const char *input, size_t len);
GuardResult*  guard_check_pii_type(const char *input, size_t len, PIIType pii_type);

/* ──────────────────────────────────────────────
   Context Guards (Hallucination Detection)
   ────────────────────────────────────────────── */
float  guard_entailment_score(const char *claim, const char *evidence);
GuardResult*  guard_check_hallucination(const char *generated,
                                        const char **retrieved_docs,
                                        size_t num_docs,
                                        float threshold);
GuardResult*  guard_verify_fact(const char *statement,
                                const char *reference_text);

/* ──────────────────────────────────────────────
   Output Guards (Factual Consistency)
   ────────────────────────────────────────────── */
NLILabel  guard_nli_classify(const char *premise, const char *hypothesis);
GuardResult*  guard_check_factual_consistency(const char *generated,
                                              const char **retrieved_docs,
                                              size_t num_docs);
GuardResult*  guard_refuse_harmful(const char *output, size_t len);

/* ──────────────────────────────────────────────
   Citation Operations
   ────────────────────────────────────────────── */
CitationList*  citation_extract(const char *generated, const char **sources,
                                const char **source_texts, size_t num_sources);
void           citation_list_destroy(CitationList *cl);
GuardResult*   guard_check_citations(const CitationList *expected,
                                     const CitationList *found);

/* ──────────────────────────────────────────────
   Full Guard Chain
   ────────────────────────────────────────────── */
typedef struct {
    GuardType active_guards;
    float     hallucination_threshold;
    float     toxicity_threshold;
    float     nli_threshold;
    bool      enforce_citations;
    bool      block_on_violation;
} GuardConfig;

GuardResult*  guard_chain_input(GuardContext *gc, const GuardConfig *cfg);
GuardResult*  guard_chain_context(GuardContext *gc, const GuardConfig *cfg);
GuardResult*  guard_chain_output(GuardContext *gc, const GuardConfig *cfg);
GuardResult*  guard_chain_full(GuardContext *gc, const GuardConfig *cfg);

#endif /* GUARDRAILS_GEN_H */
