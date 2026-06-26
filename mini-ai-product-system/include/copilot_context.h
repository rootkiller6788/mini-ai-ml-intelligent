#ifndef COPILOT_CONTEXT_H
#define COPILOT_CONTEXT_H

#include <stddef.h>
#include <stdint.h>
#include <time.h>

#define CC_MAX_FILES          128
#define CC_MAX_FILE_PATH      512
#define CC_MAX_FILE_CONTENT   (64 * 1024)
#define CC_MAX_LINES          4096
#define CC_MAX_PROMPT_LEN     (32 * 1024)
#define CC_MAX_COMPLETION     4096
#define CC_MAX_IMPORTS        256
#define CC_MAX_SNIPPET_LEN    2048

typedef struct {
    char   path[CC_MAX_FILE_PATH];
    char  *content;
    int32_t content_len;
    char   language[32];
    int32_t cursor_line;
    int32_t cursor_col;
    int64_t last_modified;
    int     is_open;
    int     is_active;
} CCFileContext;

typedef struct {
    int32_t start_line;
    int32_t end_line;
    char    text[CC_MAX_SNIPPET_LEN];
    int32_t text_len;
    float   relevance;
} CCContextSnippet;

typedef struct {
    char    path[CC_MAX_FILE_PATH];
    int32_t line;
    int32_t col;
} CCCursorPos;

typedef struct {
    struct {
        char path[CC_MAX_FILE_PATH];
        char before[CC_MAX_SNIPPET_LEN];
        char after[CC_MAX_SNIPPET_LEN];
        int32_t before_len;
        int32_t after_len;
    } edits[32];
    int32_t num_edits;
    time_t  timestamp;
} CCRecentEdits;

typedef struct {
    char   path[CC_MAX_FILE_PATH];
    char   name[128];
    char   symbol[128];
} CCImportInfo;

typedef struct {
    char         path[CC_MAX_FILE_PATH];
    char         name[128];
    CCImportInfo imports[CC_MAX_IMPORTS];
    int32_t      num_imports;
    char         language[32];
} CCProjectFile;

typedef struct {
    CCProjectFile files[CC_MAX_FILES];
    int32_t       num_files;
} CCProjectStructure;

typedef struct {
    char    path[CC_MAX_FILE_PATH];
    char    diff_text[CC_MAX_SNIPPET_LEN * 4];
    int32_t diff_len;
    int     has_changes;
} CCGitDiff;

typedef struct {
    CCFileContext     open_files[CC_MAX_FILES];
    int32_t           num_open_files;
    CCCursorPos       cursor;
    CCRecentEdits     recent_edits;
    CCProjectStructure project;
    CCGitDiff         git_diff;
    char              current_language[32];
} CCContextState;

typedef struct {
    CCContextSnippet snippets[64];
    int32_t          num_snippets;
} CCRankedContext;

typedef enum {
    CC_INTENT_COMPLETE,
    CC_INTENT_EXPLAIN,
    CC_INTENT_FIX,
    CC_INTENT_REFACTOR,
    CC_INTENT_GENERATE,
    CC_INTENT_CHAT
} CCUserIntent;

typedef struct {
    CCUserIntent intent;
    char         text[CC_MAX_SNIPPET_LEN];
    int32_t      text_len;
    char         language[32];
} CCUserRequest;

typedef struct {
    char    role[32];
    char    content[CC_MAX_PROMPT_LEN];
    int32_t content_len;
} CCPromptMessage;

typedef struct {
    CCPromptMessage messages[16];
    int32_t          num_messages;
    char             raw_prompt[CC_MAX_PROMPT_LEN];
    int32_t          raw_len;
} CCConstructedPrompt;

typedef struct {
    char    text[CC_MAX_COMPLETION];
    int32_t text_len;
    float   logprob;
    int     is_truncated;
} CCCompletion;

typedef struct {
    CCCompletion lines[8];
    int32_t      num_lines;
} CCMultiLineCompletion;

void cc_context_state_init(CCContextState *state);

void cc_gather_open_files(CCContextState *state, const char **paths, int32_t n);
void cc_set_cursor(CCContextState *state, const char *path, int32_t line, int32_t col);
void cc_gather_recent_edits(CCContextState *state, const char *path,
                            const char *before, int32_t blen,
                            const char *after, int32_t alen);
void cc_gather_imports(CCContextState *state, const char *path,
                       const char *content, int32_t len);
void cc_gather_project_structure(CCContextState *state,
                                 const char **paths, const char **names,
                                 const char **languages, int32_t n);
void cc_gather_git_diff(CCContextState *state, const char *diff_text, int32_t len);
void cc_detect_language(const char *path, char *language, int32_t max_len);

float cc_snippet_relevance(const CCContextSnippet *snip, const CCCursorPos *cursor,
                           const char *active_path);
void cc_rank_context(const CCContextState *state, CCRankedContext *ranked);

void cc_construct_system_message(CCPromptMessage *msg);
void cc_construct_prompt(const CCContextState *state, const CCUserRequest *request,
                         const CCRankedContext *ranked, CCConstructedPrompt *prompt);

void cc_completion_generate(const CCConstructedPrompt *prompt,
                            const char *prefix, float temperature, uint64_t seed,
                            CCCompletion *completion);
void cc_completion_multi_line(const CCConstructedPrompt *prompt,
                              const char *prefix, int32_t max_lines,
                              float temperature, uint64_t seed,
                              CCMultiLineCompletion *completion);
void cc_postprocess_trim(CCCompletion *completion);
void cc_postprocess_filter_incomplete(const char *text, int32_t len,
                                      int *is_complete);
void cc_inline_chat_process(const CCUserRequest *request,
                            const CCContextState *state,
                            CCConstructedPrompt *prompt);

#endif
