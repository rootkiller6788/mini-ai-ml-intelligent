#include "copilot_context.h"
#include <stdlib.h>
#include <string.h>

void cc_context_state_init(CCContextState *state) {
    memset(state, 0, sizeof(CCContextState));
    strcpy(state->current_language, "unknown");
}

void cc_gather_open_files(CCContextState *state, const char **paths, int32_t n) {
    state->num_open_files = n < CC_MAX_FILES ? n : CC_MAX_FILES;
    for (int i = 0; i < state->num_open_files; i++) {
        CCFileContext *fc = &state->open_files[i];
        strncpy(fc->path, paths[i], CC_MAX_FILE_PATH - 1);
        fc->path[CC_MAX_FILE_PATH - 1] = '\0';
        fc->is_open = 1;
        fc->is_active = 0;
        fc->content = NULL;
        fc->content_len = 0;
        cc_detect_language(paths[i], fc->language, sizeof(fc->language));
    }
}

void cc_set_cursor(CCContextState *state, const char *path, int32_t line, int32_t col) {
    strncpy(state->cursor.path, path, CC_MAX_FILE_PATH - 1);
    state->cursor.path[CC_MAX_FILE_PATH - 1] = '\0';
    state->cursor.line = line;
    state->cursor.col = col;
    for (int i = 0; i < state->num_open_files; i++) {
        state->open_files[i].is_active = (strcmp(state->open_files[i].path, path) == 0);
    }
    cc_detect_language(path, state->current_language, sizeof(state->current_language));
}

void cc_gather_recent_edits(CCContextState *state, const char *path,
                            const char *before, int32_t blen,
                            const char *after, int32_t alen) {
    if (state->recent_edits.num_edits >= 32) {
        memmove(&state->recent_edits.edits[0], &state->recent_edits.edits[1],
                (size_t)(31) * sizeof(state->recent_edits.edits[0]));
        state->recent_edits.num_edits = 31;
    }
    int idx = state->recent_edits.num_edits++;
    strncpy(state->recent_edits.edits[idx].path, path, CC_MAX_FILE_PATH - 1);
    int bl = blen < CC_MAX_SNIPPET_LEN - 1 ? blen : CC_MAX_SNIPPET_LEN - 1;
    int al = alen < CC_MAX_SNIPPET_LEN - 1 ? alen : CC_MAX_SNIPPET_LEN - 1;
    memcpy(state->recent_edits.edits[idx].before, before, (size_t)bl);
    state->recent_edits.edits[idx].before[bl] = '\0';
    state->recent_edits.edits[idx].before_len = bl;
    memcpy(state->recent_edits.edits[idx].after, after, (size_t)al);
    state->recent_edits.edits[idx].after[al] = '\0';
    state->recent_edits.edits[idx].after_len = al;
    state->recent_edits.timestamp = time(NULL);
}

void cc_gather_imports(CCContextState *state, const char *path,
                       const char *content, int32_t len) {
    char *copy = (char *)malloc((size_t)(len + 1));
    if (!copy) return;
    memcpy(copy, content, (size_t)len);
    copy[len] = '\0';
    for (int i = 0; i < state->num_open_files; i++) {
        if (strcmp(state->open_files[i].path, path) == 0) {
            state->open_files[i].content = copy;
            state->open_files[i].content_len = len;
            break;
        }
    }
}

void cc_gather_project_structure(CCContextState *state,
                                 const char **paths, const char **names,
                                 const char **languages, int32_t n) {
    state->project.num_files = n < CC_MAX_FILES ? n : CC_MAX_FILES;
    for (int i = 0; i < state->project.num_files; i++) {
        CCProjectFile *pf = &state->project.files[i];
        strncpy(pf->path, paths[i], CC_MAX_FILE_PATH - 1);
        pf->path[CC_MAX_FILE_PATH - 1] = '\0';
        strncpy(pf->name, names[i], 127);
        pf->name[127] = '\0';
        strncpy(pf->language, languages[i], 31);
        pf->language[31] = '\0';
        pf->num_imports = 0;
    }
}

void cc_gather_git_diff(CCContextState *state, const char *diff_text, int32_t len) {
    state->git_diff.has_changes = (len > 0) ? 1 : 0;
    int dl = len < CC_MAX_SNIPPET_LEN * 4 - 1 ? len : CC_MAX_SNIPPET_LEN * 4 - 1;
    memcpy(state->git_diff.diff_text, diff_text, (size_t)dl);
    state->git_diff.diff_text[dl] = '\0';
    state->git_diff.diff_len = dl;
}

void cc_detect_language(const char *path, char *language, int32_t max_len) {
    const char *ext = strrchr(path, '.');
    if (!ext) { strncpy(language, "text", (size_t)max_len - 1); return; }
    if (strcmp(ext, ".c") == 0 || strcmp(ext, ".h") == 0)
        strncpy(language, "c", (size_t)max_len - 1);
    else if (strcmp(ext, ".py") == 0)
        strncpy(language, "python", (size_t)max_len - 1);
    else if (strcmp(ext, ".js") == 0 || strcmp(ext, ".ts") == 0)
        strncpy(language, "javascript", (size_t)max_len - 1);
    else if (strcmp(ext, ".java") == 0)
        strncpy(language, "java", (size_t)max_len - 1);
    else if (strcmp(ext, ".go") == 0)
        strncpy(language, "go", (size_t)max_len - 1);
    else if (strcmp(ext, ".rs") == 0)
        strncpy(language, "rust", (size_t)max_len - 1);
    else
        strncpy(language, "text", (size_t)max_len - 1);
    language[max_len - 1] = '\0';
}

float cc_snippet_relevance(const CCContextSnippet *snip, const CCCursorPos *cursor,
                           const char *active_path) {
    (void)active_path;
    float dist = (float)(abs(snip->start_line - cursor->line) +
                         abs(snip->end_line - cursor->line));
    float proximity = 1.0f / (1.0f + dist * 0.1f);
    return snip->relevance * 0.5f + proximity * 0.5f;
}

void cc_rank_context(const CCContextState *state, CCRankedContext *ranked) {
    ranked->num_snippets = 0;
    for (int i = 0; i < state->num_open_files; i++) {
        const CCFileContext *fc = &state->open_files[i];
        if (ranked->num_snippets >= 64) break;
        CCContextSnippet *snip = &ranked->snippets[ranked->num_snippets++];
        snip->start_line = 0;
        snip->end_line = fc->content_len > 0 ? fc->content_len / 80 : 50;
        snip->relevance = fc->is_active ? 1.0f : 0.3f;
        snip->text_len = fc->content_len < CC_MAX_SNIPPET_LEN - 1 ?
                         fc->content_len : CC_MAX_SNIPPET_LEN - 1;
        if (fc->content && snip->text_len > 0) {
            memcpy(snip->text, fc->content, (size_t)snip->text_len);
            snip->text[snip->text_len] = '\0';
        } else {
            snip->text[0] = '\0';
            snip->text_len = 0;
        }
    }
    for (int i = 0; i < ranked->num_snippets; i++) {
        ranked->snippets[i].relevance = cc_snippet_relevance(
            &ranked->snippets[i], &state->cursor, state->cursor.path);
    }
    for (int i = 0; i < ranked->num_snippets - 1; i++) {
        for (int j = i + 1; j < ranked->num_snippets; j++) {
            if (ranked->snippets[j].relevance > ranked->snippets[i].relevance) {
                CCContextSnippet tmp = ranked->snippets[i];
                ranked->snippets[i] = ranked->snippets[j];
                ranked->snippets[j] = tmp;
            }
        }
    }
}

void cc_construct_system_message(CCPromptMessage *msg) {
    strcpy(msg->role, "system");
    const char *sys = "You are an AI coding assistant. You help with code completion, "
                      "explanation, fixing, refactoring, and generation. "
                      "Provide concise, accurate code suggestions based on the context.";
    int len = (int32_t)strlen(sys);
    msg->content_len = len < CC_MAX_PROMPT_LEN - 1 ? len : CC_MAX_PROMPT_LEN - 1;
    memcpy(msg->content, sys, (size_t)msg->content_len);
    msg->content[msg->content_len] = '\0';
}

void cc_construct_prompt(const CCContextState *state, const CCUserRequest *request,
                         const CCRankedContext *ranked, CCConstructedPrompt *prompt) {
    prompt->num_messages = 0;
    CCPromptMessage sys_msg;
    cc_construct_system_message(&sys_msg);
    prompt->messages[prompt->num_messages++] = sys_msg;

    CCPromptMessage ctx_msg;
    strcpy(ctx_msg.role, "context");
    int off = 0;
    off += snprintf(ctx_msg.content + off, (size_t)(CC_MAX_PROMPT_LEN - off),
                    "Language: %s\n", state->current_language);
    off += snprintf(ctx_msg.content + off, (size_t)(CC_MAX_PROMPT_LEN - off),
                    "Cursor: %s:%d:%d\n", state->cursor.path,
                    state->cursor.line, state->cursor.col);
    if (state->git_diff.has_changes) {
        off += snprintf(ctx_msg.content + off, (size_t)(CC_MAX_PROMPT_LEN - off),
                        "Git diff:\n%.1000s\n", state->git_diff.diff_text);
    }
    for (int i = 0; i < ranked->num_snippets && i < 3; i++) {
        off += snprintf(ctx_msg.content + off, (size_t)(CC_MAX_PROMPT_LEN - off),
                        "File[%d]: %s\n", i, ranked->snippets[i].text);
    }
    ctx_msg.content_len = off;
    prompt->messages[prompt->num_messages++] = ctx_msg;

    CCPromptMessage user_msg;
    strcpy(user_msg.role, "user");
    user_msg.content_len = request->text_len;
    memcpy(user_msg.content, request->text, (size_t)request->text_len);
    user_msg.content[request->text_len] = '\0';
    prompt->messages[prompt->num_messages++] = user_msg;

    off = 0;
    off += snprintf(prompt->raw_prompt + off, (size_t)(CC_MAX_PROMPT_LEN - off),
                    "System: %s\n", sys_msg.content);
    off += snprintf(prompt->raw_prompt + off, (size_t)(CC_MAX_PROMPT_LEN - off),
                    "Context: %s\n", ctx_msg.content);
    off += snprintf(prompt->raw_prompt + off, (size_t)(CC_MAX_PROMPT_LEN - off),
                    "User: %s", user_msg.content);
    prompt->raw_len = off;
}

void cc_completion_generate(const CCConstructedPrompt *prompt,
                            const char *prefix, float temperature, uint64_t seed,
                            CCCompletion *completion) {
    (void)temperature;
    (void)seed;
    const char *template = "// Generated code suggestion\n";
    int tlen = (int32_t)strlen(template);
    int plen = prefix ? (int32_t)strlen(prefix) : 0;
    completion->text_len = tlen + plen;
    if (completion->text_len >= CC_MAX_COMPLETION) completion->text_len = CC_MAX_COMPLETION - 1;
    if (prefix) memcpy(completion->text, prefix, (size_t)plen);
    memcpy(completion->text + plen, template,
           (size_t)(completion->text_len - plen));
    completion->text[completion->text_len] = '\0';
    completion->logprob = -0.5f;
    completion->is_truncated = 0;
    (void)prompt;
}

void cc_completion_multi_line(const CCConstructedPrompt *prompt,
                              const char *prefix, int32_t max_lines,
                              float temperature, uint64_t seed,
                              CCMultiLineCompletion *completion) {
    if (max_lines > 8) max_lines = 8;
    completion->num_lines = max_lines;
    for (int i = 0; i < max_lines; i++) {
        CCCompletion line_comp;
        cc_completion_generate(prompt, prefix, temperature, seed + (uint64_t)i, &line_comp);
        completion->lines[i] = line_comp;
    }
}

void cc_postprocess_trim(CCCompletion *completion) {
    int end = completion->text_len - 1;
    while (end >= 0 && (completion->text[end] == ' ' || completion->text[end] == '\n' ||
                        completion->text[end] == '\t' || completion->text[end] == '\r'))
        end--;
    completion->text_len = end + 1;
    completion->text[completion->text_len] = '\0';
}

void cc_postprocess_filter_incomplete(const char *text, int32_t len,
                                      int *is_complete) {
    *is_complete = 1;
    int depth = 0;
    for (int i = 0; i < len; i++) {
        if (text[i] == '{' || text[i] == '(' || text[i] == '[') depth++;
        else if (text[i] == '}' || text[i] == ')' || text[i] == ']') depth--;
    }
    if (depth > 0) *is_complete = 0;
    if (len > 0 && text[len - 1] != ';' && text[len - 1] != '}' &&
        text[len - 1] != '\n') *is_complete = 0;
}

void cc_inline_chat_process(const CCUserRequest *request,
                            const CCContextState *state,
                            CCConstructedPrompt *prompt) {
    CCRankedContext ranked;
    cc_rank_context(state, &ranked);
    CCUserRequest chat_req = *request;
    if (chat_req.intent == CC_INTENT_CHAT) {
        cc_construct_prompt(state, &chat_req, &ranked, prompt);
    }
}
