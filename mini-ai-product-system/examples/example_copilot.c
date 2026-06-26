#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "copilot_context.h"

int main(void) {
    uint64_t seed = 77777;
    printf("=== Copilot Context System Demo ===\n\n");

    CCContextState state;
    cc_context_state_init(&state);

    const char *open_paths[] = {
        "src/main.c",
        "src/utils.c",
        "include/common.h",
        "tests/test_main.c"
    };
    cc_gather_open_files(&state, open_paths, 4);
    printf("Gathered %d open files\n", state.num_open_files);

    cc_set_cursor(&state, "src/main.c", 42, 15);
    printf("Cursor at %s:%d:%d\n", state.cursor.path, state.cursor.line, state.cursor.col);

    const char *before_edit = "int x = 10;\nint y = 20;";
    const char *after_edit  = "int x = 10;\nint y = 20;\nint z = x + y;";
    cc_gather_recent_edits(&state, "src/main.c",
                           before_edit, (int32_t)strlen(before_edit),
                           after_edit,  (int32_t)strlen(after_edit));
    printf("Recorded recent edit (%d total edits)\n", state.recent_edits.num_edits);

    const char *file_imports = "#include <stdio.h>\n#include <stdlib.h>\n#include \"common.h\"\n";
    cc_gather_imports(&state, "src/main.c", file_imports, (int32_t)strlen(file_imports));

    const char *proj_paths[] = {"src/main.c", "src/utils.c", "include/common.h",
                                "src/math.c", "tests/test_main.c"};
    const char *proj_names[] = {"main.c", "utils.c", "common.h", "math.c", "test_main.c"};
    const char *proj_langs[] = {"c", "c", "c", "c", "c"};
    cc_gather_project_structure(&state, proj_paths, proj_names, proj_langs, 5);
    printf("Project structure: %d files\n", state.project.num_files);

    const char *diff = "diff --git a/src/main.c b/src/main.c\n"
                       "+int z = x + y;\n"
                       "-void old_func();\n";
    cc_gather_git_diff(&state, diff, (int32_t)strlen(diff));
    printf("Git diff: %s\n", state.git_diff.has_changes ? "has changes" : "no changes");

    char lang[32];
    cc_detect_language("src/main.c", lang, 32);
    printf("Detected language: %s\n", lang);
    cc_detect_language("app.py", lang, 32);
    printf("Detected language: %s\n", lang);
    cc_detect_language("utils.ts", lang, 32);
    printf("Detected language: %s\n", lang);

    printf("\n--- Context Ranking ---\n");
    CCRankedContext ranked;
    cc_rank_context(&state, &ranked);
    printf("Ranked %d context snippets:\n", ranked.num_snippets);
    for (int i = 0; i < 3 && i < ranked.num_snippets; i++)
        printf("  [%d] relevance=%.4f start=%d end=%d\n",
               i, ranked.snippets[i].relevance,
               ranked.snippets[i].start_line, ranked.snippets[i].end_line);

    printf("\n--- Prompt Construction ---\n");
    CCUserRequest request;
    request.intent = CC_INTENT_COMPLETE;
    const char *req_text = "Complete the function to calculate factorial recursively.";
    request.text_len = (int32_t)strlen(req_text);
    memcpy(request.text, req_text, (size_t)request.text_len);
    request.text[request.text_len] = '\0';
    cc_detect_language("src/main.c", request.language, sizeof(request.language));

    CCConstructedPrompt prompt;
    cc_construct_prompt(&state, &request, &ranked, &prompt);
    printf("Prompt messages: %d\n", prompt.num_messages);
    printf("Raw prompt length: %d chars\n", prompt.raw_len);
    printf("\n=== Prompt Preview (first 300 chars) ===\n%.300s\n...\n", prompt.raw_prompt);

    printf("\n--- Completion Generation ---\n");
    CCCompletion comp;
    cc_completion_generate(&prompt, "int factorial(int n) {\n", 0.7f, seed, &comp);
    printf("Completion (%d chars):\n%s\n", comp.text_len, comp.text);

    cc_postprocess_trim(&comp);
    printf("After trim (%d chars):\n%s\n", comp.text_len, comp.text);

    int is_complete;
    cc_postprocess_filter_incomplete(comp.text, comp.text_len, &is_complete);
    printf("Is complete: %s\n", is_complete ? "yes" : "no");

    printf("\n--- Multi-line Completion ---\n");
    CCMultiLineCompletion ml;
    cc_completion_multi_line(&prompt, "void sort(int arr[], int n) {\n", 3, 0.8f, seed, &ml);
    printf("%d lines generated\n", ml.num_lines);
    for (int i = 0; i < ml.num_lines; i++)
        printf("Line %d (%d chars): %.60s...\n",
               i + 1, ml.lines[i].text_len, ml.lines[i].text);

    printf("\n--- Inline Chat ---\n");
    CCUserRequest chat_req;
    chat_req.intent = CC_INTENT_CHAT;
    const char *chat_text = "What does this function do?";
    chat_req.text_len = (int32_t)strlen(chat_text);
    memcpy(chat_req.text, chat_text, (size_t)chat_req.text_len);
    chat_req.text[chat_req.text_len] = '\0';
    cc_detect_language("src/main.c", chat_req.language, sizeof(chat_req.language));
    CCConstructedPrompt chat_prompt;
    cc_inline_chat_process(&chat_req, &state, &chat_prompt);
    printf("Chat prompt constructed: %d messages, %d raw chars\n",
           chat_prompt.num_messages, chat_prompt.raw_len);

    printf("\nDone.\n");
    return 0;
}
