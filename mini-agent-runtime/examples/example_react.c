#include "react_agent.h"
#include "tool_use.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char* mock_llm_callback(const char *prompt, void *user_data) {
    (void)user_data;
    (void)prompt;
    char *response = (char*)malloc(1024);
    if (!response) return NULL;
    static int call_count = 0;
    call_count++;
    if (call_count == 1) {
        strcpy(response,
            "Thought: I need to understand what the user is asking. The user wants to know "
            "the result of a calculation. I should use the calculator tool.\n"
            "Action: calculator\n"
            "Action Input: {\"expression\": \"2 + 2\"}\n");
    } else {
        strcpy(response,
            "Thought: The calculator tool has provided the result. I now have the answer.\n"
            "Action: Final Answer\n"
            "Action Input: The result of 2 + 2 is 4.\n");
    }
    return response;
}

int main(void) {
    printf("=== ReAct Agent Example ===\n\n");

    tool_registry_t *registry = tool_registry_create();
    if (!registry) { printf("Failed to create tool registry\n"); return 1; }

    if (!tool_registry_add_builtin(registry, TOOL_TYPE_CALCULATOR)) {
        printf("Failed to add calculator tool\n");
        tool_registry_destroy(registry);
        return 1;
    }
    if (!tool_registry_add_builtin(registry, TOOL_TYPE_WEB_SEARCH)) {
        printf("Failed to add web_search tool\n");
        tool_registry_destroy(registry);
        return 1;
    }

    printf("Tool registry initialized with %d tools\n", tool_registry_count(registry));

    char tools_desc[4096];
    tool_format_for_llm(registry, tools_desc, sizeof(tools_desc));
    printf("Available tools:\n%s\n", tools_desc);

    react_agent_t *agent = react_agent_create(
        "You are a helpful assistant. Use tools when needed to answer questions accurately.");
    if (!agent) { printf("Failed to create agent\n"); tool_registry_destroy(registry); return 1; }

    react_agent_set_tools(agent, registry);
    react_agent_set_max_iterations(agent, 5);
    react_agent_set_call_format(agent, REACT_CALL_FORMAT_TEXT);
    react_agent_set_llm_callback(agent, mock_llm_callback, NULL);

    printf("\n--- Running agent with query: \"What is 2+2?\" ---\n\n");

    react_state_t initial_state = react_agent_get_state(agent);
    printf("Initial state: %d (0=IDLE)\n", (int)initial_state);

    const char *answer = react_agent_run(agent, "What is 2+2?");
    if (answer) {
        printf("\nAgent final answer: %s\n", answer);
    } else {
        printf("\nAgent failed to produce an answer.\n");
        printf("Final state: %d\n", (int)react_agent_get_state(agent));
    }

    printf("Total iterations used: %d\n", react_agent_get_iteration(agent));

    const react_step_t *last = react_agent_get_last_step(agent);
    if (last) {
        printf("\nLast step details:\n");
        printf("  Thought: %s\n", last->thought);
        printf("  Action: %s\n", last->action);
        printf("  Is final: %s\n", last->is_final ? "yes" : "no");
        if (last->is_final) {
            printf("  Final answer: %s\n", last->final_answer);
        }
    }

    int history_count = 0;
    const react_message_t *history = react_agent_get_history(agent, &history_count);
    printf("\nConversation history (%d messages):\n", history_count);
    for (int i = 0; i < history_count; i++) {
        printf("  [%s] %.80s...\n", history[i].role, history[i].content);
    }

    char prompt_buf[16384];
    react_agent_build_prompt(agent, "What is 2+2?", prompt_buf, sizeof(prompt_buf));
    printf("\n--- Generated Prompt ---\n%.500s...\n", prompt_buf);

    react_agent_destroy(agent);
    tool_registry_destroy(registry);

    printf("\nExample completed successfully.\n");
    return 0;
}
