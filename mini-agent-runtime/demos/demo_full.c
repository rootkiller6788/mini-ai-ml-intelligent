/*
 * mini-agent-runtime — Full Demo: AI Agent Runtime
 *
 * Demonstrates: ReAct agent, tool use, planning, agent memory, multi-agent.
 */
#include "../src/react_agent.h"
#include "../src/tool_use.h"
#include "../src/planning_system.h"
#include "../src/agent_memory.h"
#include "../src/multi_agent.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *mock_llm_callback(const char *prompt, void *user_data) {
    (void)user_data;
    static char response[512];
    snprintf(response, sizeof(response),
             "Thought: This is a test thought.\nAction: calculator\nAction Input: {\"expr\": \"2+2\"}\n");
    return response;
}

static char *mock_planning_llm(const char *prompt, void *user_data) {
    (void)user_data;
    static char response[256];
    snprintf(response, sizeof(response), "Step 1: Search for information. | Step 2: Compute result. | Answer: 4");
    return response;
}

int main(void) {
    printf("=== mini-agent-runtime: AI Agent Runtime Demo ===\n\n");

    /* Step 1: ReAct Agent */
    printf("-- Step 1: ReAct Agent --\n");
    react_agent_t *agent = react_agent_create("You are a helpful math assistant.");
    react_agent_set_max_iterations(agent, 5);
    react_agent_set_call_format(agent, REACT_CALL_FORMAT_TEXT);
    react_agent_set_llm_callback(agent, mock_llm_callback, NULL);
    react_agent_add_message(agent, "user", "What is 2+2?");
    react_state_t state = react_agent_get_state(agent);
    printf("ReAct agent: state=%d, max_iters=%d\n", state, react_agent_get_iteration(agent));
    react_agent_destroy(agent);

    /* Step 2: Tool Registry */
    printf("\n-- Step 2: Tool Registry --\n");
    tool_registry_t *reg = tool_registry_create();
    tool_def_t calc_def;
    memset(&calc_def, 0, sizeof(calc_def));
    strcpy(calc_def.name, "calculator");
    strcpy(calc_def.description, "Evaluate mathematical expressions");
    calc_def.type = TOOL_TYPE_CALCULATOR;
    tool_registry_add(reg, &calc_def, NULL, NULL);
    tool_def_t search_def;
    memset(&search_def, 0, sizeof(search_def));
    strcpy(search_def.name, "web_search");
    strcpy(search_def.description, "Search the web for information");
    search_def.type = TOOL_TYPE_WEB_SEARCH;
    tool_registry_add(reg, &search_def, NULL, NULL);
    printf("Tool registry: %d tools\n", tool_registry_count(reg));
    char schema_buf[2048];
    tool_get_schemas_json(reg, schema_buf, sizeof(schema_buf));
    printf("  schemas: %s\n", schema_buf);
    tool_registry_destroy(reg);

    /* Step 3: Planning System */
    printf("\n-- Step 3: Planning System (ReWOO) --\n");
    planning_system_t *ps = planning_system_create(PLANNING_STRATEGY_REWOO);
    planning_system_set_llm_callback(ps, mock_planning_llm, NULL);
    plan_t *plan = planning_system_create_plan(ps, "Calculate the sum of 1 to 10");
    printf("Plan: strategy=%s, steps=%d\n", planning_strategy_name(plan->strategy), plan->step_count);
    for (int i = 0; i < plan->step_count && i < 5; i++)
        printf("  step %d: %s\n", i, plan->steps[i].description);
    plan_destroy(plan);
    planning_system_destroy(ps);

    /* Step 4: Agent Memory */
    printf("\n-- Step 4: Agent Memory System --\n");
    agent_memory_t *mem = agent_memory_create();
    working_memory_add(&mem->working, MEMORY_ROLE_USER, "What is the capital of France?");
    working_memory_add(&mem->working, MEMORY_ROLE_ASSISTANT, "The capital of France is Paris.");
    working_memory_add(&mem->working, MEMORY_ROLE_USER, "What is its population?");
    int count;
    const memory_item_t *items = working_memory_get_recent(&mem->working, 3, &count);
    printf("Working memory: %d items\n", count);
    char wm_buf[1024];
    working_memory_format_for_llm(&mem->working, wm_buf, sizeof(wm_buf));
    printf("  LLM context: %s\n", wm_buf);
    agent_memory_destroy(mem);

    /* Step 5: Multi-Agent */
    printf("\n-- Step 5: Multi-Agent System --\n");
    multi_agent_t *ma = multi_agent_create();
    ma_agent_def_t *planner = ma_agent_create(MA_ROLE_PLANNER, "PlannerAgent", "Plan the approach.");
    ma_agent_set_persona(planner, "methodical and thorough");
    int pid = multi_agent_register(ma, planner);
    ma_agent_def_t *executor = ma_agent_create(MA_ROLE_EXECUTOR, "ExecAgent", "Execute the plan.");
    int eid = multi_agent_register(ma, executor);
    printf("Multi-agent: %d agents registered (planner=%d, executor=%d)\n", 2, pid, eid);
    ma_message_t msg;
    memset(&msg, 0, sizeof(msg));
    msg.sender_id = -1;
    msg.recipient_id = pid;
    msg.type = MA_MSG_TYPE_TASK;
    strcpy(msg.content, "Write a summary of AI safety.");
    ma_send_message(ma, &msg);
    printf("  message sent to planner\n");
    multi_agent_destroy(ma);

    printf("\nAI agent runtime demo complete!\n");
    return 0;
}
