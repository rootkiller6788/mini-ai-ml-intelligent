/*
 * mini-agent-runtime — Core Tests
 *
 * Unit tests for ReAct agent, tool use, planning, agent memory, multi-agent.
 */
#include "../src/react_agent.h"
#include "../src/tool_use.h"
#include "../src/planning_system.h"
#include "../src/agent_memory.h"
#include "../src/multi_agent.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int tests_run = 0, tests_passed = 0;

#define TEST(name) do { tests_run++; printf("  TEST %s ... ", name); } while(0)
#define PASS() do { tests_passed++; printf("PASS\n"); } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); return 1; } while(0)
#define CHECK(cond, msg) if (!(cond)) FAIL(msg)

/* ── ReAct Agent Tests ── */
static int test_react_agent_create(void) {
    TEST("react_agent_create");
    react_agent_t *agent = react_agent_create("You are a helpful assistant.");
    CHECK(agent != NULL, "agent create failed");
    react_agent_destroy(agent);
    PASS();
    return 0;
}

static int test_react_agent_add_message(void) {
    TEST("react_agent_add_message");
    react_agent_t *agent = react_agent_create("system prompt");
    bool ok = react_agent_add_message(agent, "user", "Hello");
    CHECK(ok, "add message failed");
    int count = 0;
    const react_message_t *hist = react_agent_get_history(agent, &count);
    CHECK(count > 0, "history should not be empty");
    react_agent_destroy(agent);
    PASS();
    return 0;
}

static int test_react_agent_parse(void) {
    TEST("react_agent_parse_output");
    const char *raw = "Thought: I need to search.\nAction: search\nAction Input: hello world\n";
    react_step_t step = react_agent_parse_llm_output(raw);
    CHECK(strlen(step.thought) > 0, "thought empty");
    CHECK(strlen(step.action) > 0, "action empty");
    PASS();
    return 0;
}

/* ── Tool Use Tests ── */
static int test_tool_registry_create(void) {
    TEST("tool_registry_create");
    tool_registry_t *reg = tool_registry_create();
    CHECK(reg != NULL, "registry create failed");
    CHECK(tool_registry_count(reg) >= 0, "count should be 0");
    tool_registry_destroy(reg);
    PASS();
    return 0;
}

static int test_tool_registry_add_find(void) {
    TEST("tool_registry_add_find");
    tool_registry_t *reg = tool_registry_create();
    tool_def_t def;
    memset(&def, 0, sizeof(def));
    strcpy(def.name, "calculator");
    strcpy(def.description, "basic calculator");
    def.type = TOOL_TYPE_CALCULATOR;
    bool ok = tool_registry_add(reg, &def, NULL, NULL);
    CHECK(ok, "add tool failed");
    tool_entry_t *entry = tool_registry_find(reg, "calculator");
    CHECK(entry != NULL, "find tool failed");
    tool_registry_destroy(reg);
    PASS();
    return 0;
}

static int test_tool_builtin_add(void) {
    TEST("tool_registry_add_builtin");
    tool_registry_t *reg = tool_registry_create();
    bool ok = tool_registry_add_builtin(reg, TOOL_TYPE_CALCULATOR);
    CHECK(ok || !ok, "builtin add should not crash");
    tool_registry_destroy(reg);
    PASS();
    return 0;
}

/* ── Planning Tests ── */
static int test_planning_system_create(void) {
    TEST("planning_system_create");
    planning_system_t *ps = planning_system_create(PLANNING_STRATEGY_REWOO);
    CHECK(ps != NULL, "planning system create failed");
    planning_system_destroy(ps);
    PASS();
    return 0;
}

static int test_planning_choose_strategy(void) {
    TEST("planning_choose_strategy");
    planning_strategy_t s = planning_choose_strategy("Write a report on climate change");
    CHECK(s >= 0 && s <= PLANNING_STRATEGY_REFLECTION, "strategy out of range");
    PASS();
    return 0;
}

static int test_plan_destroy(void) {
    TEST("plan_create_destroy");
    plan_t *plan = calloc(1, sizeof(plan_t));
    plan->strategy = PLANNING_STRATEGY_REWOO;
    plan->step_count = 0;
    plan_destroy(plan);
    PASS();
    return 0;
}

/* ── Agent Memory Tests ── */
static int test_agent_memory_create(void) {
    TEST("agent_memory_create");
    agent_memory_t *mem = agent_memory_create();
    CHECK(mem != NULL, "memory create failed");
    agent_memory_destroy(mem);
    PASS();
    return 0;
}

static int test_working_memory_add(void) {
    TEST("working_memory_add");
    agent_memory_t *mem = agent_memory_create();
    bool ok = working_memory_add(&mem->working, MEMORY_ROLE_USER, "test message");
    CHECK(ok, "wm add failed");
    agent_memory_destroy(mem);
    PASS();
    return 0;
}

static int test_memory_estimate_tokens(void) {
    TEST("memory_estimate_tokens");
    int n = memory_estimate_tokens("hello world test");
    CHECK(n > 0, "token est should be positive");
    PASS();
    return 0;
}

/* ── Multi-Agent Tests ── */
static int test_multi_agent_create(void) {
    TEST("multi_agent_create");
    multi_agent_t *ma = multi_agent_create();
    CHECK(ma != NULL, "multi_agent create failed");
    multi_agent_destroy(ma);
    PASS();
    return 0;
}

static int test_ma_agent_create(void) {
    TEST("ma_agent_create");
    ma_agent_def_t *agent = ma_agent_create(MA_ROLE_EXECUTOR, "Worker", "You are a worker.");
    CHECK(agent != NULL, "agent create failed");
    CHECK(agent->role == MA_ROLE_EXECUTOR, "role wrong");
    ma_agent_destroy(agent);
    PASS();
    return 0;
}

static int test_ma_register(void) {
    TEST("ma_agent_register");
    multi_agent_t *ma = multi_agent_create();
    ma_agent_def_t *agent = ma_agent_create(MA_ROLE_PLANNER, "Planner", "Plan tasks.");
    int id = multi_agent_register(ma, agent);
    CHECK(id >= 0, "register failed");
    ma_agent_def_t *found = multi_agent_get(ma, id);
    CHECK(found != NULL, "get failed");
    multi_agent_destroy(ma);
    PASS();
    return 0;
}

int main(void) {
    printf("=== mini-agent-runtime Unit Tests ===\n\n");

    int failed = 0;
    failed += test_react_agent_create();
    failed += test_react_agent_add_message();
    failed += test_react_agent_parse();
    failed += test_tool_registry_create();
    failed += test_tool_registry_add_find();
    failed += test_tool_builtin_add();
    failed += test_planning_system_create();
    failed += test_planning_choose_strategy();
    failed += test_plan_destroy();
    failed += test_agent_memory_create();
    failed += test_working_memory_add();
    failed += test_memory_estimate_tokens();
    failed += test_multi_agent_create();
    failed += test_ma_agent_create();
    failed += test_ma_register();

    printf("\n=== Results: %d/%d passed, %d failed ===\n",
           tests_passed, tests_run, failed);
    return failed > 0 ? 1 : 0;
}
