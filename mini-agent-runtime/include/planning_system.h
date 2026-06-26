#ifndef PLANNING_SYSTEM_H
#define PLANNING_SYSTEM_H

#include <stddef.h>
#include <stdbool.h>

#define PLANNING_MAX_STEPS        32
#define PLANNING_MAX_PLAN_LEN     16384
#define PLANNING_MAX_STEP_DESC    2048
#define PLANNING_MAX_STEP_RESULT  8192
#define PLANNING_MAX_DEPENDENCIES 16
#define PLANNING_MAX_DAG_NODES    64
#define PLANNING_MAX_GRAPH_NODES  64
#define PLANNING_MAX_GRAPH_EDGES  256
#define PLANNING_MAX_REFLECTION   4096
#define PLANNING_MAX_REPLAN_COUNT 5

typedef enum {
    PLANNING_STRATEGY_REWOO,
    PLANNING_STRATEGY_PLAN_EXECUTE,
    PLANNING_STRATEGY_LLM_COMPILER,
    PLANNING_STRATEGY_GRAPH_OF_THOUGHT,
    PLANNING_STRATEGY_REFLECTION
} planning_strategy_t;

typedef enum {
    PLAN_STEP_PENDING,
    PLAN_STEP_RUNNING,
    PLAN_STEP_COMPLETED,
    PLAN_STEP_FAILED,
    PLAN_STEP_SKIPPED
} plan_step_state_t;

typedef struct {
    int id;
    char description[PLANNING_MAX_STEP_DESC];
    char tool_name[64];
    char arguments[PLANNING_MAX_STEP_DESC];
    plan_step_state_t state;
    char result[PLANNING_MAX_STEP_RESULT];
    int dependency_ids[PLANNING_MAX_DEPENDENCIES];
    int dependency_count;
    bool parallelizable;
} plan_step_t;

typedef struct {
    plan_step_t steps[PLANNING_MAX_STEPS];
    int step_count;
    int current_step;
    planning_strategy_t strategy;
    char full_plan[PLANNING_MAX_PLAN_LEN];
    char aggregated_result[PLANNING_MAX_PLAN_LEN];
    int replan_count;
} plan_t;

typedef struct {
    int node_id;
    char tool_name[64];
    char args[2048];
    int dep_ids[PLANNING_MAX_DEPENDENCIES];
    int dep_count;
    char result[PLANNING_MAX_STEP_RESULT];
    bool executed;
} dag_node_t;

typedef struct {
    dag_node_t nodes[PLANNING_MAX_DAG_NODES];
    int node_count;
    int edges[PLANNING_MAX_DAG_NODES][PLANNING_MAX_DAG_NODES];
} compiler_dag_t;

typedef struct {
    int node_id;
    char thought[PLANNING_MAX_STEP_DESC];
    char score_rationale[1024];
    float score;
    bool expanded;
} got_node_t;

typedef struct {
    got_node_t nodes[PLANNING_MAX_GRAPH_NODES];
    int edges[PLANNING_MAX_GRAPH_EDGES][2];
    int node_count;
    int edge_count;
} thought_graph_t;

typedef struct {
    char critique[PLANNING_MAX_REFLECTION];
    char improvements[PLANNING_MAX_REFLECTION];
    char revised_output[PLANNING_MAX_PLAN_LEN];
    int iteration;
} reflection_t;

typedef char* (*planning_llm_callback_t)(const char *prompt, void *user_data);
typedef char* (*planning_executor_callback_t)(const char *tool_name, const char *args, void *user_data);

typedef struct planning_system_s planning_system_t;

planning_system_t* planning_system_create(planning_strategy_t strategy);
void planning_system_destroy(planning_system_t *ps);

void planning_system_set_llm_callback(planning_system_t *ps, planning_llm_callback_t cb, void *user_data);
void planning_system_set_executor_callback(planning_system_t *ps, planning_executor_callback_t cb, void *user_data);
void planning_system_set_tool_registry(planning_system_t *ps, void *tool_registry);
void planning_system_set_max_replans(planning_system_t *ps, int max_replans);

plan_t* planning_system_create_plan(planning_system_t *ps, const char *task);
const char* planning_system_execute_plan(planning_system_t *ps, plan_t *plan, char *buf, size_t buf_size);
bool planning_system_replan(planning_system_t *ps, plan_t *plan, const char *feedback);

plan_t* planning_rew00(planning_system_t *ps, const char *task);
const char* planning_rew00_execute_all(planning_system_t *ps, plan_t *plan, char *buf, size_t buf_size);
const char* planning_rew00_aggregate(planning_system_t *ps, plan_t *plan, char *buf, size_t buf_size);

plan_t* planning_plan_execute(planning_system_t *ps, const char *task);
const char* planning_plan_execute_step(planning_system_t *ps, plan_t *plan, char *buf, size_t buf_size);
bool planning_plan_execute_replan(planning_system_t *ps, plan_t *plan);

compiler_dag_t* planning_llmcompiler_create_dag(planning_system_t *ps, const char *task);
const char* planning_llmcompiler_execute(planning_system_t *ps, compiler_dag_t *dag, char *buf, size_t buf_size);
int* planning_llmcompiler_topological_order(const compiler_dag_t *dag, int *out_count);

thought_graph_t* planning_got_create_graph(planning_system_t *ps, const char *task);
thought_graph_t* planning_got_expand(planning_system_t *ps, thought_graph_t *graph, int node_id);
thought_graph_t* planning_got_evaluate(planning_system_t *ps, thought_graph_t *graph);
const char* planning_got_best_path(planning_system_t *ps, thought_graph_t *graph, char *buf, size_t buf_size);

reflection_t planning_reflect(planning_system_t *ps, const char *output, const char *context);
const char* planning_refine(planning_system_t *ps, const char *original, const reflection_t *reflection, char *buf, size_t buf_size);
bool planning_should_rethink(const char *output);

planning_strategy_t planning_choose_strategy(const char *task);
const char* planning_strategy_name(planning_strategy_t strategy);

void plan_destroy(plan_t *plan);
void compiler_dag_destroy(compiler_dag_t *dag);
void thought_graph_destroy(thought_graph_t *graph);
void plan_step_mark_failed(plan_step_t *step, const char *error);

#endif
