#include "planning_system.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>

struct planning_system_s {
    planning_strategy_t strategy;
    planning_llm_callback_t llm_callback;
    planning_executor_callback_t executor_callback;
    void *user_data;
    void *tool_registry;
    int max_replans;
};

planning_system_t* planning_system_create(planning_strategy_t strategy) {
    planning_system_t *ps = (planning_system_t*)calloc(1, sizeof(planning_system_t));
    if (!ps) return NULL;
    ps->strategy = strategy;
    ps->max_replans = PLANNING_MAX_REPLAN_COUNT;
    return ps;
}

void planning_system_destroy(planning_system_t *ps) {
    if (ps) free(ps);
}

void planning_system_set_llm_callback(planning_system_t *ps, planning_llm_callback_t cb, void *user_data) {
    if (!ps) return;
    ps->llm_callback = cb;
    ps->user_data = user_data;
}

void planning_system_set_executor_callback(planning_system_t *ps, planning_executor_callback_t cb, void *user_data) {
    if (!ps) return;
    ps->executor_callback = cb;
    ps->user_data = user_data;
}

void planning_system_set_tool_registry(planning_system_t *ps, void *tool_registry) {
    if (ps) ps->tool_registry = tool_registry;
}

void planning_system_set_max_replans(planning_system_t *ps, int max_replans) {
    if (ps) ps->max_replans = max_replans;
}

static char* ps_call_llm(planning_system_t *ps, const char *prompt) {
    if (!ps->llm_callback) return NULL;
    return ps->llm_callback(prompt, ps->user_data);
}

static char* ps_execute(planning_system_t *ps, const char *tool_name, const char *args) {
    if (!ps->executor_callback) return NULL;
    return ps->executor_callback(tool_name, args, ps->user_data);
}

plan_t* planning_system_create_plan(planning_system_t *ps, const char *task) {
    if (!ps) return NULL;
    switch (ps->strategy) {
        case PLANNING_STRATEGY_REWOO:        return planning_rew00(ps, task);
        case PLANNING_STRATEGY_PLAN_EXECUTE: return planning_plan_execute(ps, task);
        default: return NULL;
    }
}

const char* planning_system_execute_plan(planning_system_t *ps, plan_t *plan, char *buf, size_t buf_size) {
    if (!ps || !plan) return NULL;
    switch (plan->strategy) {
        case PLANNING_STRATEGY_REWOO:
            planning_rew00_execute_all(ps, plan, buf, buf_size);
            return planning_rew00_aggregate(ps, plan, buf, buf_size);
        case PLANNING_STRATEGY_PLAN_EXECUTE:
            return planning_plan_execute_step(ps, plan, buf, buf_size);
        default: return NULL;
    }
}

bool planning_system_replan(planning_system_t *ps, plan_t *plan, const char *feedback) {
    if (!ps || !plan || plan->replan_count >= ps->max_replans) return false;
    (void)feedback;
    plan->replan_count++;
    return true;
}

plan_t* planning_rew00(planning_system_t *ps, const char *task) {
    plan_t *plan = (plan_t*)calloc(1, sizeof(plan_t));
    if (!plan) return NULL;
    plan->strategy = PLANNING_STRATEGY_REWOO;
    char prompt[16384];
    snprintf(prompt, sizeof(prompt),
        "Create a plan to solve this task. List each step as:\n"
        "Step N: [TOOL_NAME] description\n\nTask: %s", task);
    char *raw = ps_call_llm(ps, prompt);
    if (raw) {
        strncpy(plan->full_plan, raw, PLANNING_MAX_PLAN_LEN - 1);
        const char *p = raw;
        int step_num = 0;
        while (*p && step_num < PLANNING_MAX_STEPS) {
            const char *step_tag = strstr(p, "Step ");
            if (!step_tag) break;
            p = step_tag + 5;
            plan_step_t *step = &plan->steps[step_num];
            step->id = step_num;
            step->state = PLAN_STEP_PENDING;
            step->parallelizable = true;
            const char *bracket = strchr(p, '[');
            if (bracket) {
                bracket++;
                const char *cb = strchr(bracket, ']');
                if (cb) {
                    size_t len = (size_t)(cb - bracket);
                    if (len >= sizeof(step->tool_name)) len = sizeof(step->tool_name) - 1;
                    strncpy(step->tool_name, bracket, len);
                    p = cb + 1;
                }
            }
            while (*p == ' ' || *p == ':') p++;
            const char *next_step = strstr(p, "Step ");
            size_t len;
            if (next_step) {
                len = (size_t)(next_step - p);
            } else {
                len = strlen(p);
            }
            if (len >= PLANNING_MAX_STEP_DESC) len = PLANNING_MAX_STEP_DESC - 1;
            strncpy(step->description, p, len);
            step_num++;
            plan->step_count = step_num;
        }
        free(raw);
    }
    return plan;
}

const char* planning_rew00_execute_all(planning_system_t *ps, plan_t *plan, char *buf, size_t buf_size) {
    if (!ps || !plan || !buf) return NULL;
    int offset = 0;
    for (int i = 0; i < plan->step_count; i++) {
        plan_step_t *step = &plan->steps[i];
        step->state = PLAN_STEP_RUNNING;
        char *result = ps_execute(ps, step->tool_name, step->description);
        if (result) {
            strncpy(step->result, result, PLANNING_MAX_STEP_RESULT - 1);
            step->state = PLAN_STEP_COMPLETED;
            offset += snprintf(buf + offset, buf_size - offset,
                              "Step %d [%s]: %s\n", i, step->tool_name, step->result);
            free(result);
        } else {
            step->state = PLAN_STEP_FAILED;
            plan_step_mark_failed(step, "Execution returned NULL");
        }
    }
    return buf;
}

const char* planning_rew00_aggregate(planning_system_t *ps, plan_t *plan, char *buf, size_t buf_size) {
    if (!ps || !plan || !buf) return NULL;
    char steps_text[PLANNING_MAX_PLAN_LEN] = {0};
    int off = 0;
    for (int i = 0; i < plan->step_count; i++) {
        off += snprintf(steps_text + off, sizeof(steps_text) - off,
                       "Step %d result: %s\n", i, plan->steps[i].result);
    }
    char prompt[16384];
    snprintf(prompt, sizeof(prompt),
        "Aggregate these step results into a final answer:\n%s", steps_text);
    char *agg = ps_call_llm(ps, prompt);
    if (agg) {
        strncpy(buf, agg, buf_size - 1);
        strncpy(plan->aggregated_result, agg, PLANNING_MAX_PLAN_LEN - 1);
        free(agg);
    } else {
        strncpy(buf, steps_text, buf_size - 1);
    }
    return buf;
}

plan_t* planning_plan_execute(planning_system_t *ps, const char *task) {
    plan_t *plan = (plan_t*)calloc(1, sizeof(plan_t));
    if (!plan) return NULL;
    plan->strategy = PLANNING_STRATEGY_PLAN_EXECUTE;
    char prompt[16384];
    snprintf(prompt, sizeof(prompt),
        "Create a step-by-step execution plan. For each step, specify the tool and action.\n"
        "Format: TOOL_NAME: action description\n\nTask: %s", task);
    char *raw = ps_call_llm(ps, prompt);
    if (raw) {
        strncpy(plan->full_plan, raw, PLANNING_MAX_PLAN_LEN - 1);
        const char *p = raw;
        int step_num = 0;
        while (*p && step_num < PLANNING_MAX_STEPS) {
            plan_step_t *step = &plan->steps[step_num];
            step->id = step_num;
            step->state = PLAN_STEP_PENDING;
            step->parallelizable = false;
            const char *colon = strchr(p, ':');
            if (colon) {
                size_t tool_len = (size_t)(colon - p);
                if (tool_len >= sizeof(step->tool_name)) tool_len = sizeof(step->tool_name) - 1;
                strncpy(step->tool_name, p, tool_len);
                p = colon + 1;
                while (*p == ' ') p++;
                const char *newline = strchr(p, '\n');
                if (!newline) newline = p + strlen(p);
                size_t desc_len = (size_t)(newline - p);
                if (desc_len >= PLANNING_MAX_STEP_DESC) desc_len = PLANNING_MAX_STEP_DESC - 1;
                strncpy(step->description, p, desc_len);
                p = newline;
                if (*p) p++;
            } else {
                break;
            }
            step_num++;
            plan->step_count = step_num;
        }
        free(raw);
    }
    return plan;
}

const char* planning_plan_execute_step(planning_system_t *ps, plan_t *plan, char *buf, size_t buf_size) {
    if (!ps || !plan || !buf) return NULL;
    int offset = 0;
    while (plan->current_step < plan->step_count) {
        int i = plan->current_step;
        plan_step_t *step = &plan->steps[i];
        step->state = PLAN_STEP_RUNNING;
        char *result = ps_execute(ps, step->tool_name, step->description);
        if (result) {
            strncpy(step->result, result, PLANNING_MAX_STEP_RESULT - 1);
            step->state = PLAN_STEP_COMPLETED;
            offset += snprintf(buf + offset, buf_size - offset,
                              "Step %d: %s\n", i, step->result);
            free(result);
            plan->current_step++;
        } else {
            step->state = PLAN_STEP_FAILED;
            offset += snprintf(buf + offset, buf_size - offset,
                              "Step %d FAILED: %s\n", i, step->tool_name);
            if (!planning_plan_execute_replan(ps, plan)) break;
        }
    }
    return buf;
}

bool planning_plan_execute_replan(planning_system_t *ps, plan_t *plan) {
    if (!ps || !plan) return false;
    if (plan->replan_count >= ps->max_replans) return false;
    plan->replan_count++;
    return true;
}

compiler_dag_t* planning_llmcompiler_create_dag(planning_system_t *ps, const char *task) {
    compiler_dag_t *dag = (compiler_dag_t*)calloc(1, sizeof(compiler_dag_t));
    if (!dag) return NULL;
    char prompt[16384];
    snprintf(prompt, sizeof(prompt),
        "Create a DAG of function calls for this task. List each function and its dependencies.\n"
        "Format: ID: FUNCTION_NAME(args) depends_on=[id1,id2,...]\n\nTask: %s", task);
    char *raw = ps_call_llm(ps, prompt);
    if (raw) {
        const char *p = raw;
        while (*p && dag->node_count < PLANNING_MAX_DAG_NODES) {
            dag_node_t *node = &dag->nodes[dag->node_count];
            node->node_id = dag->node_count;
            node->executed = false;
            const char *paren = strchr(p, '(');
            const char *colon = strchr(p, ':');
            if (paren && colon && colon < paren) {
                size_t tlen = (size_t)(paren - colon - 1);
                if (tlen >= sizeof(node->tool_name)) tlen = sizeof(node->tool_name) - 1;
                strncpy(node->tool_name, colon + 1, tlen);
            }
            if (paren) {
                const char *close = strchr(paren, ')');
                if (close) {
                    size_t alen = (size_t)(close - paren - 1);
                    if (alen >= sizeof(node->args)) alen = sizeof(node->args) - 1;
                    strncpy(node->args, paren + 1, alen);
                }
            }
            const char *dep = strstr(p, "depends_on");
            if (dep) {
                dep = strchr(dep, '[');
                if (dep) {
                    dep++;
                    while (*dep && *dep != ']' && node->dep_count < PLANNING_MAX_DEPENDENCIES) {
                        while (*dep == ' ' || *dep == ',') dep++;
                        int dep_id = 0;
                        while (isdigit((unsigned char)*dep)) {
                            dep_id = dep_id * 10 + (*dep - '0');
                            dep++;
                        }
                        node->dep_ids[node->dep_count++] = dep_id;
                        dag->edges[node->node_id][dep_id] = 1;
                    }
                }
            }
            dag->node_count++;
            p = strchr(p, '\n');
            if (p) p++; else break;
        }
        free(raw);
    }
    return dag;
}

const char* planning_llmcompiler_execute(planning_system_t *ps, compiler_dag_t *dag, char *buf, size_t buf_size) {
    if (!ps || !dag || !buf) return NULL;
    int order_count = 0;
    int *order = planning_llmcompiler_topological_order(dag, &order_count);
    int offset = 0;
    for (int i = 0; i < order_count; i++) {
        int id = order[i];
        dag_node_t *node = &dag->nodes[id];
        if (node->executed) continue;
        char *result = ps_execute(ps, node->tool_name, node->args);
        if (result) {
            strncpy(node->result, result, PLANNING_MAX_STEP_RESULT - 1);
            node->executed = true;
            offset += snprintf(buf + offset, buf_size - offset,
                              "Node %d [%s]: %s\n", id, node->tool_name, node->result);
            free(result);
        }
    }
    free(order);
    return buf;
}

int* planning_llmcompiler_topological_order(const compiler_dag_t *dag, int *out_count) {
    if (!dag || !out_count) return NULL;
    int n = dag->node_count;
    int *indegree = (int*)calloc(n, sizeof(int));
    int *result = (int*)malloc(n * sizeof(int));
    if (!indegree || !result) { free(indegree); free(result); *out_count = 0; return NULL; }
    for (int i = 0; i < n; i++) {
        for (int d = 0; d < dag->nodes[i].dep_count; d++) {
            indegree[i]++;
        }
    }
    int idx = 0;
    bool *visited = (bool*)calloc(n, sizeof(bool));
    for (int round = 0; round < n; round++) {
        for (int i = 0; i < n; i++) {
            if (!visited[i] && indegree[i] == 0) {
                result[idx++] = i;
                visited[i] = true;
                for (int j = 0; j < n; j++) {
                    for (int d = 0; d < dag->nodes[j].dep_count; d++) {
                        if (dag->nodes[j].dep_ids[d] == i) indegree[j]--;
                    }
                }
            }
        }
    }
    free(indegree);
    free(visited);
    *out_count = idx;
    return result;
}

thought_graph_t* planning_got_create_graph(planning_system_t *ps, const char *task) {
    thought_graph_t *graph = (thought_graph_t*)calloc(1, sizeof(thought_graph_t));
    if (!graph) return NULL;
    got_node_t *root = &graph->nodes[0];
    root->node_id = 0;
    root->score = 1.0f;
    root->expanded = false;
    char prompt[16384];
    snprintf(prompt, sizeof(prompt),
        "Generate initial thoughts for this task as nodes:\n%s", task);
    char *raw = ps_call_llm(ps, prompt);
    if (raw) {
        strncpy(root->thought, raw, PLANNING_MAX_STEP_DESC - 1);
        free(raw);
    } else {
        strncpy(root->thought, task, PLANNING_MAX_STEP_DESC - 1);
    }
    graph->node_count = 1;
    graph->edge_count = 0;
    return graph;
}

thought_graph_t* planning_got_expand(planning_system_t *ps, thought_graph_t *graph, int node_id) {
    if (!ps || !graph || node_id >= graph->node_count) return graph;
    got_node_t *node = &graph->nodes[node_id];
    if (node->expanded) return graph;
    node->expanded = true;
    if (graph->node_count + 3 >= PLANNING_MAX_GRAPH_NODES) return graph;
    char prompt[8192];
    snprintf(prompt, sizeof(prompt),
        "Generate up to 3 expanded thoughts based on: %s", node->thought);
    char *raw = ps_call_llm(ps, prompt);
    if (!raw) return graph;
    int added = 0;
    const char *p = raw;
    while (*p && added < 3 && graph->node_count < PLANNING_MAX_GRAPH_NODES) {
        const char *nl = strchr(p, '\n');
        if (!nl) nl = p + strlen(p);
        if (nl > p) {
            int new_id = graph->node_count;
            got_node_t *new_node = &graph->nodes[new_id];
            new_node->node_id = new_id;
            size_t len = (size_t)(nl - p);
            if (len >= PLANNING_MAX_STEP_DESC) len = PLANNING_MAX_STEP_DESC - 1;
            strncpy(new_node->thought, p, len);
            new_node->score = 0.5f;
            new_node->expanded = false;
            if (graph->edge_count < PLANNING_MAX_GRAPH_EDGES) {
                graph->edges[graph->edge_count][0] = node_id;
                graph->edges[graph->edge_count][1] = new_id;
                graph->edge_count++;
            }
            graph->node_count++;
            added++;
        }
        p = *nl ? nl + 1 : nl;
    }
    free(raw);
    return graph;
}

thought_graph_t* planning_got_evaluate(planning_system_t *ps, thought_graph_t *graph) {
    if (!ps || !graph) return graph;
    for (int i = 0; i < graph->node_count; i++) {
        got_node_t *node = &graph->nodes[i];
        char prompt[8192];
        snprintf(prompt, sizeof(prompt),
            "Score this thought on a scale of 0-1 for relevance and quality:\n%s", node->thought);
        char *raw = ps_call_llm(ps, prompt);
        if (raw) {
            float score = 0.5f;
            sscanf(raw, "%f", &score);
            if (score < 0.0f) score = 0.0f;
            if (score > 1.0f) score = 1.0f;
            node->score = score;
            free(raw);
        }
    }
    return graph;
}

const char* planning_got_best_path(planning_system_t *ps, thought_graph_t *graph, char *buf, size_t buf_size) {
    if (!graph || !buf) return NULL;
    (void)ps;
    int best_id = 0;
    float best_score = -1.0f;
    for (int i = 0; i < graph->node_count; i++) {
        if (graph->nodes[i].score > best_score) {
            best_score = graph->nodes[i].score;
            best_id = i;
        }
    }
    int path_nodes[PLANNING_MAX_GRAPH_NODES];
    int path_len = 0;
    int cur = best_id;
    while (path_len < PLANNING_MAX_GRAPH_NODES) {
        path_nodes[path_len++] = cur;
        int parent = -1;
        for (int e = 0; e < graph->edge_count; e++) {
            if (graph->edges[e][1] == cur) {
                parent = graph->edges[e][0];
                break;
            }
        }
        if (parent < 0) break;
        cur = parent;
    }
    int off = 0;
    for (int i = path_len - 1; i >= 0; i--) {
        off += snprintf(buf + off, buf_size - off, "-> %s\n", graph->nodes[path_nodes[i]].thought);
    }
    return buf;
}

reflection_t planning_reflect(planning_system_t *ps, const char *output, const char *context) {
    reflection_t ref;
    memset(&ref, 0, sizeof(ref));
    ref.iteration = 1;
    if (!ps || !output) return ref;
    char prompt[16384];
    snprintf(prompt, sizeof(prompt),
        "Critique this output and suggest improvements:\n\nContext: %s\n\nOutput: %s\n\n"
        "Provide: Critique, Improvements, and Revised Output.",
        context ? context : "None", output);
    char *raw = ps_call_llm(ps, prompt);
    if (raw) {
        const char *crit = strstr(raw, "Critique:");
        if (crit) {
            crit += 9;
            const char *imp = strstr(crit, "Improvements:");
            size_t len = imp ? (size_t)(imp - crit) : strlen(crit);
            if (len >= PLANNING_MAX_REFLECTION) len = PLANNING_MAX_REFLECTION - 1;
            strncpy(ref.critique, crit, len);
        }
        const char *imp = strstr(raw, "Improvements:");
        if (imp) {
            imp += 13;
            const char *rev = strstr(imp, "Revised");
            size_t len = rev ? (size_t)(rev - imp) : strlen(imp);
            if (len >= PLANNING_MAX_REFLECTION) len = PLANNING_MAX_REFLECTION - 1;
            strncpy(ref.improvements, imp, len);
        }
        const char *rev = strstr(raw, "Revised");
        if (rev) {
            rev = strchr(rev, ':');
            if (rev) {
                rev++;
                while (*rev == ' ' || *rev == '\n') rev++;
                strncpy(ref.revised_output, rev, PLANNING_MAX_PLAN_LEN - 1);
            }
        }
        free(raw);
    }
    return ref;
}

const char* planning_refine(planning_system_t *ps, const char *original, const reflection_t *reflection, char *buf, size_t buf_size) {
    if (!buf) return NULL;
    if (reflection && reflection->revised_output[0]) {
        strncpy(buf, reflection->revised_output, buf_size - 1);
        return buf;
    }
    char prompt[8192];
    snprintf(prompt, sizeof(prompt),
        "Refine this output based on the critique: %s\n\nOriginal: %s", 
        reflection ? reflection->critique : "", original);
    char *raw = ps_call_llm(ps, prompt);
    if (raw) {
        strncpy(buf, raw, buf_size - 1);
        free(raw);
    } else {
        strncpy(buf, original, buf_size - 1);
    }
    return buf;
}

bool planning_should_rethink(const char *output) {
    if (!output) return true;
    const char *markers[] = {"I think", "maybe", "probably", "might be", "not sure", NULL};
    for (int i = 0; markers[i]; i++) {
        if (strstr(output, markers[i])) return true;
    }
    return false;
}

planning_strategy_t planning_choose_strategy(const char *task) {
    if (!task) return PLANNING_STRATEGY_REWOO;
    if (strstr(task, "parallel") || strstr(task, "concurrent")) return PLANNING_STRATEGY_LLM_COMPILER;
    if (strstr(task, "brainstorm") || strstr(task, "creative")) return PLANNING_STRATEGY_GRAPH_OF_THOUGHT;
    if (strstr(task, "plan") || strstr(task, "steps")) return PLANNING_STRATEGY_PLAN_EXECUTE;
    return PLANNING_STRATEGY_REWOO;
}

const char* planning_strategy_name(planning_strategy_t strategy) {
    switch (strategy) {
        case PLANNING_STRATEGY_REWOO:          return "ReWOO";
        case PLANNING_STRATEGY_PLAN_EXECUTE:   return "Plan-and-Execute";
        case PLANNING_STRATEGY_LLM_COMPILER:   return "LLMCompiler";
        case PLANNING_STRATEGY_GRAPH_OF_THOUGHT: return "Graph of Thought";
        case PLANNING_STRATEGY_REFLECTION:     return "Reflection";
        default: return "Unknown";
    }
}

void plan_destroy(plan_t *plan) { if (plan) free(plan); }
void compiler_dag_destroy(compiler_dag_t *dag) { if (dag) free(dag); }
void thought_graph_destroy(thought_graph_t *graph) { if (graph) free(graph); }

void plan_step_mark_failed(plan_step_t *step, const char *error) {
    if (!step) return;
    step->state = PLAN_STEP_FAILED;
    if (error) strncpy(step->result, error, PLANNING_MAX_STEP_RESULT - 1);
}
