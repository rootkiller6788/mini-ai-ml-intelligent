#include "planning_system.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char* mock_llm_planner(const char *prompt, void *user_data) {
    (void)user_data;
    char *resp = (char*)malloc(4096);
    if (!resp) return NULL;
    if (strstr(prompt, "Create a plan")) {
        strcpy(resp,
            "Step 1: [web_search] Search for current weather data API\n"
            "Step 2: [calculator] Calculate average temperature across cities\n"
            "Step 3: [file_reader] Read stored preference data\n"
            "Step 4: [code_executor] Run data aggregation script");
    } else if (strstr(prompt, "Aggregate these step results")) {
        strcpy(resp, "Final Aggregation: All steps completed. Weather data collected, "
               "temperatures calculated, preferences loaded, and data aggregated. "
               "The weather analysis report is ready.");
    } else if (strstr(prompt, "Create a step-by-step execution plan")) {
        strcpy(resp,
            "web_search: Search for the requested information\n"
            "calculator: Process numerical data\n"
            "file_reader: Load configuration from file\n"
            "code_executor: Execute final processing script");
    } else if (strstr(prompt, "Critique this output")) {
        strcpy(resp,
            "Critique: The output is comprehensive but could be more concise.\n"
            "Improvements: Remove redundant phrases, add concrete numbers.\n"
            "Revised: Weather analysis complete. Avg temp: 22C. "
            "Data from 3 cities analyzed and aggregated.");
    } else if (strstr(prompt, "Refine this output")) {
        strcpy(resp, "Refined: Weather analysis complete. "
               "Average temperature: 22C across 3 cities. Report ready.");
    } else if (strstr(prompt, "Generate up to 3 expanded thoughts")) {
        strcpy(resp,
            "We could approach this by analyzing seasonal patterns\n"
            "Another angle: compare with historical weather data\n"
            "Third option: focus on extreme weather events prediction");
    } else if (strstr(prompt, "Score this thought")) {
        strcpy(resp, "0.85");
    } else {
        strcpy(resp, "default response");
    }
    return resp;
}

static char* mock_executor(const char *tool_name, const char *args, void *user_data) {
    (void)user_data;
    char *result = (char*)malloc(1024);
    if (!result) return NULL;
    snprintf(result, 1024, "[Executed %s with args: %.100s] - result data", tool_name, args);
    return result;
}

int main(void) {
    printf("=== Planning System Example ===\n\n");

    printf("--- Strategy Selection ---\n");
    const char *tasks[] = {
        "plan and execute this workflow",
        "brainstorm creative ideas",
        "run parallel computations",
        "just do this simple task"
    };
    for (int i = 0; i < 4; i++) {
        planning_strategy_t strat = planning_choose_strategy(tasks[i]);
        printf("  Task: \"%s\"\n", tasks[i]);
        printf("  Chosen strategy: %s\n\n", planning_strategy_name(strat));
    }

    printf("\n--- ReWOO Strategy ---\n");
    planning_system_t *ps_rew00 = planning_system_create(PLANNING_STRATEGY_REWOO);
    planning_system_set_llm_callback(ps_rew00, mock_llm_planner, NULL);
    planning_system_set_executor_callback(ps_rew00, mock_executor, NULL);
    planning_system_set_max_replans(ps_rew00, 3);

    plan_t *plan = planning_rew00(ps_rew00, "Get weather data for 3 cities and analyze");
    if (plan) {
        printf("  Plan created with %d steps:\n", plan->step_count);
        for (int i = 0; i < plan->step_count; i++) {
            printf("    Step %d: [%s] %s\n", i, plan->steps[i].tool_name, plan->steps[i].description);
        }
        printf("  Full plan text: %.200s...\n", plan->full_plan);

        char exec_buf[8192];
        planning_rew00_execute_all(ps_rew00, plan, exec_buf, sizeof(exec_buf));
        printf("\n  Execution results:\n%s\n", exec_buf);

        char agg_buf[4096];
        planning_rew00_aggregate(ps_rew00, plan, agg_buf, sizeof(agg_buf));
        printf("  Aggregated result: %s\n", agg_buf);

        plan_destroy(plan);
    }
    planning_system_destroy(ps_rew00);

    printf("\n--- Plan-and-Execute Strategy ---\n");
    planning_system_t *ps_pe = planning_system_create(PLANNING_STRATEGY_PLAN_EXECUTE);
    planning_system_set_llm_callback(ps_pe, mock_llm_planner, NULL);
    planning_system_set_executor_callback(ps_pe, mock_executor, NULL);

    plan_t *pe_plan = planning_plan_execute(ps_pe, "Research and write a weather report");
    if (pe_plan) {
        printf("  Plan created with %d steps:\n", pe_plan->step_count);
        for (int i = 0; i < pe_plan->step_count; i++) {
            printf("    Step %d: [%s] %s\n", i, pe_plan->steps[i].tool_name, pe_plan->steps[i].description);
        }
        printf("  Replan limit: %d\n", ps_pe->max_replans);

        char step_buf[8192];
        planning_plan_execute_step(ps_pe, pe_plan, step_buf, sizeof(step_buf));
        printf("\n  Step execution:\n%s\n", step_buf);

        printf("  Executed %d/%d steps\n", pe_plan->current_step, pe_plan->step_count);

        plan_destroy(pe_plan);
    }
    planning_system_destroy(ps_pe);

    printf("\n--- LLMCompiler (DAG) Strategy ---\n");
    planning_system_t *ps_lc = planning_system_create(PLANNING_STRATEGY_LLM_COMPILER);
    planning_system_set_llm_callback(ps_lc, mock_llm_planner, NULL);
    planning_system_set_executor_callback(ps_lc, mock_executor, NULL);

    compiler_dag_t *dag = planning_llmcompiler_create_dag(ps_lc, "Analyze and format data in parallel");
    if (dag) {
        printf("  DAG created with %d nodes\n", dag->node_count);
        for (int i = 0; i < dag->node_count; i++) {
            printf("    Node %d: %s(", i, dag->nodes[i].tool_name);
            for (int d = 0; d < dag->nodes[i].dep_count; d++) {
                printf("%s%d", d > 0 ? "," : "", dag->nodes[i].dep_ids[d]);
            }
            printf(")\n");
        }

        int order_count = 0;
        int *order = planning_llmcompiler_topological_order(dag, &order_count);
        if (order) {
            printf("\n  Topological order:");
            for (int i = 0; i < order_count; i++) printf(" %d", order[i]);
            printf("\n");
            free(order);
        }

        char dag_buf[8192];
        planning_llmcompiler_execute(ps_lc, dag, dag_buf, sizeof(dag_buf));
        printf("\n  Execution:\n%s\n", dag_buf);

        compiler_dag_destroy(dag);
    }
    planning_system_destroy(ps_lc);

    printf("\n--- Graph of Thought Strategy ---\n");
    planning_system_t *ps_got = planning_system_create(PLANNING_STRATEGY_GRAPH_OF_THOUGHT);
    planning_system_set_llm_callback(ps_got, mock_llm_planner, NULL);

    thought_graph_t *graph = planning_got_create_graph(ps_got, "How to improve weather prediction?");
    if (graph) {
        printf("  Initial graph: %d nodes, %d edges\n", graph->node_count, graph->edge_count);
        printf("  Root thought: %.100s...\n", graph->nodes[0].thought);

        graph = planning_got_expand(ps_got, graph, 0);
        printf("  After expansion: %d nodes, %d edges\n", graph->node_count, graph->edge_count);
        for (int i = 1; i < graph->node_count; i++) {
            printf("    Node %d (score %.2f): %.80s...\n", i, graph->nodes[i].score, graph->nodes[i].thought);
        }

        graph = planning_got_evaluate(ps_got, graph);
        printf("  After evaluation:\n");
        for (int i = 0; i < graph->node_count; i++) {
            printf("    Node %d score: %.2f\n", i, graph->nodes[i].score);
        }

        char best_buf[4096];
        planning_got_best_path(ps_got, graph, best_buf, sizeof(best_buf));
        printf("\n  Best path:\n%s\n", best_buf);

        thought_graph_destroy(graph);
    }
    planning_system_destroy(ps_got);

    printf("\n--- Reflection Strategy ---\n");
    planning_system_t *ps_ref = planning_system_create(PLANNING_STRATEGY_REFLECTION);
    planning_system_set_llm_callback(ps_ref, mock_llm_planner, NULL);

    const char *sample_output = "The weather is probably going to be nice, I think.";
    printf("  Should rethink? %s\n", planning_should_rethink(sample_output) ? "Yes" : "No");

    reflection_t ref = planning_reflect(ps_ref, sample_output, "Weather prediction task");
    printf("  Critique: %.200s...\n", ref.critique[0] ? ref.critique : "(none)");
    printf("  Improvements: %.200s...\n", ref.improvements[0] ? ref.improvements : "(none)");
    printf("  Revised: %.200s...\n", ref.revised_output[0] ? ref.revised_output : "(none)");

    char refined_buf[4096];
    planning_refine(ps_ref, sample_output, &ref, refined_buf, sizeof(refined_buf));
    printf("  Refined output: %s\n", refined_buf);

    planning_system_destroy(ps_ref);

    printf("\nExample completed successfully.\n");
    return 0;
}
