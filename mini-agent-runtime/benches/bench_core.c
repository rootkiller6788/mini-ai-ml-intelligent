/*
 * mini-agent-runtime — Core Benchmarks
 *
 * Benchmarks: ReAct agent, tool use, planning, agent memory, multi-agent.
 */
#include "../src/react_agent.h"
#include "../src/tool_use.h"
#include "../src/planning_system.h"
#include "../src/agent_memory.h"
#include "../src/multi_agent.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
    return (double)clock() / (double)(CLOCKS_PER_SEC / 1000);
}

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 5000;
    double t0, t1;
    printf("=== mini-agent-runtime Benchmarks (N=%d) ===\n\n", N);

    /* ── ReAct Agent Create/Destroy ── */
    {
        t0 = now_ms();
        for (int r = 0; r < N / 20; r++) {
            react_agent_t *agent = react_agent_create("You are a helpful assistant.");
            react_agent_destroy(agent);
        }
        t1 = now_ms();
        printf("  react_agent:         %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 20, t1 - t0, (t1 - t0) / (double)(N / 20) * 1000.0);
    }

    /* ── React Agent Parse ── */
    {
        const char *raw = "Thought: test\nAction: search\nAction Input: query\n";
        t0 = now_ms();
        for (int r = 0; r < N / 10; r++) {
            react_agent_parse_llm_output(raw);
        }
        t1 = now_ms();
        printf("  react_parse:         %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 10, t1 - t0, (t1 - t0) / (double)(N / 10) * 1000.0);
    }

    /* ── Tool Registry ── */
    {
        t0 = now_ms();
        for (int r = 0; r < N / 20; r++) {
            tool_registry_t *reg = tool_registry_create();
            tool_def_t def;
            memset(&def, 0, sizeof(def));
            strcpy(def.name, "search");
            def.type = TOOL_TYPE_WEB_SEARCH;
            tool_registry_add(reg, &def, NULL, NULL);
            tool_registry_destroy(reg);
        }
        t1 = now_ms();
        printf("  tool_registry_add:   %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 20, t1 - t0, (t1 - t0) / (double)(N / 20) * 1000.0);
    }

    /* ── Tool Schema JSON ── */
    {
        tool_registry_t *reg = tool_registry_create();
        tool_def_t def;
        memset(&def, 0, sizeof(def));
        strcpy(def.name, "calc");
        def.type = TOOL_TYPE_CALCULATOR;
        tool_registry_add(reg, &def, NULL, NULL);
        char buf[4096];
        t0 = now_ms();
        for (int r = 0; r < N / 10; r++) {
            tool_get_schemas_json(reg, buf, sizeof(buf));
        }
        t1 = now_ms();
        printf("  tool_schemas_json:   %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 10, t1 - t0, (t1 - t0) / (double)(N / 10) * 1000.0);
        tool_registry_destroy(reg);
    }

    /* ── Planning System ── */
    {
        t0 = now_ms();
        for (int r = 0; r < N / 20; r++) {
            planning_system_t *ps = planning_system_create(PLANNING_STRATEGY_REWOO);
            planning_system_destroy(ps);
        }
        t1 = now_ms();
        printf("  planning_sys_create: %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 20, t1 - t0, (t1 - t0) / (double)(N / 20) * 1000.0);
    }

    /* ── Agent Memory ── */
    {
        t0 = now_ms();
        for (int r = 0; r < N / 20; r++) {
            agent_memory_t *mem = agent_memory_create();
            working_memory_add(&mem->working, MEMORY_ROLE_USER, "test");
            agent_memory_destroy(mem);
        }
        t1 = now_ms();
        printf("  agent_memory:        %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 20, t1 - t0, (t1 - t0) / (double)(N / 20) * 1000.0);
    }

    /* ── Multi-Agent ── */
    {
        t0 = now_ms();
        for (int r = 0; r < N / 20; r++) {
            multi_agent_t *ma = multi_agent_create();
            ma_agent_def_t *a = ma_agent_create(MA_ROLE_EXECUTOR, "w", "worker");
            multi_agent_register(ma, a);
            multi_agent_destroy(ma);
        }
        t1 = now_ms();
        printf("  multi_agent:         %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 20, t1 - t0, (t1 - t0) / (double)(N / 20) * 1000.0);
    }

    printf("\nDone.\n");
    return 0;
}
