# mini-agent-runtime — 智能体运行时 (C 语言实现)

A lightweight, embeddable agent runtime framework in C99 for building LLM-powered
intelligent agents with tool use, planning, memory, and multi-agent collaboration.

## Features

| Module | Description |
|--------|-------------|
| **ReAct Agent** | Thought→Action→Observation loop with LLM-driven reasoning |
| **Tool Use** | Extensible tool registry with JSON Schema validation |
| **Planning** | ReWOO, Plan-and-Execute, LLMCompiler, Graph of Thought |
| **Memory** | Working, short-term, long-term (vector), summary, episodic memory |
| **Multi-Agent** | Roles, debate, round-robin, supervisor, group chat |

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Multi-Agent System                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │
│  │ Planner  │  │ Executor │  │    Reviewer      │   │
│  └────┬─────┘  └────┬─────┘  └────────┬─────────┘   │
│       │             │                │              │
│  ┌────┴─────────────┴────────────────┴──────────┐   │
│  │              Planning System                  │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────────┐  │   │
│  │  │  ReWOO   │ │ P&E      │ │ LLMCompiler  │  │   │
│  │  └──────────┘ └──────────┘ └──────────────┘  │   │
│  └──────────────────────┬───────────────────────┘   │
│  ┌──────────────────────┴───────────────────────┐   │
│  │                 ReAct Agent                   │   │
│  │        Thought → Action → Observation         │   │
│  └──────────────────────┬───────────────────────┘   │
│  ┌──────────────────────┴───────────────────────┐   │
│  │               Tool Use System                 │   │
│  │   calculator │ search │ file │ code_exec     │   │
│  └──────────────────────┬───────────────────────┘   │
│  ┌──────────────────────┴───────────────────────┐   │
│  │               Agent Memory                   │   │
│  │  Working │ Short-term │ Long-term │ Summary  │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

## Building

```bash
make          # Build library and all examples/demos
make lib      # Build static library libminiruntime.a
make examples # Build examples only
make demos    # Build demos only
make clean    # Clean build artifacts
```

## Quick Start

```c
#include "react_agent.h"
#include "tool_use.h"

int main(void) {
    tool_registry_t *registry = tool_registry_create();
    tool_registry_add_builtin(registry, TOOL_CALCULATOR);
    tool_registry_add_builtin(registry, TOOL_WEB_SEARCH);

    react_agent_t *agent = react_agent_create("You are a helpful assistant.");
    react_agent_set_tools(agent, registry);
    react_agent_set_model_callback(agent, my_llm_callback);

    const char *answer = react_agent_run(agent, "What is 2+2?");
    printf("Answer: %s\n", answer);

    react_agent_destroy(agent);
    tool_registry_destroy(registry);
    return 0;
}
```

## License

MIT
