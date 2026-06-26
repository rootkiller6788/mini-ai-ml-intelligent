# mini-agent-runtime Design Document

## Overview

mini-agent-runtime is a lightweight, embeddable C99 framework for building LLM-powered
intelligent agents. It provides the foundational building blocks for agent-based
AI systems: reasoning loops, tool calling, planning strategies, memory architectures,
and multi-agent collaboration.

## Design Philosophy

### 1. Minimal Dependencies

The framework is pure C99 with only the standard library (`stdlib.h`, `string.h`,
`stdio.h`, `ctype.h`, `time.h`, `math.h`). No external dependencies, JSON parsers,
or HTTP libraries. LLM calls are injected via callback functions, making the
framework backend-agnostic.

### 2. Embeddable Architecture

All state is heap-allocated via opaque structs. Memory management follows
create/destroy patterns. The library compiles to a static library that can be
embedded in any C or C++ application, including resource-constrained environments.

### 3. Modular Design

Each module (ReAct, Tool Use, Planning, Memory, Multi-Agent) is independent.
They can be used individually or composed together. Integration between modules
occurs via void pointers and callback functions rather than direct includes.

## Architecture

```
Layer 5: Multi-Agent System
  - Agent roles, message passing, debate, round-robin
  - Supervisor delegation, group chat, conversation programming

Layer 4: Planning System
  - ReWOO: Plan → Execute All → Aggregate
  - Plan-and-Execute: Plan → Step-by-step → Replan
  - LLMCompiler: DAG-based parallel execution
  - Graph of Thought: Expand → Evaluate → Best Path
  - Reflection: Critique → Improve

Layer 3: ReAct Agent
  - Thought → Action → Observation loop
  - LLM output parsing (JSON/text)
  - Max iteration guard, state tracking

Layer 2: Tool Use System
  - Registry-based tool management
  - JSON Schema parameter validation
  - Parallel tool execution
  - Built-in tools (calculator, search, file, code)

Layer 1: Agent Memory
  - Working memory (conversation buffer)
  - Short-term memory (session buffer, TTL)
  - Long-term memory (vector store, cosine similarity)
  - Summary memory (conversation compression)
  - Episodic memory (interaction episodes)
  - Reflection memory (high-level insights)
```

## Module Details

### ReAct Agent

The ReAct loop follows the canonical Thought→Action→Observation pattern:

1. **Thought**: LLM reasons about what to do next
2. **Action**: LLM selects a tool and provides arguments
3. **Observation**: Tool result is fed back to the LLM
4. **Loop**: Continue until Final Answer or max iterations

The prompt includes:
- System message with agent role
- Tool descriptions with parameter schemas
- Conversation history
- User query

Output parsing supports two formats:
- **Text format**: `Thought: ... Action: ... Action Input: ...`
- **JSON format**: `{"function": "...", "arguments": {...}}`

### Tool Use

Tools are registered with:
- Name and description
- Parameter definitions with types (string, integer, float, boolean)
- Required/optional flags
- Function pointer for execution

Built-in tools demonstrate the pattern:
- **calculator**: Simple mathematical expression evaluator
- **web_search**: Simulated web search
- **file_reader**: Simulated file reading
- **code_executor**: Simulated code execution

Parallel execution sends multiple tool calls to be resolved concurrently.

### Planning System

Five planning strategies are implemented:

**ReWOO (Reason Without Observation)**: Plans all steps first, executes them
independently, then aggregates results into a final answer. Most efficient when
steps don't depend on each other's outputs.

**Plan-and-Execute**: Traditional sequential execution. Each step's output is
available to subsequent steps. Supports replanning when steps fail.

**LLMCompiler**: Represents a plan as a Directed Acyclic Graph (DAG) where nodes
are function calls and edges represent dependencies. Topological sort determines
execution order. Enables maximum parallelism.

**Graph of Thought**: Explores the solution space by expanding thought nodes,
evaluating their quality via LLM scoring, and finding the highest-scoring path
through the graph.

**Reflection**: Runs critique on the agent's own output, identifies improvements,
and produces a refined version. Includes heuristic detection of uncertain outputs.

### Agent Memory

Memory follows a multi-tier architecture inspired by cognitive science and modern
LLM systems:

**Working Memory**: Active conversation window. Holds recent messages with decay
mechanism that reduces importance of older items.

**Short-Term Memory**: Session-level buffer with TTL expiration. Automatically
compresses into summary memory when approaching capacity.

**Long-Term Memory**: Vector store using a simple hash-based embedding function
and cosine similarity for semantic retrieval. Keys allow direct lookup.

**Summary Memory**: Compressed representations of conversations. Periodically
generated from short-term memory overflow. Can merge multiple summaries.

**Episodic Memory**: Stores complete interaction episodes with events, outcomes,
and success ratings. Supports similarity-based episode retrieval.

**Reflection Memory**: High-level insights extracted from interaction patterns.
Generated periodically to identify recurring themes and improvements.

Retrieval scoring uses a composite formula weighting recency, importance, and
relevance. Results are sorted by composite score.

### Multi-Agent System

The multi-agent framework supports several collaboration patterns:

**Agent Roles**: Specialized agents (Planner, Executor, Reviewer, Critic, Supervisor)
with distinct system prompts and personae.

**Message Passing**: Agent-to-agent and broadcast messaging with typed messages
(TASK, RESULT, QUERY, RESPONSE, CRITIQUE, DELEGATION, BROADCAST).

**Round-Robin**: Agents take turns contributing to a shared task.

**Debate**: Multiple agents exchange arguments on a topic, with round limits
and conclusion synthesis.

**Supervisor**: A supervisor agent delegates subtasks to specialized sub-agents,
collects results, and makes final decisions.

**Group Chat**: Multiple agents and a human user share a conversation space.
Messages are logged with sender attribution.

**Tool Delegation**: Agents can delegate tool calls to other agents who have
the required tools in their registry.

**Conversation Programming**: AutoGen-inspired callback-based conversation flow
control with hooks for message, turn, and completion events.

## Memory Management

All structures use heap allocation with explicit create/destroy functions.
Internal buffers use fixed-size char arrays with size constants defined in
headers (e.g., `REACT_AGENT_MAX_OBS_LEN`). This avoids dynamic resizing overhead
while providing predictable memory usage.

## Thread Safety

The current implementation is single-threaded. The parallel tool execution
(`tool_call_execute_parallel`) runs sequentially in this version. For multi-threaded
environments, add mutex locks around registry and memory operations.

## Extending

### Adding a Custom Tool

```c
tool_def_t my_tool = {
    .name = "my_tool",
    .description = "Does something useful",
    .type = TOOL_TYPE_CUSTOM,
    .param_count = 1,
    .params = {{"input", TOOL_PARAM_STRING, true, "The input value", ""}}
};

static char* my_tool_func(const char *args_json, void *user_data) {
    char *result = malloc(1024);
    snprintf(result, 1024, "Processed: %s", args_json);
    return result;
}

tool_registry_add(registry, &my_tool, my_tool_func, NULL);
```

### Adding a Custom Agent Role

```c
ma_agent_def_t *custom = ma_agent_create(MA_ROLE_CUSTOM, "MyAgent",
    "You are a specialized agent for X. Always verify Y before Z.");
ma_agent_set_persona(custom, "Expert in domain X with 10 years experience.");
multi_agent_register(ma, custom);
```

## Constants Reference

| Module | Constant | Value | Purpose |
|--------|----------|-------|---------|
| react_agent | MAX_ITERATIONS | 10 | Default ReAct loop limit |
| react_agent | MAX_THOUGHT_LEN | 4096 | Thought text buffer |
| react_agent | MAX_OBS_LEN | 8192 | Observation/answer buffer |
| tool_use | MAX_REGISTRY | 128 | Max registered tools |
| tool_use | MAX_PARALLEL_CALLS | 8 | Max parallel tool calls |
| planning | MAX_STEPS | 32 | Max plan steps |
| planning | MAX_DAG_NODES | 64 | Max DAG nodes |
| planning | MAX_GRAPH_NODES | 64 | Max GoT nodes |
| memory | MAX_WORKING_ITEMS | 256 | Working memory capacity |
| memory | MAX_SHORT_TERM_ITEMS | 1024 | Short-term memory capacity |
| memory | MAX_VECTOR_ITEMS | 10000 | Vector store capacity |
| memory | VEC_DIM | 384 | Embedding vector dimensions |
| multi_agent | MAX_AGENTS | 16 | Max registered agents |
| multi_agent | MAX_DEBATE_ROUNDS | 8 | Default debate rounds |
| multi_agent | MAX_SUBAGENTS | 8 | Max supervisor sub-agents |

## Limitations

- Embedding function uses simple hash-based vectors (not real semantic embeddings)
- No HTTP/network implementation included (LLM calls are callback-injected)
- Single-threaded execution
- No persistent storage (all memory is in-process)
- Built-in tools are simulated/stub implementations
- JSON parsing is basic string-based extraction (not a full parser)
