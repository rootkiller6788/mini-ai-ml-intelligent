# mini-agent-runtime API Reference

## react_agent.h — ReAct Agent

The ReAct agent implements the Thought → Action → Observation loop pattern.

### Types

| Type | Description |
|------|-------------|
| `react_agent_t` | Opaque agent instance |
| `react_state_t` | State enum: IDLE, THINKING, ACTING, OBSERVING, FINISHED, ERROR |
| `react_step_t` | Single step: thought, action, action_input, observation, final_answer |
| `react_message_t` | Conversation message: role + content |
| `react_call_format_t` | JSON or TEXT format for tool calls |
| `react_llm_callback_t` | `char*(*)(const char *prompt, void *user_data)` |

### Core API

```c
react_agent_t* react_agent_create(const char *system_prompt);
void react_agent_destroy(react_agent_t *agent);

void react_agent_set_max_iterations(react_agent_t *agent, int max_iter);
void react_agent_set_call_format(react_agent_t *agent, react_call_format_t format);
void react_agent_set_tools(react_agent_t *agent, void *tool_registry);
void react_agent_set_llm_callback(react_agent_t *agent, react_llm_callback_t cb, void *user_data);

const char* react_agent_run(react_agent_t *agent, const char *user_query);
```

### Helper API

```c
react_step_t react_agent_parse_llm_output(const char *raw_output);
bool react_agent_execute_action(react_agent_t *agent, react_step_t *step);
bool react_agent_has_final_answer(const react_step_t *step);
const char* react_agent_extract_thought(const char *raw, char *buf, size_t buf_size);
const char* react_agent_extract_action(const char *raw, char *buf, size_t buf_size);
bool react_agent_validate_step(const react_step_t *step);
```

---

## tool_use.h — Tool/Function Calling

Extensible tool registry with built-in tools and JSON Schema validation.

### Types

| Type | Description |
|------|-------------|
| `tool_registry_t` | Tool registry holding all registered tools |
| `tool_def_t` | Tool definition: name, description, parameters, schema |
| `tool_call_t` | Result of a tool execution call |
| `tool_batch_t` | Batch of parallel tool calls |
| `tool_func_t` | `char*(*)(const char *args_json, void *user_data)` |

### Core API

```c
tool_registry_t* tool_registry_create(void);
void tool_registry_destroy(tool_registry_t *reg);

bool tool_registry_add(tool_registry_t *reg, const tool_def_t *def, tool_func_t func, void *user_data);
bool tool_registry_add_builtin(tool_registry_t *reg, tool_type_t type);
tool_entry_t* tool_registry_find(tool_registry_t *reg, const char *name);
```

### Built-in Tools

| Type | Tool | Parameters |
|------|------|------------|
| `TOOL_TYPE_CALCULATOR` | `calculator` | `expression` (string, required) |
| `TOOL_TYPE_WEB_SEARCH` | `web_search` | `query` (string, required) |
| `TOOL_TYPE_FILE_READER` | `file_reader` | `path` (string), `encoding` (string, optional) |
| `TOOL_TYPE_CODE_EXECUTOR` | `code_executor` | `code` (string), `language` (string, optional) |

### Execution API

```c
tool_call_t tool_call_execute(tool_registry_t *reg, const char *name, const char *args_json);
tool_batch_t tool_call_execute_parallel(tool_registry_t *reg, const tool_call_t *calls, int count);
tool_call_t tool_call_parse_json(const char *json_str);
tool_call_t tool_call_parse_text(const char *text_str);
bool tool_validate_args(const tool_def_t *def, const char *args_json, char *error_buf, size_t buf_size);
```

---

## planning_system.h — Planning Strategies

Supports ReWOO, Plan-and-Execute, LLMCompiler, Graph of Thought, and Reflection.

### Strategies

| Enum | Strategy | Description |
|------|----------|-------------|
| `PLANNING_STRATEGY_REWOO` | ReWOO | Plan all steps → execute all → aggregate results |
| `PLANNING_STRATEGY_PLAN_EXECUTE` | Plan & Execute | Plan → step-by-step execution → replan on failure |
| `PLANNING_STRATEGY_LLM_COMPILER` | LLMCompiler | Parallel DAG of function calls |
| `PLANNING_STRATEGY_GRAPH_OF_THOUGHT` | GoT | Expand, evaluate, find best path through thought graph |
| `PLANNING_STRATEGY_REFLECTION` | Reflection | Critique own output → refine |

### Core API

```c
planning_system_t* planning_system_create(planning_strategy_t strategy);
void planning_system_destroy(planning_system_t *ps);

plan_t* planning_system_create_plan(planning_system_t *ps, const char *task);
const char* planning_system_execute_plan(planning_system_t *ps, plan_t *plan, char *buf, size_t buf_size);

planning_strategy_t planning_choose_strategy(const char *task);
const char* planning_strategy_name(planning_strategy_t strategy);
```

### Strategy-Specific API

**ReWOO:**
```c
plan_t* planning_rew00(planning_system_t *ps, const char *task);
const char* planning_rew00_execute_all(planning_system_t *ps, plan_t *plan, char *buf, size_t buf_size);
const char* planning_rew00_aggregate(planning_system_t *ps, plan_t *plan, char *buf, size_t buf_size);
```

**LLMCompiler:**
```c
compiler_dag_t* planning_llmcompiler_create_dag(planning_system_t *ps, const char *task);
int* planning_llmcompiler_topological_order(const compiler_dag_t *dag, int *out_count);
```

**Graph of Thought:**
```c
thought_graph_t* planning_got_create_graph(planning_system_t *ps, const char *task);
thought_graph_t* planning_got_expand(planning_system_t *ps, thought_graph_t *graph, int node_id);
const char* planning_got_best_path(planning_system_t *ps, thought_graph_t *graph, char *buf, size_t buf_size);
```

**Reflection:**
```c
reflection_t planning_reflect(planning_system_t *ps, const char *output, const char *context);
const char* planning_refine(planning_system_t *ps, const char *original, const reflection_t *reflection, char *buf, size_t buf_size);
bool planning_should_rethink(const char *output);
```

---

## agent_memory.h — Memory System

Multi-tier memory architecture: working, short-term, long-term (vector), summary, episodic, reflection.

### Memory Types

| Type | Capacity | Description |
|------|----------|-------------|
| Working Memory | 256 items | Active conversation history, recent observations |
| Short-Term Memory | 1024 items | Session buffer with TTL management |
| Long-Term Memory | 10000 vectors | Semantic memory with cosine similarity search |
| Summary Memory | 64 items | Compressed conversation summaries |
| Episodic Memory | 512 episodes | Interaction episodes with events and outcomes |
| Reflection Memory | 128 items | High-level insights extracted from interactions |

### Core API

```c
agent_memory_t* agent_memory_create(void);
void agent_memory_destroy(agent_memory_t *mem);
void agent_memory_clear(agent_memory_t *mem, memory_type_t type);

bool working_memory_add(working_memory_t *wm, memory_role_t role, const char *content);
bool short_term_memory_add(short_term_memory_t *stm, memory_role_t role, const char *content);
bool long_term_memory_store(long_term_memory_t *ltm, const char *content, const char *key);

memory_retrieval_t* memory_retrieve(agent_memory_t *mem, const char *query, int limit, int *out_count);
void memory_retrieval_sort(memory_retrieval_t *results, int count);
```

### Retrieval Scoring

Results are scored by composite formula: `0.4 * recency + 0.3 * importance + 0.3 * relevance`.

---

## multi_agent.h — Multi-Agent System

Agent roles, debate, round-robin, supervisor delegation, group chat, tool delegation.

### Agent Roles

| Role | Description |
|------|-------------|
| `MA_ROLE_PLANNER` | Breaks tasks into executable steps |
| `MA_ROLE_EXECUTOR` | Executes tasks precisely |
| `MA_ROLE_REVIEWER` | Checks work for correctness |
| `MA_ROLE_CRITIC` | Provides constructive feedback |
| `MA_ROLE_SUPERVISOR` | Delegates tasks to sub-agents |

### Core API

```c
multi_agent_t* multi_agent_create(void);
void multi_agent_destroy(multi_agent_t *ma);

int multi_agent_register(multi_agent_t *ma, ma_agent_def_t *agent);
ma_agent_def_t* ma_agent_create(ma_role_t role, const char *name, const char *system_prompt);

bool ma_send_message(multi_agent_t *ma, const ma_message_t *msg);
bool ma_broadcast(multi_agent_t *ma, const char *content, ma_message_type_t type);
```

### Collaboration Patterns

**Round-Robin:**
```c
void ma_round_robin_start(multi_agent_t *ma, const char *task);
const char* ma_round_robin_step(multi_agent_t *ma, char *buf, size_t buf_size);
```

**Debate:**
```c
ma_debate_t* ma_debate_create(multi_agent_t *ma, const char *topic, int max_rounds);
const char* ma_debate_conclude(ma_debate_t *debate, char *buf, size_t buf_size);
```

**Supervisor:**
```c
ma_supervisor_t* ma_supervisor_create(multi_agent_t *ma, ma_agent_def_t *supervisor);
bool ma_supervisor_delegate(ma_supervisor_t *sup, const char *task, int subagent_ids[], int count);
```

**Group Chat:**
```c
ma_group_chat_t* ma_group_chat_create(multi_agent_t *ma, const char *topic);
bool ma_group_chat_send(ma_group_chat_t *gc, int sender_id, const char *content, ma_message_type_t type);
const char* ma_group_chat_get_transcript(const ma_group_chat_t *gc, char *buf, size_t buf_size);
```

**Tool Delegation:**
```c
ma_tool_delegation_t* ma_tool_delegate(multi_agent_t *ma, int from_id, int to_id, const char *tool_name, const char *args);
const char* ma_tool_collect_result(ma_tool_delegation_t *delegation);
```
