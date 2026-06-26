#include "react_agent.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

struct react_agent_s {
    char system_prompt[REACT_AGENT_MAX_OBS_LEN];
    react_message_t history[REACT_AGENT_MAX_MSG_HISTORY];
    int history_count;
    react_state_t state;
    int iteration;
    int max_iterations;
    react_call_format_t call_format;
    react_step_t last_step;
    void *tool_registry;
    react_llm_callback_t llm_callback;
    void *user_data;
};

static const char *react_format_text =
    "You are an agent that follows the ReAct framework.\n"
    "Respond in the following format:\n\n"
    "Thought: <your reasoning about what to do next>\n"
    "Action: <the tool name to call, or 'Final Answer'>\n"
    "Action Input: <the input for the action>\n\n"
    "Available tools:\n%s\n\n"
    "Always end with 'Final Answer' when you have the answer.\n";

static const char *react_format_json =
    "You are an agent that calls functions using JSON.\n"
    "Respond with a JSON function call object:\n"
    "{\"function\": \"tool_name\", \"arguments\": {...}}\n"
    "Or when finished:\n"
    "{\"function\": \"final_answer\", \"arguments\": {\"answer\": \"...\"}}\n\n"
    "Available functions:\n%s\n";

react_agent_t* react_agent_create(const char *system_prompt) {
    react_agent_t *agent = (react_agent_t*)calloc(1, sizeof(react_agent_t));
    if (!agent) return NULL;
    if (system_prompt) {
        strncpy(agent->system_prompt, system_prompt, REACT_AGENT_MAX_OBS_LEN - 1);
    }
    agent->state = REACT_STATE_IDLE;
    agent->max_iterations = REACT_AGENT_MAX_ITERATIONS;
    agent->call_format = REACT_CALL_FORMAT_TEXT;
    agent->history_count = 0;
    agent->iteration = 0;
    return agent;
}

void react_agent_destroy(react_agent_t *agent) {
    if (agent) free(agent);
}

void react_agent_set_max_iterations(react_agent_t *agent, int max_iter) {
    if (agent) agent->max_iterations = max_iter;
}

void react_agent_set_call_format(react_agent_t *agent, react_call_format_t format) {
    if (agent) agent->call_format = format;
}

void react_agent_set_tools(react_agent_t *agent, void *tool_registry) {
    if (agent) agent->tool_registry = tool_registry;
}

void react_agent_set_llm_callback(react_agent_t *agent, react_llm_callback_t cb, void *user_data) {
    if (!agent) return;
    agent->llm_callback = cb;
    agent->user_data = user_data;
}

void react_agent_set_user_data(react_agent_t *agent, void *user_data) {
    if (agent) agent->user_data = user_data;
}

bool react_agent_add_message(react_agent_t *agent, const char *role, const char *content) {
    if (!agent || agent->history_count >= REACT_AGENT_MAX_MSG_HISTORY) return false;
    react_message_t *msg = &agent->history[agent->history_count];
    strncpy(msg->role, role, sizeof(msg->role) - 1);
    strncpy(msg->content, content, REACT_AGENT_MAX_OBS_LEN - 1);
    agent->history_count++;
    return true;
}

void react_agent_clear_history(react_agent_t *agent) {
    if (agent) agent->history_count = 0;
}

static const char* call_llm(react_agent_t *agent, const char *prompt) {
    if (!agent->llm_callback) return NULL;
    return agent->llm_callback(prompt, agent->user_data);
}

const char* react_agent_build_prompt(react_agent_t *agent, const char *user_query, char *buf, size_t buf_size) {
    if (!agent || !buf) return NULL;
    char tools_desc[8192] = {0};
    if (agent->tool_registry) {
        extern const char* tool_format_for_llm(const void *reg, char *buf, size_t buf_size);
        tool_format_for_llm(agent->tool_registry, tools_desc, sizeof(tools_desc));
    }
    const char *template_str = (agent->call_format == REACT_CALL_FORMAT_JSON) ? react_format_json : react_format_text;
    int offset = snprintf(buf, buf_size, "%s\n", agent->system_prompt);
    offset += snprintf(buf + offset, buf_size - offset, template_str, tools_desc);
    for (int i = 0; i < agent->history_count; i++) {
        offset += snprintf(buf + offset, buf_size - offset, "%s: %s\n",
                          agent->history[i].role, agent->history[i].content);
    }
    offset += snprintf(buf + offset, buf_size - offset, "User: %s\n", user_query);
    return buf;
}

const char* react_agent_run(react_agent_t *agent, const char *user_query) {
    if (!agent) return NULL;
    agent->state = REACT_STATE_THINKING;
    agent->iteration = 0;
    agent->last_step.is_final = false;
    react_agent_add_message(agent, "User", user_query);

    while (agent->iteration < agent->max_iterations) {
        char prompt[32768];
        react_agent_build_prompt(agent, user_query, prompt, sizeof(prompt));
        if (agent->iteration > 0) {
            const react_step_t *last = &agent->last_step;
            char feedback[8192];
            snprintf(feedback, sizeof(feedback),
                     "Observation: %s\n(If you have the final answer, respond with Final Answer.)",
                     last->observation);
            react_agent_add_message(agent, "System", feedback);
        }
        const char *raw = call_llm(agent, prompt);
        if (!raw) {
            agent->state = REACT_STATE_ERROR;
            return NULL;
        }
        react_step_t step = react_agent_parse_llm_output(raw);
        step = agent->last_step;
        step = step;
        agent->last_step = react_agent_parse_llm_output(raw);
        agent->state = REACT_STATE_ACTING;
        if (react_agent_has_final_answer(&agent->last_step)) {
            agent->state = REACT_STATE_FINISHED;
            react_agent_add_message(agent, "Assistant", agent->last_step.final_answer);
            return agent->last_step.final_answer;
        }
        react_agent_execute_action(agent, &agent->last_step);
        agent->state = REACT_STATE_OBSERVING;
        agent->iteration++;
    }
    agent->state = REACT_STATE_FINISHED;
    strncpy(agent->last_step.final_answer, "Maximum iterations reached.", REACT_AGENT_MAX_OBS_LEN - 1);
    return agent->last_step.final_answer;
}

react_step_t react_agent_parse_llm_output(const char *raw_output) {
    react_step_t step;
    memset(&step, 0, sizeof(step));
    if (!raw_output) return step;

    react_agent_extract_thought(raw_output, step.thought, sizeof(step.thought));
    react_agent_extract_action(raw_output, step.action, sizeof(step.action));
    react_agent_extract_action_input(raw_output, step.action_input, sizeof(step.action_input));

    if (strstr(raw_output, "Final Answer") || strstr(raw_output, "final_answer")) {
        step.is_final = true;
        const char *fa = strstr(raw_output, "Final Answer:");
        if (!fa) fa = strstr(raw_output, "final_answer");
        if (fa) {
            const char *start = strchr(fa, ':');
            if (!start) start = strstr(fa, "answer");
            if (start) {
                start = strchr(start, '"');
                if (start) {
                    start++;
                    const char *end = strchr(start, '"');
                    if (end) {
                        size_t len = (size_t)(end - start);
                        if (len >= REACT_AGENT_MAX_OBS_LEN) len = REACT_AGENT_MAX_OBS_LEN - 1;
                        strncpy(step.final_answer, start, len);
                    } else {
                        strncpy(step.final_answer, start, REACT_AGENT_MAX_OBS_LEN - 1);
                    }
                } else {
                    strncpy(step.final_answer, start + 1, REACT_AGENT_MAX_OBS_LEN - 1);
                }
            }
        }
    }
    return step;
}

bool react_agent_execute_action(react_agent_t *agent, react_step_t *step) {
    if (!agent || !step || step->is_final) return false;
    snprintf(step->observation, sizeof(step->observation),
             "Tool '%s' called with input '%s'. Result would appear here.",
             step->action, step->action_input);
    return true;
}

react_state_t react_agent_get_state(const react_agent_t *agent) {
    return agent ? agent->state : REACT_STATE_ERROR;
}

int react_agent_get_iteration(const react_agent_t *agent) {
    return agent ? agent->iteration : -1;
}

const react_step_t* react_agent_get_last_step(const react_agent_t *agent) {
    return agent ? &agent->last_step : NULL;
}

const react_message_t* react_agent_get_history(const react_agent_t *agent, int *count) {
    if (!agent) { if (count) *count = 0; return NULL; }
    if (count) *count = agent->history_count;
    return agent->history;
}

bool react_agent_has_final_answer(const react_step_t *step) {
    return step && step->is_final;
}

const char* react_agent_extract_thought(const char *raw, char *buf, size_t buf_size) {
    if (!raw || !buf) return NULL;
    const char *start = strstr(raw, "Thought:");
    if (!start) { buf[0] = '\0'; return buf; }
    start += 8;
    while (*start == ' ' || *start == '\t') start++;
    const char *end = strstr(start, "Action:");
    if (!end) end = strstr(start, "\n\n");
    if (!end) end = raw + strlen(raw);
    size_t len = (size_t)(end - start);
    if (len >= buf_size) len = buf_size - 1;
    strncpy(buf, start, len);
    buf[len] = '\0';
    return buf;
}

const char* react_agent_extract_action(const char *raw, char *buf, size_t buf_size) {
    if (!raw || !buf) return NULL;
    const char *start = strstr(raw, "Action:");
    if (!start) { buf[0] = '\0'; return buf; }
    start += 7;
    while (*start == ' ' || *start == '\t') start++;
    const char *end = strchr(start, '\n');
    if (!end) end = raw + strlen(raw);
    size_t len = (size_t)(end - start);
    if (len >= buf_size) len = buf_size - 1;
    strncpy(buf, start, len);
    buf[len] = '\0';
    return buf;
}

const char* react_agent_extract_action_input(const char *raw, char *buf, size_t buf_size) {
    if (!raw || !buf) return NULL;
    const char *start = strstr(raw, "Action Input:");
    if (!start) { buf[0] = '\0'; return buf; }
    start += 13;
    while (*start == ' ' || *start == '\t') start++;
    const char *end = strstr(start, "\nObservation:");
    if (!end) end = strstr(start, "\nThought:");
    if (!end) end = raw + strlen(raw);
    size_t len = (size_t)(end - start);
    if (len >= buf_size) len = buf_size - 1;
    strncpy(buf, start, len);
    buf[len] = '\0';
    return buf;
}

bool react_agent_validate_step(const react_step_t *step) {
    if (!step) return false;
    if (step->is_final) return step->final_answer[0] != '\0';
    return step->action[0] != '\0' && step->thought[0] != '\0';
}
