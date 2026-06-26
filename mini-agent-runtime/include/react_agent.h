#ifndef REACT_AGENT_H
#define REACT_AGENT_H

#include <stddef.h>
#include <stdbool.h>

#define REACT_AGENT_MAX_ITERATIONS   10
#define REACT_AGENT_MAX_THOUGHT_LEN  4096
#define REACT_AGENT_MAX_ACTION_LEN   2048
#define REACT_AGENT_MAX_OBS_LEN      8192
#define REACT_AGENT_MAX_TOOLS        64
#define REACT_AGENT_MAX_MSG_HISTORY  128

typedef enum {
    REACT_STATE_IDLE,
    REACT_STATE_THINKING,
    REACT_STATE_ACTING,
    REACT_STATE_OBSERVING,
    REACT_STATE_FINISHED,
    REACT_STATE_ERROR
} react_state_t;

typedef enum {
    REACT_CALL_FORMAT_JSON,
    REACT_CALL_FORMAT_TEXT
} react_call_format_t;

typedef struct {
    char role[32];
    char content[REACT_AGENT_MAX_OBS_LEN];
} react_message_t;

typedef struct {
    char thought[REACT_AGENT_MAX_THOUGHT_LEN];
    char action[REACT_AGENT_MAX_ACTION_LEN];
    char action_input[REACT_AGENT_MAX_ACTION_LEN];
    char observation[REACT_AGENT_MAX_OBS_LEN];
    char final_answer[REACT_AGENT_MAX_OBS_LEN];
    bool is_final;
} react_step_t;

typedef char* (*react_llm_callback_t)(const char *prompt, void *user_data);

typedef struct react_agent_s react_agent_t;

react_agent_t* react_agent_create(const char *system_prompt);
void react_agent_destroy(react_agent_t *agent);

void react_agent_set_max_iterations(react_agent_t *agent, int max_iter);
void react_agent_set_call_format(react_agent_t *agent, react_call_format_t format);
void react_agent_set_tools(react_agent_t *agent, void *tool_registry);
void react_agent_set_llm_callback(react_agent_t *agent, react_llm_callback_t cb, void *user_data);
void react_agent_set_user_data(react_agent_t *agent, void *user_data);

bool react_agent_add_message(react_agent_t *agent, const char *role, const char *content);
void react_agent_clear_history(react_agent_t *agent);

const char* react_agent_run(react_agent_t *agent, const char *user_query);

react_step_t react_agent_parse_llm_output(const char *raw_output);
bool react_agent_execute_action(react_agent_t *agent, react_step_t *step);

react_state_t react_agent_get_state(const react_agent_t *agent);
int react_agent_get_iteration(const react_agent_t *agent);
const react_step_t* react_agent_get_last_step(const react_agent_t *agent);
const react_message_t* react_agent_get_history(const react_agent_t *agent, int *count);
const char* react_agent_build_prompt(react_agent_t *agent, const char *user_query, char *buf, size_t buf_size);

bool react_agent_has_final_answer(const react_step_t *step);
const char* react_agent_extract_thought(const char *raw, char *buf, size_t buf_size);
const char* react_agent_extract_action(const char *raw, char *buf, size_t buf_size);
const char* react_agent_extract_action_input(const char *raw, char *buf, size_t buf_size);
bool react_agent_validate_step(const react_step_t *step);

#endif
