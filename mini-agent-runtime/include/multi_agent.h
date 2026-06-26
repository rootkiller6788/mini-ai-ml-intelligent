#ifndef MULTI_AGENT_H
#define MULTI_AGENT_H

#include <stddef.h>
#include <stdbool.h>

#define MA_MAX_AGENTS           16
#define MA_MAX_ROLES            8
#define MA_MAX_MESSAGE_LEN      8192
#define MA_MAX_GROUP_CHAT_MSGS  256
#define MA_MAX_DEBATE_ROUNDS    8
#define MA_MAX_SUBAGENTS        8
#define MA_MAX_TOOLS_PER_AGENT  32
#define MA_MAX_TASK_LEN         4096

typedef enum {
    MA_ROLE_PLANNER,
    MA_ROLE_EXECUTOR,
    MA_ROLE_REVIEWER,
    MA_ROLE_CRITIC,
    MA_ROLE_SUPERVISOR,
    MA_ROLE_DELEGATE,
    MA_ROLE_CUSTOM
} ma_role_t;

typedef enum {
    MA_MSG_TYPE_TASK,
    MA_MSG_TYPE_RESULT,
    MA_MSG_TYPE_QUERY,
    MA_MSG_TYPE_RESPONSE,
    MA_MSG_TYPE_CRITIQUE,
    MA_MSG_TYPE_DELEGATION,
    MA_MSG_TYPE_BROADCAST
} ma_message_type_t;

typedef struct {
    int sender_id;
    int recipient_id;
    ma_message_type_t type;
    char content[MA_MAX_MESSAGE_LEN];
    char metadata[1024];
    int timestamp;
    bool is_broadcast;
} ma_message_t;

typedef struct {
    ma_role_t role;
    char name[64];
    char system_prompt[2048];
    char persona[1024];
    void *react_agent;
    void *tool_registry;
    bool active;
    int agent_id;
} ma_agent_def_t;

typedef struct {
    ma_agent_def_t *agents[MA_MAX_AGENTS];
    int agent_count;
    ma_message_t *message_queue[MA_MAX_GROUP_CHAT_MSGS * 2];
    int queue_head;
    int queue_tail;
    int turn_index;
    char shared_context[MA_MAX_MESSAGE_LEN];
} multi_agent_t;

typedef struct {
    ma_agent_def_t *planner;
    ma_agent_def_t *executor;
    ma_agent_def_t *reviewer;
    char *task;
    char *result;
    bool completed;
} ma_workflow_t;

typedef struct {
    ma_agent_def_t *agents[MA_MAX_AGENTS];
    int agent_count;
    char topic[1024];
    int max_rounds;
    int current_round;
    char summary[MA_MAX_MESSAGE_LEN];
} ma_debate_t;

typedef struct {
    ma_agent_def_t *supervisor;
    ma_agent_def_t *subagents[MA_MAX_SUBAGENTS];
    int subagent_count;
    char *task;
} ma_supervisor_t;

typedef struct {
    ma_agent_def_t *agents[MA_MAX_AGENTS];
    int agent_count;
    char *human_user_name;
    ma_message_t messages[MA_MAX_GROUP_CHAT_MSGS];
    int message_count;
    char shared_topic[1024];
} ma_group_chat_t;

typedef struct {
    char tool_name[64];
    int from_agent_id;
    int to_agent_id;
    char args[2048];
    char result[MA_MAX_MESSAGE_LEN];
    bool completed;
} ma_tool_delegation_t;

typedef char* (*ma_llm_callback_t)(const char *system_prompt, const char *user_prompt, void *user_data);
typedef char* (*ma_agent_llm_t)(int agent_id, const char *prompt, void *user_data);

multi_agent_t* multi_agent_create(void);
void multi_agent_destroy(multi_agent_t *ma);

int multi_agent_register(multi_agent_t *ma, ma_agent_def_t *agent);
bool multi_agent_unregister(multi_agent_t *ma, int agent_id);
ma_agent_def_t* multi_agent_get(multi_agent_t *ma, int agent_id);

ma_agent_def_t* ma_agent_create(ma_role_t role, const char *name, const char *system_prompt);
void ma_agent_destroy(ma_agent_def_t *agent);
void ma_agent_set_persona(ma_agent_def_t *agent, const char *persona);
void ma_agent_set_llm(ma_agent_def_t *agent, ma_llm_callback_t cb, void *user_data);
void ma_agent_assign_tools(ma_agent_def_t *agent, void *tool_registry);

bool ma_send_message(multi_agent_t *ma, const ma_message_t *msg);
bool ma_send_to_role(multi_agent_t *ma, ma_role_t role, const char *content, ma_message_type_t type);
bool ma_broadcast(multi_agent_t *ma, const char *content, ma_message_type_t type);
ma_message_t* ma_receive_message(multi_agent_t *ma, int agent_id);
int ma_poll_messages(multi_agent_t *ma, int agent_id, ma_message_t *buf, int max_msgs);

void ma_round_robin_start(multi_agent_t *ma, const char *task);
const char* ma_round_robin_step(multi_agent_t *ma, char *buf, size_t buf_size);
bool ma_round_robin_is_done(const multi_agent_t *ma);

ma_debate_t* ma_debate_create(multi_agent_t *ma, const char *topic, int max_rounds);
void ma_debate_destroy(ma_debate_t *debate);
bool ma_debate_add_participant(ma_debate_t *debate, ma_agent_def_t *agent);
const char* ma_debate_run_round(ma_debate_t *debate, char *buf, size_t buf_size);
const char* ma_debate_conclude(ma_debate_t *debate, char *buf, size_t buf_size);

ma_supervisor_t* ma_supervisor_create(multi_agent_t *ma, ma_agent_def_t *supervisor);
void ma_supervisor_destroy(ma_supervisor_t *supervisor);
bool ma_supervisor_delegate(ma_supervisor_t *sup, const char *task, int subagent_ids[], int count);
const char* ma_supervisor_collect(ma_supervisor_t *sup, char *buf, size_t buf_size);
const char* ma_supervisor_review_and_decide(ma_supervisor_t *sup, const char *sub_results, char *buf, size_t buf_size);

ma_group_chat_t* ma_group_chat_create(multi_agent_t *ma, const char *topic);
void ma_group_chat_destroy(ma_group_chat_t *gc);
bool ma_group_chat_add_member(ma_group_chat_t *gc, ma_agent_def_t *agent);
bool ma_group_chat_send(ma_group_chat_t *gc, int sender_id, const char *content, ma_message_type_t type);
const char* ma_group_chat_get_transcript(const ma_group_chat_t *gc, char *buf, size_t buf_size);
int ma_group_chat_speaker_select(const ma_group_chat_t *gc);

ma_tool_delegation_t* ma_tool_delegate(multi_agent_t *ma, int from_id, int to_id, const char *tool_name, const char *args);
const char* ma_tool_collect_result(ma_tool_delegation_t *delegation);
void ma_tool_delegation_destroy(ma_tool_delegation_t *delegation);

typedef void (*ma_conversation_step_t)(multi_agent_t *ma, const char *trigger, void *user_data);
typedef struct {
    ma_conversation_step_t on_message;
    ma_conversation_step_t on_turn;
    ma_conversation_step_t on_complete;
    void *user_data;
} ma_conversation_program_t;

void ma_conversation_program_run(multi_agent_t *ma, ma_conversation_program_t *prog, const char *initial_task);

ma_agent_def_t* ma_create_planner(const char *name);
ma_agent_def_t* ma_create_executor(const char *name);
ma_agent_def_t* ma_create_reviewer(const char *name);
ma_agent_def_t* ma_create_critic(const char *name);

const char* ma_agent_generate_response(const ma_agent_def_t *agent, const char *prompt, char *buf, size_t buf_size);
bool ma_verify_output(const ma_agent_def_t *agent, const char *output);
void ma_reset_conversation(multi_agent_t *ma);

#endif
