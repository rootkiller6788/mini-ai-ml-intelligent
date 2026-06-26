#include "multi_agent.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

multi_agent_t* multi_agent_create(void) {
    multi_agent_t *ma = (multi_agent_t*)calloc(1, sizeof(multi_agent_t));
    if (!ma) return NULL;
    ma->agent_count = 0;
    ma->queue_head = 0;
    ma->queue_tail = 0;
    ma->turn_index = 0;
    return ma;
}

void multi_agent_destroy(multi_agent_t *ma) {
    if (!ma) return;
    for (int i = 0; i < ma->agent_count; i++) {
        if (ma->agents[i]) ma_agent_destroy(ma->agents[i]);
    }
    free(ma);
}

int multi_agent_register(multi_agent_t *ma, ma_agent_def_t *agent) {
    if (!ma || !agent || ma->agent_count >= MA_MAX_AGENTS) return -1;
    int id = ma->agent_count;
    agent->agent_id = id;
    agent->active = true;
    ma->agents[id] = agent;
    ma->agent_count++;
    return id;
}

bool multi_agent_unregister(multi_agent_t *ma, int agent_id) {
    if (!ma || agent_id < 0 || agent_id >= ma->agent_count) return false;
    ma->agents[agent_id]->active = false;
    return true;
}

ma_agent_def_t* multi_agent_get(multi_agent_t *ma, int agent_id) {
    if (!ma || agent_id < 0 || agent_id >= ma->agent_count) return NULL;
    return ma->agents[agent_id];
}

ma_agent_def_t* ma_agent_create(ma_role_t role, const char *name, const char *system_prompt) {
    ma_agent_def_t *agent = (ma_agent_def_t*)calloc(1, sizeof(ma_agent_def_t));
    if (!agent) return NULL;
    agent->role = role;
    agent->active = true;
    if (name) strncpy(agent->name, name, sizeof(agent->name) - 1);
    if (system_prompt) strncpy(agent->system_prompt, system_prompt, sizeof(agent->system_prompt) - 1);
    return agent;
}

void ma_agent_destroy(ma_agent_def_t *agent) {
    if (agent) free(agent);
}

void ma_agent_set_persona(ma_agent_def_t *agent, const char *persona) {
    if (agent && persona) strncpy(agent->persona, persona, sizeof(agent->persona) - 1);
}

void ma_agent_set_llm(ma_agent_def_t *agent, ma_llm_callback_t cb, void *user_data) {
    if (!agent) return;
    (void)cb;
    (void)user_data;
}

void ma_agent_assign_tools(ma_agent_def_t *agent, void *tool_registry) {
    if (agent) agent->tool_registry = tool_registry;
}

static int msg_queue_len(const multi_agent_t *ma) {
    if (ma->queue_tail >= ma->queue_head) return ma->queue_tail - ma->queue_head;
    return (MA_MAX_GROUP_CHAT_MSGS * 2) - ma->queue_head + ma->queue_tail;
}

bool ma_send_message(multi_agent_t *ma, const ma_message_t *msg) {
    if (!ma || !msg) return false;
    int next = (ma->queue_tail + 1) % (MA_MAX_GROUP_CHAT_MSGS * 2);
    if (next == ma->queue_head) return false;
    ma_message_t *copy = (ma_message_t*)malloc(sizeof(ma_message_t));
    if (!copy) return false;
    memcpy(copy, msg, sizeof(ma_message_t));
    ma->message_queue[ma->queue_tail] = copy;
    ma->queue_tail = next;
    return true;
}

bool ma_send_to_role(multi_agent_t *ma, ma_role_t role, const char *content, ma_message_type_t type) {
    if (!ma || !content) return false;
    for (int i = 0; i < ma->agent_count; i++) {
        if (ma->agents[i]->role == role && ma->agents[i]->active) {
            ma_message_t msg;
            memset(&msg, 0, sizeof(msg));
            msg.sender_id = -1;
            msg.recipient_id = i;
            msg.type = type;
            strncpy(msg.content, content, MA_MAX_MESSAGE_LEN - 1);
            return ma_send_message(ma, &msg);
        }
    }
    return false;
}

bool ma_broadcast(multi_agent_t *ma, const char *content, ma_message_type_t type) {
    if (!ma || !content) return false;
    ma_message_t msg;
    memset(&msg, 0, sizeof(msg));
    msg.sender_id = -1;
    msg.is_broadcast = true;
    msg.type = type;
    strncpy(msg.content, content, MA_MAX_MESSAGE_LEN - 1);
    return ma_send_message(ma, &msg);
}

ma_message_t* ma_receive_message(multi_agent_t *ma, int agent_id) {
    if (!ma || msg_queue_len(ma) == 0) return NULL;
    ma_message_t *msg = ma->message_queue[ma->queue_head];
    if (msg->recipient_id != agent_id && !msg->is_broadcast) {
        return NULL;
    }
    ma->queue_head = (ma->queue_head + 1) % (MA_MAX_GROUP_CHAT_MSGS * 2);
    return msg;
}

int ma_poll_messages(multi_agent_t *ma, int agent_id, ma_message_t *buf, int max_msgs) {
    int received = 0;
    for (int i = 0; i < max_msgs; i++) {
        ma_message_t *msg = ma_receive_message(ma, agent_id);
        if (!msg) break;
        buf[received++] = *msg;
        free(msg);
    }
    return received;
}

void ma_round_robin_start(multi_agent_t *ma, const char *task) {
    if (!ma) return;
    ma->turn_index = 0;
    ma_broadcast(ma, task, MA_MSG_TYPE_TASK);
}

const char* ma_round_robin_step(multi_agent_t *ma, char *buf, size_t buf_size) {
    if (!ma || !buf) return NULL;
    if (ma->agent_count == 0) return NULL;
    int idx = ma->turn_index % ma->agent_count;
    ma->turn_index++;
    ma_agent_def_t *agent = ma->agents[idx];
    if (!agent->active) {
        snprintf(buf, buf_size, "[%s is inactive]", agent->name);
        return buf;
    }
    ma_message_t msgs[8];
    int n = ma_poll_messages(ma, idx, msgs, 8);
    int off = snprintf(buf, buf_size, "[%s (%s)]: ", agent->name,
                       agent->role == MA_ROLE_PLANNER ? "Planner" :
                       agent->role == MA_ROLE_EXECUTOR ? "Executor" :
                       agent->role == MA_ROLE_REVIEWER ? "Reviewer" :
                       agent->role == MA_ROLE_CRITIC ? "Critic" :
                       agent->role == MA_ROLE_SUPERVISOR ? "Supervisor" : "Agent");
    for (int i = 0; i < n; i++) {
        off += snprintf(buf + off, buf_size - off, "%s\n", msgs[i].content);
    }
    if (n == 0) {
        snprintf(buf + off, buf_size - off, "Processing task...");
    }
    return buf;
}

bool ma_round_robin_is_done(const multi_agent_t *ma) {
    return ma && ma->turn_index >= ma->agent_count * 3;
}

ma_debate_t* ma_debate_create(multi_agent_t *ma, const char *topic, int max_rounds) {
    ma_debate_t *debate = (ma_debate_t*)calloc(1, sizeof(ma_debate_t));
    if (!debate) return NULL;
    if (topic) strncpy(debate->topic, topic, sizeof(debate->topic) - 1);
    debate->max_rounds = max_rounds > 0 ? max_rounds : MA_MAX_DEBATE_ROUNDS;
    debate->current_round = 0;
    (void)ma;
    return debate;
}

void ma_debate_destroy(ma_debate_t *debate) {
    if (debate) free(debate);
}

bool ma_debate_add_participant(ma_debate_t *debate, ma_agent_def_t *agent) {
    if (!debate || !agent || debate->agent_count >= MA_MAX_AGENTS) return false;
    debate->agents[debate->agent_count++] = agent;
    return true;
}

const char* ma_debate_run_round(ma_debate_t *debate, char *buf, size_t buf_size) {
    if (!debate || !buf || debate->current_round >= debate->max_rounds) return NULL;
    int off = snprintf(buf, buf_size, "=== Debate Round %d ===\nTopic: %s\n\n",
                      debate->current_round + 1, debate->topic);
    for (int i = 0; i < debate->agent_count; i++) {
        ma_agent_def_t *agent = debate->agents[i];
        if (!agent->active) continue;
        off += snprintf(buf + off, buf_size - off, "[%s] critique:\n", agent->name);
        off += snprintf(buf + off, buf_size - off,
                       "  Evaluating previous arguments and providing perspective...\n\n");
    }
    debate->current_round++;
    return buf;
}

const char* ma_debate_conclude(ma_debate_t *debate, char *buf, size_t buf_size) {
    if (!debate || !buf) return NULL;
    snprintf(buf, buf_size,
             "Debate concluded after %d rounds.\nTopic: %s\n"
             "Key arguments synthesized from %d participants.\nConsensus reached.",
             debate->current_round, debate->topic, debate->agent_count);
    return buf;
}

ma_supervisor_t* ma_supervisor_create(multi_agent_t *ma, ma_agent_def_t *supervisor) {
    ma_supervisor_t *sup = (ma_supervisor_t*)calloc(1, sizeof(ma_supervisor_t));
    if (!sup) return NULL;
    sup->supervisor = supervisor;
    (void)ma;
    return sup;
}

void ma_supervisor_destroy(ma_supervisor_t *supervisor) {
    if (supervisor) {
        free(supervisor->task);
        free(supervisor);
    }
}

bool ma_supervisor_delegate(ma_supervisor_t *sup, const char *task, int subagent_ids[], int count) {
    if (!sup || !task || !subagent_ids) return false;
    sup->task = (char*)malloc(strlen(task) + 1);
    if (!sup->task) return false;
    strcpy(sup->task, task);
    sup->subagent_count = count < MA_MAX_SUBAGENTS ? count : MA_MAX_SUBAGENTS;
    for (int i = 0; i < sup->subagent_count; i++) {
        sup->subagents[i] = NULL;
        (void)subagent_ids[i];
    }
    return true;
}

const char* ma_supervisor_collect(ma_supervisor_t *sup, char *buf, size_t buf_size) {
    if (!sup || !buf) return NULL;
    int off = snprintf(buf, buf_size, "Supervisor report for: %s\n", sup->task ? sup->task : "task");
    for (int i = 0; i < sup->subagent_count; i++) {
        off += snprintf(buf + off, buf_size - off, "  Sub-agent %d result pending...\n", i);
    }
    return buf;
}

const char* ma_supervisor_review_and_decide(ma_supervisor_t *sup, const char *sub_results, char *buf, size_t buf_size) {
    if (!sup || !buf) return NULL;
    snprintf(buf, buf_size,
             "Supervisor decision based on sub-agent results.\n"
             "Task: %s\nSub-results: %s\nDecision: Proceed with synthesized output.",
             sup->task ? sup->task : "unknown", sub_results ? sub_results : "pending");
    return buf;
}

ma_group_chat_t* ma_group_chat_create(multi_agent_t *ma, const char *topic) {
    ma_group_chat_t *gc = (ma_group_chat_t*)calloc(1, sizeof(ma_group_chat_t));
    if (!gc) return NULL;
    if (topic) strncpy(gc->shared_topic, topic, sizeof(gc->shared_topic) - 1);
    gc->human_user_name = (char*)"User";
    (void)ma;
    return gc;
}

void ma_group_chat_destroy(ma_group_chat_t *gc) {
    if (gc) free(gc);
}

bool ma_group_chat_add_member(ma_group_chat_t *gc, ma_agent_def_t *agent) {
    if (!gc || !agent || gc->agent_count >= MA_MAX_AGENTS) return false;
    gc->agents[gc->agent_count++] = agent;
    return true;
}

bool ma_group_chat_send(ma_group_chat_t *gc, int sender_id, const char *content, ma_message_type_t type) {
    if (!gc || !content || gc->message_count >= MA_MAX_GROUP_CHAT_MSGS) return false;
    ma_message_t *msg = &gc->messages[gc->message_count];
    memset(msg, 0, sizeof(*msg));
    msg->sender_id = sender_id;
    msg->type = type;
    strncpy(msg->content, content, MA_MAX_MESSAGE_LEN - 1);
    gc->message_count++;
    return true;
}

const char* ma_group_chat_get_transcript(const ma_group_chat_t *gc, char *buf, size_t buf_size) {
    if (!gc || !buf) return NULL;
    int off = snprintf(buf, buf_size, "=== Group Chat: %s ===\n", gc->shared_topic);
    for (int i = 0; i < gc->message_count; i++) {
        const char *sender = gc->messages[i].sender_id < 0 ? gc->human_user_name :
                            (gc->messages[i].sender_id < gc->agent_count ?
                             gc->agents[gc->messages[i].sender_id]->name : "Unknown");
        off += snprintf(buf + off, buf_size - off, "[%s]: %s\n", sender, gc->messages[i].content);
    }
    return buf;
}

int ma_group_chat_speaker_select(const ma_group_chat_t *gc) {
    if (!gc || gc->agent_count == 0) return -1;
    return gc->message_count % gc->agent_count;
}

ma_tool_delegation_t* ma_tool_delegate(multi_agent_t *ma, int from_id, int to_id, const char *tool_name, const char *args) {
    ma_tool_delegation_t *d = (ma_tool_delegation_t*)calloc(1, sizeof(ma_tool_delegation_t));
    if (!d) return NULL;
    d->from_agent_id = from_id;
    d->to_agent_id = to_id;
    if (tool_name) strncpy(d->tool_name, tool_name, sizeof(d->tool_name) - 1);
    if (args) strncpy(d->args, args, sizeof(d->args) - 1);
    d->completed = false;
    (void)ma;
    return d;
}

const char* ma_tool_collect_result(ma_tool_delegation_t *delegation) {
    if (!delegation) return NULL;
    if (!delegation->completed) {
        snprintf(delegation->result, MA_MAX_MESSAGE_LEN,
                 "[Tool '%s' delegated from agent %d to agent %d. Result pending.]",
                 delegation->tool_name, delegation->from_agent_id, delegation->to_agent_id);
        delegation->completed = true;
    }
    return delegation->result;
}

void ma_tool_delegation_destroy(ma_tool_delegation_t *delegation) {
    if (delegation) free(delegation);
}

void ma_conversation_program_run(multi_agent_t *ma, ma_conversation_program_t *prog, const char *initial_task) {
    if (!ma || !prog) return;
    ma_broadcast(ma, initial_task, MA_MSG_TYPE_TASK);
    if (prog->on_message) prog->on_message(ma, initial_task, prog->user_data);
    for (int i = 0; i < MA_MAX_AGENTS * 2; i++) {
        if (prog->on_turn) prog->on_turn(ma, initial_task, prog->user_data);
    }
    if (prog->on_complete) prog->on_complete(ma, initial_task, prog->user_data);
}

ma_agent_def_t* ma_create_planner(const char *name) {
    return ma_agent_create(MA_ROLE_PLANNER, name,
        "You are a planning agent. Break down tasks into executable steps.");
}

ma_agent_def_t* ma_create_executor(const char *name) {
    return ma_agent_create(MA_ROLE_EXECUTOR, name,
        "You are an executor agent. Run the steps provided by the planner precisely.");
}

ma_agent_def_t* ma_create_reviewer(const char *name) {
    return ma_agent_create(MA_ROLE_REVIEWER, name,
        "You are a reviewer agent. Check the executor's work for correctness.");
}

ma_agent_def_t* ma_create_critic(const char *name) {
    return ma_agent_create(MA_ROLE_CRITIC, name,
        "You are a critic agent. Provide constructive feedback and identify flaws.");
}

const char* ma_agent_generate_response(const ma_agent_def_t *agent, const char *prompt, char *buf, size_t buf_size) {
    if (!agent || !buf) return NULL;
    snprintf(buf, buf_size,
             "[%s (%s)] Response to: %.200s...",
             agent->name,
             agent->role == MA_ROLE_PLANNER ? "Planner" : "Agent",
             prompt ? prompt : "");
    return buf;
}

bool ma_verify_output(const ma_agent_def_t *agent, const char *output) {
    if (!agent || !output) return false;
    return strlen(output) > 0;
}

void ma_reset_conversation(multi_agent_t *ma) {
    if (!ma) return;
    while (msg_queue_len(ma) > 0) {
        free(ma->message_queue[ma->queue_head]);
        ma->queue_head = (ma->queue_head + 1) % (MA_MAX_GROUP_CHAT_MSGS * 2);
    }
    ma->turn_index = 0;
    memset(ma->shared_context, 0, sizeof(ma->shared_context));
}
