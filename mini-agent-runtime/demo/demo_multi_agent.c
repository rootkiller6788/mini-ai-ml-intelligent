#include "multi_agent.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void print_header(const char *title) {
    printf("\n");
    printf("==============================================\n");
    printf("  %s\n", title);
    printf("==============================================\n");
}

static void print_agent_info(const ma_agent_def_t *agent) {
    const char *role_name = "Unknown";
    switch (agent->role) {
        case MA_ROLE_PLANNER:    role_name = "Planner"; break;
        case MA_ROLE_EXECUTOR:   role_name = "Executor"; break;
        case MA_ROLE_REVIEWER:   role_name = "Reviewer"; break;
        case MA_ROLE_CRITIC:     role_name = "Critic"; break;
        case MA_ROLE_SUPERVISOR: role_name = "Supervisor"; break;
        case MA_ROLE_DELEGATE:   role_name = "Delegate"; break;
        case MA_ROLE_CUSTOM:     role_name = "Custom"; break;
    }
    printf("  Agent[%d] '%s' (%s)%s\n",
           agent->agent_id, agent->name, role_name,
           agent->active ? " [active]" : " [inactive]");
}

static void print_message(const ma_message_t *msg) {
    const char *type_str = "?";
    switch (msg->type) {
        case MA_MSG_TYPE_TASK:       type_str = "TASK"; break;
        case MA_MSG_TYPE_RESULT:     type_str = "RESULT"; break;
        case MA_MSG_TYPE_QUERY:      type_str = "QUERY"; break;
        case MA_MSG_TYPE_RESPONSE:   type_str = "RESPONSE"; break;
        case MA_MSG_TYPE_CRITIQUE:   type_str = "CRITIQUE"; break;
        case MA_MSG_TYPE_DELEGATION: type_str = "DELEGATE"; break;
        case MA_MSG_TYPE_BROADCAST:  type_str = "BROADCAST"; break;
    }
    printf("  MSG[%d->%d %s%s]: %.80s...\n",
           msg->sender_id, msg->recipient_id, type_str,
           msg->is_broadcast ? " (broadcast)" : "",
           msg->content);
}

int main(void) {
    printf("========== Multi-Agent System Demo ==========\n");

    multi_agent_t *ma = multi_agent_create();
    if (!ma) { printf("Failed to create multi-agent system\n"); return 1; }
    printf("Multi-agent system initialized.\n");

    print_header("1. Agent Registration");

    ma_agent_def_t *planner = ma_create_planner("AlphaPlanner");
    ma_agent_set_persona(planner, "Strategic thinker who breaks down complex tasks into manageable steps.");
    int pid = multi_agent_register(ma, planner);
    printf("Registered AlphaPlanner (id=%d)\n", pid);

    ma_agent_def_t *executor = ma_create_executor("BetaExecutor");
    ma_agent_set_persona(executor, "Efficient worker who executes tasks precisely and reports results.");
    int eid = multi_agent_register(ma, executor);
    printf("Registered BetaExecutor (id=%d)\n", eid);

    ma_agent_def_t *reviewer = ma_create_reviewer("GammaReviewer");
    ma_agent_set_persona(reviewer, "Meticulous reviewer who checks work for correctness and quality.");
    int rid = multi_agent_register(ma, reviewer);
    printf("Registered GammaReviewer (id=%d)\n", rid);

    ma_agent_def_t *critic = ma_create_critic("DeltaCritic");
    ma_agent_set_persona(critic, "Constructive critic who identifies flaws and suggests improvements.");
    int cid = multi_agent_register(ma, critic);
    printf("Registered DeltaCritic (id=%d)\n", cid);

    ma_agent_def_t *supervisor_agent = ma_agent_create(MA_ROLE_SUPERVISOR, "OmegaSupervisor",
        "You oversee a team of agents and delegate tasks based on their specialties.");
    int sid = multi_agent_register(ma, supervisor_agent);
    printf("Registered OmegaSupervisor (id=%d)\n", sid);

    printf("\nAgent roster:\n");
    for (int i = 0; i < ma->agent_count; i++) {
        print_agent_info(ma->agents[i]);
    }

    print_header("2. Message Passing");

    ma_send_to_role(ma, MA_ROLE_PLANNER,
        "Plan the Q4 marketing campaign strategy. Break it into executable tasks.",
        MA_MSG_TYPE_TASK);
    printf("Task sent to Planner role.\n");

    ma_broadcast(ma, "System initialized. All agents online.", MA_MSG_TYPE_BROADCAST);
    printf("Broadcast sent to all agents.\n");

    ma_message_t msg;
    memset(&msg, 0, sizeof(msg));
    msg.sender_id = pid;
    msg.recipient_id = eid;
    msg.type = MA_MSG_TYPE_DELEGATION;
    strcpy(msg.content, "Execute campaign data analysis: collect engagement metrics from Q1-Q3.");
    ma_send_message(ma, &msg);
    printf("Planner delegated task to Executor.\n");

    printf("\nMessage queue state:\n");
    ma_message_t poll_buf[16];
    for (int i = 0; i < ma->agent_count; i++) {
        int n = ma_poll_messages(ma, i, poll_buf, 16);
        if (n > 0) {
            printf("  Agent[%d] %s: %d pending messages\n", i, ma->agents[i]->name, n);
        }
    }

    ma_reset_conversation(ma);

    print_header("3. Round-Robin Collaboration");

    multi_agent_t *ma2 = multi_agent_create();
    ma_agent_def_t *rr_agents[3];
    rr_agents[0] = ma_create_planner("TeamLead");
    rr_agents[1] = ma_create_executor("WorkerBee");
    rr_agents[2] = ma_create_reviewer("QualityCheck");

    for (int i = 0; i < 3; i++) {
        multi_agent_register(ma2, rr_agents[i]);
    }

    const char *task = "Analyze customer feedback data and create an improvement report";
    ma_round_robin_start(ma2, task);
    printf("Round-robin started with task: '%s'\n", task);

    printf("\nRound-robin turns:\n");
    for (int turn = 0; turn < 6 && !ma_round_robin_is_done(ma2); turn++) {
        char step_buf[1024];
        ma_round_robin_step(ma2, step_buf, sizeof(step_buf));
        printf("  Turn %d: %s\n", turn + 1, step_buf);
    }
    printf("Round-robin done: %s\n", ma_round_robin_is_done(ma2) ? "Yes" : "No");

    multi_agent_destroy(ma2);

    print_header("4. Agent Debate");

    ma_debate_t *debate = ma_debate_create(ma, "Should agents prioritize speed or accuracy?", 4);
    if (debate) {
        ma_debate_add_participant(debate, planner);
        ma_debate_add_participant(debate, critic);
        ma_debate_add_participant(debate, reviewer);

        printf("Debate topic: '%s'\n", debate->topic);
        printf("Participants: Planner, Critic, Reviewer (%d total)\n", debate->agent_count);
        printf("Max rounds: %d\n\n", debate->max_rounds);

        for (int r = 0; r < 3; r++) {
            char round_buf[4096];
            const char *round_output = ma_debate_run_round(debate, round_buf, sizeof(round_buf));
            if (round_output) {
                printf("%s\n", round_output);
            } else {
                printf("Round %d: no output (max rounds reached?)\n", r + 1);
                break;
            }
        }

        char conclusion[4096];
        ma_debate_conclude(debate, conclusion, sizeof(conclusion));
        printf("Debate conclusion:\n%s\n", conclusion);

        ma_debate_destroy(debate);
    }

    print_header("5. Supervisor Agent");

    ma_supervisor_t *sup = ma_supervisor_create(ma, supervisor_agent);
    if (sup) {
        int sub_ids[3] = {eid, rid, cid};
        const char *task2 = "Design and validate a new customer onboarding flow";

        bool delegated = ma_supervisor_delegate(sup, task2, sub_ids, 3);
        printf("Supervisor delegated task '%s' to %d sub-agents: %s\n",
               task2, sup->subagent_count, delegated ? "success" : "failed");

        char report[4096];
        ma_supervisor_collect(sup, report, sizeof(report));
        printf("\nSupervisor collection report:\n%s\n", report);

        char decision[4096];
        ma_supervisor_review_and_decide(sup, "Executor: flow implemented. Reviewer: approved. Critic: 2 issues found.",
                                        decision, sizeof(decision));
        printf("Supervisor decision:\n%s\n", decision);

        ma_supervisor_destroy(sup);
    }

    print_header("6. Group Chat");

    ma_group_chat_t *gc = ma_group_chat_create(ma, "Q4 Strategy Planning Session");
    if (gc) {
        ma_group_chat_add_member(gc, planner);
        ma_group_chat_add_member(gc, executor);
        ma_group_chat_add_member(gc, reviewer);

        printf("Group chat created: '%s'\n", gc->shared_topic);
        printf("Members: %d agents + User\n\n", gc->agent_count);

        ma_group_chat_send(gc, -1, "Hi team, let's plan our Q4 strategy.", MA_MSG_TYPE_QUERY);
        ma_group_chat_send(gc, pid, "Proposal: Focus on mobile-first campaigns for Q4.", MA_MSG_TYPE_RESPONSE);
        ma_group_chat_send(gc, rid, "Mobile-first is good, but we need desktop fallback too.", MA_MSG_TYPE_CRITIQUE);
        ma_group_chat_send(gc, pid, "Agreed. I'll add desktop support to the plan.", MA_MSG_TYPE_RESPONSE);
        ma_group_chat_send(gc, eid, "I can start the mobile campaign implementation now.", MA_MSG_TYPE_RESULT);
        ma_group_chat_send(gc, -1, "Great. Any other concerns before we proceed?", MA_MSG_TYPE_QUERY);
        ma_group_chat_send(gc, rid, "Just ensure proper A/B testing is in the timeline.", MA_MSG_TYPE_CRITIQUE);

        printf("Group chat messages: %d\n", gc->message_count);

        int next_speaker = ma_group_chat_speaker_select(gc);
        printf("Next speaker by round-robin: Agent[%d] %s\n",
               next_speaker, gc->agents[next_speaker]->name);

        char transcript[16384];
        ma_group_chat_get_transcript(gc, transcript, sizeof(transcript));
        printf("\n--- Group Chat Transcript ---\n%s\n", transcript);

        ma_group_chat_destroy(gc);
    }

    print_header("7. Tool Delegation Between Agents");

    ma_tool_delegation_t *td1 = ma_tool_delegate(ma, pid, eid, "calculator",
        "{\"expression\": \"4500 * 1.15\"}");
    if (td1) {
        printf("Tool delegation created:\n");
        printf("  From: Agent[%d] %s\n", td1->from_agent_id, ma->agents[td1->from_agent_id]->name);
        printf("  To:   Agent[%d] %s\n", td1->to_agent_id, ma->agents[td1->to_agent_id]->name);
        printf("  Tool: %s\n", td1->tool_name);
        printf("  Args: %s\n", td1->args);

        const char *res = ma_tool_collect_result(td1);
        printf("  Result: %s\n", res);

        ma_tool_delegation_destroy(td1);
    }

    ma_tool_delegation_t *td2 = ma_tool_delegate(ma, rid, eid, "web_search",
        "{\"query\": \"Q4 marketing trends 2024\"}");
    if (td2) {
        printf("\nSecond delegation:\n");
        printf("  From: Agent[%d] -> To: Agent[%d], Tool: %s\n",
               td2->from_agent_id, td2->to_agent_id, td2->tool_name);
        const char *res2 = ma_tool_collect_result(td2);
        printf("  Result: %.100s...\n", res2);
        ma_tool_delegation_destroy(td2);
    }

    print_header("8. AutoGen-style Conversation Programming");

    multi_agent_t *ma3 = multi_agent_create();
    ma_agent_def_t *ag1 = ma_create_planner("AutoPlanner");
    ma_agent_def_t *ag2 = ma_create_executor("AutoExecutor");
    multi_agent_register(ma3, ag1);
    multi_agent_register(ma3, ag2);

    printf("Registered %d agents for conversation programming.\n", ma3->agent_count);

    ma_conversation_program_t prog;
    memset(&prog, 0, sizeof(prog));
    prog.user_data = NULL;

    ma_conversation_program_run(ma3, &prog, "Build a weather dashboard app");

    printf("\nConversation program completed.\n");
    printf("  Agent count: %d\n", ma3->agent_count);
    printf("  Turn index: %d\n", ma3->turn_index);

    multi_agent_destroy(ma3);

    print_header("9. Agent Response Generation");

    char response[1024];
    ma_agent_generate_response(planner, "Create a 5-step plan for Q4 launch", response, sizeof(response));
    printf("Planner response: %s\n", response);

    ma_agent_generate_response(critic, "Review this plan: step1, step2, step3", response, sizeof(response));
    printf("Critic response: %s\n", response);

    bool valid = ma_verify_output(planner, response);
    printf("Output verified: %s\n", valid ? "yes" : "no");

    print_header("10. System Operations");

    printf("Deactivating Critic agent...\n");
    multi_agent_unregister(ma, cid);
    print_agent_info(ma->agents[cid]);

    ma_agent_def_t *retrieved = multi_agent_get(ma, pid);
    if (retrieved) {
        printf("Retrieved agent by ID %d: %s\n", pid, retrieved->name);
    }

    printf("\nFinal agent roster:\n");
    for (int i = 0; i < ma->agent_count; i++) {
        print_agent_info(ma->agents[i]);
    }

    multi_agent_destroy(ma);
    printf("\nDemo completed successfully.\n");
    return 0;
}
