#include "../include/recommendation.h"
#include "../include/chatbot_rlhf.h"
#include "../include/copilot_context.h"
#include "../include/eval_monitor.h"
#include "../include/model_ab_test.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(void) {
    printf("Test 10\n");
    double f = ev_f1_score(0.8, 0.6); printf("f1=%.4f OK\n", f);
    return 0; }
