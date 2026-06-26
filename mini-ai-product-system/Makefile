CC       = gcc
CFLAGS   = -std=c99 -Wall -Wextra -O2 -march=native
LDFLAGS  = -lm
TARGETS  = demo_ai_product.exe demo_full_pipeline.exe \
           example_recommendation.exe example_chatbot.exe example_copilot.exe

OBJS = recommendation.o chatbot_rlhf.o copilot_context.o eval_monitor.o model_ab_test.o

.PHONY: all clean run-demo run-pipeline run-examples

all: $(TARGETS)

recommendation.o: recommendation.c recommendation.h
	$(CC) $(CFLAGS) -c recommendation.c -o recommendation.o

chatbot_rlhf.o: chatbot_rlhf.c chatbot_rlhf.h
	$(CC) $(CFLAGS) -c chatbot_rlhf.c -o chatbot_rlhf.o

copilot_context.o: copilot_context.c copilot_context.h
	$(CC) $(CFLAGS) -c copilot_context.c -o copilot_context.o

eval_monitor.o: eval_monitor.c eval_monitor.h
	$(CC) $(CFLAGS) -c eval_monitor.c -o eval_monitor.o

model_ab_test.o: model_ab_test.c model_ab_test.h
	$(CC) $(CFLAGS) -c model_ab_test.c -o model_ab_test.o

demo_ai_product.exe: demo_ai_product.c $(OBJS)
	$(CC) $(CFLAGS) demo_ai_product.c $(OBJS) -o demo_ai_product.exe $(LDFLAGS)

demo_full_pipeline.exe: demo_full_pipeline.c $(OBJS)
	$(CC) $(CFLAGS) demo_full_pipeline.c $(OBJS) -o demo_full_pipeline.exe $(LDFLAGS)

example_recommendation.exe: example_recommendation.c recommendation.o
	$(CC) $(CFLAGS) example_recommendation.c recommendation.o -o example_recommendation.exe $(LDFLAGS)

example_chatbot.exe: example_chatbot.c chatbot_rlhf.o
	$(CC) $(CFLAGS) example_chatbot.c chatbot_rlhf.o -o example_chatbot.exe $(LDFLAGS)

example_copilot.exe: example_copilot.c copilot_context.o
	$(CC) $(CFLAGS) example_copilot.c copilot_context.o -o example_copilot.exe $(LDFLAGS)

run-demo: demo_ai_product.exe
	.\demo_ai_product.exe

run-pipeline: demo_full_pipeline.exe
	.\demo_full_pipeline.exe

run-examples: example_recommendation.exe example_chatbot.exe example_copilot.exe
	@echo === Recommendation Example ===
	.\example_recommendation.exe
	@echo === ChatBot Example ===
	.\example_chatbot.exe
	@echo === Copilot Example ===
	.\example_copilot.exe

clean:
	del /f *.o *.exe 2>nul
