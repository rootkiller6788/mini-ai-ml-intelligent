CC       = gcc
CFLAGS   = -std=c99 -Wall -Wextra -O2 -march=native -Iinclude
LDFLAGS  = -lm
VPATH    = src:examples:tests:benches:demos

TARGETS  = demo_ai_product.exe demo_full_pipeline.exe \
           example_recommendation.exe example_chatbot.exe example_copilot.exe \
           test_core.exe bench_core.exe demo_full.exe

SRC_OBJS = recommendation.o chatbot_rlhf.o copilot_context.o \
           eval_monitor.o model_ab_test.o bandit_allocator.o

.PHONY: all clean test run-demo run-pipeline run-examples

all: $(TARGETS)

%.o: src/%.c include/%.h
	$(CC) $(CFLAGS) -c $< -o $@

%.o: src/%.c
	$(CC) $(CFLAGS) -c $< -o $@

demo_ai_product.exe: examples/demo_ai_product.c $(SRC_OBJS)
	$(CC) $(CFLAGS) $< $(SRC_OBJS) -o $@ $(LDFLAGS)

demo_full_pipeline.exe: examples/demo_full_pipeline.c $(SRC_OBJS)
	$(CC) $(CFLAGS) $< $(SRC_OBJS) -o $@ $(LDFLAGS)

example_recommendation.exe: examples/example_recommendation.c recommendation.o bandit_allocator.o
	$(CC) $(CFLAGS) $< recommendation.o bandit_allocator.o -o $@ $(LDFLAGS)

example_chatbot.exe: examples/example_chatbot.c chatbot_rlhf.o
	$(CC) $(CFLAGS) $< chatbot_rlhf.o -o $@ $(LDFLAGS)

example_copilot.exe: examples/example_copilot.c copilot_context.o
	$(CC) $(CFLAGS) $< copilot_context.o -o $@ $(LDFLAGS)

test_core.exe: tests/test_core.c $(SRC_OBJS)
	$(CC) $(CFLAGS) $< $(SRC_OBJS) -o $@ $(LDFLAGS)

bench_core.exe: benches/bench_core.c $(SRC_OBJS)
	$(CC) $(CFLAGS) $< $(SRC_OBJS) -o $@ $(LDFLAGS)

demo_full.exe: demos/demo_full.c $(SRC_OBJS)
	$(CC) $(CFLAGS) $< $(SRC_OBJS) -o $@ $(LDFLAGS)

test: test_core.exe
	@echo === Running Unit Tests ===
	./test_core.exe

run-demo: demo_ai_product.exe
	./demo_ai_product.exe

run-pipeline: demo_full_pipeline.exe
	./demo_full_pipeline.exe

run-examples: example_recommendation.exe example_chatbot.exe example_copilot.exe
	@echo === Recommendation Example ===
	./example_recommendation.exe
	@echo === ChatBot Example ===
	./example_chatbot.exe
	@echo === Copilot Example ===
	./example_copilot.exe

clean:
	rm -f *.o *.exe 2>/dev/null
