CC       = gcc
CFLAGS   = -std=c99 -Wall -Wextra -O2 -Iinclude
LDFLAGS  = -lm
AR       = ar rcs

BUILD    = build
INC      = include
SRC      = src
EXAMPLES = examples
DEMOS    = demos
TESTS    = tests

SRCS  = $(wildcard $(SRC)/*.c)
OBJS  = $(SRCS:$(SRC)/%.c=$(BUILD)/%.o)
LIB   = $(BUILD)/libtrain.a

EXAMPLE_BINS = $(BUILD)/example_ddp_train.exe \
               $(BUILD)/example_mixed_precision.exe \
               $(BUILD)/example_checkpoint_resume.exe

DEMO_BINS = $(BUILD)/demo_tune.exe \
            $(BUILD)/demo_training.exe

TEST_BIN = $(BUILD)/test_core.exe

.PHONY: all lib examples demos test clean

all: lib examples demos test

$(BUILD):
	mkdir -p "$(BUILD)"

lib: $(BUILD) $(LIB)

$(LIB): $(OBJS)
	$(AR) $@ $^

$(BUILD)/%.o: $(SRC)/%.c
	$(CC) $(CFLAGS) -c -o $@ $<

examples: $(BUILD) $(EXAMPLE_BINS)

$(BUILD)/example_%.exe: $(EXAMPLES)/example_%.c $(LIB)
	$(CC) $(CFLAGS) -o $@ $< -L$(BUILD) -ltrain $(LDFLAGS)

demos: $(BUILD) $(DEMO_BINS)

$(BUILD)/demo_%.exe: $(DEMOS)/demo_%.c $(LIB)
	$(CC) $(CFLAGS) -o $@ $< -L$(BUILD) -ltrain $(LDFLAGS)

test: $(BUILD) $(TEST_BIN)
	$(TEST_BIN)

$(TEST_BIN): $(TESTS)/test_core.c $(LIB)
	$(CC) $(CFLAGS) -o $@ $< -L$(BUILD) -ltrain $(LDFLAGS)

clean:
	rm -rf "$(BUILD)" *.txt 2>/dev/null; true
