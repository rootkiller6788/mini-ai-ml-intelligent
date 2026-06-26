CC       = gcc
CFLAGS   = -std=c99 -Wall -Wextra -O2 -Iinclude
LDFLAGS  = -lm
AR       = ar rcs

BUILD    = build
INC      = include
SRC      = src
EXAMPLES = examples
DEMOS    = demos

SRCS  = $(wildcard $(SRC)/*.c)
OBJS  = $(SRCS:$(SRC)/%.c=$(BUILD)/%.o)
LIB   = $(BUILD)/libtrain.a

EXAMPLE_BINS = $(BUILD)/example_ddp_train.exe \
               $(BUILD)/example_mixed_precision.exe \
               $(BUILD)/example_checkpoint_resume.exe

DEMO_BINS = $(BUILD)/demo_tune.exe \
            $(BUILD)/demo_training.exe

.PHONY: all lib examples demos clean

all: lib examples demos

$(BUILD):
	if not exist "$(BUILD)" mkdir "$(BUILD)"

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

clean:
	if exist "$(BUILD)" rmdir /s /q "$(BUILD)"
	if exist *.txt del *.txt
