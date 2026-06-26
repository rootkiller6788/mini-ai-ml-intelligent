CC      = gcc
CFLAGS  = -std=c99 -Wall -Wextra -O2 -fopenmp
LDFLAGS = -lm
INC_DIR = include
SRC_DIR = src
EXA_DIR = examples
DEM_DIR = demo
BLD_DIR = build
DOC_DIR = doc

SRCS = $(wildcard $(SRC_DIR)/*.c)
OBJS = $(SRCS:$(SRC_DIR)/%.c=$(BLD_DIR)/%.o)
EXAS = $(wildcard $(EXA_DIR)/*.c)
EXAB = $(EXAS:$(EXA_DIR)/%.c=$(BLD_DIR)/%)
DEMS = $(wildcard $(DEM_DIR)/*.c)
DEMB = $(DEMS:$(DEM_DIR)/%.c=$(BLD_DIR)/%)

.PHONY: all examples demos clean

all: examples demos

examples: $(EXAB)

demos: $(DEMB)

$(BLD_DIR)/%.o: $(SRC_DIR)/%.c $(INC_DIR)/%.h
	@if not exist "$(BLD_DIR)" mkdir "$(BLD_DIR)"
	$(CC) $(CFLAGS) -I$(INC_DIR) -c $< -o $@

$(BLD_DIR)/example_%: $(EXA_DIR)/example_%.c $(OBJS)
	@if not exist "$(BLD_DIR)" mkdir "$(BLD_DIR)"
	$(CC) $(CFLAGS) -I$(INC_DIR) $< $(OBJS) $(LDFLAGS) -o $@

$(BLD_DIR)/demo_%: $(DEM_DIR)/demo_%.c $(OBJS)
	@if not exist "$(BLD_DIR)" mkdir "$(BLD_DIR)"
	$(CC) $(CFLAGS) -I$(INC_DIR) $< $(OBJS) $(LDFLAGS) -o $@

clean:
	@if exist "$(BLD_DIR)" rmdir /s /q "$(BLD_DIR)"
