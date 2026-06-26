# mini-model-arch Makefile — C99, no external dependencies
CC       = gcc
CFLAGS   = -std=c99 -O2 -Wall -Wextra -pedantic -I $(INC_DIR)
LDFLAGS  = -lm
INC_DIR  = include
SRC_DIR  = src
TST_DIR  = tests
EXA_DIR  = examples
BNC_DIR  = benches

_MAIN_SRCS  = cnn_models.c rnn_lstm.c transformer_arch.c gan_model.c diffusion_model.c
_EXT_SRCS   = optimizers.c loss_functions.c normalization.c activations.c regularization.c position_encodings.c
ALL_SRCS    = $(_MAIN_SRCS) $(_EXT_SRCS)

OBJS     = $(ALL_SRCS:%.c=$(SRC_DIR)/%.o)
LIB      = libmodelarch.a

EXAMPLES = $(EXA_DIR)/example_cnn $(EXA_DIR)/example_rnn $(EXA_DIR)/example_transformer \
           $(EXA_DIR)/demo_gan $(EXA_DIR)/demo_diffusion
TEST_BIN = $(TST_DIR)/test_core
BENCH_BIN = $(BNC_DIR)/bench_core

.PHONY: all test clean lib

all: lib $(EXAMPLES)

lib: $(LIB)

$(LIB): $(OBJS)
	ar rcs $(LIB) $(OBJS)

# Main source files
$(SRC_DIR)/cnn_models.o:         $(SRC_DIR)/cnn_models.c         $(INC_DIR)/cnn_models.h
	$(CC) $(CFLAGS) -c $< -o $@

$(SRC_DIR)/rnn_lstm.o:           $(SRC_DIR)/rnn_lstm.c           $(INC_DIR)/rnn_lstm.h
	$(CC) $(CFLAGS) -c $< -o $@

$(SRC_DIR)/transformer_arch.o:   $(SRC_DIR)/transformer_arch.c   $(INC_DIR)/transformer_arch.h
	$(CC) $(CFLAGS) -c $< -o $@

$(SRC_DIR)/gan_model.o:          $(SRC_DIR)/gan_model.c          $(INC_DIR)/gan_model.h
	$(CC) $(CFLAGS) -c $< -o $@

$(SRC_DIR)/diffusion_model.o:    $(SRC_DIR)/diffusion_model.c    $(INC_DIR)/diffusion_model.h
	$(CC) $(CFLAGS) -c $< -o $@

# Extended source files
$(SRC_DIR)/optimizers.o:         $(SRC_DIR)/optimizers.c         $(INC_DIR)/optimizers.h
	$(CC) $(CFLAGS) -c $< -o $@

$(SRC_DIR)/loss_functions.o:     $(SRC_DIR)/loss_functions.c     $(INC_DIR)/loss_functions.h
	$(CC) $(CFLAGS) -c $< -o $@

$(SRC_DIR)/normalization.o:      $(SRC_DIR)/normalization.c      $(INC_DIR)/normalization.h
	$(CC) $(CFLAGS) -c $< -o $@

$(SRC_DIR)/activations.o:         $(SRC_DIR)/activations.c         $(INC_DIR)/activations.h
	$(CC) $(CFLAGS) -c $< -o $@

$(SRC_DIR)/regularization.o:     $(SRC_DIR)/regularization.c     $(INC_DIR)/regularization.h
	$(CC) $(CFLAGS) -c $< -o $@

$(SRC_DIR)/position_encodings.o: $(SRC_DIR)/position_encodings.c $(INC_DIR)/position_encodings.h
	$(CC) $(CFLAGS) -c $< -o $@

# Examples
$(EXA_DIR)/example_cnn: $(EXA_DIR)/example_cnn.c $(LIB)
	$(CC) $(CFLAGS) $< -L. -lmodelarch $(LDFLAGS) -o $@

$(EXA_DIR)/example_rnn: $(EXA_DIR)/example_rnn.c $(LIB)
	$(CC) $(CFLAGS) $< -L. -lmodelarch $(LDFLAGS) -o $@

$(EXA_DIR)/example_transformer: $(EXA_DIR)/example_transformer.c $(LIB)
	$(CC) $(CFLAGS) $< -L. -lmodelarch $(LDFLAGS) -o $@

$(EXA_DIR)/demo_gan: $(EXA_DIR)/demo_gan.c $(LIB)
	$(CC) $(CFLAGS) $< -L. -lmodelarch $(LDFLAGS) -o $@

$(EXA_DIR)/demo_diffusion: $(EXA_DIR)/demo_diffusion.c $(LIB)
	$(CC) $(CFLAGS) $< -L. -lmodelarch $(LDFLAGS) -o $@

# Test target — must pass for COMPLETE status
test: $(TEST_BIN)
	$(TST_DIR)/test_core

$(TEST_BIN): $(TST_DIR)/test_core.c $(LIB)
	$(CC) $(CFLAGS) $< -L. -lmodelarch $(LDFLAGS) -o $(TST_DIR)/test_core

# Benchmark
bench: $(BENCH_BIN)
	./$(BENCH_BIN)

$(BENCH_BIN): $(BNC_DIR)/bench_core.c $(LIB)
	$(CC) $(CFLAGS) $< -L. -lmodelarch $(LDFLAGS) -o $@

clean:
	del /f /q $(SRC_DIR)\*.o $(LIB) $(EXA_DIR)\*.exe $(TST_DIR)\*.exe $(BNC_DIR)\*.exe 2>nul || \
	rm -f $(SRC_DIR)/*.o $(LIB) $(EXAMPLES) $(TEST_BIN) $(BENCH_BIN)
