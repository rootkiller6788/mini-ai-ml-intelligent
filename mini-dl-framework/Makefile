CC = gcc
CFLAGS = -std=c99 -Wall -Wextra -O2 -Iinclude
LDFLAGS = -lm
AR = ar
ARFLAGS = rcs

SRCDIR = src
OBJDIR = obj
LIBDIR = lib
BINDIR = bin
TESTDIR = tests

LIB = $(LIBDIR)/libminidl.a

SOURCES = autograd.c tensor_ops.c nn_layers.c optimizers.c loss_funcs.c rnn_layers.c training.c
OBJECTS = $(patsubst %.c,$(OBJDIR)/%.o,$(SOURCES))

HEADERS = include/autograd.h include/tensor_ops.h include/nn_layers.h \
          include/optimizers.h include/loss_funcs.h include/rnn_layers.h \
          include/training.h

.PHONY: all clean test

all: $(LIB) test

$(OBJDIR):
	@mkdir -p "$(OBJDIR)"

$(LIBDIR):
	@mkdir -p "$(LIBDIR)"

$(BINDIR):
	@mkdir -p "$(BINDIR)"

$(OBJDIR)/%.o: $(SRCDIR)/%.c $(HEADERS) | $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(LIB): $(OBJECTS) | $(LIBDIR)
	$(AR) $(ARFLAGS) $@ $(OBJECTS)

$(BINDIR)/test_core.exe: $(TESTDIR)/test_core.c $(LIB) | $(BINDIR)
	$(CC) $(CFLAGS) $< -o $@ -L$(LIBDIR) -lminidl $(LDFLAGS)

test: $(LIB) $(BINDIR)/test_core.exe
	@echo "=== Running Tests ==="
	./$(BINDIR)/test_core.exe

clean:
	@rm -rf "$(OBJDIR)" "$(LIBDIR)" "$(BINDIR)"
