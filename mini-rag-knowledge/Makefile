CC       = gcc
CFLAGS   = -std=gnu99 -Wall -Wextra -O2 -march=native
LDFLAGS  = -lm

SRCDIR   = src
INCDIR   = include
OBJDIR   = build
LIBNAME  = libminirag.a

SRCS     = $(wildcard $(SRCDIR)/*.c)
OBJS     = $(patsubst $(SRCDIR)/%.c, $(OBJDIR)/%.o, $(SRCS))

EXAMPLES_SRC = $(wildcard examples/*.c)
EXAMPLES_BIN = $(patsubst examples/%.c, $(OBJDIR)/examples/%, $(EXAMPLES_SRC))

DEMOS_SRC    = $(wildcard demos/*.c)
DEMOS_BIN    = $(patsubst demos/%.c, $(OBJDIR)/demos/%, $(DEMOS_SRC))

TESTS_SRC    = $(wildcard tests/*.c)
TESTS_BIN    = $(patsubst tests/%.c, $(OBJDIR)/tests/%.exe, $(TESTS_SRC))

MKDIR = mkdir -p
RM    = rm -rf

.PHONY: all clean examples demos test

all: $(OBJDIR)/$(LIBNAME)

$(OBJDIR)/%.o: $(SRCDIR)/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) -I$(INCDIR) -c $< -o $@

$(OBJDIR):
	$(MKDIR) $(OBJDIR)

$(OBJDIR)/$(LIBNAME): $(OBJS)
	ar rcs $@ $^

examples: $(OBJDIR)/$(LIBNAME) $(EXAMPLES_BIN)

$(OBJDIR)/examples/%.exe: examples/%.c $(OBJDIR)/$(LIBNAME) | $(OBJDIR)/examples
	$(CC) $(CFLAGS) -I$(INCDIR) $< $(OBJDIR)/$(LIBNAME) $(LDFLAGS) -o $@

$(OBJDIR)/examples:
	$(MKDIR) $(OBJDIR)/examples

demos: $(OBJDIR)/$(LIBNAME) $(DEMOS_BIN)

$(OBJDIR)/demos/%.exe: demos/%.c $(OBJDIR)/$(LIBNAME) | $(OBJDIR)/demos
	$(CC) $(CFLAGS) -I$(INCDIR) $< $(OBJDIR)/$(LIBNAME) $(LDFLAGS) -o $@

$(OBJDIR)/demos:
	$(MKDIR) $(OBJDIR)/demos

test: $(OBJDIR)/$(LIBNAME) $(TESTS_BIN)
	@for f in $(TESTS_BIN); do echo "Running $$f..."; $$f || exit 1; done
	@echo "All tests passed."

$(OBJDIR)/tests/%.exe: tests/%.c $(OBJDIR)/$(LIBNAME) | $(OBJDIR)/tests
	$(CC) $(CFLAGS) -I$(INCDIR) $< $(OBJDIR)/$(LIBNAME) $(LDFLAGS) -o $@

$(OBJDIR)/tests:
	$(MKDIR) $(OBJDIR)/tests

clean:
	$(RM) $(OBJDIR)
