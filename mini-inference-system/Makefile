CC       = gcc
CFLAGS   = -std=c99 -Wall -Wextra -O3 -march=native -fopenmp
LDFLAGS  = -lm -lpthread -fopenmp
INCDIR   = include
SRCDIR   = src
EXAMPLEDIR = examples
DEMODIR  = demos
BUILDDIR = build

SRCS     = $(wildcard $(SRCDIR)/*.c)
OBJS     = $(patsubst $(SRCDIR)/%.c,$(BUILDDIR)/%.o,$(SRCS))
LIB      = $(BUILDDIR)/libinference.a

EXAMPLES = $(patsubst $(EXAMPLEDIR)/%.c,$(BUILDDIR)/%,$(wildcard $(EXAMPLEDIR)/*.c))
DEMOS    = $(patsubst $(DEMODIR)/%.c,$(BUILDDIR)/%,$(wildcard $(DEMODIR)/*.c))

TESTDIR  = tests
TEST     = $(BUILDDIR)/test_core
BENCHDIR = benches
BENCH    = $(BUILDDIR)/bench_core

.PHONY: all clean demo test benchmark

all: $(LIB) $(EXAMPLES) $(DEMOS)

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

$(BUILDDIR)/%.o: $(SRCDIR)/%.c | $(BUILDDIR)
	$(CC) $(CFLAGS) -I$(INCDIR) -c $< -o $@

$(LIB): $(OBJS)
	ar rcs $@ $^

$(BUILDDIR)/example_%: $(EXAMPLEDIR)/example_%.c $(LIB) | $(BUILDDIR)
	$(CC) $(CFLAGS) -I$(INCDIR) $< -L$(BUILDDIR) -linference $(LDFLAGS) -o $@

$(BUILDDIR)/demo_%: $(DEMODIR)/demo_%.c $(LIB) | $(BUILDDIR)
	$(CC) $(CFLAGS) -I$(INCDIR) $< -L$(BUILDDIR) -linference $(LDFLAGS) -o $@

demo: $(DEMOS)
	@for d in $(DEMOS); do echo "=== Running $$(basename $$d) ==="; $$d; echo; done

$(TEST): $(TESTDIR)/test_core.c $(LIB) | $(BUILDDIR)
	$(CC) $(CFLAGS) -I$(INCDIR) $< -L$(BUILDDIR) -linference $(LDFLAGS) -o $@

test: $(LIB) $(TEST)
	@echo "=== Running Tests ==="
	$(TEST)
	@echo "=== Tests Complete ==="

$(BENCH): $(BENCHDIR)/bench_core.c $(LIB) | $(BUILDDIR)
	$(CC) $(CFLAGS) -I$(INCDIR) $< -L$(BUILDDIR) -linference $(LDFLAGS) -o $@

benchmark: $(LIB) $(BENCH)
	@echo "=== Running Benchmarks ==="
	$(BENCH) 5000
	@echo "=== Benchmarks Complete ==="

clean:
	rm -rf $(BUILDDIR)
