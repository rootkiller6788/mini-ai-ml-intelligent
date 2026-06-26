SHELL    = sh
CC       = gcc
CFLAGS   = -Wall -Wextra -O2 -I include
LDFLAGS  = -lm
OUTDIR   = bin

SRCS = src/linear_models.c \
       src/svm_kernel.c    \
       src/decision_tree.c \
       src/clustering.c    \
       src/ensemble_gbdt.c \
       src/neighbor_models.c \
       src/model_selection.c \
       src/preprocessing.c

OBJS = $(SRCS:src/%.c=$(OUTDIR)/%.o)

EXAMPLES = $(OUTDIR)/example_linear \
           $(OUTDIR)/example_tree   \
           $(OUTDIR)/example_cluster \
           $(OUTDIR)/example_neighbor \
           $(OUTDIR)/example_model_sel

TESTS = $(OUTDIR)/test_core

BENCHES = $(OUTDIR)/bench_core

.PHONY: all clean dirs test bench

all: dirs $(OBJS) $(EXAMPLES)
	@echo build complete -- all objects in $(OUTDIR)/

dirs:
	@mkdir -p "$(OUTDIR)"

$(OUTDIR)/%.o: src/%.c
	$(CC) $(CFLAGS) -c $< -o $@

$(OUTDIR)/example_linear: examples/example_linear.c $(OUTDIR)/linear_models.o
	$(CC) $(CFLAGS) $< $(OUTDIR)/linear_models.o $(LDFLAGS) -o $@

$(OUTDIR)/example_tree: examples/example_tree.c $(OUTDIR)/decision_tree.o
	$(CC) $(CFLAGS) $< $(OUTDIR)/decision_tree.o $(LDFLAGS) -o $@

$(OUTDIR)/example_cluster: examples/example_cluster.c $(OUTDIR)/clustering.o
	$(CC) $(CFLAGS) $< $(OUTDIR)/clustering.o $(LDFLAGS) -o $@

$(OUTDIR)/example_neighbor: examples/example_neighbor.c $(OUTDIR)/neighbor_models.o $(OUTDIR)/preprocessing.o
	$(CC) $(CFLAGS) $< $(OUTDIR)/neighbor_models.o $(OUTDIR)/preprocessing.o $(LDFLAGS) -o $@

$(OUTDIR)/example_model_sel: examples/example_model_sel.c $(OUTDIR)/model_selection.o $(OUTDIR)/linear_models.o
	$(CC) $(CFLAGS) $< $(OUTDIR)/model_selection.o $(OUTDIR)/linear_models.o $(LDFLAGS) -o $@

$(OUTDIR)/test_core: tests/test_core.c $(OBJS)
	$(CC) $(CFLAGS) $< $(OBJS) $(LDFLAGS) -o $@

$(OUTDIR)/bench_core: benches/bench_core.c $(OBJS)
	$(CC) $(CFLAGS) $< $(OBJS) $(LDFLAGS) -o $@

test: dirs $(OUTDIR)/test_core
	@echo "=== Running Tests ==="
	$(OUTDIR)/test_core
	@echo "=== Tests Complete ==="

bench: dirs $(OUTDIR)/bench_core
	@echo "=== Running Benchmarks ==="
	$(OUTDIR)/bench_core 500
	@echo "=== Benchmarks Complete ==="

clean:
	@rm -rf "$(OUTDIR)"
