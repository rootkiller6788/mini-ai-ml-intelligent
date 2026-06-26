SHELL    = sh
CC       = gcc
CFLAGS   = -Wall -Wextra -O2 -I include
LDFLAGS  = -lm
OUTDIR   = bin

SRCS = src/linear_models.c \
       src/svm_kernel.c    \
       src/decision_tree.c \
       src/clustering.c    \
       src/ensemble_gbdt.c

OBJS = $(SRCS:src/%.c=$(OUTDIR)/%.o)

EXAMPLES = $(OUTDIR)/example_linear \
           $(OUTDIR)/example_tree   \
           $(OUTDIR)/example_cluster

.PHONY: all clean dirs

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

clean:
	@rm -rf "$(OUTDIR)"
