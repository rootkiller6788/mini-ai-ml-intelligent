CC = gcc
CFLAGS = -std=c99 -Wall -Wextra -pedantic -O2 -Iinclude
LDFLAGS = -lm
AR = ar
ARFLAGS = rcs

SRCDIR = src
OBJDIR = build
LIBDIR = build
BINDIR = build

_SRCS = react_agent.c tool_use.c planning_system.c agent_memory.c multi_agent.c
SRCS = $(_SRCS:%=$(SRCDIR)/%)

_OBJS = $(_SRCS:.c=.o)
OBJS = $(_OBJS:%=$(OBJDIR)/%)

_EXAMPLES = example_react example_tool_use example_planning
EXAMPLES = $(_EXAMPLES:%=$(BINDIR)/%)

_DEMOS = demo_memory demo_multi_agent
DEMOS = $(_DEMOS:%=$(BINDIR)/%)

LIBRARY = $(LIBDIR)/libminiruntime.a

.PHONY: all lib examples demos clean

all: lib examples demos

lib: $(LIBRARY)

examples: $(EXAMPLES)

demos: $(DEMOS)

$(LIBRARY): $(OBJS)
	@mkdir -p "$(LIBDIR)"
	$(AR) $(ARFLAGS) $@ $^

$(OBJDIR)/%.o: $(SRCDIR)/%.c
	@mkdir -p "$(OBJDIR)"
	$(CC) $(CFLAGS) -c $< -o $@

$(BINDIR)/example_react: examples/example_react.c $(LIBRARY)
	@mkdir -p "$(BINDIR)"
	$(CC) $(CFLAGS) $< -L$(LIBDIR) -lminiruntime $(LDFLAGS) -o $@

$(BINDIR)/example_tool_use: examples/example_tool_use.c $(LIBRARY)
	@mkdir -p "$(BINDIR)"
	$(CC) $(CFLAGS) $< -L$(LIBDIR) -lminiruntime $(LDFLAGS) -o $@

$(BINDIR)/example_planning: examples/example_planning.c $(LIBRARY)
	@mkdir -p "$(BINDIR)"
	$(CC) $(CFLAGS) $< -L$(LIBDIR) -lminiruntime $(LDFLAGS) -o $@

$(BINDIR)/demo_memory: demo/demo_memory.c $(LIBRARY)
	@mkdir -p "$(BINDIR)"
	$(CC) $(CFLAGS) $< -L$(LIBDIR) -lminiruntime $(LDFLAGS) -o $@

$(BINDIR)/demo_multi_agent: demo/demo_multi_agent.c $(LIBRARY)
	@mkdir -p "$(BINDIR)"
	$(CC) $(CFLAGS) $< -L$(LIBDIR) -lminiruntime $(LDFLAGS) -o $@

clean:
	rm -rf "$(OBJDIR)" "$(LIBDIR)" "$(BINDIR)"
