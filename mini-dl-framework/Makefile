CC = gcc
CFLAGS = -std=c99 -Wall -Wextra -O2 -fopenmp
LDFLAGS = -lm -fopenmp
AR = ar
ARFLAGS = rcs

SRCDIR = .
OBJDIR = obj
LIBDIR = lib
BINDIR = bin

LIB = $(LIBDIR)/libminidl.a
DLL = $(LIBDIR)/libminidl.so

SOURCES = autograd.c tensor_ops.c nn_layers.c optimizers.c loss_funcs.c
OBJECTS = $(patsubst %.c,$(OBJDIR)/%.o,$(SOURCES))

EXAMPLES = example_mlp example_convnet example_gan
DEMOS = demo_mnist demo_transfer_learning

HEADERS = autograd.h tensor_ops.h nn_layers.h optimizers.h loss_funcs.h

.PHONY: all clean demo run examples

all: $(LIB) examples

$(OBJDIR):
	@if not exist "$(OBJDIR)" mkdir "$(OBJDIR)"

$(LIBDIR):
	@if not exist "$(LIBDIR)" mkdir "$(LIBDIR)"

$(BINDIR):
	@if not exist "$(BINDIR)" mkdir "$(BINDIR)"

$(OBJDIR)/%.o: %.c $(HEADERS) | $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(LIB): $(OBJECTS) | $(LIBDIR)
	$(AR) $(ARFLAGS) $@ $(OBJECTS)

examples: $(LIB) $(addprefix $(BINDIR)/,$(EXAMPLES))

$(BINDIR)/example_%: example_%.c $(LIB) | $(BINDIR)
	$(CC) $(CFLAGS) $< -o $@ -L$(LIBDIR) -lminidl $(LDFLAGS)

demo: $(LIB) $(addprefix $(BINDIR)/,$(DEMOS))

$(BINDIR)/demo_%: demo_%.c $(LIB) | $(BINDIR)
	$(CC) $(CFLAGS) $< -o $@ -L$(LIBDIR) -lminidl $(LDFLAGS)

run: examples
	@for %%e in ($(EXAMPLES)) do @(echo === %%e === && $(BINDIR)\%%e.exe)

test: run

clean:
	@if exist "$(OBJDIR)" rmdir /s /q "$(OBJDIR)"
	@if exist "$(LIBDIR)" rmdir /s /q "$(LIBDIR)"
	@if exist "$(BINDIR)" rmdir /s /q "$(BINDIR)"
