CC       = gcc
CFLAGS   = -std=c99 -Wall -Wextra -O2 -march=native
LDFLAGS  = -lm

SRCDIR   = src
INCDIR   = include
OBJDIR   = build
LIBNAME  = libminirag.a

SRCS     = $(wildcard $(SRCDIR)/*.c)
OBJS     = $(patsubst $(SRCDIR)/%.c, $(OBJDIR)/%.o, $(SRCS))

EXAMPLES_SRC = $(wildcard examples/*.c)
EXAMPLES_BIN = $(patsubst examples/%.c, build/examples/%, $(EXAMPLES_SRC))

DEMOS_SRC    = $(wildcard demos/*.c)
DEMOS_BIN    = $(patsubst demos/%.c, build/demos/%, $(DEMOS_SRC))

TESTS_SRC    = $(wildcard tests/*.c)
TESTS_BIN    = $(patsubst tests/%.c, build/tests/%, $(TESTS_SRC))

.PHONY: all clean examples demos test

all: $(OBJDIR)/$(LIBNAME)

$(OBJDIR)/%.o: $(SRCDIR)/%.c
	@if not exist "$(OBJDIR)" mkdir "$(OBJDIR)"
	$(CC) $(CFLAGS) -I$(INCDIR) -c $< -o $@

$(OBJDIR)/$(LIBNAME): $(OBJS)
	ar rcs $@ $^

examples: $(OBJDIR)/$(LIBNAME) $(EXAMPLES_BIN)

build/examples/%: examples/%.c $(OBJDIR)/$(LIBNAME)
	@if not exist "build\examples" mkdir "build\examples"
	$(CC) $(CFLAGS) -I$(INCDIR) $< -L$(OBJDIR) -lminirag $(LDFLAGS) -o $@

demos: $(OBJDIR)/$(LIBNAME) $(DEMOS_BIN)

build/demos/%: demos/%.c $(OBJDIR)/$(LIBNAME)
	@if not exist "build\demos" mkdir "build\demos"
	$(CC) $(CFLAGS) -I$(INCDIR) $< -L$(OBJDIR) -lminirag $(LDFLAGS) -o $@

test: $(OBJDIR)/$(LIBNAME) $(TESTS_BIN)
	@for %%f in ($(TESTS_BIN)) do %%f

build/tests/%: tests/%.c $(OBJDIR)/$(LIBNAME)
	@if not exist "build\tests" mkdir "build\tests"
	$(CC) $(CFLAGS) -I$(INCDIR) $< -L$(OBJDIR) -lminirag $(LDFLAGS) -o $@

clean:
	@if exist "$(OBJDIR)" rmdir /s /q "$(OBJDIR)"
