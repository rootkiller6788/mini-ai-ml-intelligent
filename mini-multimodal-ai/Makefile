CC       = gcc
CFLAGS   = -std=c99 -Wall -Wextra -O3 -march=native -fopenmp
LDFLAGS  = -lm -fopenmp
AR       = ar rcs

SRCS     = clip_contrastive.c image_generation.c vlm_llama.c audio_whisper.c video_understanding.c
OBJS     = $(SRCS:.c=.o)
HEADERS  = clip_contrastive.h image_generation.h vlm_llama.h audio_whisper.h video_understanding.h
LIB      = libmultimodal.a

EXAMPLES = example_clip example_sd example_llava
DEMOS    = demo_multimodal demo_pipeline
TARGETS  = $(EXAMPLES) $(DEMOS)

.PHONY: all clean test

all: $(LIB) $(TARGETS)

$(LIB): $(OBJS)
	$(AR) $@ $^

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c -o $@ $<

example_clip: example_clip.c $(LIB)
	$(CC) $(CFLAGS) -o $@ $< $(LIB) $(LDFLAGS)

example_sd: example_sd.c $(LIB)
	$(CC) $(CFLAGS) -o $@ $< $(LIB) $(LDFLAGS)

example_llava: example_llava.c $(LIB)
	$(CC) $(CFLAGS) -o $@ $< $(LIB) $(LDFLAGS)

demo_multimodal: demo_multimodal.c $(LIB)
	$(CC) $(CFLAGS) -o $@ $< $(LIB) $(LDFLAGS)

demo_pipeline: demo_pipeline.c $(LIB)
	$(CC) $(CFLAGS) -o $@ $< $(LIB) $(LDFLAGS)

test: $(EXAMPLES)
	@echo "=== Running CLIP Zero-shot Example ==="
	./example_clip
	@echo "=== Running Stable Diffusion Example ==="
	./example_sd
	@echo "=== Running LLaVA Example ==="
	./example_llava

clean:
	del /Q $(OBJS) $(LIB) $(TARGETS) 2>nul || rm -f $(OBJS) $(LIB) $(TARGETS)
