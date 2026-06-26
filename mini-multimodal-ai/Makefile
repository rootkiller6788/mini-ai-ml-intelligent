CC       = gcc
CFLAGS   = -std=c99 -Wall -Wextra -O3 -march=native -Iinclude
LDFLAGS  = -lm
AR       = ar rcs

OBJS     = clip_contrastive.o image_generation.o vlm_llama.o \
           audio_whisper.o video_understanding.o
HEADERS  = include/clip_contrastive.h include/image_generation.h include/vlm_llama.h \
           include/audio_whisper.h include/video_understanding.h
LIB      = libmultimodal.a

TARGETS  = example_clip example_sd example_llava test_core

.PHONY: all clean test

all: $(LIB) $(TARGETS) demo_multimodal demo_pipeline

$(LIB): $(OBJS)
	$(AR) $@ $^

clip_contrastive.o: src/clip_contrastive.c $(HEADERS)
	$(CC) $(CFLAGS) -c -o $@ $<

image_generation.o: src/image_generation.c $(HEADERS)
	$(CC) $(CFLAGS) -c -o $@ $<

vlm_llama.o: src/vlm_llama.c $(HEADERS)
	$(CC) $(CFLAGS) -c -o $@ $<

audio_whisper.o: src/audio_whisper.c $(HEADERS)
	$(CC) $(CFLAGS) -c -o $@ $<

video_understanding.o: src/video_understanding.c $(HEADERS)
	$(CC) $(CFLAGS) -c -o $@ $<

example_clip: examples/example_clip.c $(LIB) $(HEADERS)
	$(CC) $(CFLAGS) -o $@ $< $(LIB) $(LDFLAGS)

example_sd: examples/example_sd.c $(LIB) $(HEADERS)
	$(CC) $(CFLAGS) -o $@ $< $(LIB) $(LDFLAGS)

example_llava: examples/example_llava.c $(LIB) $(HEADERS)
	$(CC) $(CFLAGS) -o $@ $< $(LIB) $(LDFLAGS)

demo_multimodal: examples/demo_multimodal.c $(LIB) $(HEADERS)
	$(CC) $(CFLAGS) -o $@ $< $(LIB) $(LDFLAGS)

demo_pipeline: examples/demo_pipeline.c $(LIB) $(HEADERS)
	$(CC) $(CFLAGS) -o $@ $< $(LIB) $(LDFLAGS)

test_core: tests/test_core.c $(LIB) $(HEADERS)
	$(CC) $(CFLAGS) -o $@ $< $(LIB) $(LDFLAGS)

test: $(TARGETS)
	@printf "=== Running Unit Tests ===\n"
	./test_core
	@printf "\n=== Running CLIP Example ===\n"
	./example_clip
	@printf "\n=== Running Stable Diffusion Example ===\n"
	./example_sd
	@printf "\n=== Running LLaVA Example ===\n"
	./example_llava
	@printf "\n=== All Tests Passed ===\n"

clean:
	-rm -f $(OBJS) $(LIB) $(TARGETS) demo_multimodal demo_pipeline demo_multimodal.exe demo_pipeline.exe test_core.exe example_clip.exe example_sd.exe example_llava.exe
