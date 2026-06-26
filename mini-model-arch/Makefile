CC = gcc
CFLAGS = -std=c99 -O2 -Wall -Wextra -pedantic -lm
SRCS = cnn_models.c rnn_lstm.c transformer_arch.c gan_model.c diffusion_model.c
OBJS = $(SRCS:.c=.o)
EXAMPLES = example_cnn example_rnn example_transformer
DEMOS = demo_gan demo_diffusion

.PHONY: all clean

all: $(EXAMPLES) $(DEMOS)

%.o: %.c %.h
	$(CC) $(CFLAGS) -c $< -o $@

example_cnn: example_cnn.c cnn_models.o
	$(CC) $(CFLAGS) $^ -o $@

example_rnn: example_rnn.c rnn_lstm.o
	$(CC) $(CFLAGS) $^ -o $@

example_transformer: example_transformer.c transformer_arch.o
	$(CC) $(CFLAGS) $^ -o $@

demo_gan: demo_gan.c gan_model.o
	$(CC) $(CFLAGS) $^ -o $@

demo_diffusion: demo_diffusion.c diffusion_model.o
	$(CC) $(CFLAGS) $^ -o $@

clean:
	del /f *.o $(EXAMPLES:.exe=) $(DEMOS:.exe=) 2>nul || rm -f *.o $(EXAMPLES) $(DEMOS)
