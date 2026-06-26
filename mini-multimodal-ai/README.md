# mini-multimodal-ai — 多模态AI (C 语言实现)

A minimal C99 library implementing core multimodal AI architectures:
CLIP contrastive learning, image generation (Stable Diffusion/VAE), vision-language models (LLaVA),
automatic speech recognition (Whisper), and video understanding.

## Modules

| Module              | Header                  | Description                                 |
|---------------------|-------------------------|---------------------------------------------|
| CLIP Contrastive    | `clip_contrastive.h`    | Dual encoder, InfoNCE loss, zero-shot cls   |
| Image Generation    | `image_generation.h`    | VAE, latent diffusion, UNet, DDIM/DPM       |
| VLM (LLaVA)         | `vlm_llama.h`           | Vision encoder + projection + LLM           |
| Audio (Whisper)     | `audio_whisper.h`       | Encoder-decoder, log-mel, multi-lang ASR    |
| Video Understanding | `video_understanding.h` | 3D CNN, TimeSformer, action recognition     |

## Build

```sh
make          # build library and all demos
make test     # run example programs
make clean    # remove build artifacts
```

## Quick Start

```c
#include "clip_contrastive.h"

// Zero-shot image classification
const char* classes[] = {"cat", "dog", "bird"};
int pred = mm_clip_zeroshot(image_embedding, classes, 3);
```

## Dependencies

- C99 compiler (GCC 9+, Clang 10+, MSVC 2019+)
- `math.h` (standard library)

## License

MIT
