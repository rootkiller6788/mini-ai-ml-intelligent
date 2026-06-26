# mini-multimodal-ai — Multimodal AI (C99)

A minimal C99 library implementing core multimodal AI architectures:
CLIP contrastive learning, image generation (Stable Diffusion/VAE),
vision-language models (LLaVA), automatic speech recognition (Whisper),
and video understanding.

## Module Status: COMPLETE ✅

- **Code lines** (include/ + src/): 5,141 ≥ 3,000 ✓
- **L1-L6**: Complete
- **L7**: Complete (5 application demos)
- **L8**: Partial (3/5 advanced topics implemented)
- **L9**: Partial (documented, not implemented)

## Modules

| Module              | Header                  | Description                                 |
|---------------------|-------------------------|---------------------------------------------|
| CLIP Contrastive    | `clip_contrastive.h`    | Dual encoder, InfoNCE loss, zero-shot cls   |
| Image Generation    | `image_generation.h`    | VAE, latent diffusion, UNet, DDIM/DPM       |
| VLM (LLaVA)         | `vlm_llama.h`           | Vision encoder + projection + LLM           |
| Audio (Whisper)     | `audio_whisper.h`       | Encoder-decoder, log-mel, multi-lang ASR    |
| Video Understanding | `video_understanding.h` | 3D CNN, TimeSformer, action recognition     |

## Knowledge Coverage (9 Layers)

### L1 — Core Definitions
- CLIP: `mm_linear_t`, `mm_self_attn_t`, `mm_transformer_block_t`, `mm_clip_model_t`
- Image Gen: `mm_conv2d_t`, `mm_groupnorm_t`, `mm_unet_t`, `mm_vae_t`, `mm_stable_diffusion_t`
- VLM: `mm_vlm_llm_t`, `mm_vlm_attention_t`, `mm_vlm_ffn_t`, `mm_vlm_model_t`
- Audio: `mm_mel_spectrogram_t`, `mm_whisper_encoder_t`, `mm_whisper_model_t`
- Video: `mm_video_clip_t`, `mm_conv3d_t`, `mm_timesformer_t`, `mm_video_model_t`

### L2 — Core Concepts
- Contrastive Learning: InfoNCE loss, cosine similarity, zero-shot classification
- Latent Diffusion: Forward/reverse diffusion, noise schedule, classifier-free guidance
- Vision-Language: Cross-modal projection, multi-turn conversation, visual QA
- Audio Processing: Mel spectrogram, STFT, VAD, encoder-decoder ASR
- Video Understanding: 3D convolution, TimeSformer, CLIP4Clip, temporal localization

### L3 — Engineering Structures
- Transformer blocks with pre-norm residual connections
- UNet encoder-decoder with skip connections and spatial transformers
- RMSNorm-based LLM decoder layers with RoPE
- Mel filterbank with triangular overlapping filters
- C3D pipeline: 3D Conv → BN → ReLU → MaxPool

### L4 — Standards / Theorems
- **InfoNCE Loss** (van den Oord et al., 2018): contrastive representation learning
- **DDIM Sampling** (Song et al., 2021): deterministic diffusion inversion
- **Classifier-Free Guidance** (Ho & Salimans, 2022)
- **DPM-Solver++ 2M** (Lu et al., 2022): fast ODE-based sampling
- **RoPE** (Su et al., 2021): rotary position embeddings
- **Group Normalization** (Wu & He, 2018)
- **SiLU/GELU** activation functions

### L5 — Algorithms / Methods
- Multi-head self-attention with scaled dot-product
- Cross-attention for text-conditioned generation
- Top-K / nucleus (top-P) sampling for autoregressive decoding
- Uniform frame sampling for video clips
- 3D Non-Maximum Suppression (tIoU-based)
- Nearest-neighbor 2x upsampling (UNet decoder)

### L6 — Canonical Problems
- Zero-shot image classification (CLIP)
- Text-to-image generation (Stable Diffusion → UNet + DDIM)
- Visual question answering (LLaVA)
- Speech-to-text transcription (Whisper)
- Action recognition from video (C3D, TimeSformer)

### L7 — Applications (≥2, Complete ✓)
1. Image-text retrieval with cosine similarity ranking
2. Inpainting: mask-guided latent diffusion
3. Video-text retrieval (CLIP4Clip)
4. Language detection from audio (Whisper)
5. Cross-modal pipeline (demo_multimodal.c)

### L8 — Advanced Topics (Partial)
- ✅ VAE latent space sampling (reparameterization trick)
- ✅ DDIM deterministic inversion
- ✅ DPM-Solver++ accelerated sampling
- ⬜ FlashAttention (memory-efficient attention)
- ⬜ Speculative decoding

### L9 — Industry Frontiers (Partial, documented only)
- ⬜ Multi-modal chain-of-thought reasoning
- ⬜ RLHF (Reinforcement Learning from Human Feedback)
- ⬜ AI Compiler integration (MLIR/Triton)
- ✅ Documented in `docs/`

## Core Definitions

```c
// CLIP dual encoder
typedef struct { mm_image_encoder_t ie; mm_text_encoder_t te; } mm_clip_model_t;

// Stable Diffusion
typedef struct { mm_vae_t vae; mm_unet_t unet; } mm_stable_diffusion_t;

// LLaVA VLM
typedef struct { mm_image_encoder_t vision; mm_vlm_llm_t llm; } mm_vlm_model_t;

// Whisper ASR
typedef struct { mm_whisper_encoder_t enc; mm_whisper_decoder_t dec; } mm_whisper_model_t;
```

## Core Theorems (with formulas)

| Theorem | Formula | Implementation |
|---------|---------|----------------|
| InfoNCE Loss | L = -log(exp(sim/τ) / Σ exp(sim/τ)) | `mm_infonce_loss()` |
| DDIM Step | x_{t-1} = √ᾱ_{t-1}·(x_t - √(1-ᾱ_t)·ε)/√ᾱ_t + √(1-ᾱ_{t-1})·ε | `mm_sd_ddim_step()` |
| CFG | ε̂ = ε_u + w·(ε_c - ε_u) | `mm_sd_cfg_guidance()` |
| RoPE | f(q,k,m) = R^d_{Θ,m}·W_q·x | `mm_vlm_rope_forward()` |
| L2 Norm | ‖x‖₂ = √(Σx_i²) | `mm_l2_norm()` |

## Core Algorithms

| Algorithm | Complexity | Function |
|-----------|------------|----------|
| Multi-head Self-Attention | O(n²d) | `mm_self_attn_forward()` |
| Cross-Attention | O(n·m·d) | `mm_cross_attn_forward()` |
| 2D Convolution (im2col-free) | O(HW·K²·C_in·C_out) | `mm_conv2d_forward()` |
| Mel Spectrogram (STFT) | O(T·N_fft·log N_fft) | `mm_audio_mel_spectrogram()` |
| 3D Convolution | O(THW·K³·C_in·C_out) | `mm_conv3d_forward()` |

## 9-School Curriculum Mapping

| School | Course | Module Coverage |
|--------|--------|-----------------|
| **MIT** | 6.858 (Security) | — |
| **Stanford** | CS 229 (ML), CS 231n (CNN) | CLIP, Video, LLaVA |
| **Berkeley** | CS 294 (AI Systems) | SD/VAE, Whisper |
| **CMU** | 15-418 (Parallel), 11-785 (DL) | C3D, TimeSformer |
| **UT Austin** | CS 395T (Systems ML) | SD Pipeline |
| **ETH** | 263-3501 (Parallel) | UNet Architecture |
| **Cambridge** | Part II: ML | All modules |
| **清华** | 计算机视觉, 自然语言处理 | CLIP, VLM |
| **Georgia Tech** | CS 7641 (ML) | Contrastive Learning |

## Build

```sh
make          # build library and all demos
make test     # run unit tests + example programs (15 tests + 3 demos)
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
