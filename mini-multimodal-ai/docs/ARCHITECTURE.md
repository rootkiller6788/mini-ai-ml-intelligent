# mini-multimodal-ai Architecture

## Overview

The library implements five core multimodal AI architectures in pure C99, designed
for educational use and embedded deployment. Each module is self-contained with
its own header/source pair and can be used independently.

## Module Architecture

### 1. CLIP Contrastive Learning (`clip_contrastive.h/.c`)

```
Input Image ──► Image Encoder (ViT) ──► L2 Norm ──┐
                                                    ├──► Cosine Similarity ──► InfoNCE Loss
Input Text  ──► Text Encoder (Transformer) ──► L2 Norm ──┘
```

- **Image Encoder**: Vision Transformer (ViT) with patch embedding, positional
  encoding, multi-head self-attention, and feed-forward layers.
- **Text Encoder**: Transformer encoder with token embedding, positional encoding,
  self-attention, and FFN.
- **Loss**: Symmetric InfoNCE (NT-Xent) contrastive loss operating on image-text
  pairs within a batch.
- **Zero-shot**: Class name is formatted as "a photo of {class}", encoded, and
  compared via cosine similarity to the image embedding.

### 2. Image Generation (`image_generation.h/.c`)

```
Text ──► Text Embedding ──┐
                          ├──► UNet (cross-attention) ──► Denoised Latent ──► VAE Decoder ──► Image
Noise ──────────────────┘
```

- **VAE**: Encoder compresses 3-channel image to 4-channel latent (8x downscale).
  Decoder reconstructs. Gaussian sampling with reparameterization.
- **UNet**: Down/mid/up blocks with ResBlocks, SpatialTransformers (cross-attention),
  and time embedding (sinusoidal).
- **Diffusion**: Linear/cosine/sqrt beta schedules. DDIM and DPM-Solver++ 2M samplers.
- **CFG**: Classifier-free guidance combines conditional and unconditional predictions.
- **Inpainting**: Mask-based latent blending during denoising.

### 3. Vision-Language Model (`vlm_llama.h/.c`)

```
Image ──► ViT Encoder ──► Projection (MLP) ──┐
                                              ├──► LLM (Transformer Decoder) ──► Text Response
Text Tokens ────────────────────────────────┘
```

- **LLaVA architecture**: Vision features projected into LLM embedding space.
  Image tokens interleaved with text tokens.
- **LLM**: Decoder-only transformer with RMSNorm, RoPE (rotary position encoding),
  SiLU-gated FFN, and multi-head self-attention with KV cache.
- **Capabilities**: Visual QA, OCR, region understanding (bounding box), multi-turn
  conversation with images.

### 4. Audio Understanding (`audio_whisper.h/.c`)

```
Audio ──► STFT ──► Mel Filterbank ──► Encoder ──► Cross-Attention ──► Decoder ──► Text
```

- **Preprocessing**: STFT with Hann window, mel filterbank (triangular filters),
  log-mel spectrogram with normalization.
- **Encoder**: 2x Conv2D (stride 2 for 4x temporal reduction) + transformer encoder
  layers with self-attention.
- **Decoder**: Transformer decoder with self-attention, cross-attention to encoder
  output, and FFN layers. Autoregressive token generation.
- **VAD**: Energy-based voice activity detection with zero-crossing rate.
- **Tasks**: Transcribe, translate, language detection.

### 5. Video Understanding (`video_understanding.h/.c`)

```
Video ──► Frame Sampling ──► 3D CNN / TimeSformer / CLIP4Clip ──► Classification / Retrieval
```

- **C3D**: 3D convolutions (time+space) with batch normalization, ReLU, and max-pooling.
  Fully connected classifier head.
- **TimeSformer**: Divided spatial-temporal attention — spatial attention within frames,
  temporal attention across frames. Patch embedding + positional + temporal encoding.
- **CLIP4Clip**: Frame-level image encoder (CLIP ViT) → mean pooling → temporal projection.
  Video-text cosine similarity for retrieval.
- **Tasks**: Action recognition (top-K), temporal localization (sliding window + NMS),
  video captioning, video-text retrieval.

## Data Flow

```
                          ┌──────────────────────┐
                          │   mini-multimodal-ai   │
                          └──────────┬───────────┘
                                     │
        ┌────────────┬───────────────┼───────────────┬────────────┐
        ▼            ▼               ▼               ▼            ▼
   ┌────────┐  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌─────────┐
   │  CLIP  │  │   SD     │  │   LLaVA   │  │  Whisper │  │  Video  │
   │ Image+ │  │  Image   │  │  Image+   │  │  Audio   │  │  Video  │
   │ Text   │  │  Gen     │  │  Text     │  │  ->Text  │  │  Tasks  │
   └────────┘  └──────────┘  └───────────┘  └──────────┘  └─────────┘
```

## Memory Management

All modules use manual memory management with init/free function pairs:
- `mm_*_init()` — allocates and initializes weights with Kaiming/Xavier initialization.
- `mm_*_free()` — recursively frees all allocated memory.
- Intermediate buffers use `malloc`/`free` with no persistent leaks.

## Numerical Precision

- All computations in float32.
- Stable softmax with max-subtraction for numerical stability.
- RMSNorm and LayerNorm with epsilon=1e-5 for numerical stability.
- Log-sum-exp trick used throughout for stable computation.

## Dependencies

- C99 standard library: `<stdlib.h>`, `<string.h>`, `<math.h>`, `<stdio.h>`
- No external dependencies beyond libm.
- Optional OpenMP for parallel loops (configured in Makefile).
