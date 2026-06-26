# Prerequisite Dependency Tree — mini-multimodal-ai

## Knowledge Dependencies

```
Linear Algebra → Matrix Multiply → Linear Layer (mm_linear_forward)
                                    ↓
Softmax + Scaled Dot-Product → Self-Attention
                                    ↓
LayerNorm / RMSNorm + Residual → Transformer Block
                                    ↓
                            ┌───────┼────────┐
                            ↓       ↓        ↓
                      CLIP Encoder  LLM     Whisper Enc/Dec
                            ↓       ↓        ↓
                      Contrastive  VLM      ASR
                            ↓
                    ┌───────┼────────┐
                    ↓       ↓        ↓
             Conv2D   GroupNorm   Cross-Attn
                    ↓       ↓        ↓
                 ResBlock → SpatialTransformer
                    ↓       ↓        ↓
                      UNet Encoder-Decoder
                            ↓
                    Latent Diffusion (SD)
                            ↓
                      VAE Encode/Decode

3D Conv → BatchNorm3D → C3D Block → C3D Model
                                          ↓
                                    Video Understanding
                                          ↓
                                   Temporal Localization

STFT → Mel Filterbank → Mel Spectrogram → Whisper Encoder
                                              ↓
                                        ASR Pipeline
```

## Module Dependency Graph

```
clip_contrastive.h  ←── image_generation.h (mm_linear_t, mm_gelu_forward)
                    ←── vlm_llama.h        (mm_image_encoder_t)
                    ←── audio_whisper.h    (mm_linear_t)
                    ←── video_understanding.h (mm_linear_t, mm_image_encoder_t)

image_generation.h  ←── audio_whisper.h    (mm_conv2d_t)
```

## Build Order
1. `clip_contrastive.c` (no module dependencies)
2. `image_generation.c` (depends on clip_contrastive.h)
3. `vlm_llama.c` (depends on clip_contrastive.h)
4. `audio_whisper.c` (depends on clip_contrastive.h + image_generation.h)
5. `video_understanding.c` (depends on clip_contrastive.h)
