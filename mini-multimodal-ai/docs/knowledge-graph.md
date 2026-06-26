# Knowledge Graph — mini-multimodal-ai

## L1: Core Definitions [COMPLETE]
- `mm_linear_t` — Linear layer (weight, bias, in_dim, out_dim)
- `mm_self_attn_t` — Multi-head self-attention (Q/K/V projections, output projection)
- `mm_transformer_block_t` — Transformer block (self-attn + FFN + LayerNorm)
- `mm_text_encoder_t` — CLIP text encoder (token embedding, positional encoding, transformer blocks)
- `mm_image_encoder_t` — CLIP image encoder (patch projection, class token, positional encoding)
- `mm_clip_model_t` — CLIP model (image encoder + text encoder + temperature)
- `mm_conv2d_t` — 2D convolution (weight, bias, kernel_size, stride, padding, groups)
- `mm_groupnorm_t` — Group Normalization (weight, bias, num_groups, num_ch)
- `mm_resblock_t` — Residual block (conv1, conv2, norm1, norm2, time embedding)
- `mm_cross_attn_t` — Cross-attention (Q from query, K/V from context)
- `mm_spatial_transformer_t` — Spatial transformer (GroupNorm + Conv + CrossAttn + Conv)
- `mm_unet_down_block_t`, `_mid_block_t`, `_up_block_t` — UNet components
- `mm_unet_t` — U-Net denoiser (encoder-decoder with skip connections)
- `mm_vae_t` — Variational Autoencoder (encoder + decoder)
- `mm_stable_diffusion_t` — Stable Diffusion model (VAE + UNet + schedule)
- `mm_vlm_token_seq_t` — Token sequence for LLM
- `mm_vlm_conversation_t` — Multi-turn conversation
- `mm_vlm_attention_t` — VLM attention with RoPE
- `mm_vlm_ffn_t` — VLM FFN (gate + up + down projections, SwiGLU)
- `mm_vlm_decoder_layer_t` — VLM decoder layer
- `mm_vlm_llm_t` — VLM language model
- `mm_vlm_model_t` — Full VLM (vision + projector + LLM)
- `mm_mel_spectrogram_t` — Mel spectrogram representation
- `mm_mel_filterbank_t` — Mel filterbank (triangular filters)
- `mm_whisper_attn_t`, `_cross_attn_t`, `_ffn_t` — Whisper components
- `mm_whisper_encoder_layer_t`, `_decoder_layer_t` — Whisper layers
- `mm_whisper_encoder_t`, `_decoder_t`, `_model_t` — Whisper model
- `mm_video_clip_t` — Video clip container
- `mm_conv3d_t` — 3D convolution
- `mm_batchnorm3d_t` — 3D Batch Normalization
- `mm_c3d_block_t`, `_model_t` — C3D network
- `mm_ts_attention_t` — TimeSformer attention
- `mm_timesformer_block_t`, `_t` — TimeSformer architecture
- `mm_clip4clip_t` — CLIP4Clip video-text model
- `mm_video_model_t` — Unified video model (C3D/TimeSformer/CLIP4Clip)

## L2: Core Concepts [COMPLETE]
- Contrastive Learning: InfoNCE loss, temperature scaling, cosine similarity
- Latent Diffusion Models: forward process, reverse process, noise prediction
- Classifier-Free Guidance: conditional/unconditional noise prediction blending
- Vision-Language Models: vision encoder → projection → LLM pipeline
- Audio Processing: STFT, Mel scale, log-mel spectrogram, VAD
- 3D Video Understanding: spatiotemporal convolutions, frame sampling

## L3: Engineering Structures [COMPLETE]
- Pre-norm Transformer with residual connections
- UNet encoder-decoder with skip connections and spatial transformers
- RMSNorm-based LLM with SwiGLU activation
- Mel filterbank with triangular overlapping filters
- C3D pipeline: 3D Conv → BatchNorm → ReLU → MaxPool
- TimeSformer: divided space-time attention

## L4: Standards/Theorems [COMPLETE]
- InfoNCE (van den Oord et al., 2018): `mm_infonce_loss()`, `mm_infonce_grad()`
- DDIM (Song et al., 2021): `mm_sd_ddim_step()`
- Classifier-Free Guidance (Ho & Salimans, 2022): `mm_sd_cfg_guidance()`
- DPM-Solver++ (Lu et al., 2022): `mm_sd_dpm_pp_2m_step()`
- RoPE (Su et al., 2021): `mm_vlm_rope_forward()`
- Group Normalization (Wu & He, 2018): `mm_groupnorm_forward()`
- GELU Activation (Hendrycks & Gimpel, 2016): `mm_gelu()`
- SiLU/Swish Activation (Ramachandran et al., 2017): `mm_vlm_silu()`

## L5: Algorithms/Methods [COMPLETE]
- Multi-head self-attention: O(n²d) — `mm_self_attn_forward()`
- Cross-attention: O(n·m·d) — `mm_cross_attn_forward()`
- 2D Convolution: O(HW·K²·C_in·C_out) — `mm_conv2d_forward()`
- 3D Convolution: O(THW·K³·C_in·C_out) — `mm_conv3d_forward()`
- Mel Spectrogram (via STFT): O(T·N_fft) — `mm_audio_mel_spectrogram()`
- Top-K / Nucleus Sampling: O(V log V) — `mm_vlm_sampler()`
- Nearest-neighbor 2x upsampling: O(N) — (in `mm_unet_forward()`)
- 3D Non-Maximum Suppression: O(N²) — `mm_nms_3d()`
- 3D IoU (temporal): O(1) — `mm_iou_3d()`

## L6: Canonical Problems [COMPLETE]
- Zero-shot image classification (CLIP): `example_clip.c`
- Text-to-image generation (SD): `example_sd.c`
- Visual Question Answering (LLaVA): `example_llava.c`
- Speech-to-text transcription (Whisper): `demo_multimodal.c`
- Action recognition (C3D/TimeSformer): `demo_multimodal.c`

## L7: Applications [COMPLETE — 5 apps]
1. Image-text retrieval: `mm_clip_retrieve()` + `example_clip.c`
2. Image inpainting: `mm_sd_inpaint()` + `example_sd.c`
3. Video-text retrieval: `mm_clip4clip_retrieve()` + `demo_multimodal.c`
4. Language detection: `mm_whisper_detect_language()` + `demo_multimodal.c`
5. Cross-modal pipeline: `demo_pipeline.c`

## L8: Advanced Topics [PARTIAL — 3/5]
✅ VAE latent sampling (reparameterization)
✅ DDIM deterministic inversion
✅ DPM-Solver++ 2M accelerated sampling
⬜ FlashAttention
⬜ Speculative decoding

## L9: Industry Frontiers [PARTIAL — documented only]
⬜ Multi-modal chain-of-thought
⬜ RLHF training pipeline
⬜ AI Compiler (MLIR/Triton) integration
