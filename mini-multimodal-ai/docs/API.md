# mini-multimodal-ai API Reference

## CLIP Contrastive Learning (`clip_contrastive.h`)

### Types
- `mm_linear_t` — Linear layer (weight, bias, in_dim, out_dim)
- `mm_self_attn_t` — Multi-head self-attention (Q/K/V/O projections)
- `mm_transformer_block_t` — Transformer block (self-attn + FFN + LayerNorm)
- `mm_text_encoder_t` — Text encoder (token embed, positional, N blocks, projection)
- `mm_image_encoder_t` — Image encoder (patch embed, class token, positional, N blocks)
- `mm_clip_model_t` — Full CLIP model (image encoder + text encoder + logit scale)
- `mm_embedding_t` — L2-normalized embedding vector
- `mm_clip_pair_t` — Image-text pair with similarity score

### Core Functions
| Function | Description |
|----------|-------------|
| `mm_l2_norm(x, n)` | Compute L2 norm of vector x |
| `mm_l2_normalize(x, n)` | Normalize vector x in-place |
| `mm_cosine_sim_matrix(img, txt, B, D, sim)` | B×B cosine similarity matrix |
| `mm_infonce_loss(sim, B)` | Symmetric InfoNCE contrastive loss |
| `mm_infonce_grad(sim, B, ig, tg)` | Gradient of InfoNCE loss |

### Text Processing
| Function | Description |
|----------|-------------|
| `mm_text_tokenize(text, ids, n)` | Tokenize text string to token IDs |
| `mm_text_encode(enc, ids, n, emb)` | Encode tokens to embedding |

### Image Processing
| Function | Description |
|----------|-------------|
| `mm_image_patchify(img, h, w, c, ps, patches)` | Convert image to patches |
| `mm_image_encode(enc, img, h, w, c, emb)` | Encode image to embedding |

### Model Interface
| Function | Description |
|----------|-------------|
| `mm_clip_init(model, dim, layers)` | Initialize CLIP model |
| `mm_clip_free(model)` | Free CLIP model |
| `mm_clip_encode_image(model, img, h, w, c, emb)` | Encode image through CLIP |
| `mm_clip_encode_text(model, text, emb)` | Encode text through CLIP |
| `mm_clip_zeroshot(img_emb, classes, N, top_k)` | Zero-shot classification |
| `mm_clip_retrieve(q, gallery, N, D, indices, K)` | Image-text retrieval |
| `mm_clip_train_step(model, imgs, texts, B, h, w, c, lr)` | Single training step |

### Building Blocks
| Function | Description |
|----------|-------------|
| `mm_linear_init/forward/free` | Linear layer |
| `mm_self_attn_init/forward/free` | Self-attention |
| `mm_transformer_block_init/forward/free` | Transformer block |
| `mm_layernorm(x, w, b, n, out)` | Layer normalization |
| `mm_softmax(x, n)` | Softmax in-place |
| `mm_gelu(x)` | GELU activation |

---

## Image Generation (`image_generation.h`)

### Types
- `mm_conv2d_t` — 2D convolution
- `mm_groupnorm_t` — Group normalization
- `mm_resblock_t` — Residual block with time embedding
- `mm_cross_attn_t` — Cross-attention (Q from latent, K/V from text)
- `mm_spatial_transformer_t` — Spatial transformer (groupnorm + conv + cross-attn)
- `mm_unet_down/mid/up_block_t` — UNet block types
- `mm_unet_t` — Full UNet denoiser
- `mm_vae_t` — Variational autoencoder
- `mm_stable_diffusion_t` — Full SD pipeline
- `mm_schedule_t` — Diffusion schedule

### Enums
- `mm_schedule_type_t` — LINEAR, COSINE, SQRT
- `mm_sd_sampler_t` — DDIM, DPM_PP_2M, EULER_A

### Core Functions
| Function | Description |
|----------|-------------|
| `mm_conv2d_init/forward/free` | 2D convolution |
| `mm_groupnorm_init/forward/free` | Group normalization |
| `mm_resblock_init/forward/free` | Residual block |
| `mm_cross_attn_init/forward/free` | Cross-attention |
| `mm_spatial_transformer_init/forward/free` | Spatial transformer |
| `mm_unet_init/forward/free` | UNet denoiser |
| `mm_vae_init/encode/decode/free` | VAE encode/decode |
| `mm_vae_sample(mean, logvar, n, latent)` | Sample latent from distribution |

### Diffusion
| Function | Description |
|----------|-------------|
| `mm_sd_init/free` | Initialize SD pipeline |
| `mm_sd_beta_schedule(s, N, start, end, type)` | Create noise schedule |
| `mm_sd_alphas_from_betas(betas, alphas, cp)` | Compute alphas/alpha_cumprod |
| `mm_sd_add_noise(x, noise, sa, so, n, out)` | Add noise to data |
| `mm_sd_predict_noise(sd, latent, t, ctx, ctx_len, unc, cond)` | UNet noise prediction |
| `mm_sd_ddim_step(xt, np, t, tp, ac, n, eta, xtp)` | DDIM sampling step |
| `mm_sd_dpm_pp_2m_step(...)` | DPM-Solver++ 2M step |
| `mm_sd_cfg_guidance(cond, uncond, scale, n)` | CFG guidance |
| `mm_sd_generate(sd, ctx, len, steps, sampler, h, w, c, img)` | Full generation |
| `mm_sd_inpaint(sd, img, mask, ctx, len, h, w, c, steps, out)` | Inpainting |
| `mm_sd_pipe_encode/decode` | Encode image / decode latent |

---

## Vision-Language Model (`vlm_llama.h`)

### Types
- `mm_vlm_token_seq_t` — Token sequence
- `mm_vlm_bbox_t` — Bounding box (x, y, w, h)
- `mm_vlm_msg_role_t` — Message role (USER, ASSISTANT, SYSTEM)
- `mm_vlm_message_t` — Conversation message (text + optional image + bbox)
- `mm_vlm_conversation_t` — Multi-turn conversation
- `mm_vlm_linear/attention/ffn/decoder_layer/llm_t` — LLM building blocks
- `mm_vlm_projection_t` — Vision-to-LLM projection
- `mm_vlm_model_t` — Full LLaVA model

### Core Functions
| Function | Description |
|----------|-------------|
| `mm_vlm_model_init/free` | Initialize/free LLaVA model |
| `mm_vlm_encode_image(model, img, h, w, c, feat)` | Vision encoding |
| `mm_vlm_project_features(proj, vfeat, N, pfeat)` | Project to LLM space |
| `mm_vlm_prepare_input(model, conv, ifeat, ids, n, pos)` | Prepare input tokens |
| `mm_vlm_generate_token(model, ids, len, pos)` | Generate single token |
| `mm_vlm_generate(model, ids, len, max, out, n)` | Autoregressive generation |
| `mm_vlm_decode_token/decode` | Decode tokens to text |

### Tasks
| Function | Description |
|----------|-------------|
| `mm_vlm_visual_qa(model, img, h, w, c, q, a, cap)` | Visual question answering |
| `mm_vlm_ocr(model, img, h, w, c, text, cap)` | OCR from image |
| `mm_vlm_region_understand(model, img, h, w, c, bbox, q, a, cap)` | Region understanding |
| `mm_vlm_multiturn(model, conv, img, h, w, c, resp, cap)` | Multi-turn conversation |

---

## Audio Understanding (`audio_whisper.h`)

### Types
- `mm_whisper_task_t` — TRANSCRIBE, TRANSLATE, LANG_DETECT
- `mm_whisper_special_token_t` — SOT, EOT, language, task, time tokens
- `mm_whisper_segment_t` — Transcription segment (time, text, probability)
- `mm_whisper_result_t` — Full transcription result
- `mm_mel_spectrogram_t` — Mel spectrogram data
- `mm_mel_filterbank_t` — Mel filterbank
- `mm_whisper_model_t` — Full Whisper model

### Audio Processing
| Function | Description |
|----------|-------------|
| `mm_mel_filterbank_init/free` | Initialize mel filterbank |
| `mm_audio_stft(audio, len, nfft, hop, real, imag, nf, nfreq)` | Short-Time Fourier Transform |
| `mm_audio_mel_spectrogram(audio, len, fbank, mel)` | Compute mel spectrogram |
| `mm_mel_spectrogram_free` | Free mel spectrogram |

### VAD
| Function | Description |
|----------|-------------|
| `mm_vad_energy(audio, len)` | Compute frame energy |
| `mm_vad_is_speech(audio, len, thresh)` | Detect if speech is present |
| `mm_vad_split(audio, len, chunk_ms, thresh, starts, ends, n)` | Split audio into speech segments |

### Transcription
| Function | Description |
|----------|-------------|
| `mm_whisper_model_init/free` | Initialize Whisper model |
| `mm_whisper_encoder_forward(enc, mel, hidden)` | Encode audio to hidden states |
| `mm_whisper_decoder_forward(dec, enc_h, elen, toks, n, logits)` | Decode next token |
| `mm_whisper_transcribe(model, audio, len, lang, task, result)` | Full transcription |
| `mm_whisper_translate(model, audio, len, result)` | Translate to English |
| `mm_whisper_detect_language(model, audio, len, lang, prob)` | Detect language |

---

## Video Understanding (`video_understanding.h`)

### Types
- `mm_video_clip_t` — Video clip (frames, height, width, channels)
- `mm_conv3d_t` — 3D convolution
- `mm_batchnorm3d_t` — 3D batch normalization
- `mm_c3d_block_t` — C3D block (conv + bn + relu + pool)
- `mm_c3d_model_t` — Full C3D model
- `mm_timesformer_block_t` — Divided spatial-temporal attention block
- `mm_timesformer_t` — Full TimeSformer model
- `mm_clip4clip_t` — CLIP4Clip video-text model
- `mm_action_segment_t` — Temporal action segment
- `mm_video_arch_t` — C3D, TIMESFORMER, CLIP4CLIP
- `mm_video_model_t` — Unified video model (union of architectures)

### Data Processing
| Function | Description |
|----------|-------------|
| `mm_video_clip_init/free` | Allocate/free video clip |
| `mm_video_sample_frames(video, nf, h, w, c, ns, clip)` | Uniform frame sampling |
| `mm_video_uniform_sample(video, nf, h, w, c, ns, idx)` | Get frame indices |

### C3D
| Function | Description |
|----------|-------------|
| `mm_conv3d_init/forward/free` | 3D convolution |
| `mm_batchnorm3d_init/forward/free` | 3D batch norm |
| `mm_c3d_block_init/forward/free` | C3D block |
| `mm_c3d_init/forward/free` | C3D model |

### TimeSformer
| Function | Description |
|----------|-------------|
| `mm_timesformer_init/forward/free` | TimeSformer model |
| `mm_timesformer_block_init/forward/free` | TimeSformer block |

### CLIP4Clip
| Function | Description |
|----------|-------------|
| `mm_clip4clip_init/free` | CLIP4Clip model |
| `mm_clip4clip_encode_video/encode_text` | Encode video/text |
| `mm_clip4clip_retrieve(vemb, tembs, N, D, idx, K)` | Video-text retrieval |

### Tasks
| Function | Description |
|----------|-------------|
| `mm_video_model_init/free` | Initialize video model (any architecture) |
| `mm_video_action_recognition(model, clip, name, cap)` | Classify action |
| `mm_video_action_recognition_topk(model, clip, k, ids, conf)` | Top-K classification |
| `mm_video_temporal_localize(model, video, nf, h, w, c, win, str, segs)` | Temporal action localization |
| `mm_video_caption(model, clip, cap, cap_cap)` | Video captioning |
| `mm_video_text_retrieval(model, clip, queries, N, sims)` | Video-text similarity |
| `mm_iou_3d(s1, e1, s2, e2)` | 3D temporal IoU |
| `mm_nms_3d(segs, n, thresh, keep, nkeep)` | 3D non-maximum suppression |
| `mm_video_kinetics_labels()` | Get Kinetics-400 class names |
| `mm_video_num_kinetics_labels()` | Count Kinetics labels |

---

## Common Patterns

All modules follow consistent patterns:

```c
// 1. Initialize
mm_module_t obj;
mm_module_init(&obj, ...);

// 2. Use (encode/decode/predict)
mm_module_forward(&obj, input, output);

// 3. Free
mm_module_free(&obj);
```

Naming convention: `mm_{module}_{operation}` where module identifies the component
and operation describes the action (init, free, forward, encode, decode, etc.).
