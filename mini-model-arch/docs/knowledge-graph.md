# Knowledge Graph — mini-model-arch

## L1: Definitions (Complete ✓)
- Tensor4D, Conv2D, Pool2D, Linear
- LeNet5, VGGBlock, ResBlock, BottleneckBlock, InceptionModule, DepthwiseSepConv
- RNNCell, LSTMCell, GRUCell, BiRNN, StackedRNN
- Matrix2D, MultiHeadAttn, PositionEncoding, FeedForward, LayerNorm
- EncoderBlock, DecoderBlock, TransformerEncoder, TransformerDecoder
- Generator, Discriminator, GAN, DCGANNormalizer
- DiffusionSchedule, UNet, DiffusionModel
- Optimizer, OptimizerType, LRSchedule, LRScheduleType, LAMBState
- BatchNorm1D, BatchNorm2D, RMSNorm, GroupNorm
- Dropout, EarlyStopping
- RoPE, ALiBi

## L2: Core Concepts (Complete ✓)
- Convolution (Conv2D, Depthwise Separable)
- Pooling (Max/Avg)
- Residual Connections (ResBlock, Bottleneck)
- Inception Multi-branch Architecture
- RNN recurrence, LSTM gating (forget/input/output/cell)
- GRU gating (reset/update)
- Bidirectional RNN
- Self-Attention / Multi-Head Attention
- Position Encoding (sinusoidal, learnable)
- Feed-Forward Network with ReLU
- Layer Normalization
- Pre-LN vs Post-LN Transformer
- Generative Adversarial Network
- Diffusion (forward process, reverse/sampling)
- Adam / SGD with Momentum / Nesterov
- Batch Normalization
- Dropout regularization

## L3: Engineering Structures (Complete ✓)
- Stacked RNN (multi-layer sequence model)
- Transformer Encoder stack
- Transformer Decoder stack (self-attn + cross-attn + FFN)
- GPT (decoder-only) pipeline
- BERT (encoder-only) pipeline
- DCGAN architecture pattern
- DDPM reverse process (iterative denoising)
- DDIM accelerated sampling
- SwiGLU FFN structure (3-weight-matrix design)

## L4: Standards/Theorems (Complete ✓)
- Adam convergence: O(1/sqrt(T)) regret bound
- Cross-Entropy = MLE for categorical distributions
- KL Divergence: D_KL(P||Q) >= 0 (Gibbs' inequality)
- Dropout: approximates geometric model averaging of 2^n networks
- BatchNorm: reduces internal covariate shift
- RoPE Theorem: (R_m q)^T(R_n k) = q^T R_{n-m} k
- ALiBi: linear bias enables O(L_train) -> O(L_test) extrapolation
- AdamW: decoupled weight decay superior to L2 regularization
- Inverted Dropout: scale at train, identity at test
- Cosine LR Schedule: smooth annealing from base to min

## L5: Algorithms/Methods (Complete ✓)
- SGD with Momentum / Nesterov Accelerated Gradient
- Adam optimizer with bias correction
- AdamW with decoupled weight decay
- RMSProp adaptive learning rates
- LAMB layer-wise adaptive moments
- Gradient clipping (norm and value)
- Cosine / Warmup-Cosine / Step / Exponential LR schedules
- Cross-Entropy with softmax numerical stability
- Focal Loss (class imbalance)
- Huber Loss (smooth L1, robust to outliers)
- Conv2D forward (naive im2col-free)
- Pool2D forward (max/avg)
- LeNet5 forward pipeline
- ResBlock forward with identity/projection shortcuts
- Inception forward (4-branch concat)
- RNN/LSTM/GRU forward (stepwise recurrence)
- Multi-Head Self-Attention (QKV projection + scaled dot-product)
- DDPM sampling: x_{t-1} = f(x_t, eps_t, t)
- DDIM sampling: deterministic with eta interpolation
- GAN training steps (D-step, G-step, WGAN-GP)
- RoPE application (pairwise 2D rotation)
- Dropout forward (inverted)
- DropConnect forward
- SwiGLU forward (silu-gated linear unit)
- Mixup augmentation (convex combination)

## L6: Canonical Problems (Complete ✓)
- Image classification (LeNet5 on MNIST-like input)
- Sequence processing (RNN/LSTM/GRU on toy sequences)
- Machine Translation (Transformer Encoder-Decoder)
- Image generation (GAN training loop)
- Image generation (Diffusion DDPM/DDIM)

## L7: Applications (Complete ✓)
- BERT inference (encoder-only Transformer for classification)
- GPT inference (decoder-only, causal, autoregressive)
- DCGAN image generation (generator + discriminator)
- DDIM fast sampling (50 steps)
- Classifier-Free Guidance (text-conditional diffusion)
- Cosine Embedding Loss (face verification, contrastive learning)
- Mixup data augmentation (improves robustness)

## L8: Advanced Topics (Partial+ ✓)
- RoPE: Rotary Position Embedding (LLaMA, Mistral, Gemma)
- ALiBi: Attention with Linear Biases (length extrapolation)
- RMSNorm: Root Mean Square Normalization (LLaMA)
- SwiGLU / GEGLU: Gated activation for FFN (LLaMA, PaLM)
- GroupNorm: channel-group normalization (Stable Diffusion)
- LAMB: Layer-wise Adaptive Moments (large-batch BERT)

## L9: Industry Frontiers (Partial — documented)
- AI Compiler (Triton, MLIR) for model optimization
- Mixture of Experts (MoE) for sparse scaling
- Speculative Decoding for inference acceleration
- KV-Cache optimization for autoregressive models
- Flash Attention for O(N) memory attention
