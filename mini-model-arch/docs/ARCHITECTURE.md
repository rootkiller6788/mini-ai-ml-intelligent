# mini-model-arch 架构说明

## 目录结构

```
mini-model-arch/
├── README.md              # 项目说明
├── Makefile               # 构建脚本
├── cnn_models.h/c         # CNN: LeNet, VGG, ResNet, Inception, DepthwiseSepConv
├── rnn_lstm.h/c           # RNN: SimpleRNN, LSTM, GRU, BiRNN, StackedRNN
├── transformer_arch.h/c   # Transformer: MHA, Encoder/Decoder, GPT, BERT
├── gan_model.h/c          # GAN: Generator, Discriminator, WGAN-GP, DCGAN
├── diffusion_model.h/c    # Diffusion: DDPM, DDIM, CFG, U-Net
├── example_cnn.c          # CNN 使用示例
├── example_rnn.c          # RNN/LSTM 使用示例
├── example_transformer.c  # Transformer 使用示例
├── demo_gan.c             # GAN 训练演示
├── demo_diffusion.c       # Diffusion 采样演示
├── API_REFERENCE.md       # API 参考文档
└── ARCHITECTURE.md        # 本文件
```

## 设计原则

1. **纯 C99**：无外部依赖，仅使用标准库 (`math.h`, `stdlib.h`, `string.h`, `time.h`)
2. **可读性优先**：每个模块独立，结构清晰，命名直观
3. **教育目的**：代码反映论文原始公式，便于理解算法原理
4. **可扩展**：模块化设计，方便添加新架构

## 数据流

### CNN 数据流
```
Input [B,C,H,W] → Conv2D → ReLU → Pool2D → ... → FC → Softmax → Output
                     ↑                       ↑
              residual connection    skip connection (ResNet)
```

### RNN 数据流
```
Sequence [T, input_size] → RNN/LSTM/GRU cell → hidden states [T, hidden_size]
                              ↓                  ↓
                         Bidirectional    Many-to-one (last state)
                                          Many-to-many (all states)
```

### Transformer 数据流
```
Tokens + Position Encoding
    ↓
Encoder: Self-Attn → Add&Norm → FFN → Add&Norm  (×N layers)
    ↓
Decoder: Masked Self-Attn → Add&Norm → Cross-Attn → Add&Norm → FFN → Add&Norm  (×N layers)
    ↓
Output projection
```

### GAN 数据流
```
Noise z ~ N(0,I) → Generator → Fake sample
                                ↓
Real sample → Discriminator ← Fake sample
                ↓
          Real/Fake score
```

### Diffusion 数据流
```
Training:
x_0 → Forward Diffusion → x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε
                               ↓
                          U-Net(ε_θ) → predict ε
                               ↓
                          Loss = MSE(ε_pred, ε_true)

Sampling (DDPM):
x_T ~ N(0,I) → U-Net(ε_θ) → denoise → x_{T-1} → ... → x_0
```

## ImageNet 架构对比

| 模型 | 年份 | Top-5 Error | 参数 | 核心创新 |
|------|------|-------------|------|----------|
| AlexNet | 2012 | 15.3% | 60M | ReLU, Dropout, GPU |
| VGG-16 | 2014 | 7.3% | 138M | 3×3 conv stack |
| GoogLeNet | 2014 | 6.7% | 6.8M | Inception module |
| ResNet-50 | 2015 | 5.3% | 25.6M | Residual connection |
| ResNet-152 | 2015 | 4.5% | 60.2M | Deep residual |
| EfficientNet | 2019 | 2.9% | 66M | Compound scaling |

## 关键公式参考

### CNN
- Conv2D: `output = sum(input * kernel) + bias`
- MaxPool: `output = max(region)`
- Residual: `output = F(x) + x`

### RNN/LSTM/GRU
- RNN: `h_t = tanh(W_x x_t + W_h h_{t-1} + b)`
- LSTM forget gate: `f_t = σ(W_f x_t + U_f h_{t-1} + b_f)`
- LSTM cell update: `c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t`
- LSTM output: `h_t = o_t ⊙ tanh(c_t)`
- GRU update: `h_t = (1-z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t`

### Transformer
- Attention: `Attention(Q,K,V) = softmax(QK^T/√d_k) V`
- Positional Encoding: `PE(pos,2i) = sin(pos/10000^(2i/d))`
- LayerNorm: `LN(x) = γ · (x-μ)/σ + β`
- FFN: `FFN(x) = max(0, xW_1+b_1)W_2 + b_2`

### GAN
- Minimax: `min_G max_D V(D,G) = E[log D(x)] + E[log(1-D(G(z)))]`
- WGAN: `W(P_r,P_g) = sup_{||f||_L≤1} E_{x~P_r}[f(x)] - E_{x~P_g}[f(x)]`
- Gradient Penalty: `GP = λ·E[(||∇_x̂ D(x̂)||₂ - 1)²]`

### Diffusion
- Forward: `q(x_t|x_0) = N(x_t; √ᾱ_t·x_0, (1-ᾱ_t)I)`
- Training loss: `L = E_{t,x_0,ε}[||ε - ε_θ(x_t,t)||²]`
- DDPM reverse: `x_{t-1} = 1/√α_t (x_t - β_t/√(1-ᾱ_t) ε) + σ_t z`
- DDIM (eta=0): `x_{t-1} = √ᾱ_{t-1}·x̂₀ + √(1-ᾱ_{t-1})·ε_θ`
- CFG: `ε̃_θ(x_t,t,c) = ε_θ(x_t,t,∅) + w(ε_θ(x_t,t,c) - ε_θ(x_t,t,∅))`

## 梯度消失问题详解

### 问题根源
标准 RNN 梯度通过时间反向传播：
```
∂L/∂h_0 = ∂L/∂h_T · ∏_{t=1}^T ∂h_t/∂h_{t-1}
∂h_t/∂h_{t-1} = diag(tanh'(·)) · W_hh
```
当 |W_hh|·|tanh'(x)| < 1 时，乘积指数衰减 → 梯度消失。

### LSTM 如何解决
LSTM 的 cell state 更新是**加法**操作：
```
c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t
∂c_t/∂c_{t-1} = f_t  ← 由网络学习！
```
当 forget gate `f_t ≈ 1` 时，梯度无衰减通过。cell state 成为"梯度高速公路"。

### GRU 如何解决
GRU 通过 update gate z 实现类似机制：
```
h_t = (1-z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
∂h_t/∂h_{t-1} = (1-z_t)  ← 可学习！
```
当 `z_t ≈ 0` 时，`h_t ≈ h_{t-1}`，梯度完整传递。

## Pre-LN vs Post-LN

Transformer 原始论文使用 Post-LN（LayerNorm 在 residual 之后）：
```
Post-LN: x ← LN(x + Sublayer(x))
Pre-LN:  x ← x + Sublayer(LN(x))     ← 本实现采用
```
**Pre-LN 优势**：
- 训练更稳定，不需要 warmup
- 梯度直接从输出传递到输入（通过 residual path）
- 现代架构（GPT-2, GPT-3, LLaMA）都使用 Pre-LN
