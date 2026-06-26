# mini-model-arch API Reference

## cnn_models.h

### Tensor4D
```c
typedef struct { int batch, channels, height, width; float *data; } Tensor4D;
void tensor4d_init(Tensor4D *t, int b, int c, int h, int w);
void tensor4d_free(Tensor4D *t);
float tensor4d_get(const Tensor4D *t, int b, int c, int y, int x);
void tensor4d_set(Tensor4D *t, int b, int c, int y, int x, float v);
```

### Conv2D
```c
Conv2D *conv2d_create(int in_c, int out_c, int k, int s, int p);
void conv2d_free(Conv2D *c);
Tensor4D *conv2d_forward(const Conv2D *c, const Tensor4D *in);
```

### Pool2D
```c
Tensor4D *pool2d_forward(const Pool2D *p, const Tensor4D *in, int mode);
// mode: 0=max_pool, 1=avg_pool
```

### LeNet-5
```c
LeNet5 *lenet5_create(void);
void lenet5_free(LeNet5 *l);
Tensor4D *lenet5_forward(const LeNet5 *l, const Tensor4D *in);
// Architecture: Conv(1→6,k=5)→MaxPool→Conv(6→16,k=5)→MaxPool→FC(400→120)→FC(120→84)→FC(84→10)
```

### ResBlock
```c
ResBlock *resblock_create(int in_c, int out_c, int stride, int bottleneck);
void resblock_free(ResBlock *r);
Tensor4D *resblock_forward(const ResBlock *r, const Tensor4D *in);
// bottleneck=0: BasicBlock (3x3→3x3)
// bottleneck=1: BottleneckBlock (1x1→3x3→1x1, expansion=4)
// Automatically adds 1x1 projection shortcut when in_c≠out_c or stride≠1
```

### InceptionModule
```c
InceptionModule *inception_create(int in_c, int c1, int c3r, int c3, int c5r, int c5, int pool_proj);
void inception_free(InceptionModule *im);
Tensor4D *inception_forward(const InceptionModule *im, const Tensor4D *in);
// Parallel branches: 1x1, 3x3(reduce), 5x5(reduce), MaxPool+1x1
```

### DepthwiseSepConv
```c
DepthwiseSepConv *dwsepconv_create(int in_c, int out_c, int k, int s, int p);
void dwsepconv_free(DepthwiseSepConv *d);
Tensor4D *dwsepconv_forward(const DepthwiseSepConv *d, const Tensor4D *in);
// depthwise: in_c × k×k (per-channel)
// pointwise: in_c × out_c × 1×1
```

### Activations
```c
void relu_inplace(Tensor4D *t);
void softmax_inplace(Tensor4D *t_channels);
```

---

## rnn_lstm.h

### RNN Cell — h_t = tanh(Wx + Uh_{t-1} + b)
```c
RNNCell *rnn_cell_create(int in_sz, int hid_sz);
void rnn_cell_free(RNNCell *c);
float *rnn_cell_forward(const RNNCell *c, const float *x, float *h, int steps);
```

### LSTM Cell — f/i/o gates + cell state
```c
LSTMCell *lstm_cell_create(int in_sz, int hid_sz);
void lstm_cell_free(LSTMCell *c);
void lstm_cell_forward(const LSTMCell *c, const float *x, float *h, float *cell, int steps);
// Gates: f = σ(W_f x + U_f h + b_f) — forget
//        i = σ(W_i x + U_i h + b_i) — input
//        o = σ(W_o x + U_o h + b_o) — output
// Cell:  c̃ = tanh(W_c x + U_c h + b_c)
//        c_t = f ⊙ c_{t-1} + i ⊙ c̃
//        h_t = o ⊙ tanh(c_t)
```

### GRU Cell — reset/update gates
```c
GRUCell *gru_cell_create(int in_sz, int hid_sz);
void gru_cell_free(GRUCell *c);
void gru_cell_forward(const GRUCell *c, const float *x, float *h, int steps);
// Gates: r = σ(W_r x + U_r h + b_r) — reset
//        z = σ(W_z x + U_z h + b_z) — update
//        h̃ = tanh(W_h x + U_h(r ⊙ h) + b_h)
//        h_t = (1-z) ⊙ h_{t-1} + z ⊙ h̃
```

### Bidirectional RNN
```c
BiRNN *birnn_create(int in_sz, int hid_sz);
void birnn_free(BiRNN *b);
float *birnn_forward(const BiRNN *b, const float *x, float *h_fw, float *h_bw, int steps);
// Output: [h_fw || h_bw] for each timestep (2 × hidden_size)
```

### Utilities
```c
float sigmoid_act(float x);
float tanh_act(float x);
float *linear_rnn(const float *w, const float *x, const float *b, int rows, int cols);
```

### Gradient Vanishing in RNNs
Standard RNN: repeated tanh derivatives shrink gradients (|tanh'(x)| ≤ 1).
LSTM mitigates via:
- Forget gate = 1 preserves gradient flow through cell state
- Cell state uses additive updates (no tanh), avoiding multiplicative vanishing
- gating mechanisms learn when to forget/remember

---

## transformer_arch.h

### Multi-Head Attention
```c
MultiHeadAttn *mha_create(int d_model, int n_heads, int causal);
void mha_free(MultiHeadAttn *m);
Matrix2D *mha_forward(const MultiHeadAttn *m, const Matrix2D *Q, const Matrix2D *K, const Matrix2D *V);
// Scaled dot-product: Attention(Q,K,V) = softmax(QK^T / √d_k) V
// causal=1: mask upper-triangular (GPT-style)
```

### Position Encoding
```c
PositionEncoding *pe_create(int d_model, int max_len, int learnable);
void pe_free(PositionEncoding *p);
void pe_apply(PositionEncoding *p, Matrix2D *x);
// Sinusoidal: PE(pos,2i)=sin(pos/10000^(2i/d)), PE(pos,2i+1)=cos(pos/10000^(2i/d))
```

### FeedForward Network
```c
FeedForward *ffn_create(int d_model, int d_ff);
void ffn_free(FeedForward *f);
Matrix2D *ffn_forward(const FeedForward *f, const Matrix2D *x);
// FFN(x) = ReLU(xW1 + b1)W2 + b2
```

### Layer Normalization
```c
LayerNorm *ln_create(int d_model);
void ln_free(LayerNorm *l);
void ln_forward(const LayerNorm *l, Matrix2D *x);
// LN(x) = γ(x-μ)/σ + β
```

### Encoder / Decoder Blocks
```c
EncoderBlock *enc_block_create(int d_model, int n_heads, int d_ff, int pre_ln);
DecoderBlock *dec_block_create(int d_model, int n_heads, int d_ff, int pre_ln);
// pre_ln=1: LayerNorm before sublayer (Post-LN was original, Pre-LN converges better)
// Decoder has causal self-attn + cross-attn to encoder output
```

### GPT (Decoder-only, causal)
```c
Matrix2D *gpt_forward(const Matrix2D *tokens, int n_layers, int d_model, int n_heads, int d_ff);
// Auto-regressive: each position attends only to previous positions
```

### BERT (Encoder-only, bidirectional)
```c
Matrix2D *bert_forward(const Matrix2D *tokens, int n_layers, int d_model, int n_heads, int d_ff);
// All positions attend to all positions (no causal mask)
```

### Matrix Operations
```c
Matrix2D *matrix2d_create(int dim, int seq);
void matrix2d_free(Matrix2D *m);
void softmax_rows(float *x, int rows, int cols);
void matmul(const float *A, const float *B, float *C, int m, int k, int n);
void matmul_transpose_b(const float *A, const float *B, float *C, int m, int k, int n);
```

---

## gan_model.h

### Generator / Discriminator
```c
Generator *gen_create(int noise_dim, int *hidden, int n_hid, int out_dim, int bn);
Discriminator *disc_create(int in_dim, int *hidden, int n_hid);
float *gen_forward(const Generator *g, const float *noise);
float disc_forward(const Discriminator *d, const float *x);
```

### GAN Training
```c
GAN *gan_create(int noise_dim, int data_dim, int *gen_hid, int n_gen,
                int *disc_hid, int n_disc, int loss_type);
void gan_train_step_d(GAN *gan, const float *real_batch, int bs, float lr);
void gan_train_step_g(GAN *gan, int bs, float lr);
```

### Loss Types
| Type | Discriminator | Generator |
|------|--------------|-----------|
| 0 (Minimax) | -log(D(x)) - log(1-D(G(z))) | -log(D(G(z))) |
| 1 (Non-saturating) | -log(D(x)) - log(1-D(G(z))) | -log(D(G(z))) |

### WGAN-GP
```c
float wgan_gp_penalty(const Discriminator *d, const float *real,
                       const float *fake, int batch_size, float lambda);
// WGAN: L = E[D(x_fake)] - E[D(x_real)] + λ * GP
// Gradient penalty enforces ||∇D||₂ ≈ 1 on interpolated samples
```

### Mode Collapse
```c
void mode_collapse_detect(const float *samples, int n_samples, int dim, float *diversity);
// Measures average pairwise distance from mean; low diversity → mode collapse
```

---

## diffusion_model.h

### Diffusion Schedule
```c
DiffusionSchedule *diff_schedule_create(int steps, float b_start, float b_end, int sched_type);
void diff_schedule_free(DiffusionSchedule *s);
// sched_type: 0=linear, 1=cosine
// Stores β_t, α_t=1-β_t, ᾱ_t=∏α_i, √ᾱ_t, √(1-ᾱ_t)
```

### Forward Process (q(x_t | x_0))
```c
void forward_diffuse(const DiffusionSchedule *s, const float *x0,
                     float *x_t, float *noise, int t, int dim);
// x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε,  ε ~ N(0,I)
```

### U-Net Model
```c
UNet *unet_create(int in_c, int hid_c, int out_c, int t_emb_dim, int n_res);
void unet_free(UNet *u);
float *unet_forward(const UNet *u, const float *x, int t_emb, int h, int w);
// Predicts noise ε given x_t and timestep embedding
```

### Training
```c
float *diff_train_step(const DiffusionModel *model, const float *x0, int bs, float lr);
// Loss = MSE(ε_predicted, ε_sampled), random t ~ Uniform(0,T-1)
```

### DDPM Sampling (reverse process)
```c
float *ddpm_sample(const DiffusionModel *model, int batch_size, int seed);
// Iterative denoising: x_{t-1} = 1/√α_t (x_t - β_t/√(1-ᾱ_t) ε_θ(x_t,t)) + σ_t z
```

### DDIM Sampling (deterministic)
```c
float *ddim_sample(const DiffusionModel *model, int steps, float eta, int seed);
// x_{t-1} = √ᾱ_{t-1}·x̂₀ + √(1-ᾱ_{t-1}-σ²)·ε + σ·z
// eta=0: deterministic (invertible), eta=1: stochastic DDPM
// Uses fewer steps via subsampling (stride = T/steps)
```

### Classifier-Free Guidance
```c
void classifier_free_guidance(float *eps_uncond, float *eps_cond, float *out,
                               float w, int dim);
// ε̃ = ε_uncond + w · (ε_cond - ε_uncond)
// w=1: no guidance, w>1: stronger conditioning
```

### Schedule Functions
```c
void beta_schedule_linear(float *betas, int T, float start, float end);
void beta_schedule_cosine(float *betas, int T, float s);
```
