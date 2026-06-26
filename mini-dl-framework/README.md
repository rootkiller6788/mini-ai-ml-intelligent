# mini-dl-framework — Deep Learning Framework (Pure C99)

A complete autodiff deep learning framework written in pure C99. Supports computational graph construction, automatic differentiation, tensor operations, neural network layers (CNN, RNN, LSTM, GRU, Transformer attention), optimizers, loss functions, data loading, and model checkpointing — all from scratch with zero external dependencies beyond libc and libm.

**Lines of code**: 4235 (include/ + src/), 15/15 tests passing.

## Module Status: COMPLETE ✅

- **L1 Definitions**: Complete — Tensor, Node, Linear, Conv2d, BatchNorm1d, LayerNorm, Dropout, MaxPool2d, Embedding, RNNCell, LSTMCell, GRUCell, MultiHeadAttention, SGD, Adam, AdamW, LRScheduler, DataLoader, Trainer, ModelCheckpoint
- **L2 Core Concepts**: Complete — Autograd (chain rule, topological sort), Tensor algebra (broadcasting, strides), Convolution (im2col→GEMM), Normalization (batch/layer), Dropout regularization, Embedding (discrete→continuous), Sequential modeling (RNN/LSTM/GRU), Attention (scaled dot-product)
- **L3 Engineering Structures**: Complete — Computational graph with ref-counted nodes, im2col/col2im transformation, broadcast dimension propagation, batch-aware matmul, data loading pipeline, training loop abstraction, parameter list management
- **L4 Standards/Theorems**: Complete — Xavier/Glorot initialization (Var(W)=2/(fan_in+fan_out)), Kaiming/He initialization (Var(W)=2/fan_in), CEC (Constant Error Carousel for LSTM), scaled dot-product attention (1/√d_k), Adam convergence analysis, AdamW decoupled weight decay, BPTT gradient flow
- **L5 Algorithms/Methods**: Complete — Backpropagation, BPTT, SGD with momentum/Nesterov, Adam/AdamW optimization, softmax with numerical stability, im2col convolution, multi-head attention, GEMM, Fisher-Yates shuffle
- **L6 Canonical Problems**: Complete — 5 executable examples (MLP, ConvNet, GAN, MNIST demo, transfer learning), unit test suite (15 tests)
- **L7 Applications**: Partial+ — MNIST classification demo, transfer learning demo, GAN training example, model checkpoint serialization, accuracy/precision/recall/F1 metrics
- **L8 Advanced Topics**: Partial+ — Multi-Head Self-Attention (Transformer core), sinusoidal positional encoding, GRU (gated recurrence)
- **L9 Industry Frontiers**: Partial (documented) — Transformer architecture, attention mechanisms, mixed precision training, gradient checkpointing

## Core Definitions (L1)

| Type | Description |
|------|-------------|
| `Tensor` | N-dimensional array with strides, dims, owned data |
| `Node` | Autograd computation graph node with value, grad, op type |
| `Linear` | Fully-connected layer: y = xW^T + b |
| `Conv2d` | 2D convolution with im2col→GEMM |
| `BatchNorm1d` | Batch normalization with running stats |
| `LayerNorm` | Layer normalization (per-sample mean/std) |
| `Dropout` | Inverted dropout regularization |
| `MaxPool2d` | 2D max pooling with argmax indices |
| `Embedding` | Token embedding lookup table |
| `RNNCell` | Elman recurrent cell: h_t = tanh(W_hh·h_{t-1}+W_ih·x_t+b) |
| `LSTMCell` | LSTM with forget/input/candidate/output gates |
| `GRUCell` | GRU with reset/update gates |
| `MultiHeadAttention` | Scaled dot-product multi-head attention |
| `SGD` | SGD with momentum, weight decay, Nesterov |
| `Adam` | Adam optimizer (β1, β2, ε) |
| `AdamW` | AdamW with decoupled weight decay |
| `DataLoader` | Batched data iterator with shuffle |
| `Trainer` | Training loop with early stopping |
| `ModelCheckpoint` | Binary model serialization |

## Core Theorems (L4)

| Theorem | Formula | Implementation |
|---------|---------|---------------|
| Chain Rule (Autograd) | dL/dx = dL/dy · dy/dx | `autograd.c: backward()` |
| Xavier Init | Var(W) = 2/(fan_in+fan_out) | `tensor_init(INIT_XAVIER_*)` |
| Kaiming Init | Var(W) = 2/fan_in | `tensor_init(INIT_KAIMING_*)` |
| CEC (LSTM) | c_t = f_t·c_{t-1} + i_t·g_t | `lstm_cell_forward()` |
| Scaled Attention | softmax(QK^T/√d_k)·V | `mha_forward()` |
| Adam Bias Correction | m̂ = m/(1-β₁ᵗ), v̂ = v/(1-β₂ᵗ) | `adam_step()` |
| Softmax Stability | softmax(x-max(x)) | `tensor_softmax()` |

## Core Algorithms (L5)

| Algorithm | Complexity | Location |
|-----------|-----------|----------|
| Backpropagation | O(|V|+|E|) | `autograd.c: backward()` |
| BPTT (RNN gradient) | O(T·H²) | `rnn_cell_backward()` |
| im2col Convolution | O(N·C·K²·H·W) | `im2col()` + GEMM |
| Multi-Head Attention | O(n²·d) | `mha_forward()` |
| Fisher-Yates Shuffle | O(N) | `dataloader_reset()` |

## Classic Problems (L6)

- **MLP Training**: `examples/example_mlp.c`
- **CNN Training**: `examples/example_convnet.c`
- **GAN Training**: `examples/example_gan.c`
- **MNIST Demo**: `examples/demo_mnist.c`
- **Transfer Learning**: `examples/demo_transfer_learning.c`

## Nine-School Curriculum Mapping (L4-L6)

| University | Course | Coverage |
|------------|--------|----------|
| **MIT** | 6.036 Intro to ML | Backprop, SGD, layers (L5) |
| **Stanford** | CS 231n ConvNets | im2col, BatchNorm, Dropout (L3+L5) |
| **Stanford** | CS 224n NLP | Embedding, RNN, LSTM, Attention (L2+L8) |
| **CMU** | 11-785 Deep Learning | Adam, AdamW, Xavier/Kaiming init (L4) |
| **Berkeley** | CS 182 Deep Neural Nets | LayerNorm, GRU, Transformer (L8) |
| **ETH** | 263-3210 Deep Learning | Conv2d, autograd, optimization (L3) |
| **Cambridge** | Part II: Machine Learning | Loss functions, regularization (L4) |
| **清华** | 深度学习 | Full framework pipeline (L6) |
| **Georgia Tech** | CS 7643 Deep Learning | MHA, positional encoding (L8) |

## Quick Start

```c
#include "tensor_ops.h"
#include "nn_layers.h"
#include "optimizers.h"
#include "loss_funcs.h"
#include "rnn_layers.h"
#include "training.h"

int main(void) {
    /* Create a simple MLP */
    int dims[] = {1, 10};
    Tensor* x = tensor_create_randomn(dims, 2);

    Linear* fc = linear_create(10, 5, true);
    Tensor* out = linear_forward(fc, x);

    Tensor* target = tensor_create_randomn((int[]){1, 5}, 2);
    float loss_val = mse_loss_value(out, target, REDUCE_MEAN);
    printf("Loss: %f\n", loss_val);

    linear_free(fc);
    tensor_free(x);
    tensor_free(out);
    tensor_free(target);
    return 0;
}
```

## Building

```
make          # build library + test
make test     # build + run test suite (15 tests)
make clean    # clean build artifacts
```

## Documentation

- [API Reference](docs/API.md)
- [Architecture](docs/ARCHITECTURE.md)

## License

MIT
