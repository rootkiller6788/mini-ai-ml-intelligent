# mini-dl-framework — 深度学习框架 (C 语言实现)

A minimal autodiff deep learning framework written in pure C99. Supports computational graph construction, automatic differentiation, tensor operations, neural network layers, optimizers, and loss functions — all from scratch.

## Features

- **Autograd Engine**: Computational graph with forward/backward passes, chain rule gradient computation, topological sort
- **Tensor Operations**: Element-wise ops, matmul (GEMM), transpose, reshape, slice, concat, softmax, broadcasting
- **NN Layers**: Linear, Conv2d (im2col→GEMM), BatchNorm1d, LayerNorm, Dropout, pooling
- **Optimizers**: SGD (momentum, weight_decay, Nesterov), Adam, AdamW with LR scheduling
- **Loss Functions**: MSE, BCEWithLogits, CrossEntropyLoss, Focal Loss, L1 Loss, Huber Loss with label smoothing
- **Pure C99** — no dependencies beyond standard library

## Building

```
make          # build library + examples
make demo     # build demos
make run      # run all examples
make clean    # clean build artifacts
```

## Quick Start

```c
#include "tensor_ops.h"
#include "nn_layers.h"
#include "optimizers.h"
#include "loss_funcs.h"

int main(void) {
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

## Documentation

- [API Reference](API.md)
- [Architecture](ARCHITECTURE.md)

## License

MIT
