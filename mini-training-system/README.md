# mini-training-system — Training System (C99 Implementation)

A complete C99 micro distributed training system covering distributed training,
mixed precision, checkpointing, hyperparameter tuning, training loop, learning
rate finding, and model averaging — seven integrated modules.

## Module Status: COMPLETE ✅

- **L1 Definitions**: Complete — 10+ structs, 7 enums, 70+ API functions
- **L2 Core Concepts**: Complete — Forward/Backward, LR scheduling, ZeRO, AMP, EMA/SWA
- **L3 Engineering Structures**: Complete — Full training context, gradient accumulation, pipeline
- **L4 Standards/Theorems**: Complete — Chain rule (backprop), Polyak-Ruppert averaging, Smith LR range test
- **L5 Algorithms**: Complete — Backpropagation, Cholesky GP, Hyperband, Ring All-Reduce, Softmax CE
- **L6 Canonical Problems**: Complete — Training loop, checkpoint/resume, distributed data parallel
- **L7 Applications**: Complete — LR finder (binary search + range test), EMA for inference, TPE sampler
- **L8 Advanced Topics**: Partial+ — Cholesky GP regression, ZeRO-3 partitioning, 3D parallelism
- **L9 Industry Frontiers**: Partial — TensorCore emulation (documented), pipeline 1F1B schedule

**include/ + src/ = 3,548 lines** (threshold: 3,000 ✅) | **13/13 tests passing** ✅

## Modules

| # | Module | Header | Lines | Description |
|---|--------|--------|-------|-------------|
| 1 | Training Loop | `training_loop.h` | 259+772 | Forward/backward/optimizer/scheduler, metrics, profiling |
| 2 | Mixed Precision | `mixed_precision.h` | 115+242 | FP16/BF16 conversion, loss scaling, TensorCore emulation |
| 3 | Checkpoint | `checkpoint_save.h` | 145+301 | Save/restore, best/periodic/manual, async, CRC32 integrity |
| 4 | Distributed Train | `distributed_train.h` | 148+298 | DDP, Ring All-Reduce, ZeRO 1-3, pipeline 1F1B, 3D hybrid |
| 5 | Hyperparameter Tune | `hyperparam_tune.h` | 185+574 | Grid/Random/Bayesian/TPE/Hyperband/CMA-ES, GP regression |
| 6 | LR Finder | `learn_rate_finder.h` | 82+193 | Smith (2017) LR Range Test, binary search, steepest descent |
| 7 | Model Averaging | `model_averaging.h` | 122+112 | EMA, SWA, LAWA with Polyak-Ruppert bias correction |

## Core Definitions (L1)

```
tl_context_t     — Full training state (model, optimizer, scheduler, metrics, datasets)
mp_context_t     — Mixed precision state (loss scale, overflow tracking)
ckpt_context_t   — Checkpoint manager (config, state, async I/O)
ddp_context_t    — Distributed data parallel state (world, rank, backend)
hpt_context_t    — Hyperparameter optimization (trials, GP model, Hyperband)
lrf_context_t    — LR range test state (history, smoothing, divergence detection)
ma_context_t     — Model averaging state (EMA buffer, SWA accumulator)
```

## Core Theorems (L4)

| Theorem | Implementation | Source |
|---------|---------------|--------|
| Chain Rule (Leibniz) | `tl_backward()` — full backprop through ReLU+linear layers | Rumelhart, Hinton, Williams (1986) |
| Softmax-CE Gradient Identity ∂CE/∂z = softmax(z) − y | `tl_backward()` numerical stable softmax | Bishop PRML §4.3.4 |
| Polyak-Ruppert Averaging w̄_T = (1/T)Σw_t | `ma_update_ema()` with bias correction | Polyak & Juditsky (1992) |
| Smith LR Range Test | `lrf_get_lr()` + `lrf_suggest_lr()` exponential schedule | Smith (2017) WACV |
| Cholesky Decomposition K = LL^T | `gp_model_fit()` — Gaussian Process regression | Rasmussen & Williams (2006) |
| CRC-32 Integrity | `ckpt_crc32()` — standard Ethernet polynomial | IEEE 802.3 |

## Core Algorithms (L5)

1. **Backpropagation** — O(L·B·d²) reverse-mode autodiff through multi-layer perceptron
2. **Softmax Cross-Entropy** — Log-sum-exp stabilized softmax with implicit Jacobian
3. **Ring All-Reduce** — O(2(N-1)) ring communication for gradient synchronization
4. **Cholesky GP Regression** — O(n³) Gaussian Process fitting with RBF kernel
5. **Hyperband** — Multi-armed bandit with successive halving (Li et al., 2017)
6. **TPE Sampling** — Tree-structured Parzen Estimator (Bergstra et al., 2011)
7. **LR Range Test** — Exponential LR sweep + divergence detection (Smith, 2017)
8. **EMA with Bias Correction** — θ̂_t = θ_t / (1 − β^t) — Kingma & Ba (2015)

## Nine-School Course Mapping

| School | Course | Coverage |
|--------|--------|----------|
| **MIT** | 6.824 Distributed Systems | Ring All-Reduce, ZeRO partitioning |
| **Stanford** | CS 229 Machine Learning | Backprop, cross-entropy, GP regression |
| **Berkeley** | CS 267 HPC | Pipeline parallelism, 3D hybrid, gradient accumulation |
| **CMU** | 15-418 Parallel | DDP, all-gather, reduce-scatter, ZeRO-3 |
| **UT Austin** | CS 395T Systems ML | Mixed precision (FP16/BF16), loss scaling, TensorCore |
| **ETH** | 263-3501 Parallel Prog | Ring all-reduce, allgather bandwidth model |
| **Cambridge** | Part II Concurrent Sys | Async checkpoint, 1F1B pipeline schedule |
| **清华** | 操作系统 | Checkpoint fault tolerance, save/restore |
| **Georgia Tech** | CS 7641 Machine Learning | Hyperparameter tuning, Bayesian optimization, Hyperband |

## Build & Test

```bash
make          # Build library, examples, demos, and run tests
make lib      # Build static library libtrain.a only
make test     # Build and run unit tests (13 tests)
make examples # Build example programs
make clean    # Clean build artifacts
```

## Usage

```c
#include "training_loop.h"
#include "learn_rate_finder.h"

int main(void) {
    tl_train_config_t train_cfg = {0};
    train_cfg.max_epochs = 10;
    train_cfg.batch_size = 32;

    tl_optimizer_config_t optim_cfg = {0};
    optim_cfg.type = TL_OPTIM_ADAM;
    optim_cfg.learning_rate = 0.001f;

    tl_scheduler_config_t sched_cfg = {0};
    sched_cfg.type = TL_SCHEDULER_COSINE;
    sched_cfg.base_lr = 0.001f;

    tl_context_t ctx;
    tl_init(&ctx, &train_cfg, &optim_cfg, &sched_cfg);

    /* Build model */
    tl_model_add_layer(&ctx.model, 784, 256, true);
    tl_model_add_layer(&ctx.model, 256, 10, true);

    /* Register data */
    tl_register_dataset(&ctx, train_data, train_labels, n, dim, classes, false);
    tl_register_dataset(&ctx, val_data, val_labels, n_val, dim, classes, true);

    /* Train */
    tl_train_run(&ctx);
    tl_free(&ctx);
    return 0;
}
```

## License

MIT
