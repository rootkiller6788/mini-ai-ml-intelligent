# mini-training-system — 训练系统 (C 语言实现)

C99 实现的微型分布式训练系统，涵盖分布式训练、混合精度、检查点、
超参数调优与训练循环五大模块。

## 模块

| 模块 | 头文件 | 说明 |
|---|---|---|
| 分布式训练 | `distributed_train.h` | DDP / Model Parallel / Pipeline / 3D / ZeRO |
| 混合精度 | `mixed_precision.h` | FP16/FP32/BF16, Loss Scaling, TensorCore |
| 检查点 | `checkpoint_save.h` | 保存/恢复, best/periodic, 异步, 压缩 |
| 超参数调优 | `hyperparam_tune.h` | Grid/Random/Bayesian, Hyperband, Early Stop |
| 训练循环 | `training_loop.h` | forward/backward/step, metrics, logging, profiling |

## 构建

```bash
make          # 构建库和所有示例
make lib      # 仅构建静态库 libtrain.a
make examples # 构建 example_*
make demos    # 构建 demo_*
make clean
```

## 使用

```c
#include "training_loop.h"
#include "mixed_precision.h"

int main(void) {
    train_config_t cfg = train_config_default();
    cfg.num_epochs = 10;
    cfg.batch_size = 32;
    cfg.learning_rate = 1e-3f;

    train_context_t* ctx = train_init(&cfg);
    train_run(ctx);
    train_free(ctx);
    return 0;
}
```

## 许可

MIT
