# mini-training-system 架构设计

## 系统概览

```
mini-training-system/
├── include/          # 5 公共头文件
│   ├── distributed_train.h
│   ├── mixed_precision.h
│   ├── checkpoint_save.h
│   ├── hyperparam_tune.h
│   └── training_loop.h
├── src/              # 5 核心模块实现
│   ├── distributed_train.c
│   ├── mixed_precision.c
│   ├── checkpoint_save.c
│   ├── hyperparam_tune.c
│   └── training_loop.c
├── examples/         # 3 功能示例
│   ├── example_ddp_train.c
│   ├── example_mixed_precision.c
│   └── example_checkpoint_resume.c
├── demos/            # 2 集成演示
│   ├── demo_tune.c
│   └── demo_training.c
├── docs/
│   ├── API_REFERENCE.md
│   └── ARCHITECTURE.md
├── README.md
└── Makefile
```

## 模块依赖关系

```
training_loop  ───  mixed_precision
     │                    │
     └── checkpoint_save  │
                              │
hyperparam_tune ─── training_loop
                              │
distributed_train (独立)
```

- `training_loop` 依赖 `mixed_precision` (AMP)
- `training_loop` 可集成 `checkpoint_save` (持久化)
- `hyperparam_tune` 调用 `training_loop` 的子训练
- `distributed_train` 是独立的通信模块

## 模块设计

### 1. distributed_train — 分布式训练

**数据并行 (DDP):**
- 每 GPU 保存完整模型副本
- Ring All-Reduce 聚合梯度
- 环形拓扑最小化带宽使用

**模型并行 (Tensor Parallel):**
- 张量按维度切分到不同 GPU
- TP 组内 all-reduce 同步

**流水线并行:**
- 1F1B 调度 (一前一后)
- 微批次处理减少流水线气泡

**ZeRO (DeepSpeed):**
- Stage 0: 无分区
- Stage 1: 优化器状态分区
- Stage 2: + 梯度分区
- Stage 3: + 参数分区 + offload

### 2. mixed_precision — 混合精度

**FP32 → FP16 转换流程:**
```
FP32 主权重 ─→ FP16 模型权重 (前向)
FP32 损失 ← 缩放到 FP16 范围 ← 反传
梯度 ← 缩放 + 检查溢出
溢出了 → 降低缩放因子 → 跳过更新
正常 → FP16 梯度 → FP32 主权重更新
```

**动态损失缩放:**
- 多步无溢出 → 增大缩放因子 (×2)
- 出现溢出 → 减小缩放因子 (÷2)
- 可配置窗口大小、上下限

### 3. checkpoint_save — 检查点

**文件格式:**
```
+----------+------------+-------------------+-----------+
| Magic(4B)| Version(4B)| Step+Epoch(4B+4B) | State     |
+----------+------------+-------------------+-----------+
| Metadata | Weights   | Optimizer | CRC64  |
+----------+-----------+-----------+--------+
```

**保存策略:**
- `CKPT_STRATEGY_BEST`: 仅当指标改善
- `CKPT_STRATEGY_PERIODIC`: 每隔 N 步
- `CKPT_STRATEGY_BOTH`: 两者皆做
- 异步保存: 不中断训练
- 故障恢复: `ckpt_recover_from_fault`

### 4. hyperparam_tune — 超参数调优

**搜索方法:**
| 方法 | 原理 | 搜索域类型 |
|---|---|---|
| Grid Search | 均匀枚举 | 离散 |
| Random Search | 随机采样 | 连续/离散 |
| Bayesian (GP) | GP 后验 + EI 采集 | 连续 |
| TPE | 对好/坏样本分别建模 | 连续/分类 |
| Hyperband | 渐进减半 | 含预算 |

**GP 模型 (贝叶斯优化):**
- RBF 核: `k(x,x') = σ² exp(-|x-x'|²/(2l²))`
- Cholesky 分解求解
- 采集函数: Expected Improvement (EI)

### 5. training_loop — 训练循环

**标准训练步骤:**
```
for epoch in epochs:
    for batch in dataloader:
        forward → compute loss
        backward → compute gradients
        (optional: gradient scaling)
        (optional: gradient accumulation)
        clip gradients
        optimizer.step()
        scheduler.step()
        log metrics
    validate()
```

**支持特性:**
- 梯度累积: N 个 micro-batch → 1 次参数更新
- 梯度检查点: 前向时不存激活 → 反传时重算
- 混合精度: `tl_context.use_amp = true`
- 性能剖析: 前向/后向时间统计
- 学习率调度: Cosine, Warmup+Cosine, Exponential

## 构建说明

```bash
make          # 编译库 libtrain.a + 所有示例/演示
make lib      # 仅编译 libtrain.a
make examples # 编译 examples/*
make demos    # 编译 demos/*
make clean    # 清理
```

生成的文件:
```
build/
├── libtrain.a
├── example_ddp_train.exe
├── example_mixed_precision.exe
├── example_checkpoint_resume.exe
├── demo_tune.exe
└── demo_training.exe
```
