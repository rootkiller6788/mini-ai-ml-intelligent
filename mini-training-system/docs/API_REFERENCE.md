# mini-training-system API Reference

## 分布式训练 `distributed_train.h`

| 函数 | 说明 |
|---|---|
| `ddp_init` | 初始化 DDP 上下文 |
| `ddp_finalize` | 清理 DDP 上下文 |
| `ddp_all_reduce` | 跨 GPU 的 all-reduce 操作 |
| `ddp_broadcast` | 从 root 广播张量 |
| `ddp_all_gather` | 收集所有 GPU 的局部张量 |
| `ring_all_reduce` | 环形拓扑的 all-reduce 算法 |
| `ring_all_reduce_async` | 异步 ring all-reduce |
| `ddp_reduce_scatter` | 规约后分散操作 |
| `ddp_gradient_is_finite` | 梯度有限性检查 |
| `zero_init` | ZeRO 优化器配置初始化 |
| `zero_partition_optimizer_state` | 分区优化器状态 |
| `zero_partition_gradients` | 分区梯度 |
| `zero_partition_parameters` | 分区参数 |
| `zero_all_gather_params` | ZeRO-3 收集参数 |
| `hybrid_3d_init` | 3D 并行初始化 |
| `tensor_model_parallel_all_reduce` | TP 组的 all-reduce |
| `tensor_model_parallel_all_gather` | TP 组的 all-gather |
| `tensor_model_parallel_split` | 按 TP 维度切分张量 |
| `pipeline_schedule_1f1b` | 一前一后的流水线调度 (1F1B) |
| `pipeline_flush_schedule` | 流水线刷新调度 |
| `allgather_ring` | 环形 all-gather 算法 |
| `allgather_estimate_bandwidth` | All-gather 带宽预估 |

## 混合精度 `mixed_precision.h`

| 函数 | 说明 |
|---|---|
| `mp_init` | 初始化混合精度上下文 |
| `fp32_to_fp16` / `fp16_to_fp32` | 批量 FP32 ↔ FP16 转换 |
| `fp32_to_bf16` / `bf16_to_fp32` | 批量 FP32 ↔ BF16 转换 |
| `fp32_to_fp16_scalar` | 标量 FP32 → FP16 |
| `fp16_to_fp32_scalar` | 标量 FP16 → FP32 |
| `fp32_to_bf16_scalar` | 标量 FP32 → BF16 |
| `bf16_to_fp32_scalar` | 标量 BF16 → FP32 |
| `mp_is_safe_fp16_op` | 算子 FP16 安全性检查 |
| `mp_scale_loss` | 损失缩放 |
| `mp_unscale_gradients` | 梯度反缩放 |
| `mp_update_loss_scale` | 动态更新缩放因子 |
| `mp_master_to_model_fp16` | FP32 主权重 → FP16 模型权重 |
| `mp_model_to_master_fp16` | FP16 模型权重 → FP32 主权重 |
| `mp_fp16_matmul` | FP16 矩阵乘法 (可选 TensorCore) |
| `mp_tensorcore_available` | TensorCore 可用性检查 |
| `mp_grad_scale_and_check` | 梯度缩放 + 溢出检测 |
| `mp_half_precision_error` | FP16 量化误差评估 |
| `mp_autocast_forward` | 前向自动转换 |
| `mp_autocast_backward` | 后向自动转换 |

## 检查点 `checkpoint_save.h`

| 函数 | 说明 |
|---|---|
| `ckpt_init` | 初始化检查点上下文 |
| `ckpt_should_save` | 判断是否应保存 |
| `ckpt_save` | 保存检查点 |
| `ckpt_save_best` | 基于指标保存最佳检查点 |
| `ckpt_save_latest` | 保存最新检查点 |
| `ckpt_load` | 载入指定检查点 |
| `ckpt_load_latest` | 载入最新检查点 |
| `ckpt_load_best` | 载入最佳检查点 |
| `ckpt_save_async_start` | 启动异步保存 |
| `ckpt_save_async_is_done` | 异步保存是否完成 |
| `ckpt_save_async_wait` | 等待异步保存完成 |
| `ckpt_compress_file` | 压缩检查点文件 |
| `ckpt_decompress_file` | 解压检查点文件 |
| `ckpt_cleanup_old` | 清理旧检查点 |
| `ckpt_list_checkpoints` | 列出所有检查点 |
| `ckpt_save_metadata` | 保存元数据 |
| `ckpt_load_metadata` | 载入元数据 |
| `ckpt_recover_from_fault` | 故障恢复 |
| `ckpt_crc32` | CRC32 校验 |

## 超参数调优 `hyperparam_tune.h`

| 函数 | 说明 |
|---|---|
| `hpt_init` | 初始化调优上下文 |
| `hpt_free` | 释放调优上下文 |
| `hpt_add_int` / `hpt_add_float` | 添加搜索参数域 |
| `hpt_add_log_float` / `hpt_add_log_int` | 添加对数域参数 |
| `hpt_add_categorical` | 添加分类参数域 |
| `hpt_suggest` | 生成下一个建议试验 |
| `hpt_report` | 报告试验结果 |
| `hpt_report_intermediate` | 报告中间结果 |
| `hpt_should_prune` | 判断试验是否应剪枝 |
| `gp_model_init` / `gp_model_free` | GP 模型生命期 |
| `gp_model_fit` | 拟合 GP 模型 |
| `gp_model_predict` | GP 模型预测 |
| `gp_model_acq_ei` | 期望改进 (EI) 采集函数 |
| `tpe_sample` | TPE 采样 |
| `hyperband_init` | 初始化 Hyperband |
| `hyperband_get_num_configs` | 某 bracket 的配置数 |
| `hyperband_select_top` | 渐进减半选出 Top-K |
| `hpt_early_stopping_check` | 早停检查 |
| `hpt_best_trial` | 获取最优试验 |

## 训练循环 `training_loop.h`

| 函数 | 说明 |
|---|---|
| `tl_init` / `tl_free` | 生命周期管理 |
| `tl_model_add_layer` / `tl_model_free` | 模型层构建/释放 |
| `tl_register_dataset` | 注册训练/验证数据集 |
| `tl_forward` | 前向传播 |
| `tl_compute_loss` | 损失计算 |
| `tl_backward` | 反向传播 |
| `tl_optimizer_step` | 优化器步进 |
| `tl_scheduler_step` | 学习率调度器步进 |
| `tl_train_epoch` | 训练一个 epoch |
| `tl_train_step` | 单步训练 |
| `tl_train_run` | 完整训练运行 |
| `tl_validate` | 验证评估 |
| `tl_eval_mode` | 切换训练/评估模式 |
| `tl_compute_accuracy` | 计算准确率 |
| `tl_compute_precision` | 计算精确率 |
| `tl_compute_recall` | 计算召回率 |
| `tl_compute_f1` | 计算 F1 分数 |
| `tl_gradient_accumulation_step` | 梯度累积步 |
| `tl_gradient_checkpoint_recompute` | 梯度检查点重算 |
| `tl_clip_grad_norm` | 梯度裁剪 |
| `tl_log_metrics` | 打印指标 |
| `tl_lr_cosine` | 余弦学习率 |
| `tl_lr_warmup_cosine` | 预热余弦学习率 |
| `tl_profile_report` | 性能报告 |
| `tl_early_stopping_update` | 早停更新 |
