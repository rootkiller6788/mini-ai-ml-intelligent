# Mini AI ML Intelligent（迷你人工智能机器学习智能）

**从零开始、零依赖的 C 语言实现**，涵盖机器学习、深度学习框架、模型架构、AI 系统和智能应用。每个模块实现从经典 ML 算法到现代深度学习 — 包括自动求导引擎、Transformer 架构、分布式训练推理、RAG 检索增强生成和 AI Agent 运行时。

## 模块总览

| 模块 | 主题 | 参考标准 |
|--------|--------|----------------|
| [mini-ml-basic](mini-ml-basic/) | 线性/Logistic 回归、SVM、决策树、随机森林、K-Means、梯度下降 | MIT 6.036, Stanford CS229 |
| [mini-dl-framework](mini-dl-framework/) | 自动求导（计算图）、张量操作、层（Linear/Conv/RNN）、优化器（SGD/Adam）、损失函数 | PyTorch, TensorFlow |
| [mini-model-arch](mini-model-arch/) | CNN（LeNet/ResNet）、RNN/LSTM、Transformer（自注意力/MHA）、GAN、扩散模型 | LeCun/Bengio/Hinton 论文 |
| [mini-training-system](mini-training-system/) | 分布式训练（数据/模型并行）、混合精度（FP16/FP32）、检查点、超参数搜索 | Megatron-LM, DeepSpeed |
| [mini-inference-system](mini-inference-system/) | 模型服务（Triton）、量化（INT8/INT4）、KV Cache、推测解码、动态批处理 | vLLM, TensorRT-LLM |
| [mini-rag-knowledge](mini-rag-knowledge/) | RAG 管线（嵌入→检索→生成）、分块策略、重排序、混合搜索、护栏 | LangChain, LlamaIndex |
| [mini-agent-runtime](mini-agent-runtime/) | ReAct Agent、工具使用/函数调用、规划（ReWOO/Plan-Execute）、记忆系统 | OpenAI Agents, LangGraph |
| [mini-multimodal-ai](mini-multimodal-ai/) | CLIP 对比学习、图像生成（扩散模型）、VLLaMA、音频（Whisper）、视频理解 | CLIP, Stable Diffusion, Whisper |
| [mini-ai-system-software](mini-ai-system-software/) | CUDA Kernel 仿真（Grid/Block/Thread）、GPU 内存层次、NCCL 集合通信、Flash Attention | CUDA 编程指南, NCCL |
| [mini-ai-product-system](mini-ai-product-system/) | 推荐系统（Embedding+召回+排序）、聊天机器人（RLHF 管线）、Copilot（上下文+补全） | Netflix/ByteDance 架构 |

## 设计理念

- **零外部依赖** — 纯 C（C99/C11），仅使用 `libc` 和 `libm`
- **模块自包含** — 每个目录自带 `Makefile`、`include/`、`src/`、`examples/`、`demos/`、`tests/`
- **用户态 AI 仿真** — 对深度学习框架、训练/推理系统和 AI 应用的教学级建模
- **理论到代码的映射** — 每个模块包含 `docs/` 目录，内有论文/课程对齐说明
- **实用演示程序** — 自动求导引擎、CNN/Transformer 推理、RAG 管线、AI Agent 运行时等

## 构建方式

每个模块相互独立。进入模块目录后运行：

```bash
cd mini-dl-framework
make all    # 构建全部
make test   # 运行测试
```

需要 **GCC** 和 **GNU Make**。

## 项目结构

```
mini-ai-ml-intelligent/
├── mini-ml-basic/               # 经典机器学习
├── mini-dl-framework/           # 深度学习框架
├── mini-model-arch/             # 模型架构
├── mini-training-system/        # 训练系统
├── mini-inference-system/       # 推理系统
├── mini-rag-knowledge/          # RAG 与知识检索
├── mini-agent-runtime/          # AI Agent 运行时
├── mini-multimodal-ai/          # 多模态 AI
├── mini-ai-system-software/     # AI 系统软件
└── mini-ai-product-system/      # AI 产品系统
```

## 许可证

MIT
