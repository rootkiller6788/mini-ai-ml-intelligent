# Mini AI ML Intelligent

**From-scratch, zero-dependency C implementations** of machine learning, deep learning frameworks, model architectures, AI systems, and intelligent applications. Each module implements everything from classical ML algorithms to modern deep learning — including autograd engines, Transformer architectures, distributed training/inference, RAG retrieval-augmented generation, and AI agent runtimes.

## Modules

| Module | Topics | Key References |
|--------|--------|----------------|
| [mini-ml-basic](mini-ml-basic/) | Linear/logistic regression, SVM, decision tree, random forest, k-means, gradient descent | MIT 6.036, Stanford CS229 |
| [mini-dl-framework](mini-dl-framework/) | Autograd (computational graph), Tensor ops, layers (Linear/Conv/RNN), optimizers (SGD/Adam), loss functions | PyTorch, TensorFlow |
| [mini-model-arch](mini-model-arch/) | CNN (LeNet/ResNet), RNN/LSTM, Transformer (self-attn/MHA), GAN, Diffusion model | LeCun/Bengio/Hinton papers |
| [mini-training-system](mini-training-system/) | Distributed (data/model parallel), mixed precision (FP16/FP32), checkpoint, hyperparameter search | Megatron-LM, DeepSpeed |
| [mini-inference-system](mini-inference-system/) | Model serving (Triton), quantization (INT8/INT4), KV cache, speculative decoding, batching | vLLM, TensorRT-LLM |
| [mini-rag-knowledge](mini-rag-knowledge/) | RAG pipeline (embed→retrieve→generate), chunking, reranking, hybrid search, guardrails | LangChain, LlamaIndex |
| [mini-agent-runtime](mini-agent-runtime/) | ReAct agent, tool use/function calling, planning (ReWOO/plan-execute), memory systems | OpenAI Agents, LangGraph |
| [mini-multimodal-ai](mini-multimodal-ai/) | CLIP contrastive, image generation (diffusion), VLLaMA, audio (Whisper), video understanding | CLIP, Stable Diffusion, Whisper |
| [mini-ai-system-software](mini-ai-system-software/) | CUDA kernel sim (grid/block/thread), GPU memory hierarchy, NCCL collectives, flash attention | CUDA Programming Guide, NCCL |
| [mini-ai-product-system](mini-ai-product-system/) | Recommendation (emb+recall+rank), chatbot (RLHF pipeline), copilot (context+completion) | Netflix/ByteDance arch |

## Design Philosophy

- **Zero external dependencies** — pure C (C99/C11), only `libc` and `libm`
- **Self-contained modules** — each directory has its own `Makefile`, `include/`, `src/`, `examples/`, `demos/`, `tests/`
- **AI simulation in user-space** — educational models of deep learning frameworks, training/inference systems, and AI applications
- **Theory-to-code mapping** — every module includes `docs/` with paper/course-alignment notes
- **Practical demos** — autograd engine, CNN/Transformer inference, RAG pipeline, AI agent runtime, and more

## Building

Each module is standalone. Navigate to a module directory and run:

```bash
cd mini-dl-framework
make all    # build everything
make test   # run tests
```

Requires **GCC** and **GNU Make**.

## Project Structure

```
mini-ai-ml-intelligent/
├── mini-ml-basic/               # Classical Machine Learning
├── mini-dl-framework/           # Deep Learning Framework
├── mini-model-arch/             # Model Architectures
├── mini-training-system/        # Training Systems
├── mini-inference-system/       # Inference Systems
├── mini-rag-knowledge/          # RAG & Knowledge Retrieval
├── mini-agent-runtime/          # AI Agent Runtime
├── mini-multimodal-ai/          # Multimodal AI
├── mini-ai-system-software/     # AI System Software
└── mini-ai-product-system/      # AI Product Systems
```

## License

MIT
