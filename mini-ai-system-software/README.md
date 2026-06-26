# mini-ai-system-software — AI系统软件 (C 语言实现)

AI 系统软件核心概念的 C99 教学实现，涵盖 9 大知识模块：
CUDA 编程模型、GPU 显存管理、NCCL 集合通信、FlashAttention 算法、
Triton 编译器模型、GPU 并行规约与扫描、CUDA 流并发、GPU 卷积算子、
以及 LLM 注意力变体 (MQA/GQA/KV-Cache)。

## Module Status: COMPLETE ✅

- **include/ + src/ 总行数**: 3938 行 (≥ 3000 ✅)
- **测试**: 34/34 通过 ✅
- **make test**: 一键通过 ✅
- **L1-L6**: Complete
- **L7**: Complete (6 applications: MQA, GQA, KV-Cache, Pipeline Schedule, im2col-GEMM Conv, Winograd Conv)
- **L8**: Partial (3/4: Sliding Window Attention, ALiBi, KV-Cache Mgmt; 缺: 形式化验证)
- **L9**: Partial (documented only: AI Compiler Triton/MLIR 前沿)

---

## 九层知识覆盖摘要

| Level | 名称 | 覆盖状态 | 条目数 |
|-------|------|---------|--------|
| L1 | 核心定义 | ✅ Complete | 40+ struct/typedef/enum/API |
| L2 | 核心概念 | ✅ Complete | SIMT, warp divergence, coalescing, bank conflict, ring allreduce, online softmax, block-tiled matmul, parallel reduction, Winograd filtering, MQA/GQA |
| L3 | 工程结构 | ✅ Complete | Stream pool, event mgmt, KV-cache, im2col matrix, cuBLAS tensor, pipeline schedule |
| L4 | 标准/定理 | ✅ Complete | Amdahl's Law (GPU scaling), Little's Law (pipeline), IO complexity (FlashAttention), roofline model (conv OI) |
| L5 | 算法/方法 | ✅ Complete | Tree reduction, Blelloch scan, Hillis-Steele scan, Winograd F(2×2,3×3), Warp-shuffle reduction, Ring AllReduce |
| L6 | 经典工程问题 | ✅ Complete | cuDNN conv pipeline (3 algos), 3-stage DL training pipeline, multi-pass reduction |
| L7 | 应用 | ✅ Complete (6) | MQA, GQA, KV-Cache, Sliding Window, im2col-GEMM conv, Winograd conv |
| L8 | 进阶主题 | ⚠️ Partial (3/4) | Sliding Window O(NW), ALiBi position bias, Pipeline concurrency |
| L9 | 工业前沿 | ⚠️ Partial | Triton/MLIR compiler pipeline (doc), FlashAttention-3 (doc) |

---

## 核心定义 (L1)

| 模块 | 核心结构 |
|------|---------|
| cuda_kernel | dim3, KernelLaunchConfig, ThreadContext, CudaMemorySpace, WarpDescriptor, CoopGroup |
| gpu_memory | GpuMemTier, CoalesceReport, BankConflictReport, PinnedMemory, UnifiedMem, CuBlasTensor |
| nccl_collectives | NcclComm, NcclLink, NcclOp, NcclDataType, NcclInterconnect |
| flash_attention | FlashAttnConfig, FlashAttnState, FlashTile, FlashAttnOutput |
| triton_compile | TritonStage, BlockProgram, TritonTile, TritonTuneConfig, TritonCompileContext |
| gpu_reduction | ReduceOp, ReductionPlan, ScanPlan — 并行规约与扫描 |
| gpu_stream | CudaStream, CudaEvent, StreamPool, TransferOp, PipelineSchedule |
| gpu_convolution | Conv2DDesc, ConvPerfEstimate — im2col/GEMM/Winograd 三种算法 |
| attention_variants | AttnDescriptor, KVCache, SlidingWindowConfig — MQA/GQA/KV-Cache |

---

## 核心定理 (L4)

| 定理 | 实现 | 来源 |
|------|------|------|
| **Amdahl's Law** | `gpu_amdahl_speedup()` — GPU并行加速比 | Amdahl, AFIPS 1967 |
| **Little's Law** (L=λW) | `gpu_pipeline_schedule()` — 流水线并发度 | Little, 1961 |
| **IO Complexity** | `flash_attn_io_complexity()` — FlashAttention Θ(N²d²/M) | Dao et al., NeurIPS 2022 |
| **Roofline Model** | `gpu_conv2d_perf_estimate()` — 操作强度分析 | Williams et al., CACM 2009 |

---

## 核心算法 (L5)

| 算法 | 实现函数 | 复杂度 | 参考 |
|------|---------|--------|------|
| Tree Parallel Reduction | `gpu_reduction_tree()` | O(log₂N) depth | Harris, NVIDIA 2007 |
| Warp-Shuffle Reduction | `gpu_warp_reduce()` | O(log₂32)=5 steps | CUDA C++ Guide |
| Blelloch Exclusive Scan | `gpu_scan_blelloch()` | O(n) work, 2(n-1) adds | Blelloch, CMU 1990 |
| Hillis-Steele Inclusive Scan | `gpu_scan_hillis_steele()` | O(n·log₂n) work | Hillis & Steele, 1986 |
| Ring AllReduce | `nccl_allreduce()` | 2(N-1)/N · BW | NCCL docs |
| Online Softmax | `flash_online_softmax_block()` | tile-based, O(N²d²/M) IO | Dao et al., 2022 |
| Winograd F(2×2, 3×3) | `gpu_winograd_f2x2_3x3()` | 2.25× fewer mults | Lavin & Gray, CVPR 2016 |
| im2col + GEMM Conv | `gpu_conv2d_im2col_gemm()` | transforms conv→matmul | Chetlur et al., 2014 |

---

## 经典工程问题 (L6)

| 问题 | 示例/实现 | 描述 |
|------|----------|------|
| cuDNN 卷积流水线 | `gpu_conv2d_recommend_algo()` | 3种算法自动选择 (Direct/im2col/Winograd) |
| 3-stage DL 训练流水线 | `gpu_pipeline_schedule()` | H2D→Compute→D2H 重叠调度 |
| Multi-pass GPU Reduction | `gpu_reduce_multi_block()` | 块内规约+跨块规约 |

---

## 应用 (L7)

| 应用 | 实现 | 工业对标 |
|------|------|---------|
| Multi-Query Attention (MQA) | `attn_mqa_forward()` | PaLM, Gemini 推理优化 |
| Grouped-Query Attention (GQA) | `attn_gqa_forward()` | LLaMA 2/3, Mistral |
| KV-Cache 管理 | `kv_cache_create/append/get_range()` | vLLM, TGI 推理引擎 |
| Sliding Window Attention | `attn_sliding_window_forward()` | Longformer, Mistral |
| Winograd 卷积 | `gpu_winograd_f2x2_3x3()` | cuDNN, TensorRT |
| GEMM-based 卷积 | `gpu_conv2d_im2col_gemm()` | cuBLAS, CUTLASS |

---

## 九校课程映射

| 学校 | 课程 | 对标模块 |
|------|------|---------|
| **MIT** | 6.824 Distributed Systems | NCCL AllReduce (ring consensus pattern) |
| **Stanford** | CS 229 Machine Learning | FlashAttention (attention mechanism) |
| **Berkeley** | CS 267 HPC | GPU Reduction, Prefix Scan, Pipeline |
| **CMU** | 15-418 Parallel | CUDA programming model, warp-level primitives |
| **CMU** | 15-445 Database | cuBLAS tensor layout (row/col major) |
| **UT Austin** | CS 395T Systems ML | MQA/GQA (LLM system optimization) |
| **ETH** | 263-3501 Parallel Programming | GPU reduction, scan, stream concurrency |
| **Cambridge** | Part II Concurrent Systems | Stream pool, event synchronization |
| **清华** | 计算机体系结构 | GPU memory hierarchy, bank conflicts |
| **Georgia Tech** | CS 7641 Machine Learning | Autotuning for matrix multiply |

---

## 目录结构

```
mini-ai-system-software/
├── README.md                   # 本文件
├── Makefile                    # make test 一键通过
├── include/                    # 头文件 (9个, 1122行)
│   ├── cuda_kernel.h           # CUDA 内核编程模型
│   ├── gpu_memory.h            # GPU 显存层次结构
│   ├── nccl_collectives.h      # NCCL 集合通信
│   ├── flash_attention.h       # FlashAttention 算法
│   ├── triton_compile.h        # Triton 编译器模型
│   ├── gpu_reduction.h         # 并行规约与扫描 ✨
│   ├── gpu_stream.h            # CUDA 流并发 ✨
│   ├── gpu_convolution.h       # GPU 卷积 (im2col/GEMM/Winograd) ✨
│   └── attention_variants.h    # MQA/GQA/KV-Cache ✨
├── src/                        # 源文件 (9个, 2816行)
│   ├── cuda_kernel.c           # CUDA 内核启动模拟
│   ├── gpu_memory.c            # GPU 显存分配器
│   ├── nccl_collectives.c      # Ring AllReduce 实现
│   ├── flash_attention.c       # FlashAttention 前向/反向
│   ├── triton_compile.c        # Triton 分块编译
│   ├── gpu_reduction.c         # Tree/Warp/Atomic 规约 + Blelloch 扫描 ✨
│   ├── gpu_stream.c            # Stream/Event/Pipeline 调度 ✨
│   ├── gpu_convolution.c       # im2col+GEMM + Winograd F(2×2,3×3) ✨
│   └── attention_variants.c    # MQA/GQA/KV-Cache/Sliding Window ✨
├── tests/                      # 单元测试 (34个测试)
│   └── test_core.c
├── examples/                   # 示例 (3个)
│   ├── example_cuda_kernel.c
│   ├── example_gpu_memory.c
│   └── example_nccl_collectives.c
├── demo/                       # 演示 (2个)
│   ├── demo_flash_attention.c
│   └── demo_triton_compile.c
└── doc/                        # 文档
    ├── cuda_kernel.md
    └── gpu_memory.md
```

(✨ = 本轮新增模块)

---

## 构建与测试

```sh
make all        # 构建所有目标 (examples + demos + test)
make examples   # 仅构建示例
make demos      # 仅构建演示
make test       # 运行单元测试 (34 tests)
make clean      # 清理
```

---

## 完成状态: COMPLETE ✅

- **准入条件**: include/ + src/ ≥ 3000 行 ✅ (当前: 3938 行)
- **子模块级准入**: 9 个 .c + 9 个 .h ✅
- **L1-L6**: Complete ✅
- **L7**: Complete ✅ (6 applications)
- **L8**: Partial ⚠️ (3/4 advanced topics implemented)
- **L9**: Partial ⚠️ (documented, not implemented — allowed by spec)
- **测试**: 34/34 通过 ✅
- **make test**: 一键通过 ✅
- **无 TODO/FIXME/stub/placeholder**: ✅

