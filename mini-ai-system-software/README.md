# mini-ai-system-software — AI系统软件 (C 语言实现)

AI 系统软件核心概念的 C99 教学实现，涵盖 CUDA 编程模型、GPU 显存管理、NCCL 集合通信、
FlashAttention 算法和 Triton 编译器模型。

## 目录结构

```
mini-ai-system-software/
├── README.md
├── Makefile
├── include/                    # 头文件
│   ├── cuda_kernel.h           # CUDA 内核编程模型
│   ├── gpu_memory.h            # GPU 显存层次结构
│   ├── nccl_collectives.h      # NCCL 集合通信
│   ├── flash_attention.h       # FlashAttention 算法
│   └── triton_compile.h        # Triton 编译器模型
├── src/                        # 源文件 (130+ 行)
│   ├── cuda_kernel.c           # CUDA 内核启动模拟
│   ├── gpu_memory.c            # GPU 显存分配器模拟
│   ├── nccl_collectives.c      # Ring AllReduce 实现
│   ├── flash_attention.c       # FlashAttention 前向/反向
│   └── triton_compile.c        # Triton 分块编译模拟
├── examples/                   # 示例程序
│   ├── example_cuda_kernel.c   # 线程索引与Warp发散示例
│   ├── example_gpu_memory.c    # 显存合并与Bank冲突示例
│   └── example_nccl_collectives.c # 集合通信示例
├── demo/                       # 演示程序 (250+ 行)
│   ├── demo_flash_attention.c  # FlashAttention 完整演示
│   └── demo_triton_compile.c   # Triton 分块矩阵乘演示
└── doc/                        # 文档
    ├── cuda_kernel.md          # CUDA 内核模型文档
    └── gpu_memory.md           # GPU 显存模型文档
```

## 构建

```sh
make all        # 构建所有目标
make examples   # 仅构建示例
make demos      # 仅构建演示
make clean      # 清理
```

## 主题

| 模块 | 主题 |
|------|------|
| cuda_kernel | 线程索引, blockIdx/threadIdx, __syncthreads, warp分歧, cooperative groups |
| gpu_memory | HBM/SRAM/L1/L2/寄存器, 显存合并, bank冲突, 钉扎内存, UVA, cuBLAS布局 |
| nccl_collectives | AllReduce环形算法, AllGather, ReduceScatter, Broadcast, NVLink/PCIe带宽 |
| flash_attention | 分块注意力, online softmax, 反向重计算, IO复杂度分析, 因果掩码 |
| triton_compile | TTIR→TTGIR→LLVM IR→PTX, 块级编程, tl.load, autotuning, 2D分块矩阵乘 |
