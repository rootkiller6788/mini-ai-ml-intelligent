# GPU 显存模型

## 1. 内存层次结构 (以 NVIDIA A100 为例)

```
Register File        [255 KB/SM]  ← 最快, per-thread, ~0 cycle
    └─ L1 / SRAM     [192 KB/SM]  ← 共享内存 + L1 缓存, ~28 cycles
        └─ L2 Cache  [40 MB]      ← 统一 L2, ~200 cycles
            └─ HBM2e [40/80 GB]   ← 全局内存, ~400 cycles
                └─ NVMe SSD       ← 主机/GPU 交换 (UVA)
```

### 每种内存的参数

| 类型 | 大小 | 带宽 | 延迟 |
|------|------|------|------|
| Register File | 65536×32-bit/SM | ~10 TB/s aggregate | ~0 cycles |
| Shared Memory | 164 KB/SM (A100) | ~10 TB/s aggregate | ~28 cycles |
| L1 Cache | 128 KB/SM (configurable) | 与共享内存共用 | ~28 cycles |
| L2 Cache | 40 MB (A100) | ~4 TB/s | ~200 cycles |
| HBM2e | 40/80 GB | 1555/2039 GB/s | ~400 cycles |

## 2. 显存合并 (Memory Coalescing)

### 原理

同一 warp 的 32 个线程访问全局内存时，如果地址落在同一 128-byte 对齐段内，硬件合并为**一次**内存事务。

### 示例

```cuda
// 完美合并 — 1×128B 事务
float x = data[threadIdx.x];         // 线程 0→addr[0], 1→addr[4], ..., 31→addr[124]

// 跨步访问 (stride=2) — 2×128B 事务
float x = data[threadIdx.x * 2];     // 线程 0→addr[0], 1→addr[8], ..., 31→addr[248]

// 随机访问 — 最多 32 次事务
float x = data[permutation[threadIdx.x]];
```

### L2 Cache Line

- A100: 64-byte sectors (8 sectors × 32B = 256B cache line)
- 写入合并: 同一 sector 的写入合并为一笔
- 关键: **线程 0 到 线程 31 应当连续访问 `addr[0]`, `addr[4]`, ..., `addr[124]`**

### 规则

1. 基础地址对齐到 128-byte 边界
2. 线程 0～31 访问地址递增 (间隔 4/8/16 字节均可)
3. 每条线程访问 1/2/4/8 字节

## 3. Shared Memory Bank 冲突

SM 有 **32 个 bank**，每个 bank 宽度为 4 字节。同一 warp 的多条线程访问同一 bank 的不同地址 → 串行化。

### 无冲突 (1-cycle)

```cuda
__shared__ float data[256];
float x = data[threadIdx.x];  // 线程 i → bank i%32, 无重复
```

### 2-way 冲突 (2 cycles)

```cuda
float x = data[threadIdx.x * 2];  // stride-2: 线程 0(bank-0), 线程 16(bank-0)
```

### n-way 冲突 (n cycles)

```cuda
float x = data[threadIdx.x * 32];  // stride-32: 所有线程命中 bank-0
// 最坏: 32-way conflict → 1/32 带宽
```

### 广播

32 个线程访问同一 bank 的**同一地址** → 广播，1 cycle (硬件多播)。

### 避免 Bank 冲突

- **步幅为奇数** (≈ 无冲突)
- **数据填充**: `__shared__ float data[256 + PAD]` 填充消除 stride-bank 对齐
- **交换维度**: row 存储 → column 访问时填充行

## 4. 钉扎内存 (Pinned / Page-Locked Memory)

CPU 页表锁定的主机内存，允许 GPU 直接通过 DMA 访问，无需 CPU 拷贝到中间缓冲区。

```cuda
cudaHostAlloc((void**)&ptr, bytes, cudaHostAllocDefault);
cudaMemcpy(dst, ptr, bytes, cudaMemcpyHostToDevice);  // ~25 GB/s (PCIe 4.0)
```

- **速度**: pinned >> pageable (pageable 需要先拷贝到 pinned 临时缓冲区)
- **内存占用**: 锁定页不可换出，占用物理 RAM
- **使用模式**: 临时分配 pinned buffer 做快速传输

## 5. 统一内存 (UVA / Managed Memory)

CPU 和 GPU 共享同一虚拟地址空间，数据自动按需迁移。

```cuda
cudaMallocManaged(&ptr, bytes);
int device = -1;
cudaMemPrefetchAsync(ptr, bytes, device);  // 预取到 GPU
```

### 迁移开销

- 页错误 → 迁移 2 MB 页 (64 KB 子页面 on Pascal+)
- 频繁换向 (CPU↔GPU) 产生"乒乓效应"
- **建议**: 使用 `cudaMemPrefetchAsync` 手动指定位置

## 6. cuBLAS Tensor 布局

cuBLAS 使用列优先 (column-major / Fortran-order) 作为默认布局:

```
Row-major (C):   A[0,0],A[0,1],A[0,2],A[1,0],...
Column-major (F): A[0,0],A[1,0],A[2,0],A[0,1],...

cuBLAS GEMM:
  C = α · op(A) · op(B) + β · C
  op(X) ∈ {N (no transpose), T (transpose)}
```

### 重要

- `lda` (leading dimension) ≥ 列数 (col-major) 或 行数 (row-major)
- `CUBLAS_OP_N` = 不转置 = col-major 的 A 或 row-major 的 A^T
- 常见错误: 混淆 row-major 和 col-major 导致结果转置

## 7. 性能调优 checklist

1. [ ] 全局内存访问是否合并? (同 warp 连续地址)
2. [ ] shared memory 是否有 bank 冲突?
3. [ ] 寄存器用量是否过高? (影响占用率)
4. [ ] 共享内存用量是否合适? (block 数量)
5. [ ] 数据是否预取到最近的内存层级?
6. [ ] 是否使用了 pinned memory 加速传输?
7. [ ] 对于 large graph, 是否启用了 UVA / 分页?
