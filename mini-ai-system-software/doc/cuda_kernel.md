# CUDA 内核编程模型

## 1. 线程层次结构

CUDA 将并行任务组织为 **grid → block → warp → thread** 的层次结构。

| 层次 | 数量 | 约束 |
|------|------|------|
| Grid | 1~65535 (x) | 三维 `<<<grid,block>>>` |
| Block | 1~1024 threads | 三维, max 1024 total |
| Warp | 32 threads | 最小执行单元 |
| Thread | 1 | 最小逻辑单元 |

### 内置变量

```cuda
dim3 threadIdx;   // 线程在 block 内的索引 (0 ~ blockDim-1)
dim3 blockIdx;    // block 在 grid 内的索引  (0 ~ gridDim-1)
dim3 blockDim;    // 每个 block 的线程数
dim3 gridDim;     // grid 中 block 的数量
```

全局线程 ID 计算:
```cuda
int tid = blockIdx.x * blockDim.x + threadIdx.x;           // 1D
int tid = blockIdx.x * blockDim.x + threadIdx.x
        + (blockIdx.y * blockDim.y + threadIdx.y)
        * gridDim.x * blockDim.x;                          // 2D
```

## 2. 内存层次结构

| 内存类型 | 位置 | 访问范围 | 速度 | 容量 |
|----------|------|----------|------|------|
| 寄存器 | SM | Per-Thread | ~0 cycles | 255×32-bit/SM |
| 共享内存 | SM | Per-Block | ~28 cycles | 48-164 KB/SM |
| L1 缓存 | SM | Per-SM | ~28 cycles | 128 KB (A100) |
| L2 缓存 | Chip | Device-wide | ~200 cycles | 40 MB (A100) |
| 全局内存 (HBM) | Chip | Device-wide | ~400 cycles | 40-80 GB |

### 共享内存声明

```cuda
__shared__ float tile[BLOCK_SIZE][BLOCK_SIZE];  // 静态
extern __shared__ float dynamic_tile[];          // 动态 (通过kernel启动参数传大小)
```

## 3. SIMT 执行模型

**SIMT** (Single Instruction, Multiple Threads): 同一 warp 的所有线程同时执行相同指令。

### Warp

- 32 个线程组成一个 warp
- 同一 warp 执行相同的指令流
- GPU 调度器以 warp 为单位发射指令

### Warp 发散 (Divergence)

当 warp 内线程走不同分支时:

```cuda
if (threadIdx.x % 2 == 0) {
    // 偶数线程执行
} else {
    // 奇数线程执行 (等待偶数线程完成)
}
// 执行时间 = path_A + path_B (序列化)
```

- 完全收敛: 1 instruction issue
- 2-way: 周期翻倍
- N-way: 周期 × N

**解决**: 尽量减少 warp 内分支，或调整数据布局使同 warp 线程走同一分支。

## 4. 线程同步

### __syncthreads()

```cuda
__global__ void kernel(float *data) {
    data[threadIdx.x] *= 2.0f;
    __syncthreads();  // block 内所有线程在此等待
    data[threadIdx.x] += 1.0f;
}
```

- **作用**: block 内 barrier
- **粒度**: warp 级 (volta+) 或 block 级
- **限制**: 只能用于 block 内, 不可跨 block

### __syncwarp()

Volta 架构引入的 warp 级同步。

### Cooperative Groups

```cuda
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

cg::thread_block block = cg::this_thread_block();  // block
cg::thread_group tile = cg::tiled_partition<32>(block);  // warp
cg::grid_group grid = cg::this_grid();  // grid
grid.sync();  // grid 级同步
```

## 5. Kernel Launch

```cuda
kernel<<<grid, block, shared_mem_bytes, stream>>>(args);
```

| 参数 | 说明 |
|------|------|
| `grid` | dim3 grid 维度 |
| `block` | dim3 block 维度 |
| `shared_mem_bytes` | 动态共享内存字节数 |
| `stream` | CUDA stream (0 = default) |

## 6. 性能注意事项

1. **占用率 (Occupancy)**: active warps / max warps per SM
   - 寄存器压力: 过多局部变量降低占用率
   - 共享内存压力: 大 tile 减少 block 数量
2. **全局内存合并**: 同 warp 连续地址 = 1 次事务
3. **Bank 冲突**: 同 warp 多线程访问同一 bank = 串行化
4. **指令级并行 (ILP)**: 独立指令可流水执行
5. **异步拷贝 (cp.async)**: A100 SM80+ 支持异步加载到 shared memory
