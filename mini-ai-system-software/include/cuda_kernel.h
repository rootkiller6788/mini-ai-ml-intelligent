#ifndef CUDA_KERNEL_H
#define CUDA_KERNEL_H

#include <stdint.h>
#include <stddef.h>

/* --- dim3: grid/block dimensions --- */
typedef struct {
    uint32_t x, y, z;
} dim3;

#define MK_DIM3(_x, _y, _z) ((dim3){(_x), (_y), (_z)})
#define MK_DIM3_1D(_x)      MK_DIM3((_x), 1, 1)
#define MK_DIM3_2D(_x, _y)  MK_DIM3((_x), (_y), 1)

/* --- Kernel launch configuration --- */
typedef struct {
    dim3   gridDim;    /* blocks per grid */
    dim3   blockDim;   /* threads per block */
    size_t sharedMem;  /* dynamic shared memory in bytes */
    int    streamId;   /* CUDA stream identifier */
} KernelLaunchConfig;

/* --- Thread indexing context --- */
typedef struct {
    dim3 threadIdx;    /* thread index within block  (0..blockDim-1) */
    dim3 blockIdx;     /* block index within grid     (0..gridDim-1)  */
    dim3 blockDim;     /* threads per block */
    dim3 gridDim;      /* blocks per grid   */
    uint32_t warpIdx;  /* warp index within block */
    uint32_t laneIdx;  /* lane index within warp (0..31) */
    uint32_t globalThreadIdx; /* flattened global thread id */
} ThreadContext;

/* --- Memory spaces --- */
typedef enum {
    MEM_GLOBAL  = 0,   /* HBM / DRAM                      */
    MEM_SHARED  = 1,   /* on-chip SRAM (per-SM, per-block) */
    MEM_LOCAL   = 2,   /* thread-local (spills to global)   */
    MEM_CONSTANT = 3,  /* read-only cached global           */
    MEM_TEXTURE  = 4   /* texture memory                    */
} CudaMemorySpace;

/* --- Shared memory descriptor --- */
typedef struct {
    void   *base;
    size_t  size;
    int     isDynamic;  /* __shared__ (static) or extern __shared__ */
} SharedMem;

/* --- Warp / SIMT model --- */
#define CUDA_WARP_SIZE 32

typedef enum {
    WARP_ACTIVE     = 0,
    WARP_INACTIVE   = 1,
    WARP_DIVERGED   = 2,  /* threads in warp took different branches */
    WARP_RECONVERGED = 3
} WarpState;

typedef struct {
    uint32_t warpId;
    uint32_t activeMask;    /* bitmask of active lanes */
    uint32_t divergenceMask; /* lanes that diverged    */
    WarpState state;
} WarpDescriptor;

/* --- Cooperative groups --- */
typedef enum {
    CG_THIS_THREAD    = 0,
    CG_THIS_WARP      = 1,
    CG_THIS_BLOCK     = 2,
    CG_THIS_GRID      = 3,
    CG_THIS_MULTI_GRID = 4
} CoopGroupType;

typedef struct {
    CoopGroupType type;
    uint32_t      size;       /* number of threads in group */
    uint32_t      threadRank; /* rank of calling thread     */
    uint32_t      numGroups;  /* number of groups in parent */
    uint32_t      groupIndex; /* index of this group        */
} CoopGroup;

/* --- Synchronization --- */
typedef enum {
    SYNC_BLOCK   = 0,  /* __syncthreads()        */
    SYNC_WARP    = 1,  /* __syncwarp()           */
    SYNC_GRID    = 2,  /* grid.sync() (cooperative groups) */
    SYNC_DEVICE  = 3   /* cudaDeviceSynchronize() */
} SyncLevel;

/* --- Kernel function type --- */
typedef void (*KernelFunc)(ThreadContext *ctx, void *args);

/* --------------------------------------------------------------------------- */
/*  API                                                                        */
/* --------------------------------------------------------------------------- */

/*  Launch a "kernel" (simulated) with the given config  */
void cuda_kernel_launch_sim(KernelFunc func, void *args,
                            const KernelLaunchConfig *cfg);

/*  Compute thread context from launch config + flat index  */
ThreadContext cuda_thread_context(const KernelLaunchConfig *cfg,
                                  uint32_t flatThreadIdx);

/*  Flat index utilities  */
uint32_t cuda_flat_block_idx(const dim3 *blockIdx, const dim3 *gridDim);
uint32_t cuda_flat_thread_idx(const dim3 *threadIdx, const dim3 *blockDim);
uint32_t cuda_global_thread_idx(const ThreadContext *ctx);

/*  Shared memory allocation (simulated)  */
SharedMem cuda_shmem_alloc(size_t bytes);
void      cuda_shmem_free(SharedMem *smem);

/*  Warp divergence analyser  */
void cuda_warp_divergence_analyze(const uint32_t *branchMask, uint32_t numWarps,
                                  WarpDescriptor *warps, uint32_t *divergedWarps);

/*  Cooperative group helpers  */
CoopGroup cuda_coop_group_create(CoopGroupType type, uint32_t size,
                                  uint32_t threadRank);
int       cuda_coop_group_is_valid(const CoopGroup *cg);

#endif /* CUDA_KERNEL_H */
