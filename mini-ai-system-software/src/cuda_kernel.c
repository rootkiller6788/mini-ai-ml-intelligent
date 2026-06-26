#include "cuda_kernel.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* ---- flat index helpers ---- */

uint32_t cuda_flat_block_idx(const dim3 *blockIdx, const dim3 *gridDim) {
    return blockIdx->x + blockIdx->y * gridDim->x
         + blockIdx->z * gridDim->x * gridDim->y;
}

uint32_t cuda_flat_thread_idx(const dim3 *threadIdx, const dim3 *blockDim) {
    return threadIdx->x + threadIdx->y * blockDim->x
         + threadIdx->z * blockDim->x * blockDim->y;
}

uint32_t cuda_global_thread_idx(const ThreadContext *ctx) {
    return ctx->blockIdx.x * ctx->blockDim.x + ctx->threadIdx.x
         + (ctx->blockIdx.y * ctx->blockDim.y + ctx->threadIdx.y)
         * ctx->gridDim.x * ctx->blockDim.x
         + (ctx->blockIdx.z * ctx->blockDim.z + ctx->threadIdx.z)
         * ctx->gridDim.x * ctx->blockDim.x
         * ctx->gridDim.y * ctx->blockDim.y;
}

/* ---- thread context ---- */

ThreadContext cuda_thread_context(const KernelLaunchConfig *cfg,
                                   uint32_t flatThreadIdx) {
    ThreadContext ctx;
    uint32_t threadsPerBlock = cfg->blockDim.x * cfg->blockDim.y * cfg->blockDim.z;
    uint32_t totalBlocks = cfg->gridDim.x * cfg->gridDim.y * cfg->gridDim.z;

    uint32_t blockFlat = flatThreadIdx / threadsPerBlock;
    uint32_t threadFlat = flatThreadIdx % threadsPerBlock;

    ctx.blockIdx.x = blockFlat % cfg->gridDim.x;
    ctx.blockIdx.y = (blockFlat / cfg->gridDim.x) % cfg->gridDim.y;
    ctx.blockIdx.z = blockFlat / (cfg->gridDim.x * cfg->gridDim.y);

    ctx.threadIdx.x = threadFlat % cfg->blockDim.x;
    ctx.threadIdx.y = (threadFlat / cfg->blockDim.x) % cfg->blockDim.y;
    ctx.threadIdx.z = threadFlat / (cfg->blockDim.x * cfg->blockDim.y);

    ctx.blockDim = cfg->blockDim;
    ctx.gridDim  = cfg->gridDim;
    ctx.globalThreadIdx = flatThreadIdx;

    uint32_t tidInBlock = cuda_flat_thread_idx(&ctx.threadIdx, &ctx.blockDim);
    ctx.warpIdx = tidInBlock / CUDA_WARP_SIZE;
    ctx.laneIdx = tidInBlock % CUDA_WARP_SIZE;

    return ctx;
}

/* ---- simulated kernel launch ---- */

void cuda_kernel_launch_sim(KernelFunc func, void *args,
                             const KernelLaunchConfig *cfg) {
    uint32_t threadsPerBlock = cfg->blockDim.x * cfg->blockDim.y * cfg->blockDim.z;
    uint32_t numBlocks = cfg->gridDim.x * cfg->gridDim.y * cfg->gridDim.z;
    uint32_t totalThreads = numBlocks * threadsPerBlock;
    printf("[CUDA] Launching kernel: %u blocks × %u threads = %u total\n",
           numBlocks, threadsPerBlock, totalThreads);
    printf("[CUDA] grid=(%u,%u,%u) block=(%u,%u,%u) shared_mem=%zu bytes\n",
           cfg->gridDim.x, cfg->gridDim.y, cfg->gridDim.z,
           cfg->blockDim.x, cfg->blockDim.y, cfg->blockDim.z,
           cfg->sharedMem);
    for (uint32_t i = 0; i < totalThreads; i++) {
        ThreadContext ctx = cuda_thread_context(cfg, i);
        func(&ctx, args);
    }
}

/* ---- shared memory ---- */

SharedMem cuda_shmem_alloc(size_t bytes) {
    SharedMem sm;
    sm.base = malloc(bytes);
    sm.size = sm.base ? bytes : 0;
    sm.isDynamic = 0;
    if (sm.base) memset(sm.base, 0, bytes);
    return sm;
}

void cuda_shmem_free(SharedMem *smem) {
    if (smem && smem->base) {
        free(smem->base);
        smem->base = NULL;
        smem->size = 0;
    }
}

/* ---- warp divergence analyser ---- */

void cuda_warp_divergence_analyze(const uint32_t *branchMask,
                                   uint32_t numWarps,
                                   WarpDescriptor *warps,
                                   uint32_t *divergedWarps) {
    uint32_t divergedCount = 0;
    for (uint32_t w = 0; w < numWarps; w++) {
        uint32_t mask = branchMask[w];
        uint32_t active = __builtin_popcount(mask);
        warps[w].warpId = w;
        warps[w].activeMask = mask;
        if (active < CUDA_WARP_SIZE && active > 0) {
            warps[w].state = WARP_DIVERGED;
            warps[w].divergenceMask = (~mask) & ((1u << CUDA_WARP_SIZE) - 1);
            divergedCount++;
        } else if (active == 0) {
            warps[w].state = WARP_INACTIVE;
            warps[w].divergenceMask = 0;
        } else {
            warps[w].state = WARP_ACTIVE;
            warps[w].divergenceMask = 0;
        }
    }
    *divergedWarps = divergedCount;
    printf("[WARP] %u/%u warps diverged (%.1f%%)\n",
           divergedCount, numWarps, 100.0 * divergedCount / numWarps);
}

/* ---- cooperative groups ---- */

CoopGroup cuda_coop_group_create(CoopGroupType type, uint32_t size,
                                  uint32_t threadRank) {
    CoopGroup cg;
    cg.type = type;
    cg.size = size;
    cg.threadRank = threadRank;
    cg.numGroups = 1;
    cg.groupIndex = 0;
    switch (type) {
    case CG_THIS_THREAD: cg.size = 1; break;
    case CG_THIS_WARP:   cg.size = CUDA_WARP_SIZE; break;
    case CG_THIS_BLOCK:  /* size defined by blockDim */ break;
    case CG_THIS_GRID:   /* size defined by grid */ break;
    case CG_THIS_MULTI_GRID: break;
    }
    return cg;
}

int cuda_coop_group_is_valid(const CoopGroup *cg) {
    return cg != NULL && cg->size > 0 && cg->threadRank < cg->size;
}

/* ---- sync simulation ---- */

static void sim_syncthreads_block(uint32_t blockThreads) {
    printf("[SYNC] __syncthreads() barrier for %u threads per block\n",
           blockThreads);
}

void cuda_sync_sim(SyncLevel level, const ThreadContext *ctx) {
    switch (level) {
    case SYNC_BLOCK:
        sim_syncthreads_block(ctx->blockDim.x * ctx->blockDim.y * ctx->blockDim.z);
        break;
    case SYNC_WARP:
        printf("[SYNC] __syncwarp() for warp %u, mask=0x%08X\n",
               ctx->warpIdx, (1u << CUDA_WARP_SIZE) - 1);
        break;
    case SYNC_GRID:
        printf("[SYNC] grid.sync() for %u blocks × %u threads\n",
               ctx->gridDim.x * ctx->gridDim.y * ctx->gridDim.z,
               ctx->blockDim.x * ctx->blockDim.y * ctx->blockDim.z);
        break;
    case SYNC_DEVICE:
        printf("[SYNC] cudaDeviceSynchronize()\n");
        break;
    }
}
