#include "cuda_kernel.h"
#include <stdio.h>
#include <stdlib.h>

/* A sample kernel: vector add (simulated per-thread) */
typedef struct {
    double *a;
    double *b;
    double *c;
    uint32_t n;
} VecAddArgs;

static void vecadd_kernel(ThreadContext *ctx, void *args) {
    VecAddArgs *va = (VecAddArgs *)args;
    uint32_t idx = cuda_global_thread_idx(ctx);
    uint32_t total = ctx->gridDim.x * ctx->blockDim.x;
    for (uint32_t i = idx; i < va->n; i += total) {
        va->c[i] = va->a[i] + va->b[i];
    }
}

/* Demonstrate thread indexing */
static void demo_thread_indexing(void) {
    printf("\n=== Thread Indexing Demo ===\n");

    KernelLaunchConfig cfg;
    cfg.gridDim   = MK_DIM3_1D(2);   /* 2 blocks */
    cfg.blockDim  = MK_DIM3_1D(32);  /* 32 threads per block */
    cfg.sharedMem = 0;
    cfg.streamId  = 0;

    double a[64], b[64], c[64];
    for (int i = 0; i < 64; i++) { a[i] = (double)i; b[i] = (double)(i * 2); }

    printf("Launching 2 blocks × 32 threads = %u threads\n",
           2u * 32u);
    printf("gridDim=(2,1,1) blockDim=(32,1,1)\n");

    for (uint32_t i = 0; i < 64; i++) {
        ThreadContext ctx = cuda_thread_context(&cfg, i);
        printf("  global[%2u] blockIdx=(%u,%u,%u) threadIdx=(%u,%u,%u) "
               "warp=%u lane=%u\n",
               ctx.globalThreadIdx,
               ctx.blockIdx.x, ctx.blockIdx.y, ctx.blockIdx.z,
               ctx.threadIdx.x, ctx.threadIdx.y, ctx.threadIdx.z,
               ctx.warpIdx, ctx.laneIdx);
        if (i % 16 == 15) printf("  --- end of warp %u ---\n", ctx.warpIdx);
    }

    VecAddArgs va = { a, b, c, 64 };
    cuda_kernel_launch_sim(vecadd_kernel, &va, &cfg);

    printf("Results (first 8): ");
    for (int i = 0; i < 8; i++) printf("%.0f ", c[i]);
    printf("\n");
}

/* Demonstrate warp divergence */
static void demo_warp_divergence(void) {
    printf("\n=== Warp Divergence Demo ===\n");

    /* Simulate a warp where threads take different branches.
     * branchMask[w]: 1 if thread took path A, 0 if path B.
     * Full convergence = 0xFFFFFFFF (all 1s or all 0s).
     * Divergence = mixed mask. */
    uint32_t branchMasks[4];
    uint32_t numWarps = 4;

    /* Warp 0: all active, no divergence */
    branchMasks[0] = 0xFFFFFFFF;
    /* Warp 1: half/half divergence */
    branchMasks[1] = 0x0000FFFF;
    /* Warp 2: single thread deviates */
    branchMasks[2] = 0xFFFFFFFE;
    /* Warp 3: inactive (all 0) */
    branchMasks[3] = 0x00000000;

    WarpDescriptor warps[4];
    uint32_t diverged = 0;

    cuda_warp_divergence_analyze(branchMasks, numWarps, warps, &diverged);

    for (uint32_t w = 0; w < numWarps; w++) {
        const char *stateStr;
        switch (warps[w].state) {
        case WARP_ACTIVE:     stateStr = "ACTIVE"; break;
        case WARP_DIVERGED:   stateStr = "DIVERGED"; break;
        case WARP_INACTIVE:   stateStr = "INACTIVE"; break;
        default:              stateStr = "?"; break;
        }
        printf("  Warp %u: mask=0x%08X state=%s active=%u diverged=%u\n",
               warps[w].warpId, warps[w].activeMask, stateStr,
               __builtin_popcount(warps[w].activeMask),
               __builtin_popcount(warps[w].divergenceMask));
    }

    printf("\nPerformance impact of warp divergence:\n");
    printf("  Fully converged: 1 instruction issue\n");
    printf("  2-way divergence: 2 instruction issues (50%% utilization per path)\n");
    printf("  N-way divergence: N instruction issues (1/N utilization per path)\n");
}

/* Demonstrate cooperative groups */
static void demo_coop_groups(void) {
    printf("\n=== Cooperative Groups Demo ===\n");

    CoopGroup cg_thread = cuda_coop_group_create(CG_THIS_THREAD, 0, 0);
    printf("  this_thread: size=%u rank=%u valid=%d\n",
           cg_thread.size, cg_thread.threadRank,
           cuda_coop_group_is_valid(&cg_thread));

    CoopGroup cg_warp = cuda_coop_group_create(CG_THIS_WARP, 0, 5);
    printf("  this_warp:   size=%u rank=%u valid=%d\n",
           cg_warp.size, cg_warp.threadRank,
           cuda_coop_group_is_valid(&cg_warp));

    CoopGroup cg_block = cuda_coop_group_create(CG_THIS_BLOCK, 256, 127);
    printf("  this_block:  size=%u rank=%u valid=%d\n",
           cg_block.size, cg_block.threadRank,
           cuda_coop_group_is_valid(&cg_block));

    printf("\nCooperative groups hierarchy:\n");
    printf("  multi-grid > grid > block > warp > thread\n");
    printf("  Size ordering: MCP > gridDim*blockDim > blockDim > 32 > 1\n");
}

int main(void) {
    printf("=== CUDA Kernel Programming Model Demo ===\n");

    demo_thread_indexing();
    demo_warp_divergence();
    demo_coop_groups();

    printf("\n=== Demo complete ===\n");
    return 0;
}
