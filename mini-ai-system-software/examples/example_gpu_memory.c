#include "gpu_memory.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Demonstrate memory coalescing */
static void demo_coalescing(void) {
    printf("\n=== Memory Coalescing Demo ===\n");
    printf("L2 cache line / memory transaction: %d bytes\n", COALESCE_GRANULARITY);

    /* Case 1: perfectly coalesced — consecutive floats (4 bytes each),
     * 32 threads accessing 128 contiguous bytes = 1 transaction */
    printf("\n--- Case 1: Perfect coalescing ---\n");
    uint32_t addrs_coalesced[32];
    for (uint32_t i = 0; i < 32; i++)
        addrs_coalesced[i] = i * 4;  /* consecutive 4-byte elements */
    CoalesceReport r1 = gpu_coalesce_analyze(addrs_coalesced, 32);
    printf("  Efficiency: %.0f%% (ideal for A100)\n", 100.0 / r1.coalesceRatio);
    printf("  Transactions: %u (best: 1)\n", r1.coalescedSegments);

    /* Case 2: strided access — every 2nd element, half wasted */
    printf("\n--- Case 2: Strided access (stride=2) ---\n");
    uint32_t addrs_strided[32];
    for (uint32_t i = 0; i < 32; i++)
        addrs_strided[i] = i * 8;  /* skip every other float */
    CoalesceReport r2 = gpu_coalesce_analyze(addrs_strided, 32);
    printf("  Efficiency: %.0f%%\n", 100.0 / r2.coalesceRatio);

    /* Case 3: random access — worst case */
    printf("\n--- Case 3: Random access ---\n");
    uint32_t addrs_random[32] = {
        0, 128, 256, 64, 192, 320, 48, 144, 272, 80, 208, 16,
        112, 240, 352, 32, 160, 288, 96, 224, 336, 8, 136, 264,
        56, 184, 312, 72, 200, 328, 24, 152
    };
    CoalesceReport r3 = gpu_coalesce_analyze(addrs_random, 32);
    printf("  Efficiency: %.0f%% (worst case)\n", 100.0 / r3.coalesceRatio);

    printf("\nCoalescing rules:\n");
    printf("  - Consecutive threads must access consecutive addresses\n");
    printf("  - Align warp base address to 128-byte boundary\n");
    printf("  - Each thread accesses 1/2/4/8/16 bytes for best results\n");
}

/* Demonstrate bank conflicts */
static void demo_bank_conflicts(void) {
    printf("\n=== Shared Memory Bank Conflict Demo ===\n");
    printf("Banks: %d, Bank width: %d bytes\n",
           GPU_SHMEM_NUM_BANKS, GPU_SHMEM_BANK_WIDTH);

    /* Case 1: no conflict — each thread to a different bank */
    printf("\n--- Case 1: No bank conflicts ---\n");
    uint32_t addrs_noconflict[32];
    for (uint32_t i = 0; i < 32; i++)
        addrs_noconflict[i] = i * 4;  /* 4 bytes apart, 32 banks */
    BankConflictReport b1 = gpu_bank_conflict_analyze(addrs_noconflict, 32);
    printf("  Max conflict depth: %u (1 = perfect)\n", b1.maxConflictDepth);

    /* Case 2: 2-way conflict — threads 0 and 16 hit bank 0 */
    printf("\n--- Case 2: 2-way bank conflict ---\n");
    uint32_t addrs_2way[32];
    for (uint32_t i = 0; i < 32; i++)
        addrs_2way[i] = (i % 16) * GPU_SHMEM_BANK_WIDTH;  /* 0-15 repeat */
    BankConflictReport b2 = gpu_bank_conflict_analyze(addrs_2way, 32);
    printf("  Max conflict depth: %u\n", b2.maxConflictDepth);

    /* Case 3: 32-way conflict — all threads hit the same bank */
    printf("\n--- Case 3: 32-way bank conflict (broadcast pattern) ---\n");
    uint32_t addrs_32way[32];
    for (uint32_t i = 0; i < 32; i++)
        addrs_32way[i] = 0;  /* all threads access same address — broadcast */
    BankConflictReport b3 = gpu_bank_conflict_analyze(addrs_32way, 32);
    printf("  Max conflict depth: %u (broadcast handled by multicast)\n",
           b3.maxConflictDepth);
    printf("  Note: Same-address broadcast is 1 cycle on modern GPUs\n");

    printf("\nBank conflict rules:\n");
    printf("  - Access different banks per thread for 1-cycle access\n");
    printf("  - Stride of 32 banks or multiples avoids conflicts\n");
    printf("  - Broadcast (same address) is 1 cycle via multicast\n");
}

/* Demonstrate cuBLAS tensor layout */
static void demo_cublas_layout(void) {
    printf("\n=== cuBLAS Tensor Layout Demo ===\n");

    CuBlasTensor tA, tB;
    tA.layout   = CUBLAS_ROW_MAJOR;
    tA.rows     = 3;
    tA.cols     = 3;
    tA.ld       = 3;
    tA.elemSize = sizeof(double);
    tA.data     = (double *)calloc(9, sizeof(double));

    tB.layout   = CUBLAS_COL_MAJOR;
    tB.rows     = 3;
    tB.cols     = 3;
    tB.ld       = 3;
    tB.elemSize = sizeof(double);
    tB.data     = (double *)calloc(9, sizeof(double));

    double vals[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    for (uint32_t i = 0; i < 9; i++) {
        tA.data[i] = vals[i];
        tB.data[i] = vals[i];
    }

    printf("Row-major (CUBLAS_OP_N):\n");
    for (uint32_t r = 0; r < 3; r++) {
        printf("  ");
        for (uint32_t c = 0; c < 3; c++)
            printf("%.0f ", cublas_tensor_get(&tA, r, c));
        printf("\n");
    }

    printf("Column-major (CUBLAS_OP_T):\n");
    for (uint32_t r = 0; r < 3; r++) {
        printf("  ");
        for (uint32_t c = 0; c < 3; c++)
            printf("%.0f ", cublas_tensor_get(&tB, r, c));
        printf("\n");
    }

    printf("\nNote: same memory layout gives different logical views.\n");
    printf("cuBLAS GEMM: C=α·op(A)·op(B)+β·C\n");
    printf("  op(X): N (no transpose, row-major) / T (transpose, col-major)\n");

    cublas_tensor_free(&tA);
    cublas_tensor_free(&tB);
}

/* Demonstrate unified memory */
static void demo_unified_memory(void) {
    printf("\n=== Unified Memory (UVA) Demo ===\n");

    UnifiedMem um = gpu_unified_alloc(1024 * 1024, 0);
    printf("  Preferred: device, Current: %s\n",
           um.currentLocation ? "host" : "device");

    gpu_unified_prefetch(&um, 1);
    printf("  After prefetch to host: current=%s, migrations=%d\n",
           um.currentLocation ? "host" : "device", um.migrationCount);

    gpu_unified_prefetch(&um, 0);
    printf("  After prefetch to device: current=%s, migrations=%d\n",
           um.currentLocation ? "host" : "device", um.migrationCount);

    printf("\nUVA advantages:\n");
    printf("  - Single pointer for CPU and GPU\n");
    printf("  - Automatic page migration on access\n");
    printf("  - Oversubscription supported (swap to CPU)\n");

    gpu_unified_free(&um);
}

int main(void) {
    printf("=== GPU Memory Hierarchy Demo ===\n");

    demo_coalescing();
    demo_bank_conflicts();
    demo_cublas_layout();
    demo_unified_memory();

    printf("\n=== Demo complete ===\n");
    return 0;
}
