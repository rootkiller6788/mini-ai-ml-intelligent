#ifndef GPU_REDUCTION_H
#define GPU_REDUCTION_H

#include <stdint.h>
#include <stddef.h>

/* ───────────────────────────────────────────────────────────────
 * Parallel Reduction & Prefix Scan — GPU Primitives
 *
 * L5 — Algorithms/Methods:
 *   Tree-based parallel reduction (log₂N steps)
 *   Warp-shuffle reduction (__shfl_down_sync)
 *   Blelloch exclusive prefix scan (work-efficient)
 *   Hillis-Steele inclusive prefix scan
 *   Bank-conflict-free shared-memory addressing
 *
 * L4 — Standards/Theorems:
 *   Amdahl's Law scalability model for GPU parallelism
 * ─────────────────────────────────────────────────────────────── */

/* --- Reduction operation --- */
typedef enum {
    REDUCE_OP_SUM  = 0,
    REDUCE_OP_PROD = 1,
    REDUCE_OP_MIN  = 2,
    REDUCE_OP_MAX  = 3,
    REDUCE_OP_AND  = 4,
    REDUCE_OP_OR   = 5
} ReduceOp;

/* --- Reduction plan (launch config for multi-pass reduction) --- */
typedef struct {
    ReduceOp op;
    uint32_t numElements;
    uint32_t blockSize;
    uint32_t numBlocks;
    uint32_t smemPerBlock;
    double   throughputEst;
} ReductionPlan;

/* --- Scan plan --- */
typedef struct {
    enum { SCAN_INCLUSIVE = 0, SCAN_EXCLUSIVE = 1 } scanType;
    uint32_t numElements;
    uint32_t blockSize;
    uint32_t numBlocks;
    int      isSegmented;
} ScanPlan;

/* ─── Reduction API ─── */

/* Plan a multi-pass tree reduction: choose block size given limits */
ReductionPlan gpu_reduction_plan(uint32_t numElements,
                                  uint32_t maxBlockSize,
                                  uint32_t smemLimitBytes);

/* Single-pass tree reduction (CPU-simulated GPU hierarchy).
 * Mirrors: load → shmem → tree-reduce → write.
 * Returns reduction result in data[0]. */
void gpu_reduction_tree(double *data, uint32_t n,
                         const ReductionPlan *plan,
                         double *result);

/* Warp-shuffle reduction (32-thread warp).
 * Simulates __shfl_down_sync with offset doubling each round.
 * O(log₂32) = 5 steps. */
double gpu_warp_reduce(double laneVal, ReduceOp op);

/* Multi-pass reduction: block-level tree reduce → final pass over partials */
double gpu_reduce_multi_block(const double *data, uint32_t n,
                               const ReductionPlan *plan);

/* Atomic-based reduction (simulated global atomic).
 * Each thread atomically updates global accumulator.
 * Demonstrates atomic serialization cost vs tree reduction. */
double gpu_reduce_atomic(const double *data, uint32_t n, ReduceOp op);

/* ─── Prefix Scan API ─── */

/* Blelloch work-efficient exclusive scan.
 * Phase 1 (upsweep): binary-tree partial reduction.
 * Phase 2 (downsweep): propagate prefix values.
 * Complexity: O(n) work, O(log₂n) depth, 2(n-1) additions. */
void gpu_scan_blelloch(double *data, uint32_t n,
                        const ScanPlan *plan,
                        double *output);

/* Hillis-Steele inclusive scan (simpler, more barriers).
 * Step d: thread i adds element max(0,i-2^d).
 * Complexity: O(n·log₂n) work. */
void gpu_scan_hillis_steele(double *data, uint32_t n, double *output);

/* Segmented scan: flag[i]==1 resets prefix to data[i].
 * Each segment scanned independently. */
void gpu_scan_segmented(const double *data, const int *flags,
                          uint32_t n, double *output);

/* ─── Performance Analysis ─── */

/* Amdahl's Law: speedup = 1 / ((1-p) + p/N)
 * p = parallel fraction, N = number of processors (SMs). */
double gpu_amdahl_speedup(double parallelFraction, uint32_t numSMs);

/* Bank-conflict-free padding for tree reduction.
 * Without padding, sequential addressing causes stride conflicts.
 * Returns offset to add for given thread at given tree level. */
uint32_t gpu_reduction_bankfree_offset(uint32_t tid, uint32_t level,
                                        uint32_t numBanks);

/* Coalescing efficiency: 1.0 = fully coalesced, 0.0 = fully strided */
double gpu_reduction_coalesce_efficiency(uint32_t blockSize,
                                          uint32_t stride);

#endif /* GPU_REDUCTION_H */
