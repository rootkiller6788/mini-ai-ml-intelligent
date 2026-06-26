/*
 * gpu_reduction.c — Parallel Reduction & Prefix Scan
 *
 * Knowledge points (L4-L5):
 *   L5: Tree-based parallel reduction (log₂N depth)
 *   L5: Warp-shuffle reduction (CUDA cooperative groups)
 *   L5: Blelloch work-efficient exclusive scan
 *   L5: Hillis-Steele inclusive scan
 *   L5: Segmented scan with flag array
 *   L4: Amdahl's Law scalability analysis
 *   L3: Bank-conflict-free shared memory addressing
 *
 * References:
 *   Harris, "Optimizing Parallel Reduction in CUDA" (NVIDIA, 2007)
 *   Blelloch, "Prefix Sums and Their Applications" (CMU, 1990)
 *   Amdahl, "Validity of the Single Processor Approach..." (AFIPS, 1967)
 */

#include "gpu_reduction.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

/* ─── Internal helpers ─── */

/* Apply reduce op on two scalars */
static double reduce_apply(ReduceOp op, double a, double b) {
    switch (op) {
    case REDUCE_OP_SUM:  return a + b;
    case REDUCE_OP_PROD: return a * b;
    case REDUCE_OP_MIN:  return (a < b) ? a : b;
    case REDUCE_OP_MAX:  return (a > b) ? a : b;
    case REDUCE_OP_AND:  return (a != 0.0 && b != 0.0) ? 1.0 : 0.0;
    case REDUCE_OP_OR:   return (a != 0.0 || b != 0.0) ? 1.0 : 0.0;
    default: return a;
    }
}

static inline uint32_t next_pow2(uint32_t n) {
    n--;
    n |= n >> 1; n |= n >> 2; n |= n >> 4;
    n |= n >> 8; n |= n >> 16;
    return n + 1;
}

static inline uint32_t ilog2(uint32_t n) {
    uint32_t r = 0;
    while (n >>= 1) r++;
    return r;
}

/* ─── Reduction Plan ─── */

ReductionPlan gpu_reduction_plan(uint32_t numElements,
                                  uint32_t maxBlockSize,
                                  uint32_t smemLimitBytes) {
    ReductionPlan plan;
    plan.numElements = numElements;
    plan.blockSize = maxBlockSize;
    plan.smemPerBlock = smemLimitBytes;
    /* Each thread needs 8 bytes (double), plus padding for bank conflicts */
    if (plan.blockSize * sizeof(double) * 2 > smemLimitBytes)
        plan.blockSize = smemLimitBytes / (2 * sizeof(double));

    /* Align to power of 2 for tree reduction */
    plan.blockSize = next_pow2(plan.blockSize);
    if (plan.blockSize < 32) plan.blockSize = 32;
    if (plan.blockSize > 1024) plan.blockSize = 1024;

    plan.numBlocks = (numElements + plan.blockSize - 1) / plan.blockSize;
    plan.op = REDUCE_OP_SUM;

    /* Throughput estimate: each block does blockSize adds in log2(blockSize)
     * steps, plus load/store. Rough model for A100: 19.5 TFLOPS fp64. */
    double flopsPerBlock = (double)plan.blockSize * (double)ilog2(plan.blockSize);
    double totalFlops = flopsPerBlock * (double)plan.numBlocks;
    plan.throughputEst = totalFlops / 1e9;  /* approximate GFLOPs */

    printf("[REDUCE] Plan: %u elements → %u blocks × %u threads, "
           "smem=%uB, ~%.1f GFLOPs\n",
           numElements, plan.numBlocks, plan.blockSize,
           plan.smemPerBlock, plan.throughputEst);
    return plan;
}

/* ─── Single-pass Tree Reduction ─── */

void gpu_reduction_tree(double *data, uint32_t n,
                         const ReductionPlan *plan,
                         double *result) {
    if (!data || n == 0) { if (result) *result = 0.0; return; }
    if (n == 1) { if (result) *result = data[0]; return; }

    ReduceOp op = plan ? plan->op : REDUCE_OP_SUM;
    /* Work on a copy since we modify in-place */
    double *temp = (double *)malloc(n * sizeof(double));
    memcpy(temp, data, n * sizeof(double));

    /* Tree reduction: stride doubles each level.
     * Level 0: stride=1, pairs (0,1), (2,3), ...
     * Level 1: stride=2, pairs (0,2), (4,6), ...
     * Level L: stride=2^L, pairs reachable */
    uint32_t remaining = n;
    uint32_t level = 0;
    printf("[REDUCE-TREE] %u elements, op=%d\n", n, (int)op);

    while (remaining > 1) {
        uint32_t stride = 1u << level;
        uint32_t pairs = remaining / 2;
        for (uint32_t i = 0; i < pairs; i++) {
            uint32_t dstIdx = i * stride * 2;
            uint32_t srcIdx = dstIdx + stride;
            if (srcIdx < n)
                temp[dstIdx] = reduce_apply(op, temp[dstIdx], temp[srcIdx]);
        }
        /* Handle odd element: carry to next level */
        if (remaining % 2 == 1) {
            uint32_t oddIdx = (remaining - 1) * stride;
            uint32_t lastPairIdx = (pairs > 0) ? (pairs - 1) * stride * 2 : 0;
            if (oddIdx != lastPairIdx && oddIdx < n && lastPairIdx < n)
                temp[lastPairIdx] = reduce_apply(op, temp[lastPairIdx],
                                                   temp[oddIdx]);
        }
        remaining = (remaining + 1) / 2;
        level++;
    }

    if (result) *result = temp[0];
    printf("[REDUCE-TREE] Result after %u levels: %g\n", level, temp[0]);
    free(temp);
}

/* ─── Warp-Shuffle Reduction ─── */

double gpu_warp_reduce(double laneVal, ReduceOp op) {
    /* Simulate __shfl_down_sync: each round, thread i gets value from i+offset.
     * Lane 0 accumulates all 32 values after 5 rounds (offset: 1,2,4,8,16).
     * This eliminates shared memory: data exchanged via registers. */
    double acc = laneVal;

    for (uint32_t offset = 1; offset <= 16; offset <<= 1) {
        /* In real CUDA: acc = reduce_apply(op, acc, __shfl_down_sync(mask, acc, offset));
         * Here we simulate by having each "lane" operate independently. */
        double neighborVal = laneVal; /* placeholder: in real HW this is peer's value */
        acc = reduce_apply(op, acc, neighborVal);
    }

    printf("[WARP-REDUCE] Input=%g, reduced=%g (5 rounds)\n", laneVal, acc);
    return acc;
}

/* ─── Multi-block Reduction ─── */

double gpu_reduce_multi_block(const double *data, uint32_t n,
                               const ReductionPlan *plan) {
    if (!data || n == 0) return 0.0;
    if (n == 1) return data[0];

    uint32_t blockSize = plan ? plan->blockSize : 256;
    ReduceOp op = plan ? plan->op : REDUCE_OP_SUM;
    uint32_t numBlocks = (n + blockSize - 1) / blockSize;

    printf("[REDUCE-MULTIBLOCK] %u elements → %u blocks × %u threads\n",
           n, numBlocks, blockSize);

    /* Pass 1: each block reduces its segment */
    double *partials = (double *)malloc(numBlocks * sizeof(double));
    for (uint32_t b = 0; b < numBlocks; b++) {
        uint32_t start = b * blockSize;
        uint32_t count = (start + blockSize <= n) ? blockSize : n - start;
        double blockResult = data[start];
        for (uint32_t i = 1; i < count; i++)
            blockResult = reduce_apply(op, blockResult, data[start + i]);
        partials[b] = blockResult;
    }

    /* Pass 2: tree-reduce partial results */
    ReductionPlan pass2Plan;
    pass2Plan.op = op;
    pass2Plan.blockSize = (numBlocks < 256) ? next_pow2(numBlocks) : 256;
    pass2Plan.numBlocks = 1;
    pass2Plan.smemPerBlock = pass2Plan.blockSize * sizeof(double) * 2;
    double finalResult;
    gpu_reduction_tree(partials, numBlocks, &pass2Plan, &finalResult);

    free(partials);
    printf("[REDUCE-MULTIBLOCK] Final result: %g\n", finalResult);
    return finalResult;
}

/* ─── Atomic-based Reduction ─── */

double gpu_reduce_atomic(const double *data, uint32_t n, ReduceOp op) {
    if (!data || n == 0) return 0.0;
    double accumulator = (op == REDUCE_OP_PROD) ? 1.0 :
                         (op == REDUCE_OP_MIN)  ? DBL_MAX :
                         (op == REDUCE_OP_MAX)  ? -DBL_MAX :
                         (op == REDUCE_OP_AND)  ? 1.0 : 0.0;

    /* Simulate global atomic: each thread serializes on the same address.
     * Atomic contention: for N threads, O(N) serialized ops.
     * Compare: tree reduction = O(log N) parallel steps. */
    uint32_t contentionCount = 0;
    for (uint32_t i = 0; i < n; i++) {
        /* In real GPU: atomicAdd(&accumulator, data[i]) */
        double old = accumulator;
        accumulator = reduce_apply(op, accumulator, data[i]);
        if (old != accumulator) contentionCount++;
    }

    printf("[REDUCE-ATOMIC] %u elements, %u atomic updates, "
           "result=%g\n", n, contentionCount, accumulator);
    printf("[REDUCE-ATOMIC] Contention: 1 address, %u serialized ops "
           "(vs log₂(%u)=%u for tree)\n",
           n, n, ilog2(next_pow2(n)));
    return accumulator;
}

/* ─── Blelloch Exclusive Scan ─── */

void gpu_scan_blelloch(double *data, uint32_t n,
                        const ScanPlan *plan,
                        double *output) {
    if (!data || n == 0) return;
    if (!output) output = (double *)malloc(n * sizeof(double));

    /* Pad to power of 2 for tree operations */
    uint32_t padded = next_pow2(n);
    double *temp = (double *)calloc(padded * 2, sizeof(double));
    memcpy(temp, data, n * sizeof(double));

    printf("[SCAN-BLELLOCH] %u elements (padded to %u)\n", n, padded);

    /* --- Phase 1: Upsweep (reduce) ---
     * Tree nodes at indices offset, offset+stride, ...
     *     offset = 1, stride = 2:   pairs (0,1), (2,3), ...
     *     offset = 2, stride = 4:   pairs (0,3), (4,7), ...
     * Each step: temp[idx+stride-1] += temp[idx+stride/2-1] */
    for (uint32_t stride = 2; stride <= padded; stride <<= 1) {
        uint32_t half = stride >> 1;
        for (uint32_t i = stride - 1; i < padded; i += stride) {
            temp[i] += temp[i - half];
        }
    }

    /* --- Root set to 0 for exclusive scan --- */
    temp[padded - 1] = 0.0;

    /* --- Phase 2: Downsweep (propagate) ---
     *     offset halves each round, from padded/2 down to 1.
     *     left_child = idx - half, right_child = idx
     *     t = left_child; left_child = right_child; right_child += t */
    for (uint32_t stride = padded; stride >= 2; stride >>= 1) {
        uint32_t half = stride >> 1;
        for (uint32_t i = stride - 1; i < padded; i += stride) {
            double t = temp[i - half];
            temp[i - half] = temp[i];
            temp[i] += t;
        }
    }

    /* Extract result: exclusive scan stored in temp[0..n-1] */
    for (uint32_t i = 0; i < n; i++)
        output[i] = temp[i];

    /* Verify: output[0] should be 0 for exclusive scan */
    printf("[SCAN-BLELLOCH] output[0]=%g output[%u]=%g\n",
           output[0], n - 1, output[n - 1]);
    free(temp);
}

/* ─── Hillis-Steele Inclusive Scan ─── */

void gpu_scan_hillis_steele(double *data, uint32_t n, double *output) {
    if (!data || n == 0) return;
    memcpy(output, data, n * sizeof(double));

    printf("[SCAN-HILLIS-STEELE] %u elements\n", n);

    /* For d = 0..log2(n)-1:
     *   If i >= 2^d: output[i] += output[i - 2^d]
     * Each step adds element offset-by-2^d.
     * Requires barrier after each step in GPU. */
    uint32_t maxStep = ilog2(n);
    for (uint32_t d = 0; d <= maxStep; d++) {
        uint32_t offset = 1u << d;
        for (uint32_t i = offset; i < n; i++) {
            output[i] += output[i - offset];
        }
    }

    printf("[SCAN-HILLIS-STEELE] Work: %u steps × O(n) = O(n·log₂n)\n",
           maxStep + 1);
}

/* ─── Segmented Scan ─── */

void gpu_scan_segmented(const double *data, const int *flags,
                          uint32_t n, double *output) {
    if (!data || n == 0) return;

    printf("[SCAN-SEGMENTED] %u elements with flags\n", n);

    /* Segmented scan: process each segment independently.
     * flag[i]==1: start new segment, output[i] = data[i].
     * flag[i]==0: continue segment, output[i] = output[i-1] + data[i].
     * i==0 is also treated as a segment start. */
    for (uint32_t i = 0; i < n; i++) {
        if (flags[i] || i == 0) {
            output[i] = data[i];        /* reset at segment boundary */
        } else {
            output[i] = output[i - 1] + data[i];
        }
    }

    uint32_t numSegments = 0;
    for (uint32_t i = 0; i < n; i++)
        if (flags[i] || i == 0) numSegments++;

    printf("[SCAN-SEGMENTED] %u segments detected\n", numSegments);
}

/* ─── Amdahl's Law ─── */

double gpu_amdahl_speedup(double parallelFraction, uint32_t numSMs) {
    if (numSMs == 0) return 1.0;
    if (parallelFraction <= 0.0) return 1.0;
    if (parallelFraction >= 1.0) return (double)numSMs;

    double speedup = 1.0 / ((1.0 - parallelFraction)
                           + parallelFraction / (double)numSMs);

    printf("[AMDAHL] p=%.2f, N=%u → speedup=%.2f× (max=%.0f×)\n",
           parallelFraction, numSMs, speedup, (double)numSMs);
    return speedup;
}

/* ─── Bank-Conflict-Free Offset ─── */

uint32_t gpu_reduction_bankfree_offset(uint32_t tid, uint32_t level,
                                        uint32_t numBanks) {
    /* At tree level L, threads access data[tid] and data[tid + 2^L].
     * Sequential addressing: tid accesses bank[tid * stride % numBanks].
     * Pad each level with numBanks extra elements to shift offsets.
     * Formula: offset = (tid >> level) & (numBanks - 1)
     * This scatters concurrent accesses across all banks. */
    if (numBanks == 0) return 0;
    uint32_t offset = ((tid >> level) ^ (tid << 1)) & (numBanks - 1);
    return offset;
}

/* ─── Coalescing Efficiency ─── */

double gpu_reduction_coalesce_efficiency(uint32_t blockSize,
                                          uint32_t stride) {
    if (stride == 0) return 0.0;

    /* Coalesced: consecutive threads access consecutive elements (stride=1).
     * Strided (stride>1): threads access elements strided apart.
     * Efficiency = elements per cache line that are used / line size.
     * For 128-byte line, 8-byte doubles: 16 doubles per line.
     * With stride S: only 16/S doubles used per line if S ≤ 16.
     * With large stride: 1 double per line → efficiency = 1/16. */
    uint32_t lineSize = 128; /* bytes per L2 cache line */
    uint32_t elemSize = 8;   /* bytes per double */
    uint32_t elemsPerLine = lineSize / elemSize;

    double efficiency;
    if (stride <= elemsPerLine)
        efficiency = (double)elemsPerLine / (double)(stride * elemsPerLine);
    else
        efficiency = 1.0 / (double)elemsPerLine;

    if (efficiency > 1.0) efficiency = 1.0;

    printf("[COALESCE-EFF] blockSize=%u stride=%u → efficiency=%.2f%%\n",
           blockSize, stride, efficiency * 100.0);
    return efficiency;
}
