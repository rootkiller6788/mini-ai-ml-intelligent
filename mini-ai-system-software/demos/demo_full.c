/*
 * mini-ai-system-software — Full Demo: AI System Software
 *
 * Demonstrates: CUDA kernel simulation, GPU memory, NCCL collectives,
 *               Flash Attention, Triton compilation.
 */
#include "../include/cuda_kernel.h"
#include "../include/gpu_memory.h"
#include "../include/nccl_collectives.h"
#include "../include/flash_attention.h"
#include "../include/triton_compile.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void demo_kernel_func(ThreadContext *ctx, void *args) {
    double *data = (double *)args;
    uint32_t idx = cuda_global_thread_idx(ctx);
    data[idx] = (double)(ctx->threadIdx.x + ctx->threadIdx.y + ctx->threadIdx.z);
}

int main(void) {
    printf("=== mini-ai-system-software: AI System Software Demo ===\n\n");

    /* Step 1: CUDA Kernel Simulation */
    printf("-- Step 1: CUDA Kernel Simulation --\n");
    KernelLaunchConfig cfg;
    cfg.gridDim = MK_DIM3_2D(2, 2);
    cfg.blockDim = MK_DIM3_2D(4, 4);
    cfg.sharedMem = 0;
    cfg.streamId = 0;
    ThreadContext ctx = cuda_thread_context(&cfg, 5);
    printf("Thread context: threadIdx=(%d,%d,%d), blockIdx=(%d,%d,%d)\n",
           ctx.threadIdx.x, ctx.threadIdx.y, ctx.threadIdx.z,
           ctx.blockIdx.x, ctx.blockIdx.y, ctx.blockIdx.z);
    printf("  globalThreadIdx=%d, warpIdx=%d, laneIdx=%d\n",
           ctx.globalThreadIdx, ctx.warpIdx, ctx.laneIdx);

    uint32_t flat = cuda_flat_thread_idx(&ctx.threadIdx, &ctx.blockDim);
    printf("Flat thread index: %d\n", flat);

    CoopGroup cg = cuda_coop_group_create(CG_THIS_WARP, 32, 5);
    printf("Cooperative group: type=%d, size=%d, rank=%d, valid=%d\n",
           cg.type, cg.size, cg.threadRank, cuda_coop_group_is_valid(&cg));

    double kernel_data[64] = {0};
    cuda_kernel_launch_sim(demo_kernel_func, kernel_data, &cfg);
    printf("Kernel launched: data[0]=%.0f, data[15]=%.0f\n", kernel_data[0], kernel_data[15]);

    SharedMem smem = cuda_shmem_alloc(4096);
    printf("Shared memory: %zu bytes @ %p, dynamic=%d\n", smem.size, smem.base, smem.isDynamic);
    cuda_shmem_free(&smem);

    /* Step 2: GPU Memory Management */
    printf("\n-- Step 2: GPU Memory Management --\n");
    uint32_t addrs[] = {0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60,
                        64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124};
    CoalesceReport cr = gpu_coalesce_analyze(addrs, 32);
    printf("Coalesce analysis: maxSeg=%d, coalesced=%d, ratio=%.2f, unaligned=%d\n",
           cr.maxSegments, cr.coalescedSegments, cr.coalesceRatio, cr.unalignedCount);

    uint32_t conflict_addrs[] = {0, 4, 0, 4, 8, 8, 12, 12};
    BankConflictReport bcr = gpu_bank_conflict_analyze(conflict_addrs, 8);
    printf("Bank conflicts: accesses=%d, conflicts=%d, maxDepth=%d\n",
           bcr.numAccesses, bcr.numConflicts, bcr.maxConflictDepth);

    PinnedMemory pm = gpu_pinned_alloc(8192);
    printf("Pinned memory: %zu bytes, pinned=%d, mapped=%d\n", pm.size, pm.isPinned, pm.isMapped);
    printf("  transfer rate: %.0f MB/s\n", gpu_pinned_transfer_rate());
    gpu_pinned_free(&pm);

    UnifiedMem um = gpu_unified_alloc(4096, 0);
    printf("Unified memory: %zu bytes, preferred=%d, current=%d\n", um.size, um.preferredLocation, um.currentLocation);
    gpu_unified_free(&um);

    CuBlasTensor t;
    t.layout = CUBLAS_ROW_MAJOR;
    t.rows = 4; t.cols = 4; t.ld = 4; t.elemSize = sizeof(double);
    t.data = calloc(16, sizeof(double));
    cublas_tensor_set(&t, 0, 1, 2.718);
    cublas_tensor_set(&t, 1, 0, 3.141);
    printf("cuBLAS tensor: [0,1]=%.3f, [1,0]=%.3f\n",
           cublas_tensor_get(&t, 0, 1), cublas_tensor_get(&t, 1, 0));
    cublas_tensor_free(&t);

    /* Step 3: NCCL Collectives */
    printf("\n-- Step 3: NCCL Collectives --\n");
    NcclComm comm = nccl_comm_init(4, 0, NCCL_INTER_NVLINK, 900.0);
    printf("NCCL communicator: rank=%d/%d, nvlink bw=%.1f GB/s, pcie bw=%.1f GB/s\n",
           comm.rank, comm.nRanks,
           nccl_nvlink_bandwidth(4), nccl_pcie_bandwidth(4));

    float sendbuf[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    float recvbuf[8] = {0};
    NcclStatus s = nccl_allreduce(sendbuf, recvbuf, 8, NCCL_FLOAT32, NCCL_OP_SUM, &comm);
    printf("AllReduce (SUM, 4 GPUs): result=[");
    for (int i = 0; i < 4; i++) printf("%.0f%s", recvbuf[i], i < 3 ? " " : "");
    printf(" ...], status=%d\n", s);

    double ar_bw = nccl_allreduce_bandwidth(8, 900.0);
    printf("AllReduce bandwidth (8 ranks, 900 GB/s link): %.1f GB/s\n", ar_bw);
    double sync_bytes = nccl_gradient_sync_bytes(7000000000, 2, 8);
    printf("Gradient sync bytes (7B params, fp16, 8 ranks): %.1f GB\n", sync_bytes / 1e9);

    /* Step 4: Flash Attention */
    printf("\n-- Step 4: Flash Attention --\n");
    FlashAttnConfig fac = flash_attn_config_create(1024, 64, 1);
    printf("FlashAttention config: seqLen=%d, headDim=%d, causal=%d, scale=%.6f\n",
           fac.seqLen, fac.headDim, fac.causal, fac.softmaxScale);
    printf("  tile: Br=%d, Bc=%d\n", fac.blockSizeBr, fac.blockSizeBc);

    double io_saved = flash_attn_io_complexity(2048, 64, 20 * 1024 * 1024);
    printf("IO complexity (N=2048, d=64, SRAM=20MB): %.1fx vs naive\n", io_saved);

    /* Allocate a small forward pass */
    uint32_t N = 128, d = 64;
    double *Q = calloc(N * d, sizeof(double));
    double *K = calloc(N * d, sizeof(double));
    double *V = calloc(N * d, sizeof(double));
    for (uint32_t i = 0; i < N * d; i++) { Q[i] = 0.01; K[i] = 0.01; V[i] = 0.01; }
    FlashAttnConfig small_cfg = flash_attn_config_create(N, d, 1);
    FlashAttnOutput *out = flash_attn_forward(Q, K, V, &small_cfg);
    printf("Forward pass: output[0]=%.4f, lse[0]=%.4f\n", out->output[0], out->lse[0]);

    double *dQ = calloc(N * d, sizeof(double));
    double *dK = calloc(N * d, sizeof(double));
    double *dV = calloc(N * d, sizeof(double));
    double *dO = calloc(N * d, sizeof(double));
    for (uint32_t i = 0; i < N * d; i++) dO[i] = 0.01;
    flash_attn_backward(Q, K, V, dO, dQ, dK, dV, &small_cfg);
    printf("Backward pass: dQ[0]=%.6f, dK[0]=%.6f, dV[0]=%.6f\n", dQ[0], dK[0], dV[0]);
    flash_attn_output_free(out);
    free(Q); free(K); free(V); free(dQ); free(dK); free(dV); free(dO);

    /* Step 5: Triton Compilation */
    printf("\n-- Step 5: Triton Compilation --\n");
    BlockProgram bp = triton_block_program(0, 64, 8, 8, 1);
    printf("Block program: pid=%d, numPids=%d, grid=(%d,%d,%d)\n",
           bp.pid, bp.numPids, bp.gridDimX, bp.gridDimY, bp.gridDimZ);

    TritonAutotuneResult tune = triton_autotune_matmul(1024, 1024, 1024, 120.0);
    printf("Autotune matmul (1024^3): %d configs, best=#%d (%.2f TFLOPS)\n",
           tune.numConfigs, tune.bestIndex, tune.bestTflops);
    for (uint32_t i = 0; i < 3 && i < tune.numConfigs; i++) {
        printf("  config[%d]: blockM=%d, blockN=%d, blockK=%d, warps=%d, stages=%d, %.2f TFLOPS\n",
               i, tune.configs[i].blockM, tune.configs[i].blockN, tune.configs[i].blockK,
               tune.configs[i].numWarps, tune.configs[i].numStages, tune.configs[i].estimatedTflops);
    }

    double tf = triton_tile_tflops(128, 128, 32, 1.41, 108);
    printf("Tile TFLOPS (128×128×32, 1.41GHz, 108 SMs): %.2f\n", tf);

    TritonCompileContext *tcc = triton_compile_begin(
        "@triton.jit\ndef matmul_kernel(A, B, C, M, N, K):\n  ...");
    printf("Compile: started at stage %d\n", tcc->currentStage);
    triton_compile_stage(tcc, TRITON_TTIR);
    triton_compile_stage(tcc, TRITON_TTGIR);
    triton_compile_stage(tcc, TRITON_LLVM_IR);
    triton_compile_stage(tcc, TRITON_PTX);
    printf("Compile: reached stage %d in %d ms\n", tcc->currentStage, tcc->compileTimeMs);
    triton_compile_print(tcc);
    triton_compile_end(tcc);

    printf("\nAI system software demo complete!\n");
    return 0;
}
