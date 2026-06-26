/*
 * mini-ai-system-software — Core Benchmarks
 *
 * Benchmarks: CUDA kernels, GPU memory, NCCL collectives,
 *             Flash Attention, Triton compilation.
 */
#include "../include/cuda_kernel.h"
#include "../include/gpu_memory.h"
#include "../include/nccl_collectives.h"
#include "../include/flash_attention.h"
#include "../include/triton_compile.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
    return (double)clock() / (double)(CLOCKS_PER_SEC / 1000);
}

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 5000;
    double t0, t1;
    printf("=== mini-ai-system-software Benchmarks (N=%d) ===\n\n", N);

    /* ── CUDA Thread Context ── */
    {
        KernelLaunchConfig cfg;
        cfg.gridDim = MK_DIM3_2D(4, 4);
        cfg.blockDim = MK_DIM3_2D(16, 16);
        t0 = now_ms();
        for (int r = 0; r < N / 10; r++) {
            cuda_thread_context(&cfg, (uint32_t)(r % (16 * 16 * 4 * 4)));
        }
        t1 = now_ms();
        printf("  cuda_thread_context:   %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 10, t1 - t0, (t1 - t0) / (double)(N / 10) * 1000.0);
    }

    /* ── CUDA Flat Indices ── */
    {
        dim3 blockIdx = {2, 1, 0};
        dim3 gridDim = {4, 4, 1};
        dim3 threadIdx = {5, 3, 0};
        dim3 blockDim = {16, 8, 1};
        t0 = now_ms();
        for (int r = 0; r < N / 10; r++) {
            cuda_flat_block_idx(&blockIdx, &gridDim);
            cuda_flat_thread_idx(&threadIdx, &blockDim);
        }
        t1 = now_ms();
        printf("  cuda_flat_indices:     %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 10, t1 - t0, (t1 - t0) / (double)(N / 10) * 1000.0);
    }

    /* ── CUDA Shared Memory Alloc/Free ── */
    {
        t0 = now_ms();
        for (int r = 0; r < N / 10; r++) {
            SharedMem sm = cuda_shmem_alloc(1024);
            cuda_shmem_free(&sm);
        }
        t1 = now_ms();
        printf("  cuda_shmem_alloc+free: %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 10, t1 - t0, (t1 - t0) / (double)(N / 10) * 1000.0);
    }

    /* ── GPU Coalesce Analyze ── */
    {
        uint32_t addrs[] = {0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60,
                            64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124};
        t0 = now_ms();
        for (int r = 0; r < N / 10; r++) {
            gpu_coalesce_analyze(addrs, 32);
        }
        t1 = now_ms();
        printf("  gpu_coalesce_analyze:  %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 10, t1 - t0, (t1 - t0) / (double)(N / 10) * 1000.0);
    }

    /* ── GPU Pinned Alloc/Free ── */
    {
        t0 = now_ms();
        for (int r = 0; r < N / 50; r++) {
            PinnedMemory pm = gpu_pinned_alloc(4096);
            gpu_pinned_free(&pm);
        }
        t1 = now_ms();
        printf("  gpu_pinned_alloc+free: %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 50, t1 - t0, (t1 - t0) / (double)(N / 50) * 1000.0);
    }

    /* ── cuBLAS Tensor Get/Set ── */
    {
        CuBlasTensor t;
        t.layout = CUBLAS_ROW_MAJOR;
        t.rows = 64; t.cols = 64; t.ld = 64; t.elemSize = sizeof(double);
        t.data = calloc(64 * 64, sizeof(double));
        t0 = now_ms();
        for (int r = 0; r < N / 10; r++) {
            cublas_tensor_set(&t, (uint32_t)(r % 64), (uint32_t)((r / 64) % 64), (double)r);
            cublas_tensor_get(&t, (uint32_t)(r % 64), (uint32_t)((r / 64) % 64));
        }
        t1 = now_ms();
        printf("  cublas_tensor_get/set: %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 10, t1 - t0, (t1 - t0) / (double)(N / 10) * 1000.0);
        cublas_tensor_free(&t);
    }

    /* ── NCCL Comm Init ── */
    {
        t0 = now_ms();
        for (int r = 0; r < N / 10; r++) {
            nccl_comm_init(8, (uint32_t)(r % 8), NCCL_INTER_NVLINK, 900.0);
        }
        t1 = now_ms();
        printf("  nccl_comm_init:        %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 10, t1 - t0, (t1 - t0) / (double)(N / 10) * 1000.0);
    }

    /* ── NCCL AllReduce ── */
    {
        NcclComm comm = nccl_comm_init(4, 0, NCCL_INTER_NVLINK, 600.0);
        float sbuf[16] = {0}, rbuf[16] = {0};
        for (int i = 0; i < 16; i++) sbuf[i] = (float)i;
        t0 = now_ms();
        for (int r = 0; r < N / 20; r++) {
            nccl_allreduce(sbuf, rbuf, 16, NCCL_FLOAT32, NCCL_OP_SUM, &comm);
        }
        t1 = now_ms();
        printf("  nccl_allreduce:        %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 20, t1 - t0, (t1 - t0) / (double)(N / 20) * 1000.0);
    }

    /* ── NCCL Bandwidth Model ── */
    {
        t0 = now_ms();
        for (int r = 0; r < N / 10; r++) {
            nccl_allreduce_bandwidth(8, 900.0);
            nccl_nvlink_bandwidth(4);
        }
        t1 = now_ms();
        printf("  nccl_bw_model:         %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 10, t1 - t0, (t1 - t0) / (double)(N / 10) * 1000.0);
    }

    /* ── Flash Attention Config ── */
    {
        t0 = now_ms();
        for (int r = 0; r < N / 10; r++) {
            flash_attn_config_create(2048, 64, 1);
        }
        t1 = now_ms();
        printf("  flash_attn_config:     %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 10, t1 - t0, (t1 - t0) / (double)(N / 10) * 1000.0);
    }

    /* ── Flash Attention IO Complexity ── */
    {
        t0 = now_ms();
        for (int r = 0; r < N / 10; r++) {
            flash_attn_io_complexity(4096, 64, 20 * 1024 * 1024);
        }
        t1 = now_ms();
        printf("  flash_attn_io_cmplx:   %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 10, t1 - t0, (t1 - t0) / (double)(N / 10) * 1000.0);
    }

    /* ── Triton Block Program ── */
    {
        t0 = now_ms();
        for (int r = 0; r < N / 10; r++) {
            triton_block_program((uint32_t)(r % 256), 256, 16, 16, 0);
        }
        t1 = now_ms();
        printf("  triton_block_program:  %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 10, t1 - t0, (t1 - t0) / (double)(N / 10) * 1000.0);
    }

    /* ── Triton Matmul Blocked ── */
    {
        double *A = calloc(256 * 256, sizeof(double));
        double *B = calloc(256 * 256, sizeof(double));
        double *C = calloc(256 * 256, sizeof(double));
        for (int i = 0; i < 256 * 256; i++) { A[i] = 1.0; B[i] = 1.0; }
        t0 = now_ms();
        for (int r = 0; r < N / 100; r++) {
            triton_matmul_blocked(A, B, C, 256, 256, 256, 64, 64, 32);
        }
        t1 = now_ms();
        printf("  triton_matmul_blocked: %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 100, t1 - t0, (t1 - t0) / (double)(N / 100) * 1000.0);
        free(A); free(B); free(C);
    }

    /* ── Triton TFLOPs Estimate ── */
    {
        t0 = now_ms();
        for (int r = 0; r < N / 10; r++) {
            triton_tile_tflops(128, 128, 32, 1.4, 108);
        }
        t1 = now_ms();
        printf("  triton_tile_tflops:    %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 10, t1 - t0, (t1 - t0) / (double)(N / 10) * 1000.0);
    }

    printf("\nDone.\n");
    return 0;
}
