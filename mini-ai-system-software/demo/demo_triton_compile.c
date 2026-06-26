#include "triton_compile.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static double random_normal_demo(void) {
    return ((double)rand() / RAND_MAX - 0.5) * 2.0;
}

static void fill_matrix(double *mat, uint32_t rows, uint32_t cols) {
    for (uint32_t i = 0; i < rows * cols; i++)
        mat[i] = random_normal_demo();
}

static void mat_print(const double *mat, uint32_t rows, uint32_t cols) {
    for (uint32_t r = 0; r < rows; r++) {
        printf("  ");
        for (uint32_t c = 0; c < cols; c++)
            printf("%7.2f ", mat[r * cols + c]);
        printf("\n");
    }
}

static void mat_mul_naive(const double *A, const double *B, double *C,
                           uint32_t M, uint32_t N, uint32_t K) {
    for (uint32_t i = 0; i < M; i++)
        for (uint32_t j = 0; j < N; j++) {
            C[i * N + j] = 0.0;
            for (uint32_t k = 0; k < K; k++)
                C[i * N + j] += A[i * K + k] * B[k * N + j];
        }
}

/* Demo 1: Block program model */
static void demo_block_model(void) {
    printf("\n=== Triton Block Programming Model ===\n");

    uint32_t M = 128, N = 64;
    uint32_t blockM = 32, blockN = 16;
    uint32_t gridM = (M + blockM - 1) / blockM;
    uint32_t gridN = (N + blockN - 1) / blockN;
    uint32_t totalPids = gridM * gridN;

    printf("Problem: C = A(M×K) × B(K×N), M=%u N=%u\n", M, N);
    printf("Tile: blockM=%u, blockN=%u\n", blockM, blockN);
    printf("Grid: %u × %u = %u blocks\n\n", gridM, gridN, totalPids);

    printf("Block program dispatch:\n");
    for (uint32_t pid = 0; pid < totalPids && pid < 12; pid++) {
        BlockProgram bp = triton_block_program(pid, totalPids,
                                                gridM, gridN, 1);
        uint32_t rowBlock = pid / gridN;
        uint32_t colBlock = pid % gridN;
        uint32_t rowStart = rowBlock * blockM;
        uint32_t colStart = colBlock * blockN;
        printf("  pid=%u → computes C[%u..%u][%u..%u] (%u×%u tile)\n",
               pid, rowStart,
               rowStart + blockM - 1 < M ? rowStart + blockM - 1 : M - 1,
               colStart,
               colStart + blockN - 1 < N ? colStart + blockN - 1 : N - 1,
               blockM, blockN);
    }
    printf("  ... (%u total blocks)\n", totalPids);

    printf("\nTriton DSL equivalent:\n");
    printf("  @triton.jit\n");
    printf("  def matmul_kernel(A, B, C, M, N, K, ...):\n");
    printf("      pid = tl.program_id(axis=0)\n");
    printf("      block_m = pid // grid_n\n");
    printf("      block_n = pid %% grid_n\n");
    printf("      # load tiles with masking...\n");
}

/* Demo 2: Compiler pipeline */
static void demo_compile_pipeline(void) {
    printf("\n=== Triton Compiler Pipeline ===\n");

    const char *dsl =
        "@triton.jit\n"
        "def add_kernel(x, y, out, N):\n"
        "    pid = tl.program_id(0)\n"
        "    offsets = pid * BLOCK\n"
        "    x_ptrs = x + offsets\n"
        "    y_ptrs = y + offsets\n"
        "    x_block = tl.load(x_ptrs, mask=...)\n"
        "    y_block = tl.load(y_ptrs, mask=...)\n"
        "    tl.store(out + offsets, x_block + y_block)\n";

    TritonCompileContext *ctx = triton_compile_begin(dsl);

    printf("Pipeline stages:\n");

    printf("  [1] DSL  (Python) → Triton High-Level IR\n");
    triton_compile_stage(ctx, TRITON_DSL);

    printf("  [2] TTIR → optimization passes (CSE, DCE, loop unroll)\n");
    triton_compile_stage(ctx, TRITON_TTIR);

    printf("  [3] TTGIR → layout conversion (blocked → distributed)\n");
    printf("         Memory promotion: local → shared → global\n");
    triton_compile_stage(ctx, TRITON_TTGIR);

    printf("  [4] LLVM IR → target-independent optimizations\n");
    triton_compile_stage(ctx, TRITON_LLVM_IR);

    printf("  [5] PTX → NVIDIA ISA (load/store, sync, barrier)\n");
    triton_compile_stage(ctx, TRITON_PTX);

    printf("  [6] CUBIN → final binary\n");
    triton_compile_stage(ctx, TRITON_CUBIN);

    triton_compile_end(ctx);
    triton_compile_print(ctx);
    free(ctx);
}

/* Demo 3: Tile load/store with masking */
static void demo_tile_load_store(void) {
    printf("\n=== Triton Tile Load/Store with Masking ===\n");

    uint32_t tensorRows = 10;
    uint32_t tensorCols = 8;
    double *tensor = (double *)calloc(tensorRows * tensorCols, sizeof(double));
    for (uint32_t i = 0; i < tensorRows * tensorCols; i++)
        tensor[i] = (double)(i + 1);

    printf("Source tensor (%u×%u):\n", tensorRows, tensorCols);
    for (uint32_t r = 0; r < tensorRows; r++) {
        printf("  ");
        for (uint32_t c = 0; c < tensorCols; c++)
            printf("%4.0f ", tensor[r * tensorCols + c]);
        printf("\n");
    }

    TritonTile tile;
    tile.rows      = 4;
    tile.cols      = 4;
    tile.ld        = tensorCols;
    tile.rowOffset = 8;  /* start at row 8 (partially out of bounds) */
    tile.colOffset = 6;  /* start at col 6 (partially out of bounds) */
    tile.hasMask   = 1;
    tile.data      = (double *)calloc(tile.rows * tile.cols, sizeof(double));

    TritonLoadOpts loadOpts;
    loadOpts.useMask       = 1;
    loadOpts.maskingValue  = -1.0;
    loadOpts.cacheModifier = 0;
    loadOpts.evictionPolicy = 0;

    triton_tile_load(&tile, tensor, tensorRows, tensorCols, &loadOpts);

    printf("Loaded tile (4×4) at offset (%u,%u) with mask:\n",
           tile.rowOffset, tile.colOffset);
    for (uint32_t r = 0; r < tile.rows; r++) {
        printf("  ");
        for (uint32_t c = 0; c < tile.cols; c++)
            printf("%4.0f ", tile.data[r * tile.cols + c]);
        printf("\n");
    }
    printf("  (-1.0 = masked out-of-bounds value)\n");

    free(tensor);
    free(tile.data);
}

/* Demo 4: Autotuning */
static void demo_autotuning(void) {
    printf("\n=== Triton Autotuning (@triton.autotune) ===\n");

    uint32_t M = 4096, N = 4096, K = 4096;
    printf("Problem: (%u×%u) × (%u×%u), total FLOPs = %.2f GFLOPs\n",
           M, K, K, N,
           2.0 * (double)M * (double)N * (double)K / 1e9);

    printf("\nAutotune search space:\n");
    printf("  blockM ∈ {16, 32, 64, 128, 256}\n");
    printf("  blockN ∈ {16, 32, 64, 128, 256}\n");
    printf("  blockK ∈ {16, 32, 64, 128, 256}\n");
    printf("  Total configs: 5³ = 125\n");

    TritonAutotuneResult result = triton_autotune_matmul(M, N, K, 312.0);

    printf("\nTop 5 configurations:\n");
    printf("  %6s %6s %6s  %6s  %6s  %12s\n",
           "bm", "bn", "bk", "warps", "stages", "TFLOPS");

    /* Bubble-sort for top 5 by tflops */
    for (uint32_t i = 0; i < result.numConfigs && i < 124; i++) {
        for (uint32_t j = i + 1; j < result.numConfigs; j++) {
            if (result.configs[j].estimatedTflops
                > result.configs[i].estimatedTflops) {
                TritonTuneConfig tmp = result.configs[i];
                result.configs[i] = result.configs[j];
                result.configs[j] = tmp;
            }
        }
    }

    for (uint32_t i = 0; i < 5 && i < result.numConfigs; i++) {
        TritonTuneConfig *tc = &result.configs[i];
        printf("  %6u %6u %6u  %6u  %6u  %12.2f\n",
               tc->blockM, tc->blockN, tc->blockK,
               tc->numWarps, tc->numStages, tc->estimatedTflops);
    }

    printf("\n@triton.autotune decorator usage:\n");
    printf("  @triton.autotune(\n");
    printf("      configs=[\n");
    printf("          triton.Config({'BLOCK_M': 128, ...}),\n");
    printf("          triton.Config({'BLOCK_M': 64, ...}),\n");
    printf("          ...\n");
    printf("      ],\n");
    printf("      key=['M', 'N', 'K'],\n");
    printf("  )\n");
}

/* Demo 5: Atomic addition */
static void demo_atomic_add(void) {
    printf("\n=== Triton Atomic Add Demo ===\n");

    double sharedMem[4] = { 1.0, 2.0, 3.0, 4.0 };
    printf("Initial values: %.1f %.1f %.1f %.1f\n",
           sharedMem[0], sharedMem[1], sharedMem[2], sharedMem[3]);

    TritonAtomicAdd ops[6] = {
        { &sharedMem[0], 10.0, 1 },
        { &sharedMem[0],  5.0, 1 },
        { &sharedMem[1], 20.0, 1 },
        { &sharedMem[2],  0.5, 1 },
        { &sharedMem[3],  7.0, 1 },
        { &sharedMem[3],  3.0, 1 },
    };

    for (int i = 0; i < 6; i++)
        triton_atomic_add(&ops[i]);

    printf("After 6 atomic_add ops: %.1f %.1f %.1f %.1f\n",
           sharedMem[0], sharedMem[1], sharedMem[2], sharedMem[3]);

    printf("\nUse case: gradient accumulation across blocks\n");
    printf("  for block_id in range(num_blocks):\n");
    printf("      partial_result = compute_block(block_id)\n");
    printf("      tl.atomic_add(output_ptr, partial_result)\n");
}

/* Demo 6: End-to-end blocked matmul */
static void demo_blocked_matmul(void) {
    printf("\n=== End-to-End Block-Tiled Matrix Multiply ===\n");

    uint32_t M = 128, K = 64, N = 96;
    printf("C(%u×%u) = A(%u×%u) × B(%u×%u)\n", M, N, M, K, K, N);

    double *A = (double *)malloc((size_t)M * K * sizeof(double));
    double *B = (double *)malloc((size_t)K * N * sizeof(double));
    double *C_blocked = (double *)calloc((size_t)M * N, sizeof(double));
    double *C_naive   = (double *)calloc((size_t)M * N, sizeof(double));

    fill_matrix(A, M, K);
    fill_matrix(B, K, N);

    uint32_t blockM = 32, blockN = 32, blockK = 32;
    triton_matmul_blocked(A, B, C_blocked, M, N, K,
                           blockM, blockN, blockK);
    mat_mul_naive(A, B, C_naive, M, N, K);

    double maxDiff = 0.0;
    for (uint32_t i = 0; i < M * N; i++) {
        double d = fabs(C_blocked[i] - C_naive[i]);
        if (d > maxDiff) maxDiff = d;
    }
    printf("Max |blocked - naive| = %.6e\n", maxDiff);
    printf("Status: %s\n", maxDiff < 1e-6 ? "PASS" : "WARN");

    printf("First 4×4 block of C:\n");
    mat_print(C_blocked, 4, 4);

    free(A); free(B); free(C_blocked); free(C_naive);
}

int main(void) {
    printf("=== Triton Compiler Model Demo ===\n");
    printf("GPU: A100 (108 SMs, 1.41 GHz, 312 TFLOPS fp16)\n\n");

    demo_block_model();
    demo_compile_pipeline();
    demo_tile_load_store();
    demo_autotuning();
    demo_atomic_add();
    demo_blocked_matmul();

    printf("\n=== Demo complete ===\n");
    return 0;
}
