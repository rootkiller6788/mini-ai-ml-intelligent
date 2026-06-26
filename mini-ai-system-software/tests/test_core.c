/*
 * mini-ai-system-software — Core Tests
 *
 * Unit tests for CUDA kernels, GPU memory, NCCL collectives,
 * Flash Attention, Triton compilation.
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

static int tests_run = 0, tests_passed = 0;

#define TEST(name) do { tests_run++; printf("  TEST %s ... ", name); } while(0)
#define PASS() do { tests_passed++; printf("PASS\n"); } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); return 1; } while(0)
#define CHECK(cond, msg) if (!(cond)) FAIL(msg)

/* ── CUDA Kernel Tests ── */
static int test_thread_context(void) {
    TEST("thread_context");
    KernelLaunchConfig cfg;
    cfg.gridDim = MK_DIM3_2D(2, 2);
    cfg.blockDim = MK_DIM3_2D(8, 8);
    ThreadContext ctx = cuda_thread_context(&cfg, 0);
    CHECK(ctx.threadIdx.x == 0, "threadIdx.x wrong");
    CHECK(ctx.blockIdx.x == 0, "blockIdx.x wrong");
    PASS();
    return 0;
}

static int test_cuda_flat_indices(void) {
    TEST("cuda_flat_indices");
    dim3 blockIdx = {1, 0, 0};
    dim3 gridDim = {2, 2, 1};
    uint32_t flat = cuda_flat_block_idx(&blockIdx, &gridDim);
    CHECK(flat == 1, "flat block idx wrong");
    dim3 threadIdx = {3, 2, 0};
    dim3 blockDim = {8, 4, 1};
    flat = cuda_flat_thread_idx(&threadIdx, &blockDim);
    CHECK(flat == 3 + 2 * 8, "flat thread idx wrong");
    PASS();
    return 0;
}

static int test_coop_group_create(void) {
    TEST("coop_group_create");
    CoopGroup cg = cuda_coop_group_create(CG_THIS_WARP, 32, 5);
    CHECK(cg.type == CG_THIS_WARP, "coop group type wrong");
    CHECK(cg.size == 32, "coop group size wrong");
    PASS();
    return 0;
}

/* ── GPU Memory Tests ── */
static int test_coalesce_analyze(void) {
    TEST("coalesce_analyze");
    uint32_t addrs[] = {0, 4, 8, 12, 16, 20, 24, 28};
    CoalesceReport r = gpu_coalesce_analyze(addrs, 8);
    CHECK(r.coalesceRatio >= 0.0 && r.coalesceRatio <= 1.0, "ratio out of range");
    PASS();
    return 0;
}

static int test_pinned_alloc(void) {
    TEST("pinned_alloc");
    PinnedMemory pm = gpu_pinned_alloc(4096);
    CHECK(pm.hostPtr != NULL, "pinned alloc failed");
    CHECK(pm.isPinned, "should be pinned");
    gpu_pinned_free(&pm);
    PASS();
    return 0;
}

static int test_cublas_tensor(void) {
    TEST("cublas_tensor");
    CuBlasTensor t;
    t.layout = CUBLAS_ROW_MAJOR;
    t.rows = 4; t.cols = 4; t.ld = 4; t.elemSize = sizeof(double);
    t.data = calloc(16, sizeof(double));
    cublas_tensor_set(&t, 0, 0, 3.14);
    double v = cublas_tensor_get(&t, 0, 0);
    CHECK(fabs(v - 3.14) < 0.001, "tensor get/set wrong");
    cublas_tensor_free(&t);
    PASS();
    return 0;
}

/* ── NCCL Tests ── */
static int test_nccl_comm_init(void) {
    TEST("nccl_comm_init");
    NcclComm comm = nccl_comm_init(4, 0, NCCL_INTER_NVLINK, 900.0);
    CHECK(comm.rank == 0, "rank wrong");
    CHECK(comm.nRanks == 4, "nRanks wrong");
    PASS();
    return 0;
}

static int test_nccl_allreduce(void) {
    TEST("nccl_allreduce");
    NcclComm comm = nccl_comm_init(2, 0, NCCL_INTER_NVLINK, 600.0);
    float sbuf[4] = {1, 2, 3, 4}, rbuf[4] = {0};
    NcclStatus s = nccl_allreduce(sbuf, rbuf, 4, NCCL_FLOAT32, NCCL_OP_SUM, &comm);
    CHECK(s == NCCL_STATUS_OK, "allreduce failed");
    PASS();
    return 0;
}

static int test_nccl_bandwidth(void) {
    TEST("nccl_bandwidth");
    double bw = nccl_allreduce_bandwidth(8, 900.0);
    CHECK(bw > 0.0, "bandwidth should be positive");
    double nv = nccl_nvlink_bandwidth(4);
    CHECK(nv > 0.0, "nvlink bw should be positive");
    PASS();
    return 0;
}

/* ── Flash Attention Tests ── */
static int test_flash_attn_config(void) {
    TEST("flash_attn_config");
    FlashAttnConfig cfg = flash_attn_config_create(1024, 64, 1);
    CHECK(cfg.seqLen == 1024, "seqLen wrong");
    CHECK(cfg.headDim == 64, "headDim wrong");
    CHECK(cfg.causal, "causal should be set");
    PASS();
    return 0;
}

static int test_flash_attn_io_complexity(void) {
    TEST("flash_attn_io_complexity");
    double io = flash_attn_io_complexity(2048, 64, 20 * 1024 * 1024);
    CHECK(io > 0.0, "IO complexity positive");
    PASS();
    return 0;
}

/* ── Triton Tests ── */
static int test_triton_block_program(void) {
    TEST("triton_block_program");
    BlockProgram bp = triton_block_program(0, 16, 4, 0, 0);
    CHECK(bp.pid == 0, "pid wrong");
    CHECK(bp.numPids == 16, "numPids wrong");
    PASS();
    return 0;
}

static int test_triton_autotune(void) {
    TEST("triton_autotune");
    TritonAutotuneResult result = triton_autotune_matmul(1024, 1024, 1024, 100.0);
    CHECK(result.numConfigs > 0, "autotune returned 0 configs");
    CHECK(result.bestIndex >= 0, "bestIndex negative");
    PASS();
    return 0;
}

static int test_triton_tile_tflops(void) {
    TEST("triton_tile_tflops");
    double tf = triton_tile_tflops(128, 128, 32, 1.4, 108);
    CHECK(tf > 0.0, "tflops should be positive");
    PASS();
    return 0;
}

int main(void) {
    printf("=== mini-ai-system-software Unit Tests ===\n\n");

    int failed = 0;
    failed += test_thread_context();
    failed += test_cuda_flat_indices();
    failed += test_coop_group_create();
    failed += test_coalesce_analyze();
    failed += test_pinned_alloc();
    failed += test_cublas_tensor();
    failed += test_nccl_comm_init();
    failed += test_nccl_allreduce();
    failed += test_nccl_bandwidth();
    failed += test_flash_attn_config();
    failed += test_flash_attn_io_complexity();
    failed += test_triton_block_program();
    failed += test_triton_autotune();
    failed += test_triton_tile_tflops();

    printf("\n=== Results: %d/%d passed, %d failed ===\n",
           tests_passed, tests_run, failed);
    return failed > 0 ? 1 : 0;
}
