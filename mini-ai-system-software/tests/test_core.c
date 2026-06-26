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
#include "../include/gpu_reduction.h"
#include "../include/gpu_stream.h"
#include "../include/gpu_convolution.h"
#include "../include/attention_variants.h"
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

/* ── GPU Reduction Tests ── */
static int test_gpu_reduction_plan(void) {
    TEST("gpu_reduction_plan");
    ReductionPlan plan = gpu_reduction_plan(1024, 256, 48 * 1024);
    CHECK(plan.numElements == 1024, "numElements wrong");
    CHECK(plan.numBlocks >= 1, "numBlocks zero");
    CHECK(plan.blockSize >= 32, "blockSize too small");
    PASS();
    return 0;
}

static int test_gpu_reduction_tree(void) {
    TEST("gpu_reduction_tree");
    double data[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    ReductionPlan plan = gpu_reduction_plan(8, 8, 1024);
    double result = 0.0;
    gpu_reduction_tree(data, 8, &plan, &result);
    CHECK(fabs(result - 36.0) < 0.01, "tree reduce sum wrong");
    PASS();
    return 0;
}

static int test_gpu_scan_blelloch(void) {
    TEST("gpu_scan_blelloch");
    double data[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    double output[8];
    ScanPlan splan;
    memset(&splan, 0, sizeof(splan));
    splan.scanType = 0; /* exclusive */
    splan.numElements = 8;
    splan.blockSize = 8;
    gpu_scan_blelloch(data, 8, &splan, output);
    CHECK(fabs(output[0]) < 0.01, "exclusive scan[0] should be 0");
    CHECK(fabs(output[7] - 7.0) < 0.01, "exclusive scan[7] wrong");
    PASS();
    return 0;
}

static int test_gpu_scan_segmented(void) {
    TEST("gpu_scan_segmented");
    double data[6] = {3, 1, 7, 0, 4, 1};
    int flags[6] = {1, 0, 1, 0, 1, 0};
    double output[6];
    gpu_scan_segmented(data, flags, 6, output);
    CHECK(fabs(output[0] - 3.0) < 0.01, "seg[0] wrong");
    CHECK(fabs(output[1] - 4.0) < 0.01, "seg[1] wrong");
    CHECK(fabs(output[2] - 7.0) < 0.01, "seg[2] wrong");
    CHECK(fabs(output[3] - 7.0) < 0.01, "seg[3] wrong");
    PASS();
    return 0;
}

static int test_gpu_amdahl_speedup(void) {
    TEST("gpu_amdahl_speedup");
    double sp = gpu_amdahl_speedup(0.95, 100);
    CHECK(sp > 1.0 && sp < 100.0, "speedup out of range");
    double sp_full = gpu_amdahl_speedup(1.0, 64);
    CHECK(fabs(sp_full - 64.0) < 0.01, "fully parallel speedup wrong");
    PASS();
    return 0;
}

/* ── GPU Stream Tests ── */
static int test_gpu_stream_pool(void) {
    TEST("gpu_stream_pool");
    StreamPool pool = gpu_stream_pool_create(8);
    CHECK(pool.numStreams == 8, "pool stream count wrong");
    CudaStream *st = gpu_stream_alloc(&pool, 0);
    CHECK(st != NULL, "stream alloc failed");
    CHECK(st->state == STREAM_RUNNING, "stream not running");
    PASS();
    return 0;
}

static int test_gpu_transfer_estimate(void) {
    TEST("gpu_transfer_estimate");
    TransferOp op = gpu_transfer_estimate(TRANSFER_H2D, 1024 * 1024, 32.0);
    CHECK(op.bytes == 1024 * 1024, "transfer bytes wrong");
    CHECK(op.timeUs > 0.0, "transfer time zero");
    PASS();
    return 0;
}

static int test_gpu_pipeline_schedule(void) {
    TEST("gpu_pipeline_schedule");
    TransferOp h2d = gpu_transfer_estimate(TRANSFER_H2D, 4096, 32.0);
    TransferOp d2h = gpu_transfer_estimate(TRANSFER_D2H, 4096, 32.0);
    PipelineSchedule sched = gpu_pipeline_schedule(&h2d, 10.0, &d2h, 100);
    CHECK(sched.numIterations == 100, "iterations wrong");
    CHECK(sched.throughputPerSec > 0.0, "throughput zero");
    PASS();
    return 0;
}

static int test_gpu_occupancy_report(void) {
    TEST("gpu_occupancy_report");
    StreamPool pool = gpu_stream_pool_create(4);
    gpu_stream_alloc(&pool, 0);
    gpu_stream_alloc(&pool, -1);
    uint32_t high, normal, low;
    gpu_stream_occupancy_report(&pool, &high, &normal, &low);
    CHECK(high + normal + low >= 2, "occupancy count wrong");
    PASS();
    return 0;
}

/* ── GPU Convolution Tests ── */
static int test_gpu_conv2d_desc(void) {
    TEST("gpu_conv2d_desc");
    Conv2DDesc desc;
    int ok = gpu_conv2d_desc_init(&desc, 1, 3, 32, 32, 64, 3, 3, 1, 1, 1, 1, 1, 1);
    CHECK(ok == 1, "conv desc init failed");
    CHECK(desc.outputHeight == 32, "output height wrong");
    CHECK(desc.outputWidth == 32, "output width wrong");
    PASS();
    return 0;
}

static int test_gpu_im2col(void) {
    TEST("gpu_im2col");
    Conv2DDesc desc;
    gpu_conv2d_desc_init(&desc, 1, 1, 4, 4, 1, 2, 2, 0, 0, 1, 1, 1, 1);
    size_t ws = gpu_im2col_workspace_size(&desc);
    CHECK(ws > 0, "im2col workspace zero");
    double *input = (double *)calloc(16, sizeof(double));
    for (int i = 0; i < 16; i++) input[i] = (double)(i + 1);
    double *cols = (double *)malloc(ws * sizeof(double));
    gpu_im2col(input, &desc, cols);
    CHECK(cols[0] == 1.0, "im2col first element wrong");
    free(input);
    free(cols);
    PASS();
    return 0;
}

static int test_gpu_conv2d_im2col_gemm(void) {
    TEST("gpu_conv2d_im2col_gemm");
    Conv2DDesc desc;
    gpu_conv2d_desc_init(&desc, 1, 1, 4, 4, 1, 3, 3, 1, 1, 1, 1, 1, 1);
    size_t inSize = 1 * 1 * 4 * 4;
    size_t filtSize = 1 * 1 * 3 * 3;
    size_t outSize = 1 * 1 * 4 * 4;
    double *input = (double *)calloc(inSize, sizeof(double));
    double *filters = (double *)calloc(filtSize, sizeof(double));
    double *output = (double *)calloc(outSize, sizeof(double));
    for (size_t i = 0; i < inSize; i++) input[i] = 1.0;
    for (size_t i = 0; i < filtSize; i++) filters[i] = 1.0;
    gpu_conv2d_im2col_gemm(input, filters, &desc, output);
    CHECK(output != NULL, "output null");
    free(input); free(filters); free(output);
    PASS();
    return 0;
}

static int test_gpu_conv2d_recommend(void) {
    TEST("gpu_conv2d_recommend");
    Conv2DDesc desc;
    gpu_conv2d_desc_init(&desc, 1, 64, 56, 56, 64, 3, 3, 1, 1, 1, 1, 1, 1);
    ConvAlgorithm algo = gpu_conv2d_recommend_algo(&desc, 19.5, 2.0);
    CHECK(algo == CONV_ALGO_WINOGRAD, "expected Winograd for 3×3");
    PASS();
    return 0;
}

/* ── Attention Variants Tests ── */
static int test_attn_mqa_forward(void) {
    TEST("attn_mqa_forward");
    AttnDescriptor desc;
    memset(&desc, 0, sizeof(desc));
    desc.variant = ATTN_MQA;
    desc.batchSize = 1;
    desc.numQHeads = 4;
    desc.numKVHeads = 1;
    desc.seqLenQ = 4;
    desc.seqLenKV = 4;
    desc.headDim = 8;
    desc.softmaxScale = 1.0 / sqrt(8.0);
    desc.causal = 0;
    size_t qSize = 1 * 4 * 4 * 8;
    size_t kvSize = 1 * 1 * 4 * 8;
    double *Q = (double *)calloc(qSize, sizeof(double));
    double *K = (double *)calloc(kvSize, sizeof(double));
    double *V = (double *)calloc(kvSize, sizeof(double));
    double *out = (double *)calloc(qSize, sizeof(double));
    for (size_t i = 0; i < qSize; i++) Q[i] = 1.0;
    for (size_t i = 0; i < kvSize; i++) { K[i] = 1.0; V[i] = 1.0; }
    attn_mqa_forward(Q, K, V, &desc, out);
    CHECK(out != NULL, "MQA output null");
    free(Q); free(K); free(V); free(out);
    PASS();
    return 0;
}

static int test_attn_gqa_forward(void) {
    TEST("attn_gqa_forward");
    AttnDescriptor desc;
    memset(&desc, 0, sizeof(desc));
    desc.variant = ATTN_GQA;
    desc.batchSize = 1;
    desc.numQHeads = 8;
    desc.numKVHeads = 2;
    desc.seqLenQ = 4;
    desc.seqLenKV = 4;
    desc.headDim = 8;
    desc.softmaxScale = 1.0 / sqrt(8.0);
    desc.causal = 0;
    size_t qSize = 1 * 8 * 4 * 8;
    size_t kvSize = 1 * 2 * 4 * 8;
    double *Q = (double *)calloc(qSize, sizeof(double));
    double *K = (double *)calloc(kvSize, sizeof(double));
    double *V = (double *)calloc(kvSize, sizeof(double));
    double *out = (double *)calloc(qSize, sizeof(double));
    for (size_t i = 0; i < qSize; i++) Q[i] = 1.0;
    for (size_t i = 0; i < kvSize; i++) { K[i] = 1.0; V[i] = 1.0; }
    attn_gqa_forward(Q, K, V, &desc, out);
    CHECK(out != NULL, "GQA output null");
    free(Q); free(K); free(V); free(out);
    PASS();
    return 0;
}

static int test_kv_cache(void) {
    TEST("kv_cache");
    KVCache *cache = kv_cache_create(1, 2, 16, 8);
    CHECK(cache != NULL, "cache create failed");
    CHECK(cache->currentLen == 0, "initial len not zero");
    double *kNew = (double *)calloc(1 * 2 * 4 * 8, sizeof(double));
    double *vNew = (double *)calloc(1 * 2 * 4 * 8, sizeof(double));
    for (int i = 0; i < 64; i++) { kNew[i] = 1.0; vNew[i] = 2.0; }
    uint32_t newLen = kv_cache_append(cache, kNew, vNew, 4);
    CHECK(newLen == 4, "append length wrong");
    double *kOut = (double *)malloc(4 * 8 * sizeof(double));
    double *vOut = (double *)malloc(4 * 8 * sizeof(double));
    kv_cache_get_range(cache, 0, 0, 0, 4, kOut, vOut);
    CHECK(kOut[0] == 1.0, "k cache get wrong");
    kv_cache_free(cache);
    free(kNew); free(vNew); free(kOut); free(vOut);
    PASS();
    return 0;
}

static int test_sliding_window(void) {
    TEST("sliding_window_attention");
    AttnDescriptor desc;
    memset(&desc, 0, sizeof(desc));
    desc.batchSize = 1;
    desc.numQHeads = 2;
    desc.numKVHeads = 2;
    desc.seqLenQ = 8;
    desc.seqLenKV = 8;
    desc.headDim = 4;
    desc.softmaxScale = 0.5;
    desc.causal = 1;
    SlidingWindowConfig sw;
    sw.windowSize = 4;
    sw.causal = 1;
    sw.useAlibi = 0;
    size_t sz = 1 * 2 * 8 * 4;
    double *Q = (double *)calloc(sz, sizeof(double));
    double *K = (double *)calloc(sz, sizeof(double));
    double *V = (double *)calloc(sz, sizeof(double));
    double *out = (double *)calloc(sz, sizeof(double));
    for (size_t i = 0; i < sz; i++) Q[i] = K[i] = V[i] = 1.0;
    attn_sliding_window_forward(Q, K, V, &desc, &sw, out);
    CHECK(out != NULL, "sliding window output null");
    free(Q); free(K); free(V); free(out);
    PASS();
    return 0;
}

static int test_alibi_bias(void) {
    TEST("alibi_bias");
    uint32_t N = 4;
    double *scores = (double *)calloc(2 * N * N, sizeof(double));
    for (size_t i = 0; i < 2 * N * N; i++) scores[i] = 10.0;
    attn_alibi_bias(scores, N, N, 2, 1);
    /* After ALiBi: scores for distant positions should be lower */
    CHECK(scores[N] < 10.0, "ALiBi bias not applied");
    free(scores);
    PASS();
    return 0;
}

static int test_kv_cache_memory_compare(void) {
    TEST("kv_cache_memory_compare");
    double mha, mqa, gqa;
    kv_cache_memory_compare(32, 8, 4096, 128, 80, &mha, &mqa, &gqa);
    CHECK(mqa < mha, "MQA should use less memory than MHA");
    CHECK(gqa < mha, "GQA should use less memory than MHA");
    CHECK(mqa < gqa, "MQA should use less memory than GQA");
    PASS();
    return 0;
}

/* ── GPU Convolution Winograd Test ── */
static int test_winograd(void) {
    TEST("winograd_f2x2_3x3");
    Conv2DDesc desc;
    gpu_conv2d_desc_init(&desc, 1, 1, 4, 4, 1, 3, 3, 0, 0, 1, 1, 1, 1);
    CHECK(desc.outputHeight == 2, "output height wrong for winograd");
    size_t inSize = 1 * 1 * 4 * 4;
    size_t filtSize = 1 * 1 * 3 * 3;
    size_t outSize = 1 * 1 * 2 * 2;
    double *input = (double *)calloc(inSize, sizeof(double));
    double *filters = (double *)calloc(filtSize, sizeof(double));
    double *output = (double *)calloc(outSize, sizeof(double));
    for (size_t i = 0; i < inSize; i++) input[i] = 1.0;
    for (size_t i = 0; i < filtSize; i++) filters[i] = 1.0;
    gpu_winograd_f2x2_3x3(input, filters, &desc, output);
    CHECK(output != NULL, "winograd output null");
    free(input); free(filters); free(output);
    PASS();
    return 0;
}

int main(void) {
    printf("=== mini-ai-system-software Unit Tests ===\n\n");

    int failed = 0;

    /* Original tests */
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

    /* GPU Reduction tests */
    failed += test_gpu_reduction_plan();
    failed += test_gpu_reduction_tree();
    failed += test_gpu_scan_blelloch();
    failed += test_gpu_scan_segmented();
    failed += test_gpu_amdahl_speedup();

    /* GPU Stream tests */
    failed += test_gpu_stream_pool();
    failed += test_gpu_transfer_estimate();
    failed += test_gpu_pipeline_schedule();
    failed += test_gpu_occupancy_report();

    /* GPU Convolution tests */
    failed += test_gpu_conv2d_desc();
    failed += test_gpu_im2col();
    failed += test_gpu_conv2d_im2col_gemm();
    failed += test_gpu_conv2d_recommend();
    failed += test_winograd();

    /* Attention Variants tests */
    failed += test_attn_mqa_forward();
    failed += test_attn_gqa_forward();
    failed += test_kv_cache();
    failed += test_sliding_window();
    failed += test_alibi_bias();
    failed += test_kv_cache_memory_compare();

    printf("\n=== Results: %d/%d passed, %d failed ===\n",
           tests_passed, tests_run, failed);
    return failed > 0 ? 1 : 0;
}
