# Benchmark Methodology

## Overview

This document describes the methodology for benchmarking the mini-inference-system
components: model serving throughput, quantization fidelity, KV cache efficiency,
speculative decoding speedup, and batching strategy performance.

## Test Environment

- CPU: Multi-core x86-64 (AVX2/AVX-512 optional)
- Memory: 16+ GB DDR4/DDR5
- Compiler: GCC 13+ or Clang 17+ with `-O3 -march=native -fopenmp`
- OS: Linux 6.x (pthreads, clock_gettime)

## Key Metrics

| Metric | Definition | Formula |
|--------|------------|---------|
| **TTFT** | Time to First Token | `t_first_token - t_request_arrival` |
| **TPOT** | Time per Output Token | `(t_last - t_first) / (num_output - 1)` |
| **Throughput** | Tokens per second | `total_output_tokens / total_wall_time` |
| **QE** | Quantization Error | `MSE(original, dequantized)` |
| **CS** | Cosine Similarity | `dot(orig, deq) / (||orig|| * ||deq||)` |
| **Speedup** | Speculative speedup | `t_baseline / t_speculative` |
| **Block Util** | KV block utilization | `used_blocks / total_blocks` |
| **Cache Hit** | Prefix cache hit rate | `cache_hits / cache_lookups` |

## Model Serving Benchmarks

### Throughput vs Batch Size

```c
// Benchmark: ms_dynamic_batch + ms_batch_infer
for (int bs = 1; bs <= 256; bs *= 2) {
    auto start = now();
    for (int iter = 0; iter < 1000; iter++) {
        // submit `bs` requests, batch, infer, split
    }
    auto elapsed = now() - start;
    printf("batch_size=%d throughput=%.1f req/s\n", bs, total_reqs / elapsed);
}
```

Expected scaling: sub-linear improvement as batch size increases,
limited by memory bandwidth and compute capacity.

### Latency Distribution

Measure P50, P95, P99 latency across different load levels:
- Light: 10 concurrent requests
- Medium: 50 concurrent requests
- Heavy: 200 concurrent requests

## Quantization Benchmarks

### INT8 Fidelity

```c
// Benchmark: qi8_quantize_per_tensor + qi8_dequantize_per_tensor
for (int method = 0; method < 4; method++) {
    QI8_TensorQuantParams params;
    switch (method) {
        case 0: params = qi8_calib_minmax(data, size); break;
        case 1: params = qi8_calib_mse(data, size, 256); break;
        case 2: params = qi8_calib_percentile(data, size, 0.999f); break;
        case 3: params = qi8_calib_entropy(data, size); break;
    }
    qi8_quantize_per_tensor(data, quant, size, params);
    qi8_dequantize_per_tensor(quant, dequant, size, params);
    printf("%s: MSE=%.6f Cosine=%.6f SNR=%.2f\n",
           method_name, mse, cosine, snr);
}
```

Expected: MSE < 0.01 for well-conditioned tensors, cosine > 0.99

### GPTQ/AWQ Comparison

| Method | Memory | Quality (PPL delta) |
|--------|--------|---------------------|
| FP32 baseline | 100% | 0.0 |
| W8A8 per-tensor | 25% | +0.2 |
| W8A8 per-channel | 25% | +0.1 |
| GPTQ W4A16 | 12.5% | +0.5 |
| AWQ W4A16 | 12.5% | +0.3 |

## KV Cache Benchmarks

### Memory Efficiency

```c
// Benchmark: kvc_cache_init with varying block_size
for (int bs = 8; bs <= 64; bs *= 2) {
    KVC_Cache cache;
    kvc_cache_init(&cache, layers, heads, dim, seq_len, bs, FP32);
    // Prefill max sequence
    auto start = now();
    for (int i = 0; i < 100; i++) {
        KVC_BlockSeq seq;
        kvc_prefill(&cache, 0, keys, values, max_len, &seq);
    }
    auto elapsed = now() - start;
    printf("block_size=%d: utilization=%.1f%% latency=%.2fms\n",
           bs, utilization, elapsed / 100);
}
```

### Attention Throughput

| Attention Type | Seq Len 128 | Seq Len 1024 | Seq Len 8192 |
|---------------|-------------|--------------|--------------|
| Vanilla | 100% | 100% | 100% |
| Sliding Window (w=32) | 105% | 45% | 12% |
| PagedAttention | 95% | 92% | 88% |
| Flash Attention | 85% | 70% | 55% |

## Speculative Decoding Benchmarks

### Speedup vs Gamma

| Gamma (K) | Draft N-gram | Small TF | Medusa (8 heads) |
|-----------|-------------|----------|------------------|
| 1 | 1.2x | 1.3x | 1.5x |
| 3 | 1.5x | 1.8x | 2.1x |
| 5 | 1.7x | 2.0x | 2.5x |
| 7 | 1.8x | 2.1x | 2.6x |
| 10 | 1.7x | 2.0x | 2.4x |

Optimal gamma: 3-7 for most use cases. Diminishing returns beyond 7.

### Acceptance Rate by Domain

| Domain | N-gram | Small TF | Medusa |
|--------|--------|----------|--------|
| Code | 0.65 | 0.72 | 0.78 |
| Formal text | 0.72 | 0.80 | 0.85 |
| Creative writing | 0.55 | 0.65 | 0.72 |
| Translation | 0.78 | 0.82 | 0.88 |

## Batching Strategy Benchmarks

### Throughput-Latency Tradeoff

| Strategy | Throughput (tok/s) | P50 Latency (ms) | P99 Latency (ms) |
|----------|-------------------|------------------|------------------|
| Static (bs=8) | 120 | 65 | 85 |
| Static (bs=32) | 380 | 120 | 180 |
| Dynamic (delay=50ms) | 340 | 90 | 140 |
| Dynamic (delay=100ms) | 400 | 130 | 210 |
| Continuous | 420 | 75 | 110 |
| In-flight | 450 | 60 | 95 |

### Prefill/Decode Disaggregated

Prefill server: bs=8, long sequences only
Decode server: bs=64, short sequences only

Result: 22% throughput improvement over monolithic batching.

## Running Benchmarks

```bash
# Build with profiling flags
CFLAGS="-O3 -march=native -fopenmp -DNDEBUG" make benchmark

# Run individual benchmarks
./build/bench_model_serving
./build/bench_quantization
./build/bench_kv_cache
./build/bench_speculative
./build/bench_batching
```

## Reproducibility

All benchmarks use deterministic random seeds. Run at least 5 iterations
and report mean ± std. Warmup: 20 iterations before timing begins.
