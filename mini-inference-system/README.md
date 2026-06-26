# mini-inference-system — 推理系统 (C 语言实现)

Production-grade LLM inference system in pure C99. Model serving with Triton-compatible
repository, INT8 quantization, KV cache with PagedAttention, speculative decoding,
and advanced batching strategies.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Inference Server                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐               │
│  │  Model   │  │  Quant   │  │  KV      │               │
│  │  Serving │  │  INT8    │  │  Cache   │               │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘               │
│       │             │             │                      │
│  ┌────┴─────────────┴─────────────┴─────┐               │
│  │         Batching Strategy             │               │
│  └────────────────┬─────────────────────┘               │
│  ┌────────────────┴─────────────────────┐               │
│  │       Speculative Decode             │               │
│  └──────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────┘
```

## Modules

| Module | Description | Lines |
|--------|-------------|-------|
| `model_serving` | Model load/unload, batching, scheduling, protocols | 130+ |
| `quantization_int8` | INT8 PTQ, GPTQ, AWQ, GEMM, activation quant | 130+ |
| `kv_cache` | PagedAttention, prefix caching, sliding window | 130+ |
| `speculative_decode` | Draft/target verification, Medusa heads | 130+ |
| `batching_strategy` | Static/dynamic/continuous/in-flight batching | 130+ |

## Build

```sh
make all          # build library and all examples/demos
make demo         # build and run demos
make test         # build and run tests
make benchmark    # build and run benchmarks
```

## Requirements

- C99 compiler (GCC 9+, Clang 12+, MSVC 2019+)
- pthreads
- OpenMP (optional, for GEMM acceleration)
- No external ML framework dependencies
