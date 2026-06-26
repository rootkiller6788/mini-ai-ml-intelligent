# Inference System Architecture

## Overview

The mini-inference-system implements a production-grade LLM inference pipeline
in pure C99. The architecture follows the standard model serving paradigm with
optimizations used in vLLM, TensorRT-LLM, and Triton Inference Server.

## Component Architecture

```
                    ┌──────────────────────────┐
                    │      Client (gRPC/HTTP)   │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │    Model Serving Layer     │
                    │  ┌──────────────────────┐ │
                    │  │ Request Queue (FIFO/  │ │
                    │  │ Priority Scheduling)  │ │
                    │  └──────────┬───────────┘ │
                    │  ┌──────────▼───────────┐ │
                    │  │ Dynamic Batching      │ │
                    │  └──────────┬───────────┘ │
                    │  ┌──────────▼───────────┐ │
                    │  │ Batch → Infer → Split │ │
                    │  └──────────────────────┘ │
                    └────────────┬─────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
┌───────▼───────┐    ┌──────────▼──────────┐    ┌────────▼────────┐
│ INT8 Quantizer│    │   KV Cache Manager   │    │  Batching       │
│               │    │                      │    │  Scheduler      │
│ ┌───────────┐ │    │ ┌──────────────────┐ │    │                 │
│ │ Calibrator│ │    │ │ PagedAttention    │ │    │ ┌─────────────┐ │
│ │ (MinMax,  │ │    │ │ Block Manager     │ │    │ │ Static      │ │
│ │  MSE, Ent)│ │    │ └────────┬─────────┘ │    │ │ Dynamic     │ │
│ └───────────┘ │    │ ┌────────▼─────────┐ │    │ │ Continuous  │ │
│ ┌───────────┐ │    │ │ Prefix Cache      │ │    │ │ In-flight   │ │
│ │ INT8 GEMM │ │    │ └────────┬─────────┘ │    │ └─────────────┘ │
│ └───────────┘ │    │ ┌────────▼─────────┐ │    │ ┌─────────────┐ │
│ ┌───────────┐ │    │ │ FP8/INT8 Quant    │ │    │ │ Preemption  │ │
│ │ W8A8/W4A16│ │    │ └──────────────────┘ │    │ │ Priority    │ │
│ └───────────┘ │    └──────────────────────┘    │ │ TTFT/TPS    │ │
│ ┌───────────┐ │                                │ └─────────────┘ │
│ │ GPTQ/AWQ  │ │                                └─────────────────┘
│ └───────────┘ │
└───────────────┘
        │
┌───────▼───────────┐
│ Speculative Decode │
│ ┌────────────────┐ │
│ │ Draft Model     │ │
│ │ (N-gram/SmallTF)│ │
│ │ → K candidates  │ │
│ └────────┬───────┘ │
│ ┌────────▼───────┐ │
│ │ Target Verify   │ │
│ │ (Rejection Smpl)│ │
│ └────────┬───────┘ │
│ ┌────────▼───────┐ │
│ │ Accept/Reject   │ │
│ │ → Speedup 2-3x  │ │
│ └────────────────┘ │
└────────────────────┘
```

## Request Lifecycle

1. **Submit**: Client submits request via gRPC/HTTP to model serving layer
2. **Queue**: Request enters priority-sorted FIFO queue
3. **Batch**: Dynamic batcher accumulates requests into optimal batch
4. **Prefill**: KV cache prefill phase computes keys/values from input
5. **Decode**: Auto-regressive decode phase reuses cached KV entries
6. **Return**: Results streamed back to client via KFServing protocol

## Batching Strategies

| Strategy | Mechanism | Latency | Throughput | Use Case |
|----------|-----------|---------|------------|----------|
| Static | Fixed batch size | Low variance | Medium | Simple workloads |
| Dynamic | Accumulate window | Medium | High | Variable traffic |
| Continuous | Add/remove in-flight | Low | High | Production serving |
| In-flight | Non-blocking join | Best | Best | Real-time systems |

## KV Cache Memory Model

```
Layer i:
  ┌─────────────────────────────────────────────────────┐
  │ Block Table [batch, heads, block_size, head_dim]     │
  │                                                       │
  │  Block 0     Block 1     Block 2     ... Block N     │
  │ ┌─────────┐ ┌─────────┐ ┌─────────┐    ┌─────────┐ │
  │ │ K V K V │ │ K V K V │ │ K V K V │    │ K V K V │ │
  │ │ ...     │ │ ...     │ │ ...     │    │ ...     │ │
  │ │ 16 tok  │ │ 16 tok  │ │ 16 tok  │    │ 16 tok  │ │
  │ └─────────┘ └─────────┘ └─────────┘    └─────────┘ │
  └─────────────────────────────────────────────────────┘
```

## Quantization Pipeline

```
FP32 Weights
    │
    ├── Calibration (MinMax / MSE / Entropy / Percentile)
    │   Collect activation ranges from calibration dataset
    │
    ├── Symmetric Quantization
    │   scale = max(|x|) / 127.0
    │   W_int8 = round(W_fp32 / scale)
    │
    ├── GPTQ (layer-wise)
    │   Hessian matrix from calibration activations
    │   Optimal rounding via column-wise OBS
    │
    └── AWQ (activation-aware)
        Scale search per channel based on activation magnitude
```

## Speculative Decoding Protocol

```
1. Draft model generates K candidate tokens (cheap)
2. Target model runs single forward on (prefix + K candidates)
3. For each draft token i:
   if random() < min(1, p_target(token_i) / p_draft(token_i)):
       ACCEPT token_i
   else:
       ACCEPT sampled_token from max(0, p_target - p_draft)
       STOP (reject remaining)
```
