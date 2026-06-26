/*
 * gpu_stream.c — CUDA Stream & Concurrency
 *
 * Knowledge points (L3-L4-L7):
 *   L3: Stream lifecycle management, event synchronization
 *   L3: Priority-based scheduling with inversion detection
 *   L4: Little's Law (L = λW) applied to GPU pipeline
 *   L7: 3-stage DL training pipeline: H2D → Compute → D2H
 *
 * References:
 *   NVIDIA CUDA C Programming Guide §3.2.5 "Streams"
 *   Little, "A Proof for the Queuing Formula L = λW" (1961)
 *   Micikevicius, "Multi-GPU Training" (NVIDIA GTC, 2017)
 */

#include "gpu_stream.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* ─── Stream Pool Creation ─── */

StreamPool gpu_stream_pool_create(uint32_t maxConcurrentKernels) {
    StreamPool pool;
    memset(&pool, 0, sizeof(pool));
    pool.concurrentKernels = maxConcurrentKernels;
    if (pool.concurrentKernels > GPU_MAX_STREAMS)
        pool.concurrentKernels = GPU_MAX_STREAMS;
    if (pool.concurrentKernels == 0)
        pool.concurrentKernels = 4; /* default: 4-way concurrency */

    /* Pre-create streams */
    for (uint32_t i = 0; i < pool.concurrentKernels; i++) {
        pool.streams[i].streamId = i;
        pool.streams[i].priority = 0;
        pool.streams[i].state = STREAM_IDLE;
        pool.streams[i].queuedOps = 0;
        pool.streams[i].completedOps = 0;
        pool.streams[i].cumulativeTimeUs = 0.0;
    }
    pool.numStreams = pool.concurrentKernels;

    printf("[STREAM-POOL] Created pool with %u streams, "
           "%u concurrent kernels max\n",
           pool.numStreams, pool.concurrentKernels);
    return pool;
}

/* ─── Stream Allocation ─── */

CudaStream *gpu_stream_alloc(StreamPool *pool, int priority) {
    if (!pool) return NULL;
    /* Find idle stream, or create new one */
    for (uint32_t i = 0; i < pool->numStreams; i++) {
        if (pool->streams[i].state == STREAM_IDLE) {
            pool->streams[i].priority = priority;
            pool->streams[i].state = STREAM_RUNNING;
            pool->streams[i].queuedOps = 0;
            printf("[STREAM] Allocated stream %u (priority %d)\n",
                   pool->streams[i].streamId, priority);
            return &pool->streams[i];
        }
    }
    /* If all busy, use the one with fewest queued ops (fallback) */
    if (pool->numStreams < GPU_MAX_STREAMS) {
        uint32_t id = pool->numStreams++;
        pool->streams[id].streamId = id;
        pool->streams[id].priority = priority;
        pool->streams[id].state = STREAM_RUNNING;
        pool->streams[id].queuedOps = 0;
        pool->streams[id].completedOps = 0;
        printf("[STREAM] Created new stream %u (priority %d)\n", id, priority);
        return &pool->streams[id];
    }
    printf("[STREAM] WARNING: all streams busy (max %u)\n", GPU_MAX_STREAMS);
    return NULL;
}

/* ─── Stream Release ─── */

void gpu_stream_free(CudaStream *stream) {
    if (!stream) return;
    printf("[STREAM] Free stream %u: completed %u ops in %.1f us\n",
           stream->streamId, stream->completedOps, stream->cumulativeTimeUs);
    stream->state = STREAM_IDLE;
    stream->queuedOps = 0;
}

/* ─── Stream Synchronize ─── */

void gpu_stream_synchronize(CudaStream *stream, StreamPool *pool) {
    if (!stream) return;
    printf("[STREAM] Synchronize stream %u (ops queued: %u, done: %u)...\n",
           stream->streamId, stream->queuedOps, stream->completedOps);
    /* Simulate blocking until all ops finish */
    stream->completedOps = stream->queuedOps;
    stream->state = STREAM_IDLE;
    printf("[STREAM] Stream %u synchronized\n", stream->streamId);
}

/* ─── Event Create ─── */

CudaEvent *gpu_event_create(StreamPool *pool) {
    if (!pool || pool->numEvents >= GPU_MAX_EVENTS) return NULL;
    uint32_t id = pool->numEvents++;
    pool->events[id].eventId = id;
    pool->events[id].isRecorded = 0;
    pool->events[id].isCompleted = 0;
    pool->events[id].streamId = 0;
    printf("[EVENT] Created event %u\n", id);
    return &pool->events[id];
}

/* ─── Event Record ─── */

void gpu_event_record(CudaEvent *event, CudaStream *stream) {
    if (!event || !stream) return;
    event->isRecorded = 1;
    event->isCompleted = 0;
    event->streamId = stream->streamId;
    printf("[EVENT] Record event %u in stream %u\n",
           event->eventId, stream->streamId);
}

/* ─── Event Query ─── */

int gpu_event_query(const CudaEvent *event) {
    if (!event) return 1;
    return event->isCompleted ? 1 : 0;
}

/* ─── Stream Wait Event ─── */

void gpu_stream_wait_event(CudaStream *stream, const CudaEvent *event) {
    if (!stream || !event) return;
    printf("[EVENT] Stream %u waiting on event %u (stream %u)...\n",
           stream->streamId, event->eventId, event->streamId);
    /* In real CUDA: cudaStreamWaitEvent inserts a barrier.
     * All subsequent ops in 'stream' are blocked until event completes. */
    if (event->isCompleted)
        printf("[EVENT] Event %u already completed — no wait needed\n",
               event->eventId);
    else
        printf("[EVENT] Stream %u will block until event %u signals\n",
               stream->streamId, event->eventId);
}

/* ─── Transfer Estimate ─── */

TransferOp gpu_transfer_estimate(TransferDirection dir, size_t bytes,
                                  double bandwidthGBps) {
    TransferOp op;
    memset(&op, 0, sizeof(op));
    op.dir = dir;
    op.bytes = bytes;
    op.bandwidthGBps = bandwidthGBps;
    /* time = bytes / bandwidth, convert GB/s → B/us: 1 GB/s = 1000 B/us */
    op.timeUs = (bandwidthGBps > 0.0)
                ? ((double)bytes / (bandwidthGBps * 1000.0)) : 1e9;

    /* H2D and D2H use copy engine → can overlap with compute kernels.
     * D2D uses NVLink P2P, also on separate engine. */
    op.overlappable = (dir != TRANSFER_D2D) ? 1 : 1;

    const char *dirNames[] = { "H2D", "D2H", "D2D" };
    printf("[XFER] %s %zu bytes @ %.1f GB/s → %.1f us (%s)\n",
           dirNames[dir], bytes, bandwidthGBps, op.timeUs,
           op.overlappable ? "overlappable" : "serial");
    return op;
}

/* ─── Transfer Overlap Check ─── */

int gpu_transfer_can_overlap(const TransferOp *transfer,
                              const CudaStream *computeStream) {
    if (!transfer || !computeStream) return 0;
    /* Overlap possible if transfer and compute use different engines.
     * Copy engine (DMA) for H2D/D2H, compute engine for kernels.
     * Different streams allow concurrency. */
    int canOverlap = transfer->overlappable &&
                     (computeStream->priority >= 0); /* normal or low priority */
    printf("[XFER-OVERLAP] Transfer(%s) + Compute(stream %u): %s\n",
           transfer->dir == TRANSFER_H2D ? "H2D" :
           transfer->dir == TRANSFER_D2H ? "D2H" : "D2D",
           computeStream->streamId,
           canOverlap ? "CAN overlap" : "CANNOT overlap");
    return canOverlap;
}

/* ─── Pipeline Scheduling (Little's Law) ─── */

PipelineSchedule gpu_pipeline_schedule(const TransferOp *h2d,
                                        double computeTimeUs,
                                        const TransferOp *d2h,
                                        uint32_t numIterations) {
    PipelineSchedule sched;
    memset(&sched, 0, sizeof(sched));
    if (h2d) sched.h2d = *h2d;
    sched.computeTimeUs = computeTimeUs;
    if (d2h) sched.d2h = *d2h;
    sched.numIterations = numIterations;

    /* Serial execution: T_serial = N × (t_H2D + t_compute + t_D2H) */
    double tSerial = (double)numIterations
                     * (sched.h2d.timeUs + computeTimeUs + sched.d2h.timeUs);

    /* Pipelined: 3-stage software pipeline.
     * Fill: (N-1) rounds of overlap after first 2 stages.
     * Total: t_H2D + t_compute + t_D2H + (N-1)×max(t_H2D, t_compute, t_D2H) */
    double bottleneck = sched.h2d.timeUs;
    if (computeTimeUs > bottleneck) bottleneck = computeTimeUs;
    if (sched.d2h.timeUs > bottleneck) bottleneck = sched.d2h.timeUs;

    double tPipeline = sched.h2d.timeUs + computeTimeUs + sched.d2h.timeUs
                     + (double)(numIterations - 1) * bottleneck;

    sched.totalTimeUs = (numIterations >= 3) ? tPipeline : tSerial;

    /* Throughput: iterations per second */
    sched.throughputPerSec = (sched.totalTimeUs > 0.0)
                              ? (double)numIterations / sched.totalTimeUs * 1e6
                              : 0.0;

    /* Little's Law: L = λ · W
     *   λ = throughput (iterations/sec) = N / T
     *   W = latency per iteration = t_H2D + t_compute + t_D2H
     *   L = average concurrency = λ · W */
    double latency = sched.h2d.timeUs + computeTimeUs + sched.d2h.timeUs;
    sched.avgConcurrency = (numIterations / 1e6)
                          * sched.throughputPerSec * latency / 1e6;

    printf("[PIPELINE] %u iterations:\n", numIterations);
    printf("  Serial: %.1f us, Pipeline: %.1f us, Speedup: %.2f×\n",
           tSerial, tPipeline, tSerial / tPipeline);
    printf("  Throughput: %.1f iter/s\n", sched.throughputPerSec);
    printf("  Little's Law: L = λW = %.2f (%s)\n",
           sched.avgConcurrency,
           sched.avgConcurrency > 2.0 ? "under-utilized" : "balanced");
    return sched;
}

/* ─── Pipeline Overlap Ratio ─── */

double gpu_pipeline_overlap_ratio(const PipelineSchedule *sched) {
    if (!sched || sched->numIterations <= 1) return 1.0;

    double bottleneck = sched->h2d.timeUs;
    if (sched->computeTimeUs > bottleneck) bottleneck = sched->computeTimeUs;
    if (sched->d2h.timeUs > bottleneck) bottleneck = sched->d2h.timeUs;

    double avgStage = (sched->h2d.timeUs + sched->computeTimeUs
                       + sched->d2h.timeUs) / 3.0;
    double ratio = avgStage / bottleneck;

    printf("[PIPELINE] Overlap ratio: %.2f (%.1f%% pipeline utilization)\n",
           ratio, ratio * 100.0);
    return ratio;
}

/* ─── Priority Inversion Detection ─── */

int gpu_detect_priority_inversion(const StreamPool *pool) {
    if (!pool) return 0;

    /* Priority inversion: a low-priority stream holds a resource
     * that a high-priority stream needs. Detection requires tracking
     * event dependencies. Here we check for low-priority streams
     * that have pending events from higher-priority sources. */
    int inversions = 0;
    for (uint32_t i = 0; i < pool->numEvents; i++) {
        if (!pool->events[i].isCompleted) {
            uint32_t sid = pool->events[i].streamId;
            if (sid < pool->numStreams) {
                int srcPriority = pool->streams[sid].priority;
                for (uint32_t j = 0; j < pool->numStreams; j++) {
                    if (pool->streams[j].priority < srcPriority
                        && pool->streams[j].state != STREAM_IDLE) {
                        inversions++;
                    }
                }
            }
        }
    }

    printf("[PRIORITY] Inversion check: %d potential inversions detected\n",
           inversions);
    return inversions;
}

/* ─── Stream Occupancy Report ─── */

void gpu_stream_occupancy_report(const StreamPool *pool,
                                  uint32_t *highCount,
                                  uint32_t *normalCount,
                                  uint32_t *lowCount) {
    uint32_t high = 0, normal = 0, low = 0;
    for (uint32_t i = 0; i < pool->numStreams; i++) {
        if (pool->streams[i].state == STREAM_IDLE) continue;
        if (pool->streams[i].priority < 0) high++;
        else if (pool->streams[i].priority == 0) normal++;
        else low++;
    }

    if (highCount)  *highCount   = high;
    if (normalCount) *normalCount = normal;
    if (lowCount)   *lowCount    = low;

    printf("[OCCUPANCY] Active streams: %u high, %u normal, %u low "
           "(%u total out of %u)\n",
           high, normal, low, high + normal + low, pool->numStreams);
}
