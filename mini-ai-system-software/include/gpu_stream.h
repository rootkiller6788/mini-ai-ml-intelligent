#ifndef GPU_STREAM_H
#define GPU_STREAM_H

#include <stdint.h>
#include <stddef.h>

/* ───────────────────────────────────────────────────────────────
 * CUDA Stream & Concurrency — Kernel Launch Pipeline
 *
 * L3 — Engineering Structures:
 *   Stream management, event synchronization
 *   Concurrent kernel execution model
 *   Overlap: compute ↔ data transfer
 *
 * L4 — Standards/Theorems:
 *   Little's Law applied to GPU pipeline scheduling
 *
 * L7 — Applications:
 *   Multi-stream pipeline for DL training (H2D → Compute → D2H)
 * ─────────────────────────────────────────────────────────────── */

/* --- Stream state --- */
typedef enum {
    STREAM_IDLE      = 0,
    STREAM_RUNNING   = 1,
    STREAM_BLOCKED   = 2,  /* waiting on event */
    STREAM_ERROR     = 3
} StreamState;

/* --- CUDA stream descriptor --- */
typedef struct {
    uint32_t    streamId;
    int         priority;      /* -1=high, 0=normal, 1=low */
    StreamState state;
    uint32_t    queuedOps;     /* number of pending operations */
    uint32_t    completedOps;
    double      cumulativeTimeUs; /* total execution time in microseconds */
} CudaStream;

/* --- CUDA event descriptor --- */
typedef struct {
    uint32_t eventId;
    int      isRecorded;       /* 1 if cudaEventRecord() was called */
    int      isCompleted;      /* 1 if all preceding stream ops finished */
    uint32_t streamId;         /* stream this event belongs to */
} CudaEvent;

/* --- Stream pool --- */
#define GPU_MAX_STREAMS 32
#define GPU_MAX_EVENTS  128

typedef struct {
    CudaStream streams[GPU_MAX_STREAMS];
    uint32_t   numStreams;
    CudaEvent  events[GPU_MAX_EVENTS];
    uint32_t   numEvents;
    uint32_t   concurrentKernels;  /* max concurrent kernels supported */
} StreamPool;

/* --- Transfer descriptor (H2D / D2H) --- */
typedef enum {
    TRANSFER_H2D = 0,   /* host → device */
    TRANSFER_D2H = 1,   /* device → host */
    TRANSFER_D2D = 2    /* device → device (peer) */
} TransferDirection;

typedef struct {
    TransferDirection dir;
    size_t   bytes;
    double   bandwidthGBps;  /* link bandwidth */
    double   timeUs;         /* estimated transfer time */
    int      overlappable;   /* can overlap with compute */
} TransferOp;

/* ─── Stream Management API ─── */

/* Create/initalize a stream pool with given concurrency level */
StreamPool gpu_stream_pool_create(uint32_t maxConcurrentKernels);

/* Allocate a stream from the pool */
CudaStream *gpu_stream_alloc(StreamPool *pool, int priority);

/* Release a stream back to the pool */
void gpu_stream_free(CudaStream *stream);

/* Synchronize (block until all ops in stream complete) */
void gpu_stream_synchronize(CudaStream *stream, StreamPool *pool);

/* ─── Event API ─── */

/* Create an event */
CudaEvent *gpu_event_create(StreamPool *pool);

/* Record an event in a stream */
void gpu_event_record(CudaEvent *event, CudaStream *stream);

/* Wait for an event to complete */
int gpu_event_query(const CudaEvent *event);

/* Make a stream wait on an event */
void gpu_stream_wait_event(CudaStream *stream, const CudaEvent *event);

/* ─── Transfer API ─── */

/* Estimate transfer time using bandwidth model.
 * H2D/D2H: PCIe/NVLink bandwidth.
 * D2D: NVLink peer-to-peer bandwidth. */
TransferOp gpu_transfer_estimate(TransferDirection dir, size_t bytes,
                                  double bandwidthGBps);

/* Check if transfer can overlap with compute (requires a different engine) */
int gpu_transfer_can_overlap(const TransferOp *transfer,
                              const CudaStream *computeStream);

/* ─── Pipeline Scheduling (Little's Law) ─── */

/* Three-stage pipeline: H2D → Compute → D2H.
 * Each stage runs on a separate stream.
 * Little's Law: avg_concurrency = throughput × latency. */
typedef struct {
    TransferOp   h2d;
    double        computeTimeUs;  /* kernel execution time */
    TransferOp   d2h;
    uint32_t     numIterations;
    double        totalTimeUs;    /* pipelined total time */
    double        throughputPerSec; /* iterations per second */
    double        avgConcurrency;   /* Little's Law: L = λW */
} PipelineSchedule;

/* Build a 3-stage pipeline schedule.
 * Returns estimated total time and throughput. */
PipelineSchedule gpu_pipeline_schedule(const TransferOp *h2d,
                                        double computeTimeUs,
                                        const TransferOp *d2h,
                                        uint32_t numIterations);

/* Compute the theoretical maximum overlap ratio.
 * If all 3 stages are balanced: ~3× throughput improvement.
 * Imbalanced: bottleneck determines throughput. */
double gpu_pipeline_overlap_ratio(const PipelineSchedule *sched);

/* ─── Priority-based scheduling ─── */

/* Priority inversion detection: a low-priority stream waiting on
 * an event from a high-priority stream (should not happen). */
int gpu_detect_priority_inversion(const StreamPool *pool);

/* Stream occupancy analyzer: how many active streams at each priority */
void gpu_stream_occupancy_report(const StreamPool *pool,
                                  uint32_t *highCount,
                                  uint32_t *normalCount,
                                  uint32_t *lowCount);

#endif /* GPU_STREAM_H */
