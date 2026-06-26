#ifndef GPU_MEMORY_H
#define GPU_MEMORY_H

#include <stdint.h>
#include <stddef.h>

/* --- Memory hierarchy levels --- */
typedef enum {
    GPU_MEM_REGISTER   = 0,  /* register file (per-thread, ~256 KB/SM) */
    GPU_MEM_L1_CACHE   = 1,  /* L1 / shared memory (per-SM, ~128 KB)  */
    GPU_MEM_L2_CACHE   = 2,  /* L2 cache (chip-wide, ~40-80 MB)        */
    GPU_MEM_GLOBAL_HBM = 3   /* HBM / global DRAM (chip-wide, 40-80 GB) */
} GpuMemLevel;

typedef struct {
    GpuMemLevel level;
    size_t      totalBytes;
    size_t      freeBytes;
    size_t      bandwidthMBps;  /* theoretical peak bandwidth */
    uint32_t    latencyCycles;  /* approximate access latency in cycles */
} GpuMemTier;

/* --- Memory coalescing analyser --- */
#define COALESCE_GRANULARITY 128  /* 128-byte transaction granularity */

typedef struct {
    const uint32_t *addresses;  /* byte-offset array for a warp */
    uint32_t         warpSize;
    uint32_t         maxSegments;   /* worst-case transactions needed */
    uint32_t         coalescedSegments; /* transactions after coalescing */
    double           coalesceRatio; /* coalescedSegments / maxSegments */
    uint32_t         unalignedCount;  /* unaligned accesses */
} CoalesceReport;

/* Analyse warp-level memory coalescing.
 * addresses: byte-offsets into a contiguous 128-byte-aligned region.
 * Returns a CoalesceReport showing how many 32/64/128-byte transactions
 * are needed.  Fully coalesced → 1×128B or 2×64B per warp. */
CoalesceReport gpu_coalesce_analyze(const uint32_t *addresses,
                                    uint32_t count);

/* --- Shared memory bank conflict detector --- */
#define GPU_SHMEM_NUM_BANKS  32
#define GPU_SHMEM_BANK_WIDTH 4   /* bytes per bank */

typedef struct {
    uint32_t numAccesses;
    uint32_t numConflicts;     /* accesses hitting same bank, same cycle */
    uint32_t bankAccessCount[GPU_SHMEM_NUM_BANKS];
    uint32_t maxConflictDepth; /* worst-case threads to one bank */
} BankConflictReport;

/* addresses: byte-offsets within shared memory for one warp.
 * returns conflict analysis assuming 32 banks × 4 bytes/bank. */
BankConflictReport gpu_bank_conflict_analyze(const uint32_t *addresses,
                                             uint32_t count);

/* --- Pinned (page-locked) memory --- */
typedef struct {
    void   *hostPtr;
    size_t  size;
    int     isPinned;  /* 1 if page-locked for fast DMA */
    int     isMapped;  /* 1 if mapped to device address space */
} PinnedMemory;

PinnedMemory gpu_pinned_alloc(size_t bytes);
void          gpu_pinned_free(PinnedMemory *pm);
double        gpu_pinned_transfer_rate(void);

/* --- Unified memory (UVA / managed memory) --- */
typedef struct {
    void   *ptr;
    size_t  size;
    int     preferredLocation;  /* 0=device, 1=host */
    int     currentLocation;    /* where data currently resides */
    int     migrationCount;     /* page migrations so far */
} UnifiedMem;

UnifiedMem gpu_unified_alloc(size_t bytes, int preferredLocation);
void        gpu_unified_prefetch(UnifiedMem *um, int dstLocation);
void        gpu_unified_free(UnifiedMem *um);

/* --- cuBLAS tensor layout --- */
typedef enum {
    CUBLAS_ROW_MAJOR = 101,   /* CUBLAS_OP_N /  C-style  */
    CUBLAS_COL_MAJOR = 102    /* CUBLAS_OP_T / Fortran-style */
} CuBlasLayout;

typedef struct {
    CuBlasLayout layout;
    uint32_t     rows;
    uint32_t     cols;
    uint32_t     ld;      /* leading dimension */
    size_t       elemSize;
    double      *data;    /* simulated tensor data on device */
} CuBlasTensor;

/* Index into cuBLAS tensor: a[r][c] */
double cublas_tensor_get(const CuBlasTensor *t, uint32_t r, uint32_t c);
void   cublas_tensor_set(CuBlasTensor *t, uint32_t r, uint32_t c, double v);
void   cublas_tensor_reshape(CuBlasTensor *t, uint32_t rows, uint32_t cols);
void   cublas_tensor_free(CuBlasTensor *t);

#endif /* GPU_MEMORY_H */
