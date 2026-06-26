#ifndef TRITON_COMPILE_H
#define TRITON_COMPILE_H

#include <stdint.h>
#include <stddef.h>

/* --- Compiler pipeline stages --- */
typedef enum {
    TRITON_DSL      = 0,  /* Python DSL (@triton.jit)     */
    TRITON_TTIR     = 1,  /* Triton IR (high-level)       */
    TRITON_TTGIR    = 2,  /* Triton GPU IR (lowered)      */
    TRITON_LLVM_IR  = 3,  /* LLVM IR                      */
    TRITON_PTX      = 4,  /* PTX assembly (NVIDIA)        */
    TRITON_CUBIN    = 5   /* binary (cubin)               */
} TritonStage;

/* --- Block program descriptor --- */
typedef struct {
    uint32_t pid;        /* program (block) id: tl.program_id(axis) */
    uint32_t numPids;    /* total number of blocks: tl.num_programs(axis) */
    uint32_t gridDimX;   /* grid dimensions */
    uint32_t gridDimY;
    uint32_t gridDimZ;
} BlockProgram;

/* --- Tensor block (tile) descriptor --- */
typedef struct {
    double  *data;       /* flat pointer to global tensor data   */
    uint32_t rows;       /* tile height                          */
    uint32_t cols;       /* tile width                           */
    uint32_t ld;         /* leading dimension of source tensor   */
    uint32_t rowOffset;  /* starting row in source tensor        */
    uint32_t colOffset;  /* starting col in source tensor        */
    int      hasMask;    /* whether loading requires boundary mask */
} TritonTile;

/* --- tl.load parameters --- */
typedef struct {
    int      useMask;         /* apply boundary-checking mask       */
    double   maskingValue;    /* fill value for out-of-bounds       */
    int      cacheModifier;   /* ".ca" / ".cg" cache hint           */
    int      evictionPolicy;  /* "evict_first" / "evict_last"       */
} TritonLoadOpts;

/* --- tl.store parameters --- */
typedef struct {
    int useMask;
    int cacheModifier;
    int evictionPolicy;
} TritonStoreOpts;

/* --- tl.atomic_add descriptor --- */
typedef struct {
    double *ptr;         /* destination pointer in global/shared mem */
    double  value;       /* value to add                            */
    int     inSharedMem; /* 1 if destination is in shared memory    */
} TritonAtomicAdd;

/* --- Autotuning configuration --- */
#define TRITON_MAX_CONFIGS 64

typedef struct {
    uint32_t blockM;       /* tile rows (M dimension)       */
    uint32_t blockN;       /* tile cols (N dimension)       */
    uint32_t blockK;       /* tile inner (K dimension)      */
    uint32_t numWarps;     /* warps per block               */
    uint32_t numStages;    /* pipeline stages               */
    double   estimatedTflops; /* performance estimate       */
} TritonTuneConfig;

typedef struct {
    TritonTuneConfig configs[TRITON_MAX_CONFIGS];
    uint32_t         numConfigs;
    double           bestTflops;
    uint32_t         bestIndex;
} TritonAutotuneResult;

/* --- Compile context (simulated) --- */
typedef struct {
    TritonStage currentStage;
    char       *sourceCode;      /* Python DSL source        */
    char       *ttirCode;        /* TTIR dump               */
    char       *ttgirCode;       /* TTGIR dump              */
    char       *llvmIR;          /* LLVM IR dump            */
    char       *ptxCode;         /* PTX dump                */
    uint32_t    compileTimeMs;   /* simulated compilation time */
} TritonCompileContext;

/* --- API --- */

/* Create a block program descriptor */
BlockProgram triton_block_program(uint32_t pid, uint32_t numPids,
                                   uint32_t gx, uint32_t gy, uint32_t gz);

/* Simulate tl.load: copy a tile from global→local registers with optional mask */
void triton_tile_load(TritonTile *dst, const double *srcTensor,
                       uint32_t tensorRows, uint32_t tensorCols,
                       const TritonLoadOpts *opts);

/* Simulate tl.store: write a tile from registers to global memory */
void triton_tile_store(double *dstTensor, const TritonTile *src,
                        uint32_t tensorRows, uint32_t tensorCols,
                        const TritonStoreOpts *opts);

/* Simulate tl.atomic_add */
void triton_atomic_add(TritonAtomicAdd *op);

/* 2D block-tiled matrix multiplication: C = A × B
 * A: M×K, B: K×N, C: M×N
 * Tiles A into blockM×blockK blocks, B into blockK×blockN blocks.
 * Uses registers for the accumulator tile. */
void triton_matmul_blocked(const double *A, const double *B, double *C,
                            uint32_t M, uint32_t N, uint32_t K,
                            uint32_t blockM, uint32_t blockN, uint32_t blockK);

/* Autotuning: search tile sizes for best estimated performance */
TritonAutotuneResult triton_autotune_matmul(uint32_t M, uint32_t N, uint32_t K,
                                             double maxTflops);

/* Compile lifecycle (simulated) */
TritonCompileContext *triton_compile_begin(const char *dslSource);
void                   triton_compile_stage(TritonCompileContext *ctx,
                                             TritonStage stage);
void                   triton_compile_end(TritonCompileContext *ctx);
void                   triton_compile_print(const TritonCompileContext *ctx);

/* Utility: compute expected TFLOPS given tile config */
double triton_tile_tflops(uint32_t blockM, uint32_t blockN, uint32_t blockK,
                           double clockGHz, uint32_t numSMs);

#endif /* TRITON_COMPILE_H */
