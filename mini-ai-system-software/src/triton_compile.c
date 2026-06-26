#include "triton_compile.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* ---- block program ---- */

BlockProgram triton_block_program(uint32_t pid, uint32_t numPids,
                                   uint32_t gx, uint32_t gy, uint32_t gz) {
    BlockProgram bp;
    bp.pid      = pid;
    bp.numPids  = numPids;
    bp.gridDimX = gx;
    bp.gridDimY = gy;
    bp.gridDimZ = gz;
    return bp;
}

/* ---- tl.load: tile from global → local registers ---- */

void triton_tile_load(TritonTile *dst, const double *srcTensor,
                       uint32_t tensorRows, uint32_t tensorCols,
                       const TritonLoadOpts *opts) {
    if (!dst || !srcTensor) return;
    for (uint32_t r = 0; r < dst->rows; r++) {
        for (uint32_t c = 0; c < dst->cols; c++) {
            uint32_t gr = dst->rowOffset + r;
            uint32_t gc = dst->colOffset + c;
            if (gr >= tensorRows || gc >= tensorCols) {
                dst->data[r * dst->cols + c] =
                    opts && opts->useMask ? opts->maskingValue : 0.0;
                continue;
            }
            dst->data[r * dst->cols + c] =
                srcTensor[gr * dst->ld + gc];
        }
    }
    printf("[TRITON] tl.load: tile (%u×%u) at (%u,%u) from tensor (%u×%u)\n",
           dst->rows, dst->cols, dst->rowOffset, dst->colOffset,
           tensorRows, tensorCols);
}

/* ---- tl.store: tile from registers → global memory ---- */

void triton_tile_store(double *dstTensor, const TritonTile *src,
                        uint32_t tensorRows, uint32_t tensorCols,
                        const TritonStoreOpts *opts) {
    if (!dstTensor || !src) return;
    for (uint32_t r = 0; r < src->rows; r++) {
        for (uint32_t c = 0; c < src->cols; c++) {
            uint32_t gr = src->rowOffset + r;
            uint32_t gc = src->colOffset + c;
            if (gr >= tensorRows || gc >= tensorCols) {
                if (!opts || !opts->useMask) continue;
                continue;
            }
            dstTensor[gr * src->ld + gc] = src->data[r * src->cols + c];
        }
    }
    printf("[TRITON] tl.store: tile (%u×%u) → tensor at (%u,%u)\n",
           src->rows, src->cols, src->rowOffset, src->colOffset);
}

/* ---- tl.atomic_add ---- */

void triton_atomic_add(TritonAtomicAdd *op) {
    if (!op || !op->ptr) return;
    double old = *op->ptr;
    *op->ptr += op->value;
    printf("[TRITON] tl.atomic_add: %p += %.6f (%s, %.6f → %.6f)\n",
           (void *)op->ptr, op->value,
           op->inSharedMem ? "shared" : "global",
           old, *op->ptr);
}

/* ---- 2D block-tiled matrix multiply ---- */

void triton_matmul_blocked(const double *A, const double *B, double *C,
                            uint32_t M, uint32_t N, uint32_t K,
                            uint32_t blockM, uint32_t blockN,
                            uint32_t blockK) {
    printf("[TRITON] Blocked matmul: (%u×%u)×(%u×%u) with tiles "
           "(bm=%u,bn=%u,bk=%u)\n", M, K, K, N, blockM, blockN, blockK);

    for (uint32_t i = 0; i < M; i += blockM) {
        uint32_t mEnd = (i + blockM < M) ? i + blockM : M;
        for (uint32_t j = 0; j < N; j += blockN) {
            uint32_t nEnd = (j + blockN < N) ? j + blockN : N;
            for (uint32_t ii = i; ii < mEnd; ii++) {
                for (uint32_t jj = j; jj < nEnd; jj++) {
                    C[ii * N + jj] = 0.0;
                }
            }
            for (uint32_t kk = 0; kk < K; kk += blockK) {
                uint32_t kEnd = (kk + blockK < K) ? kk + blockK : K;
                for (uint32_t ii = i; ii < mEnd; ii++) {
                    for (uint32_t jj = j; jj < nEnd; jj++) {
                        double acc = 0.0;
                        for (uint32_t k = kk; k < kEnd; k++)
                            acc += A[ii * K + k] * B[k * N + jj];
                        C[ii * N + jj] += acc;
                    }
                }
            }
        }
    }
}

/* ---- autotuning ---- */

double triton_tile_tflops(uint32_t blockM, uint32_t blockN, uint32_t blockK,
                           double clockGHz, uint32_t numSMs) {
    double flops = 2.0 * (double)blockM * (double)blockN * (double)blockK;
    double secondsPerBlock = flops / (clockGHz * 1e9 * 64.0); /* rough */
    return (flops * numSMs) / (secondsPerBlock * 1e12 + 1e-9);
}

TritonAutotuneResult triton_autotune_matmul(uint32_t M, uint32_t N, uint32_t K,
                                             double maxTflops) {
    TritonAutotuneResult res;
    memset(&res, 0, sizeof(res));
    res.bestTflops = 0.0;
    res.bestIndex  = 0;

    static const uint32_t candidateSizes[] = {16, 32, 64, 128, 256};
    uint32_t nc = (uint32_t)(sizeof(candidateSizes) / sizeof(candidateSizes[0]));

    for (uint32_t bm = 0; bm < nc && res.numConfigs < TRITON_MAX_CONFIGS; bm++) {
        for (uint32_t bn = 0; bn < nc && res.numConfigs < TRITON_MAX_CONFIGS; bn++) {
            for (uint32_t bk = 0; bk < nc && res.numConfigs < TRITON_MAX_CONFIGS; bk++) {
                uint32_t cm = candidateSizes[bm];
                uint32_t cn = candidateSizes[bn];
                uint32_t ck = candidateSizes[bk];
                double tflops = triton_tile_tflops(cm, cn, ck, 1.41, 108);
                uint32_t idx = res.numConfigs++;
                res.configs[idx].blockM    = cm;
                res.configs[idx].blockN    = cn;
                res.configs[idx].blockK    = ck;
                res.configs[idx].numWarps  = (cm * cn) / 128 + 1;
                res.configs[idx].numStages = 2;
                res.configs[idx].estimatedTflops = tflops;
                if (tflops > res.bestTflops) {
                    res.bestTflops = tflops;
                    res.bestIndex  = idx;
                }
            }
        }
    }
    printf("[TRITON] Autotune: %u configs, best=bm%u/bn%u/bk%u @ %.2f TFLOPs\n",
           res.numConfigs,
           res.configs[res.bestIndex].blockM,
           res.configs[res.bestIndex].blockN,
           res.configs[res.bestIndex].blockK,
           res.bestTflops);
    return res;
}

/* ---- compile lifecycle ---- */

TritonCompileContext *triton_compile_begin(const char *dslSource) {
    TritonCompileContext *ctx =
        (TritonCompileContext *)malloc(sizeof(TritonCompileContext));
    memset(ctx, 0, sizeof(*ctx));
    ctx->currentStage = TRITON_DSL;
    ctx->sourceCode   = dslSource ? _strdup(dslSource) : NULL;
    ctx->compileTimeMs = 0;
    printf("[TRITON] Compile begin (DSL stage)\n");
    return ctx;
}

void triton_compile_stage(TritonCompileContext *ctx, TritonStage stage) {
    if (!ctx) return;
    static const char *stageNames[] = {
        "DSL", "TTIR", "TTGIR", "LLVM IR", "PTX", "CUBIN"
    };
    uint32_t stageMs[] = { 0, 15, 25, 40, 30, 10 };
    ctx->compileTimeMs += stageMs[stage];
    ctx->currentStage = stage;
    printf("[TRITON] Compile → %s (elapsed %u ms)\n",
           stageNames[stage], ctx->compileTimeMs);
}

void triton_compile_end(TritonCompileContext *ctx) {
    if (!ctx) return;
    printf("[TRITON] Compile complete: total %u ms\n", ctx->compileTimeMs);
}

void triton_compile_print(const TritonCompileContext *ctx) {
    if (!ctx) return;
    printf("=== Triton Compile Context ===\n");
    printf("  Source:   %s\n", ctx->sourceCode ? ctx->sourceCode : "(null)");
    printf("  Stage:    %d\n", ctx->currentStage);
    printf("  Compile:  %u ms\n", ctx->compileTimeMs);
    printf("=============================\n");
}
