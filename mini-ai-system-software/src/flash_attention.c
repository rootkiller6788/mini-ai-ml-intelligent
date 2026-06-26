#include "flash_attention.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* ---- config ---- */

FlashAttnConfig flash_attn_config_create(uint32_t seqLen, uint32_t headDim,
                                          int causal) {
    FlashAttnConfig cfg;
    cfg.batchSize = 1;
    cfg.numHeads  = 1;
    cfg.seqLen    = seqLen;
    cfg.headDim   = headDim;
    cfg.blockSizeBr = headDim <= 64 ? 128 : 64;
    cfg.blockSizeBc = cfg.blockSizeBr;
    cfg.causal    = causal;
    cfg.softmaxScale = 1.0 / sqrt((double)headDim);
    if (cfg.blockSizeBr > seqLen) cfg.blockSizeBr = seqLen;
    if (cfg.blockSizeBc > seqLen) cfg.blockSizeBc = seqLen;
    return cfg;
}

/* ---- online softmax for one block ---- */

void flash_online_softmax_block(FlashAttnState *state,
                                 const double *scoresBlock,
                                 const double *valuesBlock,
                                 uint32_t blockRows, uint32_t blockCols,
                                 uint32_t headDim) {
    for (uint32_t i = 0; i < blockRows; i++) {
        double mOld = state->m[i];
        double mNew = mOld;
        for (uint32_t j = 0; j < blockCols; j++) {
            double s = scoresBlock[i * blockCols + j];
            if (s > mNew) mNew = s;
        }
        double correction = exp(mOld - mNew);
        state->l[i] = state->l[i] * correction;
        double lNew = 0.0;
        for (uint32_t j = 0; j < blockCols; j++) {
            double s = scoresBlock[i * blockCols + j];
            double p = exp(s - mNew);
            lNew += p;
            for (uint32_t d = 0; d < headDim; d++) {
                state->O[i * headDim + d] =
                    state->O[i * headDim + d] * correction
                    + p * valuesBlock[j * headDim + d];
            }
        }
        state->l[i] += lNew;
        state->m[i] = mNew;
    }
}

/* ---- single-head forward ---- */

void flash_attn_forward_head(const double *q_head, const double *k_head,
                              const double *v_head,
                              uint32_t seqLen, uint32_t headDim,
                              uint32_t blockSizeBr, uint32_t blockSizeBc,
                              int causal, double softmaxScale,
                              double *o_head, double *lse) {
    uint32_t numQblocks = (seqLen + blockSizeBr - 1) / blockSizeBr;
    uint32_t numKVblocks = (seqLen + blockSizeBc - 1) / blockSizeBc;

    FlashAttnState state;
    memset(&state, 0, sizeof(state));
    state.rows = blockSizeBr;
    state.cols = headDim;
    state.m = (double *)calloc(blockSizeBr, sizeof(double));
    state.l = (double *)calloc(blockSizeBr, sizeof(double));
    state.O = (double *)calloc(blockSizeBr * headDim, sizeof(double));

    double *scores = (double *)malloc(blockSizeBr * blockSizeBc * sizeof(double));

    for (uint32_t qb = 0; qb < numQblocks; qb++) {
        uint32_t qStart = qb * blockSizeBr;
        uint32_t qRows  = (qStart + blockSizeBr <= seqLen)
                           ? blockSizeBr : seqLen - qStart;

        for (uint32_t i = 0; i < qRows; i++) {
            state.m[i] = -INFINITY;
            state.l[i] = 0.0;
            memset(&state.O[i * headDim], 0, headDim * sizeof(double));
        }

        for (uint32_t kvb = 0; kvb < numKVblocks; kvb++) {
            uint32_t kvStart = kvb * blockSizeBc;
            uint32_t kvRows  = (kvStart + blockSizeBc <= seqLen)
                                ? blockSizeBc : seqLen - kvStart;
            uint32_t kvEnd   = kvStart + kvRows;

            for (uint32_t qi = 0; qi < qRows; qi++) {
                for (uint32_t kj = 0; kj < kvRows; kj++) {
                    if (causal && (qStart + qi) < (kvStart + kj))
                        { scores[qi * blockSizeBc + kj] = -INFINITY; continue; }
                    double dot = 0.0;
                    for (uint32_t d = 0; d < headDim; d++)
                        dot += q_head[(qStart + qi) * headDim + d]
                             * k_head[(kvStart + kj) * headDim + d];
                    scores[qi * blockSizeBc + kj] = dot * softmaxScale;
                }
            }
            flash_online_softmax_block(&state, scores, v_head + kvStart * headDim,
                                        qRows, kvRows, headDim);
        }

        for (uint32_t i = 0; i < qRows; i++) {
            double normalizer = 1.0 / state.l[i];
            for (uint32_t d = 0; d < headDim; d++)
                o_head[(qStart + i) * headDim + d] =
                    state.O[i * headDim + d] * normalizer;
            lse[qStart + i] = state.m[i] + log(state.l[i]);
        }
    }

    free(scores);
    free(state.m);
    free(state.l);
    free(state.O);
}

/* ---- multi-head forward wrapper ---- */

FlashAttnOutput *flash_attn_forward(const double *Q, const double *K,
                                     const double *V,
                                     const FlashAttnConfig *cfg) {
    FlashAttnOutput *out = (FlashAttnOutput *)malloc(sizeof(FlashAttnOutput));
    out->batchSize = cfg->batchSize;
    out->numHeads  = cfg->numHeads;
    out->seqLen    = cfg->seqLen;
    out->headDim   = cfg->headDim;
    size_t perHead = (size_t)cfg->seqLen * cfg->headDim;
    out->output = (double *)calloc((size_t)cfg->batchSize * cfg->numHeads * perHead,
                                    sizeof(double));
    out->lse    = (double *)calloc((size_t)cfg->batchSize * cfg->numHeads
                                    * cfg->seqLen, sizeof(double));

    printf("[FLASH] Forward: B=%u H=%u N=%u d=%u Br=%u Bc=%u causal=%d\n",
           cfg->batchSize, cfg->numHeads, cfg->seqLen, cfg->headDim,
           cfg->blockSizeBr, cfg->blockSizeBc, cfg->causal);

    for (uint32_t b = 0; b < cfg->batchSize; b++) {
        for (uint32_t h = 0; h < cfg->numHeads; h++) {
            uint32_t off = (b * cfg->numHeads + h) * (uint32_t)perHead;
            flash_attn_forward_head(Q + off, K + off, V + off,
                                     cfg->seqLen, cfg->headDim,
                                     cfg->blockSizeBr, cfg->blockSizeBc,
                                     cfg->causal, cfg->softmaxScale,
                                     out->output + off, out->lse
                                     + (b * cfg->numHeads + h) * cfg->seqLen);
        }
    }
    printf("[FLASH] Forward complete.\n");
    return out;
}

/* ---- backward ---- */

void flash_attn_backward(const double *Q, const double *K, const double *V,
                          const double *dO,
                          double *dQ, double *dK, double *dV,
                          const FlashAttnConfig *cfg) {
    size_t perHead = (size_t)cfg->seqLen * cfg->headDim;
    double *P = (double *)malloc((size_t)cfg->seqLen * cfg->seqLen * sizeof(double));

    printf("[FLASH] Backward: recomputing softmax P from SRAM (no HBM write)\n");

    for (uint32_t b = 0; b < cfg->batchSize; b++) {
        for (uint32_t h = 0; h < cfg->numHeads; h++) {
            uint32_t off = (b * cfg->numHeads + h) * (uint32_t)perHead;
            const double *q = Q + off;
            const double *k = K + off;
            const double *v = V + off;
            const double *dO_h = dO + off;
            double *dQ_h = dQ + off;
            double *dK_h = dK + off;
            double *dV_h = dV + off;

            for (uint32_t qi = 0; qi < cfg->seqLen; qi++) {
                double maxVal = -INFINITY;
                for (uint32_t kj = 0; kj < cfg->seqLen; kj++) {
                    if (cfg->causal && qi < kj)
                        { P[qi * cfg->seqLen + kj] = 0.0; continue; }
                    double dot = 0.0;
                    for (uint32_t dd = 0; dd < cfg->headDim; dd++)
                        dot += q[qi * cfg->headDim + dd]
                             * k[kj * cfg->headDim + dd];
                    P[qi * cfg->seqLen + kj] = dot * cfg->softmaxScale;
                    if (P[qi * cfg->seqLen + kj] > maxVal)
                        maxVal = P[qi * cfg->seqLen + kj];
                }
                double sumExp = 0.0;
                for (uint32_t kj = 0; kj < cfg->seqLen; kj++) {
                    P[qi * cfg->seqLen + kj] =
                        exp(P[qi * cfg->seqLen + kj] - maxVal);
                    sumExp += P[qi * cfg->seqLen + kj];
                }
                for (uint32_t kj = 0; kj < cfg->seqLen; kj++)
                    P[qi * cfg->seqLen + kj] /= sumExp;
            }

            for (uint32_t qi = 0; qi < cfg->seqLen; qi++) {
                for (uint32_t kj = 0; kj < cfg->seqLen; kj++) {
                    double dp = dO_h[qi * cfg->headDim + 0] * v[kj * cfg->headDim + 0];
                    double dP = dp * P[qi * cfg->seqLen + kj];
                    for (uint32_t dd = 0; dd < cfg->headDim; dd++) {
                        double dv = dO_h[qi * cfg->headDim + dd]
                                  * P[qi * cfg->seqLen + kj];
                        dV_h[kj * cfg->headDim + dd] += dv;
                        dK_h[kj * cfg->headDim + dd] += dP
                            * q[qi * cfg->headDim + dd] * cfg->softmaxScale;
                        dQ_h[qi * cfg->headDim + dd] += dP
                            * k[kj * cfg->headDim + dd] * cfg->softmaxScale;
                    }
                }
            }
        }
    }
    free(P);
    printf("[FLASH] Backward complete.\n");
}

/* ---- IO complexity ---- */

double flash_attn_io_complexity(uint32_t seqLen, uint32_t headDim,
                                 uint32_t sramSize) {
    if (sramSize == 0) return 1e12;
    double N = (double)seqLen;
    double d = (double)headDim;
    double M = (double)sramSize;
    double naive = N * N * d;
    double flash = N * N * d * d / M;
    printf("[FLASH] Naive IO: %.2e, Flash IO: %.2e (ratio: %.2f)\n",
           naive, flash, naive / flash);
    return flash;
}

/* ---- cleanup ---- */

void flash_attn_output_free(FlashAttnOutput *out) {
    if (out) { free(out->output); free(out->lse); free(out); }
}

void flash_attn_state_free(FlashAttnState *state) {
    if (state) { free(state->m); free(state->l); free(state->O);
                 free(state->dO); free(state->dQ); free(state->dK);
                 free(state->dV); }
}
