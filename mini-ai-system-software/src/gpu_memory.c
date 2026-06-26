#include "gpu_memory.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* --- Memory tier information --- */

static const GpuMemTier g_gpuA100Tiers[] = {
    { GPU_MEM_REGISTER,   256 * 1024,  256 * 1024, 0, 0 },     /* per SM */
    { GPU_MEM_L1_CACHE,   192 * 1024,  128 * 1024, 0, 28 },    /* per SM */
    { GPU_MEM_L2_CACHE,   40  * 1024 * 1024, 38  * 1024 * 1024, 0, 200 },
    { GPU_MEM_GLOBAL_HBM, 80ull * 1024 * 1024 * 1024,
                          78ull * 1024 * 1024 * 1024,
                          2039000, 400 }
};

void gpu_mem_tier_info(GpuMemLevel level, GpuMemTier *tier) {
    if (level <= GPU_MEM_GLOBAL_HBM) *tier = g_gpuA100Tiers[level];
    else memset(tier, 0, sizeof(*tier));
}

/* ---- coalescing analyser ---- */

CoalesceReport gpu_coalesce_analyze(const uint32_t *addresses,
                                     uint32_t count) {
    CoalesceReport rpt;
    memset(&rpt, 0, sizeof(rpt));
    rpt.addresses = addresses;
    rpt.warpSize = count;
    rpt.maxSegments = count;

    if (count == 0) return rpt;

    uint32_t segLo = addresses[0] / COALESCE_GRANULARITY;
    uint32_t segHi = segLo;
    for (uint32_t i = 0; i < count; i++) {
        uint32_t a = addresses[i];
        uint32_t seg = a / COALESCE_GRANULARITY;
        if (a % 4 != 0) rpt.unalignedCount++;
        if (seg < segLo) segLo = seg;
        if (seg > segHi) segHi = seg;
    }
    rpt.coalescedSegments = segHi - segLo + 1;
    if (rpt.coalescedSegments == 0) rpt.coalescedSegments = 1;
    rpt.coalesceRatio = (double)rpt.coalescedSegments / (double)rpt.maxSegments;

    printf("[COALESCE] %u threads access %u segments (%.0f%% coalesced, %u unaligned)\n",
           count, rpt.coalescedSegments,
           100.0 / rpt.coalesceRatio,
           rpt.unalignedCount);
    return rpt;
}

/* ---- bank conflict detector ---- */

static uint32_t shmem_bank(uint32_t byteAddr) {
    return (byteAddr / GPU_SHMEM_BANK_WIDTH) % GPU_SHMEM_NUM_BANKS;
}

BankConflictReport gpu_bank_conflict_analyze(const uint32_t *addresses,
                                              uint32_t count) {
    BankConflictReport rpt;
    memset(&rpt, 0, sizeof(rpt));
    rpt.numAccesses = count;

    for (uint32_t i = 0; i < count; i++) {
        uint32_t bank = shmem_bank(addresses[i]);
        if (bank >= GPU_SHMEM_NUM_BANKS) continue;
        rpt.bankAccessCount[bank]++;
        if (rpt.bankAccessCount[bank] > rpt.maxConflictDepth)
            rpt.maxConflictDepth = rpt.bankAccessCount[bank];
    }

    for (uint32_t b = 0; b < GPU_SHMEM_NUM_BANKS; b++) {
        if (rpt.bankAccessCount[b] > 1)
            rpt.numConflicts += rpt.bankAccessCount[b] - 1;
    }

    printf("[BANK] %u-way conflict depth, %u total conflicts (over %u banks)\n",
           rpt.maxConflictDepth, rpt.numConflicts, GPU_SHMEM_NUM_BANKS);
    return rpt;
}

/* ---- pinned memory ---- */

PinnedMemory gpu_pinned_alloc(size_t bytes) {
    PinnedMemory pm;
    pm.size = bytes;
    pm.isPinned = 1;
    pm.isMapped = 0;
    pm.hostPtr = malloc(bytes);
    if (pm.hostPtr) memset(pm.hostPtr, 0, bytes);
    printf("[PINNED] Allocated %zu bytes page-locked memory\n", bytes);
    return pm;
}

void gpu_pinned_free(PinnedMemory *pm) {
    if (pm && pm->hostPtr) { free(pm->hostPtr); pm->hostPtr = NULL; }
}

double gpu_pinned_transfer_rate(void) {
    return 25.0; /* GB/s typical for PCIe 4.0 x16 pinned transfer */
}

/* ---- unified memory ---- */

UnifiedMem gpu_unified_alloc(size_t bytes, int preferredLocation) {
    UnifiedMem um;
    um.size = bytes;
    um.preferredLocation = preferredLocation;
    um.currentLocation = preferredLocation;
    um.migrationCount = 0;
    um.ptr = malloc(bytes);
    if (um.ptr) memset(um.ptr, 0, bytes);
    printf("[UVA] Allocated %zu bytes unified memory (preferred: %s)\n",
           bytes, preferredLocation ? "host" : "device");
    return um;
}

void gpu_unified_prefetch(UnifiedMem *um, int dstLocation) {
    if (!um || um->currentLocation == dstLocation) return;
    printf("[UVA] Migrating %zu bytes: %s → %s\n",
           um->size,
           um->currentLocation ? "host" : "device",
           dstLocation ? "host" : "device");
    um->currentLocation = dstLocation;
    um->migrationCount++;
}

void gpu_unified_free(UnifiedMem *um) {
    if (um && um->ptr) { free(um->ptr); um->ptr = NULL; }
}

/* ---- cuBLAS tensor ---- */

double cublas_tensor_get(const CuBlasTensor *t, uint32_t r, uint32_t c) {
    if (!t || !t->data || r >= t->rows || c >= t->cols) return 0.0;
    if (t->layout == CUBLAS_ROW_MAJOR)
        return t->data[r * t->ld + c];
    else
        return t->data[c * t->ld + r];
}

void cublas_tensor_set(CuBlasTensor *t, uint32_t r, uint32_t c, double v) {
    if (!t || !t->data || r >= t->rows || c >= t->cols) return;
    if (t->layout == CUBLAS_ROW_MAJOR)
        t->data[r * t->ld + c] = v;
    else
        t->data[c * t->ld + r] = v;
}

void cublas_tensor_reshape(CuBlasTensor *t, uint32_t rows, uint32_t cols) {
    if (!t) return;
    if (t->ld < cols) t->ld = cols;
    t->rows = rows;
    t->cols = cols;
}

void cublas_tensor_free(CuBlasTensor *t) {
    if (t && t->data) { free(t->data); t->data = NULL; }
}
