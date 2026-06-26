#include "nccl_collectives.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* ---- communicator init ---- */

NcclComm nccl_comm_init(uint32_t gpuCount, uint32_t thisRank,
                         NcclInterconnect linkType, double bandwidthGBps) {
    NcclComm comm;
    memset(&comm, 0, sizeof(comm));
    comm.rank   = thisRank;
    comm.nRanks = gpuCount;
    if (gpuCount > NCCL_MAX_GPUS) gpuCount = NCCL_MAX_GPUS;
    for (uint32_t i = 0; i < gpuCount; i++) {
        NcclLink link;
        link.type = linkType;
        link.bandwidthGBps = bandwidthGBps;
        comm.links[thisRank][i] = link;
        comm.links[i][thisRank] = link;
        comm.ringOrder[i] = i;
    }
    nccl_comm_build_ring(&comm);
    printf("[NCCL] Communicator: rank=%u/%u, link=%d, BW=%.1f GB/s\n",
           thisRank, gpuCount, linkType, bandwidthGBps);
    return comm;
}

/* ---- ring builder: greedy nearest-neighbour ---- */

void nccl_comm_build_ring(NcclComm *comm) {
    uint32_t n = comm->nRanks;
    if (n <= 1) return;
    uint8_t visited[NCCL_MAX_GPUS];
    memset(visited, 0, sizeof(visited));
    visited[0] = 1;
    comm->ringOrder[0] = 0;
    uint32_t cur = 0;
    for (uint32_t step = 1; step < n; step++) {
        double bestBW = -1.0;
        uint32_t best = cur;
        for (uint32_t i = 0; i < n; i++) {
            if (visited[i]) continue;
            double bw = comm->links[cur][i].bandwidthGBps;
            if (bw > bestBW) { bestBW = bw; best = i; }
        }
        visited[best] = 1;
        comm->ringOrder[step] = best;
        cur = best;
    }
    printf("[NCCL] Ring order: ");
    for (uint32_t i = 0; i < n; i++)
        printf("%u ", comm->ringOrder[i]);
    printf("\n");
}

/* ---- point-to-point ---- */

NcclStatus nccl_send(const void *src, size_t count, NcclDataType dtype,
                     uint32_t dstRank, const NcclComm *comm) {
    if (!src || dstRank >= comm->nRanks) return NCCL_STATUS_ERROR;
    size_t elemSize = (dtype == NCCL_FLOAT64) ? 8 : 4;
    printf("[NCCL] rank %u → rank %u: sending %zu elements (%.2f KB)\n",
           comm->rank, dstRank, count, (double)(count * elemSize) / 1024.0);
    return NCCL_STATUS_OK;
}

NcclStatus nccl_recv(void *dst, size_t count, NcclDataType dtype,
                     uint32_t srcRank, const NcclComm *comm) {
    if (!dst || srcRank >= comm->nRanks) return NCCL_STATUS_ERROR;
    size_t elemSize = (dtype == NCCL_FLOAT64) ? 8 : 4;
    printf("[NCCL] rank %u ← rank %u: receiving %zu elements (%.2f KB)\n",
           comm->rank, srcRank, count, (double)(count * elemSize) / 1024.0);
    return NCCL_STATUS_OK;
}

/* ---- reduce function ---- */

static void nccl_reduce_op(void *dst, const void *src, size_t count,
                            NcclDataType dtype, NcclOp op) {
    if (dtype == NCCL_FLOAT64) {
        double *d = (double *)dst;
        const double *s = (const double *)src;
        for (size_t i = 0; i < count; i++) {
            switch (op) {
            case NCCL_OP_SUM:  d[i] += s[i]; break;
            case NCCL_OP_PROD: d[i] *= s[i]; break;
            case NCCL_OP_MIN:  if (s[i] < d[i]) d[i] = s[i]; break;
            case NCCL_OP_MAX:  if (s[i] > d[i]) d[i] = s[i]; break;
            }
        }
    } else {
        float *d = (float *)dst;
        const float *s = (const float *)src;
        for (size_t i = 0; i < count; i++) {
            switch (op) {
            case NCCL_OP_SUM:  d[i] += s[i]; break;
            case NCCL_OP_PROD: d[i] *= s[i]; break;
            case NCCL_OP_MIN:  if (s[i] < d[i]) d[i] = s[i]; break;
            case NCCL_OP_MAX:  if (s[i] > d[i]) d[i] = s[i]; break;
            }
        }
    }
}

/* ---- AllReduce: ring algorithm (N-1 rounds) ---- */

NcclStatus nccl_allreduce(const void *sendbuff, void *recvbuff,
                           size_t count, NcclDataType dtype, NcclOp op,
                           const NcclComm *comm) {
    if (!sendbuff || !recvbuff) return NCCL_STATUS_ERROR;
    size_t elemSize = (dtype == NCCL_FLOAT64) ? 8 : 4;
    uint32_t n = comm->nRanks;
    uint32_t rank = comm->rank;
    size_t totalBytes = count * elemSize;

    printf("[NCCL] AllReduce: rank=%u/%u, count=%zu, total=%.2f KB\n",
           rank, n, count, (double)totalBytes / 1024.0);

    memcpy(recvbuff, sendbuff, totalBytes);
    size_t chunkSize = count; /* simplified: one chunk */

    uint32_t nextRank = (rank + 1) % n;
    uint32_t prevRank = (rank + n - 1) % n;

    for (uint32_t step = 0; step < n - 1; step++) {
        printf("[NCCL] AllReduce round %u/%u: recv from rank %u, "
               "send to rank %u\n",
               step + 1, n - 1, prevRank, nextRank);
        /* In a real NCCL ring, sends and receives are overlapped.
         * Here we simulate the data flow. */
        size_t offset = ((rank - step + n) % n) * chunkSize / n;
        printf("[NCCL]   chunk offset=%zu, size=%zu\n", offset,
               chunkSize / n);
    }

    double bw = nccl_allreduce_bandwidth(n,
                  comm->links[rank][nextRank].bandwidthGBps);
    printf("[NCCL] AllReduce complete. Effective bandwidth: %.2f GB/s\n", bw);

    return NCCL_STATUS_OK;
}

/* ---- Broadcast ---- */

NcclStatus nccl_broadcast(const void *sendbuff, void *recvbuff,
                           size_t count, NcclDataType dtype,
                           uint32_t root, const NcclComm *comm) {
    if (!sendbuff || !recvbuff) return NCCL_STATUS_ERROR;
    size_t elemSize = (dtype == NCCL_FLOAT64) ? 8 : 4;
    if (comm->rank == root)
        memcpy(recvbuff, sendbuff, count * elemSize);
    printf("[NCCL] Broadcast: root=%u → %u ranks, size=%zu elements\n",
           root, comm->nRanks, count);
    return NCCL_STATUS_OK;
}

/* ---- AllGather ---- */

NcclStatus nccl_allgather(const void *sendbuff, void *recvbuff,
                           size_t sendCount, NcclDataType dtype,
                           const NcclComm *comm) {
    if (!sendbuff || !recvbuff) return NCCL_STATUS_ERROR;
    size_t elemSize = (dtype == NCCL_FLOAT64) ? 8 : 4;
    printf("[NCCL] AllGather: %u ranks × %zu elements each = %zu total "
           "(%.2f KB)\n",
           comm->nRanks, sendCount, sendCount * comm->nRanks,
           (double)(sendCount * comm->nRanks * elemSize) / 1024.0);
    return NCCL_STATUS_OK;
}

/* ---- ReduceScatter ---- */

NcclStatus nccl_reducescatter(const void *sendbuff, void *recvbuff,
                               size_t recvCount, NcclDataType dtype,
                               NcclOp op, const NcclComm *comm) {
    if (!sendbuff || !recvbuff) return NCCL_STATUS_ERROR;
    size_t elemSize = (dtype == NCCL_FLOAT64) ? 8 : 4;
    printf("[NCCL] ReduceScatter: %u GPUs reduce → scatter %zu elements "
           "each (%.2f KB)\n",
           comm->nRanks, recvCount,
           (double)(recvCount * elemSize) / 1024.0);
    return NCCL_STATUS_OK;
}

/* ---- bandwidth models ---- */

double nccl_allreduce_bandwidth(uint32_t nRanks, double linkBW) {
    if (nRanks <= 1) return 0.0;
    return 2.0 * (double)(nRanks - 1) / (double)nRanks * linkBW;
}

double nccl_nvlink_bandwidth(int generation) {
    switch (generation) {
    case 1: return 20.0;   /* P100 */
    case 2: return 50.0;   /* V100 */
    case 3: return 100.0;  /* A100 (per link) */
    case 4: return 150.0;  /* H100 */
    default: return 0.0;
    }
}

double nccl_pcie_bandwidth(int generation) {
    switch (generation) {
    case 3: return 15.75;  /* PCIe 3.0 x16 */
    case 4: return 31.5;   /* PCIe 4.0 x16 */
    case 5: return 63.0;   /* PCIe 5.0 x16 */
    default: return 0.0;
    }
}

double nccl_gradient_sync_bytes(size_t paramCount, size_t bytesPerParam,
                                 uint32_t nRanks) {
    if (nRanks <= 1) return 0.0;
    return 2.0 * (double)(nRanks - 1) / (double)nRanks
           * (double)paramCount * (double)bytesPerParam;
}
