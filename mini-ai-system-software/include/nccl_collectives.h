#ifndef NCCL_COLLECTIVES_H
#define NCCL_COLLECTIVES_H

#include <stdint.h>
#include <stddef.h>

/* --- Collective operations --- */
typedef enum {
    NCCL_OP_SUM  = 0,
    NCCL_OP_PROD = 1,
    NCCL_OP_MIN  = 2,
    NCCL_OP_MAX  = 3
} NcclOp;

/* --- Reduction data type --- */
typedef enum {
    NCCL_FLOAT32 = 0,
    NCCL_FLOAT64 = 1,
    NCCL_INT32   = 2
} NcclDataType;

/* --- Interconnect type --- */
typedef enum {
    NCCL_INTER_NVLINK = 0,
    NCCL_INTER_PCIE   = 1,
    NCCL_INTER_IB     = 2   /* InfiniBand */
} NcclInterconnect;

/* --- Bandwidth pair (uni-directional) --- */
typedef struct {
    NcclInterconnect type;
    double           bandwidthGBps;  /* GB/s per direction */
} NcclLink;

/* --- Communicator: set of GPUs with topology --- */
#define NCCL_MAX_GPUS 16

typedef struct {
    uint32_t rank;           /* this GPU's rank             */
    uint32_t nRanks;         /* total number of GPUs        */
    NcclLink links[NCCL_MAX_GPUS][NCCL_MAX_GPUS]; /* topology matrix */
    uint32_t ringOrder[NCCL_MAX_GPUS]; /* nearest-neighbour ring */
} NcclComm;

/* Initialize a communicator with a given topology.
 * gpuCount: number of GPUs
 * linkType: uniform link type between all pairs (for simplicity)
 * bandwidthGBps: bandwidth of each link */
NcclComm nccl_comm_init(uint32_t gpuCount, uint32_t thisRank,
                         NcclInterconnect linkType, double bandwidthGBps);

/* Build the ring ordering: greedy nearest-neighbour based on bandwidth. */
void nccl_comm_build_ring(NcclComm *comm);

/* --- Point-to-point --- */
typedef enum {
    NCCL_STATUS_OK    = 0,
    NCCL_STATUS_ERROR = 1
} NcclStatus;

NcclStatus nccl_send(const void *src, size_t count, NcclDataType dtype,
                     uint32_t dstRank, const NcclComm *comm);
NcclStatus nccl_recv(void *dst, size_t count, NcclDataType dtype,
                     uint32_t srcRank, const NcclComm *comm);

/* --- Collectives --- */

/* AllReduce: reduce values across all GPUs, distribute result to all.
 * Ring algorithm: N-1 rounds of send→recv→reduce. */
NcclStatus nccl_allreduce(const void *sendbuff, void *recvbuff,
                           size_t count, NcclDataType dtype, NcclOp op,
                           const NcclComm *comm);

/* Broadcast: root sends data to all other GPUs.
 * Ring algorithm: log2(N) rounds (tree) or N-1 rounds (chain). */
NcclStatus nccl_broadcast(const void *sendbuff, void *recvbuff,
                           size_t count, NcclDataType dtype,
                           uint32_t root, const NcclComm *comm);

/* AllGather: each GPU sends its data, result is concatenation for all. */
NcclStatus nccl_allgather(const void *sendbuff, void *recvbuff,
                           size_t sendCount, NcclDataType dtype,
                           const NcclComm *comm);

/* ReduceScatter: reduce values, scatter result slices among GPUs. */
NcclStatus nccl_reducescatter(const void *sendbuff, void *recvbuff,
                               size_t recvCount, NcclDataType dtype,
                               NcclOp op, const NcclComm *comm);

/* --- Bandwidth / performance models --- */

/* Theoretical AllReduce bandwidth: 2*(N-1)/N * BW (ring algorithm) */
double nccl_allreduce_bandwidth(uint32_t nRanks, double linkBW);

/* NVLink vs PCIe: NVLink typically 50-900 GB/s, PCIe 4.0 ~32 GB/s x16 */
double nccl_nvlink_bandwidth(int generation);
double nccl_pcie_bandwidth(int generation);

/* Gradient synchronization: bytes transferred = 2*(N-1)/N * total_params */
double nccl_gradient_sync_bytes(size_t paramCount, size_t bytesPerParam,
                                 uint32_t nRanks);

#endif /* NCCL_COLLECTIVES_H */
