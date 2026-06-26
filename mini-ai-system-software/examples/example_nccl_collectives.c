#include "nccl_collectives.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Simulate an AllReduce gradient sync across multiple GPUs */
static void demo_allreduce_ring(void) {
    printf("\n=== NCCL AllReduce Ring Algorithm ===\n");

    uint32_t numGPUs = 8;
    NcclComm comm = nccl_comm_init(numGPUs, 0, NCCL_INTER_NVLINK, 100.0);

    printf("Ring order: ");
    for (uint32_t i = 0; i < numGPUs; i++)
        printf("GPU%u ", comm.ringOrder[i]);
    printf("\n");

    /* Simulate per-GPU gradient data */
    double gradients[128];
    double reduced[128];
    for (int i = 0; i < 128; i++) gradients[i] = (double)(i + 1) * 0.01;

    NcclStatus st = nccl_allreduce(gradients, reduced, 128,
                                    NCCL_FLOAT64, NCCL_OP_SUM, &comm);
    printf("AllReduce status: %s\n", st == NCCL_STATUS_OK ? "OK" : "ERROR");

    /* Bandwidth analysis */
    double effBW = nccl_allreduce_bandwidth(numGPUs, 100.0);
    printf("\nAnalytical bandwidth:\n");
    printf("  Formula: 2*(N-1)/N * BW\n");
    printf("  N=%u: 2*(%u)/%u * 100 GB/s = %.2f GB/s\n",
           numGPUs, numGPUs - 1, numGPUs, effBW);
}

/* Demonstrate all collectives */
static void demo_collectives(void) {
    printf("\n=== NCCL Collective Operations ===\n");

    uint32_t numGPUs = 4;
    NcclComm comm = nccl_comm_init(numGPUs, 0, NCCL_INTER_NVLINK, 50.0);

    double send[32], recv[128];
    memset(send, 0, sizeof(send));
    memset(recv, 0, sizeof(recv));

    nccl_broadcast(send, recv, 32, NCCL_FLOAT64, 0, &comm);

    nccl_allgather(send, recv, 32, NCCL_FLOAT64, &comm);

    nccl_reducescatter(send, recv, 32, NCCL_FLOAT64, NCCL_OP_SUM, &comm);

    nccl_send(send, 32, NCCL_FLOAT64, 1, &comm);
    nccl_recv(recv, 32, NCCL_FLOAT64, 1, &comm);

    printf("\nCollective summary:\n");
    printf("  AllReduce:  reduce all → broadcast result\n");
    printf("  Broadcast:  one root → all GPUs\n");
    printf("  AllGather:  N chunks → N×N concatenation\n");
    printf("  ReduceScatter: reduce N×N → N chunks (scattered)\n");
    printf("  Send/Recv:   point-to-point unicast\n");
}

/* Compare NVLink vs PCIe bandwidth */
static void demo_interconnect_compare(void) {
    printf("\n=== NVLink vs PCIe Bandwidth ===\n");

    printf("NVLink generations:\n");
    for (int gen = 1; gen <= 4; gen++) {
        double bw = nccl_nvlink_bandwidth(gen);
        printf("  Gen %d: %.0f GB/s per link (×%d links on A100)\n",
               gen, bw, gen == 3 ? 12 : gen == 4 ? 18 : 6);
    }

    printf("PCIe generations (×16):\n");
    for (int gen = 3; gen <= 5; gen++) {
        double bw = nccl_pcie_bandwidth(gen);
        printf("  Gen %d: %.1f GB/s\n", gen, bw);
    }

    printf("\nGradient sync bytes for 175B params (fp16, 8 GPUs):\n");
    double gb = nccl_gradient_sync_bytes(175000000000ULL, 2, 8);
    printf("  %.2f GB transferred during AllReduce\n", gb / 1e9);
}

/* Demonstrate ring ordering and topology */
static void demo_ring_topology(void) {
    printf("\n=== Ring Topology Demo ===\n");

    uint32_t numGPUs = 6;

    /* Case: 2 nodes × 3 GPUs, high intra-node NVLINK, low inter-node PCIe */
    printf("2 nodes × 3 GPUs (intra: NVLink 100 GB/s, inter: PCIe 32 GB/s)\n");
    NcclComm comm = nccl_comm_init(numGPUs, 0, NCCL_INTER_NVLINK, 100.0);

    printf("Ring traversal:\n");
    for (uint32_t i = 0; i < numGPUs; i++) {
        uint32_t cur  = comm.ringOrder[i];
        uint32_t next = comm.ringOrder[(i + 1) % numGPUs];
        printf("  GPU%u → GPU%u\n", cur, next);
    }

    printf("\nAllReduce phases (ring):\n");
    printf("  Phase 1 - ReduceScatter (N-1 steps):\n");
    for (uint32_t step = 0; step < numGPUs - 1; step++)
        printf("    Step %u: each GPU sends 1/N chunk → neighbor reduce\n", step);
    printf("  Phase 2 - AllGather (N-1 steps):\n");
    for (uint32_t step = 0; step < numGPUs - 1; step++)
        printf("    Step %u: each GPU sends reduced chunk → neighbor\n", step);
}

int main(void) {
    printf("=== NCCL Collective Communication Demo ===\n");

    demo_allreduce_ring();
    demo_collectives();
    demo_interconnect_compare();
    demo_ring_topology();

    printf("\n=== Demo complete ===\n");
    return 0;
}
