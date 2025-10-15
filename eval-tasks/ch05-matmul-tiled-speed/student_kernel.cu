// student_kernel.cu
#include <cuda_runtime.h>
#include <cstdio>

#ifndef TILE
#define TILE 16
#endif

// Students implement this kernel: C[M x K] = A[M x N] * B[N x K]
// One thread computes one C element; shared-memory tiled load of A and B.
__global__ void matmul_tiled_student_kernel(const float* __restrict__ A,
                                            const float* __restrict__ B,
                                            float* __restrict__ C,
                                            int M, int N, int K) {
    // TODO:
    // - Compute (row, col) from blockIdx/threadIdx
    // - Loop over tiles of N dimension
    // - Use shared memory tiles for A (TILE x TILE) and B (TILE x TILE)
    // - Guard for out-of-bounds loads/stores
    // - Accumulate sum into a register and store to C[row*K + col]

    // Hints (remove after implementing):
    // extern __shared__ float smem[]; // or static shared tiles
    // __shared__ float As[TILE][TILE], Bs[TILE][TILE];
    // int row = blockIdx.y * blockDim.y + threadIdx.y;
    // int col = blockIdx.x * blockDim.x + threadIdx.x;

    // --- your code here ---

    // Placeholder stub (compiles but gives wrong results):
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < K) {
        C[row * K + col] = 0.0f; // TODO: replace with proper tiled computation
    }
}

// Host wrapper called by test harness
extern "C" void matmul_student(const float* dA, const float* dB, float* dC,
                               int M, int N, int K, int tile) {
    dim3 block(TILE, TILE);
    dim3 grid((K + TILE - 1)/TILE, (M + TILE - 1)/TILE);
    // You may ignore `tile` and use TILE macro; the harness passes TILE=16.
    matmul_tiled_student_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
}