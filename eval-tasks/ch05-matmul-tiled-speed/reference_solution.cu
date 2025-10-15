// reference_solution.cu
#include <cuda_runtime.h>
#include <cstdio>

#ifndef TILE
#define TILE 16
#endif

// --------- Reference naive kernel (correctness baseline) ----------
__global__ void matmul_naive_kernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= K) return;

    float acc = 0.f;
    for (int n = 0; n < N; ++n) {
        acc += A[row * N + n] * B[n * K + col];
    }
    C[row * K + col] = acc;
}

// --------- Reference tiled kernel (optimized & correct) ----------
__global__ void matmul_tiled_ref_kernel(const float* __restrict__ A,
                                        const float* __restrict__ B,
                                        float* __restrict__ C,
                                        int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * blockDim.y + threadIdx.y; // [0..M)
    int col = blockIdx.x * blockDim.x + threadIdx.x; // [0..K)

    float acc = 0.f;
    // Iterate over tiles along N
    for (int t = 0; t < (N + TILE - 1)/TILE; ++t) {
        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;

        // guarded loads
        As[threadIdx.y][threadIdx.x] = (row < M && aCol < N) ? A[row * N + aCol] : 0.f;
        Bs[threadIdx.y][threadIdx.x] = (bRow < N && col < K) ? B[bRow * K + col] : 0.f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < K) {
        C[row * K + col] = acc;
    }
}

// Host wrappers used by test harness
extern "C" void matmul_ref_naive(const float* dA, const float* dB, float* dC,
                                 int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid((K + block.x - 1)/block.x, (M + block.y - 1)/block.y);
    matmul_naive_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
}

extern "C" void matmul_ref_tiled(const float* dA, const float* dB, float* dC,
                                 int M, int N, int K, int tile /*unused*/) {
    dim3 block(TILE, TILE);
    dim3 grid((K + TILE - 1)/TILE, (M + TILE - 1)/TILE);
    matmul_tiled_ref_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
}