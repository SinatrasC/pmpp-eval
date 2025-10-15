// BAD: Missing bounds checking - out-of-bounds memory access
#include <cuda_runtime.h>

extern "C" void launch_student(const float* A, const float* B, float* C,
                               int M, int N, int K, int blockSize);

__global__ void matmul_tiled_student(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
                                     int M, int N, int K)
{
    const int TILE = 16;
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    float acc = 0.0f;
    int tiles = (N + TILE - 1) / TILE;

    for (int t = 0; t < tiles; ++t) {
        int aRow = row;
        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;
        int bCol = col;

        // BAD: No bounds checking - will access out-of-bounds memory
        As[threadIdx.y][threadIdx.x] = A[aRow * N + aCol];
        Bs[threadIdx.y][threadIdx.x] = B[bRow * K + bCol];

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    // BAD: No output bounds checking either
    C[row * K + col] = acc;
}

extern "C" void launch_student(const float* A, const float* B, float* C,
                               int M, int N, int K, int /*blockSize*/)
{
    dim3 block(16, 16);
    dim3 grid((K + 15) / 16, (M + 15) / 16);
    matmul_tiled_student<<<grid, block>>>(A, B, C, M, N, K);
}