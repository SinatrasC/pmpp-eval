// BAD: No shared memory - naive implementation without tiling
#include <cuda_runtime.h>

extern "C" void launch_student(const float* A, const float* B, float* C,
                               int M, int N, int K, int blockSize);

__global__ void matmul_tiled_student(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
                                     int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < K) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * K + col];  // No tiling, direct global access
        }
        C[row * K + col] = sum;
    }
}

extern "C" void launch_student(const float* A, const float* B, float* C,
                               int M, int N, int K, int /*blockSize*/)
{
    dim3 block(16, 16);
    dim3 grid((K + 15) / 16, (M + 15) / 16);
    matmul_tiled_student<<<grid, block>>>(A, B, C, M, N, K);
}