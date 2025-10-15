#include <cuda_runtime.h>

// BAD: Missing bounds check - will cause out-of-bounds writes
__global__ void vecMulKernel(const float* A, const float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    C[i] = A[i] * B[i];  // Missing: if (i < n)
}