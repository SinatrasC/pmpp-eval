#include <cuda_runtime.h>

// BAD: Does addition instead of multiplication
__global__ void vecMulKernel(const float* A, const float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] + B[i];  // Should be A[i] * B[i]
}