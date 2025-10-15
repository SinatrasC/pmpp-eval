#include <cuda_runtime.h>

// BAD: Wrong thread index calculation
__global__ void vecMulKernel(const float* A, const float* B, float* C, int n) {
    int i = threadIdx.x;  // Missing blockIdx.x * blockDim.x
    if (i < n) C[i] = A[i] * B[i];
}