#include <cuda_runtime.h>

// BAD: Off-by-one indexing error
__global__ void vecMulKernel(const float* A, const float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;  // Extra +1
    if (i < n) C[i] = A[i] * B[i];
}