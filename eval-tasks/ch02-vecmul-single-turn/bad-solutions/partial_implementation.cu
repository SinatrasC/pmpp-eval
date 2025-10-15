#include <cuda_runtime.h>

// BAD: Only works for even indices 
__global__ void vecMulKernel(const float* A, const float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && i % 2 == 0) {  // Only processes even indices
        C[i] = A[i] * B[i];
    }
}