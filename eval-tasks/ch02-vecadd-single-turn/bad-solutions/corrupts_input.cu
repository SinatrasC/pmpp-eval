#include <cuda_runtime.h>

// BAD: Tries to modify const input (breaks const correctness)
__global__ void vecAddKernel(const float* A, const float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        ((float*)A)[i] = 999.0f;  // Cast away const and corrupt input
        C[i] = A[i] + B[i];
    }
}