#include <cuda_runtime.h>

__global__ void vecAddKernel(const float* A, const float* B, float* C, int n) {
    // TODO: Implement vector addition
    // Hints:
    // - Calculate global thread index i from blockIdx.x, blockDim.x, and threadIdx.x
    // - Add bounds check to ensure i < n
    // - Compute C[i] = A[i] + B[i]
}