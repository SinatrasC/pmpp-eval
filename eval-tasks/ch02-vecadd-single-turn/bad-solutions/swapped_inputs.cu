#include <cuda_runtime.h>

// BAD: Swapped order of inputs (may still work for addition but tests the harness)
__global__ void vecAddKernel(const float* A, const float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = B[i] + A[i];  // Order shouldn't matter for +, but tests our data
}