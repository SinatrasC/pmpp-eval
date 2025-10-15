#include <cuda_runtime.h>

// BAD: Does subtraction instead of addition
__global__ void vecAddKernel(const float* A, const float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] - B[i];  // Should be A[i] + B[i]
}