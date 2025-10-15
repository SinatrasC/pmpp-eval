#include <cuda_runtime.h>

// BAD: Missing bounds check - will corrupt canaries
__global__ void vecAddKernel(const float* A, const float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    C[i] = A[i] + B[i];  // Missing: if (i < n)
}