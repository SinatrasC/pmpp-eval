#include <cuda_runtime.h>

// BAD: Writes to adjacent memory locations 
__global__ void vecAddKernel(const float* A, const float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
        if (i + 1 < n) C[i + 1] = 0.0f;  // Corrupts next element
    }
}