#include <cuda_runtime.h>

// BAD: Deliberately massive out-of-bounds writes
__global__ void vecMulKernel(const float* A, const float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // Write far beyond bounds to definitely hit canaries
    if (i < 1000) {  // Will write way past n for small test cases
        C[i] = A[i] * B[i];
    }
}