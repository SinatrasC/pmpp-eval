#include <cuda_runtime.h>

// BAD: Writes to input arrays (casts away const)
__global__ void vecMulKernel(const float* A, const float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float* mutable_A = const_cast<float*>(A);  // Cast away const
        mutable_A[i] = A[i] * B[i];  // Corrupts input A
        C[i] = mutable_A[i];
    }
}