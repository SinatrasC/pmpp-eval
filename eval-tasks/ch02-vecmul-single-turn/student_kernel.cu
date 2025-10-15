#include <cuda_runtime.h>

__global__ void vecMulKernel(const float* A, const float* B, float* C, int n) {
    // TODO: Each thread i computes: C[i] = A[i] * B[i]  (if i < n)
    // Hints:
    //  - Derive global index i from block and thread indices
    //  - Guard against i >= n
}