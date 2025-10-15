#include <cuda_runtime.h>

extern "C" __global__
void matrixMulColKernel(const float* __restrict__ M,
                        const float* __restrict__ N,
                        float* __restrict__ P,
                        int size) {
    // TODO:
    // - Each thread computes ONE output column 'col'
    // - Guard: if (col < size)  
    // - For each row 'row', compute dot(M[row, :], N[:, col])
    // - Write P[row * size + col]
    // Hints:
    //   int col = blockIdx.x * blockDim.x + threadIdx.x;
    //   for (int row = 0; row < size; ++row) { ... }
}