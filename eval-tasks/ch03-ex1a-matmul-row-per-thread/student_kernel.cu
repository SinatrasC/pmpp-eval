#include <cuda_runtime.h>

extern "C" __global__
void matrixMulRowKernel(const float* __restrict__ M,
                        const float* __restrict__ N,
                        float* __restrict__ P,
                        int size) {
    // TODO:
    // - Each thread computes ONE output row 'row'
    // - Guard: if (row < size)
    // - For each column 'col', compute dot(M[row, :], N[:, col])
    // - Write P[row * size + col]
    // Hints:
    //   int row = blockIdx.x * blockDim.x + threadIdx.x;
    //   for (int col = 0; col < size; ++col) { ... }
}