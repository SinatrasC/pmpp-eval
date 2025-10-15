#include <cuda_runtime.h>

extern "C" __global__
void matrixMulColKernel(const float* __restrict__ M,
                        const float* __restrict__ N,
                        float* __restrict__ P,
                        int size) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < size) {
        for (int row = 0; row < size; ++row) {
            float sum = 0.0f;
            for (int j = 0; j < size; ++j) {
                sum += M[row * size + j] * N[j * size + col];
            }
            P[row * size + col] = sum;
        }
    }
}