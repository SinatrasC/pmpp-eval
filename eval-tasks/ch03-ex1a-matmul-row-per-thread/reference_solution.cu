#include <cuda_runtime.h>

extern "C" __global__
void matrixMulRowKernel(const float* __restrict__ M,
                        const float* __restrict__ N,
                        float* __restrict__ P,
                        int size) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < size) {
        for (int col = 0; col < size; ++col) {
            float sum = 0.0f;
            for (int j = 0; j < size; ++j) {
                sum += M[row * size + j] * N[j * size + col];
            }
            P[row * size + col] = sum;
        }
    }
}