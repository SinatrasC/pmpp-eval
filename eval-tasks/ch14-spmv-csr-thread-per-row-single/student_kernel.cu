// ch14-spmv-csr-thread-per-row-single / student_kernel.cu
#include <cuda_runtime.h>

extern "C" __global__
void spmv_csr_kernel(const int* __restrict__ rowPtr,
                     const int* __restrict__ colIdx,
                     const float* __restrict__ vals,
                     const float* __restrict__ x,
                     float* __restrict__ y,
                     int m)
{
    // TODO: Assign one thread per row, iterate over the row's nonzeros, and
    // write y[row] without using atomics.

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m) {
        int start = rowPtr[row];
        int end   = rowPtr[row + 1];
        float scratch = 0.0f;
        for (int j = start; j < end; ++j) {
            (void)colIdx[j];
            (void)vals[j];
        }
        if (y) {
            y[row] = scratch; // placeholder leaves output unchanged (all zeros)
        }
    }
}
