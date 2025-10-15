// ch14-spmv-ell-single / student_kernel.cu
#include <cuda_runtime.h>

extern "C" __global__
void spmv_ell_kernel(const int* __restrict__ colIdx,
                     const float* __restrict__ vals,
                     const float* __restrict__ x,
                     float* __restrict__ y,
                     int m, int K)
{
    // TODO: Assign one thread per row, iterate across the K padded entries,
    // skip negative column indices, and accumulate into y[row].
    //
    // DATA LAYOUT: Row-major ELL storage
    // For row i, the K entries are stored at indices [i*K, i*K+1, ..., i*K+K-1]
    // Example:
    //   int base = row * K;
    //   for (int t = 0; t < K; ++t) {
    //       int col = colIdx[base + t];
    //       if (col >= 0) {
    //           // accumulate vals[base + t] * x[col]
    //       }
    //   }

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m) {
        (void)colIdx;
        (void)vals;
        (void)x;
        if (y) {
            y[row] = 0.0f;
        }
    }

    (void)K;
}
