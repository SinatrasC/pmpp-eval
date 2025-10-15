// ch14-spmv-hyb-single / student_kernel.cu
#include <cuda_runtime.h>

extern "C" __global__
void spmv_ell_rows_kernel(const int* __restrict__ colEll,
                          const float* __restrict__ valEll,
                          const float* __restrict__ x,
                          float* __restrict__ y,
                          int m, int K)
{
    // TODO: Compute the ELL portion of the HYB product row-by-row.
    //
    // DATA LAYOUT: Row-major ELL storage
    // For row i, the K entries are stored at indices [i*K, i*K+1, ..., i*K+K-1]
    // Example:
    //   int row = blockIdx.x * blockDim.x + threadIdx.x;
    //   if (row < m) {
    //       float sum = 0.0f;
    //       int base = row * K;
    //       for (int t = 0; t < K; ++t) {
    //           int col = colEll[base + t];
    //           if (col >= 0) sum += valEll[base + t] * x[col];
    //       }
    //       y[row] = sum;
    //   }
    if (colEll && valEll && x && y) {
        (void)m;
        (void)K;
    }
}

extern "C" __global__
void spmv_coo_accum_kernel(const int* __restrict__ rowCoo,
                           const int* __restrict__ colCoo,
                           const float* __restrict__ valCoo,
                           const float* __restrict__ x,
                           float* __restrict__ y,
                           int nnzC)
{
    // TODO: Accumulate the COO tail using atomicAdd.
    if (rowCoo && colCoo && valCoo && x && y) {
        (void)nnzC;
    }
}

extern "C" void spmv_hyb(const int* colEll, const float* valEll, int m, int K,
                         const int* rowCoo, const int* colCoo, const float* valCoo, int nnzC,
                         const float* x, float* y)
{
    (void)colEll; (void)valEll; (void)m; (void)K;
    (void)rowCoo; (void)colCoo; (void)valCoo; (void)nnzC;
    (void)x; (void)y;

    // TODO: Launch spmv_ell_rows_kernel followed by spmv_coo_accum_kernel to
    // form the full HYB SpMV result.
}
