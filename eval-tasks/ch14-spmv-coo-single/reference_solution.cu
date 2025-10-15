// ch14-spmv-coo-single / reference_solution.cu
#include <cuda_runtime.h>

extern "C" __global__
void spmv_coo_kernel(const int* __restrict__ row_idx,
                     const int* __restrict__ col_idx,
                     const float* __restrict__ vals,
                     const float* __restrict__ x,
                     float* __restrict__ y,
                     int nnz)
{
    for (int k = blockIdx.x * blockDim.x + threadIdx.x;
         k < nnz;
         k += blockDim.x * gridDim.x)
    {
        const int r = row_idx[k];
        const int c = col_idx[k];
        const float a = vals[k];
        atomicAdd(&y[r], a * x[c]);
    }
}