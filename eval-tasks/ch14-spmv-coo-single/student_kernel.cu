// ch14-spmv-coo-single / student_kernel.cu
#include <cuda_runtime.h>

extern "C" __global__
void spmv_coo_kernel(const int* __restrict__ row_idx,
                     const int* __restrict__ col_idx,
                     const float* __restrict__ vals,
                     const float* __restrict__ x,
                     float* __restrict__ y,
                     int nnz)
{
    // TODO: Implement COO-format SpMV using a grid-stride loop and atomicAdd on
    // y[row] to handle duplicate entries.

    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < nnz) {
        int r = row_idx[k];
        int c = col_idx[k];
        (void)r;
        (void)c;
        (void)vals[k];
        (void)x;
        (void)y;
    }
}
