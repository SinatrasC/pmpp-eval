// ch14-spmv-csr-thread-per-row-single / reference_solution.cu
#include <cuda_runtime.h>

extern "C" __global__
void spmv_csr_kernel(const int* __restrict__ rowPtr,
                     const int* __restrict__ colIdx,
                     const float* __restrict__ vals,
                     const float* __restrict__ x,
                     float* __restrict__ y,
                     int m)
{
    for (int row = blockIdx.x*blockDim.x + threadIdx.x;
         row < m;
         row += blockDim.x*gridDim.x)
    {
        const int start = rowPtr[row];
        const int end   = rowPtr[row+1];
        float sum = 0.f;
        for (int j=start;j<end;++j){
            sum += vals[j] * x[colIdx[j]];
        }
        y[row] = sum;
    }
}