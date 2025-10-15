// ch14-spmv-ell-single / reference_solution.cu
#include <cuda_runtime.h>

extern "C" __global__
void spmv_ell_kernel(const int* __restrict__ colIdx,
                     const float* __restrict__ vals,
                     const float* __restrict__ x,
                     float* __restrict__ y,
                     int m, int K)
{
    for (int row = blockIdx.x*blockDim.x + threadIdx.x;
         row < m;
         row += blockDim.x*gridDim.x)
    {
        float sum = 0.f;
        int base = row * K;
        for (int t=0; t<K; ++t) {
            int c = colIdx[base + t];
            if (c >= 0) sum += vals[base + t] * x[c];
        }
        y[row] = sum;
    }
}