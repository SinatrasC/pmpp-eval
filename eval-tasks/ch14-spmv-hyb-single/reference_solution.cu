// ch14-spmv-hyb-single / reference_solution.cu
#include <cuda_runtime.h>

extern "C" __global__
void spmv_ell_rows_kernel(const int* __restrict__ colEll,
                          const float* __restrict__ valEll,
                          const float* __restrict__ x,
                          float* __restrict__ y,
                          int m, int K)
{
    for(int row = blockIdx.x*blockDim.x + threadIdx.x;
        row < m;
        row += blockDim.x*gridDim.x){
        float s = 0.f;
        int base = row * K;
        for(int t=0;t<K;++t){
            int c = colEll[base + t];
            if(c >= 0) s += valEll[base + t] * x[c];
        }
        y[row] = s; // overwrite
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
    for(int k = blockIdx.x*blockDim.x + threadIdx.x;
        k < nnzC;
        k += blockDim.x*gridDim.x){
        int r = rowCoo[k], c = colCoo[k];
        atomicAdd(&y[r], valCoo[k] * x[c]);
    }
}

extern "C" void spmv_hyb(const int* colEll, const float* valEll, int m, int K,
                         const int* rowCoo, const int* colCoo, const float* valCoo, int nnzC,
                         const float* x, float* y)
{
    dim3 block(256);
    auto cdiv=[](int a,int b){return (a+b-1)/b;};
    int gridEll = (m>0)? std::max(1, cdiv(m, (int)block.x)) : 1;
    int gridCoo = (nnzC>0)? std::max(1, cdiv(nnzC, (int)block.x)) : 1;
    gridEll = std::min(gridEll, 65535);
    gridCoo = std::min(gridCoo, 65535);

    spmv_ell_rows_kernel<<<gridEll, block>>>(colEll, valEll, x, y, m, K);
    cudaDeviceSynchronize();
    spmv_coo_accum_kernel<<<gridCoo, block>>>(rowCoo, colCoo, valCoo, x, y, nnzC);
    cudaDeviceSynchronize();
}