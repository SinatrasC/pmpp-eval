// ch14-spmv-jds-single / reference_solution.cu
#include <cuda_runtime.h>

extern "C" __global__
void spmv_jds_kernel(const int* __restrict__ colJds,
                     const float* __restrict__ valJds,
                     const int* __restrict__ permute,
                     const int* __restrict__ jdPtr,
                     const float* __restrict__ x,
                     float* __restrict__ y,
                     int m, int maxJ)
{
    for(int tid = blockIdx.x * blockDim.x + threadIdx.x;
        tid < m;
        tid += blockDim.x * gridDim.x) {
        int orig_row = permute[tid];
        float sum = 0.f;
        for(int d = 0; d < maxJ; d++) {
            int diag_size = jdPtr[d+1] - jdPtr[d];
            if(tid < diag_size) {
                int jds_idx = jdPtr[d] + tid;
                sum += valJds[jds_idx] * x[colJds[jds_idx]];
            }
        }
        y[orig_row] = sum;
    }
}

extern "C" void spmv_jds(const int* colJds, const float* valJds,
                         const int* permute, const int* jdPtr,
                         const float* x, float* y, int m, int maxJ)
{
    dim3 block(256);
    int grid = std::max(1, (m + (int)block.x - 1) / (int)block.x);
    spmv_jds_kernel<<<grid, block>>>(colJds, valJds, permute, jdPtr, x, y, m, maxJ);
    cudaDeviceSynchronize();
}