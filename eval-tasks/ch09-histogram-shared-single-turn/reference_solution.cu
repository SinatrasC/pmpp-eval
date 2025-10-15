// reference_solution.cu
#include <cuda_runtime.h>
#include <cstddef>

__global__ void histogram_kernel(const int* in, unsigned int* hist,
                                 size_t N, int num_bins)
{
    extern __shared__ unsigned int s_hist[];

    // 1. Cooperatively zero shared histogram
    for (int bin = threadIdx.x; bin < num_bins; bin += blockDim.x) {
        s_hist[bin] = 0u;
    }
    __syncthreads();

    // 2. Grid-stride loop with shared accumulation
    size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    size_t stride = size_t(blockDim.x) * gridDim.x;
    for (; i < N; i += stride) {
        int bin = in[i];
        if (bin >= 0 && bin < num_bins) {
            atomicAdd(&s_hist[bin], 1u);
        }
    }
    __syncthreads();

    // 3. Cooperatively merge to global memory
    for (int bin = threadIdx.x; bin < num_bins; bin += blockDim.x) {
        if (s_hist[bin] > 0) {
            atomicAdd(&hist[bin], s_hist[bin]);
        }
    }
}