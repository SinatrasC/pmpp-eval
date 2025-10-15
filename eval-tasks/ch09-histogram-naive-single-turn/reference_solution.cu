// reference_solution.cu
#include <cuda_runtime.h>
#include <cstddef>

__global__ void histogram_kernel(const int* in, unsigned int* hist,
                                 size_t N, int num_bins)
{
    size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    size_t stride = size_t(blockDim.x) * gridDim.x;

    for (; i < N; i += stride) {
        int b = in[i];
        if (b >= 0 && b < num_bins) {
            atomicAdd(&hist[b], 1u);
        }
    }
}