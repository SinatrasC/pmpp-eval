// student_kernel.cu
#include <cuda_runtime.h>
#include <cstddef>

// TODO: Implement naive global-atomic histogram.
//
// Requirements:
// - Use global-memory atomicAdd(&hist[bin], 1u)
// - Grid-stride loop over N
// - Ignore out-of-range bin indices
// - Do not write to 'in'
// - No shared memory
//
// Signature must not change.
__global__ void histogram_kernel(const int* in, unsigned int* hist,
                                 size_t N, int num_bins)
{
    // TODO:
    // size_t i = ...
    // size_t stride = ...
    // for (; i < N; i += stride) {
    //   int bin = in[i];
    //   if (0 <= bin && bin < num_bins) atomicAdd(&hist[bin], 1u);
    // }
}