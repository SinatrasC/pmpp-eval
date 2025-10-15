// student_kernel.cu
#include <cuda_runtime.h>
#include <cstddef>

// TODO: Implement shared-memory privatized histogram
//
// Requirements:
//  - Use shared memory to reduce global atomic contention
//  - Each block maintains its own private histogram in shared memory
//  - Accumulate counts into shared histogram using atomicAdd
//  - Flush shared histogram to global histogram at the end
//  - Process all N input elements across the grid
//
// Algorithm steps:
//  1. Initialize shared memory histogram to zero (cooperative initialization)
//     - Each thread zeros multiple bins if num_bins > blockDim.x
//     - Use loop: for (int bin = threadIdx.x; bin < num_bins; bin += blockDim.x)
//     - Synchronize after initialization
//
//  2. Accumulate input values into shared histogram
//     - Use grid-stride loop to process input elements
//     - Read input value, treat it as bin index
//     - Validate bin is in range [0, num_bins)
//     - Use atomicAdd to increment shared histogram bin
//     - Synchronize after accumulation
//
//  3. Merge shared histogram into global histogram
//     - Each thread handles multiple bins if needed
//     - Use atomicAdd to add shared counts to global histogram
//
// Hints:
//  - Shared memory is declared: extern __shared__ unsigned int s_hist[];
//  - Grid-stride loop: for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < N; i += blockDim.x*gridDim.x)
//  - Atomic operations: atomicAdd(&s_hist[bin], 1u) for shared, atomicAdd(&hist[bin], count) for global
//  - Synchronization required between phases: __syncthreads()

__global__ void histogram_kernel(const int* in, unsigned int* hist,
                                 size_t N, int num_bins)
{
    extern __shared__ unsigned int s_hist[];

    // TODO: Implement 3-phase histogram: init shared → accumulate → flush to global
    (void)in; (void)hist; (void)N; (void)num_bins; (void)s_hist;
}
