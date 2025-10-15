// student_kernel.cu
// TODO: Implement a naive 1-bit parallel radix sort that fully sorts 32-bit keys
// across 32 passes (LSB -> MSB), stable per pass.
//
// Requirements:
//  - API: extern "C" void radix_sort_1bit_host(unsigned int* data, int n)
//  - In-place: modify data in-place
//  - Stability: within each bit-partition, preserve relative order (use zerosBefore(i) / onesBefore(i))
//  - Multi-pass orchestration: 32 passes, swap ping-pong buffers every pass
//  - Correct for arbitrary n (n can be 0)
//  - No OOB writes: only write within [0..n)
//
// Hints (not mandatory, but aligned with the reference):
//  - kFlagZeros: flagsZero[i] = 1 if ((x >> bit)&1)==0 else 0
//  - kBlockExclusiveScan + host scan over block sums for robustness
//  - kAddBlockOffsets to turn per-block exclusive scans into global exclusive scan
//  - kScatter uses stable positions:
//        if bit==0: pos = zerosBefore(i)
//        else      : pos = totalZeros + (i - zerosBefore(i))

#include <cuda_runtime.h>

// TODO: Implement naive 1-bit radix sort using LSD approach.
// Contract summary:
//  - Stable: equal elements maintain relative order
//  - Process 1 bit per pass, 32 passes total for uint32_t
//  - Use parallel counting, prefix sum, and scattering
//  - Sort in ascending order

extern "C" __global__
void radix_sort_1bit_kernel(unsigned int* __restrict__ data,
                           unsigned int* __restrict__ temp,
                           int n,
                           int bit)
{
    // TODO: Implement single-bit radix sort pass
    // 1. Count elements with bit=0 and bit=1 using shared memory
    // 2. Compute prefix sums to find output positions
    // 3. Scatter elements to correct positions based on bit value
    // 4. Ensure stable sorting (preserve relative order for equal keys)
}

extern "C"
void radix_sort_1bit_host(unsigned int* data, int n)
{
    // TODO: Implement host function that orchestrates 32 sorting passes
    // 1. Allocate temporary buffer
    // 2. For each bit position (0 to 31):
    //    - Launch radix_sort_1bit_kernel
    //    - Swap data and temp pointers
    // 3. Ensure final result is in original data array
    // 4. Clean up temporary buffer
}