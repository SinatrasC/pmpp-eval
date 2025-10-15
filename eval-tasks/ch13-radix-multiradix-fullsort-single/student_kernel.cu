// student_kernel.cu
#include <cuda_runtime.h>
#include <stdint.h>
#include <cstdio>

static inline void CK(cudaError_t e, const char* m){
    if(e != cudaSuccess){
        std::fprintf(stderr, "CUDA %s: %s\n", m, cudaGetErrorString(e));
        std::exit(2);
    }
}

// TODO: Implement a multiradix (RADIX_BITS=4) stable radix sort in-place.
// Contract:
//   extern "C" void radix_sort_multiradix_host(unsigned int* data, int n);
// Requirements:
//   - Sort ascending, stable per pass (4-bit buckets, 8 passes total).
//   - Arbitrary n (including 0 / non-multiples of block size).
//   - No OOB writes (tests use guarded buffers).
//   - In-place on `data` (you may use an internal device temp buffer).

// Choose your radix size (recommended: 2-bit or 4-bit)
#define RADIX_BITS 2
#define RADIX_SIZE (1 << RADIX_BITS)  // 2^RADIX_BITS buckets
#define RADIX_MASK (RADIX_SIZE - 1)

extern "C" __global__
void radix_sort_multiradix_kernel(unsigned int* __restrict__ data,
                                 unsigned int* __restrict__ temp,
                                 int n,
                                 int shift)
{
    // TODO: Implement multi-radix sort pass
    // 1. Count elements for each bucket (RADIX_SIZE buckets) using shared memory
    // 2. Compute prefix sums to find output positions for each bucket
    // 3. Scatter elements to correct positions based on radix value
    // 4. Ensure stable sorting (preserve relative order for equal keys)
}

extern "C"
void radix_sort_multiradix_host(unsigned int* data, int n)
{
    if (n <= 1) return;
    // Starter behavior: shallow copy → will fail non-trivial tests.
    unsigned int* tmp = nullptr;
    CK(cudaMalloc(&tmp, n*sizeof(unsigned int)), "malloc tmp");
    CK(cudaMemcpy(tmp, data, n*sizeof(unsigned int), cudaMemcpyDeviceToDevice), "copy tmp");
    CK(cudaMemcpy(data, tmp, n*sizeof(unsigned int), cudaMemcpyDeviceToDevice), "copy back");
    cudaFree(tmp);
}